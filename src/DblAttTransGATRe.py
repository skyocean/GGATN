import os
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datetime import datetime, timedelta

from torch_geometric.nn import GATConv

from torch.utils.data import TensorDataset, DataLoader

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


# =========================================================
# GATEncoder
# =========================================================
    

class ActivityGATEncoder(nn.Module):
    """
    GAT encoder that produces structure-aware activity embeddings
    from the global process graph.

    Parameters
    ----------
    num_activities : int   – size of activity vocabulary
    d_model        : int   – embedding dimension (must be divisible by heads)
    edge_dim       : int   – edge attribute dimension (default 2: log_freq, mean_log_dt)
    num_layers     : int   – number of GAT layers (default 2 for 2-hop receptive field)
    heads          : int   – attention heads per layer
    dropout        : float – output dropout rate (applied after each layer)
    use_norm       : bool  – apply LayerNorm after each layer
    init_scale     : float – std for initial node embedding
    """

    def __init__(
        self,
        num_activities: int,
        d_model: int,
        edge_dim: int = 2,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        use_norm: bool = True,
        init_scale: float = 0.01,
    ):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"

        self.num_activities = num_activities
        self.d_model        = d_model
        self.num_layers     = num_layers
        self.dropout_rate   = dropout
        self.use_norm       = use_norm

        # learnable initial node embeddings
        self.node_emb = nn.Parameter(
            torch.randn(num_activities, d_model) * init_scale
        )

        # GAT layers
        # dropout=0 inside GATConv — we handle dropout explicitly after each layer so there is a single clean control point
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=d_model,
                out_channels=d_model // heads,
                heads=heads,
                concat=True,          # output shape stays d_model
                edge_dim=edge_dim,
                dropout=0.0,          # no internal dropout — see note above
            )
            for _ in range(num_layers)
        ])

        if use_norm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_layers)
            ])

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def forward(self, edge_index, edge_attr, return_attention: bool = True):
        """
        Parameters
        ----------
        edge_index       : LongTensor  (2, E)
        edge_attr        : FloatTensor (E, edge_dim)
        return_attention : bool – whether to return last-layer attention weights

        Returns
        -------
        E_act  : FloatTensor (num_activities, d_model)
        ei     : LongTensor  (2, E)   – only if return_attention=True
        alpha  : FloatTensor (E, H)   – only if return_attention=True
        """
        x = self.node_emb   # (N, d_model)
        ei, alpha = None, None

        for i, gat in enumerate(self.gat_layers):
            is_last = (i == self.num_layers - 1)

            # capture attention weights only on the last layer
            if is_last:
                out, (ei, alpha) = gat(
                    x, edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
            else:
                out = gat(x, edge_index, edge_attr=edge_attr)

            out = out + x                           # residual

            if self.use_norm:
                out = self.norms[i](out)

            if self.dropout_rate > 0:
                out = self.dropout(out)

            x = out

        if return_attention:
            return x, ei, alpha
        return x


# =========================================================
# Positional encoding
# =========================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, L: int) -> torch.Tensor:
        return self.pe[:L]

# =========================================================
# Transformer encoder with attention extraction: 
# To contextualize each sequence position using global self attention before pass representations into the second attention stage.
# =========================================================

class TransformerLayerPreLN(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_mult: int = 4, dropout: float = 0.1): 
        super().__init__()
        #ff_mult: the size of the feed forward expansion
        dim_ff = ff_mult * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        

        self.mha = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = nhead,
            dropout = dropout,
            batch_first = True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                                  # (B, L, d)
        key_padding_mask: Optional[torch.Tensor] = None,   # (B, L) bool True pad
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """
        Returns
        -------
        x    : (B, L, d_model)
        attn : (B, nhead, L(q), L(k)) if need_weights else None
        """
        
        qkv = self.norm1(x)
        out, attn = self.mha(
            qkv, qkv, qkv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=False
        )
        x = x + self.drop(out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn


class TransformerEncoderWithAttn(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, ff_mult: int = 4, dropout: float = 0.1):
        
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayerPreLN(d_model, nhead, ff_mult = ff_mult, dropout = dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                                  # (B, L, d)
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_all_attn: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        """
        Returns
        -------
        x         : (B, L, d_model)
        attn_list : list of (B, nhead, L, L)
                    length == num_layers if return_all_attn=True
                    length == 1          otherwise (last layer only)
        """
        
        attn_list: List[torch.Tensor] = []
        n = len(self.layers)

        for i, layer in enumerate(self.layers):
            is_last = (i == n - 1)
            need = is_last or return_all_attn
            x, attn = layer(x, 
                            key_padding_mask = key_padding_mask, 
                            attn_mask=attn_mask,
                            need_weights = need)
            if need and attn is not None:
                attn_list.append(attn)

        x = self.final_norm(x)
        
        return x, attn_list


# =========================================================
# Graph cross-attention: positions attend to activity nodes
# =========================================================

class GraphCrossAttention(nn.Module):
    """
    Sequence positions (queries) attend to GAT activity embeddings
    (keys and values), injecting process-graph structure into H_seq.

    """

    def __init__(
        self,
        d_model   : int,
        nhead     : int,
        ff_mult_x   : int   = 4,     # keep small — see docstring
        dropout   : float = 0.1,
        gate_init : float = 0.1,
    ):
        super().__init__()
        dim_ff = ff_mult_x * d_model  # ff_mult=1 → d_model→d_model, no expansion

        self.norm1    = nn.LayerNorm(d_model)   # query norm (Pre-LN)
        self.norm_out = nn.LayerNorm(d_model)   # post-injection norm
        self.norm2    = nn.LayerNorm(d_model)   # FF input norm

        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = nhead,
            dropout     = dropout,
            batch_first = True,
        )

        # intentionally small FF — no expansion
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        self.drop = nn.Dropout(dropout)

        # learnable gate via tanh → (-1, 1); small init = conservative start
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(
        self,
        H_seq                : torch.Tensor,                   # (B, L, d)
        E_act                : torch.Tensor,                   # (A, d)
        act_key_padding_mask : Optional[torch.Tensor] = None,  # (B, A) bool True=masked
        need_weights         : bool = False,                   # False during training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns
        -------
        H_out : (B, L, d_model)
        attn  : (B, nhead, L, A) if need_weights=True else None
        """
        B, L, d = H_seq.shape
        A       = E_act.shape[0]

        # expand to batch — view only, no copy
        # passed separately as key and value: MHA projects independently via Wk/Wv
        nodes = E_act.unsqueeze(0).expand(B, A, d)

        # --- cross-attention (Pre-LN on query) ---
        q = self.norm1(H_seq)
        out, attn = self.attn(
            query                = q,
            key                  = nodes,
            value                = nodes,
            key_padding_mask     = act_key_padding_mask,
            need_weights         = need_weights,
            average_attn_weights = False,
        )

        # gated residual + output norm
        g     = torch.tanh(self.gate)
        H_tmp = self.norm_out(H_seq + g * self.drop(out))

        # small FF — transforms injected info without overfitting
        H_out = H_tmp + self.drop(self.ff(self.norm2(H_tmp)))

        return H_out, attn


# =========================================================
# Utilities
# =========================================================

def build_ar_decoder_inputs(y_activity: torch.Tensor, sos_id: int) -> torch.Tensor:
    """
    y_activity: (B, L) target sequence including EOS and PAD after EOS
    returns decoder input shifted right with SOS at position 0
    """
    dec_in = torch.empty_like(y_activity)
    dec_in[:, 0] = sos_id
    dec_in[:, 1:] = y_activity[:, :-1]
    return dec_in
    
def build_causal_attn_mask(L: int, device: torch.device) -> torch.Tensor:
    """
    Returns (L, L) bool mask for nn.MultiheadAttention
    True means blocked.
    """
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    w = mask.unsqueeze(-1)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1e-8)

def mask_attention_square(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    attn: (B, heads, L, L)
    mask: (B, L) float or bool
    """
    m = mask.bool()
    attn = attn * m[:, None, None, :]  # keys
    attn = attn * m[:, None, :, None]  # queries
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return attn


def mask_attention_rect(attn: torch.Tensor, q_mask: torch.Tensor, k_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    attn: (B, heads, Lq, Lk)
    q_mask: (B, Lq)
    k_mask: (B, Lk) optional
    """
    qm = q_mask.bool()
    attn = attn * qm[:, None, :, None]  # queries

    if k_mask is not None:
        km = k_mask.bool()
        attn = attn * km[:, None, None, :]  # keys

    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return attn

def compute_activity_accuracy(logits, targets, mask):
    """
    logits  : (B, L, A)
    targets : (B, L)
    mask    : (B, L) float or bool

    returns scalar accuracy
    """
    with torch.no_grad():
        valid = mask.bool()

        preds = logits.argmax(dim=-1)

        correct = (preds == targets) & valid

        total_correct = correct.sum().float()
        total_valid = valid.sum().clamp_min(1).float()

        acc = total_correct / total_valid

    return acc
    
def compute_activity_accuracy_no_eos(logits, targets, mask, eos_id: int = 2):
    with torch.no_grad():
        valid = mask.bool() & (targets != eos_id)
        preds = logits.argmax(dim=-1)
        correct = (preds == targets) & valid
        acc = correct.sum().float() / valid.sum().clamp_min(1).float()
    return acc
    
def compute_final_activity_accuracy(logits, targets, lengths):
    """
    Final-position activity accuracy.
    With EOS-based training, this is mostly EOS accuracy.
    """

    with torch.no_grad():

        device = logits.device
        lengths = lengths.to(device).long().clamp(min=1, max=logits.shape[1])

        final_pos = lengths - 1
        batch_idx = torch.arange(logits.shape[0], device=device)

        preds = logits.argmax(dim=-1)

        final_preds = preds[batch_idx, final_pos]
        final_targets = targets[batch_idx, final_pos]

        valid = (
            (final_targets != 0) &   # PAD
            (final_targets != 1)     # UNK
        )

        if valid.any():
            acc = (final_preds[valid] == final_targets[valid]).float().mean()
        else:
            acc = torch.tensor(0.0, device=device)

    return acc    

def compute_preterminal_activity_accuracy(logits, targets, lengths):
    """
    Preterminal-position activity accuracy.
    This is usually the more informative 'last real activity' accuracy.
    """

    with torch.no_grad():

        device = logits.device
        B, L, _ = logits.shape

        lengths = lengths.to(device).long()

        # only sequences with length >= 2 have a preterminal event
        valid_len = lengths >= 2
        if not valid_len.any():
            return torch.tensor(0.0, device=device)

        batch_idx = torch.arange(B, device=device)[valid_len]

        pre_pos = (lengths[valid_len] - 2).clamp(min=0)

        preds = logits.argmax(dim=-1)

        pre_preds = preds[batch_idx, pre_pos]
        pre_targets = targets[batch_idx, pre_pos]

        valid_targets = (
            (pre_targets != 0) &   # PAD
            (pre_targets != 1)     # UNK
        )

        if valid_targets.any():
            acc = (pre_preds[valid_targets] == pre_targets[valid_targets]).float().mean()
        else:
            acc = torch.tensor(0.0, device=device)

    return acc
    
def compute_seq_cat_accuracy(outputs, batch):
    accs = []
    for name, logits in outputs["seq_categorical"].items():
        preds = logits.argmax(dim=-1)
        targets = batch["seq_cat"][name]
        acc = (preds == targets).float().mean()
        accs.append(acc)
    if len(accs) == 0:
        return torch.tensor(0.0)
    return torch.stack(accs).mean()

def compute_seq_num_rmse(outputs, batch):
    rmses = []
    for name, preds in outputs["seq_numeric"].items():
        targets = batch["seq_numeric"][name]
        rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
        rmses.append(rmse)
    if len(rmses) == 0:
        return torch.tensor(0.0)
    return torch.stack(rmses).mean()    

def compute_event_cat_accuracy(outputs, batch, eos_id: int = 2):
    """
    Event categorical accuracy on real events only, excluding EOS.
    """
    y_act = batch["y_activity"]
    valid_mask = batch["mask"].bool()
    non_eos_mask = valid_mask & (y_act != eos_id)

    accs = []
    for name, logits in outputs["event_categorical"].items():
        preds = logits.argmax(dim=-1)
        targets = batch["event_cat"][name]

        if non_eos_mask.any():
            correct = (preds == targets) & non_eos_mask
            acc = correct.sum().float() / non_eos_mask.sum().clamp_min(1)
            accs.append(acc)

    if len(accs) == 0:
        return torch.tensor(0.0, device=y_act.device)

    return torch.stack(accs).mean()


def compute_event_num_rmse(outputs, batch, eos_id: int = 2):
    """
    Event numeric RMSE on real events only, excluding EOS.
    """
    y_act = batch["y_activity"]
    valid_mask = batch["mask"].bool()
    non_eos_mask = valid_mask & (y_act != eos_id)

    rmses = []
    for name, preds in outputs["event_numeric"].items():
        targets = batch["event_numeric"][name]

        if non_eos_mask.any():
            rmse = torch.sqrt(torch.mean((preds[non_eos_mask] - targets[non_eos_mask]) ** 2))
            rmses.append(rmse)

    if len(rmses) == 0:
        return torch.tensor(0.0, device=y_act.device)

    return torch.stack(rmses).mean()

def compute_time_rmse(outputs, batch, eos_id: int = 2):
    """
    Time RMSE on real events only, excluding EOS.
    Must match training loss masking.
    """
    y_act = batch["y_activity"]
    valid_mask = batch["mask"].bool()
    non_eos_mask = valid_mask & (y_act != eos_id)

    if non_eos_mask.any():
        preds = outputs["time_pred"][non_eos_mask]
        targets = batch["y_time_log"][non_eos_mask]
        return torch.sqrt(torch.mean((preds - targets) ** 2))

    return torch.tensor(0.0, device=outputs["time_pred"].device)

def compute_len_accuracy(outputs, batch):
    preds = outputs["length_logits"].argmax(dim=-1)
    targets = batch["length"] - 1
    return (preds == targets).float().mean()

    

# =========================================================
# Config
# =========================================================

@dataclass
class GraphSequenceDoubleAttentionConfig:
    L_max: int
    num_activities: int

    d_model: int = 128
    d_case_num: int = 5  #  sequence_head dims
    ff_mult: int = 4
    ff_mult_x: int = 4
    n_layers: int = 4
    de_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    gate_init: float = 0.05

    latent_dim: int = 0
    latent_std: float = 1.0

    length_bins: int = 64

    # loss weights
    w_activity: float = 1.0
    w_time: float = 0.1
    w_length: float = 0.25
    w_event_cat: float = 0.25
    w_event_num: float = 0.0
    # case (sequence-level) prediction weights
    w_seq_cat: float = 0.2
    w_seq_num: float = 0.0
    w_transition_penalty: float = 0

    # decoding stabilizer
    use_activity_bias: bool = True

    # activity ids reserved
    pad_id: int = 0
    unk_id: int = 1
    eos_id: int = 2
    sos_id: int = 3

    # head
    activity_decoder: str = "cosine_linear" # allow values "cosine_linear", "cosine", "linear"

    # final activity setup
    w_final_activity: float = 5.0
    w_preterminal_activity: float = 3.0

    # feedback block
    use_activity_feedback: bool = False
    n_refine_layers: int = 1
    activity_feedback_gate_init: float = 0.10
    activity_feedback_temp: float = 0.5

    # pre transition 
    use_transition_bias: bool = True
    transition_bias_weight: float = 1.0

    #generator part
    unary_weight: float = 0.5


# =========================================================
# Main model
# =========================================================

class GraphSequenceDoubleAttentionOneShotModel(nn.Module):
    """
    Final architecture:
      Stage 1: external ActivityGATEncoder -> E_act (A, d)
      Stage 2: transformer over positions -> H_seq (B, L, d)
      Stage 3: graph cross attention -> H_struct (B, L, d)
      Decode: activity_logits = H_struct @ E_act^T
      Heads: time, length, event attributes on H_struct
    """

    def __init__(self, cfg: GraphSequenceDoubleAttentionConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.cfg = cfg

        self.case_proj = nn.Linear(cfg.d_case_num, cfg.d_model)

        if cfg.latent_dim > 0:
            self.z_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        else:
            self.z_proj = None

        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.L_max)
        self.pos_emb = nn.Embedding(cfg.L_max, cfg.d_model)

        self.transformer = TransformerEncoderWithAttn(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            num_layers=cfg.n_layers,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout
        )

        self.graph_cross = GraphCrossAttention(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            ff_mult_x=cfg.ff_mult_x,
            dropout=cfg.dropout,
            gate_init=cfg.gate_init
        )
        self.activity_token_emb = nn.Embedding(cfg.num_activities, cfg.d_model)
        self.ar_pos_emb = nn.Embedding(cfg.L_max, cfg.d_model)
        
        self.ar_decoder = TransformerEncoderWithAttn(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            num_layers=cfg.de_layers,          # start small
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout
        )
        
        self.use_activity_feedback = cfg.use_activity_feedback
        if self.use_activity_feedback:
            self.act_feedback_proj = nn.Linear(cfg.d_model, cfg.d_model)
        
            nn.init.normal_(self.act_feedback_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.act_feedback_proj.bias)
        
            self.act_feedback_gate = nn.Parameter(
                torch.tensor(cfg.activity_feedback_gate_init)
            )
        
            self.act_feedback_norm = nn.LayerNorm(cfg.d_model)
        
            self.refine_transformer = TransformerEncoderWithAttn(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                num_layers=cfg.n_refine_layers,
                ff_mult=cfg.ff_mult,
                dropout=cfg.dropout
            )
        else:
            self.act_feedback_proj = None
            self.act_feedback_gate = None
            self.act_feedback_norm = None
            self.refine_transformer = None


        self.drop = nn.Dropout(cfg.dropout)

        self.time_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1)
        )
        
        self.length_head = nn.Linear(cfg.d_model, cfg.length_bins)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0))) 

        self.activity_bias = nn.Parameter(torch.zeros(cfg.num_activities)) if cfg.use_activity_bias else None

        self.activity_linear = nn.Linear(cfg.d_model, cfg.num_activities)
        nn.init.zeros_(self.activity_linear.bias)
        nn.init.normal_(self.activity_linear.weight, mean=0.0, std=0.02)

        self.use_transition_bias = cfg.use_transition_bias
        if self.use_transition_bias:
            self.transition_bias = nn.Parameter(
                torch.zeros(cfg.num_activities, cfg.num_activities)
            )
        else:
            self.transition_bias = None

        # a knob (start)
        self.activity_linear_weight = nn.Parameter(torch.tensor(0.2))


        self.cat_heads = nn.ModuleDict()
        self.num_heads = nn.ModuleDict()

        self.seq_cat_heads = nn.ModuleDict()
        self.seq_num_heads = nn.ModuleDict()
        
    def _build_ar_inputs(
        self,
        decoder_tokens: torch.Tensor,   # (B, L)
        H_cond: torch.Tensor            # (B, L, d)
    ) -> torch.Tensor:
        B, L = decoder_tokens.shape
        device = decoder_tokens.device
    
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        tok = self.activity_token_emb(decoder_tokens)
        pos = self.ar_pos_emb(pos_ids)
    
        H_ar0 = tok + pos + H_cond
        return self.drop(H_ar0)

    def _apply_transition_bias_train(
        self,
        logits: torch.Tensor,          # (B, L, A)
        prev_tokens: torch.Tensor      # (B, L) previous-token ids for each position
    ) -> torch.Tensor:
        
        if (not self.use_transition_bias) or (self.transition_bias is None):
            return logits
    
        trans = self.transition_bias[prev_tokens]   # (B, L, A)
        w = self.cfg.transition_bias_weight
        return logits + w * trans

    def _build_prev_tokens_train(
        self,
        y_act: torch.Tensor
    ) -> torch.Tensor:

        B, L = y_act.shape
        prev = torch.full_like(y_act, fill_value=self.cfg.sos_id)
        if L > 1:
            prev[:, 1:] = y_act[:, :-1]
        return prev
    
    def _decode_activity_logits(
        self,
        H: torch.Tensor,
        E_act: torch.Tensor
    ) -> torch.Tensor:
        Hn = F.normalize(H, dim=-1)
        En = F.normalize(E_act, dim=-1)
    
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        cos_logits = scale * torch.matmul(Hn, En.t())
    
        decoder = self.cfg.activity_decoder
    
        if decoder == "cosine_linear":
            lin_logits = self.activity_linear(H)
            w_lin = torch.clamp(self.activity_linear_weight, 0.0, 3.0)
            logits = cos_logits + w_lin * lin_logits
    
        elif decoder == "cosine":
            logits = cos_logits
    
        elif decoder == "linear":
            logits = self.activity_linear(H)
    
        else:
            raise ValueError("Unknown activity_decoder")
    
        if self.activity_bias is not None:
            logits = logits + self.activity_bias.view(1, 1, -1)
    
        mask_ids = torch.tensor(
            [self.cfg.pad_id, self.cfg.unk_id, self.cfg.sos_id],
            device=logits.device
        )
        logits = logits.clone()
        logits.index_fill_(-1, mask_ids, -1e9)
    
        return logits


    def register_attribute_heads(
        self,
        event_cat_dims: Dict[str, int],
        event_num_names: List[str]
    ) -> None:
        d_in = 2 * self.cfg.d_model
        d_hidden = self.cfg.d_model
        drop = self.cfg.dropout
    
        for name, dim in event_cat_dims.items():
            self.cat_heads[name] = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d_hidden, dim)
            )
    
        for name in event_num_names:
            self.num_heads[name] = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d_hidden, 1)
            )
        
    def register_sequence_heads(
        self,
        seq_cat_dims: Dict[str, int],
        seq_num_names: List[str]
    ) -> None:
        d_in = 2 * self.cfg.d_model
        d_hidden = self.cfg.d_model
        drop = self.cfg.dropout
    
        for name, dim in seq_cat_dims.items():
            self.seq_cat_heads[name] = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d_hidden, dim)
            )
    
        for name in seq_num_names:
            self.seq_num_heads[name] = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d_hidden, 1)
            )
        
    def _build_input(
        self,
        sequence_head: torch.Tensor,    # (B, d_case_num)
        mask: torch.Tensor,             # (B, L)
        drop_length_feature: bool = True,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L = mask.shape
        x_case = sequence_head

        # in case the sequence_head layout is [length_norm, sin_hour, cos_hour, sin_wday, cos_wday]
        if drop_length_feature and x_case.shape[1] >= 1:
            x_case = x_case.clone()
            x_case[:, 0] = 0.0

        cond = self.case_proj(x_case)  # (B, d)

        if self.z_proj is not None:
            if z is None:
                z = torch.randn(B, self.cfg.latent_dim, device=sequence_head.device) * self.cfg.latent_std
            cond = cond + self.z_proj(z)


        pos_ids = torch.arange(L, device=sequence_head.device)
        pos_ids = pos_ids.unsqueeze(0).expand(B, L)

        H0 = cond.unsqueeze(1).expand(B, L, self.cfg.d_model)
        # H0 = H0 + self.pos_emb(pos_ids)
        H0 = H0 + self.pos_enc(L).unsqueeze(0) + self.pos_emb(pos_ids) 
        
        return self.drop(H0)

    def _build_activity_key_padding_mask(self, B: int, E_act: torch.Tensor) -> torch.Tensor:
        A = E_act.shape[0]
        kpm = torch.zeros((B, A), dtype=torch.bool, device=E_act.device)
    
        if 0 <= self.cfg.pad_id < A:
            kpm[:, self.cfg.pad_id] = True
    
        if 0 <= self.cfg.unk_id < A:
            kpm[:, self.cfg.unk_id] = True
    
        if 0 <= self.cfg.eos_id < A:
            kpm[:, self.cfg.eos_id] = True

        if 0 <= self.cfg.sos_id < A:
            kpm[:, self.cfg.sos_id] = True
    
        return kpm

    def forward(
        self,
        batch: Dict[str, Any],
        E_act: torch.Tensor,                       # (A, d)
        drop_length_feature: bool = True,
        return_all_attn: bool = False,
        return_graph_attn: bool = True,
        z: Optional[torch.Tensor] = None,
        use_teacher_forcing_transition: bool = True,
        use_teacher_forcing_attributes: bool = True
        
    ) -> Dict[str, Any]:
        
        seq_head = batch["sequence_head"]  # (B, d_case_num)
        mask = batch["mask"]               # (B, L) float
        B, L = mask.shape

        H0 = self._build_input(seq_head, mask, drop_length_feature=drop_length_feature, z=z)

        key_padding_mask = (mask <= 0.0)  # (B, L) bool True pad
        H_seq, tr_attn_list = self.transformer(
            H0,
            key_padding_mask=key_padding_mask,
            return_all_attn=return_all_attn
        )
        if len(tr_attn_list):
            tr_attn_list = [mask_attention_square(a, mask) for a in tr_attn_list]

        act_kpm = self._build_activity_key_padding_mask(B, E_act)
        H_struct, graph_attn = self.graph_cross(H_seq,
                                                E_act,
                                                act_key_padding_mask=act_kpm,
                                                need_weights=return_graph_attn
                                               )

        if return_graph_attn and graph_attn is not None:
            k_mask = (~act_kpm).float()
            graph_attn = mask_attention_rect(graph_attn, q_mask=mask, k_mask=k_mask)
        else:
            graph_attn = None


        # ---- cosine prototype decoding ----
        assert E_act.shape[0] == self.cfg.num_activities

        # --------------------------------------------------
        if "y_activity" not in batch:
           raise ValueError("AR forward currently requires y_activity. Use autoregressive generate() once implemented.")

        H_final = H_struct   # first AR version: no activity feedback refinement
        
        decoder_tokens = build_ar_decoder_inputs(batch["y_activity"], self.cfg.sos_id)
        
        H_ar_in = self._build_ar_inputs(decoder_tokens, H_final)
        
        causal_mask = build_causal_attn_mask(L, H_ar_in.device)
        key_padding_mask = (mask <= 0.0)
        
        H_ar, ar_attn_list = self.ar_decoder(
            H_ar_in,
            key_padding_mask=key_padding_mask,
            attn_mask=causal_mask,
            return_all_attn=return_all_attn
        )
        
        activity_logits_1 = self._decode_activity_logits(H_ar, E_act)
        activity_logits = activity_logits_1
        
        if use_teacher_forcing_transition:
            activity_logits = self._apply_transition_bias_train(activity_logits, decoder_tokens)
         
        # --------------------------------------------------
        # stage 3: generation branch
        # --------------------------------------------------

        # --------------------------------------------------
        # length branch
        # --------------------------------------------------
        case_emb_gen = masked_mean(H_final, mask, dim=1)
        length_logits = self.length_head(case_emb_gen)

        # --------------------------------------------------
        # attribute + time branch
        # default: no teacher forcing for attributes
        # --------------------------------------------------
        if use_teacher_forcing_attributes and ("y_activity" in batch):
            act_ids_attr = batch["y_activity"].clamp(min=0)
        else:
            act_ids_attr = activity_logits.argmax(dim=-1)
        
        act_emb_attr = E_act[act_ids_attr]
        event_attr_in = torch.cat([H_ar, act_emb_attr], dim=-1)
        time_pred = self.time_head(event_attr_in).squeeze(-1)

        # sequence-level heads
        if use_teacher_forcing_attributes and ("y_activity" in batch):
            seq_pool_mask = (mask.bool() & (batch["y_activity"] != self.cfg.eos_id)).float()
        else:
            seq_pool_mask = ((act_ids_attr != self.cfg.eos_id) & mask.bool()).float()

        case_emb_attr = masked_mean(H_ar, seq_pool_mask, dim=1)
        act_emb_case_attr = masked_mean(act_emb_attr, seq_pool_mask, dim=1)

        seq_attr_in = torch.cat([case_emb_attr, act_emb_case_attr], dim=-1)
        
        event_cat_out = {
            name: head(event_attr_in)
            for name, head in self.cat_heads.items()
        }

        event_num_out = {
            name: head(event_attr_in).squeeze(-1)
            for name, head in self.num_heads.items()
        }

        seq_cat_out = {
            name: head(seq_attr_in)
            for name, head in self.seq_cat_heads.items()
        }

        seq_num_out = {
            name: head(seq_attr_in).squeeze(-1)
            for name, head in self.seq_num_heads.items()
        }


        return {
            "H_seq": H_seq,
            "H_cond": H_final,
            "H_ar": H_ar,
            "activity_logits_1": activity_logits_1,
            "activity_logits": activity_logits,
            "time_pred": time_pred,
            "length_logits": length_logits,
            "encoder_attn": tr_attn_list,
            "transformer_attn": ar_attn_list,
            "graph_attn": graph_attn,
            "event_categorical": event_cat_out,
            "event_numeric": event_num_out,
            "seq_categorical": seq_cat_out,
            "seq_numeric": seq_num_out
        }

    def compute_loss(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        adj_matrix: Optional[torch.Tensor] = None,
        w_activity: Optional[float] = None,
        w_time: Optional[float] = None,
        w_length: Optional[float] = None,
        w_transition_penalty: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        
        cfg = self.cfg
        wa = cfg.w_activity if w_activity is None else w_activity
        wt = cfg.w_time if w_time is None else w_time
        wl = cfg.w_length if w_length is None else w_length
        wtp = cfg.w_transition_penalty if w_transition_penalty is None else w_transition_penalty

        mask = batch["mask"]                                  # (B, L)
        valid_mask = mask.bool()

        logits = outputs["activity_logits"]                   # (B, L, A)
        y_act = batch["y_activity"]                           # (B, L)
    
        B, L, A = logits.shape
        lengths = batch["length"].to(logits.device).long().clamp(min=1, max=L)
    
        eos_id = cfg.eos_id 
    
        # --------------------------------------------------
        # weighted activity loss (terminal + preterminal)
        # --------------------------------------------------
        
        act_loss_per_pos = F.cross_entropy(
            logits.transpose(1, 2),      # (B, A, L)
            y_act,
            reduction="none"
        )                                # (B, L)
        
        mask0 = valid_mask.float()
        
        final_pos = lengths - 1
        pre_pos = (lengths - 2).clamp(min=0)
    
        final_mask = torch.zeros_like(mask0)
        pre_mask = torch.zeros_like(mask0)
    
        final_mask.scatter_(1, final_pos.unsqueeze(1), 1.0)
        pre_mask.scatter_(1, pre_pos.unsqueeze(1), 1.0)
    
        final_mask = final_mask * mask0
        pre_mask = pre_mask * mask0
        pre_mask = pre_mask * (1.0 - final_mask)
    
        pos_weights = mask0.clone()
        pos_weights = pos_weights + (cfg.w_final_activity - 1.0) * final_mask
        pos_weights = pos_weights + (cfg.w_preterminal_activity - 1.0) * pre_mask
    
        act_loss = (
            (act_loss_per_pos * pos_weights).sum()
            / pos_weights.sum().clamp_min(1.0)
        )
    
        final_activity_loss = (
            (act_loss_per_pos * final_mask).sum()
            / final_mask.sum().clamp_min(1.0)
        )
    
        preterminal_activity_loss = (
            (act_loss_per_pos * pre_mask).sum()
            / pre_mask.sum().clamp_min(1.0)
        )

        # --------------------------------------------------
        # non-EOS mask for auxiliary event/time heads
        # --------------------------------------------------
        non_eos_mask = valid_mask & (y_act != eos_id)  
        
        # --------------------------------------------------
        # time loss (exclude EOS)
        # --------------------------------------------------
        pred_time = outputs["time_pred"]
        y_time = batch["y_time_log"]
    
        if non_eos_mask.any():
            time_loss = F.smooth_l1_loss(
                pred_time[non_eos_mask],
                y_time[non_eos_mask],
                reduction="mean",
                beta=0.5
            )
        else:
            time_loss = torch.tensor(0.0, device=mask.device)

        # --------------------------------------------------
        # length loss
        # lengths now include EOS, which is correct
        # --------------------------------------------------
        y_len = batch["length"].long()       # (B,)
        y_len_clipped = (y_len.clamp(1, cfg.length_bins) - 1)
        length_loss = F.cross_entropy(outputs["length_logits"], y_len_clipped, reduction="mean")
        
        # --------------------------------------------------
        # event categorical losses (exclude EOS)
        # --------------------------------------------------
        event_cat_loss = torch.tensor(0.0, device=mask.device)
        cat_dict = outputs.get("event_categorical", {})
        if cat_dict:
            for name, logits_attr in cat_dict.items():
                targets = batch["event_cat"][name]
                if non_eos_mask.any():
                    event_cat_loss = event_cat_loss + F.cross_entropy(
                        logits_attr[non_eos_mask],
                        targets[non_eos_mask],
                        reduction="mean",
                        label_smoothing=0.05
                    )
            event_cat_loss = event_cat_loss / max(1, len(cat_dict))
            
        # --------------------------------------------------
        # event numeric losses (exclude EOS)
        # --------------------------------------------------
        event_num_loss = torch.tensor(0.0, device=mask.device)
        num_dict = outputs.get("event_numeric", {})
        if num_dict:
            for name, preds_attr in num_dict.items():
                targets = batch["event_numeric"][name]
                if non_eos_mask.any():
                    event_num_loss = event_num_loss + F.mse_loss(
                        preds_attr[non_eos_mask],
                        targets[non_eos_mask],
                        reduction="mean"
                    )
            event_num_loss = event_num_loss / max(1, len(num_dict))
            
        # --------------------------------------------------
        # sequence losses
        # --------------------------------------------------
        seq_cat_loss = torch.tensor(0.0, device=mask.device)
        seq_cat_dict = outputs.get("seq_categorical", {})
        if seq_cat_dict:
            for name, logits_attr in seq_cat_dict.items():
                targets = batch["seq_cat"][name].to(mask.device)
                seq_cat_loss = seq_cat_loss + F.cross_entropy(logits_attr, targets, reduction="mean", label_smoothing=0.05)
            seq_cat_loss = seq_cat_loss / max(1, len(seq_cat_dict))
    
        seq_num_loss = torch.tensor(0.0, device=mask.device)
        seq_num_dict = outputs.get("seq_numeric", {})
        if seq_num_dict:
            for name, preds_attr in seq_num_dict.items():
                targets = batch["seq_numeric"][name].to(mask.device).float()
                seq_num_loss = seq_num_loss + F.mse_loss(preds_attr, targets, reduction="mean")
            seq_num_loss = seq_num_loss / max(1, len(seq_num_dict))

        # --------------------------------------------------
        # transition penalty
        # valid pairs inside the supervised sequence, including real->EOS
        # --------------------------------------------------
        trans_pen = torch.tensor(0.0, device=mask.device)
        if wtp > 0.0 and adj_matrix is not None:
            probs = F.softmax(logits, dim=-1)                  # (B, L, A)
            adj = adj_matrix.to(probs.device)
            invalid = 1.0 - adj
        
            pair_mask = valid_mask[:, :-1] & valid_mask[:, 1:]   # includes real->EOS, excludes EOS->PAD
            if pair_mask.any():
                p_t = probs[:, :-1, :]
                p_n = probs[:, 1:, :]
    
                A = probs.shape[-1]
                k = min(10, A)
    
                topk_t_vals, topk_t_idx = torch.topk(p_t, k, dim=-1)
                topk_n_vals, topk_n_idx = torch.topk(p_n, k, dim=-1)
    
                prob_outer = topk_t_vals.unsqueeze(-1) * topk_n_vals.unsqueeze(-2)
    
                src_idx = topk_t_idx.unsqueeze(-1).expand(-1, -1, -1, k)
                dst_idx = topk_n_idx.unsqueeze(-2).expand(-1, -1, k, -1)
                invalid_pairs = invalid[src_idx, dst_idx]
    
                penalty_mass = prob_outer * invalid_pairs
                trans_pen = (
                    penalty_mass.sum(dim=(-1, -2))[pair_mask].mean()
                )
            else:
                trans_pen = torch.tensor(0.0, device=mask.device)
 
        total = (
            wa * act_loss +
            wt * time_loss +
            wl * length_loss +
            wtp * trans_pen +
            cfg.w_event_cat * event_cat_loss +
            cfg.w_event_num * event_num_loss +
            cfg.w_seq_cat * seq_cat_loss +
            cfg.w_seq_num * seq_num_loss
        )

        return {
            "loss": total,
            "activity_loss": act_loss.detach(),
            "final_activity_loss": final_activity_loss.detach(),
            "preterminal_activity_loss": preterminal_activity_loss.detach(),
            "time_loss": time_loss.detach(),
            "length_loss": length_loss.detach(),
            "transition_penalty": trans_pen.detach(),
            "event_cat_loss": event_cat_loss.detach(),
            "event_num_loss": event_num_loss.detach(),
            "seq_cat_loss": seq_cat_loss.detach(),
            "seq_num_loss": seq_num_loss.detach()
        }
        
    @torch.no_grad()
    def generate(
        self,
        sequence_head: torch.Tensor,
        E_act: torch.Tensor,
        temperature: float = 1.0,
        greedy: bool = True,
        length_override: Optional[torch.Tensor] = None,   # real-event length, excludes EOS
        adj_matrix: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        time_sigma: float = 0.05,
        bigram_prior: Optional[torch.Tensor] = None,
        time_min_scaled: Optional[float] = None,
        time_max_scaled: Optional[float] = None,
        return_attn: bool = False
    ) -> Dict[str, Any]:
    
        B = sequence_head.shape[0]
        L = self.cfg.L_max
        device = sequence_head.device
        A = self.cfg.num_activities
    
        pad_id = self.cfg.pad_id
        unk_id = self.cfg.unk_id
        eos_id = self.cfg.eos_id
        sos_id = self.cfg.sos_id
    
        if time_min_scaled is None:
            time_min_scaled = -10.0
        if time_max_scaled is None:
            time_max_scaled = 10.0
    
        # --------------------------------------------------
        # 1. build encoder-side conditioning once
        # --------------------------------------------------
        full_mask = torch.ones((B, L), device=device, dtype=torch.float32)
    
        H0 = self._build_input(
            sequence_head,
            full_mask,
            drop_length_feature=True,
            z=None
        )
    
        key_padding_mask = (full_mask <= 0.0)
    
        H_seq, enc_attn_list = self.transformer(
            H0,
            key_padding_mask=key_padding_mask,
            attn_mask=None,
            return_all_attn=return_attn
        )
    
        if len(enc_attn_list):
            enc_attn_list = [mask_attention_square(a, full_mask) for a in enc_attn_list]
    
        act_kpm = self._build_activity_key_padding_mask(B, E_act)
        H_cond, graph_attn = self.graph_cross(
            H_seq,
            E_act,
            act_key_padding_mask=act_kpm,
            need_weights=return_attn
        )
    
        if return_attn and graph_attn is not None:
            k_mask = (~act_kpm).float()
            graph_attn = mask_attention_rect(graph_attn, q_mask=full_mask, k_mask=k_mask)
        else:
            graph_attn = None
    
        # optional length budget
        if length_override is not None:
            forced_real_lengths = length_override.to(device).long().clamp(min=1, max=L - 1)
        else:
            forced_real_lengths = None
    
        # --------------------------------------------------
        # 2. prepare transition scores / priors
        # --------------------------------------------------
        pair_scores = torch.zeros((A, A), device=device)
    
        if self.use_transition_bias and self.transition_bias is not None:
            pair_scores = pair_scores + self.cfg.transition_bias_weight * self.transition_bias.detach().to(device)
    
        bigram = None
        if bigram_prior is not None:
            bigram = bigram_prior.to(device).float().clone()
            bigram[:, pad_id] = 0.0
            bigram[:, unk_id] = 0.0
    
            row_sums = bigram.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0
            bigram = bigram / row_sums
    
            pair_scores = pair_scores + torch.log(bigram.clamp_min(1e-9))
    
        # hard constraints on pair scores
        pair_scores[:, pad_id] = -1e9
        pair_scores[:, unk_id] = -1e9
        pair_scores[:, sos_id] = -1e9
    
        pair_scores[pad_id, :] = -1e9
        pair_scores[unk_id, :] = -1e9
        pair_scores[eos_id, :] = -1e9
    
        if adj_matrix is not None:
            adj = adj_matrix.to(device).float()
            valid = adj > 0
            valid[:, eos_id] = True
            pair_scores = torch.where(valid, pair_scores, torch.full_like(pair_scores, -1e9))
    
        # --------------------------------------------------
        # 3. autoregressive rollout
        # --------------------------------------------------
        y_act = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        y_act[:, 0] = sos_id
    
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        eos_step = torch.full((B,), fill_value=-1, dtype=torch.long, device=device)
    
        step_logits_history = torch.full((B, L, A), -1e9, dtype=torch.float32, device=device)
    
        last_ar_attn = None
    
        max_steps = L - 1  # positions after SOS slot in decoder input space
    
        for t in range(max_steps):
            # decoder input is current token table y_act
            H_ar_in = self._build_ar_inputs(y_act, H_cond)
            causal_mask = build_causal_attn_mask(L, device)
    
            H_ar, ar_attn_list = self.ar_decoder(
                H_ar_in,
                key_padding_mask=None,
                attn_mask=causal_mask,
                return_all_attn=return_attn
            )
    
            if return_attn:
                last_ar_attn = ar_attn_list
    
            logits_all = self._decode_activity_logits(H_ar, E_act)   # (B, L, A)
            step_logits = logits_all[:, t, :].clone()                # predict token for position t in target space
    
            prev_tokens = y_act[:, t]
            if self.use_transition_bias and self.transition_bias is not None:
                step_logits = step_logits + self.cfg.transition_bias_weight * self.transition_bias[prev_tokens]
    
            if bigram is not None:
                step_logits = step_logits + torch.log(bigram[prev_tokens].clamp_min(1e-9))
    
            # hard masks
            step_logits[:, pad_id] = -1e9
            step_logits[:, unk_id] = -1e9
            step_logits[:, sos_id] = -1e9
    
            if adj_matrix is not None:
                allowed = adj[prev_tokens] > 0
                allowed[:, eos_id] = True
                step_logits = torch.where(allowed, step_logits, torch.full_like(step_logits, -1e9))
    
            # if forced length is provided:
            # before forced_real_lengths -> forbid EOS
            # at forced_real_lengths -> force EOS
            if forced_real_lengths is not None:
                must_continue = t < forced_real_lengths
                must_stop = t == forced_real_lengths
    
                step_logits[must_continue, eos_id] = -1e9
    
                force_rows = must_stop.nonzero(as_tuple=False).squeeze(-1)
                if force_rows.numel() > 0:
                    step_logits[force_rows, :] = -1e9
                    step_logits[force_rows, eos_id] = 0.0
    
            # once finished, keep EOS frozen
            if finished.any():
                step_logits[finished, :] = -1e9
                step_logits[finished, eos_id] = 0.0
    
            # top-k filter
            if top_k is not None and top_k < A:
                vals, idx = torch.topk(step_logits, top_k, dim=-1)
                filtered = torch.full_like(step_logits, -1e9)
                filtered.scatter_(1, idx, vals)
                step_logits = filtered
    
            step_logits_history[:, t, :] = step_logits
    
            # decode next token
            if greedy:
                next_tok = step_logits.argmax(dim=-1)
            else:
                step_probs = F.softmax(step_logits / max(temperature, 1e-8), dim=-1)
                next_tok = torch.multinomial(step_probs, 1).squeeze(-1)
    
            y_act[:, t + 1] = next_tok
    
            new_eos = (~finished) & (next_tok == eos_id)
            eos_step[new_eos] = t + 1
            finished = finished | new_eos
    
            if finished.all():
                break
    
        # --------------------------------------------------
        # 4. ensure every case terminates
        # --------------------------------------------------
        no_eos = eos_step < 0
        if no_eos.any():
            eos_step[no_eos] = L - 1
            y_act[no_eos, L - 1] = eos_id
    
        # zero everything after first EOS
        final_y = torch.full_like(y_act, pad_id)
    
        lengths = torch.zeros(B, dtype=torch.long, device=device)             # includes EOS
        real_event_lengths = torch.zeros(B, dtype=torch.long, device=device)  # excludes EOS
    
        for b in range(B):
            epos = int(eos_step[b].item())
            final_y[b, 0:epos + 1] = y_act[b, 0:epos + 1]
            lengths[b] = epos + 1
            real_event_lengths[b] = max(epos, 1) - 1 if epos > 0 else 0
    
        y_act = final_y
    
        # --------------------------------------------------
        # 5. convert decoder-state tokens to target tokens
        # y_act layout currently includes SOS at position 0 and EOS somewhere later
        # target layout should be real events from index 0, then EOS, then PAD
        # --------------------------------------------------
        y_target = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        target_lengths = torch.zeros(B, dtype=torch.long, device=device)
        target_real_lengths = torch.zeros(B, dtype=torch.long, device=device)
    
        for b in range(B):
            epos = int(eos_step[b].item())   # EOS position in decoder-token layout
            # tokens 1..epos-1 are real events, token epos is EOS
            real_len = max(epos - 1, 0)
            target_real_lengths[b] = real_len
            target_lengths[b] = min(real_len + 1, L)
    
            if real_len > 0:
                y_target[b, :real_len] = y_act[b, 1:1 + real_len]
            eos_target_pos = real_len
            if eos_target_pos < L:
                y_target[b, eos_target_pos] = eos_id
    
        y_act = y_target
        real_event_lengths = target_real_lengths
        lengths = target_lengths
    
        arange = torch.arange(L, device=device).unsqueeze(0)
        gen_mask = arange < lengths.unsqueeze(1)
        non_eos_gen_mask = gen_mask & (y_act != eos_id)
    
        # --------------------------------------------------
        # 6. one final aligned AR pass on generated sequence
        # --------------------------------------------------
        decoder_tokens_final = build_ar_decoder_inputs(y_act, sos_id)
        H_ar_in = self._build_ar_inputs(decoder_tokens_final, H_cond)
        causal_mask = build_causal_attn_mask(L, device)
    
        H_ar_final, ar_attn_final = self.ar_decoder(
            H_ar_in,
            key_padding_mask=(~gen_mask),
            attn_mask=causal_mask,
            return_all_attn=return_attn
        )
    
        final_activity_logits = self._decode_activity_logits(H_ar_final, E_act)
    
        if self.use_transition_bias and self.transition_bias is not None:
            final_activity_logits = self._apply_transition_bias_train(final_activity_logits, decoder_tokens_final)
    
        # --------------------------------------------------
        # 7. time + attributes from final generated path
        # --------------------------------------------------
        act_emb_attr = E_act[y_act].clone()
        act_emb_attr = act_emb_attr * gen_mask.unsqueeze(-1)
    
        event_attr_in = torch.cat([H_ar_final, act_emb_attr], dim=-1)
    
        raw_time = self.time_head(event_attr_in).squeeze(-1)
        raw_time = torch.nan_to_num(raw_time, nan=0.0, posinf=0.0, neginf=0.0)
        y_time = torch.clamp(raw_time, min=time_min_scaled, max=time_max_scaled)
    
        if not greedy and time_sigma > 0:
            y_time = y_time + torch.randn_like(y_time) * time_sigma
            y_time = torch.clamp(y_time, min=time_min_scaled, max=time_max_scaled)
    
        y_time = torch.nan_to_num(y_time, nan=0.0, posinf=0.0, neginf=0.0)
        y_time = y_time * non_eos_gen_mask.float()
    
        for b in range(B):
            T = int(real_event_lengths[b].item())
            if T > 0:
                y_time[b, 0] = time_min_scaled
    
        event_cat_logits = {
            name: head(event_attr_in)
            for name, head in self.cat_heads.items()
        }
        event_cat_logits = {
            name: torch.where(
                non_eos_gen_mask.unsqueeze(-1),
                logits,
                torch.zeros_like(logits)
            )
            for name, logits in event_cat_logits.items()
        }
    
        event_num_preds = {
            name: head(event_attr_in).squeeze(-1)
            for name, head in self.num_heads.items()
        }
        event_num_preds = {
            name: torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0) * non_eos_gen_mask.float()
            for name, preds in event_num_preds.items()
        }
    
        case_emb_attr = masked_mean(H_ar_final, non_eos_gen_mask.float(), dim=1)
        act_emb_case_attr = masked_mean(act_emb_attr, non_eos_gen_mask.float(), dim=1)
        seq_attr_in = torch.cat([case_emb_attr, act_emb_case_attr], dim=-1)
    
        seq_cat_logits = {
            name: head(seq_attr_in)
            for name, head in self.seq_cat_heads.items()
        }
    
        seq_num_preds = {
            name: torch.nan_to_num(head(seq_attr_in).squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
            for name, head in self.seq_num_heads.items()
        }
    
        result = {
            "lengths": lengths,
            "real_event_lengths": real_event_lengths,
            "mask": gen_mask,
            "non_eos_mask": non_eos_gen_mask,
            "y_activity": y_act,
            "y_time_log": y_time,
            "activity_logits": final_activity_logits,
            "event_cat_logits": event_cat_logits,
            "event_num_preds": event_num_preds,
            "seq_cat_logits": seq_cat_logits,
            "seq_num_preds": seq_num_preds,
        }
    
        if return_attn:
            result["encoder_attn"] = enc_attn_list
            result["transformer_attn"] = ar_attn_final
            result["graph_attn"] = graph_attn
    
        return result
     

# =========================================================
# Factory
# =========================================================

def build_model(
    L_max: int,
    num_activities: int,
    d_case_num: int,
    d_model: int = 256,
    n_layers: int = 4,
    de_layers: int = 2,
    n_heads: int = 4,
    dropout: float = 0.2,
    ff_mult: int = 4,
    ff_mult_x: int = 4,
    gate_init: float = 0.05,
 
    length_bins: int = 64,
    latent_dim: int = 0,
    latent_std: float = 1.0,
    
    # loss weights
    w_activity: float = 1.0,
    w_time: float = 0.1,
    w_length: float = 0.25,
    w_event_cat: float = 0.15,
    w_event_num: float = 0.1,
    # case (sequence-level) prediction weights
    w_seq_cat: float = 0.05,
    w_seq_num: float = 0.05,
    w_transition_penalty: float = 0,

    # decoding stabilizer
    use_activity_bias: bool = True,

    # activity ids reserved
    pad_id: int = 0,
    unk_id: int = 1,
    eos_id: int = 2,
    sos_id: int = 3,
    
    # head
    activity_decoder: str = "cosine_linear", # allow values "cosine_linear", "cosine", "linear"

    # final activity setup
    w_final_activity: float = 5.0,
    w_preterminal_activity: float = 3.0,

    use_activity_feedback: bool = True,
    n_refine_layers: int = 1,
    activity_feedback_gate_init: float = 0.10,
    activity_feedback_temp: float = 0.5,

    # pre transition 
    use_transition_bias: bool = True,
    transition_bias_weight: float = 1.0,
    unary_weight: float = 0.5,

    event_cat_dims: Optional[Dict[str, int]] = None,
    event_num_names: Optional[List[str]] = None,
    seq_cat_dims: Optional[Dict[str, int]] = None,
    seq_num_names: Optional[List[str]] = None    
    
) -> GraphSequenceDoubleAttentionOneShotModel:
    cfg = GraphSequenceDoubleAttentionConfig(
        L_max=L_max,
        num_activities=num_activities,
        d_model=d_model,
        d_case_num=d_case_num,
        n_layers=n_layers,
        de_layers=de_layers,
        n_heads=n_heads,
        dropout=dropout,
        ff_mult=ff_mult,
        ff_mult_x=ff_mult_x,
        gate_init=gate_init,
        
        length_bins=min(length_bins, L_max),
        latent_dim=latent_dim,
        latent_std=latent_std,

        w_activity=w_activity,
        w_time=w_time,
        w_length=w_length,
        w_event_cat=w_event_cat,
        w_event_num=w_event_num,
        w_seq_cat=w_seq_cat,
        w_seq_num=w_seq_num,
        w_transition_penalty=w_transition_penalty,

        use_activity_bias=use_activity_bias,
        pad_id=pad_id,
        unk_id=unk_id,
        eos_id=eos_id,
        sos_id=sos_id,
        
        activity_decoder=activity_decoder,
        
        w_final_activity=w_final_activity,
        w_preterminal_activity=w_preterminal_activity,

        use_activity_feedback=use_activity_feedback,    
        n_refine_layers=n_refine_layers,                
        activity_feedback_gate_init=activity_feedback_gate_init,
        activity_feedback_temp=activity_feedback_temp,

        use_transition_bias=use_transition_bias,
        transition_bias_weight=transition_bias_weight,
        unary_weight = unary_weight,        
    )
    model = GraphSequenceDoubleAttentionOneShotModel(cfg)

    if event_cat_dims or event_num_names:
        model.register_attribute_heads(
            event_cat_dims=event_cat_dims or {},
            event_num_names=event_num_names or []
        )
        
    if seq_cat_dims or seq_num_names:
        model.register_sequence_heads(
            seq_cat_dims=seq_cat_dims or {},
            seq_num_names=seq_num_names or [],
        )

    return model


#-----

def compute_E_act_cache(
    gat_encoder,
    head_graph,
    device,
    normalize: bool = True,
    return_attention: bool = True,
):
    """
    Compute activity embeddings once per epoch (frozen regime).
    """

    gat_encoder.eval()

    edge_index = head_graph['edge_index'].to(device)
    edge_attr  = head_graph['edge_attr'].to(device)

    with torch.no_grad():
        if return_attention:
            E_act, ei, alpha = gat_encoder(
                edge_index,
                edge_attr,
                return_attention=True
            )
        else:
            E_act = gat_encoder(
                edge_index,
                edge_attr,
                return_attention=False
            )
            ei, alpha = None, None

    if normalize:
        E_act = F.normalize(E_act, dim=-1)

    return E_act, ei, alpha

#----
def move_batch_to_device(batch, device):
    """
    Recursively move all tensors inside nested dicts/lists to device.
    """
    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}

    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]

    return batch

def train_one_epoch(
    model,
    gat_encoder,
    train_loader,
    head_graph,
    optimizer,
    scheduler,
    device,
    epoch: int,
    capture_attn: bool = True,
    attn_batch_idx: int = 0,     # capture only this batch
):
    model.train()
    gat_encoder.eval()

    # store one bundle only (avoid memory blow-up)
    E_act, ei_head, alpha_head = compute_E_act_cache(
        gat_encoder,
        head_graph,
        device=device,
        normalize=True,
        return_attention=capture_attn
    )
    
    if capture_attn:
        attn_bundle = {
            "gat_edge_index": ei_head.detach().cpu() if ei_head is not None else None,
            "gat_alpha": alpha_head.detach().cpu() if alpha_head is not None else None,
            "encoder_attn": None,
            "transformer_attn": None,   # keep this as AR decoder attn for compatibility
            "graph_attn": None,
        }
    else:
        attn_bundle = None

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    total_time_rmse = 0
    total_len_acc = 0
    total_seq_cat_acc = 0.0
    total_seq_num_rmse = 0.0
    total_event_cat_acc = 0.0
    total_event_num_rmse = 0.0
    total_final_acc = 0.0
    total_preterminal_acc = 0.0
    total_no_eos_acc = 0.0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        batch = move_batch_to_device(batch, device)

        # ask model to return attentions ONLY for the capture batch
        want_attn = (capture_attn and batch_idx == attn_batch_idx)

        outputs = model(
            batch,
            E_act,
            return_all_attn=want_attn,          # all attention layers for one batch
            return_graph_attn=want_attn     # only compute graph attn when needed
        )

        loss_dict = model.compute_loss(
            outputs,
            batch,
            adj_matrix=head_graph["adj_matrix"].to(device)
        )

        # optional debug print
        if epoch == 0 and batch_idx == 0:
            print("\n===== LOSS BREAKDOWN =====")
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    print(f"{k}: {v.item():.4f}")
            print("==========================\n")

        loss = loss_dict["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )
        optimizer.step()
        scheduler.step()
        
        # ---------- accuracy ----------
        acc = compute_activity_accuracy(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["mask"]
        )

        final_acc = compute_final_activity_accuracy(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["length"]
        )

        preterminal_acc = compute_preterminal_activity_accuracy(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["length"]
        )
        
        total_preterminal_acc += preterminal_acc.item()

        no_eos_acc = compute_activity_accuracy_no_eos(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["mask"],
            eos_id=model.cfg.eos_id
        )
        total_no_eos_acc += no_eos_acc.item()
        
        len_acc = compute_len_accuracy(outputs, batch)
        time_rmse = compute_time_rmse(outputs, batch, eos_id=model.cfg.eos_id)
        total_len_acc += len_acc.item()
        total_time_rmse += time_rmse.item()
        
        event_cat_acc = compute_event_cat_accuracy(outputs, batch, eos_id=model.cfg.eos_id)
        event_num_rmse = compute_event_num_rmse(outputs, batch, eos_id=model.cfg.eos_id)
        total_event_cat_acc += event_cat_acc.item()
        total_event_num_rmse += event_num_rmse.item()
        
        seq_cat_acc = compute_seq_cat_accuracy(outputs, batch)
        seq_num_rmse = compute_seq_num_rmse(outputs, batch)
        total_seq_cat_acc += seq_cat_acc.item()
        total_seq_num_rmse += seq_num_rmse.item()

        total_loss += float(loss.item())
        total_acc += acc.item()
        total_final_acc += final_acc.item()
        n_batches += 1

        # capture attentions once
        if want_attn and attn_bundle is not None:
            attn_bundle["mask"] = batch["mask"].detach().cpu()
            attn_bundle["y_activity"] = batch["y_activity"].detach().cpu()
            attn_bundle["pred_activity"] = outputs["activity_logits"].argmax(dim=-1).detach().cpu()
            attn_bundle["activity_logits_1"] = outputs["activity_logits_1"].detach().cpu()
            attn_bundle["activity_logits"] = outputs["activity_logits"].detach().cpu()

            enc_attn = outputs.get("encoder_attn", None)
            if enc_attn is not None:
                if isinstance(enc_attn, list):
                    attn_bundle["encoder_attn"] = [a.detach().cpu() for a in enc_attn]
                else:
                    attn_bundle["encoder_attn"] = enc_attn.detach().cpu()     
                    
            # transformer attention: list[(B, heads, L, L)] or []
            tr_attn = outputs.get("transformer_attn", None)
            if tr_attn is not None:
                # keep last layer only if list
                if isinstance(tr_attn, list):
                    attn_bundle["transformer_attn"] = [a.detach().cpu() for a in tr_attn]
                else:
                    attn_bundle["transformer_attn"] = tr_attn.detach().cpu()

            # graph cross attention: (B, heads, L, A)
            g_attn = outputs.get("graph_attn", None)
            if g_attn is not None:
                attn_bundle["graph_attn"] = g_attn.detach().cpu()

    mean_loss = total_loss / max(1, n_batches)
    mean_acc =  total_acc / max(1, n_batches)
    
    mean_len_acc = total_len_acc / max(1, n_batches)
    mean_time_rmse = total_time_rmse / max(1, n_batches)
    
    mean_seq_cat_acc = total_seq_cat_acc / max(1, n_batches)
    mean_seq_num_rmse = total_seq_num_rmse / max(1, n_batches)
    
    mean_event_cat_acc = total_event_cat_acc / max(1, n_batches)
    mean_event_num_rmse = total_event_num_rmse / max(1, n_batches)
    mean_final_acc = total_final_acc / max(1, n_batches)
    mean_preterminal_acc = total_preterminal_acc / max(1, n_batches)
    mean_no_eos_acc = total_no_eos_acc / max(1, n_batches)

    E_act_his = E_act.detach().cpu()

    return mean_loss, mean_acc, mean_no_eos_acc, mean_final_acc, mean_preterminal_acc, mean_len_acc, mean_time_rmse, mean_event_cat_acc, mean_event_num_rmse, mean_seq_cat_acc, mean_seq_num_rmse, E_act_his, attn_bundle

def train_one_epoch_joint(
    model,
    gat_encoder,
    train_loader,
    head_graph,
    opt_seq,
    opt_gat,
    sched_seq,
    sched_gat,
    device,
    epoch: int,
    train_gat: bool,
    capture_attn: bool = True,
    attn_batch_idx: int = 0
):
    
    model.train()
    if train_gat:
        gat_encoder.train()
    else:
        gat_encoder.eval()

    # ---- static tensors ----
    adj_matrix = head_graph["adj_matrix"].to(device)

    edge_index = head_graph["edge_index"].to(device)
    edge_attr = head_graph["edge_attr"].to(device)

    # ---- attention container ----
    attn_bundle = {
        "gat_edge_index": None,
        "gat_alpha": None,
        "encoder_attn": None,
        "transformer_attn": None,   # keep this as AR decoder attn for compatibility
        "graph_attn": None,
    } if capture_attn else None    

 
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    total_time_rmse = 0
    total_len_acc = 0
    total_seq_cat_acc = 0.0
    total_seq_num_rmse = 0.0
    total_event_cat_acc = 0.0
    total_event_num_rmse = 0.0
    total_final_acc = 0.0
    total_preterminal_acc = 0.0
    total_no_eos_acc = 0.0

    for batch_idx, batch in enumerate(train_loader):

        opt_seq.zero_grad(set_to_none=True)
        if train_gat:
            opt_gat.zero_grad(set_to_none=True)

        batch = move_batch_to_device(batch, device)

        # ask model to return attentions ONLY for the capture batch
        want_attn = (capture_attn and batch_idx == attn_batch_idx)

        if want_attn:
            E_act, ei_head, alpha_head = gat_encoder(
                    edge_index,
                    edge_attr,
                    return_attention=True
                )
        else:
            E_act = gat_encoder(
                    edge_index,
                    edge_attr,
                    return_attention=False
                )
            ei_head, alpha_head = None, None
        
        E_act = F.normalize(E_act, dim=-1)
        # freeze GAT before unfreeze epoch
        if not train_gat:
            E_act = E_act.detach()
            
        if want_attn and attn_bundle is not None:
            if ei_head is not None:
                attn_bundle["gat_edge_index"] = ei_head.detach().cpu()
            if alpha_head is not None:
                attn_bundle["gat_alpha"] = alpha_head.detach().cpu()
            

        outputs = model(
            batch,
            E_act,
            return_all_attn=want_attn,          # last layer only
            return_graph_attn=want_attn     # only compute graph attn when needed
        )

        loss_dict = model.compute_loss(
            outputs,
            batch,
            adj_matrix=adj_matrix
        )

        # optional debug print
        if epoch == 0 and batch_idx == 0:
            print("\n===== LOSS BREAKDOWN =====")
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    print(f"{k}: {v.item():.4f}")
            print("==========================\n")

        loss = loss_dict["loss"]
        loss.backward()

        if train_gat:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(gat_encoder.parameters()),
                max_norm=5.0
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )    

        opt_seq.step()
        sched_seq.step()        

        if train_gat:
            opt_gat.step()
            sched_gat.step()

        
        # ---------- accuracy ----------
        acc = compute_activity_accuracy(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["mask"]
        )

        final_acc = compute_final_activity_accuracy(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["length"]
        )

        preterminal_acc = compute_preterminal_activity_accuracy(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["length"]
        )
        
        total_preterminal_acc += preterminal_acc.item()

        no_eos_acc = compute_activity_accuracy_no_eos(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["mask"],
            eos_id=model.cfg.eos_id
        )
        total_no_eos_acc += no_eos_acc.item()
        
        len_acc = compute_len_accuracy(outputs, batch)
        time_rmse = compute_time_rmse(outputs, batch, eos_id=model.cfg.eos_id)
        total_len_acc += len_acc.item()
        total_time_rmse += time_rmse.item()
        
        event_cat_acc = compute_event_cat_accuracy(outputs, batch, eos_id=model.cfg.eos_id)
        event_num_rmse = compute_event_num_rmse(outputs, batch, eos_id=model.cfg.eos_id)
        total_event_cat_acc += event_cat_acc.item()
        total_event_num_rmse += event_num_rmse.item()
        
        seq_cat_acc = compute_seq_cat_accuracy(outputs, batch)
        seq_num_rmse = compute_seq_num_rmse(outputs, batch)
        total_seq_cat_acc += seq_cat_acc.item()
        total_seq_num_rmse += seq_num_rmse.item()

        total_loss += float(loss.item())
        total_acc += acc.item()
        total_final_acc += final_acc.item()
        n_batches += 1

        # capture attentions once
        if want_attn and attn_bundle is not None:
            attn_bundle["mask"] = batch["mask"].detach().cpu()
            attn_bundle["y_activity"] = batch["y_activity"].detach().cpu()
            attn_bundle["pred_activity"] = outputs["activity_logits"].argmax(dim=-1).detach().cpu()
            attn_bundle["activity_logits_1"] = outputs["activity_logits_1"].detach().cpu()
            attn_bundle["activity_logits"] = outputs["activity_logits"].detach().cpu()

            enc_attn = outputs.get("encoder_attn", None)
            if enc_attn is not None:
                if isinstance(enc_attn, list):
                    attn_bundle["encoder_attn"] = [a.detach().cpu() for a in enc_attn]
                else:
                    attn_bundle["encoder_attn"] = enc_attn.detach().cpu()
                    
            # transformer attention: list[(B, heads, L, L)] or []
            tr_attn = outputs.get("transformer_attn", None)
            if tr_attn is not None:
                # keep last layer only if list
                if isinstance(tr_attn, list):
                    attn_bundle["transformer_attn"] = [a.detach().cpu() for a in tr_attn]
                else:
                    attn_bundle["transformer_attn"] = tr_attn.detach().cpu()

            # graph cross attention: (B, heads, L, A)
            g_attn = outputs.get("graph_attn", None)
            if g_attn is not None:
                attn_bundle["graph_attn"] = g_attn.detach().cpu()

    mean_loss = total_loss / max(1, n_batches)
    mean_acc =  total_acc / max(1, n_batches)
    
    mean_len_acc = total_len_acc / max(1, n_batches)
    mean_time_rmse = total_time_rmse / max(1, n_batches)
    
    mean_seq_cat_acc = total_seq_cat_acc / max(1, n_batches)
    mean_seq_num_rmse = total_seq_num_rmse / max(1, n_batches)
    
    mean_event_cat_acc = total_event_cat_acc / max(1, n_batches)
    mean_event_num_rmse = total_event_num_rmse / max(1, n_batches)
    mean_final_acc = total_final_acc / max(1, n_batches)
    mean_preterminal_acc = total_preterminal_acc / max(1, n_batches)
    mean_no_eos_acc = total_no_eos_acc / max(1, n_batches)
    E_act_his = E_act.detach().cpu()

    return mean_loss, mean_acc, mean_no_eos_acc, mean_final_acc, mean_preterminal_acc, mean_len_acc, mean_time_rmse, mean_event_cat_acc, mean_event_num_rmse, mean_seq_cat_acc, mean_seq_num_rmse, E_act_his, attn_bundle


@torch.no_grad()
def evaluate(
    model,
    gat_encoder,
    val_loader,
    head_graph,
    device,
    capture_attn: bool = True,
    attn_batch_idx: int = 0,
):
    model.eval()
    gat_encoder.eval()

    E_act, ei_head, alpha_head = compute_E_act_cache(
        gat_encoder,
        head_graph,
        device=device,
        normalize=True,
        return_attention=capture_attn
    )
    
    if capture_attn:
        attn_bundle = {
            "gat_edge_index": ei_head.detach().cpu() if ei_head is not None else None,
            "gat_alpha": alpha_head.detach().cpu() if alpha_head is not None else None,
            "encoder_attn": None,
            "transformer_attn": None,   # keep this as AR decoder attn for compatibility
            "graph_attn": None,
        }
    else:
        attn_bundle = None

    total_loss = 0.0
    total_acc = 0.0
    total_final_acc = 0.0
    n_batches = 0
    total_time_rmse = 0
    total_len_acc = 0
    total_seq_cat_acc = 0.0
    total_seq_num_rmse = 0.0
    total_event_cat_acc = 0.0
    total_event_num_rmse = 0.0
    total_preterminal_acc = 0.0
    total_no_eos_acc = 0.0
    
    for batch_idx, batch in enumerate(val_loader):
        batch = move_batch_to_device(batch, device)

        want_attn = (capture_attn and batch_idx == attn_batch_idx)

        outputs = model(
            batch,
            E_act,
            return_all_attn=want_attn,
            return_graph_attn=want_attn,
            use_teacher_forcing_transition=True,
            use_teacher_forcing_attributes=True
        )

        loss_dict = model.compute_loss(
            outputs,
            batch,
            adj_matrix=head_graph["adj_matrix"].to(device)
        )
        
        acc = compute_activity_accuracy(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["mask"]
        )

        final_acc = compute_final_activity_accuracy(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["length"]
        )

        preterminal_acc = compute_preterminal_activity_accuracy(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["length"]
        )        
        total_preterminal_acc += preterminal_acc.item()

        no_eos_acc = compute_activity_accuracy_no_eos(
            outputs["activity_logits"],
            batch["y_activity"],
            batch["mask"],
            eos_id=model.cfg.eos_id
        )
        total_no_eos_acc += no_eos_acc.item()

        
        len_acc = compute_len_accuracy(outputs, batch)
        time_rmse = compute_time_rmse(outputs, batch, eos_id=model.cfg.eos_id)
        total_len_acc += len_acc.item()
        total_time_rmse += time_rmse.item()
        
        event_cat_acc = compute_event_cat_accuracy(outputs, batch, eos_id=model.cfg.eos_id)
        event_num_rmse = compute_event_num_rmse(outputs, batch, eos_id=model.cfg.eos_id)
        total_event_cat_acc += event_cat_acc.item()
        total_event_num_rmse += event_num_rmse.item()
        
        seq_cat_acc = compute_seq_cat_accuracy(outputs, batch)
        seq_num_rmse = compute_seq_num_rmse(outputs, batch)
        total_seq_cat_acc += seq_cat_acc.item()
        total_seq_num_rmse += seq_num_rmse.item()

        total_loss += float(loss_dict["loss"].item())
        total_acc += acc.item()
        total_final_acc += final_acc.item()
        n_batches += 1

        if want_attn and attn_bundle is not None:
            attn_bundle["mask"] = batch["mask"].detach().cpu()
            attn_bundle["y_activity"] = batch["y_activity"].detach().cpu()
            attn_bundle["pred_activity"] = outputs["activity_logits"].argmax(dim=-1).detach().cpu()
            attn_bundle["activity_logits_1"] = outputs["activity_logits_1"].detach().cpu()
            attn_bundle["activity_logits"] = outputs["activity_logits"].detach().cpu()
            
            enc_attn = outputs.get("encoder_attn", None)
            if enc_attn is not None:
                if isinstance(enc_attn, list):
                    attn_bundle["encoder_attn"] = [a.detach().cpu() for a in enc_attn]
                else:
                    attn_bundle["encoder_attn"] = enc_attn.detach().cpu()
            
            tr_attn = outputs.get("transformer_attn", None)
            if tr_attn is not None:
                if isinstance(tr_attn, list):
                    attn_bundle["transformer_attn"] = [a.detach().cpu() for a in tr_attn]
                else:
                    attn_bundle["transformer_attn"] = tr_attn.detach().cpu()

            g_attn = outputs.get("graph_attn", None)
            if g_attn is not None:
                attn_bundle["graph_attn"] = g_attn.detach().cpu()

    mean_loss = total_loss / max(1, n_batches)
    mean_acc =  total_acc / max(1, n_batches)
    
    mean_len_acc = total_len_acc / max(1, n_batches)
    mean_time_rmse = total_time_rmse / max(1, n_batches)
    
    mean_seq_cat_acc = total_seq_cat_acc / max(1, n_batches)
    mean_seq_num_rmse = total_seq_num_rmse / max(1, n_batches)
    
    mean_event_cat_acc = total_event_cat_acc / max(1, n_batches)
    mean_event_num_rmse = total_event_num_rmse / max(1, n_batches)
    mean_final_acc = total_final_acc / max(1, n_batches)
    mean_preterminal_acc = total_preterminal_acc / max(1, n_batches)
    mean_no_eos_acc = total_no_eos_acc / max(1, n_batches)
    E_act_his = E_act.detach().cpu()

    return mean_loss, mean_acc, mean_no_eos_acc, mean_final_acc, mean_preterminal_acc, mean_len_acc, mean_time_rmse, mean_event_cat_acc, mean_event_num_rmse, mean_seq_cat_acc, mean_seq_num_rmse, E_act_his, attn_bundle


class EarlyStopping:
    """
    Early stopping based on validation metric (higher is better).
    Also handles checkpoint saving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        checkpoint_path: str = "best_model.pt"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path

        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_score, act_acc, model, gat_encoder, optimizer, scheduler, epoch):
        """
        Returns True if checkpoint saved.
        """
        improved = val_score > self.best_score + self.min_delta

        if improved:
            self.best_score = val_score
            self.best_act_acc = act_acc            
            self.counter = 0
            self.save_checkpoint(model, gat_encoder, optimizer, scheduler, epoch)
            return True

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

            return False

    def save_checkpoint(self, model, gat_encoder, optimizer, scheduler, epoch):
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "gat_state": gat_encoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_score": self.best_score,
            "best_act_acc": self.best_act_acc 
        }
        torch.save(state, self.checkpoint_path)

class EarlyStoppingJoint:
    """
    Early stopping based on validation metric (higher is better).
    Also handles checkpoint saving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        checkpoint_path: str = "best_model.pt"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path

        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(
        self,
        val_score,
        act_acc,
        model,
        gat_encoder,
        opt_seq,
        opt_gat=None,
        sched_seq=None,
        sched_gat=None,
        epoch=None
    ):
        """
        Returns True if checkpoint saved.
        """
        improved = val_score > self.best_score + self.min_delta

        if improved:
            self.best_score = val_score
            self.best_act_acc = act_acc            
            self.counter = 0
            self.save_checkpoint(model, gat_encoder, opt_seq, opt_gat, sched_seq, sched_gat, epoch)
            return True

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

            return False

    def save_checkpoint(
        self,
        model,
        gat_encoder,
        opt_seq,
        opt_gat,
        sched_seq,
        sched_gat,
        epoch
    ):
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "gat_state": gat_encoder.state_dict(),
    
            # optimizers
            "opt_seq_state": opt_seq.state_dict() if opt_seq is not None else None,
            "opt_gat_state": opt_gat.state_dict() if opt_gat is not None else None,
    
            # schedulers
            "sched_seq_state": sched_seq.state_dict() if sched_seq is not None else None,
            "sched_gat_state": sched_gat.state_dict() if sched_gat is not None else None,
    
            # metrics
            "best_score": self.best_score,
            "best_act_acc": self.best_act_acc
        }
    
        torch.save(state, self.checkpoint_path)

        
#--------
def split_decay_params(module: nn.Module):
    decay, no_decay = [], []

    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue

        if name.endswith(".bias"):
            no_decay.append(p)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            no_decay.append(p)
        elif "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    return decay, no_decay

def build_optimizer(
    seq_model: nn.Module,
    lr_base: float = 3e-4,
    weight_decay: float = 1e-4,
):
    seq_decay, seq_no_decay = split_decay_params(seq_model)

    param_groups = [
        {"params": seq_decay,    "lr": lr_base, "weight_decay": weight_decay},
        {"params": seq_no_decay, "lr": lr_base, "weight_decay": 0.0},
    ]

    optim = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)
    return optim

def build_optimizer_joint(
    seq_model: nn.Module,
    gat_encoder: nn.Module,
    lr_seq: float = 3e-4,
    lr_gat: float = 3e-5,
    weight_decay: float = 1e-4,
):
    seq_decay, seq_no_decay = split_decay_params(seq_model)
    gat_decay, gat_no_decay = split_decay_params(gat_encoder)

    opt_seq = torch.optim.AdamW(
        [
            {"params": seq_decay, "lr": lr_seq, "weight_decay": weight_decay},
            {"params": seq_no_decay, "lr": lr_seq, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    opt_gat = torch.optim.AdamW(
        [
            {"params": gat_decay, "lr": lr_gat, "weight_decay": weight_decay},
            {"params": gat_no_decay, "lr": lr_gat, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    return opt_seq, opt_gat

#----------
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_mult: float = 0.1, last_epoch: int = -1):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_mult = min_lr_mult
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # starts at 0
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_mult + (1 - self.min_lr_mult) * cosine)
            lrs.append(lr)
        return lrs

#-----
def build_schedulers(opt_seq, opt_gat, warmup_steps, total_steps, min_lr_mult):
    sched_seq = WarmupCosineScheduler(opt_seq, warmup_steps, total_steps, min_lr_mult)
    sched_gat = WarmupCosineScheduler(opt_gat, warmup_steps, total_steps, min_lr_mult)
    return sched_seq, sched_gat

#---------- generating the databse------
def build_bigram_prior(
    df,
    case_index,
    core_event,
    time_col,
    act2id,
    n_act,
    include_eos: bool = True
):
    df = df.copy()
    df = df.sort_values([case_index, time_col])

    counts = np.zeros((n_act, n_act), dtype=np.float32)

    pad = act2id.get("<PAD>")
    unk = act2id.get("<UNK>")
    eos = act2id.get("<EOS>")
    sos = act2id.get("<SOS>")

    for _, g in df.groupby(case_index):
        acts = g[core_event].map(act2id).dropna().astype(int).tolist()

        if len(acts) == 0:
            continue

        # --------------------------------------------------
        # SOS -> first activity
        # --------------------------------------------------
        if sos is not None:
            first = acts[0]
            counts[sos, first] += 1.0

        for i in range(len(acts) - 1):
            a = acts[i]
            b = acts[i + 1]
            counts[a, b] += 1.0

        if include_eos and eos is not None:
            #acts = acts + [eos]
            counts[acts[-1], eos] += 1.0

    valid_rows = np.ones(n_act, dtype=bool)
    for idx in [pad, unk, eos]:
        if idx is not None:
            valid_rows[idx] = False

    counts[valid_rows] += 1e-6

    # remove invalid rows/cols influence
    if pad is not None:
        counts[pad, :] = 0.0
        counts[:, pad] = 0.0

    if unk is not None:
        counts[unk, :] = 0.0
        counts[:, unk] = 0.0

    # EOS should have no outgoing transitions
    if eos is not None:
        counts[eos, :] = 0.0

    # SOS should have no incoming transitions
    if sos is not None:
        counts[:, sos] = 0.0

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    counts = counts / row_sums

    return torch.tensor(counts, dtype=torch.float32)

def invert_vocab(vocab: dict) -> dict:
    return {i: k for k, i in vocab.items()}


@torch.no_grad()
def decode_generation_to_rows(
    gen: dict,
    case_ids: list,
    start_times: list,
    id2activity: dict,
    time_stats: dict,
):
    """
    Decodes only real events.
    Uses provided start_times as the timestamp anchor.

    Returns
    -------
    df_rows : one row per real generated event
    pos_map : dict {case_id: [position_indices_kept]}
    """

    y_act = gen["y_activity"].detach().cpu().numpy().astype(int)
    y_time_log = gen["y_time_log"].detach().cpu().numpy().astype(float)

    if "non_eos_mask" in gen:
        non_eos_mask = gen["non_eos_mask"].detach().cpu().numpy().astype(bool)
    else:
        non_eos_mask = None

    rows = []
    pos_map = {}
    B, L = y_act.shape

    eos_id = next((i for i, a in id2activity.items() if a == "<EOS>"), None)
    pad_id = next((i for i, a in id2activity.items() if a == "<PAD>"), None)
    sos_id = next((i for i, a in id2activity.items() if a == "<SOS>"), None)

    for b in range(B):
        cid = case_ids[b]
        t0 = pd.Timestamp(start_times[b])
        cur_time = t0

        kept_positions = []

        for i in range(L):
            act_id = int(y_act[b, i])

            if eos_id is not None and act_id == eos_id:
                break
            if pad_id is not None and act_id == pad_id:
                break
            if sos_id is not None and act_id == sos_id:
                continue
            if non_eos_mask is not None and not non_eos_mask[b, i]:
                continue

            act = id2activity.get(act_id, "UNK")

            dt_log = y_time_log[b, i] * time_stats["std"] + time_stats["mean"]
            dt_sec = float(np.expm1(dt_log))
            
            if not np.isfinite(dt_sec) or dt_sec < 0:
                dt_sec = 0.0

            if len(kept_positions) == 0:
                dt_sec = 0.0
            else:
                cur_time = cur_time + pd.to_timedelta(dt_sec, unit="s")

            rows.append({
                "case_id": cid,
                "pos": len(kept_positions),
                "src_pos": i,
                "activity": act,
                "time_timestamp": cur_time,
                "delta_seconds": dt_sec,
            })

            kept_positions.append(i)

        pos_map[cid] = kept_positions

    return pd.DataFrame(rows), pos_map

@torch.no_grad()
def decode_sequence_attributes(
    gen: dict,
    case_ids: list,
    seq_cat_decoders: dict = None,
    seq_num_post: dict = None,
):
    seq_cat_decoders = seq_cat_decoders or {}
    seq_num_post = seq_num_post or {}

    rows = []

    seq_cat_preds = {}
    for name, logits in gen.get("seq_cat_logits", {}).items():
        ids = logits.argmax(dim=-1).detach().cpu().numpy()
        id2label = seq_cat_decoders.get(name, None)

        if id2label:
            decoded = [id2label.get(int(x), "UNK") for x in ids]
        else:
            decoded = ids

        seq_cat_preds[name] = decoded

    seq_num_preds = {}
    for name, preds in gen.get("seq_num_preds", {}).items():
        arr = preds.detach().cpu().numpy()
        if name in seq_num_post:
            arr = seq_num_post[name](arr)
        seq_num_preds[name] = arr

    B = len(case_ids)

    for b in range(B):
        row = {"case_id": case_ids[b]}

        for name, arr in seq_cat_preds.items():
            row[name] = arr[b]

        for name, arr in seq_num_preds.items():
            row[name] = arr[b]

        rows.append(row)

    return pd.DataFrame(rows)

@torch.no_grad()
def attach_event_attributes(
    df_rows: pd.DataFrame,
    gen: dict,
    case_ids: list,
    pos_map: dict,
    event_cat_decoders: dict = None,
    event_num_post: dict = None,
):
    event_cat_decoders = event_cat_decoders or {}
    event_num_post = event_num_post or {}

    case_to_batch = {cid: i for i, cid in enumerate(case_ids)}

    cat_logits = gen.get("event_cat_logits", {})
    cat_preds = {}

    for name, logits in cat_logits.items():
        pred_ids = logits.argmax(dim=-1).detach().cpu().numpy()
        id2cat = event_cat_decoders.get(name, None)

        if id2cat:
            decoded = np.array(
                [[id2cat.get(int(x), "UNK") for x in row] for row in pred_ids],
                dtype=object
            )
        else:
            decoded = pred_ids

        cat_preds[name] = decoded

    num_preds = {}
    for name, preds in gen.get("event_num_preds", {}).items():
        arr = preds.detach().cpu().numpy().copy()
        if name in event_num_post:
            arr = event_num_post[name](arr)
        num_preds[name] = arr

    for row_idx in df_rows.index:
        cid = df_rows.at[row_idx, "case_id"]
        src_pos = int(df_rows.at[row_idx, "src_pos"])
        b = case_to_batch[cid]

        for name, arr in cat_preds.items():
            df_rows.at[row_idx, name] = arr[b, src_pos]

        for name, arr in num_preds.items():
            val = arr[b, src_pos]
            df_rows.at[row_idx, name] = val if np.isfinite(val) else np.nan

    return df_rows
    
@torch.no_grad()
def generate_full_log(
    model,
    gat_encoder,
    head_graph,
    batch,
    device,
    vocabs,
    num_stats,
    time_stats,
    temperature,
    greedy,
    top_k,
    time_sigma,
    bigram_prior,
    time_min_scaled,
    time_max_scaled,
    adj_m
):
    case_ids = batch["case_id"]
    start_times = batch["start_time"]

    E_act, _, _ = compute_E_act_cache(
        gat_encoder,
        head_graph,
        device=device,
        normalize=True,
        return_attention=False,
    )

    adj_matrix = head_graph["adj_matrix"].to(device) if adj_m else None
    length_override = batch["length"].detach().cpu()

    gen = model.generate(
        sequence_head=batch["sequence_head"].to(device),
        E_act=E_act,
        temperature=temperature,
        greedy=greedy,
        length_override=length_override,
        adj_matrix=adj_matrix,
        top_k=top_k,
        time_sigma=time_sigma,
        bigram_prior=bigram_prior.to(device) if bigram_prior is not None else None,
        time_min_scaled=time_min_scaled,
        time_max_scaled=time_max_scaled,
        return_attn=False
    )

    print("gen real lengths:", gen["real_event_lengths"][:10].cpu().tolist())
    print("gen total lengths:", gen["lengths"][:10].cpu().tolist())
    print("target real lens:", length_override[:10].tolist())

    df_events, pos_map = decode_generation_to_rows(
        gen,
        case_ids,
        start_times,
        invert_vocab(vocabs["activity"]),
        time_stats=time_stats
    )

    df_events = attach_event_attributes(
        df_events,
        gen,
        case_ids=case_ids,
        pos_map=pos_map,
        event_cat_decoders={
            c: invert_vocab(vocabs["event_cat"][c])
            for c in vocabs["event_cat"]
        },
        event_num_post={
            c: (lambda arr, m=num_stats["event"][c][0], s=num_stats["event"][c][1]: arr * s + m)
            for c in num_stats["event"]
        },
    )

    df_cases = decode_sequence_attributes(
        gen,
        case_ids,
        seq_cat_decoders={
            c: invert_vocab(vocabs["seq_cat"][c])
            for c in vocabs["seq_cat"]
        },
        seq_num_post={
            c: (lambda arr, m=num_stats["seq"][c][0], s=num_stats["seq"][c][1]: arr * s + m)
            for c in num_stats["seq"]
        },
    )

    df_final = df_events.merge(df_cases, on="case_id", how="left")
    return df_final, gen
    
#----------
def sanity_check_one_batch(
    gat_encoder,
    seq_model,
    train_loader,
    head_graph,
    device,
):
    """
    head_graph must contain:
      - edge_index
      - edge_attr
      - adj_matrix
      - num_activities
    """

    print("=== SANITY CHECK START ===")

    # -------------------------------------------------
    # Step 1 — Compute cached E_act (once per epoch)
    # -------------------------------------------------
    edge_index = head_graph["edge_index"].to(device)
    edge_attr = head_graph["edge_attr"].to(device)

    gat_encoder = gat_encoder.to(device)

    with torch.no_grad():
        gat_encoder.eval()
        E_act, _, _ = gat_encoder(edge_index, edge_attr, return_attention=True)
        E_act = torch.nn.functional.normalize(E_act, dim=-1)

    print("E_act shape:", tuple(E_act.shape))

    A = head_graph["num_activities"]
    d = seq_model.cfg.d_model

    assert E_act.shape[0] == A, f"Expected A={A}, got {E_act.shape[0]}"
    assert E_act.shape[1] == d, f"Expected d_model={d}, got {E_act.shape[1]}"

    print("E_act check OK")

    # -------------------------------------------------
    # Step 2 — Fetch ONE batch
    # -------------------------------------------------
    batch = next(iter(train_loader))

    # move batch tensors to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            # nested dicts for attributes
            for kk, vv in v.items():
                batch[k][kk] = vv.to(device)

    B, L = batch["mask"].shape
    print("Batch size:", B)
    print("Seq length:", L)

    # -------------------------------------------------
    # Step 3 — Forward pass
    # -------------------------------------------------
    seq_model = seq_model.to(device)
    seq_model.eval()

    with torch.no_grad():
        outputs = seq_model(
            batch=batch,
            E_act=E_act,
            drop_length_feature=True,
            return_all_attn=False,
            return_graph_attn=False,
        )

    # -------------------------------------------------
    # Step 4 — Shape checks
    # -------------------------------------------------
    act_logits = outputs["activity_logits"]

    print("activity_logits shape:", tuple(act_logits.shape))

    assert act_logits.shape[0] == B
    assert act_logits.shape[1] == L
    assert act_logits.shape[2] == A

    print("Activity logits check OK")

    # optional extra checks (recommended)
    print("time_pred shape:", tuple(outputs["time_pred"].shape))
    print("length_logits shape:", tuple(outputs["length_logits"].shape))

    print("=== SANITY CHECK PASSED ===")        

#--------------------after training-----#

def load_for_inference(model, gat_encoder, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    gat_encoder.load_state_dict(checkpoint["gat_state"])
    model.eval()
    gat_encoder.eval()
    return checkpoint

def load_for_training(model, gat_encoder, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    gat_encoder.load_state_dict(ckpt["gat_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"] + 1

def load_for_training_joint(model, gat_encoder, opt_seq, opt_gat, sched_seq, sched_gat, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    gat_encoder.load_state_dict(ckpt["gat_state"])
    if opt_seq is not None and ckpt["opt_seq_state"] is not None:
        opt_seq.load_state_dict(ckpt["opt_seq_state"])
    if opt_gat is not None and ckpt["opt_gat_state"] is not None:
        opt_gat.load_state_dict(ckpt["opt_gat_state"])    
    if sched_seq is not None and ckpt["sched_seq_state"] is not None:
        sched_seq.load_state_dict(ckpt["sched_seq_state"])    
    if sched_gat is not None and ckpt["sched_gat_state"] is not None:
        sched_gat.load_state_dict(ckpt["sched_gat_state"])
    return ckpt["epoch"] + 1


#--- normalization loss----
def rmse_to_score(x, c=0.5):
    return c / (c + x)