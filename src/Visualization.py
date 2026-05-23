import os
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import torch.nn.functional as F
import networkx as nx
import plotly

def process_transformer_attention(
    attn_list,          # list of (B, H, L, L)
    mask,               # (B, L)
    y_activity,         # (B, L)
    pad_id=0,
    unk_id=1,
    eos_id=2,
    sos_id=3,
    layer_mode="mean",  # "mean" or "last"
    head_mode="mean",   # "mean" or "keep"
    remove_eos=True,
):
    """
    Returns:
        attn: (B, L, L) if head_mode='mean'
              (B, H, L, L) if head_mode='keep'
    """

    if not isinstance(attn_list, (list, tuple)) or len(attn_list) == 0:
        raise ValueError("attn_list must be a non-empty list of tensors")

    # stack layers -> (NL, B, H, L, L)
    attn = torch.stack(attn_list, dim=0)

    # layer aggregation
    if layer_mode == "mean":
        attn = attn.mean(dim=0)          # (B, H, L, L)
    elif layer_mode == "last":
        attn = attn[-1]                  # (B, H, L, L)
    else:
        raise ValueError("layer_mode must be 'mean' or 'last'")

    # remove special sequence positions
    special = (
        (y_activity == pad_id) |
        (y_activity == unk_id) |
        (y_activity == sos_id)
    )
    if remove_eos:
        special = special | (y_activity == eos_id)

    valid_pos = mask.bool() & (~special)   # (B, L)

    # apply masking before head aggregation
    attn = attn * valid_pos[:, None, None, :]   # keys
    attn = attn * valid_pos[:, None, :, None]   # queries

    # renormalize over keys
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    # head aggregation
    if head_mode == "mean":
        attn = attn.mean(dim=1)           # (B, L, L)

        # re-mask and re-normalize after averaging
        attn = attn * valid_pos[:, None, :]
        attn = attn * valid_pos[:, :, None]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    elif head_mode == "keep":
        pass
    else:
        raise ValueError("head_mode must be 'mean' or 'keep'")

    return attn, valid_pos


def extract_case_transformer_attention(
    attn,            # (B, L, L) or (B, H, L, L)
    valid_pos,       # (B, L)
    case_index,
):
    """
    Returns one clean case attention matrix:
        (T, T) if attn is (B, L, L)
        (H, T, T) if attn is (B, H, L, L)
    where T = number of valid positions for that case.
    """

    keep_idx = torch.where(valid_pos[case_index])[0]
    if keep_idx.numel() == 0:
        raise ValueError(f"No valid positions for case_index={case_index}")

    if attn.dim() == 3:
        return attn[case_index][keep_idx][:, keep_idx]
    elif attn.dim() == 4:
        return attn[case_index][:, keep_idx][:, :, keep_idx]
    else:
        raise ValueError("attn must have shape (B,L,L) or (B,H,L,L)")


def process_graph_attention(
    graph_attn,       # (B, H, L, A)
    mask,             # (B, L)
    y_activity,       # (B, L)
    num_activities,   # A
    pad_id=0,
    unk_id=1,
    eos_id=2,
    sos_id=3,
    head_mode="mean", # "mean" or "keep"
    remove_eos=True,
):
    """
    Returns:
        attn: (B, L, A) if head_mode='mean'
              (B, H, L, A) if head_mode='keep'
    """

    if graph_attn is None:
        raise ValueError("graph_attn is None")

    if graph_attn.dim() != 4:
        raise ValueError("graph_attn must have shape (B, H, L, A)")

    # valid query positions
    special_q = (
        (y_activity == pad_id) |
        (y_activity == unk_id) |
        (y_activity == sos_id)
    )
    if remove_eos:
        special_q = special_q | (y_activity == eos_id)

    valid_q = mask.bool() & (~special_q)   # (B, L)

    # valid activity nodes
    valid_k = torch.ones(num_activities, dtype=torch.bool, device=graph_attn.device)
    for idx in [pad_id, unk_id, sos_id] + ([eos_id] if remove_eos else []):
        if idx is not None and 0 <= idx < num_activities:
            valid_k[idx] = False

    # apply masks
    graph_attn = graph_attn * valid_q[:, None, :, None]      # queries
    graph_attn = graph_attn * valid_k[None, None, None, :]   # keys

    # renormalize over activity keys
    graph_attn = graph_attn / graph_attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    # head aggregation
    if head_mode == "mean":
        graph_attn = graph_attn.mean(dim=1)   # (B, L, A)

        # re-mask and re-normalize after averaging
        graph_attn = graph_attn * valid_q[:, :, None]
        graph_attn = graph_attn * valid_k[None, None, :]
        graph_attn = graph_attn / graph_attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    elif head_mode == "keep":
        pass
    else:
        raise ValueError("head_mode must be 'mean' or 'keep'")

    return graph_attn, valid_q, valid_k


def extract_case_graph_attention(
    graph_attn,       # (B, L, A) or (B, H, L, A)
    valid_q,          # (B, L)
    valid_k,          # (A,)
    case_index,
):
    """
    Returns one clean case graph attention:
        (T, K) if graph_attn is (B, L, A)
        (H, T, K) if graph_attn is (B, H, L, A)
    where:
        T = number of valid sequence positions
        K = number of valid activity nodes
    """

    q_idx = torch.where(valid_q[case_index])[0]
    k_idx = torch.where(valid_k)[0]

    if q_idx.numel() == 0:
        raise ValueError(f"No valid query positions for case_index={case_index}")
    if k_idx.numel() == 0:
        raise ValueError("No valid activity nodes after masking")

    if graph_attn.dim() == 3:
        return graph_attn[case_index][q_idx][:, k_idx]
    elif graph_attn.dim() == 4:
        return graph_attn[case_index][:, q_idx][:, :, k_idx]
    else:
        raise ValueError("graph_attn must have shape (B,L,A) or (B,H,L,A)")

def process_gat_attention(
    gat_edge_index,   # (2, E)
    gat_alpha,        # (E, H)
    head_mode="mean", # "mean" or "keep"
):
    if gat_edge_index is None or gat_alpha is None:
        raise ValueError("gat_edge_index and gat_alpha must not be None")

    if head_mode == "mean":
        alpha = gat_alpha.mean(dim=-1)   # (E,)
    elif head_mode == "keep":
        alpha = gat_alpha                # (E, H)
    else:
        raise ValueError("head_mode must be 'mean' or 'keep'")

    return gat_edge_index, alpha


def build_next_activity(y_activity, pad_id=0, unk_id=1, eos_id=2, sos_id=3):
    """
    Build next-token target aligned to current positions.
    Invalid last positions get -1.
    """
    y_next = torch.full_like(y_activity, fill_value=-1)
    y_next[:, :-1] = y_activity[:, 1:]

    invalid_next = (
        (y_next == pad_id) |
        (y_next == unk_id) |
        (y_next == sos_id)
    )
    y_next[invalid_next] = -1
    return y_next

# =========================================================
# Dualstage attention routing plot
# =========================================================

# =========================================================
# small helpers
# =========================================================

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _activity_name(id2activity, idx):
    return str(id2activity.get(int(idx), f"id_{int(idx)}"))

def choose_representative_case(
    attn_pack,
    min_len=8,
    prefer_correct=True,
    avoid_all_same=True,
    exclude_eos=True,
    pad_id=0,
    unk_id=1,
    eos_id=2,
    sos_id=3,
):
    y_true = attn_pack["y_activity"]
    y_pred = attn_pack["pred_activity"]
    mask = attn_pack["mask"]

    if torch.is_tensor(y_true):
        y_true = y_true.cpu()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu()
    if torch.is_tensor(mask):
        mask = mask.cpu()

    B, L = y_true.shape
    rows = []

    for b in range(B):
        yt = y_true[b]
        yp = y_pred[b]
        m = mask[b].bool()

        special = (yt == pad_id) | (yt == unk_id) | (yt == sos_id)
        if exclude_eos:
            special = special | (yt == eos_id)

        valid = m & (~special)
        idx = torch.where(valid)[0]
        if idx.numel() < min_len:
            continue

        ytv = yt[idx]
        ypv = yp[idx]

        rows.append({
            "case_index": b,
            "length": int(idx.numel()),
            "is_correct": bool(torch.all(ytv == ypv).item()),
            "n_unique": int(torch.unique(ytv).numel()),
        })

    if not rows:
        raise ValueError("No suitable case found")

    cand = rows
    if prefer_correct:
        tmp = [r for r in cand if r["is_correct"]]
        if tmp:
            cand = tmp
    if avoid_all_same:
        tmp = [r for r in cand if r["n_unique"] > 1]
        if tmp:
            cand = tmp

    lengths = np.array([r["length"] for r in cand])
    med = np.median(lengths)
    best = min(cand, key=lambda r: abs(r["length"] - med))
    return best["case_index"]

def build_alias_map(activity_ids, id2activity):
    alias_map = {}
    legend_rows = []
    for i, aid in enumerate(activity_ids, start=1):
        alias = f"A{i}"
        alias_map[int(aid)] = alias
        legend_rows.append((alias, _activity_name(id2activity, aid)))
    return alias_map, legend_rows


# =========================================================
# reorder graph attention columns
# =========================================================

def reorder_case_graph_columns(case_gx, valid_activity_ids, true_ids, pred_ids):
    """
    Put case-relevant activities first:
    1) activities appearing in true sequence, in order of first appearance
    2) predicted-only activities, in order of first appearance
    3) remaining valid graph nodes
    """
    valid_activity_ids = [int(x) for x in valid_activity_ids]
    true_ids = [int(x) for x in true_ids]
    pred_ids = [int(x) for x in pred_ids]

    ordered = []
    seen = set()

    for a in true_ids:
        if a in valid_activity_ids and a not in seen:
            ordered.append(a)
            seen.add(a)

    for a in pred_ids:
        if a in valid_activity_ids and a not in seen:
            ordered.append(a)
            seen.add(a)

    for a in valid_activity_ids:
        if a not in seen:
            ordered.append(a)
            seen.add(a)

    old_pos = {aid: i for i, aid in enumerate(valid_activity_ids)}
    new_cols = [old_pos[aid] for aid in ordered]
    case_gx_reordered = case_gx[:, new_cols]

    return case_gx_reordered, ordered

# =========================================================
# Helper functions (assumed to be defined elsewhere)
# process_transformer_attention, process_graph_attention
# extract_case_transformer_attention, extract_case_graph_attention,
# choose_representative_case, build_alias_map, etc.
# =========================================================
def plot_dual_stage_attention_figure(
    attn_pack,
    case_index=None,
    min_len=8,
    save_path=None,
    layer_mode="mean",
    tr_head_mode="mean",
    gx_head_mode="mean",
    remove_eos=True,
    figsize=(16, 11),
    dpi=300,
    cmap_attn="vlag",
    acc_col = 3
):
    import numpy as np
    import torch
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle, Patch
    
    local_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial Unicode MS", "Arial"]
    }

    with plt.style.context(matplotlib.RcParams(local_rc)):

        required = [
            "transformer_attn", "graph_attn", "mask",
            "y_activity", "pred_activity", "id2activity"
        ]
        for k in required:
            if k not in attn_pack or attn_pack[k] is None:
                raise ValueError(f"attn_pack missing required key: {k}")

        if case_index is None:
            case_index = choose_representative_case(
                attn_pack,
                min_len=min_len,
                prefer_correct=True,
                avoid_all_same=True,
                exclude_eos=True,
                pad_id=0,
                unk_id=1,
                eos_id=2,
                sos_id=3,
            )

        id2activity = attn_pack["id2activity"]
        y_true = attn_pack["y_activity"]
        y_pred = attn_pack["pred_activity"]
        mask = attn_pack["mask"]
        tr_attn_list = attn_pack["transformer_attn"]
        graph_attn = attn_pack["graph_attn"]

        # =========================
        # Panel A data
        # =========================
        tr_proc, valid_pos = process_transformer_attention(
            attn_list=tr_attn_list,
            mask=mask,
            y_activity=y_true,
            layer_mode=layer_mode,
            head_mode=tr_head_mode,
            remove_eos=remove_eos,
        )
        case_tr = extract_case_transformer_attention(
            attn=tr_proc,
            valid_pos=valid_pos,
            case_index=case_index,
        )
        case_tr = case_tr.detach().cpu().numpy()

        # =========================
        # Panel B data
        # =========================
        A = int(graph_attn.shape[-1])
        gx_proc, valid_q, valid_k = process_graph_attention(
            graph_attn=graph_attn,
            mask=mask,
            y_activity=y_true,
            num_activities=A,
            head_mode=gx_head_mode,
            remove_eos=remove_eos,
        )
        case_gx = extract_case_graph_attention(
            graph_attn=gx_proc,
            valid_q=valid_q,
            valid_k=valid_k,
            case_index=case_index,
        )
        case_gx = case_gx.detach().cpu().numpy()

        # =========================
        # valid positions / tokens
        # =========================
        vp = valid_pos[case_index]
        if torch.is_tensor(vp):
            vp = vp.cpu()
        pos_idx = torch.where(vp)[0]

        true_ids = y_true[case_index][pos_idx].cpu().numpy().astype(int).tolist()
        pred_ids = y_pred[case_index][pos_idx].cpu().numpy().astype(int).tolist()
        mismatch = [p != t for p, t in zip(pred_ids, true_ids)]

        # =========================
        # valid graph activity nodes and column ordering
        # =========================
        valid_activity_ids = torch.where(valid_k.cpu())[0].numpy().astype(int).tolist()
        
        case_gx, ordered_activity_ids = reorder_case_graph_columns(
            case_gx=case_gx,
            valid_activity_ids=valid_activity_ids,
            true_ids=true_ids,
            pred_ids=pred_ids,
        )
        
        alias_map, legend_rows = build_alias_map(ordered_activity_ids, id2activity)
        x_labels_graph = [alias_map[a] for a in ordered_activity_ids]
        pos_labels = [str(i) for i in range(len(true_ids))]

        # =========================
        # graph attention concentration summary
        # =========================
        eps = 1e-12
        
        # ensure row normalized attention
        case_gx_sum = case_gx.sum(axis=1, keepdims=True)
        case_gx_norm = case_gx / np.maximum(case_gx_sum, eps)
        
        # concentration statistics
        entropy = -(case_gx_norm * np.log(case_gx_norm + eps)).sum(axis=1)
        
        # lookup activity id -> graph column index
        activity_to_col = {int(aid): j for j, aid in enumerate(ordered_activity_ids)}
        
        true_weight = []
        pred_weight = []
        
        for i, (t_id, p_id) in enumerate(zip(true_ids, pred_ids)):
            t_col = activity_to_col.get(int(t_id), None)
            p_col = activity_to_col.get(int(p_id), None)
        
            true_weight.append(case_gx_norm[i, t_col] if t_col is not None else np.nan)
            pred_weight.append(case_gx_norm[i, p_col] if p_col is not None else np.nan)

        # =========================
        # figure layout
        # =========================
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)

        # Outer grid: single column so A/C/D/Legend can really span full width
        outer = fig.add_gridspec(
            nrows=5,
            ncols=1,
            height_ratios=[6.0, 6.2, 1.3, 1.3, 2.2],
            hspace=0.55,
        )

        # Row A: heatmap + colorbar, full width
        gsA = outer[0].subgridspec(1, 2, width_ratios=[40, 1.4], wspace=0.08)
        axA = fig.add_subplot(gsA[0, 0])
        caxA = fig.add_subplot(gsA[0, 1])

        # Row B: left heatmap + colorbar + right table
        gsB = outer[1].subgridspec(1, 3, width_ratios=[34, 1.4, 12], wspace=0.12)
        axB = fig.add_subplot(gsB[0, 0])
        caxB = fig.add_subplot(gsB[0, 1])
        axBR = fig.add_subplot(gsB[0, 2])

        # Full width rows
        axC = fig.add_subplot(outer[2, 0])
        axD = fig.add_subplot(outer[3, 0])
        axL = fig.add_subplot(outer[4, 0])

        # =========================
        # Panel A
        # =========================
        sns.heatmap(
            case_tr,
            ax=axA,
            cbar=True,
            cbar_ax=caxA,
            cmap=cmap_attn,
            square=False,
            linewidths=0.5,
            linecolor="white",
            xticklabels=pos_labels,
            yticklabels=pos_labels,
        )
        axA.set_title("A. Transformer self attention (valid positions)", fontsize=13, pad=8)
        axA.set_xlabel("Key position", fontsize=11)
        axA.set_ylabel("Query position", fontsize=11)
        caxA.set_ylabel("Weight", fontsize=10)
        caxA.yaxis.set_label_position("left")
        caxA.yaxis.tick_left()
        caxA.tick_params(labelsize=8)

        # =========================
        # Panel B left
        # =========================
        sns.heatmap(
            case_gx_norm,
            ax=axB,
            cbar=True,
            cbar_ax=caxB,
            cmap=cmap_attn,
            square=False,
            linewidths=0.5,
            linecolor="white",
            xticklabels=x_labels_graph,
            yticklabels=pos_labels,
        )
        axB.set_title("B. Graph cross attention (positions → activities)", fontsize=13, pad=8)
        axB.set_xlabel("Activity node", fontsize=11)
        axB.set_ylabel("Sequence position", fontsize=11)
        #caxB.set_ylabel("Weight", fontsize=10)
        #caxB.yaxis.set_label_position("left")
        caxB.yaxis.tick_left()
        caxB.tick_params(labelsize=8)

        true_unique = []
        seen = set()
        for a in true_ids:
            if a not in seen and a in ordered_activity_ids:
                true_unique.append(a)
                seen.add(a)
        split_idx = len(true_unique)
        if 0 < split_idx < len(ordered_activity_ids):
            axB.axvline(split_idx - 0.5, linewidth=1.2, color="gray", linestyle="--")

        # =========================
        # Panel B right
        # =========================
        axBR.axis("off")
        axBR.set_title("Graph attention concentration", fontsize=13, pad=8)
        
        table_rows = []
        for i in range(len(true_ids)):
            table_rows.append([
                str(i),
                "Y" if pred_ids[i] == true_ids[i] else "N",
                f"{entropy[i]:.2f}",
                f"{true_weight[i]:.3f}" if not np.isnan(true_weight[i]) else "NA",
                f"{pred_weight[i]:.3f}" if not np.isnan(pred_weight[i]) else "NA",
            ])
        
        col_labels = ["Pos", "Correct", "Entropy",  "W_True", "W_Pred"]
        
        tbl = axBR.table(
            cellText=table_rows,
            colLabels=col_labels,
            cellLoc="center",
            colLoc="center",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 0.95],
        )
        
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.30)
        
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("lightgray")
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f2f2f2")
        
        # highlight incorrect prediction rows
        for i, bad in enumerate(mismatch, start=1):
            if bad:
                for c in range(len(col_labels)):
                    cell = tbl[(i, c)]
                    cell.set_edgecolor("red")
                    cell.set_linewidth(1.4)
        # =========================
        # Panels C and D
        # =========================
        unique_acts = list(dict.fromkeys(true_ids + pred_ids))
        palette = sns.color_palette("tab20", n_colors=len(unique_acts))
        act_color = {act: palette[i] for i, act in enumerate(unique_acts)}

        def draw_colored_strip(ax, ids, title, mismatch_mask=None):
            ax.clear()
            ax.set_facecolor("white")
            n = len(ids)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(-0.5, 0.5)

            for j, act in enumerate(ids):
                rect = Rectangle(
                    (j - 0.5, -0.5), 1, 1,
                    facecolor=act_color.get(act, "gray"),
                    edgecolor="black",
                    linewidth=0.5
                )
                ax.add_patch(rect)
                ax.text(
                    j, 0,
                    alias_map.get(act, str(act)),
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black"
                )

            ax.set_xticks(np.arange(n))
            ax.set_xticklabels(np.arange(n), fontsize=9)
            ax.set_yticks([])
            ax.set_ylabel(title, fontsize=11, rotation=0, labelpad=18, ha="right", va="center")

            if mismatch_mask is not None:
                for j, bad in enumerate(mismatch_mask):
                    if bad:
                        ax.add_patch(
                            Rectangle(
                                (j - 0.5, -0.5), 1, 1,
                                fill=False,
                                linewidth=2,
                                edgecolor="red"
                            )
                        )

            for spine in ax.spines.values():
                spine.set_visible(False)

        draw_colored_strip(axC, pred_ids, "C. Pred", mismatch)
        draw_colored_strip(axD, true_ids, "D. True")
        axD.set_xlabel("Valid position index", fontsize=11)

        # =========================
        # Legend
        # =========================
        legend_patches = []
        for alias, name in legend_rows:
            act_id = next(a for a, al in alias_map.items() if al == alias)
            color = act_color.get(act_id, "gray")
            legend_patches.append(
                Patch(facecolor=color, edgecolor="black", label=f"{alias} = {name}")
            )

        axL.legend(
            handles=legend_patches,
            loc="center",
            ncol=acc_col,
            fontsize=9,
            frameon=False
        )
        axL.axis("off")

        # =========================
        # overall title
        # =========================
        exact_match = all(p == t for p, t in zip(pred_ids, true_ids))
        fig.suptitle(
            f"Dual stage attention analysis for case {case_index}   |   "
            f"valid length = {len(true_ids)}   |   exact match = {exact_match}",
            fontsize=14,
            y=0.985
        )

        fig.subplots_adjust(top=0.94, left=0.055, right=0.98, bottom=0.05)

        if save_path is not None:
            plt.savefig(f"{save_path}.png", dpi=dpi, bbox_inches="tight")
            plt.savefig(f"{save_path}.pdf", dpi=dpi, bbox_inches="tight")
            print(f"Saved: {save_path}.png and {save_path}.pdf")

        plt.show()
##--------the legal successor attention mass figure-------###

def build_relative_position_bins(n_bins=5):
    """
    Returns bin edges on [0, 1].
    Example for 5 bins:
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    """
    return np.linspace(0.0, 1.0, n_bins + 1)


def assign_rel_bin(rel_pos, bin_edges):
    """
    rel_pos in [0, 1]
    returns integer bin id in [0, n_bins-1]
    """
    b = np.searchsorted(bin_edges, rel_pos, side="right") - 1
    b = max(0, min(b, len(bin_edges) - 2))
    return b


def build_legal_successor_attention_table(
    attn_pack,
    adj_matrix,
    num_activities,
    pad_id=0,
    unk_id=1,
    eos_id=2,
    sos_id=3,
    n_bins=5,
    use_pred_as_current=False,
):
    """
    Build a long dataframe with one row per valid position.

    Measures graph cross attention mass assigned to:
        1. legal successors of current activity
        2. illegal nodes
        3. true next activity
        4. predicted next activity

    Notes
    -----
    We use graph cross attention of shape (B, L, A) after averaging heads.
    We exclude positions where current or next token is special.
    We use relative position bins over valid real events only.

    Parameters
    ----------
    use_pred_as_current:
        False -> legal successor set based on true current activity
        True  -> legal successor set based on predicted current activity
        For the paper, use False first. It is cleaner.
    """

    graph_attn_raw = attn_pack["graph_attn"]          # (B, H, L, A) or already processed
    mask = attn_pack["mask"]                          # (B, L)
    y_true = attn_pack["y_activity"]                  # (B, L)
    y_pred = attn_pack["pred_activity"]               # (B, L)

    # Process graph attention with your helper
    graph_attn, valid_q, valid_k = process_graph_attention(
        graph_attn=graph_attn_raw,
        mask=mask,
        y_activity=y_true,
        num_activities=num_activities,
        pad_id=pad_id,
        unk_id=unk_id,
        eos_id=eos_id,
        sos_id=sos_id,
        head_mode="mean",
        remove_eos=True,
    )
    # graph_attn: (B, L, A)

    if torch.is_tensor(adj_matrix):
        adj = adj_matrix.detach().cpu().float()
    else:
        adj = torch.tensor(adj_matrix, dtype=torch.float32)

    graph_attn = graph_attn.detach().cpu()
    valid_q = valid_q.detach().cpu()
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    B, L, A = graph_attn.shape
    assert A == num_activities

    valid_node_mask = torch.ones(A, dtype=torch.bool)
    for idx in [pad_id, unk_id, sos_id, eos_id]:
        if 0 <= idx < A:
            valid_node_mask[idx] = False

    rows = []
    bin_edges = build_relative_position_bins(n_bins=n_bins)

    for b in range(B):
        pos_idx = torch.where(valid_q[b])[0]

        # Need at least 2 real positions to define next activity
        if pos_idx.numel() < 2:
            continue

        # Map valid real positions to compact trace positions 0..T-1
        T = pos_idx.numel()

        for local_t in range(T - 1):
            src_pos = pos_idx[local_t].item()
            nxt_pos = pos_idx[local_t + 1].item()

            cur_true = int(y_true[b, src_pos].item())
            nxt_true = int(y_true[b, nxt_pos].item())
            nxt_pred = int(y_pred[b, nxt_pos].item())

            if use_pred_as_current:
                cur_act = int(y_pred[b, src_pos].item())
            else:
                cur_act = cur_true

            # skip if source or next is special
            if cur_act in [pad_id, unk_id, eos_id, sos_id]:
                continue
            if nxt_true in [pad_id, unk_id, eos_id, sos_id]:
                continue

            attn_vec = graph_attn[b, src_pos].clone()   # (A,)

            # valid activity keys only
            attn_vec[~valid_node_mask] = 0.0
            denom = attn_vec.sum().item()
            if denom <= 0:
                continue
            attn_vec = attn_vec / denom

            legal_mask = (adj[cur_act] > 0)
            legal_mask = legal_mask & valid_node_mask

            # In case adjacency omitted EOS and specials already masked anyway
            legal_mass = float(attn_vec[legal_mask].sum().item())
            illegal_mass = float(attn_vec[(~legal_mask) & valid_node_mask].sum().item())

            true_next_mass = float(attn_vec[nxt_true].item()) if valid_node_mask[nxt_true] else np.nan
            pred_next_mass = float(attn_vec[nxt_pred].item()) if valid_node_mask[nxt_pred] else np.nan

            top1_idx = int(torch.argmax(attn_vec).item())
            top1_mass = float(attn_vec[top1_idx].item())

            sorted_vals, sorted_idx = torch.sort(attn_vec, descending=True)
            top2_mass = float(sorted_vals[1].item()) if len(sorted_vals) > 1 else 0.0
            margin = top1_mass - top2_mass

            rel_pos = local_t / max(1, T - 2) if T > 2 else 0.0
            rel_bin = assign_rel_bin(rel_pos, bin_edges)

            rows.append({
                "case_index": b,
                "src_pos": src_pos,
                "next_pos": nxt_pos,
                "trace_len_real": T,
                "local_t": local_t,
                "rel_pos": rel_pos,
                "rel_bin": rel_bin,
                "cur_true": cur_true,
                "next_true": nxt_true,
                "next_pred": nxt_pred,
                "top1_node": top1_idx,
                "top1_mass": top1_mass,
                "top1_margin": margin,
                "legal_mass": legal_mass,
                "illegal_mass": illegal_mass,
                "true_next_mass": true_next_mass,
                "pred_next_mass": pred_next_mass,
                "top1_is_legal": int(bool(legal_mask[top1_idx].item())),
                "top1_is_true_next": int(top1_idx == nxt_true),
                "top1_is_pred_next": int(top1_idx == nxt_pred),
            })

    df = pd.DataFrame(rows)
    return df


def summarize_attention_mass_by_bin(df):
    """
    Aggregate mean, std, sem by relative bin.
    """
    metrics = ["legal_mass", "illegal_mass", "true_next_mass", "pred_next_mass", "top1_mass", "top1_margin"]

    grouped = df.groupby("rel_bin")[metrics].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped = grouped.reset_index()

    for m in metrics:
        grouped[f"{m}_sem"] = grouped[f"{m}_std"] / np.sqrt(grouped[f"{m}_count"].clip(lower=1))

    return grouped
def plot_combined_attention_mass(
    df,
    n_bins=5,
    figsize=(15, 5.8),
    use_sem=True,
    smooth=False,
    title=None,
    save_path=None,
    dpi=300,
):
    """
    Create a combined figure with two subplots:
        Left: Attention mass decomposition.
        Right: Admissibility gap with target activity attention.
    """

    summary = summarize_attention_mass_by_bin(df)
    x = np.arange(n_bins)

    def maybe_smooth(y):
        y = np.asarray(y, dtype=float)
        if not smooth or len(y) < 3:
            return y
        y2 = y.copy()
        for i in range(1, len(y) - 1):
            y2[i] = 0.25 * y[i - 1] + 0.5 * y[i] + 0.25 * y[i + 1]
        return y2

    # Extract values
    legal = maybe_smooth(summary["legal_mass_mean"].values)
    illegal = maybe_smooth(summary["illegal_mass_mean"].values)
    true_next = maybe_smooth(summary["true_next_mass_mean"].values)
    pred_next = maybe_smooth(summary["pred_next_mass_mean"].values)

    # Standard errors
    legal_err = (summary["legal_mass_sem"] if use_sem else summary["legal_mass_std"]).values
    illegal_err = (summary["illegal_mass_sem"] if use_sem else summary["illegal_mass_std"]).values

    if smooth:
        legal_err = maybe_smooth(legal_err)
        illegal_err = maybe_smooth(illegal_err)

    # Gap and propagated error
    gap = legal - illegal
    gap_err = np.sqrt(legal_err**2 + illegal_err**2)

    # Wider left subplot, narrower right subplot
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    # Short legend labels
    label_legal = "Admissible transition attention"
    label_illegal = "Inadmissible transition attention"
    label_true = "True target activity attention"
    label_pred = "Predicted target activity attention"
    label_gap = "Admissibility gap (Admissible - Inadmissible)"

    # ---------- Left subplot ----------
    ax_left.plot(
        x, legal, "o-",
        color="#2c7bb6",
        linewidth=2,
        markersize=6,
        label=label_legal
    )
    ax_left.fill_between(
        x,
        legal - legal_err,
        legal + legal_err,
        color="#2c7bb6",
        alpha=0.2
    )

    ax_left.plot(
        x, illegal, "s-",
        color="#d7191c",
        linewidth=2,
        markersize=6,
        label=label_illegal
    )
    ax_left.fill_between(
        x,
        illegal - illegal_err,
        illegal + illegal_err,
        color="#d7191c",
        alpha=0.2
    )

    ax_left.plot(
        x, true_next, "^--",
        color="#1a9850",
        linewidth=2,
        markersize=6,
        label=label_true
    )

    ax_left.plot(
        x, pred_next, "D:",
        color="#ff8c00",
        linewidth=2,
        markersize=6,
        label=label_pred
    )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"B{i + 1}" for i in range(n_bins)])
    ax_left.set_xlabel("Relative sequence position bin", fontsize=9)
    ax_left.set_ylabel("Attention weight", fontsize=9)
    ax_left.set_ylim(0, 1)
    ax_left.set_title("(a) Graph cross attention decomposition", fontsize=9)

    # ---------- Right subplot ----------
    ax_right.bar(
        x, gap,
        yerr=gap_err,
        capsize=3,
        color="#fdae61",
        edgecolor="black",
        alpha=0.7,
        label=label_gap
    )

    ax_right.plot(
        x, true_next, "^--",
        color="#1a9850",
        linewidth=2,
        markersize=6,
        label=label_true
    )

    ax_right.plot(
        x, pred_next, "D:",
        color="#ff8c00",
        linewidth=2,
        markersize=6,
        label=label_pred
    )

    ax_right.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([f"B{i + 1}" for i in range(n_bins)])
    ax_right.set_xlabel("Relative sequence position bin", fontsize=9)
    ax_right.set_ylabel("Attention weight", fontsize=9)
    ax_right.set_title("(b) Gap and target activity attention", fontsize=9)

    # ---------- Shared legend ----------
    handles_labels = {}
    for ax in [ax_left, ax_right]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in handles_labels:
                handles_labels[label] = handle

    order = [
        label_legal,
        label_illegal,
        label_true,
        label_pred,
        label_gap,
    ]

    legend_handles = [handles_labels[label] for label in order if label in handles_labels]
    legend_labels = [label for label in order if label in handles_labels]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.4,
        handlelength=2.2,
        handletextpad=0.5
    )

    if title is not None:
        fig.suptitle(title, fontsize=10, y=1.055)

    plt.tight_layout(
        rect=[0, 0.02, 1, 0.90],
        pad=0.5,
        w_pad=1.2
    )

    if save_path is not None:
        base = save_path
        plt.savefig(f"{base}.png", dpi=dpi, bbox_inches="tight")
        plt.savefig(f"{base}.pdf", dpi=dpi, bbox_inches="tight")
        print(f"Saved: {base}.png and {base}.pdf")

    plt.show()
    return 


def plot_combined_attention_mass_comparison(
    df_correct,
    df_incorrect,
    n_bins=7,
    figsize=(16, 7),
    use_sem=True,
    smooth=False,
    group_labels=("Correct next", "Incorrect next"),
    title=None,
    save_path=None,
    dpi=300,
):
    """
    Create a combined figure comparing two groups (correct vs incorrect predictions).
    Handles groups that may have different numbers of bins.
    Legend is merged and placed outside the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    def summarize(df):
        """Compute mean, std, count, and sem per bin."""
        if df.empty:
            return pd.DataFrame(columns=['rel_bin'])
        metrics = ["legal_mass", "illegal_mass", "true_next_mass", "pred_next_mass", "top1_mass", "top1_margin"]
        grouped = df.groupby("rel_bin")[metrics].agg(["mean", "std", "count"])
        grouped.columns = ["_".join(c) for c in grouped.columns]
        grouped = grouped.reset_index()
        for m in metrics:
            grouped[f"{m}_sem"] = grouped[f"{m}_std"] / np.sqrt(grouped[f"{m}_count"].clip(lower=1))
        return grouped

    def maybe_smooth(y):
        y = np.asarray(y, dtype=float)
        if not smooth or len(y) < 3:
            return y
        y2 = y.copy()
        for i in range(1, len(y)-1):
            if np.isfinite(y[i-1]) and np.isfinite(y[i]) and np.isfinite(y[i+1]):
                y2[i] = 0.25*y[i-1] + 0.5*y[i] + 0.25*y[i+1]
        return y2

    # Summaries (keep only bins that exist)
    sum_correct = summarize(df_correct)
    sum_incorrect = summarize(df_incorrect)

    if sum_correct.empty and sum_incorrect.empty:
        raise ValueError("Both DataFrames are empty – cannot plot.")
    if sum_correct.empty:
        print("Warning: df_correct is empty. Plotting only incorrect group.")
    if sum_incorrect.empty:
        print("Warning: df_incorrect is empty. Plotting only correct group.")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    # ----------------------------------------------------------------------
    # Left subplot: line plots with ribbons
    # ----------------------------------------------------------------------
    # Correct group
    if not sum_correct.empty:
        x_c = sum_correct['rel_bin'].values
        # main curves
        leg_c = maybe_smooth(sum_correct["legal_mass_mean"].values)
        ill_c = maybe_smooth(sum_correct["illegal_mass_mean"].values)
        true_c = maybe_smooth(sum_correct["true_next_mass_mean"].values)
        pred_c = maybe_smooth(sum_correct["pred_next_mass_mean"].values)

        # error bounds (no smoothing to preserve original)
        err_leg_c = (sum_correct["legal_mass_sem"] if use_sem else sum_correct["legal_mass_std"]).values
        err_ill_c = (sum_correct["illegal_mass_sem"] if use_sem else sum_correct["illegal_mass_std"]).values

        # Legal mass
        ax_left.plot(x_c, leg_c, 'o-', color='#2c7bb6', linewidth=2, markersize=6,
                     label=f'Admissible transition attention ({group_labels[0]})')
        ax_left.fill_between(x_c, leg_c - err_leg_c, leg_c + err_leg_c,
                             color='#2c7bb6', alpha=0.15)

        # Illegal mass
        ax_left.plot(x_c, ill_c, 's-', color='#d7191c', linewidth=2, markersize=6,
                     label=f'Inadmissible transition attention ({group_labels[0]})')
        ax_left.fill_between(x_c, ill_c - err_ill_c, ill_c + err_ill_c,
                             color='#d7191c', alpha=0.15)

        # True next mass
        ax_left.plot(x_c, true_c, '^--', color='#1a9850', linewidth=2, markersize=6,
                     label=f'True activity attention ({group_labels[0]})')
        # Predicted next mass
        ax_left.plot(x_c, pred_c, 'D:', color='#ff8c00', linewidth=2, markersize=6,
                     label=f'Predicted activity attention ({group_labels[0]})')

    # Incorrect group
    if not sum_incorrect.empty:
        x_i = sum_incorrect['rel_bin'].values
        leg_i = maybe_smooth(sum_incorrect["legal_mass_mean"].values)
        ill_i = maybe_smooth(sum_incorrect["illegal_mass_mean"].values)
        true_i = maybe_smooth(sum_incorrect["true_next_mass_mean"].values)
        pred_i = maybe_smooth(sum_incorrect["pred_next_mass_mean"].values)

        err_leg_i = (sum_incorrect["legal_mass_sem"] if use_sem else sum_incorrect["legal_mass_std"]).values
        err_ill_i = (sum_incorrect["illegal_mass_sem"] if use_sem else sum_incorrect["illegal_mass_std"]).values

        # Legal mass
        ax_left.plot(x_i, leg_i, 'o--', color='#6baed6', linewidth=1.8, markersize=5,
                     label=f'Admissible transition attention ({group_labels[1]})')
        ax_left.fill_between(x_i, leg_i - err_leg_i, leg_i + err_leg_i,
                             color='#6baed6', alpha=0.1)

        # Illegal mass
        ax_left.plot(x_i, ill_i, 's--', color='#fd8d3c', linewidth=1.8, markersize=5,
                     label=f'Indmissible transition attention ({group_labels[1]})')
        ax_left.fill_between(x_i, ill_i - err_ill_i, ill_i + err_ill_i,
                             color='#fd8d3c', alpha=0.1)

        # True next mass
        ax_left.plot(x_i, true_i, '^-.', color='#a6d96a', linewidth=1.8, markersize=5,
                     label=f'True next activity attention ({group_labels[1]})')
        # Predicted next mass
        ax_left.plot(x_i, pred_i, 'D-.', color='#fdae61', linewidth=1.8, markersize=5,
                     label=f'Predicted next activity attention ({group_labels[1]})')

    ax_left.set_xticks(range(n_bins))
    ax_left.set_xticklabels([f'B{i+1}' for i in range(n_bins)])
    ax_left.set_xlabel('Relative sequence position bin')
    ax_left.set_ylabel('Attention weight')
    ax_left.set_ylim(0, 1)
    ax_left.set_title('(a) Graph cross attention decomposition')

    # ----------------------------------------------------------------------
    # Right subplot: grouped bar chart for legality gap
    # ----------------------------------------------------------------------
    def prepare_bar_data(df):
        """Return full-length arrays (n_bins) for gap and its error."""
        if df.empty:
            return np.full(n_bins, np.nan), np.full(n_bins, np.nan)
        leg = df["legal_mass_mean"].values
        ill = df["illegal_mass_mean"].values
        gap = leg - ill
        if use_sem:
            err_leg = df["legal_mass_sem"].values
            err_ill = df["illegal_mass_sem"].values
        else:
            err_leg = df["legal_mass_std"].values
            err_ill = df["illegal_mass_std"].values
        err_gap = np.sqrt(err_leg**2 + err_ill**2)

        bins = df["rel_bin"].values
        gap_full = np.full(n_bins, np.nan)
        err_full = np.full(n_bins, np.nan)
        for b, g, e in zip(bins, gap, err_gap):
            gap_full[b] = g
            err_full[b] = e
        return gap_full, err_full

    gap_c, err_c = prepare_bar_data(sum_correct)
    gap_i, err_i = prepare_bar_data(sum_incorrect)

    bar_width = 0.35
    x = np.arange(n_bins)
    positions_c = x - bar_width/2
    positions_i = x + bar_width/2

    mask_c = ~np.isnan(gap_c)
    mask_i = ~np.isnan(gap_i)

    if mask_c.any():
        ax_right.bar(positions_c[mask_c], gap_c[mask_c], yerr=err_c[mask_c],
                     capsize=3, width=bar_width,
                     color='#2c7bb6', edgecolor='black', alpha=0.7,
                     label=f'Inadmissibility gap ({group_labels[0]})')
    if mask_i.any():
        ax_right.bar(positions_i[mask_i], gap_i[mask_i], yerr=err_i[mask_i],
                     capsize=3, width=bar_width,
                     color='#fd8d3c', edgecolor='black', alpha=0.7,
                     label=f'Admissibility gap ({group_labels[1]})')

    # Overlay lines for true next and predicted next masses
    def get_full_values(df, col):
        if df.empty:
            return np.full(n_bins, np.nan)
        bins = df["rel_bin"].values
        vals = df[col].values
        full = np.full(n_bins, np.nan)
        for b, v in zip(bins, vals):
            full[b] = v
        return maybe_smooth(full)

    true_c_full = get_full_values(sum_correct, "true_next_mass_mean")
    pred_c_full = get_full_values(sum_correct, "pred_next_mass_mean")
    true_i_full = get_full_values(sum_incorrect, "true_next_mass_mean")
    pred_i_full = get_full_values(sum_incorrect, "pred_next_mass_mean")

    ax_right.plot(x, true_c_full, '^--', color='#1a9850', linewidth=2, markersize=6,
                  label=f'True next activity attention ({group_labels[0]})')
    ax_right.plot(x, pred_c_full, 'D:', color='#ff8c00', linewidth=2, markersize=6,
                  label=f'Predicted next activity attention  ({group_labels[0]})')
    ax_right.plot(x, true_i_full, '^-.', color='#a6d96a', linewidth=1.8, markersize=5,
                  label=f'True next activity attention  ({group_labels[1]})')
    ax_right.plot(x, pred_i_full, 'D-.', color='#fdae61', linewidth=1.8, markersize=5,
                  label=f'Predicted next activity attention  ({group_labels[1]})')

    ax_right.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([f'B{i+1}' for i in range(n_bins)])
    ax_right.set_xlabel('Relative sequence position bin')
    ax_right.set_ylabel('Attention Weight')
    ax_right.set_title('(b) Admissibility gap and target activity attention')

    # ----------------------------------------------------------------------
    # Merged legend placed outside the figure
    # ----------------------------------------------------------------------
    # Collect handles and labels from both subplots
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    all_handles = handles_left + handles_right
    all_labels = labels_left + labels_right

    # Remove duplicates (keep first occurrence)
    unique = {}
    for h, l in zip(all_handles, all_labels):
        if l not in unique:
            unique[l] = h
    unique_handles = list(unique.values())
    unique_labels = list(unique.keys())

    # Add the merged legend to the figure
    fig.legend(unique_handles, unique_labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=10, handlelength=1.5)

    # Remove the individual subplot legends (they are now redundant)
    if handles_left:
        ax_left.legend().remove()
    if handles_right:
        ax_right.legend().remove()

    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])   # make room for the top legend

    if save_path is not None:
        base = save_path
        plt.savefig(f"{base}.png", dpi=dpi, bbox_inches='tight')
        plt.savefig(f"{base}.pdf", dpi=dpi, bbox_inches='tight')
        print(f"Saved: {base}.png and {base}.pdf")

    plt.show()
    return

#-------cloud panel----#

# =========================================================
# helpers
# =========================================================

def build_relative_position_bins(n_bins=5):
    return np.linspace(0.0, 1.0, n_bins + 1)


def assign_rel_bin(rel_pos, bin_edges):
    b = np.searchsorted(bin_edges, rel_pos, side="right") - 1
    return max(0, min(b, len(bin_edges) - 2))


def safe_entropy(probs, eps=1e-12):
    probs = np.clip(probs, eps, 1.0)
    return -(probs * np.log(probs)).sum(axis=-1)


def js_divergence(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=-1)
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=-1)
    return 0.5 * (kl_pm + kl_qm)


# =========================================================
# build dataframe
# =========================================================

def build_refinement_shift_table(
    attn_pack,
    pad_id=0,
    unk_id=1,
    eos_id=2,
    sos_id=3,
    n_bins=5,
    remove_eos=True,
):
    logits_1 = attn_pack["activity_logits_1"]
    logits_f = attn_pack["activity_logits"]
    y_true = attn_pack["y_activity"]
    mask = attn_pack["mask"]

    if logits_1 is None or logits_f is None:
        raise ValueError("attn_pack must contain 'activity_logits_1' and 'activity_logits'")

    logits_1 = logits_1.detach().cpu()
    logits_f = logits_f.detach().cpu()
    y_true = y_true.detach().cpu()
    mask = mask.detach().cpu().bool()

    bin_edges = build_relative_position_bins(n_bins=n_bins)

    special = (y_true == pad_id) | (y_true == unk_id) | (y_true == sos_id)
    if remove_eos:
        special = special | (y_true == eos_id)

    valid = mask & (~special)

    probs_1 = F.softmax(logits_1, dim=-1).numpy()
    probs_f = F.softmax(logits_f, dim=-1).numpy()

    pred_1 = logits_1.argmax(dim=-1).numpy()
    pred_f = logits_f.argmax(dim=-1).numpy()
    y_np = y_true.numpy()

    rows = []
    B, L = y_true.shape

    for b in range(B):
        pos_idx = torch.where(valid[b])[0]
        T = pos_idx.numel()

        if T == 0:
            continue

        for local_t, src_pos in enumerate(pos_idx.tolist()):
            y = int(y_np[b, src_pos])

            p1 = probs_1[b, src_pos]
            pf = probs_f[b, src_pos]

            p_true_1 = float(p1[y])
            p_true_f = float(pf[y])

            delta_true = p_true_f - p_true_1
            ent_1 = float(safe_entropy(p1))
            ent_f = float(safe_entropy(pf))
            delta_entropy = ent_f - ent_1
            js = float(js_divergence(p1[None, :], pf[None, :])[0])

            was_correct_1 = int(pred_1[b, src_pos] == y)
            is_correct_f = int(pred_f[b, src_pos] == y)

            rel_pos = local_t / max(1, T - 1) if T > 1 else 0.0
            rel_bin = assign_rel_bin(rel_pos, bin_edges)

            rows.append({
                "case_index": b,
                "src_pos": src_pos,
                "trace_len_real": T,
                "local_t": local_t,
                "rel_pos": rel_pos,
                "rel_bin": rel_bin,
                "true_label": y,
                "pred_1": int(pred_1[b, src_pos]),
                "pred_f": int(pred_f[b, src_pos]),
                "p_true_1": p_true_1,
                "p_true_f": p_true_f,
                "delta_true_prob": delta_true,
                "entropy_1": ent_1,
                "entropy_f": ent_f,
                "delta_entropy": delta_entropy,
                "js_div": js,
                "was_correct_1": was_correct_1,
                "is_correct_f": is_correct_f,
                "provisional_status": "Provisionally Correct" if was_correct_1 == 1 else "Provisionally Misclassified",
            })

    return pd.DataFrame(rows)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D


# ── Palette ────────────────────────────────────────────────────────────────────
# rcParams applied only for the duration of plot_refinement_raincloud_panel()
# via matplotlib.style.context — all other plots are unaffected.
_RAINCLOUD_RC = {
    "font.family":       "serif",
    "font.serif":        ["Georgia", "Times New Roman", "DejaVu Serif"],
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.facecolor":  "#FAFAF8",
    "axes.facecolor":    "#FAFAF8",
    "savefig.facecolor": "#FAFAF8",
}


# ── KDE helper ────────────────────────────────────────────────────────────────

def _simple_gaussian_kde(samples, grid, bw=None):
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.zeros_like(grid)
    if bw is None:
        std = np.std(x, ddof=1) if len(x) > 1 else 1e-3
        bw  = 1.06 * max(std, 1e-3) * (len(x) ** (-1 / 5))
        bw  = max(bw, 1e-3)
    diffs = (grid[:, None] - x[None, :]) / bw
    dens  = np.exp(-0.5 * diffs ** 2).sum(axis=1)
    dens  = dens / (len(x) * bw * np.sqrt(2 * np.pi))
    return dens


# ── Drawing primitives ────────────────────────────────────────────────────────

def _draw_half_density(ax, data, x_center, y_grid, side="left",
                       width=0.18, color="#4C72B0", light_color=None, alpha=0.55):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return
    dens = _simple_gaussian_kde(data, y_grid)
    if dens.max() <= 0:
        return
    dens = dens / dens.max() * width
    lc   = light_color if light_color else color

    if side == "left":
        x_edge = x_center - dens
        ax.fill_betweenx(y_grid, x_center, x_edge,
                         color=lc, alpha=alpha, linewidth=0, zorder=2)
        ax.plot(x_edge, y_grid, color=color, linewidth=0.8, alpha=0.7, zorder=3)
    else:
        x_edge = x_center + dens
        ax.fill_betweenx(y_grid, x_center, x_edge,
                         color=lc, alpha=alpha, linewidth=0, zorder=2)
        ax.plot(x_edge, y_grid, color=color, linewidth=0.8, alpha=0.7, zorder=3)


def _draw_box(ax, data, x_center, color="#4C72B0", width=0.07, alpha=0.88):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return
    bp = ax.boxplot(
        [data],
        positions=[x_center],
        widths=width,
        vert=True,
        patch_artist=True,
        showfliers=False,
        whis=(10, 90),
        zorder=5,
    )
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor=color, alpha=alpha, linewidth=1.4)
    for med in bp["medians"]:
        med.set(color=color, linewidth=2.2)
    for whisk in bp["whiskers"]:
        whisk.set(color=color, linewidth=1.1, linestyle="-")
    for cap in bp["caps"]:
        cap.set(color=color, linewidth=1.1)


def _draw_jitter(ax, data, x_center, color="#4C72B0",
                 spread=0.028, alpha=0.22, size=9, seed=7):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return
    rng = np.random.default_rng(seed)
    x   = x_center + rng.uniform(-spread, spread, size=len(data))
    ax.scatter(x, data, s=size, color=color, alpha=alpha,
               edgecolors="none", zorder=4)


def _style_ax(ax, ylabel, ylim, zero_line=False, hide_bottom_spine=False):
    """Unified axis styling."""
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Grid — horizontal hairlines only
    ax.yaxis.grid(True, color="#E5E5E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    ax.tick_params(axis="both", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=9.5, colors="#666666")

    if zero_line:
        ax.axhline(0.0, color="#888888", linewidth=0.9,
                   linestyle=(0, (4, 3)), zorder=1, alpha=0.8)


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_refinement_raincloud_panel(
    df,
    n_bins=5,
    figsize=(13, 10),
    left_group="Provisionally Misclassified",
    right_group="Provisionally Correct",
    left_color="#E07B39",   # warm amber-orange
    right_color="#3B7DC8",   # cool slate-blue
    y_limits_left=(-0.4, 0.4),
    y_limits_right=(0, 0.3),
    zero_line_left=True,
    zero_line_right=False,
    save_path=None,
    dpi=300,
):
    """Public entry point — applies the raincloud rcParams only for its own
    execution, then restores whatever was active before."""
    
    # Define light colors based on the passed colors
    LEFT_LIGHT = "#F5C49A"
    RIGHT_LIGHT = "#A8CBF0"
    
    import matplotlib
    with plt.style.context(matplotlib.RcParams(_RAINCLOUD_RC)):
        return _raincloud_inner(
            df=df, n_bins=n_bins, figsize=figsize,
            left_group=left_group, right_group=right_group,
            left_color=left_color, right_color=right_color,
            left_light=LEFT_LIGHT, right_light=RIGHT_LIGHT,
            y_limits_left=y_limits_left, y_limits_right=y_limits_right,
            zero_line_left=zero_line_left, zero_line_right=zero_line_right,
            save_path=save_path, dpi=dpi,
        )


def _raincloud_inner(
    df, n_bins, figsize, left_group, right_group,
    left_color, right_color, left_light, right_light,
    y_limits_left, y_limits_right,
    zero_line_left, zero_line_right, save_path, dpi,
):
    required = ["rel_bin", "provisional_status", "delta_true_prob", "js_div"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")

    bin_labels      = [f"B{i+1}" for i in range(n_bins)]
    base_positions  = np.arange(n_bins, dtype=float)
    left_x          = base_positions - 0.20
    right_x         = base_positions + 0.20

    y_grid_left  = np.linspace(y_limits_left[0],  y_limits_left[1],  400)
    y_grid_right = np.linspace(y_limits_right[0], y_limits_right[1], 400)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor="#FAFAF8")

    # Two rows with a thin gap row between them for the divider
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, figure=fig, hspace=0.12)
    ax_top    = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    # ── Top panel: delta_true_prob ────────────────────────────────────────────
    for b in range(n_bins):
        wd = df.loc[(df["rel_bin"] == b) & (df["provisional_status"] == left_group),
                    "delta_true_prob"].dropna().values
        cd = df.loc[(df["rel_bin"] == b) & (df["provisional_status"] == right_group),
                    "delta_true_prob"].dropna().values

        # Density half-violins
        _draw_half_density(ax_top, wd, left_x[b],  y_grid_left, side="left",
                           color=left_color,  light_color=left_light,  alpha=0.5)
        _draw_half_density(ax_top, cd, right_x[b], y_grid_left, side="right",
                           color=right_color, light_color=right_light, alpha=0.5)

        # Jitter (behind boxes)
        _draw_jitter(ax_top, wd, left_x[b],  color=left_color,  spread=0.028,
                     alpha=0.20, size=8, seed=100 + b)
        _draw_jitter(ax_top, cd, right_x[b], color=right_color, spread=0.028,
                     alpha=0.20, size=8, seed=200 + b)

        # Boxes (on top)
        _draw_box(ax_top, wd, left_x[b],  color=left_color,  width=0.075, alpha=0.95)
        _draw_box(ax_top, cd, right_x[b], color=right_color, width=0.075, alpha=0.95)

    _style_ax(ax_top,
              ylabel=r"$\Delta\,p_{\mathrm{true}}$",
              ylim=y_limits_left,
              zero_line=zero_line_left)

    ax_top.set_title(
        "Change in true activity probability after refinement",
        fontsize=12, fontweight="normal", pad=14,
        color="#222222",
    )

    # Bin label ticks hidden (shared via bottom)
    ax_top.set_xticks(base_positions)
    ax_top.set_xticklabels([""] * n_bins)

    # Subtle bin separators
    for b in range(n_bins - 1):
        ax_top.axvline(b + 0.5, color="#E0E0DA", linewidth=0.6, zorder=0)

    # ── Bottom panel: JS divergence ───────────────────────────────────────────
    for b in range(n_bins):
        wd = df.loc[(df["rel_bin"] == b) & (df["provisional_status"] == left_group),
                    "js_div"].dropna().values
        cd = df.loc[(df["rel_bin"] == b) & (df["provisional_status"] == right_group),
                    "js_div"].dropna().values

        _draw_half_density(ax_bottom, wd, left_x[b],  y_grid_right, side="left",
                           color=left_color,  light_color=left_light,  alpha=0.5)
        _draw_half_density(ax_bottom, cd, right_x[b], y_grid_right, side="right",
                           color=right_color, light_color=right_light, alpha=0.5)

        _draw_jitter(ax_bottom, wd, left_x[b],  color=left_color,  spread=0.028,
                     alpha=0.20, size=8, seed=300 + b)
        _draw_jitter(ax_bottom, cd, right_x[b], color=right_color, spread=0.028,
                     alpha=0.20, size=8, seed=400 + b)

        _draw_box(ax_bottom, wd, left_x[b],  color=left_color,  width=0.075, alpha=0.95)
        _draw_box(ax_bottom, cd, right_x[b], color=right_color, width=0.075, alpha=0.95)

    if zero_line_right:
        ax_bottom.axhline(0.0, color="#888888", linewidth=0.9,
                          linestyle=(0, (4, 3)), zorder=1, alpha=0.8)

    _style_ax(ax_bottom,
              ylabel="Jensen–Shannon divergence",
              ylim=y_limits_right,
              zero_line=zero_line_right)

    ax_bottom.set_title(
        "Distributional shift between provisional and final predictions",
        fontsize=13, fontweight="normal", pad=14,
        color="#222222",
    )

    ax_bottom.set_xticks(base_positions)
    ax_bottom.set_xticklabels(bin_labels, fontsize=10, color="#444444")
    ax_bottom.set_xlabel("Relative sequence position bin", fontsize=11, labelpad=10)
    ax_bottom.xaxis.set_label_coords(0.05, -0.08)

    for b in range(n_bins - 1):
        ax_bottom.axvline(b + 0.5, color="#E0E0DA", linewidth=0.6, zorder=0)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor=left_light,  edgecolor=left_color,
              linewidth=1.2, alpha=0.85, label=left_group.capitalize()),
        Patch(facecolor=right_light, edgecolor=right_color,
              linewidth=1.2, alpha=0.85, label=right_group.capitalize()),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.92, 0.05),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=1.6,
        handleheight=0.9,
        columnspacing=2.0,
    )

    fig.patch.set_facecolor("#FAFAF8")
    #plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.subplots_adjust(top=0.95, hspace=0.15)

    if save_path is not None:
        base = save_path
        plt.savefig(f"{base}.png", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.savefig(f"{base}.pdf", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {base}.png  and  {base}.pdf")

    plt.show()
    return 

#----- Sankey---
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

import numpy as np
import pandas as pd
import torch


def compute_can_reach_eos(adj_matrix, eos_id=2):
    """
    Returns a boolean mask where can_reach[i] indicates whether node i
    can reach EOS through at least one directed path in the process graph.
    """
    if torch.is_tensor(adj_matrix):
        adj = adj_matrix.detach().cpu().numpy().astype(bool)
    else:
        adj = np.asarray(adj_matrix).astype(bool)

    n_nodes = adj.shape[0]
    reverse_adj = [[] for _ in range(n_nodes)]

    for src in range(n_nodes):
        for dst in range(n_nodes):
            if adj[src, dst]:
                reverse_adj[dst].append(src)

    can_reach = np.zeros(n_nodes, dtype=bool)
    can_reach[eos_id] = True
    stack = [eos_id]

    while stack:
        node = stack.pop()
        for prev_node in reverse_adj[node]:
            if not can_reach[prev_node]:
                can_reach[prev_node] = True
                stack.append(prev_node)

    return can_reach


def build_transition_rescue_rows_from_gen(
    gen,
    adj_matrix,
    pad_id=0,
    unk_id=1,
    eos_id=2,
    sos_id=3,
    top_k=None,
    n_bins=5,
):
    """
    Build one rescue-analysis row per generated real-event position.

    Parameters
    ----------
    gen : dict
        Output of model.generate(...)

    adj_matrix : torch.Tensor or np.ndarray
        Process-graph adjacency matrix used for transition admissibility.

    Returns
    -------
    rows : list of dict
        Each row describes:
        - raw unary top candidate
        - final structured decoded token
        - transition admissibility category
        - relative trace bin
    """

    if torch.is_tensor(adj_matrix):
        adj = adj_matrix.detach().cpu().bool()
    else:
        adj = torch.tensor(adj_matrix, dtype=torch.bool)

    can_reach_eos = compute_can_reach_eos(adj, eos_id=eos_id)

    y_final = gen["y_activity"].detach().cpu()
    logits = gen["activity_logits"].detach().cpu()
    real_lengths = gen["real_event_lengths"].detach().cpu().long()

    batch_size, _, n_activities = logits.shape
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []

    for b in range(batch_size):
        trace_len = int(real_lengths[b].item())
        if trace_len <= 0:
            continue

        prev_final = sos_id

        for t in range(trace_len):
            unary = logits[b, t].clone()

            unary[pad_id] = -1e9
            unary[unk_id] = -1e9
            unary[eos_id] = -1e9
            unary[sos_id] = -1e9

            if top_k is not None and top_k < n_activities:
                vals, idx = torch.topk(unary, top_k, dim=-1)
                filtered = torch.full_like(unary, -1e9)
                filtered.scatter_(0, idx, vals)
                unary = filtered

            raw_top1 = int(torch.argmax(unary).item())
            final_tok = int(y_final[b, t].item())

            raw_is_admissible = bool(adj[prev_final, raw_top1].item())
            final_is_admissible = bool(adj[prev_final, final_tok].item())
            raw_is_terminally_reachable = bool(can_reach_eos[raw_top1])

            if raw_is_admissible and raw_is_terminally_reachable:
                raw_status = "transition_admissible_and_terminally_reachable"
            elif raw_is_admissible and (not raw_is_terminally_reachable):
                raw_status = "transition_admissible_but_terminally_unreachable"
            else:
                raw_status = "transition_inadmissible"

            changed = (raw_top1 != final_tok)

            if (not changed) and final_is_admissible:
                final_status = "retained_under_structured_decoding"
            elif changed and final_is_admissible and raw_status in [
                "transition_inadmissible",
                "transition_admissible_but_terminally_unreachable",
            ]:
                final_status = "corrected_to_an_admissible_transition"
            elif changed and final_is_admissible and raw_status == "transition_admissible_and_terminally_reachable":
                final_status = "reassigned_among_admissible_transitions"
            else:
                final_status = "other_decoding_outcome"

            rel_pos = t / max(1, trace_len - 1) if trace_len > 1 else 0.0
            rel_bin = np.searchsorted(bin_edges, rel_pos, side="right") - 1
            rel_bin = max(0, min(rel_bin, n_bins - 1))

            rows.append({
                "case_local_index": b,
                "t": t,
                "trace_len_real": trace_len,
                "rel_pos": rel_pos,
                "rel_bin": rel_bin,
                "prev_final": prev_final,
                "raw_top1": raw_top1,
                "final_tok": final_tok,
                "raw_status": raw_status,
                "final_status": final_status,
                "changed": int(changed),
                "raw_is_admissible": int(raw_is_admissible),
                "final_is_admissible": int(final_is_admissible),
                "raw_is_terminally_reachable": int(raw_is_terminally_reachable),
            })

            prev_final = final_tok

    return rows

def plot_transition_rescue(
    df_rescue,
    n_bins=5,
    title="Transition Correction under Structured Decoding",
    figsize=(16, 9),
    save_path=None,
):
    """
    Create a beautiful Sankey diagram with summary table on the right.
    """
    # Convert inches to pixels (approx 100 DPI)
    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)

    # Node labels
    bin_nodes = [f"Bin {i+1}" for i in range(n_bins)]
    raw_nodes = [
        "transition_admissible_and_terminally_reachable",
        "transition_admissible_but_terminally_unreachable",
        "transition_inadmissible",
    ]
    final_nodes = [
        "retained_under_structured_decoding",
        "corrected_to_an_admissible_transition",
        "reassigned_among_admissible_transitions",
        "other_decoding_outcome",
    ]

    raw_labels = {
        "transition_admissible_and_terminally_reachable": "Admissible + EOS reachable",
        "transition_admissible_but_terminally_unreachable": "Admissible but EOS unreachable",
        "transition_inadmissible": "Inadmissible",
    }
    final_labels = {
        "retained_under_structured_decoding": "Retained",
        "corrected_to_an_admissible_transition": "Corrected to admissible",
        "reassigned_among_admissible_transitions": "Reassigned among admissible",
        "other_decoding_outcome": "Other",
    }

    # Colors
    bin_colors = px.colors.qualitative.Set2[:n_bins]
    raw_colors = {
        "transition_admissible_and_terminally_reachable": "#1B7CFA",
        "transition_admissible_but_terminally_unreachable": "#F17D01",
        "transition_inadmissible": "#A03DFC",
    }
    final_colors = {
        "retained_under_structured_decoding": "#47BCFF",
        "corrected_to_an_admissible_transition": "#FF6F00",
        "reassigned_among_admissible_transitions": "#7412FC",
        "other_decoding_outcome": "#6C757D",
    }

    # Node list
    labels = (bin_nodes + [raw_labels[x] for x in raw_nodes] + [final_labels[x] for x in final_nodes])
    node_colors = (bin_colors + [raw_colors[x] for x in raw_nodes] + [final_colors[x] for x in final_nodes])
    node_index = {label: i for i, label in enumerate(labels)}

    # Compute flows
    flow1 = df_rescue.groupby(["rel_bin", "raw_status"]).size().reset_index(name="value")
    flow2 = df_rescue.groupby(["raw_status", "final_status"]).size().reset_index(name="value")

    sources, targets, values, link_colors = [], [], [], []

    # Bin -> Raw
    for _, row in flow1.iterrows():
        bin_label = f"Bin {int(row['rel_bin']) + 1}"
        raw_label = raw_labels[row["raw_status"]]
        val = int(row["value"])
        sources.append(node_index[bin_label])
        targets.append(node_index[raw_label])
        values.append(val)
        color = raw_colors[row["raw_status"]]
        link_colors.append(f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}")

    # Raw -> Final
    for _, row in flow2.iterrows():
        raw_label = raw_labels[row["raw_status"]]
        final_label = final_labels[row["final_status"]]
        val = int(row["value"])
        sources.append(node_index[raw_label])
        targets.append(node_index[final_label])
        values.append(val)
        color = raw_colors[row["raw_status"]]
        link_colors.append(f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}")

    # Build summary table (filter zero)
    raw_totals = flow1.groupby("raw_status")["value"].sum().reset_index()
    raw_totals = raw_totals[raw_totals["value"] > 0]
    raw_totals["raw_label"] = raw_totals["raw_status"].map(raw_labels)

    final_totals = flow2.groupby("final_status")["value"].sum().reset_index()
    final_totals = final_totals[final_totals["value"] > 0]
    final_totals["final_label"] = final_totals["final_status"].map(final_labels)

    # Table text
    table_text = "<b>Summary of Flows</b><br><br><b>Raw Status Totals</b><br>"
    for _, row in raw_totals.iterrows():
        table_text += f"{row['raw_label']}: {row['value']}<br>"
    table_text += "<br><b>Final Outcome Totals</b><br>"
    for _, row in final_totals.iterrows():
        table_text += f"{row['final_label']}: {row['value']}<br>"

    # Create Sankey figure
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            hovertemplate="%{label}<br>Total flow: %{value}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="Flow: %{value}<extra></extra>",
        ),
    ))

    # Add summary table as annotation to the right
    fig.add_annotation(
        text=table_text,
        xref="paper",
        yref="paper",
        x=1.01,            # slightly right of the plot
        y=0.5,
        showarrow=False,
        align="left",
        font=dict(size=10, family="monospace"),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="black",
        borderwidth=1,
        borderpad=8,
        xanchor="left",
        yanchor="middle",
    )

    # Adjust layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Arial", weight="bold"),
            x=0.5,
            xanchor="center",
            y=0.96,          # lower title
        ),
        font=dict(size=11, family="Arial"),
        width=width,
        height=height,
        margin=dict(l=50, r=250, t=50, b=50),  # large right margin for table
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    if save_path:
        fig.write_html(save_path + ".html")
        # Save as PDF (vector format)
        fig.write_image(save_path + ".pdf", scale=2, width=width, height=height)
        fig.write_image(save_path + ".png", scale=2,width=width, height=height)  # needs kaleido
        print(f"Saved: {save_path}.html {save_path}.pdf and {save_path}.png")

    fig.show()
    return 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_transition_rescue_stacked_area(
    df_rescue,
    n_bins=5,
    title="Structured decoding outcomes across relative sequence bins",
    figsize=(10, 6),
    save_path=None,
    dpi=300,
    color_palette="Set2",
    legend_loc='lower left',
    legend_bbox_to_anchor=(0.02, 0.98),
    show_annotations=False,
):
    """
    Create a clean, publication‑ready stacked area plot.
    """
    # Define display names for final statuses (if not already defined)
    FINAL_STATUS_DISPLAY = {
        "retained_under_structured_decoding": "Retained",
        "corrected_to_an_admissible_transition": "Corrected to admissible",
        "reassigned_among_admissible_transitions": "Reassigned among admissible",
        "other_decoding_outcome": "Other",
    }
    
    # Define order of outcomes (consistent with previous plots)
    order = [
        "retained_under_structured_decoding",
        "corrected_to_an_admissible_transition",
        "reassigned_among_admissible_transitions",
        "other_decoding_outcome",
    ]

    # Compute fractions per bin
    per_bin = (
        df_rescue.groupby(["rel_bin", "final_status"])
        .size()
        .reset_index(name="count")
    )
    per_bin["fraction"] = per_bin.groupby("rel_bin")["count"].transform(
        lambda x: x / x.sum()
    )

    # Pivot to get fractions per outcome
    pivot = (
        per_bin.pivot(index="rel_bin", columns="final_status", values="fraction")
        .fillna(0.0)
        .reindex(range(n_bins), fill_value=0.0)
    )
    for col in order:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[order]

    x = np.arange(n_bins)
    bin_labels = [f"Bin {i+1}" for i in x]

    # Colors
    colors = sns.color_palette(color_palette, n_colors=len(order))

    # Seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    fig, ax = plt.subplots(figsize=figsize)

    # Stackplot
    ax.stackplot(
        x,
        [pivot[c].values for c in order],
        labels=[FINAL_STATUS_DISPLAY[c] for c in order],
        colors=colors,
        alpha=0.85,
        edgecolor='none',
    )

    # Grid and spines
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.set_xlabel("Relative sequence position bin", fontsize=11)
    ax.set_ylabel("Proportion of decoded positions", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylim(0, 1.0)

    # Reference line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    # Legend placed inside the plot (top left) with a semi‑transparent background
    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9,
        title="Outcome",
        title_fontsize=10,
        facecolor='white',
        edgecolor='lightgray',
        framealpha=0.9,
    )

    # Optional annotations (disabled by default)
    if show_annotations:
        for i in range(n_bins):
            max_frac = pivot.iloc[i].max()
            if max_frac > 0.5:
                max_outcome = pivot.iloc[i].idxmax()
                display_name = FINAL_STATUS_DISPLAY[max_outcome]
                ax.text(
                    i-0.05, 0.98, f"→ {display_name}",
                    ha='right', va='top', fontsize=8,
                    rotation=90, alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.6, edgecolor='none')
                )

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        base = save_path
        plt.savefig(f"{base}.png", dpi=dpi, bbox_inches='tight')
        plt.savefig(f"{base}.pdf", dpi=dpi, bbox_inches='tight')
        print(f"Saved: {base}.png and {base}.pdf")

    plt.show()
    return 



#----- Ablation----#
import numpy as np
import torch
import matplotlib.pyplot as plt

def compute_E_act_drift_curve(E_act_history):
    """
    E_act_history: list or array-like of shape (E, A, d), but E can differ by regime

    Returns
    -------
    drift : np.ndarray of shape (E,)
        Mean L2 drift relative to epoch 0
    """
    if len(E_act_history) == 0:
        return np.array([], dtype=float)

    tensors = []
    for x in E_act_history:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        tensors.append(x.float())

    E0 = tensors[0]
    drift = []

    for Et in tensors:
        d = torch.norm(Et - E0, dim=-1).mean().item()
        drift.append(d)

    return np.asarray(drift, dtype=float)


def make_regime_bundle(
    name,
    history,
    acc_key="val_act_no_eos_acc",
    E_key="val_E_act",
):
    """
    Builds one regime bundle even if early stopping made epoch counts different.
    """
    acc = np.asarray(history[acc_key], dtype=float)
    drift = compute_E_act_drift_curve(history[E_key])

    n = min(len(acc), len(drift))

    return {
        "name": name,
        "epochs": np.arange(1, n + 1),
        "acc": acc[:n],
        "drift": drift[:n],
    }

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_graph_centric_ablation_panel(
    regime_bundles,
    unfreeze_epochs=[5, 10],
    title_left="Validation activity accuracy excluding EOS across graph training regimes",
    title_right="Graph embedding drift from initialization",
    figsize=(12, 5),
    save_path=None,
    dpi=300,
    acc_ylim=None,
    drift_ylim=None,
):
    """
    Create a publication‑ready side‑by‑side panel comparing frozen/unfrozen GAT training.
    
    Parameters
    ----------
    regime_bundles : list of dict
        Each dict must contain:
            'name' (str), 'epochs' (list/array), 'acc' (list/array), 'drift' (list/array)
    unfreeze_epochs : list
        Epochs where GAT was unfrozen; vertical dashed lines.
    title_left, title_right : str
        Subplot titles.
    figsize : tuple
        Figure size (width, height).
    save_path : str, optional
        Base filename (no extension) to save PNG and PDF.
    dpi : int
        Resolution.
    acc_ylim, drift_ylim : tuple, optional
        Y‑axis limits for each subplot.
    """
    # Seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize, sharex=True)

    # Define styles for up to 4 curves
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    colors = sns.color_palette("Set1", n_colors=4)

    # Find frozen model baseline
    baseline_acc = None
    baseline_name = None
    for bundle in regime_bundles:
        if "frozen" in bundle["name"].lower():
            acc = bundle["acc"]
            if len(acc) > 0:
                baseline_acc = acc[-1]
                baseline_name = bundle["name"]
            break

    # Plot each regime on both subplots
    for idx, bundle in enumerate(regime_bundles):
        name = bundle["name"]
        x = bundle["epochs"]
        acc = bundle["acc"]
        drift = bundle["drift"]

        if len(x) == 0:
            continue

        style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]

        ax_left.plot(x, acc, label=name, linestyle=style, marker=marker,
                     markevery=5, linewidth=2, markersize=5, color=color)
        ax_right.plot(x, drift, label=name, linestyle=style, marker=marker,
                      markevery=5, linewidth=2, markersize=5, color=color)

    # Baseline horizontal line (left plot)
    if baseline_acc is not None:
        ax_left.axhline(y=baseline_acc, color='gray', linestyle=':', linewidth=1.5,
                        alpha=0.7, label=f'{baseline_name}: Main Model')

    # Vertical lines for unfreeze epochs
    for ep in unfreeze_epochs:
        ax_left.axvline(ep, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax_right.axvline(ep, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Left plot styling
    ax_left.set_ylabel("Val activity accuracy excluding EOS", fontsize=11)
    ax_left.set_title(title_left, fontsize=12, pad=10)
    if acc_ylim is not None:
        ax_left.set_ylim(acc_ylim)

    # Right plot styling
    ax_right.set_ylabel("Mean embedding drift", fontsize=11)
    ax_right.set_title(title_right, fontsize=12, pad=10)
    ax_right.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    if drift_ylim is not None:
        ax_right.set_ylim(drift_ylim)

    # Shared x‑axis
    max_epoch = 1
    for bundle in regime_bundles:
        epochs = bundle["epochs"]
        if len(epochs) > 0:
            max_epoch = max(max_epoch, max(epochs))
    ax_left.set_xlim(1, max_epoch)
    ax_right.set_xlim(1, max_epoch)
    ax_right.set_xlabel("Epoch", fontsize=11)

    # Grid and spines
    for ax in [ax_left, ax_right]:
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend (shared, placed above the figure)
    handles, labels = ax_left.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                   ncol=5, frameon=False, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        base = save_path
        plt.savefig(f"{base}.png", dpi=dpi, bbox_inches='tight')
        plt.savefig(f"{base}.pdf", dpi=dpi, bbox_inches='tight')
        print(f"Saved: {base}.png and {base}.pdf")

    plt.show()
    return 

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_gat_attention_panel(
    regime_data,
    remove_nodes=[0, 1],
    figsize=(14, 10),   # width is used, height will be auto adjusted
    save_path=None,
    dpi=300,
    edge_cmap='viridis',
    node_palette='tab20',
    node_size=400,
    edge_thresh=0.01,
    layout_seed=42,
    layout_k=1.8,
    main_title="Comparison of GAT Attention across Training Regimes",
    legend_ncol=8,
    legend_fontsize=9,
):
    """
    1 x 4 panel of GAT attention graphs.
    Keeps the original graph drawing logic unchanged.
    Only fixes the layout so the colorbar and activity legend do not overlap.
    """

    nrows, ncols = 1, 4

    # --- Global node mapping ---
    first_pack = regime_data[0]['attn_pack']
    id2activity = first_pack['id2activity']
    valid_ids = [aid for aid in id2activity if aid not in remove_nodes]

    colors = sns.color_palette(node_palette, n_colors=len(valid_ids))
    node_color_map = {aid: colors[i] for i, aid in enumerate(valid_ids)}

    # --- Global edge weight normalization ---
    all_weights = []
    for reg in regime_data:
        attn_pack = reg['attn_pack']
        _, alpha = process_gat_attention(
            attn_pack['gat_edge_index'],
            attn_pack['gat_alpha'],
            head_mode='mean'
        )
        weights = alpha.cpu().numpy()
        all_weights.extend(weights[weights > edge_thresh])

    norm = Normalize(vmin=min(all_weights), vmax=max(all_weights)) if all_weights else Normalize(vmin=0, vmax=1)

    # --- Styling ---
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'normal'

    # --- Dynamic figure height based on legend size ---
    n_legend_items = len(valid_ids)
    n_legend_rows = int(np.ceil(n_legend_items / legend_ncol))

    # keep plots large, expand figure downward for legend
    plot_row_h = 4.8
    cbar_row_h = 0.55
    legend_row_h = max(1, 0.36 * n_legend_rows)
    title_h = 0.45 if main_title else 0.15

    fig_width = figsize[0]
    fig_height = plot_row_h + cbar_row_h + legend_row_h + title_h

    fig = plt.figure(figsize=(fig_width, fig_height))

    # --- Grid: plots / colorbar / legend ---
    gs = fig.add_gridspec(
        3, ncols,
        height_ratios=[plot_row_h, cbar_row_h, legend_row_h],
        hspace=0.10,
        wspace=0.08,
        left=0.03,
        right=0.98,
        top=0.90 if main_title else 0.96,
        bottom=0.04
    )

    axes = [fig.add_subplot(gs[0, j]) for j in range(ncols)]
    cbar_host_ax = fig.add_subplot(gs[1, :])
    legend_ax = fig.add_subplot(gs[2, :])

    cbar_host_ax.axis('off')
    legend_ax.axis('off')

    # --- Draw each subplot ---
    for idx, reg in enumerate(regime_data[:4]):
        ax = axes[idx]

        attn_pack = reg['attn_pack']
        title = reg['title']

        gat_edge_index, gat_alpha = process_gat_attention(
            attn_pack['gat_edge_index'],
            attn_pack['gat_alpha'],
            head_mode='mean'
        )

        # Build graph (only valid nodes)
        G = nx.DiGraph()
        G.add_nodes_from(valid_ids)

        E = gat_alpha.shape[0]
        edges = [(int(gat_edge_index[0, i]), int(gat_edge_index[1, i])) for i in range(E)]
        weights = gat_alpha.cpu().numpy()

        # Add edges, skipping self loops and low weight edges
        for (u, v), w in zip(edges, weights):
            if u == v:
                continue
            if w > edge_thresh and u in valid_ids and v in valid_ids:
                G.add_edge(u, v, weight=w)

        # Layout
        # unchanged from your original logic
        pos = nx.spring_layout(G, seed=layout_seed, k=layout_k)

        # Draw nodes
        node_colors = [node_color_map[n] for n in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_size,
            node_color=node_colors,
            edgecolors='black',
            linewidths=0.5,
            ax=ax
        )

        # Draw edges
        # unchanged from your original logic
        if G.edges():
            edge_weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
            edge_colors = [norm(w) for w in edge_weights]
            nx.draw_networkx_edges(
                G, pos,
                edge_color=edge_colors,
                edge_cmap=plt.colormaps[edge_cmap],
                width=[w * 5 for w in edge_weights],
                arrows=True,
                arrowstyle='-|>',
                arrowsize=12,
                node_size=node_size,
                ax=ax
            )

        # Numeric labels
        labels = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=9,
            ax=ax,
            font_color='black'
        )

        ax.set_title(title, fontsize=11, pad=8)
        ax.axis('off')

    # --- Colorbar in its own row ---
    sm = ScalarMappable(norm=norm, cmap=plt.colormaps[edge_cmap])
    sm.set_array([])

    cbar_ax = cbar_host_ax.inset_axes([0.38, 0.25, 0.24, 0.45])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Edge weight', fontsize=10)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.label.set_visible(True)
    cbar.ax.tick_params(labelsize=8)

    # --- Legend for activity mapping in its own row ---
    legend_handles = []
    legend_labels = []
    for aid in valid_ids:
        patch = plt.Rectangle((0, 0), 1, 1,
                              facecolor=node_color_map[aid],
                              edgecolor='black')
        legend_handles.append(patch)
        legend_labels.append(f"{aid} = {id2activity[aid]}")

    legend = legend_ax.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95), 
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.2,
        labelspacing=0.50,
        borderaxespad=0.0
    )

    legend_ax.text(
        0.5, 0.98,
        'Activity mapping',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='normal',
        transform=legend_ax.transAxes
    )

    # --- Overall title ---
    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight='normal', y=0.98)

    # --- Save ---
    if save_path:
        base = save_path
        plt.savefig(f"{base}.png", dpi=dpi, bbox_inches='tight')
        plt.savefig(f"{base}.pdf", dpi=dpi, bbox_inches='tight')
        print(f"Saved: {base}.png and {base}.pdf")

    plt.show()
    return 