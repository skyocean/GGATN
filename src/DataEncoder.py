from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, StandardScaler

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

#-----Split-----#

def compute_unseen_activities(train_cases, test_cases, case2acts):
    train_set = set()
    for cid in train_cases:
        train_set.update(case2acts.get(cid, set()))
    test_set = set()
    for cid in test_cases:
        test_set.update(case2acts.get(cid, set()))
    return test_set - train_set

def deterministic_length_stratified_split(
    event,
    case_index,
    activity_col,
    test_size=0.1,
    n_bins=5,
    random_state=42,
    verbose=True,
):

    rng = np.random.RandomState(random_state)

    # --- case lengths ---
    sequence_lengths = event.groupby(case_index).size()

    # --- activity sets per case ---
    case2acts = (
        event.groupby(case_index)[activity_col]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    # --- singleton activity detection ---
    act_case_counts = event.groupby(activity_col)[case_index].nunique()
    singleton_acts = set(act_case_counts[act_case_counts == 1].index.astype(str))

    tmp = event[[case_index, activity_col]].copy()
    tmp[activity_col] = tmp[activity_col].astype(str)

    forced_train = set(
        tmp[tmp[activity_col].isin(singleton_acts)][case_index].unique()
    )

    # --- enforce global longest -> TRAIN ---
    ordered_by_len = sequence_lengths.sort_values(ascending=False).index.tolist()
    if ordered_by_len:
        forced_train.add(ordered_by_len[0])

    train_cases = set(forced_train)
    test_cases = set()

    # --- track activities already covered in TRAIN ---
    train_acts = set()
    for c in train_cases:
        train_acts.update(case2acts[c])

    # --- candidate pool ---
    remaining = list(set(sequence_lengths.index) - train_cases)

    if remaining:

        lengths = sequence_lengths.loc[remaining]

        bins = pd.qcut(
            lengths,
            q=min(n_bins, len(lengths)),
            labels=False,
            duplicates="drop"
        )

        target_test_total = max(1, int(len(sequence_lengths) * test_size))

        # process longest first (improves coverage stability)
        ordered = lengths.sort_values(ascending=False).index.tolist()

        for cid in ordered:

            acts = case2acts[cid]

            # cannot place in test if activities not yet in TRAIN
            if not acts.issubset(train_acts):
                train_cases.add(cid)
                train_acts.update(acts)
                continue

            if len(test_cases) < target_test_total:

                if rng.rand() < test_size:
                    test_cases.add(cid)
                    continue

            train_cases.add(cid)
            train_acts.update(acts)

    # --- final validation (should never fail now) ---
    unseen = compute_unseen_activities(train_cases, test_cases, case2acts)

    if unseen:
        raise RuntimeError(
            f"Split construction error: {len(unseen)} unseen activities."
        )

    if verbose:
        print(f"Train cases: {len(train_cases)} | Test cases: {len(test_cases)}")

    return sorted(train_cases), sorted(test_cases)

#----Time----#
    
def add_delta_time(event, case_index, time_col, new_col="delta_time", norm=False):
    event = event.copy()
    event[time_col] = pd.to_datetime(event[time_col])
    event = event.sort_values([case_index, time_col])

    event[new_col] = event.groupby(case_index)[time_col].diff().dt.total_seconds()
    event[new_col] = event[new_col].fillna(0)

    # stabilize distribution
    if norm == True: 
       event[new_col] = np.log1p(event[new_col])

    return event


def encode_start_time_features(start_dt):
    # start_dt: pandas Timestamp
    hour = start_dt.hour + start_dt.minute / 60.0
    weekday = start_dt.weekday()  # Monday=0

    hour_rad = 2 * np.pi * (hour / 24.0)
    wday_rad = 2 * np.pi * (weekday / 7.0)

    return np.array([
        np.sin(hour_rad), np.cos(hour_rad),
        np.sin(wday_rad), np.cos(wday_rad),
    ], dtype=np.float32)
    

def build_vocabs(
    train_df,
    core_event,
    cat_cols_event=None,
    cat_cols_seq=None,
    add_eos_activity: bool = True,
    add_eos_event_cat: bool = True,
    add_sos_activity: bool = True,
):
    if cat_cols_event is None:
        cat_cols_event = []
    if cat_cols_seq is None:
        cat_cols_seq = []

    def make_vocab(values, add_eos: bool = False, add_sos: bool = False):
        vocab_list = ["<PAD>", "<UNK>"]
        if add_eos:
            vocab_list.append("<EOS>")
        if add_sos:
            vocab_list.append("<SOS>")
        vocab_list += sorted(values)
        return {v: i for i, v in enumerate(vocab_list)}

    vocabs = {}

    act_vals = train_df[core_event].astype(str).unique()
    vocabs["activity"] = make_vocab(
        act_vals,
        add_eos=add_eos_activity,
        add_sos=add_sos_activity,
    )

    vocabs["event_cat"] = {}
    for col in cat_cols_event:
        vals = train_df[col].fillna("<NA>").astype(str).unique()
        vocabs["event_cat"][col] = make_vocab(vals, add_eos=add_eos_event_cat, add_sos=False)

    vocabs["seq_cat"] = {}
    for col in cat_cols_seq:
        vals = train_df[col].fillna("<NA>").astype(str).unique()
        vocabs["seq_cat"][col] = make_vocab(vals, add_eos=False, add_sos=False)

    return vocabs
    
def compute_L_max(df, case_index, add_eos: bool = True):
    lengths = df.groupby(case_index).size()
    L_max = max(lengths)
    if add_eos:
        L_max += 1
    return L_max

def encode_numeric_features(
    df,
    case_index,
    time_col,
    num_cols_event=None,
    num_cols_seq=None,
    L_max=None,
    stats=None,
    fit=False,
    positive_event_cols=None,
    positive_seq_cols=None,
):
    if num_cols_event is None:
        num_cols_event = []
    if num_cols_seq is None:
        num_cols_seq = []

    if positive_event_cols is None:
        positive_event_cols = []
    if positive_seq_cols is None:
        positive_seq_cols = []

    if num_cols_event and L_max is None:
        raise ValueError("L_max required for event numeric.")

    df = df.sort_values([case_index, time_col]).copy()
    groups = list(df.groupby(case_index, sort=False))
    N = len(groups)

    def _transform_vals(vals, positive=False):
        vals = vals.astype(float, copy=True)
        if positive:
            vals[vals < 0] = np.nan
            vals = np.log1p(vals)
        return vals

    if fit:
        stats = {"event": {}, "seq": {}}

        for c in num_cols_event:
            positive = c in positive_event_cols
            vals = df[c].astype(float).values.copy()
            vals = _transform_vals(vals, positive=positive)
            vals = vals[np.isfinite(vals)]

            mean = vals.mean() if len(vals) else 0.0
            std = vals.std() + 1e-8 if len(vals) else 1.0

            stats["event"][c] = {
                "mean": mean,
                "std": std,
                "positive": positive,
            }

        for c in num_cols_seq:
            positive = c in positive_seq_cols
            vals = df[c].astype(float).values.copy()
            vals = _transform_vals(vals, positive=positive)
            vals = vals[np.isfinite(vals)]

            mean = vals.mean() if len(vals) else 0.0
            std = vals.std() + 1e-8 if len(vals) else 1.0

            stats["seq"][c] = {
                "mean": mean,
                "std": std,
                "positive": positive,
            }

    elif stats is None:
        raise ValueError("Stats must be provided when fit=False")

    event_numeric = {
        c: torch.zeros((N, L_max), dtype=torch.float32)
        for c in num_cols_event
    }
    seq_numeric = {
        c: torch.zeros(N, dtype=torch.float32)
        for c in num_cols_seq
    }

    for i, (_, g) in enumerate(groups):
        for c in num_cols_seq:
            meta = stats["seq"][c]
            mean = meta["mean"]
            std = meta["std"]
            positive = meta["positive"]

            v = np.array([g.iloc[0][c]], dtype=float)
            v = _transform_vals(v, positive=positive)

            if not np.isfinite(v[0]):
                v[0] = mean

            seq_numeric[c][i] = (v[0] - mean) / std

        for c in num_cols_event:
            meta = stats["event"][c]
            mean = meta["mean"]
            std = meta["std"]
            positive = meta["positive"]

            vals = g[c].astype(float).values.copy()
            vals = _transform_vals(vals, positive=positive)
            vals[~np.isfinite(vals)] = mean
            vals = (vals - mean) / std

            L_real = len(vals)
            event_numeric[c][i, :L_real] = torch.tensor(vals[:L_real], dtype=torch.float32)

    return event_numeric, seq_numeric, stats

def encode_numeric_features_simple(
    df,
    case_index,
    time_col,
    num_cols_event=None,
    num_cols_seq=None,
    L_max=None,
    stats=None,
    fit=False,
):
    if num_cols_event is None:
        num_cols_event = []
    if num_cols_seq is None:
        num_cols_seq = []

    if num_cols_event and L_max is None:
        raise ValueError("L_max required for event numeric.")

    df = df.sort_values([case_index, time_col]).copy()
    groups = list(df.groupby(case_index, sort=False))
    N = len(groups)

  
    if fit:
        stats = {"event": {}, "seq": {}}

        for c in num_cols_event:
            vals = df[c].astype(float).values.copy()
            vals = vals[np.isfinite(vals)]
            mean = vals.mean() if len(vals) else 0.0
            std = vals.std() + 1e-8 if len(vals) else 1.0
            stats["event"][c] = (mean, std)

        for c in num_cols_seq:
            vals = df[c].astype(float).values.copy()
            vals = vals[np.isfinite(vals)]
            mean = vals.mean() if len(vals) else 0.0
            std = vals.std() + 1e-8 if len(vals) else 1.0
            stats["seq"][c] = (mean, std)

    elif stats is None:
        raise ValueError("Stats must be provided when fit=False")

    event_numeric = {c: torch.zeros((N, L_max), dtype=torch.float32) for c in num_cols_event}
    seq_numeric = {c: torch.zeros(N, dtype=torch.float32) for c in num_cols_seq}

    for i, (_, g) in enumerate(groups):
        for c in num_cols_seq:
            mean, std = stats["seq"][c]
            v = float(g.iloc[0][c])
            if not np.isfinite(v):
                v = mean
            seq_numeric[c][i] = (v - mean) / std

        for c in num_cols_event:
            mean, std = stats["event"][c]
            vals = g[c].astype(float).values.copy()
            vals[~np.isfinite(vals)] = mean
            vals = (vals - mean) / std

            L_real = len(vals)  
            event_numeric[c][i, :L_real] = torch.tensor(vals[:L_real], dtype=torch.float32)

    return event_numeric, seq_numeric, stats


def encode_categorical_features(
    df,
    case_index,
    time_col,
    core_event,
    cat_cols_event,
    cat_cols_seq,
    vocabs,
    L_max
):

    df = df.sort_values([case_index, time_col]).copy()

    groups = list(df.groupby(case_index, sort=False))
    N = len(groups)

    y_activity = torch.zeros((N, L_max), dtype=torch.long)
    mask = torch.zeros((N, L_max), dtype=torch.float32)

    event_cat = {c: torch.zeros((N, L_max), dtype=torch.long) for c in cat_cols_event}
    seq_cat = {c: torch.zeros(N, dtype=torch.long) for c in cat_cols_seq}

    act_vocab = vocabs["activity"]
    unk_act = act_vocab["<UNK>"]
    eos_act = act_vocab["<EOS>"]

    lengths = []

    for i, (_, g) in enumerate(groups):
        raw_len = len(g)
        L_real = raw_len  # reserve 1 slot for EOS
        L_total = raw_len + 1  # real events + EOS
        lengths.append(L_total)

        acts = g[core_event].astype(str).values

        # real activities
        for t in range(L_real):
            y_activity[i, t] = act_vocab.get(acts[t], unk_act)
            mask[i, t] = 1.0

        # EOS
        y_activity[i, L_real] = eos_act
        mask[i, L_real] = 1.0

        # event categorical
        for c in cat_cols_event:
            vocab = vocabs["event_cat"][c]
            unk = vocab["<UNK>"]
            eos = vocab["<EOS>"]

            vals = g[c].fillna("<NA>").astype(str).values

            for t in range(L_real):
                event_cat[c][i, t] = vocab.get(vals[t], unk)
            
            event_cat[c][i, L_real] = eos


        # sequence categorical
        for c in cat_cols_seq:
            vocab = vocabs["seq_cat"][c]
            unk = vocab["<UNK>"]

            val = g.iloc[0][c]
            if pd.isna(val):
                val = "<NA>"
            val = str(val)

            seq_cat[c][i] = vocab.get(val, unk)

    lengths = torch.tensor(lengths, dtype=torch.long)

    return y_activity, event_cat, seq_cat, mask, lengths


def encode_time_targets(
    df,
    case_index,
    time_col,
    delta_col,
    L_max,
    stats=None,
    fit=False,
):
    df = df.sort_values([case_index, time_col]).copy()
    groups = list(df.groupby(case_index, sort=False))
    N = len(groups)

    if fit:
        vals = df[delta_col].astype(float).values.copy()
        vals[~np.isfinite(vals)] = 0.0
        vals = np.log1p(np.clip(vals, a_min=0.0, a_max=None))
        mean = vals.mean() if len(vals) else 0.0
        std = vals.std() + 1e-8 if len(vals) else 1.0
        stats = {"mean": mean, "std": std}
    elif stats is None:
        raise ValueError("stats must be provided when fit=False")

    y_time_log = torch.zeros((N, L_max), dtype=torch.float32)

    for i, (_, g) in enumerate(groups):
        vals = g[delta_col].astype(float).values.copy()
        vals[~np.isfinite(vals)] = 0.0
        vals = np.log1p(np.clip(vals, a_min=0.0, a_max=None))
        vals = (vals - stats["mean"]) / stats["std"]

        L_real = len(vals)
        y_time_log[i, :L_real] = torch.tensor(vals[:L_real], dtype=torch.float32)

    return y_time_log, stats

def time_cap(train_df, delta_col, time_stats):
    train_log_deltas = train_df[delta_col].astype(float).values.copy()
    train_log_deltas[~np.isfinite(train_log_deltas)] = 0.0
    train_log_deltas = np.log1p(np.clip(train_log_deltas, a_min=0.0, a_max=None))
    
    # upper cap in raw log space from train distribution
    time_log_cap_raw = float(np.quantile(train_log_deltas, 0.999))
    
    # convert both bounds into standardized model space
    time_min_scaled = (0.0 - time_stats["mean"]) / time_stats["std"]
    time_max_scaled = (time_log_cap_raw - time_stats["mean"]) / time_stats["std"]
    return time_min_scaled, time_max_scaled



def build_sequence_head(
    df,
    case_index,
    time_col,
    lengths,
    L_max
):
    df = df.sort_values([case_index, time_col]).copy()
    groups = list(df.groupby(case_index, sort=False))
    N = len(groups)

    case_feat_list = []
    case_ids = []
    start_times = []

    for i, (cid, g) in enumerate(groups):

        # ---- normalized length ----
        L = min(lengths[i].item(), L_max)
        length_norm = np.array([L / L_max], dtype=np.float32)

        # ---- start time encoding ----
        start_ts = pd.to_datetime(g.iloc[0][time_col])
        time_feat = encode_start_time_features(start_ts)

        # ---- combine ----
        feats = np.concatenate([length_norm, time_feat])

        case_feat_list.append(feats)

        # NEW metadata capture
        case_ids.append(cid)
        start_times.append(start_ts)

    case_feat = torch.tensor(np.stack(case_feat_list), dtype=torch.float32)

    return case_feat, case_ids, start_times


class BPMOneShotDataset(Dataset):
    def __init__(
        self,
        sequence_head,     # (N, d_case_num) float32
        seq_cat,               # dict col -> (N,) long
        seq_numeric,           # dict col -> (N,) float32
        y_activity,            # (N, L_max) long
        y_time_log,            # (N, L_max) float32
        event_cat,             # dict col -> (N, L_max) long
        event_numeric,         # dict col -> (N, L_max) float32
        mask,                  # (N, L_max) float32
        lengths,
        case_ids,        # NEW: list or array length N
        start_times      # NEW: list or array length N (pandas Timestamp)
    ):
        self.sequence_head = sequence_head
        self.seq_cat = seq_cat
        self.seq_numeric = seq_numeric

        self.y_activity = y_activity
        self.y_time_log = y_time_log

        self.event_cat = event_cat
        self.event_numeric = event_numeric

        self.mask = mask
        self.lengths = lengths

        self.case_ids = case_ids
        self.start_times = start_times

        self.N = sequence_head.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample = {
            "sequence_head": self.sequence_head[idx],  # (d_case_num,)
            "mask": self.mask[idx],                            # (L_max,)
            "length": self.lengths[idx],                       # ()
            "y_activity": self.y_activity[idx],                # (L_max,)
            "y_time_log": self.y_time_log[idx],                # (L_max,)
        }

        sample["seq_cat"] = {c: self.seq_cat[c][idx] for c in self.seq_cat}
        sample["seq_numeric"] = {c: self.seq_numeric[c][idx] for c in self.seq_numeric}

        sample["event_cat"] = {c: self.event_cat[c][idx] for c in self.event_cat}
        sample["event_numeric"] = {c: self.event_numeric[c][idx] for c in self.event_numeric}
        sample["case_id"] = self.case_ids[idx]
        sample["start_time"] = self.start_times[idx]

        return sample

def audit_encoded_split(name, encoded, vocabs=None):
    print(f"\n========== AUDIT: {name} ==========")

    # ---- required tensors ----
    required = [
        "seq_head",
        "mask",
        "lengths",
        "y_activity",
        "y_time_log",
    ]

    for k in required:
        if k not in encoded:
            raise KeyError(f"Missing key: {k}")

    N = encoded["seq_head"].shape[0]
    L = encoded["mask"].shape[1]

    def shp(t):
        return tuple(t.shape), t.dtype

    print("N cases:", N, "| L_max:", L)
    print("seq_head:", shp(encoded["seq_head"]))
    print("mask:", shp(encoded["mask"]))
    print("lengths:", shp(encoded["lengths"]))
    print("y_activity:", shp(encoded["y_activity"]))
    print("y_time_log:", shp(encoded["y_time_log"]))

    # ---- basic consistency ----
    assert encoded["mask"].shape == (N, L)
    assert encoded["y_activity"].shape == (N, L)
    assert encoded["y_time_log"].shape == (N, L)
    assert encoded["lengths"].shape == (N,)

    # ---- truncation check ----
    raw_max = int(encoded["lengths"].max())
    trunc_count = int((encoded["lengths"] > L).sum())
    print("Max raw length:", raw_max)
    print("Cases truncated:", trunc_count)

    # =====================================================
    # SEQUENCE CATEGORICAL
    # =====================================================
    seq_cat = encoded.get("seq_cat", {})
    print("\nSeq categorical cols:", list(seq_cat.keys()))

    for c, t in seq_cat.items():
        print(f"  seq_cat[{c}]:", shp(t))
        assert t.shape == (N,)
        assert t.dtype == torch.long

        if vocabs:
            vmax = int(t.max())
            vsize = len(vocabs["seq_cat"][c])
            assert vmax < vsize, f"{c} has id out of range"

    # =====================================================
    # EVENT CATEGORICAL
    # =====================================================
    event_cat = encoded.get("event_cat", {})
    print("\nEvent categorical cols:", list(event_cat.keys()))

    for c, t in event_cat.items():
        print(f"  event_cat[{c}]:", shp(t))
        assert t.shape == (N, L)
        assert t.dtype == torch.long

        if vocabs:
            vmax = int(t.max())
            vsize = len(vocabs["event_cat"][c])
            assert vmax < vsize, f"{c} has id out of range"

    # =====================================================
    # SEQUENCE NUMERIC
    # =====================================================
    seq_num = encoded.get("seq_numeric", {})
    print("\nSeq numeric cols:", list(seq_num.keys()))

    for c, t in seq_num.items():
        print(f"  seq_numeric[{c}]:", shp(t))
        assert t.shape == (N,)
        assert torch.is_floating_point(t)

    # =====================================================
    # EVENT NUMERIC
    # =====================================================
    event_num = encoded.get("event_numeric", {})
    print("\nEvent numeric cols:", list(event_num.keys()))

    for c, t in event_num.items():
        print(f"  event_numeric[{c}]:", shp(t))
        assert t.shape == (N, L)
        assert torch.is_floating_point(t)

    # =====================================================
    # MASK SANITY
    # =====================================================
    mask = encoded["mask"]

    pad_positions = int((mask == 0).sum())
    valid_positions = int((mask == 1).sum())

    print("\nMask summary:")
    print("  Valid positions:", valid_positions)
    print("  Padded positions:", pad_positions)

    # padded positions must be beyond actual lengths
    for i in range(N):
        L_i = int(min(encoded["lengths"][i], L))
        assert mask[i, :L_i].sum() == L_i
        assert mask[i, L_i:].sum() == 0

    print("\n========== AUDIT PASSED ==========")        


def build_process_graph(
    event,
    core_event,
    case_index,
    time_col,
    delta_col,
    vocabs
):
    df = event.copy()

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([case_index, time_col]).reset_index(drop=True)

    activity2id = vocabs["activity"]
    id2activity = {i: a for a, i in activity2id.items()}
    num_acts = len(activity2id)

    pad_id = activity2id["<PAD>"]
    unk_id = activity2id["<UNK>"]
    eos_id = activity2id["<EOS>"]
    sos_id = activity2id["<SOS>"]


    trans_counts = {}
    trans_log_dt = {}
 
    for _, group in df.groupby(case_index, sort=False):

        acts = group[core_event].map(activity2id).values
        dts = group[delta_col].values.astype(float)

        # -------------------------
        # start transition: SOS → first activity
        # -------------------------
        first_act = int(acts[0])

        if first_act not in (pad_id, unk_id):
            key = (sos_id, first_act)

            trans_counts[key] = trans_counts.get(key, 0) + 1

            dt_val = dts[0]
            if np.isfinite(dt_val) and dt_val >= 0:
                trans_log_dt.setdefault(key, []).append(np.log1p(dt_val))
            else:
                trans_log_dt.setdefault(key, []).append(0.0)

        # normal transitions
        for i in range(len(acts) - 1):
            src = int(acts[i])
            dst = int(acts[i + 1])

            if src in (pad_id, unk_id) or dst in (pad_id, unk_id):
                continue

            key = (src, dst)

            trans_counts[key] = trans_counts.get(key, 0) + 1

            dt_val = dts[i + 1]
            if np.isfinite(dt_val) and dt_val >= 0:
                trans_log_dt.setdefault(key, []).append(np.log1p(dt_val))

        # terminal transition: last activity → EOS
        last_act = int(acts[-1])

        if last_act not in (pad_id, unk_id):
            key = (last_act, eos_id)

            trans_counts[key] = trans_counts.get(key, 0) + 1

            # no real time gap to EOS → use 0
            trans_log_dt.setdefault(key, []).append(0.0)

    edges = []
    attrs = []

    for (src, dst), count in trans_counts.items():

        edges.append([src, dst])

        log_freq = np.log1p(count)
        log_dt_list = trans_log_dt.get((src, dst), [])
        mean_log_dt = float(np.mean(log_dt_list)) if log_dt_list else 0.0

        attrs.append([log_freq, mean_log_dt])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.float32)

    adj = torch.zeros((num_acts, num_acts), dtype=torch.float32)

    for (src, dst) in trans_counts:
        adj[src, dst] = 1.0

    # forbid transitions FROM EOS
    adj[eos_id, :] = 0.0
    # forbid transitions INTO SOS
    adj[:, sos_id] = 0.0

    # forbid PAD and UNK completely
    adj[pad_id, :] = 0.0
    adj[:, pad_id] = 0.0
    adj[unk_id, :] = 0.0
    adj[:, unk_id] = 0.0

    return {
        "id2activity": id2activity,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "adj_matrix": adj,
        "num_activities": num_acts
    }


def bpm_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate list of BPMOneShotDataset samples into a batch dict.

    Returns keys:
      sequence_head: (B, d_case_num)
      mask:          (B, L)
      length:        (B,)
      y_activity:    (B, L)
      y_time_log:    (B, L)

      seq_cat:       dict[name -> (B,)]           if present
      seq_numeric:   dict[name -> (B,)]           if present
      event_cat:     dict[name -> (B, L)]         if present
      event_numeric: dict[name -> (B, L)]         if present
    """
    # required tensors
    batch = {
        "sequence_head": torch.stack([s["sequence_head"] for s in samples], dim=0),
        "mask":          torch.stack([s["mask"] for s in samples], dim=0),
        "length":        torch.stack([s["length"] for s in samples], dim=0).long(),
        "y_activity":    torch.stack([s["y_activity"] for s in samples], dim=0).long(),
        "y_time_log":    torch.stack([s["y_time_log"] for s in samples], dim=0).float(),
    }

    # optional dicts: infer presence from first sample
    if "seq_cat" in samples[0] and isinstance(samples[0]["seq_cat"], dict) and len(samples[0]["seq_cat"]) > 0:
        keys = samples[0]["seq_cat"].keys()
        batch["seq_cat"] = {k: torch.stack([s["seq_cat"][k] for s in samples], dim=0).long() for k in keys}
    else:
        batch["seq_cat"] = {}

    if "seq_numeric" in samples[0] and isinstance(samples[0]["seq_numeric"], dict) and len(samples[0]["seq_numeric"]) > 0:
        keys = samples[0]["seq_numeric"].keys()
        batch["seq_numeric"] = {k: torch.stack([s["seq_numeric"][k] for s in samples], dim=0).float() for k in keys}
    else:
        batch["seq_numeric"] = {}

    if "event_cat" in samples[0] and isinstance(samples[0]["event_cat"], dict) and len(samples[0]["event_cat"]) > 0:
        keys = samples[0]["event_cat"].keys()
        batch["event_cat"] = {k: torch.stack([s["event_cat"][k] for s in samples], dim=0).long() for k in keys}
    else:
        batch["event_cat"] = {}

    if "event_numeric" in samples[0] and isinstance(samples[0]["event_numeric"], dict) and len(samples[0]["event_numeric"]) > 0:
        keys = samples[0]["event_numeric"].keys()
        batch["event_numeric"] = {k: torch.stack([s["event_numeric"][k] for s in samples], dim=0).float() for k in keys}
    else:
        batch["event_numeric"] = {}
    
    batch["case_id"] = [s["case_id"] for s in samples]
    batch["start_time"] = [s["start_time"] for s in samples]

    return batch
#------------------------#
#      Holdout           #
#------------------------#

def build_holdout_head(
    df,
    case_index,
    time_col,
    L_max
):
    df = df.sort_values([case_index, time_col]).copy()
    groups = list(df.groupby(case_index, sort=False))
    N = len(groups)

    case_feat_list = []
    case_ids = []
    start_times = []
    lengths = []

    for i, (cid, g) in enumerate(groups):

        # ---- normalized length ----
        L = min(len(g), L_max)
        lengths.append(len(g))
        length_norm = np.array([L / L_max], dtype=np.float32)

        # ---- start time encoding ----
        start_ts = pd.to_datetime(g.iloc[0][time_col])
        time_feat = encode_start_time_features(start_ts)

        # ---- combine ----
        feats = np.concatenate([length_norm, time_feat])

        case_feat_list.append(feats)

        # NEW metadata capture
        case_ids.append(cid)
        start_times.append(start_ts)

    case_feat = torch.tensor(np.stack(case_feat_list), dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return case_feat, case_ids, start_times, lengths

class HoldoutGenerationDataset(Dataset):
    def __init__(self, seq_head, lengths, case_ids, start_times):
        self.seq_head = seq_head
        self.lengths = lengths
        self.case_ids = case_ids
        self.start_times = start_times

    def __len__(self):
        return len(self.seq_head)

    def __getitem__(self, idx):
        return {
            "sequence_head": self.seq_head[idx],
            "length": self.lengths[idx],
            "case_id": self.case_ids[idx],
            "start_time": self.start_times[idx],
        }

def bpm_collate_fn_hold(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    # required tensors
    batch = {
        "sequence_head": torch.stack([s["sequence_head"] for s in samples], dim=0),
        "length":        torch.stack([s["length"] for s in samples], dim=0).long()
    }
    
    batch["case_id"] = [s["case_id"] for s in samples]
    batch["start_time"] = [s["start_time"] for s in samples]

    return batch