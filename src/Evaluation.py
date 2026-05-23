from difflib import SequenceMatcher
import pandas as pd
import numpy as np
from collections import Counter

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import pyxdameraulevenshtein as dl
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import pyxdameraulevenshtein as dl


def seq_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def dl_similarity(a, b):
    if max(len(a), len(b)) == 0:
        return 1.0
    dist = dl.damerau_levenshtein_distance(a, b)
    return 1 - dist / max(len(a), len(b))


def get_traces(df, case_id_col, activity_col, order_col):
    df_sorted = df.sort_values([case_id_col, order_col]).copy()
    df_sorted = df_sorted[df_sorted[activity_col].notna()]
    return df_sorted.groupby(case_id_col)[activity_col].apply(list)


# -----------------------------
# CONDITIONAL versions
# -----------------------------
def trace_similarity_conditional(gen_df, gt_df, case_id_col, activity_col, time_col, pos_col):
    gen = get_traces(gen_df, case_id_col, activity_col, pos_col)
    gt = get_traces(gt_df, case_id_col, activity_col, time_col)

    common = set(gen.index) & set(gt.index)
    scores = [seq_similarity(gt[c], gen[c]) for c in common]

    return float(np.mean(scores)) if scores else 0.0


def dl_trace_similarity_conditional(gen_df, gt_df, case_id_col, activity_col, time_col, pos_col):
    gen = get_traces(gen_df, case_id_col, activity_col, pos_col)
    gt = get_traces(gt_df, case_id_col, activity_col, time_col)

    common = set(gen.index) & set(gt.index)
    scores = [dl_similarity(gt[c], gen[c]) for c in common]

    return float(np.mean(scores)) if scores else 0.0


# -----------------------------
# UNCONDITIONAL versions
# average over ALL ground-truth cases
# missing generated case => score 0
# -----------------------------
def trace_similarity_unconditional(gen_df, gt_df, case_id_col, activity_col, time_col, pos_col):
    gen = get_traces(gen_df, case_id_col, activity_col, pos_col)
    gt = get_traces(gt_df, case_id_col, activity_col, time_col)

    scores = []
    for c in gt.index:
        if c in gen.index and isinstance(gen[c], list) and len(gen[c]) > 0:
            scores.append(seq_similarity(gt[c], gen[c]))
        else:
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


def dl_trace_similarity_unconditional(gen_df, gt_df, case_id_col, activity_col, time_col, pos_col):
    gen = get_traces(gen_df, case_id_col, activity_col, pos_col)
    gt = get_traces(gt_df, case_id_col, activity_col, time_col)

    scores = []
    for c in gt.index:
        if c in gen.index and isinstance(gen[c], list) and len(gen[c]) > 0:
            scores.append(dl_similarity(gt[c], gen[c]))
        else:
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


# -----------------------------
# Coverage
# -----------------------------
def case_coverage(gen_df, gt_df, case_id_col):
    gt_cases = set(gt_df[case_id_col].dropna().unique())
    gen_cases = set(gen_df[case_id_col].dropna().unique())

    if len(gt_cases) == 0:
        return np.nan

    return len(gt_cases & gen_cases) / len(gt_cases)


# -----------------------------
# Bigrams
# -----------------------------
def bigram_jsd(gen_df, gt_df, case_id_col, activity_col, time_col, pos_col):
    def bigrams(traces):
        bg = []
        for t in traces:
            if len(t) >= 2:
                bg += [f"{t[i]}->{t[i+1]}" for i in range(len(t) - 1)]
        if len(bg) == 0:
            return pd.Series(dtype=float)
        return pd.Series(bg).value_counts(normalize=True)

    gen_tr = get_traces(gen_df, case_id_col, activity_col, pos_col)
    gt_tr = get_traces(gt_df, case_id_col, activity_col, time_col)

    gen_bg = bigrams(gen_tr)
    gt_bg = bigrams(gt_tr)

    all_bg = sorted(set(gen_bg.index) | set(gt_bg.index))
    if len(all_bg) == 0:
        return np.nan

    g = np.array([gen_bg.get(x, 0.0) for x in all_bg], dtype=float)
    t = np.array([gt_bg.get(x, 0.0) for x in all_bg], dtype=float)

    if g.sum() == 0 or t.sum() == 0:
        return np.nan

    return float(jensenshannon(g, t))


def bigram_jsd_with_coverage_penalty(
    gen_df, gt_df, case_id_col, activity_col, time_col, pos_col, lambda_penalty=1.0
):
    jsd = bigram_jsd(gen_df, gt_df, case_id_col, activity_col, time_col, pos_col)
    cov = case_coverage(gen_df, gt_df, case_id_col)

    if np.isnan(jsd) or np.isnan(cov):
        return np.nan

    return jsd + lambda_penalty * (1.0 - cov)


# -----------------------------
# Duration realism
# sorting not needed here
# -----------------------------
def case_duration(df, case_id_col, time_col):
    times = pd.to_datetime(df[time_col], errors="coerce")
    grouped = times.groupby(df[case_id_col])
    return (grouped.max() - grouped.min()).dt.total_seconds()


def duration_wd(gen_df, gt_df, case_id_col, time_col):
    gen_dur = case_duration(gen_df, case_id_col, time_col).dropna()
    gt_dur = case_duration(gt_df, case_id_col, time_col).dropna()

    if len(gen_dur) == 0 or len(gt_dur) == 0:
        return np.nan

    return float(wasserstein_distance(gen_dur, gt_dur))


def duration_wd_with_coverage_penalty(gen_df, gt_df, case_id_col, time_col):
    wd = duration_wd(gen_df, gt_df, case_id_col, time_col)
    cov = case_coverage(gen_df, gt_df, case_id_col)

    gt_dur = case_duration(gt_df, case_id_col, time_col).dropna()
    if np.isnan(wd) or np.isnan(cov) or len(gt_dur) == 0:
        return np.nan

    scale = np.median(gt_dur)
    return wd + (1.0 - cov) * scale

    
def evaluate_light(
    gen_df,
    gt_df,
    case_id_col,
    activity_col,
    time_col,
    pos_col,
    name="Model",
    jsd_lambda=1.0
):
    results = {
        "seq_coverage": case_coverage(gen_df, gt_df, case_id_col),

        "bigram_jsd_unconditional": bigram_jsd_with_coverage_penalty(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col, lambda_penalty=jsd_lambda
        ),
        "trace_similarity_unconditional": trace_similarity_unconditional(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col
        ),
        "duration_wd_unconditional": duration_wd_with_coverage_penalty(
            gen_df, gt_df, case_id_col, time_col
        ),
        "dl_similarity_unconditional": dl_trace_similarity_unconditional(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col
        ),

        "bigram_jsd_conditional": bigram_jsd(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col
        ),
        "trace_similarity_conditional": trace_similarity_conditional(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col
        ),
        "duration_wd_conditional": duration_wd(
            gen_df, gt_df, case_id_col, time_col
        ),
        "dl_similarity_conditional": dl_trace_similarity_conditional(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col
        ),
    }

    return pd.DataFrame([results], index=[name])

#------
    
import numpy as np
import pandas as pd


# =========================================================
# INTERNAL HELPERS
# =========================================================
def _prepare_eval_frames(gt_df, gen_df, case_id_col, time_col, pos_col):
    """
    Prepare GT and GEN for aligned evaluation.

    GT:
        sorted by real time within case
    GEN:
        sorted by generated position within case

    Then rebuild a clean within-case event index:
        0, 1, 2, ...

    This index is used for event alignment.
    """
    gt = gt_df.sort_values([case_id_col, time_col]).copy()
    gen = gen_df.sort_values([case_id_col, pos_col]).copy()

    gt["_event_idx"] = gt.groupby(case_id_col).cumcount()
    gen["_event_idx"] = gen.groupby(case_id_col).cumcount()

    return gt, gen


def _filter_common_cases(gt_df, gen_df, case_id_col):
    """
    Keep only cases that exist in both GT and GEN.
    Used by all conditional metrics.
    """
    gt_cases = set(gt_df[case_id_col].dropna().unique())
    gen_cases = set(gen_df[case_id_col].dropna().unique())
    common_cases = gt_cases & gen_cases

    gt_sub = gt_df[gt_df[case_id_col].isin(common_cases)].copy()
    gen_sub = gen_df[gen_df[case_id_col].isin(common_cases)].copy()

    return gt_sub, gen_sub


def _aligned_event_frame(gt_df, gen_df, case_id_col, time_col, pos_col, cols):
    """
    Align events by:
        [case_id, rebuilt within-case event index]

    Left join GEN onto GT.
    Therefore ALL GT events are preserved.
    This is the correct base for unconditional event metrics.
    It is also fine for conditional metrics AFTER filtering to common cases.
    """
    gt, gen = _prepare_eval_frames(gt_df, gen_df, case_id_col, time_col, pos_col)

    key_cols = [case_id_col, "_event_idx"]

    gt_keep = key_cols + [c for c in cols if c in gt.columns]
    gen_keep = key_cols + [c for c in cols if c in gen.columns]

    merged = gt[gt_keep].merge(
        gen[gen_keep],
        on=key_cols,
        how="left",
        suffixes=("_gt", "_gen"),
    )
    return merged

def _collapse_sequence_attributes(df, case_id_col, cols, strict=True):
    """
    Collapse event rows to one row per case for sequence-level attributes.

    If strict=True:
        - if a case has more than one distinct non-null value for a column,
          the collapsed value is set to NaN

    If strict=False:
        - use the mode among non-null values
        - ties are broken by pandas mode().iloc[0]

    Returns
    -------
    collapsed_df : pd.DataFrame
        one row per case
    """
    rows = []

    for case_id, group in df.groupby(case_id_col):
        row = {case_id_col: case_id}

        for col in cols:
            if col not in group.columns:
                row[col] = np.nan
                continue

            vals = group[col].dropna()

            if len(vals) == 0:
                row[col] = np.nan
                continue

            uniq = pd.Series(vals).nunique(dropna=True)

            if strict:
                if uniq == 1:
                    row[col] = vals.iloc[0]
                else:
                    row[col] = np.nan
            else:
                mode_vals = pd.Series(vals).mode()
                row[col] = mode_vals.iloc[0] if not mode_vals.empty else np.nan

        rows.append(row)

    return pd.DataFrame(rows)
    
def _aligned_sequence_frame(gt_df, gen_df, case_id_col, time_col, pos_col, cols, strict_gen=True):
    """
    One row per case for sequence-level evaluation.

    GT is collapsed per case.
    GEN is collapsed per case.

    If strict_gen=True:
        inconsistent generated sequence attributes become NaN.
    """
    gt, gen = _prepare_eval_frames(gt_df, gen_df, case_id_col, time_col, pos_col)

    gt_case = _collapse_sequence_attributes(gt, case_id_col, cols, strict=True)
    gen_case = _collapse_sequence_attributes(gen, case_id_col, cols, strict=strict_gen)

    gt_keep = [case_id_col] + [c for c in cols if c in gt_case.columns]
    gen_keep = [case_id_col] + [c for c in cols if c in gen_case.columns]

    merged = gt_case[gt_keep].merge(
        gen_case[gen_keep],
        on=case_id_col,
        how="left",
        suffixes=("_gt", "_gen"),
    )
    return merged

def _numeric_series_or_nan(df, col):
    """
    Safe numeric conversion. Returns float series or all-NaN series if col missing.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _datetime_series_or_nat(df, col):
    """
    Safe datetime conversion. Returns datetime series or all-NaT series if col missing.
    """
    if col in df.columns:
        return pd.to_datetime(df[col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


# =========================================================
# PENALTY BUILDERS
# max observed valid absolute error in matched rows
# =========================================================
def build_sequence_numeric_penalties(
    gen_df,
    gt_df,
    case_id_col,
    sequence_attr_cols_num,
    time_col,
    pos_col,
    fallback=1.0,
):
    merged = _aligned_sequence_frame(
        gt_df, gen_df, case_id_col, time_col, pos_col, sequence_attr_cols_num
    )

    penalties = {}

    for col in sequence_attr_cols_num:
        gt_col = f"{col}_gt"
        gen_col = f"{col}_gen"

        gt_vals = _numeric_series_or_nan(merged, gt_col)
        gen_vals = _numeric_series_or_nan(merged, gen_col)

        valid = ~(gt_vals.isna() | gen_vals.isna())

        if valid.any():
            penalties[col] = float((gen_vals[valid] - gt_vals[valid]).abs().max())
        else:
            penalties[col] = float(fallback)

    return penalties


def build_event_numeric_penalties(
    gen_df,
    gt_df,
    case_id_col,
    event_attr_cols_num,
    time_col,
    pos_col,
    fallback=1.0,
):
    merged = _aligned_event_frame(
        gt_df, gen_df, case_id_col, time_col, pos_col, event_attr_cols_num
    )

    penalties = {}

    for col in event_attr_cols_num:
        gt_col = f"{col}_gt"
        gen_col = f"{col}_gen"

        gt_vals = _numeric_series_or_nan(merged, gt_col)
        gen_vals = _numeric_series_or_nan(merged, gen_col)

        valid = ~(gt_vals.isna() | gen_vals.isna())

        if valid.any():
            penalties[col] = float((gen_vals[valid] - gt_vals[valid]).abs().max())
        else:
            penalties[col] = float(fallback)

    return penalties


def build_timestamp_penalty(
    gen_df,
    gt_df,
    case_id_col,
    time_col,
    pos_col,
    fallback=1.0,
):
    merged = _aligned_event_frame(
        gt_df, gen_df, case_id_col, time_col, pos_col, [time_col]
    )

    gt_times = _datetime_series_or_nat(merged, f"{time_col}_gt")
    gen_times = _datetime_series_or_nat(merged, f"{time_col}_gen")

    valid = ~(gt_times.isna() | gen_times.isna())

    if valid.any():
        diff_sec = (gen_times[valid] - gt_times[valid]).abs().dt.total_seconds()
        return float(diff_sec.max())

    return float(fallback)


# =========================================================
# EVENT categorical accuracy
# =========================================================
def event_attribute_accuracy_conditional(
    gen_df,
    gt_df,
    case_id_col,
    event_attr_cols,
    time_col,
    pos_col,
):
    gt_sub, gen_sub = _filter_common_cases(gt_df, gen_df, case_id_col)

    merged = _aligned_event_frame(
        gt_sub, gen_sub, case_id_col, time_col, pos_col, event_attr_cols
    )

    results = {}

    for col in event_attr_cols:
        gt_col = f"{col}_gt"
        gen_col = f"{col}_gen"

        if gt_col not in merged.columns or gen_col not in merged.columns:
            results[f"event_{col}_accuracy_conditional"] = 0.0
            continue

        valid = ~(merged[gt_col].isna() | merged[gen_col].isna())

        if valid.any():
            acc = (
                merged.loc[valid, gt_col].astype(str)
                == merged.loc[valid, gen_col].astype(str)
            ).mean()
            results[f"event_{col}_accuracy_conditional"] = float(acc)
        else:
            results[f"event_{col}_accuracy_conditional"] = 0.0

    return results


def event_attribute_accuracy_unconditional(
    gen_df,
    gt_df,
    case_id_col,
    event_attr_cols,
    time_col,
    pos_col,
):
    merged = _aligned_event_frame(
        gt_df, gen_df, case_id_col, time_col, pos_col, event_attr_cols
    )

    results = {}

    for col in event_attr_cols:
        gt_col = f"{col}_gt"
        gen_col = f"{col}_gen"

        if gt_col not in merged.columns:
            results[f"event_{col}_accuracy_unconditional"] = 0.0
            continue

        if gen_col not in merged.columns:
            results[f"event_{col}_accuracy_unconditional"] = 0.0
            continue

        gt_exists = merged[gt_col].notna()

        match = (
            gt_exists
            & merged[gen_col].notna()
            & (merged[gt_col].astype(str) == merged[gen_col].astype(str))
        ).astype(float)

        if gt_exists.any():
            results[f"event_{col}_accuracy_unconditional"] = float(match.loc[gt_exists].mean())
        else:
            results[f"event_{col}_accuracy_unconditional"] = 0.0

    return results

# =========================================================
# SEQUENCE categorical accuracy
# =========================================================
def sequence_attribute_accuracy_conditional(
    gen_df,
    gt_df,
    case_id_col,
    sequence_attr_cols,
    time_col,
    pos_col,
):
    gt_sub, gen_sub = _filter_common_cases(gt_df, gen_df, case_id_col)

    merged = _aligned_sequence_frame(
        gt_sub, gen_sub, case_id_col, time_col, pos_col, sequence_attr_cols
    )

    results = {}

    for col in sequence_attr_cols:
        gt_col = f"{col}_gt"
        gen_col = f"{col}_gen"

        if gt_col not in merged.columns or gen_col not in merged.columns:
            results[f"seq_{col}_accuracy_conditional"] = 0.0
            continue

        valid = ~(merged[gt_col].isna() | merged[gen_col].isna())

        if valid.any():
            acc = (
                merged.loc[valid, gt_col].astype(str)
                == merged.loc[valid, gen_col].astype(str)
            ).mean()
            results[f"seq_{col}_accuracy_conditional"] = float(acc)
        else:
            results[f"seq_{col}_accuracy_conditional"] = 0.0

    return results


def sequence_attribute_accuracy_unconditional(
    gen_df,
    gt_df,
    case_id_col,
    sequence_attr_cols,
    time_col,
    pos_col,
):
    merged = _aligned_sequence_frame(
        gt_df, gen_df, case_id_col, time_col, pos_col, sequence_attr_cols
    )

    results = {}

    for col in sequence_attr_cols:
        gt_col = f"{col}_gt"
        gen_col = f"{col}_gen"

        if gt_col not in merged.columns:
            results[f"seq_{col}_accuracy_unconditional"] = 0.0
            continue

        if gen_col not in merged.columns:
            results[f"seq_{col}_accuracy_unconditional"] = 0.0
            continue

        gt_exists = merged[gt_col].notna()

        match = (
            gt_exists
            & merged[gen_col].notna()
            & (merged[gt_col].astype(str) == merged[gen_col].astype(str))
        ).astype(float)

        results[f"seq_{col}_accuracy_unconditional"] = (
            float(match.loc[gt_exists].mean()) if gt_exists.any() else 0.0
        )

    return results

# =========================================================
# SEQUENCE numeric MAE
# =========================================================
def sequence_attribute_mae_conditional(
    gen_df,
    gt_df,
    case_id_col,
    sequence_attr_cols_num,
    time_col,
    pos_col,
):
    gt_sub, gen_sub = _filter_common_cases(gt_df, gen_df, case_id_col)

    merged = _aligned_sequence_frame(
        gt_sub, gen_sub, case_id_col, time_col, pos_col, sequence_attr_cols_num
    )

    results = {}

    for col in sequence_attr_cols_num:
        gt_col = f"{col}_gt"
        gen_col = f"{col}_gen"

        gt_vals = _numeric_series_or_nan(merged, gt_col)
        gen_vals = _numeric_series_or_nan(merged, gen_col)

        valid = ~(gt_vals.isna() | gen_vals.isna())

        if valid.any():
            results[f"seq_{col}_mae_conditional"] = float(
                (gen_vals[valid] - gt_vals[valid]).abs().mean()
            )
        else:
            results[f"seq_{col}_mae_conditional"] = np.nan

    return results


def sequence_attribute_mae_unconditional(
    gen_df,
    gt_df,
    case_id_col,
    sequence_attr_cols_num,
    time_col,
    pos_col,
    penalties,
):
    merged = _aligned_sequence_frame(
        gt_df, gen_df, case_id_col, time_col, pos_col, sequence_attr_cols_num
    )

    results = {}

    for col in sequence_attr_cols_num:
        gt_col = f"{col}_gt"
        gen_col = f"{col}_gen"
        penalty = penalties.get(col, np.nan)

        gt_vals = _numeric_series_or_nan(merged, gt_col)
        gen_vals = _numeric_series_or_nan(merged, gen_col)

        gt_exists = ~gt_vals.isna()
        valid = ~(gt_vals.isna() | gen_vals.isna())

        errors = pd.Series(np.nan, index=merged.index, dtype=float)
        errors.loc[gt_exists] = penalty
        errors.loc[valid] = (gen_vals[valid] - gt_vals[valid]).abs()

        results[f"seq_{col}_mae_unconditional"] = (
            float(errors.loc[gt_exists].mean()) if gt_exists.any() else np.nan
        )

    return results


# =========================================================
# EVENT numeric MAE
# =========================================================
def event_attribute_mae_conditional(
    gen_df,
    gt_df,
    case_id_col,
    event_attr_cols_num,
    time_col,
    pos_col,
):
    gt_sub, gen_sub = _filter_common_cases(gt_df, gen_df, case_id_col)

    merged = _aligned_event_frame(
        gt_sub, gen_sub, case_id_col, time_col, pos_col, event_attr_cols_num
    )

    results = {}

    for col in event_attr_cols_num:
        gt_col = f"{col}_gt"
        gen_col = f"{col}_gen"

        gt_vals = _numeric_series_or_nan(merged, gt_col)
        gen_vals = _numeric_series_or_nan(merged, gen_col)

        valid = ~(gt_vals.isna() | gen_vals.isna())

        if valid.any():
            results[f"event_{col}_mae_conditional"] = float(
                (gen_vals[valid] - gt_vals[valid]).abs().mean()
            )
        else:
            results[f"event_{col}_mae_conditional"] = np.nan

    return results


def event_attribute_mae_unconditional(
    gen_df,
    gt_df,
    case_id_col,
    event_attr_cols_num,
    time_col,
    pos_col,
    penalties,
):
    merged = _aligned_event_frame(
        gt_df, gen_df, case_id_col, time_col, pos_col, event_attr_cols_num
    )

    results = {}

    for col in event_attr_cols_num:
        gt_col = f"{col}_gt"
        gen_col = f"{col}_gen"
        penalty = penalties.get(col, np.nan)

        gt_vals = _numeric_series_or_nan(merged, gt_col)
        gen_vals = _numeric_series_or_nan(merged, gen_col)

        gt_exists = ~gt_vals.isna()
        valid = ~(gt_vals.isna() | gen_vals.isna())

        errors = pd.Series(np.nan, index=merged.index, dtype=float)
        errors.loc[gt_exists] = penalty
        errors.loc[valid] = (gen_vals[valid] - gt_vals[valid]).abs()

        results[f"event_{col}_mae_unconditional"] = (
            float(errors.loc[gt_exists].mean()) if gt_exists.any() else np.nan
        )

    return results


# =========================================================
# TIMESTAMP MAE
# =========================================================
def timestamp_mae_conditional(
    gen_df,
    gt_df,
    case_id_col,
    time_col,
    pos_col,
):
    gt_sub, gen_sub = _filter_common_cases(gt_df, gen_df, case_id_col)

    merged = _aligned_event_frame(
        gt_sub, gen_sub, case_id_col, time_col, pos_col, [time_col]
    )

    gt_times = _datetime_series_or_nat(merged, f"{time_col}_gt")
    gen_times = _datetime_series_or_nat(merged, f"{time_col}_gen")

    valid = ~(gt_times.isna() | gen_times.isna())

    if valid.any():
        diff_sec = (gen_times[valid] - gt_times[valid]).abs().dt.total_seconds()
        return float(diff_sec.mean())

    return np.nan


def timestamp_mae_unconditional(
    gen_df,
    gt_df,
    case_id_col,
    time_col,
    pos_col,
    penalty,
):
    merged = _aligned_event_frame(
        gt_df, gen_df, case_id_col, time_col, pos_col, [time_col]
    )

    gt_times = _datetime_series_or_nat(merged, f"{time_col}_gt")
    gen_times = _datetime_series_or_nat(merged, f"{time_col}_gen")

    gt_exists = ~gt_times.isna()
    valid = ~(gt_times.isna() | gen_times.isna())

    errors = pd.Series(np.nan, index=merged.index, dtype=float)
    errors.loc[gt_exists] = penalty
    errors.loc[valid] = (gen_times[valid] - gt_times[valid]).abs().dt.total_seconds()

    return float(errors.loc[gt_exists].mean()) if gt_exists.any() else np.nan


# =========================================================
# SEQUENCE consistency
# sanity check only
# =========================================================
def check_sequence_attribute_consistency(gen_df, case_id_col, sequence_attr_cols):
    inconsistencies = {}
    n_cases = gen_df[case_id_col].nunique()

    for col in sequence_attr_cols:
        if col not in gen_df.columns:
            continue

        inconsistent_cases = int(
            (gen_df.groupby(case_id_col)[col].nunique(dropna=False) > 1).sum()
        )

        inconsistencies[f"seq_{col}_inconsistent_cases"] = inconsistent_cases
        inconsistencies[f"seq_{col}_inconsistent_pct"] = (
            float(inconsistent_cases / n_cases) if n_cases > 0 else np.nan
        )

    return inconsistencies


# =========================================================
# ACTIVITY vocabulary
# =========================================================
def activity_vocabulary_metrics(gen_df, gt_df, activity_col):
    valid_activities = set(gt_df[activity_col].dropna().unique())
    gen_activities = set(gen_df[activity_col].dropna().unique())

    hallucinated = gen_activities - valid_activities
    generated_valid = gen_activities & valid_activities

    return {
        "hallucinated_activities": len(hallucinated),
        "activity_recall": (
            len(generated_valid) / len(valid_activities)
            if valid_activities else 0.0
        ),
    }
    
def evaluate_comprehensive(
    gen_df,
    gt_df,
    case_id_col,
    activity_col,
    time_col,
    pos_col,

    # event-level attributes
    event_attr_cols_cat=None,
    event_attr_cols_num=None,
    event_attr_cols_bol=None,

    # sequence-level attributes
    sequence_attr_cols_cat=None,
    sequence_attr_cols_num=None,
    sequence_attr_cols_bol=None,

    name="Model",
    jsd_lambda=1.0,
):
    """
    Comprehensive evaluation for generated event logs.

    Design:
    - Conditional metrics:
        evaluate only on common cases, and only on valid aligned pairs
    - Unconditional metrics:
        keep all GT cases/events and penalize missing generation

    Notes:
    - event categorical accuracy is MICRO over aligned event positions
      so it matches model-style token/event accuracy
    - sequence attributes are collapsed to one row per case using strict consistency
    - boolean attributes are treated as categorical for accuracy metrics
    """

    # -----------------------------
    # normalize optional inputs
    # -----------------------------
    event_attr_cols_cat = event_attr_cols_cat or []
    event_attr_cols_num = event_attr_cols_num or []
    event_attr_cols_bol = event_attr_cols_bol or []

    sequence_attr_cols_cat = sequence_attr_cols_cat or []
    sequence_attr_cols_num = sequence_attr_cols_num or []
    sequence_attr_cols_bol = sequence_attr_cols_bol or []

    # treat booleans as categorical for accuracy
    all_event_cat = event_attr_cols_cat + event_attr_cols_bol
    all_seq_cat = sequence_attr_cols_cat + sequence_attr_cols_bol

    # -----------------------------
    # build penalties once
    # -----------------------------
    event_num_penalties = (
        build_event_numeric_penalties(
            gen_df=gen_df,
            gt_df=gt_df,
            case_id_col=case_id_col,
            event_attr_cols_num=event_attr_cols_num,
            time_col=time_col,
            pos_col=pos_col,
            fallback=1.0,
        )
        if event_attr_cols_num else {}
    )

    seq_num_penalties = (
        build_sequence_numeric_penalties(
            gen_df=gen_df,
            gt_df=gt_df,
            case_id_col=case_id_col,
            sequence_attr_cols_num=sequence_attr_cols_num,
            time_col=time_col,
            pos_col=pos_col,
            fallback=1.0,
        )
        if sequence_attr_cols_num else {}
    )

    ts_penalty = build_timestamp_penalty(
        gen_df=gen_df,
        gt_df=gt_df,
        case_id_col=case_id_col,
        time_col=time_col,
        pos_col=pos_col,
        fallback=1.0,
    )

    # -----------------------------
    # core results
    # -----------------------------
    results = {
        # coverage
        "seq_coverage": case_coverage(gen_df, gt_df, case_id_col),

        # control-flow / trace quality
        "bigram_jsd_unconditional": bigram_jsd_with_coverage_penalty(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col,
            lambda_penalty=jsd_lambda
        ),
        "trace_similarity_unconditional": trace_similarity_unconditional(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col
        ),
        "duration_wd_unconditional": duration_wd_with_coverage_penalty(
            gen_df, gt_df, case_id_col, time_col
        ),
        "dl_similarity_unconditional": dl_trace_similarity_unconditional(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col
        ),

        "bigram_jsd_conditional": bigram_jsd(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col
        ),
        "trace_similarity_conditional": trace_similarity_conditional(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col
        ),
        "duration_wd_conditional": duration_wd(
            gen_df, gt_df, case_id_col, time_col
        ),
        "dl_similarity_conditional": dl_trace_similarity_conditional(
            gen_df, gt_df, case_id_col, activity_col, time_col, pos_col
        ),

        # activity vocabulary
        **activity_vocabulary_metrics(gen_df, gt_df, activity_col),
    }

    # -----------------------------
    # event categorical / boolean
    # -----------------------------
    if all_event_cat:
        results.update(
            event_attribute_accuracy_unconditional(
                gen_df=gen_df,
                gt_df=gt_df,
                case_id_col=case_id_col,
                event_attr_cols=all_event_cat,
                time_col=time_col,
                pos_col=pos_col,
            )
        )
        results.update(
            event_attribute_accuracy_conditional(
                gen_df=gen_df,
                gt_df=gt_df,
                case_id_col=case_id_col,
                event_attr_cols=all_event_cat,
                time_col=time_col,
                pos_col=pos_col,
            )
        )

    # -----------------------------
    # event numeric
    # -----------------------------
    if event_attr_cols_num:
        results.update(
            event_attribute_mae_unconditional(
                gen_df=gen_df,
                gt_df=gt_df,
                case_id_col=case_id_col,
                event_attr_cols_num=event_attr_cols_num,
                time_col=time_col,
                pos_col=pos_col,
                penalties=event_num_penalties,
            )
        )
        results.update(
            event_attribute_mae_conditional(
                gen_df=gen_df,
                gt_df=gt_df,
                case_id_col=case_id_col,
                event_attr_cols_num=event_attr_cols_num,
                time_col=time_col,
                pos_col=pos_col,
            )
        )

    # -----------------------------
    # sequence categorical / boolean
    # -----------------------------
    if all_seq_cat:
        results.update(
            sequence_attribute_accuracy_unconditional(
                gen_df=gen_df,
                gt_df=gt_df,
                case_id_col=case_id_col,
                sequence_attr_cols=all_seq_cat,
                time_col=time_col,
                pos_col=pos_col,
            )
        )
        results.update(
            sequence_attribute_accuracy_conditional(
                gen_df=gen_df,
                gt_df=gt_df,
                case_id_col=case_id_col,
                sequence_attr_cols=all_seq_cat,
                time_col=time_col,
                pos_col=pos_col,
            )
        )

    # -----------------------------
    # sequence numeric
    # -----------------------------
    if sequence_attr_cols_num:
        results.update(
            sequence_attribute_mae_unconditional(
                gen_df=gen_df,
                gt_df=gt_df,
                case_id_col=case_id_col,
                sequence_attr_cols_num=sequence_attr_cols_num,
                time_col=time_col,
                pos_col=pos_col,
                penalties=seq_num_penalties,
            )
        )
        results.update(
            sequence_attribute_mae_conditional(
                gen_df=gen_df,
                gt_df=gt_df,
                case_id_col=case_id_col,
                sequence_attr_cols_num=sequence_attr_cols_num,
                time_col=time_col,
                pos_col=pos_col,
            )
        )

    # -----------------------------
    # timestamp
    # -----------------------------
    results["timestamp_mae_unconditional"] = timestamp_mae_unconditional(
        gen_df=gen_df,
        gt_df=gt_df,
        case_id_col=case_id_col,
        time_col=time_col,
        pos_col=pos_col,
        penalty=ts_penalty,
    )
    results["timestamp_mae_conditional"] = timestamp_mae_conditional(
        gen_df=gen_df,
        gt_df=gt_df,
        case_id_col=case_id_col,
        time_col=time_col,
        pos_col=pos_col,
    )

    # -----------------------------
    # sequence consistency diagnostics
    # -----------------------------
    all_seq_attrs = all_seq_cat + sequence_attr_cols_num
    if all_seq_attrs:
        results.update(
            check_sequence_attribute_consistency(
                gen_df=gen_df,
                case_id_col=case_id_col,
                sequence_attr_cols=all_seq_attrs,
            )
        )

    return pd.DataFrame([results], index=[name])