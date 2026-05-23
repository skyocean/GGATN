import pandas as pd
import numpy as np
from dateutil import parser
from pathlib import Path
import pickle
import json
import random
import re
from ollama import Client

def sanitize(obj):
    """Convert numpy/pandas scalars to pure Python types."""
    import numpy as np

    # None stays None
    if obj is None:
        return None

    # numpy integer → python int
    if isinstance(obj, (np.integer,)):
        return int(obj)

    # numpy float → python float
    if isinstance(obj, (np.floating,)):
        return float(obj)

    # numpy boolean → python bool
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # python primitives
    if isinstance(obj, (int, float, bool, str)):
        return obj

    # list → sanitize each element
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]

    # dict → sanitize each value
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}

    # fallback: convert to string
    return str(obj)

def make_seq2seq_pairs(
        event,
        case_id_col="case_concept_name",
        timestamp_col="time_timestamp",
        activity_col="concept_name",

        # event-level typed attributes
        event_attr_cols_cat=None,
        event_attr_cols_num=None,
        event_attr_cols_bol=None,

        # sequence-level typed attributes
        sequence_attr_cols_cat=None,
        sequence_attr_cols_num=None,
        sequence_attr_cols_bol=None,

        output_file="seq2seq_pairs",
        return_pairs=False):
    
    """
    Unified seq2seq dataset builder:
      ✔ Builds <INPUT>/<OUTPUT> text format
      ✔ Computes metadata with correct typing
      ✔ Writes txt file
      ✔ Optionally returns structured pairs with typed metadata
    """

    # Ensure lists
    event_attr_cols_cat = event_attr_cols_cat or []
    event_attr_cols_num = event_attr_cols_num or []
    event_attr_cols_bol = event_attr_cols_bol or []

    sequence_attr_cols_cat = sequence_attr_cols_cat or []
    sequence_attr_cols_num = sequence_attr_cols_num or []
    sequence_attr_cols_bol = sequence_attr_cols_bol or []

    # Containers
    file_lines = []
    structured_pairs = []

    # ================================================================
    # GROUP BY CASE
    # ================================================================
    for case_id, group in event.groupby(case_id_col):

        # Already sorted by timestamp outside
        case_length = len(group)

        # Core metadata
        start_time = group[timestamp_col].iloc[0]
        end_time   = group[timestamp_col].iloc[-1]

        case_cycle_time_mins = (end_time - start_time).total_seconds() / 60
        activities = group[activity_col].values
        timestamps = group[timestamp_col].astype(str).values

        unique_activities = pd.unique(activities).size
        has_loops = group[activity_col].duplicated().any()

        # Mean inter-event
        diffs = group[timestamp_col].diff().dt.total_seconds().to_numpy()
        diffs = diffs[~np.isnan(diffs)]
        mean_inter_event_mins = diffs.mean() / 60 if diffs.size > 0 else 0.0

        start_weekday = start_time.day_name()
        start_hour = start_time.hour

        # ------------------------------------------------------------
        # EVENT-LEVEL ATTRIBUTE SUMMARIES (type consistent)
        # ------------------------------------------------------------
        event_summary = {}

        # categorical summaries
        for col in event_attr_cols_cat:
            s = group[col].astype(str)
            uniq = s.nunique()
            mode_vals = s.mode()
            mode_val = mode_vals.iloc[0] if not mode_vals.empty else None
            freq = (s == mode_val).sum()

            event_summary[f"{col}_unique"] = int(uniq)
            event_summary[f"{col}_mode"] = str(mode_val)
            event_summary[f"{col}_mode_count"] = int(freq)

        # numerical summaries
        for col in event_attr_cols_num:
            s = pd.to_numeric(group[col], errors="coerce")
            if s.notna().any():
                event_summary[f"{col}_mean"] = float(s.mean(skipna=True))
                event_summary[f"{col}_std"]  = float(s.std(skipna=True))
                event_summary[f"{col}_min"]  = float(s.min(skipna=True))
                event_summary[f"{col}_max"]  = float(s.max(skipna=True))
            else:
                event_summary[f"{col}_mean"] = None
                event_summary[f"{col}_std"]  = None
                event_summary[f"{col}_min"]  = None
                event_summary[f"{col}_max"] = None


        # boolean summaries
        for col in event_attr_cols_bol:
            s = group[col].astype(float)
            event_summary[f"{col}_true_pct"] = round(float(s.mean()), 3)

        # ------------------------------------------------------------
        # BUILD METADATA (typed dict)
        # ------------------------------------------------------------
        metadata = {
            "case_id": str(case_id),
            "case_length": int(case_length),
            "case_cycle_time_mins": float(case_cycle_time_mins),
            "unique_activities": int(unique_activities),
            "mean_inter_event_mins": float(mean_inter_event_mins),
            "has_loops": bool(has_loops),

            "start_time": str(start_time),
            "start_weekday": str(start_weekday),
            "start_hour": int(start_hour),
        }

        # sequence categorical
        first_row = group.iloc[0]
        for col in sequence_attr_cols_cat:
            metadata[col] = str(first_row[col])

        # sequence numerical
        for col in sequence_attr_cols_num:
            metadata[col] = float(first_row[col])

        # sequence boolean
        for col in sequence_attr_cols_bol:
            metadata[col] = bool(first_row[col])

        # add event-level summaries (already typed)
        metadata.update(event_summary)

        # ------------------------------------------------------------
        # BUILD INPUT TEXT BLOCK
        # ------------------------------------------------------------
        input_lines = ["Case attributes:"]
        for k, v in metadata.items():
            input_lines.append(f"  {k} = {v}")

        input_lines.append("Task: generate the full event trace until completion.")
        input_text = "\n".join(input_lines)

        # ------------------------------------------------------------
        # BUILD OUTPUT TEXT BLOCK
        # ------------------------------------------------------------
        all_event_cols = (event_attr_cols_cat
                          + event_attr_cols_num
                          + event_attr_cols_bol
                          + sequence_attr_cols_cat
                          + sequence_attr_cols_num
                          + sequence_attr_cols_bol
                         )

        ev_arrays = {col: group[col].astype(str).values for col in event_attr_cols_cat + event_attr_cols_num + event_attr_cols_bol}
        seq_arrays = {col: [str(group.iloc[0][col])] * case_length 
                      for col in sequence_attr_cols_cat + sequence_attr_cols_num + sequence_attr_cols_bol}
        all_arrays = {**ev_arrays, **seq_arrays}

        output_lines = [ f"{i+1}. {activities[i]} | "
                         + " | ".join(f"{col}:{all_arrays[col][i]}" for col in all_event_cols) 
                         + f" | {timestamp_col}:{timestamps[i]}"
                         for i in range(case_length)
                       ]

        output_lines.append(f"{case_length+1}. END")
        output_text = "\n".join(output_lines)

        # ---------------------------
        # APPEND TO TXT BUFFER
        # ---------------------------
        file_lines.append("<INPUT>")
        file_lines.append(input_text)
        file_lines.append("<OUTPUT>")
        file_lines.append(output_text)
        file_lines.append("========")

        # ---------------------------
        # ADD STRUCTURED PAIR
        # ---------------------------
        if return_pairs:
            clean_metadata = sanitize(metadata)
            structured_pairs.append({
                "input": input_text,
                "output": output_text,
                "metadata": clean_metadata
            })

    # ================================================================
    # WRITE FILES
    # ================================================================
    # Create paths for different file types
    txt_out = Path(f"{output_file}.txt")
    pkl_out = Path(f"{output_file}.pkl")
    jsonl_out = Path(f"{output_file}.jsonl")
    
    
    txt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(file_lines))

    print(f"Saved seq2seq training file to: {txt_out}")

    if return_pairs:
        # Save pickle
        with open(pkl_out, "wb") as f:
            pickle.dump(structured_pairs, f)
        
        # Save JSONL
        with open(jsonl_out, "w", encoding="utf-8") as f:
            for row in structured_pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return structured_pairs

def sample_cases(pairs, k, min_len=None, max_len=None):
    """
    Automatically builds a few-shot prompt:
      - Loads training pairs
      - Optionally filters by case_length ∈ [min_len, max_len)
      - Samples k examples (or fewer if not available)
    """
    
    # ---------------------------------------------------------
    # 1. FILTER BY LENGTH RANGE IF PROVIDED
    # ---------------------------------------------------------
    if min_len is not None and max_len is not None:
        filtered = []
        for p in pairs:
            cl = p["metadata"].get("case_length", None)
            if cl is not None and min_len <= cl < max_len:
                filtered.append(p)
                
        if len(filtered) == 0:
            print(f"[WARN] No examples with case_length in [{min_len}, {max_len}). "
                  f"Returned an empty list. Adjust length constraints.")
            return []  
    else:
        filtered = pairs

    # ---------------------------------------------------------
    # 2. HANDLE CASE WHERE k > available samples
    # ---------------------------------------------------------
    if k > len(filtered):
        print(f"[WARN] Requested k={k}, but only {len(filtered)} examples available. "
              f"Using all {len(filtered)} examples.")
        sampled = filtered
    else:
        sampled = random.sample(filtered, k)

    return sampled

def filter_cases_by_length(pairs, min_len=None, max_len=None):
    """
    Filter cases by case_length ∈ [min_len, max_len).

    Returns all matching cases (order preserved).
    """
    if min_len is None and max_len is None:
        return pairs

    filtered = []
    for p in pairs:
        cl = p["metadata"].get("case_length", None)
        if cl is None:
            continue
        if max_len is None:
            if cl >= min_len:
                filtered.append(p)
        else:
            if min_len <= cl < max_len:
                filtered.append(p)

    if len(filtered) == 0:
        print(f"[WARN] No cases with case_length in [{min_len}, {max_len})."
              f"Returned an empty list. Adjust length constraints.")
    return filtered


def sample_few_shots(train_pairs, k, min_len, max_len, max_expand=3):
    for expand in range(max_expand + 1):
        lo = max(1, min_len - expand)
        if max_len is None:
            hi = None
        else:
            hi = max_len + expand

        candidates = filter_cases_by_length(train_pairs, lo, hi)

        if len(candidates) >= k:
            return random.sample(candidates, k)

    # final fallback: ignore length constraint
    print("[WARN] Falling back to global sampling (length constraint dropped).")
    return random.sample(train_pairs, min(k, len(train_pairs)))


def reduce_input_block(parsed_entry):
    """
    Create reduced inference-time <INPUT> from a parsed metadata entry.
    
    parsed_entry is one element from make_seq2seq_pairs, i.e.:
    {
        "input": "...original block...",
        "output": "...",
        "metadata": { key: value, ... }
        }
    """

    meta = parsed_entry["metadata"]

    # Allowed inference-time fields
    allowed_fields = [
        "case_id",
        "case_length",
        "start_time",
        "start_weekday",
        "start_hour"
    ]

    lines = ["Case attributes:"]

    # Append safe metadata
    for key in allowed_fields:
        if key in meta:
            lines.append(f"  {key} = {meta[key]}")

    # allow using case_length for stronger conditioning
    #if use_case_length and "case_length" in parsed_entry:
        #lines.append(f"  case_length = {parsed_entry['case_length']}")

    lines.append("Task: generate the full event trace until completion.")
    return "\n".join(lines)


def make_seq2seq_pairs_reduced(
        event,
        case_id_col="case_concept_name",
        timestamp_col="time_timestamp",
        activity_col="concept_name",

        # event-level typed attributes
        event_attr_cols_cat=None,
        event_attr_cols_num=None,
        event_attr_cols_bol=None,

        # sequence-level typed attributes
        sequence_attr_cols_cat=None,
        sequence_attr_cols_num=None,
        sequence_attr_cols_bol=None,

        output_file="seq2seq_pairs",
        return_pairs=False):
    
    """
    Unified seq2seq dataset builder:
      ✔ Builds <INPUT>/<OUTPUT> text format
      ✔ Computes metadata with correct typing
      ✔ Writes txt file
      ✔ Optionally returns structured pairs with typed metadata
    """

    # Ensure lists
    event_attr_cols_cat = event_attr_cols_cat or []
    event_attr_cols_num = event_attr_cols_num or []
    event_attr_cols_bol = event_attr_cols_bol or []

    sequence_attr_cols_cat = sequence_attr_cols_cat or []
    sequence_attr_cols_num = sequence_attr_cols_num or []
    sequence_attr_cols_bol = sequence_attr_cols_bol or []

    # Containers
    file_lines = []
    structured_pairs = []

    # ================================================================
    # GROUP BY CASE
    # ================================================================
    for case_id, group in event.groupby(case_id_col):

        # Already sorted by timestamp outside
        case_length = len(group)

        # Core metadata
        start_time = group[timestamp_col].iloc[0]
        end_time   = group[timestamp_col].iloc[-1]

        case_cycle_time_mins = (end_time - start_time).total_seconds() / 60
        activities = group[activity_col].values
        timestamps = group[timestamp_col].astype(str).values

        unique_activities = pd.unique(activities).size
        has_loops = group[activity_col].duplicated().any()

        # Mean inter-event
        diffs = group[timestamp_col].diff().dt.total_seconds().to_numpy()
        diffs = diffs[~np.isnan(diffs)]
        mean_inter_event_mins = diffs.mean() / 60 if diffs.size > 0 else 0.0

        start_weekday = start_time.day_name()
        start_hour = start_time.hour

        # ------------------------------------------------------------
        # EVENT-LEVEL ATTRIBUTE SUMMARIES (type consistent)
        # ------------------------------------------------------------
        event_summary = {}

        # categorical summaries
        for col in event_attr_cols_cat:
            s = group[col].astype(str)
            uniq = s.nunique()
            mode_vals = s.mode()
            mode_val = mode_vals.iloc[0] if not mode_vals.empty else None
            freq = (s == mode_val).sum()

            event_summary[f"{col}_unique"] = int(uniq)
            event_summary[f"{col}_mode"] = str(mode_val)
            event_summary[f"{col}_mode_count"] = int(freq)

        # numerical summaries
        for col in event_attr_cols_num:
            s = pd.to_numeric(group[col], errors="coerce")
            if s.notna().any():
                event_summary[f"{col}_mean"] = float(s.mean(skipna=True))
                event_summary[f"{col}_std"]  = float(s.std(skipna=True))
                event_summary[f"{col}_min"]  = float(s.min(skipna=True))
                event_summary[f"{col}_max"]  = float(s.max(skipna=True))
            else:
                event_summary[f"{col}_mean"] = None
                event_summary[f"{col}_std"]  = None
                event_summary[f"{col}_min"]  = None
                event_summary[f"{col}_max"] = None


        # boolean summaries
        for col in event_attr_cols_bol:
            s = group[col].astype(float)
            event_summary[f"{col}_true_pct"] = round(float(s.mean()), 3)

        # ------------------------------------------------------------
        # BUILD METADATA (typed dict)
        # ------------------------------------------------------------
        metadata = {
            "case_id": str(case_id),
            "case_length": int(case_length),
            "case_cycle_time_mins": float(case_cycle_time_mins),
            "unique_activities": int(unique_activities),
            "mean_inter_event_mins": float(mean_inter_event_mins),
            "has_loops": bool(has_loops),

            "start_time": str(start_time),
            "start_weekday": str(start_weekday),
            "start_hour": int(start_hour),
        }

        # sequence categorical
        first_row = group.iloc[0]
        for col in sequence_attr_cols_cat:
            metadata[col] = str(first_row[col])

        # sequence numerical
        for col in sequence_attr_cols_num:
            metadata[col] = float(first_row[col])

        # sequence boolean
        for col in sequence_attr_cols_bol:
            metadata[col] = bool(first_row[col])

        # add event-level summaries (already typed)
        metadata.update(event_summary)

        # ------------------------------------------------------------
        # BUILD INPUT TEXT BLOCK
        input_lines = ["Case attributes:"]
        input_lines.append(f"  case_id = {metadata['case_id']}")
        input_lines.append(f"  start_time = {metadata['start_time']}")
        input_lines.append(f"  start_weekday = {metadata['start_weekday']}")
        input_lines.append(f"  start_hour = {metadata['start_hour']}")
        input_lines.append(f"  case_length = {metadata['case_length']}")

        input_lines.append("Task: generate the full event trace until completion.")
        input_text = "\n".join(input_lines)

        # ------------------------------------------------------------
        # BUILD OUTPUT TEXT BLOCK
        # ------------------------------------------------------------
        all_event_cols = (event_attr_cols_cat
                          + event_attr_cols_num
                          + event_attr_cols_bol
                          + sequence_attr_cols_cat
                          + sequence_attr_cols_num
                          + sequence_attr_cols_bol
                         )

        ev_arrays = {col: group[col].astype(str).values for col in event_attr_cols_cat + event_attr_cols_num + event_attr_cols_bol}
        seq_arrays = {col: [str(group.iloc[0][col])] * case_length 
                      for col in sequence_attr_cols_cat + sequence_attr_cols_num + sequence_attr_cols_bol}
        all_arrays = {**ev_arrays, **seq_arrays}

        output_lines = [ f"{i+1}. {activities[i]} | "
                         + " | ".join(f"{col}:{all_arrays[col][i]}" for col in all_event_cols) 
                         + f" | {timestamp_col}:{timestamps[i]}"
                         for i in range(case_length)
                       ]


        output_lines.append(f"{case_length+1}. END")
        output_text = "\n".join(output_lines)

        # ---------------------------
        # APPEND TO TXT BUFFER
        # ---------------------------
        file_lines.append("<INPUT>")
        file_lines.append(input_text)
        file_lines.append("<OUTPUT>")
        file_lines.append(output_text)
        file_lines.append("========")

        # ---------------------------
        # ADD STRUCTURED PAIR
        # ---------------------------
        if return_pairs:
            clean_metadata = sanitize(metadata)
            structured_pairs.append({
                "input": input_text,
                "output": output_text,
                "metadata": clean_metadata
            })

    # ================================================================
    # WRITE FILES
    # ================================================================
    # Create paths for different file types
    txt_out = Path(f"{output_file}.txt")
    pkl_out = Path(f"{output_file}.pkl")
    jsonl_out = Path(f"{output_file}.jsonl")
    
    
    txt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(file_lines))

    print(f"Saved seq2seq training file to: {txt_out}")

    if return_pairs:
        # Save pickle
        with open(pkl_out, "wb") as f:
            pickle.dump(structured_pairs, f)
        
        # Save JSONL
        with open(jsonl_out, "w", encoding="utf-8") as f:
            for row in structured_pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return structured_pairs

import re

def normalize_llm_trace(raw_text, case_length):
    """
    Normalize a raw LLM-generated trace into a clean, numbered event log:

      - Drops explanation / apology / meta-text lines.
      - Keeps only lines that look like events (numbered or containing '|').
      - Strips original numbering and re-numbers from 1..case_length.
      - Truncates to case_length events.
      - Appends final 'case_length+1. END'.

    Returns:
        cleaned_trace (str) or "" if nothing usable found.
    """

    lines = raw_text.splitlines()
    event_bodies = []

    # Words that strongly indicate explanation, not log events
    ban_keywords = [
        "unfortunately", "unable", "apolog", "because",
        "for example", "for instance", "note that",
        "as an ai", "i am not", "cannot provide"
    ]

    for line in lines:
        s = line.strip()
        if not s:
            continue

        low = s.lower()

        # Drop obvious commentary / explanation
        if any(k in low for k in ban_keywords):
            continue

        # Hard drop pure END lines; we will generate our own
        if s.upper() == "END" or re.match(r"^\d+\.\s*END\s*$", s):
            continue

        # Candidate event if:
        # - starts with a number + dot (e.g., "1. Assign ...")
        # - OR contains '|' and ':' (our attribute style)
        if re.match(r"^\d+\.", s) or ("|" in s and ":" in s):
            # If numbered, strip the number and dot
            m = re.match(r"^\d+\.\s*(.*)", s)
            body = m.group(1).strip() if m else s

            # Drop anything that clearly isn't an event
            # (no pipe AND no colon -> probably narrative)
            if "|" not in body and ":" not in body:
                continue

            event_bodies.append(body)

    # If nothing usable, return empty string and let caller handle failure
    if len(event_bodies) == 0:
        return ""

    # Enforce exact case_length events
    event_bodies = event_bodies[:case_length]

    # Renumber
    out_lines = []
    for i, body in enumerate(event_bodies, start=1):
        out_lines.append(f"{i}. {body}")

    # Final END line
    out_lines.append(f"{case_length+1}. END")

    return "\n".join(out_lines)

def assemble_prompt_from_fewshots(few_shots, new_input_block):
    """
    Build a prompt using pre-sampled few_shots + the new input block.
    """
    instruction = (
        "You are an expert BPM event-log generator.\n"
        
        "You MUST output ONLY the content between <TRACE> and </TRACE>.\n"
        "Anything outside these tags is strictly forbidden.\n"
        
        "Follow these rules strictly:\n"
        "1. Your answer MUST ONLY be the event trace, nothing else.\n"
        "2. Output the full event trace in EXACTLY the same style as the examples.\n"
        "3. Each line MUST start with an integer index followed by a dot, e.g. '1.' '2.' '3.'.\n"
        "4. Each line MUST preserve attribute names and MUST follow the format:\n"
        "   <index>. <activity> | <attr1>:<value1> | <attr2>:<value2> | ... | <timestamp>\n"
        "5. You MUST maintain chronological timestamps.\n"
        "6. Let case_length be the number given in the input. You MUST produce exactly case_length events.\n"
        "7. After those events, you MUST output exactly one final line: (case_length+1). END\n"
        "8. Do NOT output any explanations, apologies, comments, or meta-text.\n"
        "9. If information is missing or uncertain, make a reasonable guess and STILL output a full trace.\n"
        "10. NEVER mention that information is missing. Just output the best possible trace.\n"
    )

    # Build few-shot examples
    example_blocks = []
    for i, ex in enumerate(few_shots, start=1):
        block = (
            f"\n### Example {i}\n"
            f"<INPUT>\n{ex['input']}\n"
            f"<OUTPUT>\n{ex['output']}\n"
        )
        example_blocks.append(block)

    examples_text = "".join(example_blocks)
    
    # Final query
    final_query = (
        "\n### New case\n"
        "<INPUT>\n"
        f"{new_input_block}\n"
        "<OUTPUT>\n"
    )

    prompt = instruction + examples_text + final_query
    return prompt


def generate_traces_for_batch(new_cases, few_shots, model, client, context_size=32768):
    """
    new_cases : list of INPUT blocks (each string)
    few_shots : pre-sampled few-shot examples
    model : model name for client.chat
    client : Ollama client instance

    Returns a list of outputs (same order as new_cases)
    """

    results = []

    for new_input in new_cases:
        prompt = assemble_prompt_from_fewshots(few_shots, new_input)

        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={  # Add this!
                "num_ctx": context_size
            }
        )

        content = response["message"]["content"]
        print(content)
        results.append({
            "input": new_input,
            "generated_trace": content
        })

    return results

import re
import pandas as pd

def llm_results_to_eventlog(results, case_id_key=None, activity_key=None, time_key=None, attr_keys=None):
    """
    Convert LLM-generated traces into a structured event log.
    Handles messy outputs and case IDs with spaces.
    """
    TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    attr_keys = attr_keys or []
    rows = []

    for r in results:
        # -------- case_id - improved to capture full ID with spaces --------
        case_id = None
        input_text = r.get("input", "")
        
        # Try different patterns
        patterns = [
            r"case_id\s*=\s*(.+?)(?:,|\n|$)",        # case_id = request for payment 73337
            r"case_id\s*:\s*(.+?)(?:,|\n|$)",        # case_id: request for payment 73337
            r'"case_id"\s*:\s*"([^"]+)"',             # "case_id": "request for payment 73337"
            r"case_id\s*=\s*'([^']+)'",              # case_id = 'request for payment 73337'
        ]
        for pat in patterns:
            m = re.search(pat, input_text)
            if m:
                case_id = m.group(1).strip()
                break
        
        # Fallback: if still None, try to find after "case_id =" anywhere
        if not case_id:
            m = re.search(r"case_id\s*=\s*(\S.*?)(?:,|\n|$)", input_text)
            if m:
                case_id = m.group(1).strip()
        
        # -------- bucket --------
        length_bucket = r.get("length_bucket", None)

        # -------- parse generated trace --------
        rebuilt_event_index = 0
        for line in r["generated_trace"].splitlines():
            line = line.strip()
            if not line:
                continue

            # Only lines that start with a number and a dot
            m = re.match(r"^(\d+)\.\s*(.*)$", line)
            if not m:
                # Also allow lines that start with a number and a dot with possible leading spaces
                m = re.match(r"^\s*(\d+)\.\s*(.*)$", line)
                if not m:
                    continue

            rest = m.group(2).strip()
            if rest.upper().startswith("END"):
                break

            # Split by pipe, but be careful: some events may have no pipe
            if "|" in rest:
                parts = [p.strip() for p in rest.split("|")]
            else:
                # No pipe: whole rest is activity, attributes empty
                parts = [rest]

    
            activity = parts[0].strip() if parts else None


            timestamp = None
            attr_vals = {k: None for k in attr_keys}

            # Parse remaining parts (attributes)
            for p in parts[1:]:
                # Try to find timestamp anywhere in the part
                tm = TIMESTAMP_RE.search(p)
                if tm:
                    timestamp = tm.group(0)
                # Extract attributes: they are usually in form key:value
                for k in attr_keys:
                    # Look for "key:" at beginning of part or after a space
                    if re.search(rf"\b{k}\s*:", p, re.IGNORECASE):
                        # Extract value after colon
                        val = p.split(":", 1)[1].strip()
                        # Remove any trailing backticks or quotes
                        val = val.strip('`').strip("'").strip('"')
                        attr_vals[k] = val

            # Build row
            row = {
                case_id_key: case_id,
                "pos": rebuilt_event_index,
                activity_key: activity,
                time_key: timestamp,
                "length_bucket": length_bucket
            }
            row.update(attr_vals)
            rows.append(row)
            rebuilt_event_index += 1

    return pd.DataFrame(rows)

def gt_add_col(gt_df, length_buckets, case_id_col, time_col, event_index_col, length_bucket_col):
    
    gt_df = gt_df.copy()
    
    # Ensure correct ordering
    gt_df = gt_df.sort_values([case_id_col, time_col])
    
    # Recompute event index
    gt_df[event_index_col] = gt_df.groupby(case_id_col).cumcount()
    
    # Compute case lengths
    case_lengths = gt_df.groupby(case_id_col).size()
    
    # Assign buckets
    case_to_bucket = {}
    for cid, length in case_lengths.items():
        for name, (lo, hi) in length_buckets.items():
            if hi is None:
                if length >= lo:
                    case_to_bucket[cid] = name
                    break
            else:
                if lo <= length < hi:
                    case_to_bucket[cid] = name
                    break
    
    # Add bucket column
    gt_df[length_bucket_col] = gt_df[case_id_col].map(case_to_bucket)
    
    return gt_df

def generate_traces_for_batch_hf(new_cases, few_shots, pipe, max_new_token_num):
    """
    HF equivalent of Ollama batch generation
    """

    results = []

    for new_input in new_cases:
        prompt = assemble_prompt_from_fewshots(few_shots, new_input)

        outputs = pipe(
            prompt,
            max_new_tokens=max_new_token_num,
            do_sample=False,
            return_full_text=False
        )

        content = outputs[0].get("generated_text", "").strip()

        # Extract TRACE block if present
        if "<TRACE>" in content:
            content = content.split("<TRACE>", 1)[1]
        if "</TRACE>" in content:
            content = content.split("</TRACE>", 1)[0]

        # Trim everything before first event index
        m = re.search(r"\b1\.\s", content)
        if m:
            content = content[m.start():]

        results.append({
            "input": new_input,
            "generated_trace": content.strip()
        })

    return results

import math

def in_bucket(length, min_len, max_len):
    if max_len is None:
        return length >= min_len
    return (length >= min_len) and (length < max_len)


def get_bucket_size(train_pairs, min_len, max_len):
    return sum(
        1
        for p in train_pairs
        if in_bucket(p["metadata"]["case_length"], min_len, max_len)
    )


def compute_bucket_k(bucket_size, min_k=3, max_k=10, scale=0.4):
    if bucket_size <= 0:
        return 0
    k = min(max_k, max(min_k, math.floor(0.2 * bucket_size)))
    return k


from pathlib import Path
import json
import os
import pickle
import pandas as pd

def save_checkpoint(state, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)

    os.replace(tmp_path, path)


def load_pickle(path):
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def append_df_to_csv(df, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)



