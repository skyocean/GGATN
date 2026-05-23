#!/usr/bin/env python
# coding: utf-8

import re
import json
import pandas as pd
import numpy as np
from collections import Counter
import pickle
from pathlib import Path

from LLMGen import make_seq2seq_pairs, reduce_input_block, make_seq2seq_pairs_reduced
from LLMGen import filter_cases_by_length, sample_few_shots
from LLMGen import assemble_prompt_from_fewshots, generate_traces_for_batch, llm_results_to_eventlog, gt_add_col, get_bucket_size, compute_bucket_k
from LLMGen import save_checkpoint, load_checkpoint, save_pickle, load_pickle, append_df_to_csv

from Evaluation import evaluate_comprehensive, evaluate_light
from ollama import Client


# In[2]:

#dataname = "helpdesk"
#dataname = "sepsis"
#dataname = "BPI13I"
#dataname = "BPI13C"
#dataname = "BPI20"
#dataname = "BPI12W"
#dataname = "BPI12"
#dataname = "BPI17"
dataname = "BPI20R"

context_size = 4096


# Reuse the previous train-test-split datasets as if runing the second LLM model

# In[3]:


train_event = pd.read_csv("../output/data_processed/" + dataname + "_train.csv")
test_event =  pd.read_csv("../output/data_processed/" + dataname + "_hold.csv")


# Prepare for LLM input

# In[4]:


# Rename all columns
def rename(event, timecol, sequence_id):
    event = event.copy()
    event[timecol] = pd.to_datetime(event[timecol])
    event = event.sort_values([sequence_id, timecol])
    event.columns = [c.replace(":", "_") for c in event.columns]
    return event

train_event = rename(train_event, "time:timestamp", "case:concept:name")
test_event = rename(test_event,  "time:timestamp", "case:concept:name")


# In[5]:


case_index = 'case_concept_name'
time_col = 'time_timestamp'
core_event = "concept_name"
delta_col='delta_time'


# In[6]:


if dataname == "helpdesk":
    cat_cols_event = ['org_resource']
    num_cols_event = []
    cat_cols_seq = ['case_variant']
    num_cols_seq = []
elif dataname == "BPI12" or dataname == "BPI12W":
    train_event[case_index] = train_event[case_index].astype(str)
    test_event[case_index] = test_event[case_index].astype(str)
    #train_df[case_index] = train_df[case_index].astype(str)
    #val_df[case_index] = val_df[case_index].astype(str)
    cat_cols_event = ['org_resource']
    num_cols_event = []
    cat_cols_seq = []
    num_cols_seq = ['case_AMOUNT_REQ']
elif dataname == "BPI13I" or dataname == "BPI13C":
    cat_cols_event = ['org_group', "resource country", "org_resource", "organization involved", "org_role"]
    num_cols_event = []
    cat_cols_seq = ["organization country", "impact", "product"]
    num_cols_seq = []
elif dataname == "sepsis":
    cat_cols_event = ['org_group']
    num_cols_event = ['Leucocytes', 'CRP', 'LacticAcid']
    cat_cols_seq = ['InfectionSuspected', 'DiagnosticBlood',     'DisfuncOrg',  'SIRSCritTachypnea', 'Hypotensie',       'SIRSCritHeartRate', 
                    'Infusion',           'DiagnosticArtAstrup', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther',
                    'SIRSCriteria2OrMore', 'DiagnosticXthorax',  'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie', 
                    'DiagnosticLacticAcid', 'Diagnose',          'Hypoxie',             'DiagnosticUrinarySediment', 'DiagnosticECG']
    num_cols_seq = [ 'Age']
elif dataname == "BPI17":
    cat_cols_event = ['Action', 'org_resource', 'EventOrigin', 'Accepted', 'Selected', "OfferID"]
    num_cols_event = ['FirstWithdrawalAmount', 'NumberOfTerms', 'MonthlyCost',  'CreditScore', 'OfferedAmount']
    cat_cols_seq = [ 'case_LoanGoal', 'case_ApplicationType']
    num_cols_seq = ['case_RequestedAmount']
elif dataname == "BPI20":
    cat_cols_event = ['org_role']
    num_cols_event = []
    cat_cols_seq = ["case_OrganizationalEntity", "case_Project"]
    num_cols_seq = ["case_RequestedAmount", "case_Permit RequestedBudget"] 
elif dataname == "BPI20R":
    cat_cols_event = ['org_resource','org_role']
    num_cols_event = []
    cat_cols_seq = ["case_OrganizationalEntity", "case_Project", "case_RfpNumber", "case_Task", "case_Activity"]
    num_cols_seq = ["case_RequestedAmount"]  

attr_cols = cat_cols_event + num_cols_event + cat_cols_seq + num_cols_seq


# In[7]:

model = "llama3.1:8b-instruct-q5_K_M"
#model = "mistral:7b-instruct-q5_K_M"

# this version has limited metadata
if model == "llama3.1:8b-instruct-q5_K_M" and context_size == 32768:
    train_pairs = make_seq2seq_pairs_reduced(event = train_event,
                                 case_id_col = case_index,
                                 timestamp_col = time_col,
                                 activity_col = core_event,
                                 # event-level attributes by type
                                 event_attr_cols_cat = cat_cols_event,
                                 event_attr_cols_num = num_cols_event,
                                 event_attr_cols_bol = None,
                                 # sequence-level attributes by type
                                 sequence_attr_cols_cat = cat_cols_seq,
                                 sequence_attr_cols_num = num_cols_seq,
                                 sequence_attr_cols_bol = None,
                                 output_file = "../output/data_processed/" + dataname +"_train_seq2seq_reduced",
                                 return_pairs = True
                                )

    test_pairs = make_seq2seq_pairs_reduced(event = test_event,
                                 case_id_col = case_index,
                                 timestamp_col = time_col,
                                 activity_col = core_event,
                                 # event-level attributes by type
                                 event_attr_cols_cat = cat_cols_event,
                                 event_attr_cols_num = num_cols_event,
                                 event_attr_cols_bol = None,
                                 # sequence-level attributes by type
                                 sequence_attr_cols_cat = cat_cols_seq,
                                 sequence_attr_cols_num = num_cols_seq,
                                 sequence_attr_cols_bol = None,
                                 output_file = "../output/data_processed/" + dataname + "_hold_seq2seq_reduced",
                                 return_pairs = True
                                )
else:
    # Reuse the previous stored pairs for the second model
    with open("../output/data_processed/" + dataname + "_train_seq2seq_reduced.pkl", "rb") as f:
        train_pairs  = pickle.load(f)
    
    with open("../output/data_processed/" + dataname + "_hold_seq2seq_reduced.pkl", "rb") as f:
        test_pairs  = pickle.load(f)
        


# In[9]:


# helpdesk dataset
if dataname == "helpdesk":
    length_buckets = {
        "short":  (2, 5),   # 271 traces
        "medium": (5, 7),   # 154 traces
        "long":   (7, None) # 32 traces (7–11 merged)
    }
elif dataname == "sepsis":
    length_buckets = {
    "short":  (3, 8),
    "medium": (8, 18),
    "long":   (18, None)
    }
elif dataname == "BPI13I":
    length_buckets = {
    "short":  (1, 6),    # 1–5
    "medium": (6, 16),   # 6–15
    "long":   (16, None) # 16+
    }
elif dataname == "BPI13C":
    length_buckets = {
    "short":  (1, 4),    # 1–3
    "medium": (4, 8),    # 4–7
    "long":   (8, None)  # 8+
}
elif dataname == "BPI20":
    length_buckets = {
    "short":  (1, 8),    # 1–7
    "medium": (8, 12),   # 8–11
    "long":   (12, None) # 12+
    }
elif dataname == "BPI12W":
    length_buckets = {
    "short":  (2, 8),     # 2–7
    "medium": (8, 20),    # 8–19
    "long":   (20, 46),   # 20–45
    "xlong":  (46, None)  # 46+
}
elif dataname == "BPI12":
    length_buckets = {
        "short":  (3, 8),     # 3–7
        "medium": (8, 24),    # 8–23
        "long":   (24, 61),   # 24–60
        "xlong":  (61, None)  # 61+
    }
elif dataname == "BPI17":
    length_buckets = {
    "short":  (10, 21),    # 10–20
    "medium": (21, 41),    # 21–40
    "long":   (41, 61),    # 41–60
    "xlong":  (61, None)   # 61+
}
elif dataname == "BPI20R":
    length_buckets = {
    "short":  (1, 5),
    "medium": (5, 7),
    "long":   (7, None)
}

# Generate Traces

# In[ ]:


all_generated = []

client = Client()


# In[ ]:


if model == "llama3.1:8b-instruct-q5_K_M" and context_size == 32768:
    model_tag = "llama_32"
elif model == "llama3.1:8b-instruct-q5_K_M" and context_size == 4096:
    model_tag = "llama"    
elif model == "mistral:7b-instruct-q5_K_M" and context_size == 4096:
    model_tag = "mistral"

checkpoint_path = f"../output/runs/{dataname}_{model_tag}_checkpoint.json"
parsed_csv_path = f"../output/runs/{dataname}_{model_tag}_parsed_rows.csv"
fewshots_dir = Path(f"../output/runs/{dataname}_{model_tag}_fewshots")
raw_jsonl_path = f"../output/runs/{dataname}_{model_tag}_raw_generations.jsonl"

# =========================================================
# SETTINGS
# =========================================================
save_every_n = 10


# =========================================================
# CHECKPOINT
# =========================================================
checkpoint = load_checkpoint(checkpoint_path)

if checkpoint is None:
    checkpoint = {
        "bucket_idx": 0,
        "case_idx": 0
    }

bucket_items = list(length_buckets.items())

# =========================================================
# MAIN LOOP
# =========================================================
for bucket_idx, (bucket_name, (min_len, max_len)) in enumerate(bucket_items):

    if bucket_idx < checkpoint["bucket_idx"]:
        continue

    print(f"\n=== Bucket: {bucket_name} ({min_len}, {max_len}) ===")

    bucket_size = get_bucket_size(train_pairs, min_len, max_len)
    k = compute_bucket_k(bucket_size, min_k=3, max_k=10, scale=0.4)
    print(f"{bucket_name}: bucket_size={bucket_size}, k={k}")

    # exact same few-shots on resume
    fewshot_path = fewshots_dir / f"{bucket_name}.pkl"
    few_shots = load_pickle(fewshot_path)

    if few_shots is None:
        few_shots = sample_few_shots(
            train_pairs,
            k,
            min_len=min_len,
            max_len=max_len,
            max_expand=5
        )
        save_pickle(few_shots, fewshot_path)

    if len(few_shots) == 0:
        print("[SKIP] No few-shot examples available.")
        checkpoint["bucket_idx"] = bucket_idx + 1
        checkpoint["case_idx"] = 0
        save_checkpoint(checkpoint, checkpoint_path)
        continue

    selected_holdouts = filter_cases_by_length(
        test_pairs,
        min_len=min_len,
        max_len=max_len
    )

    if len(selected_holdouts) == 0:
        print("[SKIP] No holdout cases in this bucket.")
        checkpoint["bucket_idx"] = bucket_idx + 1
        checkpoint["case_idx"] = 0
        save_checkpoint(checkpoint, checkpoint_path)
        continue

    selected_holdouts_inputs = [reduce_input_block(p) for p in selected_holdouts]

    start_case_idx = checkpoint["case_idx"] if bucket_idx == checkpoint["bucket_idx"] else 0

    all_generated = []

    for case_idx in range(start_case_idx, len(selected_holdouts_inputs)):
        new_input = selected_holdouts_inputs[case_idx]

        print(f"[RUN] bucket={bucket_name}, case_idx={case_idx}")

        result = generate_traces_for_batch([new_input], few_shots, model, client, context_size=context_size)[0]
        result["length_bucket"] = bucket_name
        result["bucket_case_idx"] = case_idx
        
        all_generated.append(result)

        if len(all_generated) >= save_every_n:
            
            with open(raw_jsonl_path, "a", encoding="utf-8") as f:
               for item in all_generated:
                   f.write(json.dumps(item, ensure_ascii=False) + "\n")            
            gen_df = llm_results_to_eventlog(
                all_generated,
                case_id_key=case_index,
                activity_key=core_event,
                time_key=time_col,
                attr_keys=attr_cols
            )


            if len(gen_df) > 0:
                append_df_to_csv(gen_df, parsed_csv_path)

            all_generated = []

            checkpoint["bucket_idx"] = bucket_idx
            checkpoint["case_idx"] = case_idx + 1
            save_checkpoint(checkpoint, checkpoint_path)

    # flush remaining cases in this bucket
    if len(all_generated) > 0:

        with open(raw_jsonl_path, "a", encoding="utf-8") as f:
            for item in all_generated:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        gen_df = llm_results_to_eventlog(
            all_generated,
            case_id_key=case_index,
            activity_key=core_event,
            time_key=time_col,
            attr_keys=attr_cols
        )

        if len(gen_df) > 0:
            append_df_to_csv(gen_df, parsed_csv_path)

        all_generated = []

    checkpoint["bucket_idx"] = bucket_idx + 1
    checkpoint["case_idx"] = 0
    save_checkpoint(checkpoint, checkpoint_path)

