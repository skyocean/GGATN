# Graph Grounded Cross Attention Transformer Neural Network for Full Sequence Generation in Predictive Business Process Monitoring

***Graph grounded and interpretable sequence generation framework*** for full event sequence generation in predictive business process monitoring (PBPM). This repository implements GGATN, a Graph Grounded Cross Attention Transformer Neural Network that combines global process graph learning, Transformer based sequence contextualization, graph grounded cross attention, activity feedback refinement, and graph constrained structured decoding.

**Authors**: Fang Wang (Florence Wong), Ernesto Damiani  

**Repository**: Code and demonstrations for the associated research article.

---

## 📖 Overview

This repository implements **GGATN**, a graph grounded cross attention Transformer nerual network framework for full sequence generation in predictive business process monitoring (PBPM).

GGATN generates complete event sequences, including activities, timestamps, attributes, sequence length, and explicit termination. The model combines a global process graph encoder, a sequence Transformer encoder, graph grounded cross attention, activity feedback refinement, and graph constrained Viterbi style decoding.

The codebase includes training, generation, evaluation, local LLM baselines, ablation notebooks, and visualization utilities for interpreting how graph structure and sequence context affect generated event sequences.

### ✨ Key Features

- **Full sequence generation** for activities, timestamps, attributes, length, and termination
- **Global process graph encoder** based on global activity transitions and temporal gap information
- **Sequence Transformer encoder** for position aware contextual modeling per sequence
- **Graph grounded cross attention** between sequence positions and graph learned activity embeddings
- **Graph grounded refinement and decoding**, combining cosine prototype decoding, activity feedback refinement, and graph constrained Viterbi style sequence recovery
- **Evaluation utilities** for sequence similarity, control flow similarity, duration plausibility, attributes, coverage, and hallucination checks
- **Local LLM baseline pipeline** for prompt based generation and comparison
- **Visualization utilities** for graph attention, self attention, refinement, and decoding diagnostics

---

## ⚙️ Installation and Requirements

This code has been developed for Python based experiments using PyTorch and PyTorch Geometric.

Recommended environment:

- **Python:** 3.11 or later
- **PyTorch**
- **PyTorch Geometric**
- **NumPy**
- **Pandas**
- **scikit learn**
- **Matplotlib**
- **NetworkX**

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🧩 Repository Structure

```text
.
├── data/
│   ├── BPI13C.csv
│   ├── BPI13I.csv
│   ├── BPI20PrepaidTravelCost.csv
│   ├── Sepsis Cases - Event Log.csv
│   └── helpdesk.csv
│
├── src/
│   ├── DataEncoder.py
│   ├── DataProcess.ipynb
│   ├── DblAttTransGAT.py
│   ├── DblAttnsTransGATCall.ipynb
│   ├── DblAttnsTransGATCall_ablt_joint.ipynb
│   ├── DblAttnsTransGATCall_ablt_stage.ipynb
│   ├── DblAttnsTransGATCall_ablt_stage10.ipynb
│   ├── Evaluation.py
│   ├── LLMGen.py
│   ├── LLM_GenEvaCall.ipynb
│   └── Visualization.py
│
└── requirements.txt
```

---

## 🔧 Core Files

| File | Description |
|------|-------------|
| `DblAttTransGAT.py` | Main GGATN model implementation, including sequence Transformer encoder, graph grounded cross attention, activity feedback refinement, graph grounded decoding, and Viterbi style generation |
| `DataEncoder.py` | Data encoding utilities for activities, timestamps, event level attributes, sequence level attributes, categorical variables, numerical variables, and special symbols |
| `Evaluation.py` | Evaluation metrics for generated event sequences, including sequence similarity, Damerau Levenshtein similarity, bigram based control flow similarity, temporal error, and attribute level metrics |
| `LLMGen.py` | Local LLM generation utilities for structured prompt construction, output parsing, and full sequence generation baselines |
| `Visualization.py` | Visualization utilities for model diagnostics, attention analysis, control flow comparison, and interpretability figures |
| `requirements.txt` | Python dependency list |

---

## 📓 Demonstration Notebooks

| Notebook | Description |
|----------|-------------|
| `DataProcess.ipynb` | Event log preprocessing and data preparation |
| `DblAttnsTransGATCall.ipynb` | Main GGATN training, generation, and evaluation workflow |
| `DblAttnsTransGATCall_ablt_joint.ipynb` | Joint ablation workflow for analyzing model components |
| `DblAttnsTransGATCall_ablt_stage.ipynb` | Stage based ablation workflow for studying architectural contributions |
| `DblAttnsTransGATCall_ablt_stage10.ipynb` | Extended stage ablation workflow |
| `LLM_GenEvaCall.ipynb` | Local LLM baseline generation and evaluation workflow |

---

## 📂 Data

The `data/` directory contains benchmark event logs used for full sequence generation experiments.

| Dataset File | Description |
|--------------|-------------|
| `BPI13C.csv` | BPI Challenge 2013 closed problems event log |
| `BPI13I.csv` | BPI Challenge 2013 incidents event log |
| `BPI20PrepaidTravelCost.csv` | BPI Challenge 2020 prepaid travel cost event log |
| `Sepsis Cases - Event Log.csv` | Sepsis patient pathway event log |
| `helpdesk.csv` | Helpdesk event log |

The preprocessing workflow standardizes event logs into the input format expected by GGATN. The model uses activity labels, timestamps, event level attributes, sequence level attributes, and sequence identifiers.

---

## 🧠 Model Summary

GGATN follows a graph sequence generation pipeline:

```text
Global process graph (GAT encoder)
Sequence Transformer encoder
→ Graph grounded cross attention
→ Activity feedback refinement + Graph constrained Viterbi decoding
```
The model first learns graph based activity embeddings from observed transitions and temporal gap information. A Transformer encoder then contextualizes sequence positions, and graph grounded cross attention lets each position retrieve process aware activity representations from the learned graph.

For generation, GGATN produces all position level scores in a single forward pass. Final activity paths are recovered with Viterbi style decoding over the valid activity graph, using neural activity scores, transition regularity, adjacency constraints, and explicit EOS termination.

## 📊 Evaluation

The repository supports evaluation at several levels.

| Evaluation Level | Metrics |
|------------------|---------|
| Activity sequence quality | Sequence Coverage, Sequence similarity, Damerau Levenshtein similarity |
| Control flow realism | Bigram Jensen Shannon distance |
| Temporal plausibility | Duration distribution distance, timestamp error |
| Attribute generation | Activity accuracy Event level categorical accuracy, event level numerical error, sequence level categorical accuracy, sequence level numerical error |
| Generation validity | Coverage, hallucinated activity rate, sequence level inconsistency rate |

---

## 🔍 Interpretability

GGATN includes multi level interpretability analysis for understanding how sequence context and graph structure shape generated outputs.

Supported analyses include:

- Global GAT Graph Attention Comparision across Training Regimes  
- Position Level Dual Stage Attention Analysis
- Graph Grounded Cross Attention Decomposition Analysis
- Refinement Based Activity Distribution Reshaping Analysis
- Structured Decoding and Transition Correction

---

## 🤖 Local LLM Baselines

The repository includes utilities for locally deployed LLM baselines using an **Ollama local server**. Full sequence generation is reformulated as structured text generation with input output prompts. Generated text is parsed back into event log format and evaluated using the same metrics as GGATN.

The baseline models include:

- `llama3.1:8b-instruct-q5_K_M` with a 4096 token context window
- `mistral:7b-instruct-q5_K_M` with a 4096 token context window
- `llama3.1:8b-instruct-q5_K_M` with an extended 32768 token context window

Few shot examples are selected within length based buckets, so each target sequence is prompted using examples with comparable sequence lengths.

The LLM baseline workflow includes:

- prompt construction with bucketed few shot examples
- local generation through Ollama
- structured output parsing
- event log reconstruction
- metric based comparison with GGATN

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess data

Open:

```text
src/DataProcess.ipynb
```

and prepare the selected event log.

### 3. Train and evaluate GGATN

Open:

```text
src/DblAttnsTransGATCall.ipynb
```

to train GGATN, generate full sequences, and compute evaluation metrics.

### 4. Run LLM baselines

Open:

```text
src/LLM_GenEvaCall.ipynb
```

to run local LLM generation and evaluation.

---

## 📜 Citation

If you use this code or model, please cite the associated paper:

```bibtex
@article{wang2026ggatn,
  title={Graph Grounded Cross Attention Transformer Neural Network: A Framework for Full Sequence Generation in Predictive Business Process Monitoring},
  author={Wang, Fang and Damiani, Ernesto},
  journal={Expert Systems with Applications},
  year={2026},
  note={Under review}
}
```

---

## 📌 Notes

This repository is intended for research use. The codebase focuses on full sequence generation in PBPM, graph grounded neural generation, structured decoding, and interpretable comparison with local LLM baselines.
