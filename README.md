# MXene-GAN

A grammar-constrained generative adversarial framework for designing novel MXene compositions. The system uses a GPT-2-based generator guided by a context-free grammar to produce syntactically valid MXene structures, paired with a Transformer discriminator trained via on-policy sampling and optional human-in-the-loop feedback.

## Overview

MXenes are a family of 2D transition metal carbides and nitrides with the general formula M<sub>n+1</sub>X<sub>n</sub>, where M is an early transition metal and X is carbon or nitrogen. This project generates candidate MXene compositions by:

1. Defining a **context-free grammar** that encodes the structural rules of MXene stoichiometries (MX1, MX2, MX3)
2. Training a **generator** (GPT-2) to produce sequences of production rules, with grammar masking to guarantee syntactic validity
3. Training a **discriminator** (Transformer encoder) to distinguish chemically plausible compositions from implausible ones
4. Incorporating **chemical constraints** as auxiliary losses (occupancy ordering, periodic-group penalties)
5. Supporting a **human-in-the-loop (HITL)** workflow for iterative refinement with expert feedback

### Supported Stoichiometries

| Stoichiometry | Layer Structure | Description |
|---|---|---|
| **MX1** | M X M | Single metal on both outer layers |
| **MX2** | M X M' X M | Two outer layers + one distinct inner metal |
| **MX3** | M X M' M' X M | Two outer layers + paired inner metals |

### Supported Elements

- **Metals (M):** Cr, Mo, W, Sc, V, Nb, Ta, Ti, Zr, Hf
- **X elements:** C, N

## Project Structure

```
MXene/
├── cli.py                 # Unified CLI entry point
├── grammar.py             # Context-free grammar for MXene structures
├── validator.py           # Chemical & structural validation rules
├── augment.py             # Data augmentation pipeline
├── data.py                # CSV/formula parsing and dataset construction
├── datasets.py            # PyTorch datasets, collation, feature extraction
├── train_gen.py           # Generator training loop
├── train_disc.py          # Discriminator training loop
├── eval.py                # Evaluation and metrics
├── hitl.py                # Human-in-the-loop sampling and ingestion
├── utils.py               # Shared utilities (ranking, grouping, losses)
├── models/
│   ├── generator.py       # GPT-2 GeneratorModel + grammar-masked sampling
│   └── discriminator.py   # Transformer discriminator (TransformerDisc)
├── configs/
│   └── ablate/            # Example configs for ablation experiments
├── data/
│   ├── raw/               # Raw CSV data (data.csv, unstable.csv)
│   └── aug_v1/            # Augmented training data (dataset.jsonl, rule_vocab.pkl)
├── runs/                  # Training checkpoints and evaluation outputs
└── requirements.txt
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- NumPy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- tqdm >= 4.65.0

## Usage

All commands are run through the unified CLI (`cli.py`).

### Training Pipelines

There are two main training workflows:

#### 1. RL Pipeline (adversarial training with REINFORCE)

1. **Train generator** (pretrain):
   ```bash
   python cli.py train-gen --config configs/ablate/gen_ablate_A1.json
   ```

2. **Train discriminator** (uses generator from step 1):
   ```bash
   python cli.py train-disc --config configs/ablate/disc_ablate.json
   ```
   *(Ensure `gen_ckpt` in `disc_ablate.json` points to the checkpoint from step 1.)*

3. **RL training** (fine-tune generator with discriminator reward):
   ```bash
   python cli.py train-gen --config configs/ablate/gen_ablate_A5_with_rl.json
   ```
   *(Ensure `resume_ckpt` points to the generator from step 1, and `disc_ckpt` points to the discriminator from step 2.)*

#### 2. Human-in-the-Loop (HITL) Pipeline

1. **Train generator** (pretrain):
   ```bash
   python cli.py train-gen --config configs/ablate/gen_ablate_A1.json
   ```

2. **HITL fine-tuning** (use `hitl.sample` and `hitl.ingest` to produce HITL data, then):
   ```bash
   python cli.py train-gen --config configs/ablate/gen_hitl.json
   ```
   *(Ensure `resume_ckpt` and `extra_jsonls` in `gen_hitl.json` point to the pretrained generator and HITL-labeled data.)*

**Prerequisite:** Before either pipeline, build the dataset:
```bash
python cli.py augment --data data/raw/data.csv --out_dir data/aug_v1
```

### Evaluation

Evaluate a trained generator (optionally with a discriminator):

```bash
python cli.py eval \
  --config configs/ablate/eval_ablate_A0.json \
  --gen_config configs/ablate/gen_ablate_A1.json \
  --disc_config configs/ablate/disc_ablate.json
```

The evaluation config specifies `gen_ckpt`, `disc_ckpt`, `data_jsonl`, and `N` (number of samples). Outputs are written to `run_dir`: `eval_report.json` (validity rate, discriminator AUC/accuracy), `samples.jsonl`, `samples.csv`, and `unique_mxenes.txt` / `unique_mxenes.jsonl`.

## Data Format

Training data is stored as JSONL with one example per line:

```json
{
  "rules": [0, 3, 15, 6, 25],
  "text": "<S>-><MX1> <MX1>->STOICH_MX1,<M_outer_L>,<X>,<M_outer_R> <M_outer_L>->Ti <X>->C <M_outer_R>->Ti",
  "outer": "Ti",
  "inner": "Ti",
  "x": "C",
  "stoich": "MX1",
  "weight": 1.0
}
```

| Field | Description |
|---|---|
| `rules` | List of production rule IDs (integers) |
| `text` | Human-readable linearization of the rule sequence |
| `outer` | Outer metal element |
| `inner` | Inner metal element (same as outer for MX1) |
| `x` | X element (C or N) |
| `stoich` | Stoichiometry class (MX1, MX2, or MX3) |
| `weight` | Sample weight for training |

## Grammar

The grammar defines valid MXene compositions via context-free production rules. The start symbol `<S>` derives one of three stoichiometry templates, each expanding into metal and X-element choices:

```
<S>      -> <MX1> | <MX2> | <MX3>
<MX1>    -> STOICH_MX1, <M_outer_L>, <X>, <M_outer_R>
<MX2>    -> STOICH_MX2, <M_outer_L>, <X>, <M_inner>, <X>, <M_outer_R>
<MX3>    -> STOICH_MX3, <M_outer_L>, <X>, <M_inner_L>, <M_inner_R>, <X>, <M_outer_R>
<M_*>    -> Cr | Mo | W | Sc | V | Nb | Ta | Ti | Zr | Hf
<X>      -> C | N
```

During generation, a **grammar mask** restricts the model's output logits at each step to only the production rules that are valid continuations of the current partial derivation, guaranteeing syntactic correctness.

## Validation

The `validator.py` module checks generated sequences against multiple criteria:

- **Syntactic validity** — the rule sequence must be parseable by the grammar
- **Single X element** — all X positions must use the same element (C or N)
- **Symmetry** — outer metals must match (outer_L == outer_R); for MX3, inner metals must also match (inner_L == inner_R)
- **Preference ordering** — outer metal rank should not exceed inner metal rank
- **Group constraint** — metals from the same periodic-table group should not appear together in different positions

## Ablation Experiments

The `configs/ablate/` directory contains configs for systematic ablation studies:

| Config | Description |
|---|---|
| `gen_ablate_A2_occ` | No Occupancy loss |
| `gen_ablate_A3_group` | No Group penalty |
| `gen_ablate_A4_all` | All auxiliary losses disabled (baseline) |
| `gen_ablate_A5_with_rl` | RL shaping enabled |
| `gen_hitl` | Fine-tuning with HITL data |
| `eval_ablate_A0`–`A6` | Corresponding evaluation configs |
