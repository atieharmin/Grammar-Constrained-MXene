# Grammar-Constrained Generative Modeling of Layer-Resolved MXene Compositions 

Code for **Grammar-Constrained Generative Modeling of Layer-Resolved MXene Compositions**.
 
This project proposes new MXene compositions, layer by layer. A small GPT-2 model (the *generator*) writes each candidate as a sequence of grammar rules, so every output is a well-formed MXene structure by construction. A second model (the *discriminator*) scores how chemically plausible each candidate is. The generator can then be improved in two ways: by learning from the discriminator's scores (reinforcement learning), or by learning from candidates that a human expert has reviewed (human-in-the-loop).
 
## Background
 
MXenes are two-dimensional carbides and nitrides with the general formula M<sub>n+1</sub>X<sub>n</sub>, where M is an early transition metal and X is carbon or nitrogen. They are built from alternating metal and X layers. This project covers three layer patterns, including ordered double-metal MXenes where a second metal (M′) sits in the inner layers:
 
| Class | Formula | Layer order | Example |
|---|---|---|---|
| MX1 | M<sub>2</sub>X | M · X · M | `Ti C Ti` (Ti<sub>2</sub>C) |
| MX2 | M<sub>2</sub>M′X<sub>2</sub> | M · X · M′ · X · M | `Sc C Nb C Sc` (Sc<sub>2</sub>NbC<sub>2</sub>) |
| MX3 | M<sub>2</sub>M′<sub>2</sub>X<sub>3</sub> | M · X · M′ · X · M′ · X · M | `Ta C Ti C Ti C Ta` (Ta<sub>2</sub>Ti<sub>2</sub>C<sub>3</sub>) |
 
For MX2 and MX3, M′ may be the same metal as M, which gives the single-metal forms M<sub>3</sub>X<sub>2</sub> and M<sub>4</sub>X<sub>3</sub>.
 
**Metals (M, M′):** Cr, Mo, W, Sc, V, Nb, Ta, Ti, Zr, Hf
**X elements:** C, N
 
## How it works
 
1. **Grammar.** A context-free grammar (`grammar.py`) spells out every allowed layer pattern. A candidate MXene is written as the list of grammar rules used to build it, not as free text.
2. **Generator.** A GPT-2 model learns to produce these rule lists. At every step, choices the grammar does not allow are blocked, so the output is always structurally well-formed.
3. **Chemistry-based training penalties.** Two optional extra loss terms nudge the generator toward chemically sensible choices:
   - *Occupancy ordering* — discourages placing a less-preferred metal on the outside and a more-preferred one on the inside.
   - *Group penalty* — discourages pairing two different metals from the same periodic-table group (for example Ti with Zr).
4. **Discriminator.** A small Transformer classifier, combined with 15 hand-built chemistry features, learns to separate plausible from implausible candidates. Its training set comes from the generator's own samples, labeled by the rule checker (`validator.py`) and by a list of compositions known to be unstable.
5. **Refinement.** The pretrained generator is fine-tuned either
   - with **reinforcement learning** (REINFORCE), using the discriminator's score as the reward, or
   - with **human-in-the-loop (HITL)** feedback, using candidates an expert has marked as valid.

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
│   └── hitl/              # Human labeled data
├── runs/                  # Training checkpoints and evaluation outputs
└── requirements.txt
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch ≥ 2.0, Transformers ≥ 4.30, NumPy ≥ 1.24, pandas ≥ 2.0, scikit-learn ≥ 1.3, and tqdm ≥ 4.65. The provided configs use `"device": "cuda"`; set it to `"auto"` or `"cpu"` to run without a GPU.


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

## Experiments
 
| Config | What changes |
|---|---|
| `gen_ablate_A1` | Full model: both chemistry penalties on (weight 0.5 each) |
| `gen_ablate_A2_occ` | Occupancy-ordering penalty off |
| `gen_ablate_A3_group` | Group penalty off |
| `gen_ablate_A4_all` | Both penalties off (plain grammar-constrained baseline) |
| `gen_ablate_A5_with_rl` | A1 fine-tuned with reinforcement learning (reward weight 0.1) |
| `gen_hitl_round1` | A1 fine-tuned on human-reviewed candidates |
| `eval_ablate_A0`–`A6` | Matching evaluation configs (1,000 samples each) |
 
### Included results
 
The evaluation reports in this repository (1,000 samples, temperature 1.0, top-p 0.9):
 
| Generator | Validity rate | Unique valid MXenes | Discriminator AUC | Discriminator accuracy |
|---|---|---|---|---|
| Pretrained (A1) | 96.0% | 165 | 0.967 | 0.816 |
| + Reinforcement learning (A5) | 97.7% | 153 | 0.962 | 0.929 |
| + Human-in-the-loop (A6) | 96.2% | 147 | 0.927 | 0.914 |
 
In all three runs, most remaining failures come from the `preference` check. The full lists of unique compositions are in `unique_mxenes.jsonl`, `unique_mxenes-rl.jsonl`, and `unique_mxenes-hitl.jsonl`.
 
## Citation
 
If you use this code, please cite:
 
```
Atieh Armin, Mohammad Mozafari, Daniel Schwartz, Ali Shokoufandeh, Masoud Soroush.
Grammar-Constrained Generative Modeling of Layer-Resolved MXene Compositions. In revision.
```