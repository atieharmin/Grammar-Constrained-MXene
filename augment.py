from __future__ import annotations
from typing import Dict, Any, List
import os, json
from grammar import MXeneGrammar
from data import infer_rows, rows_to_examples, enumerate_valid, write_jsonl, load_stable_rows
from utils import ensure_dir

def augment_pipeline(data_csv: str, out_dir: str, alpha: float=0.6, beta: float=0.3, gamma: float=0.1) -> None:
    ensure_dir(out_dir)
    grammar = MXeneGrammar.create_default()
    # 1) load synthesized data
    rows = infer_rows(data_csv) if os.path.exists(data_csv) else []
    
    # Load stable data from unstable.csv
    unstable_path = "unstable.csv"
    # Check if unstable.csv is in same dir as data_csv
    if data_csv and os.path.dirname(data_csv):
        candidate = os.path.join(os.path.dirname(data_csv), "unstable.csv")
        if os.path.exists(candidate):
            unstable_path = candidate
            
    if os.path.exists(unstable_path):
        stable_rows = load_stable_rows(unstable_path)
        print(f"[augment] loaded {len(stable_rows)} stable rows from {unstable_path}")
        rows.extend(stable_rows)

    examples = rows_to_examples(grammar, rows)
    # 2) enumerate additional valid structures
    enum = enumerate_valid(grammar, alpha=alpha, beta=beta, gamma=gamma)
    enum_examples = rows_to_examples(grammar, enum)
    # merge
    all_examples = examples + enum_examples
    # normalize weights (if missing, they were set to default in rows_to_examples)
    # 3) write outputs
    jsonl_path = os.path.join(out_dir, "dataset.jsonl")
    write_jsonl(all_examples, jsonl_path)
    # 5) save vocabulary
    vocab_pkl = os.path.join(out_dir, "rule_vocab.pkl")
    grammar.build_vocab().save(vocab_pkl)

    print(f"[augment] wrote {jsonl_path} and {vocab_pkl}")
