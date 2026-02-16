from __future__ import annotations

import os
import csv
import json
from typing import Optional, Tuple, Dict, Any, List

import torch

from grammar import MXeneGrammar
from models.generator import GeneratorModel, sample_with_grammar
from validator import validate_rule_sequence
from datasets import feature_vector_16
from models.discriminator import TransformerDisc
from utils import set_seed


# ---------------------------
# Helpers
# ---------------------------

def decode_sides_and_layers(grammar: MXeneGrammar, pids: List[int]) -> Tuple[Dict[str, Any], str]:
    """Turn a token-id sequence into side fields + a readable layer string."""
    outer_L, outer_R, inner_L, inner_R, inner, x, stoich = grammar.extract_fields(pids)
    sides: Dict[str, Any] = {
        "stoich":  stoich,
        "x":       x,
        "outer_L": outer_L,
        "outer_R": outer_R,
        "inner":   inner,     # MX2
        "inner_L": inner_L,   # MX3
        "inner_R": inner_R,   # MX3
    }
    layers = grammar.sequence_to_tokens(pids)

    return sides, layers


def _truthy(s: str) -> bool:
    return (s or "").strip().lower() in {"1", "y", "yes", "true", "t"}


def _load_generator_from_ckpt(gen_ckpt: str, grammar: MXeneGrammar, device: str) -> GeneratorModel:
    """Instantiate GeneratorModel and load weights from checkpoint (robust to simple ckpt formats)."""
    if not (gen_ckpt and os.path.exists(gen_ckpt)):
        raise FileNotFoundError(f"gen_ckpt not found: {gen_ckpt}")
    ckpt = torch.load(gen_ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model_cfg", {})
    model = GeneratorModel(grammar, model_cfg)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _load_discriminator_from_ckpt(disc_ckpt: str, grammar: MXeneGrammar, device: str) -> TransformerDisc:
    """Instantiate TransformerDisc and load weights from checkpoint."""
    if not (disc_ckpt and os.path.exists(disc_ckpt)):
        raise FileNotFoundError(f"disc_ckpt not found: {disc_ckpt}")
    ckpt = torch.load(disc_ckpt, map_location="cpu")
    dcfg = ckpt.get("cfg", {}).get("model_cfg", {})
    vocab_size = int(ckpt.get("vocab_size", getattr(grammar, "vocab_size", 512)))
    disc = TransformerDisc(
        vocab_size=vocab_size,
        pad_id=grammar.pad_id,
        **dcfg,
    )
    state = ckpt.get("model", ckpt)
    disc.load_state_dict(state, strict=False)
    disc.to(device)
    disc.eval()
    return disc


# ---------------------------
# Simple HITL sampler
# ---------------------------

@torch.no_grad()
def sample_simple_for_hitl(
    gen_ckpt: str,
    out_csv: str,
    num_samples: int = 5000,
    top_p: float = 0.9,
    temperature: float = 1.0,
    max_len: int = 16,
    seed: int = 1337,
    device: Optional[str] = None,
    batch_size: int = 32,
    disc_ckpt: Optional[str] = None,
    disc_threshold: float = 0.5,   # threshold for disc_valid = 1{disc_prob >= threshold}
) -> str:

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    grammar = MXeneGrammar.create_default() if hasattr(MXeneGrammar, "create_default") else MXeneGrammar()
    gen = _load_generator_from_ckpt(gen_ckpt, grammar, device)
    disc = _load_discriminator_from_ckpt(disc_ckpt, grammar, device) if disc_ckpt else None

    header = [
        "outer_L", "inner", "inner_L", "inner_R", "outer_R", "x", "stoich",
        "disc_prob", "disc_valid",  # from discriminator (prob + label via threshold)
        "rules", "layers",
        "human_valid",           # to be filled by reviewers
        "reason",                # human reason (if invalid)
    ]

    written = 0
    seen_compositions = set()  # Track unique compositions
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        while written < num_samples:
            bs = min(batch_size, num_samples - written)
            batch_sequences: List[List[int]] = sample_with_grammar(
                model=gen,
                grammar=grammar,
                max_len=max_len,
                top_p=top_p,
                temperature=temperature,
                device=device,
                batch_size=bs,
            )

            for rules in batch_sequences:
                # sides
                sides, layers = decode_sides_and_layers(grammar, rules)

                # Skip if we've already seen this composition
                if layers in seen_compositions:
                    continue
                seen_compositions.add(layers)

                # discriminator score + label
                dprob: Any = ""
                dvalid: Any = ""
                if disc is not None:
                    X = torch.tensor(rules, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
                    F16 = torch.tensor(
                        feature_vector_16(grammar, rules),
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)  # [1, 16]
                    logit = disc(X, F16)  # [1]
                    p = float(torch.sigmoid(logit).item())
                    dprob = p
                    dvalid = 1 if p >= float(disc_threshold) else 0

                # write row
                w.writerow([
                    sides.get("outer_L", "") or "",
                    sides.get("inner", "") or "",
                    sides.get("inner_L", "") or "",
                    sides.get("inner_R", "") or "",
                    sides.get("outer_R", "") or "",
                    sides.get("x", "") or "",
                    sides.get("stoich", "") or "",
                    dprob, dvalid,
                    " ".join(map(str, rules)),
                    layers,
                    "",  # reviewer fills this
                    "",  # human reason (if invalid)
                ])

                written += 1
                if written >= num_samples:
                    break

    return os.path.abspath(out_csv)


def ingest_simple_csv(in_csv: str, out_dir: str) -> Tuple[str, str]:
    """
    Convert a reviewed CSV into:
      - hitl_gen.jsonl  : positives for generator fine-tuning
      - hitl_disc.jsonl : all rows with labels for discriminator
    Required CSV columns: outer_L, inner, inner_L, inner_R, outer_R, x, stoich, human_valid
    """
    os.makedirs(out_dir, exist_ok=True)
    gen_jsonl = os.path.join(out_dir, "hitl_gen.jsonl")
    disc_jsonl = os.path.join(out_dir, "hitl_disc.jsonl")

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    with open(in_csv, "r", newline="", encoding="utf-8-sig") as f_in, \
         open(gen_jsonl, "w") as f_g, \
         open(disc_jsonl, "w") as f_d:

        reader = csv.DictReader(f_in)
        required = {"outer_L", "inner", "inner_L", "inner_R", "outer_R", "x", "stoich", "human_valid", "rules"}
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        for row in reader:
            is_pos = _truthy(row.get("human_valid", ""))

            rules_str = (row.get("rules") or "").strip()
            if not rules_str:
                # Skip rows with no sequence (nothing to train on)
                continue
            rules = []
            for tok in rules_str.split():
                try:
                    rules.append(int(tok))
                except Exception:
                    # ignore non-integer tokens safely
                    pass

            if is_pos:
                f_g.write(json.dumps({"rules": rules, "weight": 1.0}) + "\n")

            f_d.write(json.dumps({
                "rules":  rules,
                "label": 1 if is_pos else 0,
                "disc_prob": float(row["disc_prob"]) if (row.get("disc_prob") not in (None, "")) else None,
                "disc_valid": int(row["disc_valid"]) if (row.get("disc_valid") not in (None, "")) else None,
                "reason": (row.get("reason") or "").strip(),  # ok if empty
             }) + "\n")
                

    return os.path.abspath(gen_jsonl), os.path.abspath(disc_jsonl)


# --- Tiny CLI (optional direct use) ---
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Simple HITL (sample -> CSV, then ingest -> JSONLs).")
    sub = ap.add_subparsers(dest="cmd")

    sp = sub.add_parser("sample", help="Sample candidates for human review.")
    sp.add_argument("--gen_ckpt", required=True)
    sp.add_argument("--out_csv", required=True)
    sp.add_argument("--num_samples", type=int, default=5000)
    sp.add_argument("--top_p", type=float, default=0.9)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--max_len", type=int, default=16)
    sp.add_argument("--seed", type=int, default=1337)
    sp.add_argument("--device", default=None)
    sp.add_argument("--batch_size", type=int, default=32)
    sp.add_argument("--disc_ckpt", default=None, help="Optional discriminator checkpoint to fill disc_prob/disc_valid")
    sp.add_argument("--disc_threshold", type=float, default=0.5, help="Threshold for disc_valid = 1{prob >= threshold}")

    ip = sub.add_parser("ingest", help="Convert reviewed CSV into JSONLs for training.")
    ip.add_argument("--in_csv", required=True)
    ip.add_argument("--out_dir", required=True)

    args = ap.parse_args()
    if args.cmd == "sample":
        path = sample_simple_for_hitl(
            gen_ckpt=args.gen_ckpt,
            out_csv=args.out_csv,
            num_samples=args.num_samples,
            top_p=args.top_p,
            temperature=args.temperature,
            max_len=args.max_len,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            disc_ckpt=args.disc_ckpt,
            disc_threshold=args.disc_threshold,
        )
        print(f"Wrote review CSV: {path}")
    elif args.cmd == "ingest":
        g, d = ingest_simple_csv(args.in_csv, args.out_dir)
        print(f"Wrote generator JSONL:   {g}")
        print(f"Wrote discriminator JSONL:{d}")
    else:
        ap.print_help()
