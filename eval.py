from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, json, uuid, datetime, collections, math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from grammar import MXeneGrammar
from models.generator import GeneratorModel, sample_with_grammar
from models.discriminator import TransformerDisc
from validator import validate_rule_sequence
from utils import ensure_dir, sha256_file
from tqdm import tqdm
import csv
from datasets import feature_vector_16

# --- Helpers ---

def token_entropy(samples: List[List[int]], V: int) -> float:
    cnt = np.zeros((V,), dtype=np.float64)
    for s in samples:
        for t in s:
            if 0 <= t < V:
                cnt[t] += 1
    p = cnt / max(1, cnt.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def ngram_set(seq: List[int], n: int) -> set[tuple[int, ...]]:
    return set(tuple(seq[i:i + n]) for i in range(0, max(0, len(seq) - n + 1)))

def decode_sides_and_layers(grammar: MXeneGrammar, seq: List[int]) -> Tuple[Dict[str, Any], str]:
    outer_L, outer_R, inner_L, inner_R, inner, x, stoich = grammar.extract_fields(seq)
    sides = {
        "stoich":  stoich,
        "x":       x,
        "outer_L": outer_L,
        "outer_R": outer_R,
        "inner":   inner,
        "inner_L": inner_L,
        "inner_R": inner_R,
    }
    layers = grammar.sequence_to_tokens(seq)
    return sides, layers


# --- Main ---

def evaluate(config: Dict[str, Any], gen_cfg: Dict[str, Any], disc_cfg: Dict[str, Any]) -> None:
    run_dir = config["run_dir"]; ensure_dir(run_dir)
    grammar = MXeneGrammar.create_default()
    V = len(grammar.prods) + 3

    # Load generator
    gen = GeneratorModel(grammar, gen_cfg.get("model_cfg", {}))
    ckpt = torch.load(config["gen_ckpt"], map_location="cpu")
    gen.load_state_dict(ckpt["model"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen.to(device).eval()

    # Load discriminator
    disc = TransformerDisc(vocab_size=V, pad_id=grammar.pad_id, **disc_cfg.get("model_cfg", {}))
    dckpt = torch.load(disc_cfg["disc_ckpt"], map_location="cpu")
    disc.load_state_dict(dckpt["model"], strict=False)
    disc.to(device).eval()

    # Sampling
    samp_cfg = config.get("sampling", {"temperature": 1.0, "top_p": 0.9, "max_len": 64})
    N = int(config.get("N", 1000))
    samples: List[List[int]] = []
    pbar = tqdm(total=N, desc="[eval] sampling")
    while len(samples) < N:
        bs = min(64, N - len(samples))
        seqs = sample_with_grammar(
            gen, grammar,
            max_len=samp_cfg.get("max_len", 64),
            temperature=samp_cfg.get("temperature", 1.0),
            top_p=samp_cfg.get("top_p", 0.9),
            device=device,
            batch_size=bs
        )
        samples.extend(seqs)
        pbar.update(len(seqs))
    pbar.close()

    # Per-sample validation + decode actual sides
    reasons_counter = collections.Counter()
    val_flags: List[bool] = []
    per_reason: List[str | None] = []
    sides_list: List[Dict[str, str | None]] = []
    unique_layer_strings = set()
    unique_mxenes: List[Dict[str, Any]] = []
    
    for s in tqdm(samples, desc="[eval] validating"):
        ok, reason = validate_rule_sequence(grammar, s)
        val_flags.append(ok)
        per_reason.append(None if ok else reason)
        if not ok and reason:
            reasons_counter[reason] += 1
        sides,layers = decode_sides_and_layers(grammar, s)
        sides_list.append(sides)
        
        # Collect unique valid MXenes
        if ok and layers and layers not in unique_layer_strings:
            unique_layer_strings.add(layers)
            unique_mxenes.append({
                "layer_string": layers,
                "sides": sides,
                "rules": s,
            })

    validity_rate = float(np.mean([1.0 if v else 0.0 for v in val_flags]))

    # # Diversity
    # formulas = []
    # Ms = []; Mis = []; Xs = []; Sto = []
    # for sides in tqdm(sides_list, desc="[eval] diversity fields"):
    #     outL, outR, inner, x, stoich = sides["outer_L"], sides["outer_R"], sides["inner"], sides["x"], sides["stoich"]
    #     Ms.append(outL)  # store left outer as representative M
    #     Mis.append(inner if inner else "None")
    #     Xs.append(x)
    #     Sto.append(stoich)
    #     formulas.append(f"{outL}-{inner or 'None'}-{x}-{stoich}")
    # unique_formulas = len(set(formulas))
    # diversity = {
    #     "unique_formulas": unique_formulas,
    #     "M_counts": dict(collections.Counter(Ms)),
    #     "Mi_counts": dict(collections.Counter(Mis)),
    #     "X_counts": dict(collections.Counter(Xs)),
    #     "stoich_counts": dict(collections.Counter(Sto)),
    #     "token_entropy": token_entropy(samples, V)
    # }

    # Novelty vs training set
    train_jsonl = config.get("data_jsonl")
    train_seqs = []
    if train_jsonl and os.path.exists(train_jsonl):
        with open(train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                train_seqs.append(json.loads(line)["rules"])
    train_ngrams = set()
    for s in train_seqs:
        train_ngrams |= ngram_set(s, 3)
        train_ngrams |= ngram_set(s, 4)
    nov_scores = []
    for s in tqdm(samples, desc="[eval] novelty"):
        sset = ngram_set(s, 3) | ngram_set(s, 4)
        inter = len(train_ngrams & sset)
        union = len(train_ngrams | sset) if len(train_ngrams | sset) > 0 else 1
        nov_scores.append(1.0 - float(inter) / float(union))
    novelty = {"avg_jaccard_complement": float(np.mean(nov_scores))}

    # Discriminator scores vs validator (on these samples)
    X = [torch.tensor(s, dtype=torch.long) for s in samples]
    maxL = max(len(t) for t in X)
    Xpad = torch.full((len(X), maxL), grammar.pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(X):
        Xpad[i, :len(t)] = t.to(device)
    F16_list = [feature_vector_16(grammar, s) for s in samples]
    F16 = torch.tensor(F16_list, dtype=torch.float, device=device)
    with torch.no_grad():
        dlogits = disc(Xpad, F16).cpu().numpy()
    dprobs = 1.0 / (1.0 + np.exp(-dlogits))
    auc = float(roc_auc_score(np.array(val_flags, dtype=np.float32), dprobs))
    acc = float(((dprobs >= 0.5).astype(int) == (np.array(val_flags).astype(int))).mean())

    # --- Write artifacts (JSONL/CSV + layer strings) ---
    jsonl_path = os.path.join(run_dir, "samples.jsonl")
    csv_path = os.path.join(run_dir, "samples.csv")
    layers_all_path = os.path.join(run_dir, "samples_layers_all.txt")
    layers_valid_path = os.path.join(run_dir, "samples_layers_valid.txt")
    unique_mxenes_path = os.path.join(run_dir, "unique_mxenes.txt")
    unique_mxenes_jsonl_path = os.path.join(run_dir, "unique_mxenes.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as fj, \
         open(csv_path, "w", newline="", encoding="utf-8") as fc, \
         open(layers_all_path, "w", encoding="utf-8") as fa, \
         open(layers_valid_path, "w", encoding="utf-8") as fv:

        w = csv.writer(fc)
        # NOTE: now exposing outer_L and outer_R explicitly
        w.writerow(["outer_L", "inner", "outer_R", "x", "stoich", "valid", "reason", "disc_prob", "rules", "layers"])

        for i, rids in enumerate(samples):
            sides = sides_list[i]
            valid = bool(val_flags[i])
            reason = per_reason[i] or ""
            layers = grammar.sequence_to_tokens(rids)
            dprob = float(dprobs[i])

            # JSONL row
            fj.write(json.dumps({
                "outer_L": sides["outer_L"],
                "inner": sides["inner"],
                "outer_R": sides["outer_R"],
                "x": sides["x"],
                "stoich": sides["stoich"],
                "valid": valid,
                "reason": (None if valid else reason),
                "disc_prob": dprob,
                "rules": rids,
                "layers": layers
            }) + "\n")

            # CSV row
            w.writerow([
                sides["outer_L"], sides["inner"] or "", sides["outer_R"],
                sides["x"], sides["stoich"], int(valid), reason, dprob,
                " ".join(map(str, rids)), layers
            ])

            # TXT outputs
            if layers:
                fa.write(layers + "\n")
                if valid:
                    fv.write(layers + "\n")

    # Write unique MXenes files
    with open(unique_mxenes_path, "w", encoding="utf-8") as fu, \
         open(unique_mxenes_jsonl_path, "w", encoding="utf-8") as fuj:
        
        for mxene in unique_mxenes:
            # Write layer string to text file
            fu.write(mxene["layer_string"] + "\n")
            
            # Write detailed info to JSONL file
            fuj.write(json.dumps({
                "layer_string": mxene["layer_string"],
                "outer_L": mxene["sides"]["outer_L"],
                "inner": mxene["sides"]["inner"],
                "outer_R": mxene["sides"]["outer_R"],
                "x": mxene["sides"]["x"],
                "stoich": mxene["sides"]["stoich"],
                "inner_L": mxene["sides"]["inner_L"],
                "inner_R": mxene["sides"]["inner_R"],
                "rules": mxene["rules"]
            }) + "\n")

    # Final report
    report = {
        "run_id": str(uuid.uuid4()),
        "date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "dataset_hash": sha256_file(config.get("data_jsonl")) if os.path.exists(config.get("data_jsonl", "")) else "",
        "generator": {
            "ckpt": config["gen_ckpt"],
            "validity_rate": validity_rate,
            "novelty": novelty,
            "invalid_reasons": dict(reasons_counter),
            "unique_mxenes_count": len(unique_mxenes)
        },
        "discriminator": {
            "ckpt": config.get("disc_ckpt", ""),
            "auc": auc,
            "accuracy": acc
        },
        "sampling": {"N": N, **samp_cfg},
        "artifacts": {
            "samples_jsonl": jsonl_path,
            "samples_csv": csv_path,
            "layers_all_txt": layers_all_path,
            "layers_valid_txt": layers_valid_path,
            "unique_mxenes_txt": unique_mxenes_path,
            "unique_mxenes_jsonl": unique_mxenes_jsonl_path
        }
    }
    out_path = os.path.join(run_dir, "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] Saved report to {out_path}")
    print(f"[eval] Saved samples to {jsonl_path} and {csv_path}")
    print(f"[eval] Saved layer strings to {layers_all_path} (all) and {layers_valid_path} (valid only)")
    print(f"[eval] Found {len(unique_mxenes)} unique MXenes out of {sum(val_flags)} valid samples")
    print(f"[eval] Saved unique MXenes to {unique_mxenes_path} and {unique_mxenes_jsonl_path}")
