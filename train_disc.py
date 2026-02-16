from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score
from grammar import MXeneGrammar
from datasets import feature_vector_16, DiscDataset, pad_collate
from models.discriminator import TransformerDisc
from models.generator import GeneratorModel, sample_with_grammar
from utils import ensure_dir, set_seed, pick_device
from validator import validate_rule_sequence
from tqdm import tqdm
import json


def build_on_policy_set(config: Dict[str,Any], grammar: MXeneGrammar, vocab_size: int, device: torch.device) -> Tuple[List[List[int]], List[int], List[List[float]]]:
    # Load generator ckpt
    gen_ckpt = config.get("gen_ckpt")
    if not (gen_ckpt and os.path.exists(gen_ckpt)):
        raise FileNotFoundError(f"gen_ckpt not found: {gen_ckpt}")
    gen = GeneratorModel(grammar, config.get("gen_model_cfg", {}))
    ckpt = torch.load(gen_ckpt, map_location="cpu")
    gen.load_state_dict(ckpt["model"], strict=False)
    gen.to(device)

    # Sample sequences
    sp = config.get("on_policy", {})
    seqs = sample_with_grammar(
        gen, grammar,
        max_len=int(sp.get("max_len",64)),
        temperature=float(sp.get("temperature",1.2)),
        top_p=float(sp.get("top_p",0.95)),
        device=device,
        batch_size=int(sp.get("num_samples",800))
    )
    # Label via validator (+ known unstable override when available provided downstream)
    unstable_compositions = set()
    if os.path.exists(config.get("unstable_csv","")):
        from data import load_unstable
        unstable_compositions = set(load_unstable(config["unstable_csv"]))
    labels = []
    feats = []
    for seq in seqs:
        is_valid, reason = validate_rule_sequence(grammar, seq)
        
        # Check composition string
        comp_str = grammar.sequence_to_tokens(seq)
        if comp_str in unstable_compositions:
            is_valid = False
        labels.append(1 if is_valid else 0)
        feats.append(feature_vector_16(grammar, seq))
    return seqs, labels, feats

def load_discriminator_for_shaping(
    ckpt_path: str,
    vocab_size: int,
    pad_id: int,
    disc_model_cfg: Dict[str, Any],
    device: torch.device,
):
    disc = TransformerDisc(
        vocab_size=vocab_size,
        pad_id=pad_id,
        **disc_model_cfg,
    ).to(device)

    if not ckpt_path or not os.path.exists(ckpt_path):
        print("[train-gen] no discriminator ckpt provided; shaping disabled")
        disc.eval()
        return disc

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)

    def _pop_if_mismatch(k: str):
        t = sd.get(k)
        if t is not None and isinstance(t, torch.Tensor) and t.shape[0] != vocab_size:
            print(f"[train-gen] drop {k}: ckpt {tuple(t.shape)} != current ({vocab_size}, ...)")
            sd.pop(k, None)

    _pop_if_mismatch("tok.weight")
    _pop_if_mismatch("tok_bias")
    _pop_if_mismatch("out.weight")
    _pop_if_mismatch("out.bias")

    missing, unexpected = disc.load_state_dict(sd, strict=False)
    if missing:
        print(f"[train-gen] disc missing keys: {sorted(missing)[:6]}{' ...' if len(missing)>6 else ''}")
    if unexpected:
        print(f"[train-gen] disc unexpected keys: {sorted(unexpected)[:6]}{' ...' if len(unexpected)>6 else ''}")
    disc.eval()
    return disc


def train_discriminator(config: Dict[str, Any]) -> None:
    run_dir = config["run_dir"]; ensure_dir(run_dir)
    set_seed(config.get("seed", 123))
    device = pick_device(config.get("device","auto"))
    grammar = MXeneGrammar.create_default()
    V = len(grammar.prods) + 3
    seqs, labels, feats = build_on_policy_set(config, grammar, V, device)
    # Optional HITL labeled data: append to the pool
    for path in config.get("extra_disc_jsonls", []):
        if not (path and os.path.exists(path)): continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                r = ex["rules"]
                y = int(ex["label"])
                
                feats.append(feature_vector_16(grammar, r))
                seqs.append(r)
                labels.append(y)
    # Build dataset
    ds = DiscDataset(seqs, labels, feats)
    val_frac = float(config.get("val_frac", 0.1))
    val_size = 0 if len(ds) <= 1 else max(1, int(len(ds) * val_frac))
    if val_size >= len(ds):
        val_size = len(ds) - 1
    if val_size > 0:
        train_ds, val_ds = random_split(
            ds,
            [len(ds) - val_size, val_size],
            generator=torch.Generator().manual_seed(config.get("seed", 123)),
        )
    else:
        train_ds, val_ds = ds, None

    train_dl = DataLoader(train_ds, batch_size=config.get("batch_size",32), shuffle=True, collate_fn=lambda b: pad_collate(b, grammar.pad_id))
    val_dl = DataLoader(val_ds, batch_size=config.get("batch_size",32), shuffle=False, collate_fn=lambda b: pad_collate(b, grammar.pad_id)) if val_ds else None
    disc = TransformerDisc(vocab_size=V, **config.get("model_cfg", {}), pad_id=grammar.pad_id).to(device)
    resume_path = config.get("resume_ckpt","")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        sd = ckpt.get("model", ckpt)
        ckpt_vocab = ckpt.get("vocab_size")
        if ckpt_vocab is not None and ckpt_vocab != V:
            raise ValueError(f"[disc] resume_ckpt vocab_size {ckpt_vocab} does not match current {V}")
        model_sd = disc.state_dict()
        shape_mismatch = [k for k, t in sd.items() if k in model_sd and isinstance(t, torch.Tensor) and model_sd[k].shape != t.shape]
        if shape_mismatch:
            raise ValueError(f"[disc] resume_ckpt has shape mismatch for keys: {shape_mismatch[:6]}{' ...' if len(shape_mismatch)>6 else ''}")
        missing_keys = [k for k in model_sd.keys() if k not in sd]
        unexpected_keys = [k for k in sd.keys() if k not in model_sd]
        if missing_keys or unexpected_keys:
            raise ValueError(f"[disc] resume_ckpt key mismatch; missing={missing_keys[:6]}{' ...' if len(missing_keys)>6 else ''}, unexpected={unexpected_keys[:6]}{' ...' if len(unexpected_keys)>6 else ''}")
        disc.load_state_dict(sd, strict=True)
    opt = torch.optim.AdamW(disc.parameters(), lr=config.get("lr", 1e-3), betas=tuple(config.get("betas",[0.9,0.999])))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.get("pos_weight",1.0), device=device))
    for epoch in tqdm(range(1, config.get("epochs",3)+1), desc="[disc] epochs"):
        disc.train()
        for X, y, F16 in tqdm(train_dl, desc=f"[disc] epoch {epoch}", total=len(train_dl)):
            X = X.to(device); y = y.to(device); F16 = F16.to(device)
            logit = disc(X, F16)
            loss = bce(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt.step()
        # Eval
        if val_dl:
            disc.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for X, y, F16 in tqdm(val_dl, desc=f"[disc] eval {epoch}", total=len(val_dl)):
                    X = X.to(device); y = y.to(device); F16 = F16.to(device)
                    logit = disc(X, F16)
                    all_logits.append(logit.cpu()); all_y.append(y.cpu())
            logits = torch.cat(all_logits).numpy()
            ys = torch.cat(all_y).numpy()
            acc = float(((logits>=0).astype(int) == ys.astype(int)).mean())
            try:
                auc = float(roc_auc_score(ys, 1.0/(1.0+np.exp(-logits))))
            except Exception:
                auc = 0.5
            print(f"[disc] epoch {epoch} val_acc={acc:.3f} val_auc={auc:.3f}")
        if epoch % int(config.get("save_every",1)) == 0:
            torch.save({"model": disc.state_dict(), "cfg": config, "vocab_size": V}, os.path.join(run_dir, "discriminator.pt"))
    torch.save({"model": disc.state_dict(), "cfg": config, "vocab_size": V}, os.path.join(run_dir, "discriminator.pt"))
