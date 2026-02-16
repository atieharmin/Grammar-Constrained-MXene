
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import json, math, os, random
import torch
from torch.utils.data import Dataset
from grammar import MXeneGrammar
from utils import GROUP_MAP, PREFERRED_OUTER, PREFERRED_INNER, rank_index_outer, rank_index_inner, rank_norm
from tqdm import tqdm

class InMemoryDataset(Dataset):
    def __init__(self, data: List[Any]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]

class JSONLSequences(Dataset):
    def __init__(self, jsonl_path: str, grammar: MXeneGrammar, bos_id: int, eos_id: int, pad_id: int, use_weights: bool=True):
        self.grammar = grammar
        self.bos_id, self.eos_id, self.pad_id = bos_id, eos_id, pad_id
        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="[data] reading jsonl"):
                ex = json.loads(line)
                ex["weight"] = float(ex.get("weight", 1.0))
                self.samples.append(ex)
        self.use_weights = use_weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.samples[idx]
        rules: List[int] = ex["rules"]
        # Build input: [BOS] + rules ; target: rules + [EOS]
        input_ids = [self.bos_id] + rules
        labels = rules + [self.eos_id]
        # Build allowed masks per position based on prefix
        allowed_masks: List[List[int]] = []
        prefix: List[int] = []
        for t in range(len(input_ids)):
            if t == 0:
                allowed = self.grammar.allowed_next(prefix)  # expand <S>
            else:
                prefix.append(labels[t-1])  # previous gold rule (teacher forcing)
                allowed = self.grammar.allowed_next(prefix)
            allowed_masks.append(allowed)
        # metadata for custom losses
        pos_tags = self.grammar.annotate_positions(rules)
        meta = {"outer": ex.get("outer"), "inner": ex.get("inner"), "x": ex.get("x"), "stoich": ex.get("stoich"), "pos_tags": pos_tags}
        return {"input_ids": input_ids, "labels": labels, "allowed_masks": allowed_masks, "weight": ex.get("weight", 1.0), "meta": meta}

def collate_batch(batch: List[Dict[str, Any]], vocab_size: int, pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    allowed_mask = torch.zeros((B, max_len, vocab_size), dtype=torch.bool)
    weights = torch.ones((B,), dtype=torch.float)
    metas = []
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i,:L] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i,:L] = torch.tensor(ex["labels"], dtype=torch.long)
        for t, allowed in enumerate(ex["allowed_masks"]):
            allowed_mask[i,t,allowed] = True
        weights[i] = float(ex.get("weight", 1.0))
        metas.append(ex["meta"])
    return {"input_ids": input_ids, "labels": labels, "allowed_mask": allowed_mask, "weights": weights, "metas": metas}

# --- Discriminator features ---

def extract_fields_from_rules(grammar: MXeneGrammar, rules: List[int]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    return grammar.extract_fields(rules)

def feature_vector_16(grammar: MXeneGrammar, rules: List[int]) -> List[float]:
    outer_L, outer_R, inner_L, inner_R, inner, x, stoich = grammar.extract_fields(rules)
    outer = outer_L or outer_R
    if inner is None:
        inner = inner_L or inner_R
    # groups one-hot 3,4,5,6 for outer and inner
    go = GROUP_MAP.get(outer); gi = GROUP_MAP.get(inner)
    outer_oh = [1.0 if go == g else 0.0 for g in [3,4,5,6]]
    inner_oh = [1.0 if gi == g else 0.0 for g in [3,4,5,6]]
    # ranks normalized
    ro = rank_norm(rank_index_outer(outer), len(PREFERRED_OUTER))
    ri = rank_norm(rank_index_inner(inner), len(PREFERRED_INNER))
    rd = ro - ri
    x_c = 1.0 if x == "C" else 0.0
    x_n = 1.0 if x == "N" else 0.0
    same_group = 1.0 if (go is not None and gi is not None and go == gi and outer != inner) else 0.0
    
    has_inner = 1.0 if inner is not None else 0.0
    vec = outer_oh + inner_oh + [ro, ri, rd, x_c, x_n, same_group, has_inner]
    assert len(vec) == 15
    return vec

class DiscDataset(Dataset):
    def __init__(self, seqs: List[List[int]], labels: List[int], features: List[List[float]]):
        self.seqs = seqs
        self.labels = labels
        self.features = features

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx: int):
        x = self.seqs[idx]
        y = self.labels[idx]
        f = self.features[idx]
        return x, y, f

def pad_collate(batch, pad_id: int):
    maxL = max(len(x[0]) for x in batch)
    B = len(batch)
    X = torch.full((B,maxL), pad_id, dtype=torch.long)
    y = torch.tensor([x[1] for x in batch], dtype=torch.float)
    F16 = torch.tensor([x[2] for x in batch], dtype=torch.float)
    for i,(seq,_,_) in enumerate(batch):
        X[i,:len(seq)] = torch.tensor(seq, dtype=torch.long)
    return X, y, F16
