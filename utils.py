from __future__ import annotations
import os, json, random, hashlib, math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import torch

PREFERRED_OUTER = ["Cr","Mo","W","Sc","V","Nb","Ta","Ti","Zr","Hf"]  # best -> worst
PREFERRED_INNER = list(reversed(PREFERRED_OUTER))
GROUP_MAP = {"Sc":3, "Ti":4, "V":5, "Cr":6, "Zr":4, "Nb":5, "Mo":6, "Hf":4, "Ta":5, "W":6}
GROUPS = [3,4,5,6]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def json_dump(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def json_load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_weights(ws: List[float]) -> List[float]:
    arr = np.asarray(ws, dtype=np.float64)
    if arr.size == 0: return []
    arr = arr - arr.min() if arr.max() > arr.min() else arr
    tot = float(arr.sum()) if arr.sum() > 0 else 1.0
    return list((arr / tot).astype(np.float64))

def rank_index_outer(m: str) -> int:
    return PREFERRED_OUTER.index(m) if m in PREFERRED_OUTER else len(PREFERRED_OUTER)-1

def rank_index_inner(m: str) -> int:
    return PREFERRED_INNER.index(m) if m in PREFERRED_INNER else len(PREFERRED_INNER)-1

def rank_norm(idx: int, L: int) -> float:
    return float(idx) / float(L-1)

def one_hot_group(g: int | None) -> List[float]:
    return [1.0 if g == k else 0.0 for k in GROUPS]

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-x))

def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    return (p * (p/q).log()).sum(-1)

def js_div(p: torch.Tensor, q: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    m = 0.5*(p+q)
    return 0.5*kl_div(p, m, eps) + 0.5*kl_div(q, m, eps)
