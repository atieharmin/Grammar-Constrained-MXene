from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config
from grammar import MXeneGrammar
from tqdm import tqdm

class GeneratorModel(nn.Module):
    def __init__(self, grammar: MXeneGrammar, model_cfg: Dict[str, Any]):
        super().__init__()
        vocab_size = len(grammar.prods) + 3  # + specials
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=model_cfg.get("n_positions",128),
            n_ctx=model_cfg.get("n_ctx",128),
            n_layer=model_cfg.get("n_layer",4),
            n_head=model_cfg.get("n_head",4),
            n_embd=model_cfg.get("n_embd",192),
            resid_pdrop=model_cfg.get("dropout",0.1),
            embd_pdrop=model_cfg.get("dropout",0.1),
            attn_pdrop=model_cfg.get("dropout",0.1),
            bos_token_id=grammar.bos_id,
            eos_token_id=grammar.eos_id
        )
        self.lm = GPT2LMHeadModel(cfg)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        return self.lm(input_ids=input_ids, attention_mask=attention_mask)

def apply_grammar_mask(logits: torch.Tensor, allowed_mask: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, T, V]
    allowed_mask: [B, T, V] boolean True where allowed
    """
    masked = logits.masked_fill(~allowed_mask, -1e9)
    return masked

@torch.no_grad()
def sample_with_grammar(
    model: GeneratorModel,
    grammar: MXeneGrammar,
    max_len: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 32,
) -> List[List[int]]:
    model.eval()
    V = len(grammar.prods) + 3
    bos = grammar.bos_id
    sequences: List[List[int]] = []
    for _ in tqdm(range(batch_size), desc="[gen] sampling batches"):
        seq: List[int] = []
        prefix: List[int] = []
        # first position: allowed expansions of <S>
        for t in range(max_len):
            allowed = grammar.allowed_next(prefix)
            # Build logits for single step: feed current prefix
            input_ids = torch.tensor([[bos] + prefix], dtype=torch.long, device=device)
            out = model(input_ids=input_ids)
            logits = out.logits[:, -1, :]  # [1, V]
            # mask
            mask = torch.zeros((1, V), dtype=torch.bool, device=device)
            mask[0, allowed] = True
            logits = logits / max(temperature, 1e-6)
            logits = logits.masked_fill(~mask, -1e9)
            # nucleus top-p
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumprobs > top_p).float().argmax(dim=-1, keepdim=True)
            # keep at least 1
            cutoff = torch.maximum(cutoff, torch.zeros_like(cutoff))
            mask_keep = torch.zeros_like(sorted_probs, dtype=torch.bool)
            for i in range(mask_keep.size(0)):
                k = int(cutoff[i].item()) + 1
                mask_keep[i, :k] = True
            filtered_probs = sorted_probs * mask_keep
            filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + 1e-9)
            idx = torch.multinomial(filtered_probs, num_samples=1)
            next_id = sorted_idx.gather(-1, idx).item()
            # stop?
            if next_id == grammar.eos_id:
                break
            seq.append(next_id)
            prefix.append(next_id)
            if len(grammar.allowed_next(prefix)) == 1 and grammar.allowed_next(prefix)[0] == grammar.eos_id:
                break
        sequences.append(seq)
    return sequences
