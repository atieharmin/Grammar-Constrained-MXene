from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, json, time, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from grammar import MXeneGrammar
from tqdm import tqdm
from datasets import JSONLSequences, collate_batch, feature_vector_16, InMemoryDataset
from models.generator import GeneratorModel, apply_grammar_mask
from utils import ensure_dir, set_seed, pick_device, js_div, rank_index_outer, rank_index_inner, rank_norm, GROUP_MAP, PREFERRED_OUTER, PREFERRED_INNER
from torch.distributions import Categorical
from train_disc import load_discriminator_for_shaping

def rollout_autoregressive_with_logprobs(
    grammar: MXeneGrammar,
    model: GeneratorModel,
    device: torch.device,
    batch_size: int,
    max_len: int,
    V: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
):

    bos = grammar.bos_id
    eos = grammar.eos_id
    pad = grammar.pad_id

    seqs: List[List[int]] = []
    logprob_sums: List[torch.Tensor] = []

    # We'll generate each sequence independently (batch is small, so this is fine)
    for _ in range(batch_size):
        prefix: List[int] = []   # production IDs only (no BOS/EOS/PAD)
        seq: List[int] = []      # what we actually store for D
        logprob_sum = torch.zeros((), dtype=torch.float, device=device)

        for t in range(max_len):
            allowed = grammar.allowed_next(prefix)
            # If only EOS is allowed, we can stop
            if len(allowed) == 1 and allowed[0] == eos:
                break

            # Build input_ids: [BOS] + prefix
            input_ids = torch.tensor([[bos] + prefix], dtype=torch.long, device=device)
            out = model(input_ids=input_ids)
            logits = out.logits[:, -1, :]   # [1, V] logits for next position

            # Mask out invalid rules
            mask = torch.zeros((1, V), dtype=torch.bool, device=device)
            mask[0, allowed] = True

            # Temperature scaling
            logits = logits / max(temperature, 1e-6)
            logits = logits.masked_fill(~mask, float('-inf'))

            # Softmax
            probs = torch.softmax(logits, dim=-1)  # [1, V]

            # Optional top-p (nucleus) sampling, same spirit as sample_with_grammar
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumprobs > top_p).float().argmax(dim=-1, keepdim=True)
                cutoff = torch.maximum(cutoff, torch.zeros_like(cutoff))

                mask_keep = torch.zeros_like(sorted_probs, dtype=torch.bool)
                for i in range(mask_keep.size(0)):  # here it's just 1 row
                    k = int(cutoff[i].item()) + 1
                    mask_keep[i, :k] = True

                filtered_probs = sorted_probs * mask_keep
                filtered_probs = filtered_probs / (
                    filtered_probs.sum(dim=-1, keepdim=True) + 1e-9
                )

                idx_sorted = torch.multinomial(filtered_probs, num_samples=1)  # [1,1]
                next_id = int(sorted_idx.gather(-1, idx_sorted).item())
                # log-prob under the truncated distribution
                prob_selected = filtered_probs[0, int(idx_sorted.item())]
                log_prob = torch.log(prob_selected + 1e-9)
            else:
                # Plain categorical sampling over masked, temperature-scaled probs
                dist = Categorical(probs)
                next_token = dist.sample()  # [1]
                next_id = int(next_token.item())
                log_prob = dist.log_prob(next_token)[0]

            # Stop on EOS (do not include EOS in seq)
            if next_id == eos:
                break

            # Record choice
            prefix.append(next_id)
            seq.append(next_id)
            logprob_sum = logprob_sum + log_prob

            # Early stop if next step would be forced EOS
            allowed_next = grammar.allowed_next(prefix)
            if len(allowed_next) == 1 and allowed_next[0] == eos:
                break

        seqs.append(seq)
        logprob_sums.append(logprob_sum)

    if len(logprob_sums) == 0:
        return [], torch.zeros((0,), dtype=torch.float, device=device)

    logprob_sums_tensor = torch.stack(logprob_sums, dim=0)  # [B]
    return seqs, logprob_sums_tensor

# def compute_symmetry_loss(grammar: MXeneGrammar, logits: torch.Tensor, labels: torch.Tensor, metas: List[Dict[str,Any]]) -> torch.Tensor:

#     B, T, V = logits.shape
#     loss_terms = []
#     probs = torch.softmax(logits, dim=-1)  # [B,T,V]

#     # Build mappings for different rule types
#     outer_map = grammar.metal2pids_outer_slot  # dict metal->set(pid)
#     inner_map = grammar.metal2pids_inner_slot  # dict metal->set(pid)
    
#     # Get all metal and X rule IDs
#     metal_pids = set()
#     for pids in outer_map.values():
#         metal_pids.update(pids)
#     for pids in inner_map.values():
#         metal_pids.update(pids)
    
#     x_pids = set()
#     for prod in grammar.prods:
#         if prod.lhs == "<X>":
#             x_pids.add(prod.pid)
    
#     metals = list(outer_map.keys())
#     metal_idx = {m:i for i,m in enumerate(metals)}

#     for b, meta in enumerate(metas):
#         pos = meta["pos_tags"]
        
#         # Handle outer metal symmetry (outer_L vs outer_R)
#         if pos["outer_L"] and pos["outer_R"]:
#             i = pos["outer_L"][0]; j = pos["outer_R"][0]
#             # distribution over metals at each pos
#             Pi = torch.zeros((len(metals),), device=logits.device)
#             Pj = torch.zeros((len(metals),), device=logits.device)
#             for m, pids in outer_map.items():
#                 idxs = torch.tensor(list(pids), dtype=torch.long, device=logits.device)
#                 Pi[metal_idx[m]] = probs[b, i, idxs].sum()
#                 Pj[metal_idx[m]] = probs[b, j, idxs].sum()
#             # normalize
#             Pi = Pi / (Pi.sum() + 1e-9)
#             Pj = Pj / (Pj.sum() + 1e-9)
#             loss_terms.append(js_div(Pi, Pj))
        
#         # Handle inner metal symmetry (inner_L vs inner_R) for MX3 structures
#         if pos["inner_L"] and pos["inner_R"]:
#             i = pos["inner_L"][0]; j = pos["inner_R"][0]
#             # distribution over metals at each pos
#             Pi = torch.zeros((len(metals),), device=logits.device)
#             Pj = torch.zeros((len(metals),), device=logits.device)
#             for m, pids in inner_map.items():
#                 idxs = torch.tensor(list(pids), dtype=torch.long, device=logits.device)
#                 Pi[metal_idx[m]] = probs[b, i, idxs].sum()
#                 Pj[metal_idx[m]] = probs[b, j, idxs].sum()
#             # normalize
#             Pi = Pi / (Pi.sum() + 1e-9)
#             Pj = Pj / (Pj.sum() + 1e-9)
#             loss_terms.append(js_div(Pi, Pj))
        
#         # Handle X element symmetry - compare all X positions
#         if pos["x"] and len(pos["x"]) >= 2:
#             x_positions = pos["x"]
#             # Compare each X position with its mirrored counterpart
#             for i in range(len(x_positions) // 2):
#                 j = len(x_positions) - 1 - i  # mirror position
#                 if i != j:  # don't compare position with itself
#                     pos_i = x_positions[i]
#                     pos_j = x_positions[j]
                    
#                     # Get distributions over X elements
#                     x_idxs = torch.tensor(list(x_pids), dtype=torch.long, device=logits.device)
#                     if len(x_idxs) > 0:
#                         Pi = probs[b, pos_i, x_idxs]
#                         Pj = probs[b, pos_j, x_idxs]
#                         # normalize
#                         Pi = Pi / (Pi.sum() + 1e-9)
#                         Pj = Pj / (Pj.sum() + 1e-9)
#                         loss_terms.append(js_div(Pi, Pj))
    
#     if len(loss_terms) == 0:
#         return torch.tensor(0.0, device=logits.device)
#     return torch.stack(loss_terms).mean()

def compute_occupancy_loss(grammar: MXeneGrammar, logits: torch.Tensor, labels: torch.Tensor, metas: List[Dict[str,Any]]) -> torch.Tensor:
    """
    Relative Ordering Loss:
    Penalize if Expected_Rank(Outer) > Expected_Rank(Inner).
    Uses the same PREFERRED_OUTER list for both ranks (lower index = lighter/better for outer).
    """
    probs = torch.softmax(logits, dim=-1) 
    loss_terms = []
    ranks_len = len(PREFERRED_OUTER) # 10
    device = logits.device
    
    outer_map = grammar.metal2pids_outer_slot
    inner_map = grammar.metal2pids_inner_slot

    for b, meta in enumerate(metas):
        pos = meta["pos_tags"]
        outer_positions = pos["outer_L"] + pos["outer_R"]
        inner_positions = pos["inner"] + pos["inner_L"] + pos["inner_R"]

        # If we don't have both outer and inner, we can't compare ordering.
        if not outer_positions or not inner_positions:
            continue

        # 1. Calculate Expected Rank for Outer Positions
        # We average the expected rank across all outer slots
        E_rank_outer = torch.tensor(0.0, device=device)
        for t in outer_positions:
            step_rank = torch.tensor(0.0, device=device)
            for m, pids in outer_map.items():
                if not pids: continue
                idxs = torch.tensor(list(pids), device=device, dtype=torch.long)
                p_m = probs[b, t, idxs].sum()
                # Rank: 0 (Cr) -> 9 (Hf) normalized to [0,1]
                r = float(rank_index_outer(m)) / (ranks_len - 1)
                step_rank = step_rank + (p_m * r)
            E_rank_outer = E_rank_outer + step_rank
        E_rank_outer = E_rank_outer / len(outer_positions)

        # 2. Calculate Expected Rank for Inner Positions
        # CRITICAL: We use rank_index_outer for Inner too, so we compare on SAME scale.
        E_rank_inner = torch.tensor(0.0, device=device)
        for t in inner_positions:
            step_rank = torch.tensor(0.0, device=device)
            for m, pids in inner_map.items():
                if not pids: continue
                idxs = torch.tensor(list(pids), device=device, dtype=torch.long)
                p_m = probs[b, t, idxs].sum()
                # Rank: 0 (Cr) -> 9 (Hf) normalized to [0,1]
                r = float(rank_index_outer(m)) / (ranks_len - 1)
                step_rank = step_rank + (p_m * r)
            E_rank_inner = E_rank_inner + step_rank
        E_rank_inner = E_rank_inner / len(inner_positions)

        # 3. Loss = ReLU(Outer_Rank - Inner_Rank)
        # We want Outer < Inner (Lighter Outside).
        # If Outer (e.g. 8) > Inner (e.g. 2), Loss = 6.
        diff = E_rank_outer - E_rank_inner
        loss_terms.append(torch.relu(diff))

    if len(loss_terms) == 0:
        return torch.tensor(0.0, device=device)
    
    return torch.stack(loss_terms).mean()

def compute_group_penalty(grammar: MXeneGrammar, logits: torch.Tensor, labels: torch.Tensor, metas: List[Dict[str,Any]]) -> torch.Tensor:
    """
    Penalize the model for predicting high probabilities of same-group metal combinations.
    This guides the model to avoid chemically unfavorable same-group pairings.
    """
    B, T, V = logits.shape
    probs = torch.softmax(logits, dim=-1)
    penalty_terms = []
    
    # Build mappings for metal rule IDs
    outer_map = grammar.metal2pids_outer_slot
    inner_map = grammar.metal2pids_inner_slot
    
    for b, meta in enumerate(metas):
        pos = meta["pos_tags"]
        
        # Get outer and inner positions
        outer_positions = pos["outer_L"] + pos["outer_R"]
        inner_positions = pos["inner"] + pos["inner_L"] + pos["inner_R"]
        
        # Calculate penalty for each outer-inner pair
        for outer_pos in outer_positions:
            for inner_pos in inner_positions:
                if outer_pos == inner_pos:
                    continue
                
                # Get probability distributions over metals at each position
                outer_probs = torch.zeros(len(outer_map), device=logits.device)
                inner_probs = torch.zeros(len(inner_map), device=logits.device)
                
                # Extract probabilities for each metal
                for i, (metal, pids) in enumerate(outer_map.items()):
                    if pids:
                        idxs = torch.tensor(list(pids), dtype=torch.long, device=logits.device)
                        outer_probs[i] = probs[b, outer_pos, idxs].sum()
                
                for i, (metal, pids) in enumerate(inner_map.items()):
                    if pids:
                        idxs = torch.tensor(list(pids), dtype=torch.long, device=logits.device)
                        inner_probs[i] = probs[b, inner_pos, idxs].sum()
                
                # Normalize
                outer_probs = outer_probs / (outer_probs.sum() + 1e-9)
                inner_probs = inner_probs / (inner_probs.sum() + 1e-9)
                
                # Calculate penalty for same-group combinations
                metals = list(outer_map.keys())
                penalty = 0.0
                for i, outer_metal in enumerate(metals):
                    for j, inner_metal in enumerate(metals):
                        if i < len(outer_probs) and j < len(inner_probs):
                            # Skip if same element (valid single-metal structure)
                            if outer_metal == inner_metal:
                                continue
                                
                            outer_group = GROUP_MAP.get(outer_metal)
                            inner_group = GROUP_MAP.get(inner_metal)
                            
                            # Penalize only if different elements from same group
                            if (outer_group is not None and inner_group is not None and 
                                outer_group == inner_group):
                                penalty += outer_probs[i] * inner_probs[j]
                
                penalty_terms.append(penalty)
    
    if len(penalty_terms) == 0:
        return torch.tensor(0.0, device=logits.device)
    
    return torch.stack(penalty_terms).mean()

def train_generator(config: Dict[str, Any]) -> None:
    run_dir = config["run_dir"]; ensure_dir(run_dir)
    set_seed(config.get("seed", 123))
    device = pick_device(config.get("device", "auto"))

    grammar = MXeneGrammar.create_default()
    # dataset = JSONLSequences(config["data_jsonl"], grammar, grammar.bos_id, grammar.eos_id, grammar.pad_id, use_weights=True)

     # Support multiple jsonls: base + HITL
    jsonls = [config["data_jsonl"]] + list(config.get("extra_jsonls", []))
    samples = []
    for p in jsonls:
        ds = JSONLSequences(p, grammar, grammar.bos_id, grammar.eos_id, grammar.pad_id, use_weights=True)
        samples.extend([ds[i] for i in range(len(ds))])
    # tiny adapter Dataset from in-memory list
    dataset = InMemoryDataset(samples)
    V = len(grammar.prods) + 3

    model = GeneratorModel(grammar, config.get("model_cfg", {})).to(device)

     # Optionally resume from checkpoint
    if os.path.exists(config.get("resume_ckpt","")):
        ckpt = torch.load(config["resume_ckpt"], map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)

    opt = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3), betas=tuple(config.get("betas",[0.9,0.95])))
    train_loader = DataLoader(dataset, batch_size=config.get("batch_size", 16), shuffle=True,
                              collate_fn=lambda b: collate_batch(b, V, grammar.pad_id), num_workers=8)
    total_steps = config.get("epochs", 2) * max(1, len(train_loader))
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    ce = nn.CrossEntropyLoss(reduction="none")
    global_step = 0

    rl_weight = float(config.get("rl_weight", 0.0))
    rl_start_epoch = int(config.get("rl_start_epoch", 1))
    disc = None
    disc_ckpt = config.get("disc_ckpt", None)
    if disc_ckpt and os.path.exists(disc_ckpt):
        
        disc = load_discriminator_for_shaping(
            ckpt_path=disc_ckpt,
            vocab_size=V,
            pad_id=grammar.pad_id,
            disc_model_cfg=config.get("disc_model_cfg", {}),
            device=device,
        )

    for epoch in tqdm(range(1, config.get("epochs", 2)+1), desc="[gen] epochs"):
        model.train()
        batch_iter = tqdm(train_loader, desc=f"[gen] epoch {epoch}", total=len(train_loader))
        for batch in batch_iter:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            allowed_mask = batch["allowed_mask"].to(device)
            weights = batch["weights"].to(device)
            metas = batch["metas"]

            out = model(input_ids=input_ids)
            logits = out.logits  # [B,T,V]
            masked_logits = apply_grammar_mask(logits, allowed_mask)
            # CE on each step
            loss_tok = ce(masked_logits.transpose(1,2), labels)  # [B,T]
            # weight by sample weight and average over tokens != -100
            mask_valid = (labels != -100).float()
            ce_loss = ((loss_tok * mask_valid).sum(dim=1) / (mask_valid.sum(dim=1) + 1e-9))
            ce_loss = (ce_loss * weights).mean()
            # Custom losses with ramping
            warmup = int(config.get("warmup_steps", 50))
            ramp = min(1.0, float(global_step) / float(max(1, 2*warmup)))
            # symm_coeff = ramp * float(config.get("symm_weight", 0.0))
            occ_coeff = ramp * float(config.get("occ_weight", 0.0))
            grp_coeff = ramp * float(config.get("group_weight", 0.0))

            # symm_loss = compute_symmetry_loss(grammar, masked_logits, labels, metas) * symm_coeff
            occ_loss = compute_occupancy_loss(grammar, masked_logits, labels, metas) * occ_coeff
            grp_loss = compute_group_penalty(grammar, masked_logits, labels, metas) * grp_coeff
            total_loss = ce_loss + occ_loss + grp_loss

            # # Adversarial shaping (safe; no grad to D)
            # if adv_weight > 0 and disc is not None and epoch >= adv_after:
            #     with torch.no_grad():
            #         # Build compact sequences for D (exclude PADs)
            #         seqs = []
            #         for i in range(input_ids.size(0)):
            #             toks = input_ids[i].tolist()
            #             # strip leading BOS and trailing PAD
            #             toks = [t for t in toks if t != grammar.pad_id]
            #             if toks and toks[0] == grammar.bos_id: toks = toks[1:]
            #             seqs.append(toks)
            #         # Pad to max
            #         maxL = max(len(s) for s in seqs)
            #         X = torch.full((len(seqs), maxL), grammar.pad_id, dtype=torch.long, device=device)
            #         feats = []
            #         for i,s in enumerate(seqs):
            #             X[i,:len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            #             feats.append(feature_vector_16(grammar, s))
                    
            #         feats_tensor = torch.tensor(feats, dtype=torch.float, device=device)
            #         logits_d = disc(X, features16=feats_tensor)
            #         p_valid = torch.sigmoid(logits_d).mean()
            #         adv_pen = (1.0 - p_valid)
            #     total_loss = total_loss + adv_weight * adv_pen

            # --- Adversarial shaping via full autoregressive REINFORCE ---
            if rl_weight > 0 and disc is not None and epoch >= rl_start_epoch:
                # 1) Roll out sequences from BOS using the current generator policy
                B_cur = input_ids.size(0)
                sampling_cfg = config.get("sampling", {})
                max_len_rl = sampling_cfg.get("max_len", 64)
                temp_rl = sampling_cfg.get("temperature", 1.0)
                top_p_rl = sampling_cfg.get("top_p", 0.9)

                seqs, logprob_sums = rollout_autoregressive_with_logprobs(
                    grammar=grammar,
                    model=model,
                    device=device,
                    batch_size=B_cur,
                    max_len=max_len_rl,
                    V=V,
                    temperature=temp_rl,
                    top_p=top_p_rl,
                )  # seqs: List[List[int]]; logprob_sums: [B]

                # 2) Build discriminator inputs from sampled sequences
                with torch.no_grad():
                    if len(seqs) > 0 and any(len(s) > 0 for s in seqs):
                        maxL = max(len(s) for s in seqs if len(s) > 0)
                        X = torch.full(
                            (len(seqs), maxL),
                            grammar.pad_id,
                            dtype=torch.long,
                            device=device,
                        )
                        feats = []
                        for i, s in enumerate(seqs):
                            if len(s) > 0:
                                X[i, :len(s)] = torch.tensor(
                                    s, dtype=torch.long, device=device
                                )
                            feats.append(feature_vector_16(grammar, s))

                        feats_tensor = torch.tensor(
                            feats, dtype=torch.float, device=device
                        )

                        logits_d = disc(X, features16=feats_tensor).view(-1)  # [B]
                        p_valid = torch.sigmoid(logits_d)                      # [B]
                    else:
                        # Edge case: all empty sequences (shouldn't really happen)
                        p_valid = torch.zeros(B_cur, device=device)

                    rewards = p_valid  # in [0,1]: higher = more "valid" per D
                # 3) REINFORCE loss: -(reward - baseline) * log p_theta(sequence)
                baseline = rewards.mean()
                advantage = rewards - baseline                # [B]
                # Detach advantage so gradients don't flow into D
                rl_loss = -(advantage.detach() * logprob_sums).mean()
                total_loss = total_loss + rl_weight * rl_loss

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step()
            global_step += 1
            # update progress bar postfix with current losses
            try:
                batch_iter.set_postfix({
                    "ce": float(ce_loss.detach().cpu().item()),
                    # "sym": float(symm_loss.detach().cpu().item()),
                    "occ": float(occ_loss.detach().cpu().item()),
                    "grp": float(grp_loss.detach().cpu().item()),
                    "rl": float(rl_loss.detach().cpu().item())
                })
            except Exception:
                pass

        if epoch % int(config.get("save_every", 1)) == 0:
            torch.save({"model": model.state_dict(), "cfg": config, "vocab_size": V}, os.path.join(run_dir, "generator.pt"))

    # Final save
    torch.save({"model": model.state_dict(), "cfg": config, "vocab_size": V}, os.path.join(run_dir, "generator.pt"))
