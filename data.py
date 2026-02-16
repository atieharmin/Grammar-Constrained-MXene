from __future__ import annotations
import os, csv, json, random
from typing import Dict, Any, List, Tuple, Optional, Iterable, Set
import pandas as pd
from grammar import MXeneGrammar
from utils import ensure_dir, normalize_weights, PREFERRED_OUTER, PREFERRED_INNER, GROUP_MAP, rank_index_outer, rank_index_inner, rank_norm

def _parse_formula_like(s: str) -> Dict[str, Any]:
    # Accept strings like "M=Ti;M'=V;X=C;stoich=MX2"
    # Also handle direct formula strings like "Ti2C", "V2C", etc.
    if not s or s.lower() in ("none", "nan", ""):
        return {}
    
    # If it contains "=", parse as key=value pairs
    if "=" in s:
        parts = [p.strip() for p in s.replace(",", ";").split(";") if p.strip()]
        out: Dict[str, Any] = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k in ("m","outer"): out["outer"] = v
                elif k in ("m'","inner","m_inner"): out["inner"] = v
                elif k in ("x",): out["x"] = v
                elif k in ("stoich","stoichiometry"): out["stoich"] = v
        return out
    
    # If it's a direct formula like "Ti2C", try to extract components
    # This is a simple heuristic - could be improved with more sophisticated parsing
    s_clean = s.strip()
    if len(s_clean) >= 3:
        # Look for patterns like "Ti2C", "V2C", "Mo2C", etc.
        # Extract the metal part (before the number) and X part (after the number)
        import re
        match = re.match(r'^([A-Za-z]+)(\d+)([A-Za-z]+)$', s_clean)
        if match:
            metal_part, count, x_part = match.groups()
            out = {"outer": metal_part, "x": x_part}
            # Infer stoichiometry based on count
            if count == "2":
                out["stoich"] = "MX1"  # M2X -> MX1
            elif count == "3":
                out["stoich"] = "MX2"  # M3X2 -> MX2  
            elif count == "4":
                out["stoich"] = "MX3"  # M4X3 -> MX3
            return out
    
    return {}

# def load_unstable(unstable_csv: str) -> List[str]:
#     """
#     Load unstable compositions from CSV file.
#     Returns list of composition strings (e.g., "Sc C Nb C Sc") that are unstable.
#     """
#     compositions: List[str] = []
#     if not os.path.exists(unstable_csv):
#         return compositions
    
#     df = pd.read_csv(unstable_csv)
    
#     # Check if we have the Tokens column (preferred format for whole compositions)
#     if "Tokens" in df.columns:
#         # Check if we also have Stability Label to filter by it
#         if "Stability Label" in df.columns:
#             # Only include unstable compositions (Stability Label = 0)
#             for _, row in df.iterrows():
#                 stability = row["Stability Label"]
#                 if stability == 0:
#                     tokens = str(row["Tokens"]).strip()
#                     if tokens and tokens != "nan":
#                         compositions.append(tokens)
#         else:
#             # No stability label, assume all are unstable
#             for val in df["Tokens"].astype(str).tolist():
#                 val = val.strip()
#                 if val and val != "nan":
#                     compositions.append(val)
    
#     # Remove duplicates while preserving order
#     seen = set()
#     unique_compositions = []
#     for comp in compositions:
#         if comp not in seen:
#             seen.add(comp)
#             unique_compositions.append(comp)
    
#     return unique_compositions

def load_unstable(unstable_csv: str) -> List[str]:
    """
    Load unstable compositions from CSV file.
    Returns list of composition strings (e.g., "Sc C Nb C Sc") that are unstable.
    """
    compositions: List[str] = []
    if not os.path.exists(unstable_csv):
        return compositions
    
    df = pd.read_csv(unstable_csv)
    
    # Check if we have the Tokens column (preferred format for whole compositions)
    if "Tokens" in df.columns:
        # Check if we also have Stability Label to filter by it
        if "Stability Label" in df.columns:
            # Only include unstable compositions (Stability Label = 0)
            for _, row in df.iterrows():
                stability = row["Stability Label"]
                if stability == 0:
                    tokens = str(row["Tokens"]).strip()
                    if tokens and tokens != "nan":
                        compositions.append(tokens)
        else:
            # No stability label, assume all are unstable
            for val in df["Tokens"].astype(str).tolist():
                val = val.strip()
                if val and val != "nan":
                    compositions.append(val)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_compositions = []
    for comp in compositions:
        if comp not in seen:
            seen.add(comp)
            unique_compositions.append(comp)
    
    return unique_compositions

def load_stable_rows(unstable_csv: str) -> List[Dict[str, Any]]:
    """
    Load stable compositions from CSV file (Stability Label = 1).
    Returns list of dicts compatible with rows_to_examples.
    """
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(unstable_csv):
        return rows
    
    df = pd.read_csv(unstable_csv)
    
    # Check if we have Stability Label
    if "Stability Label" not in df.columns:
        return rows
        
    for _, r in df.iterrows():
        if r["Stability Label"] == 1:
            row: Dict[str, Any] = {}
            
            # Map columns
            if "M" in r and pd.notna(r["M"]): row["outer"] = str(r["M"]).strip()
            if "M'" in r and pd.notna(r["M'"]): row["inner"] = str(r["M'"]).strip()
            if "X" in r and pd.notna(r["X"]): row["x"] = str(r["X"]).strip()
            
            # Map Structure to stoich
            if "Structure" in r:
                s = str(r["Structure"])
                if "M2M'C2" in s or "M2M'X2" in s:
                    row["stoich"] = "MX2"
                elif "M2M'2C3" in s or "M2M'2X3" in s:
                    row["stoich"] = "MX3"
                elif "M2C" in s or "M2X" in s: 
                    row["stoich"] = "MX1"
            
            # Only add if we have minimal fields
            if "outer" in row:
                rows.append(row)
                
    return rows
    
def infer_rows(data_csv: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(data_csv)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        rdict = {k.lower(): (str(r[k]) if pd.notna(r[k]) else None) for k in r.index}
        row: Dict[str, Any] = {}
        
        # Map actual CSV columns to expected fields
        # Direct column mappings
        if "outer" in rdict and rdict["outer"] not in (None, "", "nan"):
            row["outer"] = rdict["outer"]
        if "inner" in rdict and rdict["inner"] not in (None, "", "nan", "none"):
            row["inner"] = rdict["inner"]
        elif "outer" in rdict and rdict["outer"] not in (None, "", "nan"):
            # For single-metal structures, inner should be same as outer
            row["inner"] = rdict["outer"]
        if "x" in rdict and rdict["x"] not in (None, "", "nan"):
            row["x"] = rdict["x"]
        
        # Infer stoichiometry from Grammar Type column
        if "grammar type" in rdict and rdict["grammar type"] not in (None, "", "nan"):
            grammar_type = rdict["grammar type"]
            if grammar_type == "M'2X":
                row["stoich"] = "MX1"
            elif grammar_type in ["M'3X2", "M'2M''X2"]:
                row["stoich"] = "MX2"
            elif grammar_type in ["M'4X3", "M'2M''2X3"]:
                row["stoich"] = "MX3"
         
        
        # Try to extract from id column if it's a formula
        if "id" in rdict and rdict["id"] not in (None, "", "nan"):
            id_value = rdict["id"]
            # If we don't have outer/x from direct columns, try parsing the id
            if "outer" not in row or "x" not in row:
                parsed = _parse_formula_like(id_value)
                if "outer" in parsed and "outer" not in row:
                    row["outer"] = parsed["outer"]
                if "x" in parsed and "x" not in row:
                    row["x"] = parsed["x"]
                if "stoich" in parsed and "stoich" not in row:
                    row["stoich"] = parsed["stoich"]
        
        
        # Normalize values
        if "stoich" in row:
            s = str(row["stoich"]).upper()
            if s in ("MX1", "MX2", "MX3"):
                row["stoich"] = s
            else:
                # Default to MX2 if not recognized
                row["stoich"] = "MX2"
        
        # Ensure x is valid
        if "x" in row:
            x_val = str(row["x"]).upper()
            if x_val in ("C", "N"):
                row["x"] = x_val
            else:
                row["x"] = "C"  # Default to C
        
        # Clean up None values
        row = {k: v for k, v in row.items() if v is not None and str(v).lower() not in ("none", "nan", "")}
        
        rows.append(row)
    return rows

def enumerate_valid(grammar: MXeneGrammar, alpha: float=0.6, beta: float=0.3, gamma: float=0.1) -> List[Dict[str, Any]]:
    """Enumerate a modest set of valid canonical structures under grammar constraints."""
    xs = ["C","N"]
    stoichs = ["MX1","MX2","MX3"]
    items: List[Dict[str, Any]] = []
    for stoich in stoichs:
        for x in xs:
            # outer choices
            for outer in grammar.build_vocab().metals:
                if stoich == "MX1":
                    # no inner
                    items.append({"outer": outer, "inner": None, "x": x, "stoich": stoich})
                else:
                    # choose inner distinct group and worse rank than outer (since inner pref is inverse)
                    for inner in grammar.build_vocab().metals:
                        if outer == inner: 
                            continue
                        # group mismatch rule
                        go = GROUP_MAP.get(outer); gi = GROUP_MAP.get(inner)
                        if go is not None and gi is not None and go == gi:
                            continue
                        # preference: outer rank better than inner (inner prefers inverse)
                        ro = rank_index_outer(outer); ri = rank_index_outer(inner)
                        if (ro > ri): 
                            continue
                        items.append({"outer": outer, "inner": inner, "x": x, "stoich": stoich})
    # weights
    ranks_len = len(PREFERRED_OUTER)
    weights: List[float] = []
    for it in items:
        ro = rank_norm(rank_index_outer(it["outer"]), ranks_len)
        ri = 0.0
        groups_differ = 1.0
        if it["inner"]:
            ri = rank_norm(rank_index_inner(it["inner"]), ranks_len)
            go, gi = GROUP_MAP.get(it["outer"]), GROUP_MAP.get(it["inner"])
            groups_differ = 1.0 if (go is None or gi is None or go != gi) else 0.0
        w = alpha*(1.0 - ro) + beta*(1.0 - ri) + gamma*groups_differ
        weights.append(float(w))
    weights = normalize_weights(weights)
    for it, w in zip(items, weights):
        it["weight"] = float(w)
    return items


def rows_to_examples(grammar: MXeneGrammar, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    skipped = 0
    logged = 0
    for r in rows:
        outer = r.get("outer")
        inner = r.get("inner", None)
        x = (r.get("x") or "C")
        if isinstance(x, str): x = x.strip().upper()
        stoich = (r.get("stoich") or "MX2")
        if isinstance(stoich, str): stoich = stoich.strip().upper()
        if isinstance(outer, str): outer = outer.strip()
        if isinstance(inner, str): inner = inner.strip()

        if outer is None or outer == "":
            skipped += 1
            if logged < 5:
                print(f"[rows_to_examples] skip: missing outer in row={r}")
                logged += 1
            continue
        try:
            pids = grammar.parse_to_rules(outer, inner, x, stoich)
        except Exception as e:
            skipped += 1
            if logged < 5:
                print(f"[rows_to_examples] skip row={r} due to: {repr(e)}")
                logged += 1
            continue
        text = grammar.linearize(pids)
        # weight = r.get("weight", 1.0)
        weight = 1.0
        out.append({"rules": pids, "text": text, "outer": outer, "inner": inner, "x": x, "stoich": stoich, "weight": weight})
    print(f"[rows_to_examples] built {len(out)} examples, skipped {skipped}")
    return out

def write_jsonl(examples: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
