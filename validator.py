from __future__ import annotations
from typing import Optional, Tuple, List, Dict
from grammar import MXeneGrammar
from utils import PREFERRED_OUTER, PREFERRED_INNER, GROUP_MAP

INVALID_REASONS = {"symmetry","single_x","preference","same_group","rule_invalid"}

def validate_rule_sequence(grammar: MXeneGrammar, pids: List[int]) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_valid, reason) with reason in:
    {"symmetry","single_x","preference","same_group","rule_invalid"} or None if valid.
    """
    # First: syntactic check by replaying allowed-next
    prefix = []
    try:
        for pid in pids:
            allowed = grammar.allowed_next(prefix)
            if pid not in allowed:
                return False, "rule_invalid"
            prefix.append(pid)
    except AssertionError:
        return False, "rule_invalid"

    outerL, outerR, innerL, innerR, inner, x, stoich = grammar.extract_fields(pids)

    # Single X
    if x not in ("C","N"):
        return False, "single_x"

    # Symmetry: for MX2/MX3 require OUTER_L == OUTER_R if both present
    pos = grammar.annotate_positions(pids)
    if pos.get("outer_L") and pos.get("outer_R"):
        if outerL is not None and outerR is not None and outerL != outerR:
            return False, "symmetry"

    # MX3: if both inner_L and inner_R exist, require equality
    if pos.get("inner_L") and pos.get("inner_R"):
        if innerL is not None and innerR is not None and innerL != innerR:
            return False, "symmetry"



    # Preference: outer rank strictly better than inner when inner exists
    
    if inner is None:
        inner = innerL or innerR
    # Define outer from L/R if available (prefer L, else R)
    outer = outerL or outerR
    try:
        outer_rank = PREFERRED_OUTER.index(outer) if outer in PREFERRED_OUTER else len(PREFERRED_OUTER)-1
        inner_rank = PREFERRED_OUTER.index(inner) if inner in PREFERRED_OUTER else len(PREFERRED_OUTER)-1
    except Exception:
        outer_rank, inner_rank = 0, 0
    if (outer_rank > inner_rank):
        return False, "preference"

    # Group mismatch: outer and inner must not be same group when inner exists
    g_outer = GROUP_MAP.get(outer)
    g_inner = GROUP_MAP.get(inner)
    if g_outer is not None and g_inner is not None and g_outer == g_inner and outer != inner:
        return False, "same_group"

    return True, None
