from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional, Any
import pickle, os

METALS = ["Cr","Mo","W","Sc","V","Nb","Ta","Ti","Zr","Hf"] 
X_TERMS = ["C","N"]

# Nonterminals
NT_S = "<S>"
NT_MX1 = "<MX1>"
NT_MX2 = "<MX2>"
NT_MX3 = "<MX3>"
NT_X = "<X>"

NT_M_OUTER_L = "<M_outer_L>"
NT_M_OUTER_R = "<M_outer_R>"
NT_M_INNER = "<M_inner>"
NT_M_INNER_L = "<M_inner_L>"
NT_M_INNER_R = "<M_inner_R>"

EOS = "<EOS>"
BOS = "<BOS>"
PAD = "<PAD>"

@dataclass
class Production:
    pid: int
    lhs: str
    rhs: List[str]
    text: str   # human-readable e.g., "<X>->C"

@dataclass
class RuleVocab:
    pid2prod: Dict[int, Production]
    prodkey2pid: Dict[Tuple[str, Tuple[str, ...]], int]
    specials: Dict[str, int]  # BOS, EOS, PAD
    metals: List[str]
    x_terms: List[str]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "RuleVocab":
        with open(path, "rb") as f:
            return pickle.load(f)

@dataclass
class MXeneGrammar:
    rules: Dict[str, List[List[str]]]
    terminals: Set[str]
    non_terminals: Set[str]
    prods: List[Production]
    prodkey2pid: Dict[Tuple[str, Tuple[str, ...]], int]
    bos_id: int
    eos_id: int
    pad_id: int
    metal2pids_outer_slot: Dict[str, Set[int]]
    metal2pids_inner_slot: Dict[str, Set[int]]
    slot_nts: Set[str]

    @classmethod
    def create_default(cls) -> "MXeneGrammar":
        rules = {
            NT_S:   [[NT_MX1], [NT_MX2], [NT_MX3]],

            # MX1: M_L X M_R (lets us check L==R in validator)
            NT_MX1: [["STOICH_MX1", NT_M_OUTER_L, NT_X, NT_M_OUTER_R]],

            # MX2: M_L X M' X M_R
            NT_MX2: [["STOICH_MX2", NT_M_OUTER_L, NT_X, NT_M_INNER, NT_X, NT_M_OUTER_R]],

            # MX3: M_L X M'_L X M'_R X M_R  (validator will require M'_L == M'_R)
            NT_MX3: [["STOICH_MX3", NT_M_OUTER_L, NT_X, NT_M_INNER_L, NT_X, NT_M_INNER_R, NT_X, NT_M_OUTER_R]],

            NT_X: [[x] for x in X_TERMS],
            NT_M_OUTER_L: [[m] for m in METALS],
            NT_M_OUTER_R: [[m] for m in METALS],
            NT_M_INNER:   [[m] for m in METALS],     # used by MX2 (single inner)
            NT_M_INNER_L: [[m] for m in METALS],     # used by MX3 (left inner)
            NT_M_INNER_R: [[m] for m in METALS],     # used by MX3 (right inner)
        }

        terminals = set(["STOICH_MX1","STOICH_MX2","STOICH_MX3"] + X_TERMS + METALS + [EOS, BOS, PAD])
        non_terminals = set(rules.keys())

         # Compile productions
        prods: List[Production] = []
        prodkey2pid: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        pid = 0
        for lhs, rhss in rules.items():
            for rhs in rhss:
                key = (lhs, tuple(rhs))
                text = f"{lhs}->{','.join(rhs)}"
                prodkey2pid[key] = pid
                prods.append(Production(pid=pid, lhs=lhs, rhs=rhs, text=text))
                pid += 1
        # Specials
        bos_id = pid; pid += 1
        eos_id = pid; pid += 1
        pad_id = pid; pid += 1


        # Precompute metal->pid mapping for slots (extend to inner_L/R)
        metal2pids_outer_slot = {m:set() for m in METALS}
        metal2pids_inner_slot = {m:set() for m in METALS}
        for p in prods:
            if p.lhs in (NT_M_OUTER_L, NT_M_OUTER_R) and len(p.rhs)==1 and p.rhs[0] in METALS:
                metal2pids_outer_slot[p.rhs[0]].add(p.pid)
            if p.lhs in (NT_M_INNER, NT_M_INNER_L, NT_M_INNER_R) and len(p.rhs)==1 and p.rhs[0] in METALS:
                metal2pids_inner_slot[p.rhs[0]].add(p.pid)

        slot_nts = {NT_M_OUTER_L, NT_M_OUTER_R, NT_M_INNER, NT_M_INNER_L, NT_M_INNER_R}

        return cls(
            rules=rules,
            terminals=terminals,
            non_terminals=non_terminals,
            prods=prods,
            prodkey2pid=prodkey2pid,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            metal2pids_outer_slot=metal2pids_outer_slot,
            metal2pids_inner_slot=metal2pids_inner_slot,
            slot_nts=slot_nts
        )

    # --- Parsing and masking utilities ---

    def allowed_next(self, prefix_pids: List[int]) -> List[int]:
        """
        Given a prefix of production IDs, return the list of allowed next production IDs
        based on the parse stack. If parsing is complete (stack empty), only EOS is allowed.
        Start state: stack = [<S>].
        """
        stack: List[str] = [NT_S]
        # Apply prefix productions sequentially
        for pid in prefix_pids:
            if pid in (self.bos_id, self.pad_id):  # ignore BOS/PAD if present
                continue
            if pid == self.eos_id:
                # Once EOS is applied, no further expansions
                return [self.eos_id]
            if not stack:
                # Already done; only EOS
                return [self.eos_id]
            top = stack.pop()
            prod = self.prods[pid]
            assert prod.lhs == top, f"Prefix violates grammar at pid={pid}, lhs={prod.lhs}, expected={top}"
            # push RHS in reverse
            for sym in reversed(prod.rhs):
                if sym in self.non_terminals:
                    stack.append(sym)
                # terminals are emitted implicitly; no stack action
        # Find next non-terminal to expand
        while stack and stack[-1] not in self.non_terminals:
            stack.pop()
        if not stack:
            return [self.eos_id]
        lhs = stack[-1]
        # allowed productions are all with this lhs
        return [p.pid for p in self.prods if p.lhs == lhs]

    def parse_to_rules(self, outer: str, inner: Optional[str], x: str, stoich: str) -> List[int]:
        if stoich not in ("MX1","MX2","MX3"):
            raise ValueError(f"Unsupported stoich: {stoich}")

        # 1) S -> <MXk>
        nt_map = {"MX1": "<MX1>", "MX2": "<MX2>", "MX3": "<MX3>"}
        seq: List[int] = []
        seq.append(self.prodkey2pid[("<S>", (nt_map[stoich],))])

        # 2) expand selected MXk rule (find its single RHS from self.rules)
        mx_nt = nt_map[stoich]
        mx_rhs_list = self.rules[mx_nt]
        if len(mx_rhs_list) != 1:
            raise ValueError(f"Expected exactly 1 production for {mx_nt}, got {len(mx_rhs_list)}")
        mx_rhs = tuple(mx_rhs_list[0])
        seq.append(self.prodkey2pid[(mx_nt, mx_rhs)])

        # 3) walk the RHS; for each NT, emit the chosen production
        #    (terminals: no production emitted; only NTs produce productions)
        for sym in mx_rhs:
            if sym not in self.non_terminals:
                # terminals like STOICH_MXk are emitted implicitly (no prod)
                continue

            if sym == "<X>":
                seq.append(self.prodkey2pid[("<X>", (x,))])


            elif sym == "<M_outer_L>":
                seq.append(self.prodkey2pid[("<M_outer_L>", (outer,))])

            elif sym == "<M_outer_R>":
                seq.append(self.prodkey2pid[("<M_outer_R>", (outer,))])

            elif sym == "<M_inner>":
                seq.append(self.prodkey2pid[("<M_inner>", (inner,))])

            elif sym == "<M_inner_L>":
                seq.append(self.prodkey2pid[("<M_inner_L>", (inner,))])

            elif sym == "<M_inner_R>":
                seq.append(self.prodkey2pid[("<M_inner_R>", (inner,))])

            else:
                raise ValueError(f"Unknown non-terminal in MX rule RHS: {sym}")

        return seq

    def linearize(self, pids: List[int]) -> str:
        return " ".join([self.prods[pid].text if pid < len(self.prods) else ("<EOS>" if pid == self.eos_id else "<BOS/PAD>") for pid in pids])

    def annotate_positions(self, pids: List[int]) -> Dict[str, List[int]]:
        """
        Return index lists where metals or X choices occur.
        Keys: 'outer_single', 'outer_L', 'outer_R', 'inner', 'x'
        """
        positions = {"outer_L": [], "outer_R": [], "inner": [], "inner_L": [], "inner_R": [], "x": []}

        stack = [NT_S]
        t = 0
        for pid in pids:
            if pid in (self.bos_id, self.pad_id, self.eos_id): 
                t += 1
                continue
            top = stack.pop() if stack else None
            prod = self.prods[pid]
            # mark positions
            if prod.lhs == NT_X:
                positions["x"].append(t)
            elif prod.lhs == NT_M_OUTER_L:
                positions["outer_L"].append(t)
            elif prod.lhs == NT_M_OUTER_R:
                positions["outer_R"].append(t)
            elif prod.lhs == NT_M_INNER:
                positions["inner"].append(t)
            elif prod.lhs == NT_M_INNER_L:
                positions["inner_L"].append(t)
            elif prod.lhs == NT_M_INNER_R:
                positions["inner_R"].append(t)
            # push rhs
            for sym in reversed(prod.rhs):
                if sym in self.non_terminals:
                    stack.append(sym)
            t += 1
        return positions

    def extract_fields(self, pids: List[int]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Extract (outer_L, outer_R, inner_L, inner_R, inner, x, stoich) from production sequence.
        """
        stack = [NT_S]
        stoich = None
        outer_L = outer_R = inner_L = inner_R = inner = x = None
        for pid in pids:
            if pid == self.eos_id: break
            prod = self.prods[pid] if pid < len(self.prods) else None
            if not prod: continue
            if prod.lhs in (NT_MX1, NT_MX2, NT_MX3):
                stoich = prod.rhs[0].replace("STOICH_", "")
            if prod.lhs == NT_X: x = prod.rhs[0]
            if prod.lhs == NT_M_OUTER_L: outer_L = prod.rhs[0]
            if prod.lhs == NT_M_OUTER_R: outer_R = prod.rhs[0]
            if prod.lhs == NT_M_INNER: inner = prod.rhs[0]
            if prod.lhs == NT_M_INNER_L: inner_L = prod.rhs[0]
            if prod.lhs == NT_M_INNER_R: inner_R = prod.rhs[0]
        return outer_L, outer_R, inner_L, inner_R, inner, x, stoich

    def sequence_to_tokens(self, seq: List[int]) -> str:
        """
        Convert a rule sequence to a space-separated token string.
        For example: [rules for Sc2NbC2] -> "Sc C Nb C Sc"
        """
        outer_L, outer_R, inner_L, inner_R, inner, x, stoich = self.extract_fields(seq)
        
        # Build the token string based on stoichiometry
        if stoich == "MX1":
            # M_outer_L X M_outer_R
            left = outer_L or outer_R
            right = outer_R or outer_L or left
            return f"{left} {x} {right}"
        elif stoich == "MX2":
            # M_outer_L X M_inner X M_outer_R
            left = outer_L or outer_R
            right = outer_R or outer_L or left
            mid = inner
            return f"{left} {x} {mid} {x} {right}"
        elif stoich == "MX3":
            # M_outer_L X M_inner_L X M_inner_R X M_outer_R
            left = outer_L or outer_R
            right = outer_R or outer_L or left
            mid_L = inner_L or inner
            mid_R = inner_R or inner_L or inner
            return f"{left} {x} {mid_L} {x} {mid_R} {x} {right}"
        else:
            # Unknown stoichiometry
            return ""
            
    def build_vocab(self) -> RuleVocab:
        pid2prod = {p.pid: p for p in self.prods}
        specials = {BOS: self.bos_id, EOS: self.eos_id, PAD: self.pad_id}
        return RuleVocab(
            pid2prod=pid2prod,
            prodkey2pid=self.prodkey2pid,
            specials=specials,
            metals=METALS,
            x_terms=X_TERMS
        )
