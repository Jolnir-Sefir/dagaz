"""
dagaz_runtime.py — Pure Python MeTTa Evaluator for Project Dagaz.

DESIGN PRINCIPLES COMPLIANCE:
  - Bottom-Up: No domain-specific hacks (no hardcoded select-action).
  - Emergent: Only implements universal structural/math primitives.
  - Transparent: Native Python tracebacks for logical unifications.
"""

from __future__ import annotations
from pathlib import Path
import math
import sys
from typing import Any

EMPTY = ()

def is_var(a) -> bool:
    return isinstance(a, str) and len(a) > 1 and a[0] == '$'

def is_space_ref(a) -> bool:
    return isinstance(a, str) and len(a) > 1 and a[0] == '&'

def is_cons(e) -> bool:
    return isinstance(e, tuple) and len(e) >= 3 and e[-2] == '.'

def atom_to_str(atom, max_depth=6) -> str:
    if max_depth <= 0: return "..."
    if atom is True: return "True"
    if atom is False: return "False"
    if isinstance(atom, (int, float)): return str(atom)
    if isinstance(atom, str): return atom
    if isinstance(atom, tuple):
        if len(atom) == 0: return "()"
        return f"({' '.join(atom_to_str(e, max_depth - 1) for e in atom)})"
    return repr(atom)

def unwrap(val):
    """Safely unwrap nested 1-element tuples from collapse returns."""
    while isinstance(val, tuple) and len(val) == 1:
        val = val[0]
    return val

# =============================================================================
# TOKENIZER & PARSER
# =============================================================================

def tokenize(source: str) -> list[str]:
    if source and source[0] == '\ufeff': source = source[1:]
    tokens =[]
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c in ' \t\r\n': i += 1
        elif c == ';':
            while i < n and source[i] != '\n': i += 1
        elif c in '()': tokens.append(c); i += 1
        elif c == '"':
            j = i + 1
            while j < n and source[j] != '"':
                if source[j] == '\\': j += 1
                j += 1
            tokens.append(source[i:j+1] if j < n else source[i:j])
            i = j + 1
        elif c == '!' and (i + 1 < n and source[i + 1] in ' \t\r\n('):
            tokens.append('!'); i += 1
        else:
            j = i
            while j < n and source[j] not in ' \t\r\n();':
                if source[j] == '"': break
                j += 1
            if token := source[i:j]: tokens.append(token)
            i = j
    return tokens

def _parse_token(tok: str):
    if tok == 'True': return True
    if tok == 'False': return False
    try: return int(tok)
    except ValueError: pass
    try: return float(tok)
    except ValueError: pass
    return tok

def parse(source: str) -> list:
    tokens = tokenize(source)
    atoms, pos =[], 0

    def _parse_at(p):
        if p >= len(tokens): raise SyntaxError("Unexpected EOF")
        tok = tokens[p]
        if tok == '(':
            elems, p2 =[], p + 1
            while p2 < len(tokens) and tokens[p2] != ')':
                elem, p2 = _parse_at(p2)
                elems.append(elem)
            if p2 >= len(tokens): raise SyntaxError("Missing ')'")
            return tuple(elems), p2 + 1
        if tok == ')': raise SyntaxError("Unexpected ')'")
        if tok == '!':
            inner, p2 = _parse_at(p + 1)
            return ('!', inner), p2
        return _parse_token(tok), p + 1

    while pos < len(tokens):
        atom, pos = _parse_at(pos)
        atoms.append(atom)
    return atoms

# =============================================================================
# UNIFICATION & SPACES
# =============================================================================

def unify(pattern, target, bindings: dict | None = None) -> dict | None:
    if bindings is None: bindings = {}
    
    if is_var(pattern):
        if pattern in bindings:
            return bindings if bindings[pattern] == target else None
        bindings[pattern] = target
        return bindings

    if isinstance(pattern, tuple) and isinstance(target, tuple):
        if is_cons(pattern):
            head_len = len(pattern) - 2
            if len(target) < head_len: return None
            for i in range(head_len):
                if (bindings := unify(pattern[i], target[i], bindings)) is None: return None
            return unify(pattern[-1], target[head_len:], bindings)
        if len(pattern) != len(target): return None
        for p, t in zip(pattern, target):
            if (bindings := unify(p, t, bindings)) is None: return None
        return bindings

    if isinstance(pattern, (int, float)) and isinstance(target, (int, float)):
        if float(pattern) == float(target): return bindings
    elif pattern == target:
        return bindings
    return None

class Space:
    def __init__(self, name: str = ""):
        self.name = name
        self.atoms: list = []
        self._fn_index: dict[str, list[tuple]] = {}
        self._data_index: dict = {}

    def add(self, atom) -> None:
        self.atoms.append(atom)
        if (isinstance(atom, tuple) and len(atom) == 3 and atom[0] == '=' 
            and isinstance(atom[1], tuple) and len(atom[1]) > 0 and isinstance(atom[1][0], str)):
            self._fn_index.setdefault(atom[1][0], []).append((atom[1], atom[2]))
        if isinstance(atom, tuple) and len(atom) > 0:
            self._data_index.setdefault(atom[0],[]).append(atom)

    def remove(self, atom) -> bool:
        try:
            self.atoms.remove(atom)
            if isinstance(atom, tuple) and len(atom) == 3 and atom[0] == '=':
                try: self._fn_index[atom[1][0]].remove((atom[1], atom[2]))
                except (KeyError, ValueError): pass
            if isinstance(atom, tuple) and len(atom) > 0:
                try: self._data_index[atom[0]].remove(atom)
                except (KeyError, ValueError): pass
            return True
        except ValueError:
            return False

    def query(self, pattern) -> list[dict]:
        if isinstance(pattern, tuple) and len(pattern) > 0:
            head = pattern[0]
            if not is_var(head) and head != '.' and head in self._data_index:
                return [b for a in self._data_index[head] if (b := unify(pattern, a, {})) is not None]
        return[b for a in self.atoms if (b := unify(pattern, a, {})) is not None]

# =============================================================================
# ENGINE EVALUATOR
# =============================================================================

class Runtime:
    def __init__(self):
        self.spaces: dict[str, Space] = {'&self': Space('self'), '&state': Space('state'), '&ontology': Space('ontology')}
        self.debug = False
        self.max_depth = 500

    @property
    def code(self) -> Space: return self.spaces['&self']
    @property
    def state(self) -> Space: return self.spaces['&state']

    def sub(self, expr, env: dict):
        if is_var(expr): return env.get(expr, expr)
        if isinstance(expr, tuple): return tuple(self.sub(e, env) for e in expr)
        return expr

    def eval(self, expr, env: dict | None = None, d: int = 0):
        if env is None: env = {}
        if d > self.max_depth: return EMPTY
        if isinstance(expr, (int, float, bool)): return expr
        if isinstance(expr, str): return env.get(expr, expr) if is_var(expr) else expr
        if not isinstance(expr, tuple) or len(expr) == 0: return expr

        head = expr[0]
        if is_var(head) and (resolved := env.get(head, head)) != head:
            return self.eval((resolved,) + expr[1:], env, d + 1)

        if is_cons(expr):
            hl = len(expr) - 2
            heads = tuple(self.eval(expr[i], env, d + 1) for i in range(hl))
            tail = self.eval(expr[-1], env, d + 1)
            return heads + tail if isinstance(tail, tuple) else heads + (tail,)

        # ── SPECIAL FORMS ──
        if head == 'let' and len(expr) == 4:
            val = self.eval(expr[2], env, d + 1)
            if val is EMPTY and isinstance(expr[1], tuple): return EMPTY
            b = unify(expr[1], val, {})
            if b is None and isinstance(expr[1], tuple): return EMPTY
            new_env = dict(env)
            if b is not None: new_env.update(b)
            elif is_var(expr[1]): new_env[expr[1]] = val
            elif expr[1] != val: return EMPTY
            return self.eval(expr[3], new_env, d + 1)

        if head == 'let*' and len(expr) == 3:
            new_env = dict(env)
            for pat, vexpr in (expr[1] if isinstance(expr[1], tuple) else[]):
                val = self.eval(vexpr, new_env, d + 1)
                b = unify(pat, val, {})
                if b is not None: new_env.update(b)
                elif is_var(pat): new_env[pat] = val
                elif pat != val: return EMPTY
            return self.eval(expr[2], new_env, d + 1)

        if head == 'if' and len(expr) >= 3:
            return self.eval(expr[2], env, d+1) if self.eval(expr[1], env, d+1) is True else (self.eval(expr[3], env, d+1) if len(expr) == 4 else EMPTY)

        if head == 'case' and len(expr) == 3:
            val = self.eval(expr[1], env, d + 1)
            branches = expr[2] if isinstance(expr[2], tuple) else ()
            for branch in branches:
                if not isinstance(branch, tuple) or len(branch) != 2:
                    continue
                pat, body = branch
                # () as pattern matches EMPTY
                if pat == () and (val is EMPTY or val == ()):
                    return self.eval(body, env, d + 1)
                b = unify(pat, val, dict(env))
                if b is not None:
                    return self.eval(body, b, d + 1)
            return EMPTY

        if head == 'sequential':
            res = EMPTY
            for e in expr[1:]: res = self.eval(e, env, d+1)
            return res

        if head == 'match' and len(expr) == 4:
            sname = self.eval(expr[1], env, d+1) if is_var(expr[1]) else expr[1]
            results = self.spaces.get(sname, self.code).query(self.sub(expr[2], env))
            if results:
                merged = dict(env)
                merged.update(results[0])
                return self.eval(expr[3], merged, d + 1)
            return EMPTY

        if head == 'collapse' and len(expr) == 2:
            res = []
            if expr[1] and expr[1][0] == 'match' and len(expr[1]) == 4:
                sname = self.eval(expr[1][1], env, d+1) if is_var(expr[1][1]) else expr[1][1]
                for b in self.spaces.get(sname, self.code).query(self.sub(expr[1][2], env)):
                    merged = dict(env); merged.update(b)
                    v = self.eval(expr[1][3], merged, d + 1)
                    if v is not EMPTY and v != 'empty': res.append(v)
            else:
                v = self.eval(expr[1], env, d+1)
                if v is not EMPTY and v != 'empty': res.append(v)
            return tuple(res)

        if head == 'add-atom' and len(expr) == 3:
            sname = expr[1] if isinstance(expr[1], str) else self.eval(expr[1], env, d+1)
            atom = self.eval(expr[2], env, d + 1)
            self.spaces.get(sname, self.code).add(atom)
            return atom

        if head == 'remove-atom' and len(expr) == 3:
            sname = expr[1] if isinstance(expr[1], str) else self.eval(expr[1], env, d+1)
            atom = self.eval(expr[2], env, d + 1)
            self.spaces.get(sname, self.code).remove(atom)
            return atom
            
        if head in ('empty', 'nop'): return EMPTY

        # ── UNIVERSAL MATH PRIMITIVES ──
        if head in {'+', '-', '*', '/', '>', '<', '>=', '<=', '==', 'abs', 'min', 'max', 'pow', '**', '^', 'log', 'ln', 'exp', 'sign', 'round'}:
            args =[unwrap(self.eval(e, env, d + 1)) for e in expr[1:]]
            if head == '==' and len(args) == 2: 
                return float(args[0]) == float(args[1]) if isinstance(args[0], (int,float)) and isinstance(args[1], (int,float)) else args[0] == args[1]
            if not all(isinstance(a, (int, float)) for a in args): pass 
            else:
                if head == '+' and args: return sum(args)
                if head == '*' and args: return math.prod(args)
                if head == '-' and len(args) == 2: return args[0] - args[1]
                if head == '-' and len(args) == 1: return -args[0]
                if head == '/' and len(args) == 2: return args[0] / args[1] if args[1] != 0 else float('inf')
                if head == '>': return args[0] > args[1]
                if head == '<': return args[0] < args[1]
                if head == '>=': return args[0] >= args[1]
                if head == '<=': return args[0] <= args[1]
                if head == 'abs' and len(args) == 1: return abs(args[0])
                if head in ('min', 'max') and args: return min(args) if head == 'min' else max(args)
                if head in ('pow', '**', '^') and len(args) == 2: return math.pow(args[0], args[1])
                if head in ('log', 'ln') and len(args) == 1 and args[0] > 0: return math.log(args[0])
                if head == 'exp' and len(args) == 1: return math.exp(args[0])
                if head == 'sign' and len(args) == 1: return 1.0 if args[0] > 0 else (-1.0 if args[0] < 0 else 0.0)
                if head == 'round' and len(args) == 1: return float(round(args[0]))

        # ── LIST MATH PRIMITIVES ──
        if head in ('sum', 'sum-list', 'average') and len(expr) == 2:
            lst = unwrap(self.eval(expr[1], env, d + 1))
            if isinstance(lst, (int, float)): return float(lst)
            if isinstance(lst, tuple):
                nums =[float(unwrap(x)) for x in lst if isinstance(unwrap(x), (int, float))]
                if head in ('sum', 'sum-list'): return sum(nums)
                if head == 'average': return sum(nums) / len(nums) if nums else 0.0

        # ── LOGIC PRIMITIVES ──
        if head == 'and': return all(self.eval(e, env, d+1) is True for e in expr[1:])
        if head == 'or': return any(self.eval(e, env, d+1) is True for e in expr[1:])
        if head == 'not' and len(expr) == 2: return self.eval(expr[1], env, d+1) is not True

        # ── FUNCTION DISPATCH ──
        ev_expr = tuple(self.eval(e, env, d + 1) for e in expr)
        ev_head = ev_expr[0]

        if isinstance(ev_head, str) and not is_var(ev_head):
            for pat, body in self.code._fn_index.get(ev_head,[]):
                if (b := unify(pat, ev_expr, {})) is not None:
                    return self.eval(body, b, d + 1)

        return ev_expr

    # =========================================================================
    # LOADING & ROUTING
    # =========================================================================

    def load_dagaz(self, base_path: str, verbose: bool = False) -> int:
        modules =[
            'foundations', 'beliefs', 'affect', 'actions', 'safety',
            'structure_learning', 'atom_lifecycle', 'cycle', 'policy_efe',
            'planning', 'action_learning', 'self_model',
            'analogy_blending', 'conversation_model', 'perception',
            'proprioception', 'abduction', 'semantic_primitives',
            'dimensional_primitives', 'semantic_grounding',
            'action_grounding', 'grounding_hypotheses'
        ]
        
        loaded = 0
        base = Path(base_path)
        
        for mod in modules:
            fpath = base / f"{mod}.metta" if (base / f"{mod}.metta").exists() else base / "core" / f"{mod}.metta"
            if fpath.exists():
                source = fpath.read_text(encoding='utf-8')
                for atom in parse(source):
                    if isinstance(atom, tuple) and len(atom) == 2 and atom[0] == '!':
                        inner = atom[1]
                        if not (isinstance(inner, tuple) and inner and inner[0] in ('import!', 'bind!')):
                            try: self.eval(inner)
                            except Exception: pass
                    else:
                        if isinstance(atom, tuple) and len(atom) > 0:
                            head = atom[0]
                            
                            # 1. WORKING MEMORY (&state)
                            if head in {
                                'belief', 'observation', 'action-model', 'action-cost', 
                                'current-action', 'preference', 'viability-bound',
                                'cycle-count', 'error-history', 'metabolic-capital',
                                'error-trace', 'policy-trace', 'unexplained-error-ratio',
                                'perceptual-expansion-needed'
                            }:
                                self.state.add(atom)
                                
                            # 2. IMMUTABLE SEMANTIC MEMORY (&ontology)
                            elif head in {
                                ':', 'contrast', 'entails', 'dimension', 'semantic-primitive'
                            }:
                                self.spaces['&ontology'].add(atom)
                                
                            # 3. PROCEDURAL MEMORY & RULES (&self)
                            else:
                                self.code.add(atom)
                                
                        # 4. CATCH-ALL FOR BARE ATOMS (e.g., strings/ints)
                        else:
                            self.code.add(atom)
                loaded += 1
        return loaded

    def run(self, source: str):
        res = EMPTY
        for atom in parse(source):
            res = self.eval(atom[1]) if isinstance(atom, tuple) and len(atom) == 2 and atom[0] == '!' else atom
        return res

if __name__ == '__main__':
    rt = Runtime()
    print("Dagaz Universal Runtime (type 'quit' to exit)")
    while True:
        try: line = input("metta> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if line in ('quit', 'exit'): break
        if line: print(f"  = {atom_to_str(rt.run(line))}")