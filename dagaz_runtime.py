"""
dagaz_runtime.py — Pure Python MeTTa Evaluator for Project Dagaz.

DESIGN:
  One invariant: evaluation produces a SET of results.
  Deterministic evaluation is the special case |set| = 1.

  Two sources of nondeterminism:
    1. Function dispatch — multiple definitions with the same pattern
    2. Match — a pattern against a space can bind in multiple ways

  Everything else propagates nondeterminism but does not create it.
  Side effects (add-atom, remove-atom) are linear — execute once.

  No domain-specific logic. Only universal MeTTa primitives.
"""

from __future__ import annotations
from pathlib import Path
import math
import sys
import itertools
from typing import Any

EMPTY = ()

# =============================================================================
# PREDICATES
# =============================================================================

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
        return "({})".format(' '.join(atom_to_str(e, max_depth - 1) for e in atom))
    return repr(atom)

def unwrap(val):
    while isinstance(val, tuple) and len(val) == 1:
        val = val[0]
    return val

# =============================================================================
# TOKENIZER & PARSER
# =============================================================================

def tokenize(source: str) -> list[str]:
    if source and source[0] == '\ufeff':
        source = source[1:]
    tokens = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c in ' \t\r\n':
            i += 1
        elif c == ';':
            while i < n and source[i] != '\n': i += 1
        elif c in '()':
            tokens.append(c); i += 1
        elif c == '"':
            j = i + 1
            while j < n and source[j] != '"':
                if source[j] == '\\': j += 1
                j += 1
            tokens.append(source[i:j + 1] if j < n else source[i:j])
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
    atoms, pos = [], 0
    def _parse_at(p):
        if p >= len(tokens): raise SyntaxError("Unexpected EOF")
        tok = tokens[p]
        if tok == '(':
            elems, p2 = [], p + 1
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
# UNIFICATION
# =============================================================================

def unify(pattern, target, bindings: dict | None = None) -> dict | None:
    if bindings is None: bindings = {}

    # Inlined is_var check (was 900K function calls)
    if isinstance(pattern, str):
        if len(pattern) > 1 and pattern[0] == '$':
            if pattern in bindings:
                return bindings if bindings[pattern] == target else None
            bindings[pattern] = target
            return bindings
        # Non-variable string: direct equality
        return bindings if pattern == target else None

    if isinstance(pattern, tuple):
        if not isinstance(target, tuple):
            return None
        plen = len(pattern)
        # Inlined is_cons check
        if plen >= 3 and pattern[-2] == '.':
            head_len = plen - 2
            if len(target) < head_len: return None
            for i in range(head_len):
                if (bindings := unify(pattern[i], target[i], bindings)) is None:
                    return None
            return unify(pattern[-1], target[head_len:], bindings)
        # Fixed-length tuple
        if plen != len(target): return None
        for p, t in zip(pattern, target):
            if (bindings := unify(p, t, bindings)) is None: return None
        return bindings

    if isinstance(pattern, (int, float)) and isinstance(target, (int, float)):
        if float(pattern) == float(target): return bindings
    elif pattern == target:
        return bindings
    return None

# =============================================================================
# SPACE
# =============================================================================

class Space:
    """Atom space with compound-key indexing.

    Three index levels:
      _data_index[head]                → all atoms with this head symbol
      _key2_index[(head, field1)]      → atoms where first 2 elements match
      _key3_index[(head, f1, f2)]      → atoms where first 3 elements match

    Query selects the deepest applicable index, reducing unification
    candidates from O(all atoms with head) to O(1) for exact lookups.
    """
    def __init__(self, name: str = ""):
        self.name = name
        self.atoms: list = []
        self._fn_index: dict[str, list[tuple]] = {}
        self._data_index: dict = {}
        self._key2_index: dict = {}
        self._key3_index: dict = {}

    def __len__(self):
        return len(self.atoms)

    @staticmethod
    def _is_concrete(v):
        """True if v is a concrete value (not a variable)."""
        return not (isinstance(v, str) and len(v) > 1 and v[0] == '$')

    def add(self, atom) -> None:
        self.atoms.append(atom)
        if isinstance(atom, tuple) and len(atom) == 3 and atom[0] == '=':
            head = atom[1]
            if isinstance(head, tuple) and len(head) > 0 and isinstance(head[0], str):
                self._fn_index.setdefault(head[0], []).append((head, atom[2]))
            elif isinstance(head, str):
                self._fn_index.setdefault(head, []).append(((head,), atom[2]))
        if isinstance(atom, tuple) and len(atom) > 0:
            h = atom[0]
            self._data_index.setdefault(h, []).append(atom)
            if len(atom) >= 2 and self._is_concrete(atom[1]):
                self._key2_index.setdefault((h, atom[1]), []).append(atom)
                if len(atom) >= 3 and self._is_concrete(atom[2]):
                    self._key3_index.setdefault((h, atom[1], atom[2]), []).append(atom)

    def remove(self, atom) -> bool:
        try:
            self.atoms.remove(atom)
        except ValueError:
            return False
        if isinstance(atom, tuple) and len(atom) == 3 and atom[0] == '=':
            head = atom[1]
            if isinstance(head, tuple) and len(head) > 0 and isinstance(head[0], str):
                try: self._fn_index[head[0]].remove((head, atom[2]))
                except (KeyError, ValueError): pass
            elif isinstance(head, str):
                try: self._fn_index[head].remove(((head,), atom[2]))
                except (KeyError, ValueError): pass
        if isinstance(atom, tuple) and len(atom) > 0:
            h = atom[0]
            try: self._data_index[h].remove(atom)
            except (KeyError, ValueError): pass
            if len(atom) >= 2 and self._is_concrete(atom[1]):
                try: self._key2_index[(h, atom[1])].remove(atom)
                except (KeyError, ValueError): pass
                if len(atom) >= 3 and self._is_concrete(atom[2]):
                    try: self._key3_index[(h, atom[1], atom[2])].remove(atom)
                    except (KeyError, ValueError): pass
        return True

    def query(self, pattern) -> list[dict]:
        if not isinstance(pattern, tuple) or len(pattern) == 0:
            return [b for a in self.atoms
                    if (b := unify(pattern, a, {})) is not None]

        h = pattern[0]
        # Variable head — must scan everything
        if isinstance(h, str) and len(h) > 1 and h[0] == '$':
            return [b for a in self.atoms
                    if (b := unify(pattern, a, {})) is not None]

        # Try deepest applicable compound index
        if len(pattern) >= 3:
            p1, p2 = pattern[1], pattern[2]
            if self._is_concrete(p1) and self._is_concrete(p2):
                key3 = (h, p1, p2)
                if key3 in self._key3_index:
                    return [b for a in self._key3_index[key3]
                            if (b := unify(pattern, a, {})) is not None]
                return []  # no matches possible

        if len(pattern) >= 2:
            p1 = pattern[1]
            if self._is_concrete(p1):
                key2 = (h, p1)
                if key2 in self._key2_index:
                    return [b for a in self._key2_index[key2]
                            if (b := unify(pattern, a, {})) is not None]
                return []

        # Fall back to head-only index
        if h in self._data_index:
            return [b for a in self._data_index[h]
                    if (b := unify(pattern, a, {})) is not None]
        return []

    def fn_defs(self, name: str) -> list[tuple]:
        return list(self._fn_index.get(name, []))

# =============================================================================
# MATH DISPATCH TABLE
# =============================================================================

def _safe_div(a, b): return a / b if b != 0 else float('inf')
def _sign(x): return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

_MATH_OPS = {
    '+':     (1, None, lambda a: sum(a)),
    '-':     (1, 2,    lambda a: -a[0] if len(a) == 1 else a[0] - a[1]),
    '*':     (1, None, lambda a: math.prod(a)),
    '/':     (2, 2,    lambda a: _safe_div(a[0], a[1])),
    '>':     (2, 2,    lambda a: a[0] > a[1]),
    '<':     (2, 2,    lambda a: a[0] < a[1]),
    '>=':    (2, 2,    lambda a: a[0] >= a[1]),
    '<=':    (2, 2,    lambda a: a[0] <= a[1]),
    'abs':   (1, 1,    lambda a: abs(a[0])),
    'min':   (1, None, lambda a: min(a)),
    'max':   (1, None, lambda a: max(a)),
    'pow':   (2, 2,    lambda a: math.pow(a[0], a[1])),
    '**':    (2, 2,    lambda a: math.pow(a[0], a[1])),
    '^':     (2, 2,    lambda a: math.pow(a[0], a[1])),
    'log':   (1, 1,    lambda a: math.log(a[0]) if a[0] > 0 else float('-inf')),
    'ln':    (1, 1,    lambda a: math.log(a[0]) if a[0] > 0 else float('-inf')),
    'exp':   (1, 1,    lambda a: math.exp(a[0])),
    'sign':  (1, 1,    lambda a: _sign(a[0])),
    'round': (1, 1,    lambda a: float(round(a[0]))),
}
_MATH_HEADS = frozenset(_MATH_OPS.keys())

# =============================================================================
# EVALUATOR
# =============================================================================

class Runtime:
    def __init__(self):
        self.spaces: dict[str, Space] = {
            '&self': Space('self'), '&state': Space('state'),
            '&ontology': Space('ontology'),
        }
        self.debug = False
        self.max_depth = 500

    @property
    def code(self) -> Space: return self.spaces['&self']
    @property
    def state(self) -> Space: return self.spaces['&state']
    @property
    def state_space_name(self) -> str: return '&state'

    def get_space(self, name: str) -> Space:
        if name not in self.spaces:
            self.spaces[name] = Space(name)
        return self.spaces[name]

    def sub(self, expr, env: dict):
        if is_var(expr): return env.get(expr, expr)
        if isinstance(expr, tuple): return tuple(self.sub(e, env) for e in expr)
        return expr

    # -----------------------------------------------------------------
    # SHARED BINDING
    # -----------------------------------------------------------------

    def _bind(self, pat, val, env):
        if val is EMPTY and isinstance(pat, tuple): return None
        b = unify(pat, val, {})
        if b is not None:
            new_env = dict(env); new_env.update(b); return new_env
        if is_var(pat):
            new_env = dict(env); new_env[pat] = val; return new_env
        if pat == val: return dict(env)
        return None

    # -----------------------------------------------------------------
    # DETERMINISTIC EVAL — returns single result
    # -----------------------------------------------------------------

    def eval(self, expr, env: dict | None = None, d: int = 0):
        if env is None: env = {}
        if d > self.max_depth: return EMPTY
        if isinstance(expr, (int, float, bool)): return expr
        if isinstance(expr, str):
            return env.get(expr, expr) if is_var(expr) else expr
        if not isinstance(expr, tuple) or len(expr) == 0: return expr

        head = expr[0]
        if is_var(head):
            resolved = env.get(head, head)
            if resolved != head:
                return self.eval((resolved,) + expr[1:], env, d + 1)

        if is_cons(expr):
            hl = len(expr) - 2
            heads = tuple(self.eval(expr[i], env, d + 1) for i in range(hl))
            tail = self.eval(expr[-1], env, d + 1)
            return heads + tail if isinstance(tail, tuple) else heads + (tail,)

        if head == 'let' and len(expr) == 4:
            val = self.eval(expr[2], env, d + 1)
            new_env = self._bind(expr[1], val, env)
            return self.eval(expr[3], new_env, d + 1) if new_env else EMPTY

        if head == 'let*' and len(expr) == 3:
            new_env = dict(env)
            for pat, vexpr in (expr[1] if isinstance(expr[1], tuple) else []):
                val = self.eval(vexpr, new_env, d + 1)
                bound = self._bind(pat, val, new_env)
                if bound is None: return EMPTY
                new_env = bound
            return self.eval(expr[2], new_env, d + 1)

        if head == 'if' and len(expr) >= 3:
            cond = self.eval(expr[1], env, d + 1)
            if cond is True: return self.eval(expr[2], env, d + 1)
            return self.eval(expr[3], env, d + 1) if len(expr) == 4 else EMPTY

        if head == 'case' and len(expr) == 3:
            val = self.eval(expr[1], env, d + 1)
            for branch in (expr[2] if isinstance(expr[2], tuple) else ()):
                if not isinstance(branch, tuple) or len(branch) != 2: continue
                pat, body = branch
                if pat == () and (val is EMPTY or val == ()):
                    return self.eval(body, env, d + 1)
                b = unify(pat, val, dict(env))
                if b is not None: return self.eval(body, b, d + 1)
            return EMPTY

        if head == 'sequential':
            res = EMPTY
            for e in expr[1:]: res = self.eval(e, env, d + 1)
            return res

        if head == 'match' and len(expr) == 4:
            sname = expr[1] if isinstance(expr[1], str) else self.eval(expr[1], env, d + 1)
            results = self.spaces.get(sname, self.code).query(self.sub(expr[2], env))
            if results:
                merged = dict(env); merged.update(results[0])
                return self.eval(expr[3], merged, d + 1)
            return EMPTY

        if head == 'collapse' and len(expr) == 2:
            return self._eval_collapse(expr[1], env, d)

        if head == 'add-atom' and len(expr) == 3:
            sname = expr[1] if isinstance(expr[1], str) else self.eval(expr[1], env, d + 1)
            atom = self.eval(expr[2], env, d + 1)
            self.spaces.get(sname, self.code).add(atom)
            return atom

        if head == 'remove-atom' and len(expr) == 3:
            sname = expr[1] if isinstance(expr[1], str) else self.eval(expr[1], env, d + 1)
            atom = self.eval(expr[2], env, d + 1)
            self.spaces.get(sname, self.code).remove(atom)
            return atom

        if head in ('empty', 'nop'): return EMPTY

        # ── MATH ──
        if head in _MATH_HEADS:
            return self._eval_math(head, expr[1:], env, d)
        if head == '==' and len(expr) == 3:
            a = unwrap(self.eval(expr[1], env, d + 1))
            b = unwrap(self.eval(expr[2], env, d + 1))
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return float(a) == float(b)
            return a == b
        if head in ('sum', 'sum-list', 'average') and len(expr) == 2:
            return self._eval_list_agg(head, expr[1], env, d)

        # ── LOGIC ──
        if head == 'and': return all(self.eval(e, env, d+1) is True for e in expr[1:])
        if head == 'or': return any(self.eval(e, env, d+1) is True for e in expr[1:])
        if head == 'not' and len(expr) == 2: return self.eval(expr[1], env, d+1) is not True

        # ── FUNCTION DISPATCH (first match) ──
        ev_expr = tuple(self.eval(e, env, d + 1) for e in expr)
        ev_head = ev_expr[0]
        if isinstance(ev_head, str) and not is_var(ev_head):
            for pat, body in self.code._fn_index.get(ev_head, []):
                if (b := unify(pat, ev_expr, {})) is not None:
                    return self.eval(body, b, d + 1)
        return ev_expr

    # -----------------------------------------------------------------
    # NONDETERMINISTIC EVAL — returns list[result]
    # -----------------------------------------------------------------

    def eval_all(self, expr, env: dict | None = None, d: int = 0) -> list:
        if env is None: env = {}
        if d > self.max_depth: return []
        if isinstance(expr, (int, float, bool)): return [expr]
        if isinstance(expr, str):
            return [env.get(expr, expr)] if is_var(expr) else [expr]
        if not isinstance(expr, tuple) or len(expr) == 0: return [expr]

        head = expr[0]
        if is_var(head):
            resolved = env.get(head, head)
            if resolved != head:
                return self.eval_all((resolved,) + expr[1:], env, d + 1)

        if is_cons(expr):
            hl = len(expr) - 2
            parts = [self.eval_all(expr[i], env, d + 1) for i in range(hl)]
            parts.append(self.eval_all(expr[-1], env, d + 1))
            results = []
            for combo in itertools.product(*parts):
                heads = combo[:-1]; tail = combo[-1]
                results.append(heads + tail if isinstance(tail, tuple) else heads + (tail,))
            return results or [EMPTY]

        if head == 'let' and len(expr) == 4:
            results = []
            for val in self.eval_all(expr[2], env, d + 1):
                new_env = self._bind(expr[1], val, env)
                if new_env is not None:
                    results.extend(self.eval_all(expr[3], new_env, d + 1))
            return results or [EMPTY]

        if head == 'let*' and len(expr) == 3:
            bindings = expr[1] if isinstance(expr[1], tuple) else ()
            return self._eval_all_let_star(list(bindings), expr[2], env, d)

        if head == 'if' and len(expr) >= 3:
            cond = self.eval(expr[1], env, d + 1)
            if cond is True: return self.eval_all(expr[2], env, d + 1)
            return self.eval_all(expr[3], env, d + 1) if len(expr) == 4 else [EMPTY]

        if head == 'case' and len(expr) == 3:
            val = self.eval(expr[1], env, d + 1)
            for branch in (expr[2] if isinstance(expr[2], tuple) else ()):
                if not isinstance(branch, tuple) or len(branch) != 2: continue
                pat, body = branch
                if pat == () and (val is EMPTY or val == ()):
                    return self.eval_all(body, env, d + 1)
                b = unify(pat, val, dict(env))
                if b is not None: return self.eval_all(body, b, d + 1)
            return [EMPTY]

        if head == 'sequential':
            for e in expr[1:-1]: self.eval(e, env, d + 1)
            return self.eval_all(expr[-1], env, d + 1) if len(expr) > 1 else [EMPTY]

        if head == 'match' and len(expr) == 4:
            sname = expr[1] if isinstance(expr[1], str) else self.eval(expr[1], env, d + 1)
            results = self.spaces.get(sname, self.code).query(self.sub(expr[2], env))
            if results:
                merged = dict(env); merged.update(results[0])
                return self.eval_all(expr[3], merged, d + 1)
            return [EMPTY]

        if head == 'collapse' and len(expr) == 2:
            inner = expr[1]
            if isinstance(inner, tuple) and inner and inner[0] == 'match' and len(inner) == 4:
                sname = inner[1] if isinstance(inner[1], str) else self.eval(inner[1], env, d + 1)
                match_results = []
                for b in self.spaces.get(sname, self.code).query(self.sub(inner[2], env)):
                    merged = dict(env); merged.update(b)
                    for v in self._collapse_body(inner[3], merged, d + 1):
                        if v is not EMPTY and v != 'empty': match_results.append(v)
                return [tuple(match_results)]
            else:
                inner_results = self.eval_all(inner, env, d + 1)
                return [tuple(r for r in inner_results if r is not EMPTY and r != 'empty')]

        # Side effects — linear
        if head == 'add-atom' and len(expr) == 3: return [self.eval(expr, env, d)]
        if head == 'remove-atom' and len(expr) == 3: return [self.eval(expr, env, d)]
        if head in ('empty', 'nop'): return [EMPTY]

        # Deterministic primitives — delegate
        if head in _MATH_HEADS or head == '==': return [self.eval(expr, env, d)]
        if head in ('sum', 'sum-list', 'average'): return [self.eval(expr, env, d)]
        if head in ('and', 'or', 'not'): return [self.eval(expr, env, d)]

        # ── NONDETERMINISTIC FUNCTION DISPATCH ──
        sub_results = [self.eval_all(e, env, d + 1) for e in expr]
        all_results = []
        for combo in itertools.product(*sub_results):
            ev_expr = tuple(combo)
            ev_head = ev_expr[0]
            if isinstance(ev_head, str) and not is_var(ev_head):
                exact_results = []
                first_variable = None
                for pat, body in self.code._fn_index.get(ev_head, []):
                    b = unify(pat, ev_expr, {})
                    if b is not None:
                        if not b:
                            exact_results.extend(self.eval_all(body, b, d + 1))
                        elif first_variable is None:
                            first_variable = (body, b)
                if exact_results:
                    all_results.extend(exact_results)
                elif first_variable is not None:
                    body, b = first_variable
                    all_results.extend(self.eval_all(body, b, d + 1))
                else:
                    all_results.append(ev_expr)
            else:
                all_results.append(ev_expr)
        return all_results or [EMPTY]

    def _eval_all_let_star(self, bindings, body, env, d):
        if not bindings: return self.eval_all(body, env, d + 1)
        pat, vexpr = bindings[0]
        results = []
        for val in self.eval_all(vexpr, env, d + 1):
            new_env = self._bind(pat, val, env)
            if new_env is not None:
                results.extend(self._eval_all_let_star(bindings[1:], body, new_env, d))
        return results or [EMPTY]

    # -----------------------------------------------------------------
    # SHARED HELPERS
    # -----------------------------------------------------------------

    def _eval_collapse(self, inner, env, d):
        if isinstance(inner, tuple) and inner and inner[0] == 'match' and len(inner) == 4:
            sname = inner[1] if isinstance(inner[1], str) else self.eval(inner[1], env, d + 1)
            res = []
            for b in self.spaces.get(sname, self.code).query(self.sub(inner[2], env)):
                merged = dict(env); merged.update(b)
                for v in self._collapse_body(inner[3], merged, d + 1):
                    if v is not EMPTY and v != 'empty': res.append(v)
            return tuple(res)
        else:
            results = self.eval_all(inner, env, d + 1)
            return tuple(r for r in results if r is not EMPTY and r != 'empty')

    def _collapse_body(self, expr, env, d):
        """Evaluate a collapse body, recursing into nested match.

        Inside a collapse context, every match must iterate all results —
        not just the first. This recurses to arbitrary depth so that
        patterns like collapse(match(... match(... match(...)))) work.
        """
        if isinstance(expr, tuple) and expr and expr[0] == 'match' and len(expr) == 4:
            sname = expr[1] if isinstance(expr[1], str) else self.eval(expr[1], env, d + 1)
            res = []
            for b in self.spaces.get(sname, self.code).query(self.sub(expr[2], env)):
                merged = dict(env); merged.update(b)
                res.extend(self._collapse_body(expr[3], merged, d + 1))
            return res
        else:
            v = self.eval(expr, env, d)
            return [v]

    def _eval_math(self, op, arg_exprs, env, d):
        args = [unwrap(self.eval(e, env, d + 1)) for e in arg_exprs]
        for i, a in enumerate(args):
            if isinstance(a, str) and not is_var(a):
                for pat, body in self.code._fn_index.get(a, []):
                    if len(pat) == 1:
                        resolved = self.eval(body, {}, d + 1)
                        if isinstance(resolved, (int, float)):
                            args[i] = resolved
                        break
        if not all(isinstance(a, (int, float)) for a in args): return EMPTY
        min_a, max_a, fn = _MATH_OPS[op]
        if len(args) < min_a: return EMPTY
        if max_a is not None and len(args) > max_a: return EMPTY
        return fn(args)

    def _eval_list_agg(self, op, arg_expr, env, d):
        lst = unwrap(self.eval(arg_expr, env, d + 1))
        if isinstance(lst, (int, float)): return float(lst)
        if isinstance(lst, tuple):
            nums = [float(unwrap(x)) for x in lst if isinstance(unwrap(x), (int, float))]
            if op in ('sum', 'sum-list'): return sum(nums)
            if op == 'average': return sum(nums) / len(nums) if nums else 0.0
        return EMPTY

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def run(self, source: str):
        res = EMPTY
        for atom in parse(source):
            if isinstance(atom, tuple) and len(atom) == 2 and atom[0] == '!':
                res = self.eval(atom[1])
            else:
                self.code.add(atom); res = atom
        return res

    def run_expr(self, source: str):
        atoms = parse(source)
        if not atoms: return EMPTY
        expr = atoms[0]
        if isinstance(expr, tuple) and len(expr) == 2 and expr[0] == '!':
            return self.eval(expr[1])
        return self.eval(expr)

    def stats(self) -> dict:
        fn_count = sum(len(defs) for defs in self.code._fn_index.values())
        return {
            'spaces': {name: len(space) for name, space in self.spaces.items()},
            'functions': fn_count,
        }

    # =========================================================================
    # LOADING
    # =========================================================================

    _STATE_HEADS = frozenset({
        'belief', 'observation', 'action-model', 'action-cost',
        'current-action', 'preference', 'viability-bound',
        'cycle-count', 'error-history', 'metabolic-capital',
        'error-trace', 'policy-trace', 'unexplained-error-ratio',
        'perceptual-expansion-needed',
    })

    _ONTOLOGY_HEADS = frozenset({
        ':', 'contrast', 'entails', 'dimension', 'semantic-primitive',
    })

    def load_file(self, fpath: str, verbose: bool = False) -> None:
        source = Path(fpath).read_text(encoding='utf-8')
        if verbose: print("  Loading {} ...".format(Path(fpath).stem))
        for atom in parse(source):
            if isinstance(atom, tuple) and len(atom) == 2 and atom[0] == '!':
                inner = atom[1]
                if isinstance(inner, tuple) and inner and inner[0] in ('import!', 'bind!'):
                    continue
                try: self.eval(inner)
                except Exception: pass
            elif isinstance(atom, tuple) and len(atom) > 0:
                head = atom[0]
                if head in self._STATE_HEADS: self.state.add(atom)
                elif head in self._ONTOLOGY_HEADS: self.spaces['&ontology'].add(atom)
                else: self.code.add(atom)
            else:
                self.code.add(atom)

    def load_dagaz(self, base_path: str, verbose: bool = False) -> int:
        loaded = 0
        for mod in DAGAZ_MODULES:
            fpath = resolve_module_path(base_path, mod)
            if fpath is not None:
                self.load_file(str(fpath), verbose=verbose)
                loaded += 1
        return loaded

# =============================================================================
# MODULE RESOLUTION
# =============================================================================

DAGAZ_MODULES = [
    'foundations', 'beliefs', 'affect', 'actions', 'safety',
    'structure_learning', 'atom_lifecycle', 'cycle', 'policy_efe',
    'planning', 'action_learning', 'self_model',
    'analogy_blending', 'conversation_model', 'perception',
    'proprioception', 'abduction', 'semantic_primitives',
    'dimensional_primitives', 'semantic_grounding',
    'action_grounding', 'grounding_hypotheses',
    'domain',
]

def resolve_module_path(base_path: str, module: str):
    base = Path(base_path)
    for pattern in [
        base / 'core' / '{}.metta'.format(module),
        base / 'core_{}.metta'.format(module),
        base / '{}.metta'.format(module),
    ]:
        if pattern.exists(): return pattern
    return None

# =============================================================================
# REPL
# =============================================================================

if __name__ == '__main__':
    rt = Runtime()
    print("Dagaz MeTTa Runtime (type 'quit' to exit)")
    while True:
        try: line = input("metta> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if line in ('quit', 'exit'): break
        if line: print("  = {}".format(atom_to_str(rt.run(line))))
