# Trie Crash Investigation — Findings

## Summary

The Hyperon trie crash at `trie.rs:179` is triggered by **adding data atoms
to a GroundingSpace whose trie index has already accumulated a critical mass
of expressions, then querying the newly-inserted symbol**. The crash is not
about file count, module count, symbol collisions, or total atom count. It
is about modifying a trie that has grown past a stability threshold.

## The Bug (Hyperon Side)

### Precise Trigger

```
1. Load ≥1,900 expressions into a GroundingSpace (function defs, types, data)
2. Add a new data atom: (belief test-obs 0.5 0.5)
3. Query it: !(match &self (belief $o $v $p) $o)
4. CRASH: trie.rs:179 — unwrap() on None
```

### Characterization (from isolated test battery)

| Test | Result | Meaning |
|------|--------|---------|
| 10 modules (3,210 exprs), no data atoms, probe belief | **PASS** | Trie builds correctly during loading |
| 10 modules + 2 belief atoms, probe belief | **CRASH** | Data atoms after threshold corrupt trie |
| 10 modules + 1 belief atom, probe belief | **CRASH** | Single atom suffices |
| 2 modules (466 exprs) + belief atom | **PASS** | Below threshold — no corruption |
| 5 modules (1,025 exprs) + belief atom | **PASS** | Still below threshold |
| 6 modules (1,900 exprs) + belief atom | **CRASH** | **Threshold is between 1,025–1,900** |
| 10 modules + (xbelief ...) fresh symbol, probe xbelief | **CRASH** | Not symbol-specific — any new atom |
| 10 modules + belief data, probe observation | **PASS** | Corruption is localized to inserted branch |
| 10 modules + belief data, no &ontology | **CRASH** | &ontology is irrelevant |

### Key Properties

1. **Loading is always safe.** All 22 modules (4,733+ expressions) load
   without error in any order. The trie builds correctly during bulk insertion.

2. **The crash requires post-load mutation.** The trie only corrupts when
   `add-atom` is called AFTER the initial loading phase. Static loading
   works; dynamic insertion doesn't.

3. **The crash is size-dependent, not content-dependent.** Any symbol
   crashes (even `xbelief`, which appears nowhere in any module). The
   threshold is ~1,025–1,900 cumulative expressions.

4. **Corruption is branch-local.** After adding `(belief ...)` atoms, a
   `match` on `(observation ...)` still succeeds. The corrupted node is
   on the newly-inserted symbol's trie branch, not a global corruption.

5. **The threshold is in the trie structure, not atom count.** It's
   specifically about the trie's internal node layout after indexing
   ~1,000+ expressions. Something in the rebalancing or node-splitting
   logic during post-load insertion creates a dangling reference.

### Minimal Reproduction

```python
from hyperon import MeTTa
import re

metta = MeTTa()

# Load 6 modules (foundations through structure_learning)
for mod in ["foundations", "beliefs", "affect", "actions",
            "safety", "structure_learning"]:
    with open(f"core/{mod}.metta", encoding="utf-8") as f:
        code = f.read()
    code = re.sub(r'!\(import!\s+[^)]*\)', '', code)
    metta.run(code)

# This succeeds (no belief data atoms exist):
metta.run("!(match &self (belief $o $v $p) $o)")  # → [[]]

# Add a data atom:
metta.run("(belief test-obs 0.5 0.5)")

# This crashes at trie.rs:179:
metta.run("!(match &self (belief $o $v $p) $o)")  # → PANIC
```

## Why This Blocks Dagaz

The cognitive loop's fundamental operation is:

```
perceive → update beliefs → compute errors → affect → EFE → act → learn → repeat
```

Every cycle MUST:
- `add-atom &self (belief $obs $new-val $new-prec)` — update beliefs
- `remove-atom &self (belief $obs $old-val $old-prec)` — clear stale beliefs
- `add-atom &self (observation $obs $val $prec $time)` — record percepts
- `add-atom &self (suspicion-link ...)` — structure learning
- `add-atom &self (passive-model ...)` — causal discovery

There are **43 distinct symbol heads** used as runtime state atoms that get
added to and queried from `&self` during operation. Every one of these
`add-atom` calls is a potential trie crash once the space exceeds ~1,000
expressions — and Dagaz needs ~3,000+ expressions just for its function
definitions.

## The Dagaz-Side Fix: Three-Space Architecture

The insight: **`&self` should be a static code space, never modified after
loading.** All runtime state moves to a separate `&state` space that stays
small and never hits the trie threshold.

### Architecture

```
&self       = CODE (static after loading)
              Function definitions, type declarations.
              NEVER modified by add-atom/remove-atom during operation.
              ~3,000+ expressions — above trie mutation threshold, but
              that's fine because it's never mutated.

&state      = RUNTIME STATE (small, dynamic)
              Beliefs, observations, action-models, suspicion-links,
              metabolic energy, cycle count, error history, etc.
              ~50-200 atoms at any time — far below trie threshold.
              add-atom and remove-atom operate here safely.

&ontology   = SEMANTIC KNOWLEDGE (static reference data)
              Primitives, scales, grounding chains.
              ~328 atoms, never modified.
```

### What Moves to &state

All 43 runtime state symbols. The primary categories:

**Epistemic state** (beliefs, observations):
- `belief`, `observation`, `preference`, `viability-bound`

**Action state** (models, selection):
- `action-model`, `action-cost`, `action-stats`, `current-action`
- `pre-action-belief`, `last-action-taken`

**Structure learning state** (causal discovery):
- `suspicion-link`, `passive-model`, `latent-variable`, `latent-member`
- `metabolic-energy`, `lifecycle-energy`, `created-at`
- `prediction-stats`, `deduction-origin`, `deduction-source`
- `structural-atoms-this-cycle`, `latent-var-counter`
- `perceptual-expansion-needed`, `unexplained-error-ratio`

**Cycle state** (bookkeeping):
- `cycle-count`, `error-history`, `metabolic-capital`
- `demotion-log`, `policy-trace`

**Safety state**:
- `system-status`, `constitutional-atom-count`, `halt-reason`
- `goal-modification-log`

**Conversation state**:
- `conversation-turn`, `current-topics`, `partner-belief`
- `partner-goal-belief`

**Perception state**:
- `perception-log`, `proprioceptive-log`, `unresolved-topic-log`

**Planning state**:
- `fractal-pruned`, `fractal-pruned-at`

**Lifecycle state**:
- `chunked-action`, `chunked-action-policy`

### The Code Change

Every `match &self (belief ...)` in function definitions becomes
`match &state (belief ...)`. Every `add-atom &self (belief ...)`
becomes `add-atom &state (belief ...)`. Same for all 43 state symbols.

This is a mechanical transformation:
- `match &self (<state-symbol> ...)` → `match &state (<state-symbol> ...)`
- `add-atom &self (<state-symbol> ...)` → `add-atom &state (<state-symbol> ...)`
- `remove-atom &self (<state-symbol> ...)` → `remove-atom &state (<state-symbol> ...)`
- `collapse (match &self (<state-symbol> ...)...)` → `collapse (match &state ...)`

Functions stay in `&self`. They just reach into `&state` for data.

### Why This Works

1. **`&self`'s trie is never mutated after loading.** The bug requires
   post-load `add-atom` to a large trie. If `&self` is static, the bug
   never triggers.

2. **`&state`'s trie stays small.** Even with all runtime state, the
   `&state` space has ~50-200 atoms at any time. Well below the ~1,025
   expression threshold. `add-atom` and `remove-atom` work fine.

3. **Cross-space queries are verified working.** The earlier multi-space
   performance tests confirmed `match &state` from functions in `&self`
   works at 3,530 Hz — negligible overhead.

4. **Architecturally cleaner.** This is the biological distinction between
   long-term memory (code knowledge) and working memory (current state).
   It's what `PERFORMANCE_OPTIMIZATION.md` was reaching toward, but
   applied to the correct boundary.

### Previous Multi-Space Attempts and Why They Failed

The earlier multi-space work focused on moving ontological DATA from `&self`
to `&ontology`. This reduced `&self`'s atom count but didn't prevent the crash
because:

1. We still loaded all function definitions into `&self` (3,000+ expressions)
2. The cognitive loop still called `add-atom &self (belief ...)` during
   operation, hitting the mutation-after-threshold crash

The fix isn't about moving static data OUT of `&self`. It's about moving
dynamic state OUT — ensuring `&self` is never mutated after initial loading.

### Relationship to Cons-Cell Bug

The cons-cell bug (`$head . $rest` not reducing) is independent. The
three-space fix addresses the trie crash. The cons-cell patches (replacing
paired empty/cons definitions with `if-decons-expr` single definitions)
address evaluation correctness. Both are needed for end-to-end operation.

However, the cons-cell patches have a secondary benefit for the trie:
collapsing paired definitions into single definitions reduces expression
count by ~17 in the first 6 modules. This provides marginal headroom
but does not solve the fundamental problem (runtime mutation of a large
trie).

## What We Now Know Was Wrong

1. **"The crash is about file count."** — Wrong. It's about trie mutation
   after a size threshold. File count is a proxy for expression count, which
   is a proxy for trie size, but the actual trigger is add-atom + match.

2. **"Moving data to &ontology fixes it."** — Insufficient. Removing 328
   data atoms from &self doesn't help when the cognitive loop adds new
   atoms every cycle. The fix must prevent ALL runtime mutations to &self.

3. **"Merging files into fewer modules helps."** — Irrelevant. Merging
   doesn't reduce expression count. 15 files with 4,733 expressions is
   the same trie as 22 files with 4,733 expressions.

4. **"The threshold is 15-16 files."** — Misleading coincidence. The
   earlier observation that "15 files stable, 16 crashes" was correct
   but the causal model was wrong. File 16 (semantic_primitives) happened
   to contain belief data atoms that triggered the mutation crash. Any
   module that seeds data atoms would have crashed, regardless of file count.

## Implementation Priority

1. **&state space separation** — The Dagaz-side fix. Mechanical refactor of
   match/add-atom/remove-atom targets. Unblocks the cognitive loop.

2. **Cons-cell patches** — if-decons-expr rewrites. Required for correct
   evaluation of recursive list functions.

3. **Bug report to Hyperon** — The minimal reproduction case is clean.
   Whether or not the &state fix resolves it for Dagaz, the trie mutation
   bug is real and affects any MeTTa program that dynamically adds atoms
   to a large space.

4. **&ontology for semantic data** — Still architecturally valuable, but
   not required for the trie crash fix. Can be done after &state works.
