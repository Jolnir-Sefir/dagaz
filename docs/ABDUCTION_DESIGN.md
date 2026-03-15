# Abductive Inference â€” Phase 1.6 Design

## Summary

Inverse model activation: when an effect is surprisingly high and a
passive-model links causeâ†’effect, hypothesize the cause. The hypothesis
enters as a low-precision belief that drives emergent verification via
EFE information gain.

**Added**: ~330 lines MeTTa (`abduction.metta`), ~5 config values
(`foundations.metta`), epistemic source tracking, ~1400-line Python
benchmark (`test_abduction.py`).


## The Peircean Triad

| Phase | Mode | Creates | Mechanism |
|-------|------|---------|-----------|
| 1 | Induction | Structure | Co-error â†’ suspicion â†’ passive-model |
| 1.5 | Deduction | Structure | Aâ†’B, Bâ†’C â‡’ Aâ†’C (transitive closure) |
| **1.6** | **Abduction** | **State** | **Observe E, have Câ†’E â‡’ hypothesize C** |

The critical distinction: Induction and Deduction modify the causal
graph (create/extend passive-model atoms). Abduction leaves the graph
intact and modifies *beliefs*. It runs the causal graph backward.


## Design Decisions

### Decision 1: Precision-Weighted Merge, Not Set

**Problem**: The original proposal called `update-belief!` with a
hardcoded value of 0.8. This doesn't exist in the belief API, and
would overwrite existing knowledge.

**Resolution**: Hypotheses merge into existing beliefs using the same
precision-weighted formula as observation updates:

```
new_val = old_val + lr Ã— (hyp_prec / (belief_prec + hyp_prec)) Ã— (hyp_val - old_val)
```

With `hyp_prec = 0.10` and typical `belief_prec = 0.50`:
- Mixing weight = 0.10 / 0.60 = 0.167
- Effective influence = 0.12 Ã— 0.167 â‰ˆ **2% per cycle**

This is deliberately weak. A single abductive inference barely moves
the needle. Consistent evidence over 5+ cycles moves the belief
meaningfully. The system earns its confidence.


### Decision 2: Computed Hypothesis Values

**Problem**: Hardcoded `hyp_val = 0.8` ignores the evidence.

**Resolution**: Invert the forward model:

```
Forward:   cause_val Ã— weight â‰ˆ effect_delta
Inverse:   hyp_val = clamp(obs_effect / weight, 0, 1)
```

For inhibitory links, the direction flips:
- Excitatory causeâ†’effect + high effect â†’ cause is HIGH
- Inhibitory causeâ†’effect + high effect â†’ cause is LOW

This makes hypotheses *quantitatively* grounded in the evidence, not
just qualitatively.


### Decision 3: Epistemic Source Tracking

**Problem**: The honesty principle requires the system to distinguish
"I measured this" from "I guessed this." Without markers, all beliefs
look the same.

**Resolution**: Lightweight markers:

```metta
(belief-source fire abduced 42)     ; "I hypothesized this at cycle 42"
(belief-source smoke observed 41)   ; "I observed this at cycle 41"
```

Rules:
- Observations supersede hypotheses (when real data arrives, upgrade)
- Source markers don't affect EFE (beliefs are uniform in computation)
- Self-model can query: `(is-abduced? fire)` â†’ True
- Self-report: "I believe there's fire, but that's a hypothesis"


### Decision 4: Single-Step Propagation Per Cycle

**Problem**: If Aâ†’Bâ†’Câ†’D and D spikes, should abduction cascade
Aâ†Bâ†Câ†D in one cycle?

**Resolution**: No. Abduction hypothesizes immediate causes only.
Multi-hop reasoning emerges over multiple cycles:

- Cycle N: D spikes â†’ hypothesize C
- Cycle N+1: C's new belief creates prediction error â†’ hypothesize B
- Cycle N+2: B's new belief â†’ hypothesize A

This is:
- **Transparent**: Each step is independently traceable
- **Rate-limited**: No runaway cascades in a single cycle
- **Debuggable**: Watch the hypothesis ripple backward
- **Realistic**: Deeper inferences take more time

The budget cap (5 hypotheses/cycle) provides a hard bound even for
wide causal graphs.


### Decision 5: Placement After Suspicion, Before Action Selection

```
2b. Compute surprise, record error traces
4b-a. Update suspicion links (uses raw errors)      â† honest errors
4b-b. Phase 1 triggers (causal links)
4b-c. Phase 1.5 triggers (deduction)
4b-d. Phase 2 triggers (hub detection)
4b-e. Metabolic pruning
4b-f. ABDUCTIVE INFERENCE (Phase 1.6)               â† HERE
5.   Action selection (uses beliefs including hypotheses)  â† enriched
```

This ensures:
- Structure learning sees honest errors (pre-abduction)
- Suspicion accumulation isn't contaminated by hypothetical beliefs
- Action selection sees the enriched belief state
- The Sherlock Holmes effect works: hypotheses influence EFE


### Decision 6: Abduction Requires Metabolic Fitness

A passive model must have sufficient metabolic energy to be used for
abduction. This means:
- Only empirically validated links drive hypotheses
- Dying/unproven links don't produce noise
- The same selection pressure that governs structure governs inference

Uses the same `abductive-min-energy` threshold as deduction (0.5).


## The Sherlock Holmes Effect (Emergent Verification)

This is the primary payoff. The sequence:

1. Agent observes `smoke = 0.9` (high surprise)
2. Passive model `(passive-model fire smoke 0 0.8 excitatory)` exists
3. Abduction: hypothesize `fire = 0.9/0.8 = 1.0` (clamped), precision 0.10
4. Fire belief: value shifts slightly toward 1.0, precision *drops*
5. EFE for `observe(heat)`:
   - Fire predicts heat (via `passive-model fire heat`)
   - Fire has LOW precision â†’ observing heat has HIGH info gain
   - `observe(heat)` gets low EFE â†’ selected
6. Agent checks for heat â€” **emergent verification behavior**

No "if smoke then check heat" rule exists anywhere. The behavior
emerges from:
- Abduction (creates low-precision hypothesis)
- EFE (rewards info gain on uncertain beliefs)
- Causal graph (connects fire to heat)

This is the architecture working as designed: one invariant (minimize
EFE), uniform treatment of all observables, emergent behavior.


## The Explaining-Away Limitation

Full explaining-away (Aâ†’E confirmed â‡’ reduce belief in B where Bâ†’E)
requires computing posterior odds ratios over competing explanations.
This is expensive and brings us toward full Bayesian network inference.

Instead, we do **parallel abduction with metabolic selection**:
- Both A and B get hypothesized with low precision
- Subsequent observations confirm A's other predictions but not B's
- B's hypothesis doesn't get reinforced; precision stays low
- Over cycles, A's belief gains confidence while B's stagnates

This is coarser than Bayesian explaining-away but reaches the same
equilibrium given enough cycles. It's architecturally consistent with
the metabolic philosophy: hypotheses that don't pay rent fade.


## Integration Changes

### `foundations.metta` (~5 new config lines)

```metta
; --- Abductive Inference ---
(config abductive-surprise-threshold 0.05)
(config abductive-precision 0.10)
(config abductive-min-link-weight 0.15)
(config abductive-min-energy 0.5)
(config abductive-max-cause-precision 0.50)
(config abductive-budget-per-cycle 5)
```

### `cycle.metta` â€” Integration point

In `cognitive-cycle!` and variants, add after structure learning,
before action selection:

```metta
; 4b-f. ABDUCTIVE INFERENCE (Phase 1.6)
($abduction (abductive-step! $timestamp))
```

In `cognitive-cycle-v2!`, add after `structure-learning-step-v2!`:

```metta
; Phase 1.6: Abductive inference (state from structure)
($abduction (abductive-step! $timestamp))
```

### `loader.metta` â€” Import

```metta
; After structure_learning, before cycle:
!(import! &self core/abduction)
```

### `self_model.metta` â€” Introspection

```metta
; Abductive self-knowledge
(= (abductive-hypotheses)
   (collapse (match &self (belief-source $obs abduced $c) $obs)))

(= (hypothesis-count)
   (length (abductive-hypotheses)))
```

### Files NOT Modified

| File | Why |
|------|-----|
| `beliefs.metta` | Abduction uses existing `set-belief!` mechanism with computed values. No new API needed. |
| `actions.metta` | Abduction doesn't create new actions or modify EFE. The info gain from low-precision hypotheses is already computed by existing EFE machinery. |
| `policy_efe.metta` | No interaction. Abduction enriches beliefs; policy EFE reads beliefs. |
| `structure_learning.metta` | Abduction reads `passive-model` atoms but doesn't create or modify them. Clean read-only dependency. |
| `affect.metta` | Affect is derived from prediction errors. Abduction changes beliefs, which changes errors next cycle, which changes affect. The propagation is automatic. |
| `safety.metta` | Abduction only modifies beliefs (stratum 3/learned). Constitutional constraints are untouched. |


## Benchmark Scenarios (3 cases)

### Scenario 1: Confirmation ("Hidden Cause")

Setup: Latent L causes A and B. Both links weight 0.8.
- Inject A = 0.9 (high surprise)
- Expect: L hypothesized high, observe(B) selected for info gain
- Inject B = 0.9 (confirms L)
- Expect: L's precision increases over subsequent cycles

### Scenario 2: Falsification ("Wrong Guess")

Setup: Aâ†’E (excitatory, 0.8), Bâ†’E (excitatory, 0.3).
Ground truth: A caused E, not B.
- Inject E = 0.9
- Expect: Both A and B hypothesized (A more strongly due to higher weight)
- Inject evidence confirming A, contradicting B
- Expect: A's hypothesis strengthens, B's hypothesis fails

### Scenario 3: Competing Explanations with Asymmetric Evidence

Setup: Aâ†’E (0.8), Bâ†’E (0.3), Aâ†’F (0.7). F is the discriminating observable.
- Inject E = 0.9 â†’ both A and B hypothesized
- Inject F = 0.85 â†’ confirms A (A predicts F), doesn't help B
- Expect: A's belief gains precision, B's stagnates
- Expect: System chose observe(F) because A predicted F with low precision

This tests the full Sherlock Holmes effect: the system identifies the
discriminating observation and checks it.


## Known Limitations

1. **Graph-constrained**: Can only hypothesize causes in the existing
   causal graph. If the true cause has no passive-model link, abduction
   can't find it. (Phase 2 latent variable invention partially addresses
   this for hidden common causes.)

2. **No competitive explaining-away**: Parallel abduction, not
   competitive. Converges to the right answer but slower than exact
   Bayesian inference.

3. **Single-step per cycle**: Deep causal chains take multiple cycles
   to propagate backward. This is a feature (transparency, rate-limiting)
   but means the system is slower to form deep hypotheses.

4. **Precision coupling**: Abduction slightly decreases cause precision
   (-0.02 per hypothesis). In a dense causal graph where many effects
   abduct the same cause, precision could drop more than intended.
   The budget cap mitigates this.

5. **Observation-supersession timing**: If an observation and abduction
   arrive in the same cycle for the same observable, the ordering within
   the cycle determines which "wins." Currently, abduction runs before
   belief updates, so the observation update (which runs after) naturally
   overrides. This is correct but worth documenting.
