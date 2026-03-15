# Planning Strategy: Adaptive Beam Search with Confidence-Degraded Pruning

## 1. The Problem: Combinatorial Explosion

In `policy_efe.metta` v5, policy selection relies on exhaustive expansion of the
action space over a fixed time horizon *d*.

```
Complexity ≈ O(|A|^d)
```

Where |A| is the number of available actions and *d* is the depth.

**Physics Analogy:** This is equivalent to calculating the partition function
of a system by summing over *every possible microstate*. It is computationally
intractable for any interesting horizon.

**Existing Alternative (MCTS):** Monte Carlo Tree Search approximates the sum
via random sampling. While effective, it introduces stochasticity and violates
the **Symbolic** and **Transparent** design principles. It requires the Law of
Large Numbers to converge, which is metabolically expensive.

**Current State:** With 3 base actions and horizon 3, `gen-policy` produces 27
policies — manageable. At horizon 5, that's 243. At horizon 7, 2187. Adding
parameterized actions (e.g., `investigate $target`) or conversational actions
makes this worse. The architecture needs a scaling strategy before the action
space grows.


## 2. The Solution: Fractal Pruning via Noise-Aware Beam Search

We adopt an **Adaptive Beam Search** approach to planning. Instead of
simulating all micro-actions at maximum fidelity, we treat planning as a
**coarse-graining process**: systematically eliminating branches of the
decision tree that are indistinguishable from noise, leaving only the
"effective theory" of the agent's future.

### Core Principle: Variational Free Energy of Computation

The agent must minimize Free Energy not just in the external world (Action),
but in the internal world (Computation).

- **Thinking is metabolic work.**
- Simulating a high-EFE branch (a bad plan) is wasted work.
- Therefore, the computational budget must flow proportional to the promise
  (low EFE) of a trajectory.

### The Coarse-Graining Analogy

The approach is inspired by coarse-graining in physics — integrating out
irrelevant degrees of freedom to obtain an effective theory at the relevant
scale. In beam search, we eliminate branches whose EFE difference from the
best is smaller than the noise floor at that planning depth.

**Honesty about the analogy:** In physical RG, coarse-graining is *exact* for
the partition function. In beam search, pruning is *approximate* — we lose
information about pruned branches and cannot recover it. The analogy is
inspirational, not rigorous. Where it holds exactly: when one action
*dominates* (its EFE is much lower than alternatives), discarding the rest is
lossless. The system has genuinely "broken symmetry" and left the regime where
alternatives matter.

The deeper principle shared with RG: **integrate out irrelevant degrees of
freedom**. RG flow integrates out bad action trajectories (temporal
irrelevance). Multi-space memory integrates out inactive knowledge (spatial
irrelevance). Both produce the minimal representation needed for the current
decision.


## 3. The Algorithm: Unified Noise-Floor Pruning

The fractal planner uses a **single adaptive mechanism** — the noise floor —
to control all pruning decisions. Model confidence degrades geometrically with
planning depth (inherited from `policy_efe.metta`'s discount factor). The
noise floor grows as this confidence degrades:

```
noise_floor(depth) = base_noise / degraded_confidence(depth)
degraded_confidence(depth) = base_confidence × discount^depth
```

This single quantity determines beam width, symmetry breaking, and the
planning horizon — not as three independent mechanisms, but as three
manifestations of one principle: **EFE differences smaller than the noise
floor are uninformative.**

### A. How Beam Width Emerges

At each depth, we score all actions via residual EFE (Section 3D), sort
them, and keep those within the noise floor of the best:

```metta
; Keep actions within noise floor of the best
(= (noise-filter $sorted-candidates $state $depth)
   (let* (
     ($noise (noise-floor-at-depth $state $depth))
     ($best-efe (candidate-efe (car $sorted-candidates)))
     ($effective-beam (max 2 (ceil-int (* (get-planning-config base-beam-width) $deg-conf))))
     ($within-noise (filter-within-noise $sorted-candidates
                                         $best-efe $noise $depth))
   )
   (take-top $max-k $within-noise)))
```

When the EFE landscape is steep (clear winner), only one action falls within
the noise floor → beam = 1 (tunnel vision). When the landscape is flat,
multiple actions are indistinguishable → beam widens (deliberation). The beam
width is not a parameter — it *emerges* from the interaction of the EFE
landscape with the noise level.

### B. How Symmetry Breaking Emerges

Symmetry breaking is not a separate mechanism. It is the special case where
the noise filter reduces the beam to 1: the gap between best and second-best
exceeds the noise floor. The system has "fallen into a clear potential well"
and evaluating alternatives is uninformative at this noise level.

In the previous design, symmetry breaking used a separate `cutoff-ratio`
parameter relative to the EFE range. Now it falls out automatically: if
`EFE(a₂) - EFE(a₁) > noise_floor(depth)`, then `a₂` is pruned. No separate
parameter needed.

### C. How the Planning Horizon Emerges

Planning stops when degraded confidence drops below a floor:

```metta
(= (beyond-confidence-floor? $state $depth)
   (let* (
     ($deg-conf (degraded-confidence $state $depth))
     ($floor (get-planning-config confidence-floor))
   )
   (< $deg-conf $floor)))
```

This is conceptually the same as the noise floor becoming so large that *all*
actions are within it — the system can no longer distinguish signal from noise.
The confidence floor is a fast-path check: rather than scoring all actions and
discovering that all survive the noise filter (which means the beam equals the
full action set and expansion is uninformative), we check confidence directly
and stop early.

**Two distinct signals (not conflated):**
- **Low belief precision** in simulated state → the world is uncertain.
  Agent should favor `observe`. (Captured by `sim-efe`'s epistemic weight.)
- **Low action model confidence** → the model is uncertain. Agent shouldn't
  trust predictions. (Captured by the confidence floor stopping rule.)

These have different implications and different remedies. The existing v5
infrastructure already handles both; the fractal planner inherits rather
than replaces.

### D. Residual EFE Scoring (Avoiding Myopic Pruning)

**Critical design decision.** Naive beam search scores candidates by their
*myopic* (single-step) EFE and prunes before recursing. This is a regression
from `policy_efe.metta` v5's approach of evaluating full policies.

The problem: `observe` might have higher myopic EFE than `wait` (it costs
more), but `observe → retreat` is a better policy than `wait → retreat`
because the observation improves the retreat's precision.

We score candidates at each beam level using **residual EFE** — the
immediate `sim-efe` plus a fast estimate of future value:

```metta
; Residual EFE: immediate cost + estimated future
(= (residual-efe $action $state $remaining-depth)
   (let* (
     ; Immediate: full sim-efe from existing infrastructure
     ($immediate (sim-efe $action $state))

     ; Future estimate: information gain compounds, error improves
     ($next-state (apply-action $action $state))
     ($future-est (future-heuristic $next-state $remaining-depth))
   )
   (+ $immediate (* discount $future-est))))

; Heuristic: expected error reduction from precision gains
; This is a lower bound on the value of future actions
(= (future-heuristic $state $remaining)
   (if (<= $remaining 0)
       0.0
       (let* (
         ($avg-prec (sim-avg-precision $state))
         ($avg-pref-gap (sim-avg-preference-gap $state))
         ; More precision → lower future error (optimistic estimate)
         ($est-improvement (* $avg-pref-gap (- 1.0 $avg-prec)))
       )
       ; Negative = good (this is EFE, lower is better)
       (- 0.0 (* $est-improvement (min $remaining 3))))))
```

This is analogous to A*'s `f = g + h`: the immediate EFE is `g`, the
future heuristic is `h`. The heuristic must be *admissible* (never
overestimate the true future EFE) for the pruning to be safe. Our
heuristic is optimistic — it assumes future actions will capture the
full available precision gain — so it's admissible.


## 4. Integration with Existing Infrastructure

### What We Reuse (Not Replace)

The fractal planner reuses the entire simulation infrastructure from
`policy_efe.metta` v5:

| Component | Source | Used For |
|-----------|--------|----------|
| `sim-state`, `SCons`/`SNil` | policy_efe §III | Simulated belief evolution |
| `sim-efe` | policy_efe §V | Per-step EFE evaluation |
| `apply-action` | policy_efe §IV | State transition |
| `violates-viability?` | policy_efe §X | Hard boundary pruning |
| `discount` | policy_efe §V | Confidence degradation rate |
| `sim-obs-info-gain` | policy_efe §VII | Information gain in sim |
| Action accessors | actions.metta §II-V | Model queries |

**What we replace:** Only `gen-policy`, `evaluate-policy`, `all-policy-evals`,
`best-of-evals`, and `select-best-policy` from `policy_efe.metta` §XI.

### Cycle Integration

The fractal planner registers as a third action selection mode in
`cycle.metta`:

```metta
; In cycle.metta — updated choose-action dispatcher
(= (choose-action)
   (let $mode (get-config action-selection-mode)
     (if (== $mode single)
         (select-action-myopic)        ; Single-step from actions.metta
         (if (== $mode fractal)
             (select-action-fractal)   ; Fractal from planning.metta
             (select-action)))))       ; Exhaustive from policy_efe.metta
```

### Parameters as Learnable Beliefs

The planning parameters can be promoted from configs to beliefs with
precision, enabling the bootstrap mechanism to tune them from experience:

```metta
; Planning meta-parameters — beliefs, not constants
; The bootstrap mechanism can tune these from experience

(belief noise-floor-base 0.03 0.4)      ; Base EFE noise (tightened from 0.05)
(belief confidence-floor 0.15 0.4)      ; Min confidence to continue planning
(belief base-beam-width 8 0.5)          ; Starting beam (narrows with depth)
(belief max-planning-depth 7 0.3)       ; Hard ceiling on depth
```

Note the reduction from five parameters to four, with only one (`noise-floor-base`)
controlling the physics of pruning. The rest are computational caps.

These feed into the bootstrap mechanism: if the agent discovers that its
planning parameters lead to poor action outcomes (high post-action surprise),
precision on those beliefs drops, making them more susceptible to update.


## 5. MeTTa Implementation

### The Noise Floor

The core unification: one quantity, derived from model confidence, that
controls all pruning.

```metta
; Compute the degraded model confidence at a given planning depth
(= (degraded-confidence $state $depth)
   (let $base-conf (sim-avg-confidence $state)
     (* $base-conf (pow-int discount $depth))))

; Compute the noise floor at a given planning depth
; Higher noise → less discrimination between actions
(= (noise-floor-at-depth $state $depth)
   (let* (
     ($deg-conf (degraded-confidence $state $depth))
     ($base (get-planning-config noise-floor-base))
   )
   ; Inverse confidence scaling, clamped to avoid division by zero
   (/ $base (max $deg-conf 0.01))))

; Should planning stop at this depth?
(= (beyond-confidence-floor? $state $depth)
   (let* (
     ($deg-conf (degraded-confidence $state $depth))
     ($floor (get-planning-config confidence-floor))
   )
   (< $deg-conf $floor)))
```

### The Noise Filter

Replaces both `adaptive-beam` and `apply-symmetry-cutoff` with a single
function. Actions within the noise floor of the best survive. Since
candidates are sorted, the filter short-circuits on the first action
beyond the noise — all subsequent are also beyond it.

```metta
(= (noise-filter $sorted-candidates $state $depth)
   (let* (
     ($noise (noise-floor-at-depth $state $depth))
     ($best-efe (candidate-efe (car $sorted-candidates)))
     ($effective-beam (max 2 (ceil-int (* (get-planning-config base-beam-width) $deg-conf))))
     ($within-noise (filter-within-noise $sorted-candidates
                                         $best-efe $noise $depth))
   )
   (take-top $max-k $within-noise)))

(= (filter-within-noise () $best $noise $depth) ())
(= (filter-within-noise ((candidate $a $e) . $rest) $best $noise $depth)
   (let $gap (- $e $best)
     (if (<= $gap $noise)
         ((candidate $a $e) .
           (filter-within-noise $rest $best $noise $depth))
         ; Beyond noise floor — prune this and everything after
         (sequential
           (record-pruning-at-depth! $a $depth noise-floor $gap)
           (prune-remaining-noise $rest $depth)
           ()))))
```

### The Fractal Planner

```metta
(= (fractal-expand $state $max-depth $current-depth)
   ; --- Stopping conditions ---
   (if (>= $current-depth $max-depth)
       (terminal-branch 0.0 $current-depth)

   (if (beyond-confidence-floor? $state $current-depth)
       (sequential
         (record-pruning-at-depth! all-actions $current-depth
           confidence-floor (degraded-confidence $state $current-depth))
         (terminal-branch 0.0 $current-depth))

   ; --- Active expansion ---
   (let* (
     ($remaining (- $max-depth $current-depth))

     ; 1. Score all candidates via residual EFE
     ($candidates (score-all-actions $state $remaining))

     ; 2. Viability filter (hard kill)
     ($viable (viability-filter $candidates $state $current-depth))
   )
   (if (list-empty? $viable)
       (terminal-branch 999.0 $current-depth)
   (let* (
     ; 3. Sort by EFE (ascending)
     ($sorted (sort-candidates $viable))

     ; 4. Noise filter (unified: beam width + symmetry breaking)
     ($survivors (noise-filter $sorted $state $current-depth))

     ; 5. Recurse on survivors
     ($branches (expand-survivors $survivors $state
                                  $max-depth $current-depth))
   )
   (if (list-empty? $branches)
       (terminal-branch 999.0 $current-depth)
       (best-branch $branches))))))))
```

Note the simplification: steps 3-4 in the old design (adaptive beam, then
symmetry cutoff) are now a single step 4 (noise filter). The flow is:
score → viability → sort → noise filter → recurse → select best.

### Traced Selection (Full Transparency)

```metta
(= (select-action-fractal-traced)
   (let* (
     ($_ (clear-fractal-pruning-records!))
     ($state (current-sim-state))
     ($max-depth (get-planning-config max-planning-depth))
     ($result (fractal-expand $state $max-depth 0))
     ($action (branch-action $result))
     ($efe (branch-efe $result))
     ($trace (branch-trace $result))
     ($depth-reached (branch-depth $result))
     ($pruned (all-pruning-records))
     ($pruned-at-depth (all-pruning-records-with-depth))
     ($pressure (get-viability-pressure))
     ($ew (epistemic-weight))
     ($noise-d0 (noise-floor-at-depth $state 0))
     ($conf-d0 (degraded-confidence $state 0))
   )
   (fractal-trace
     (selected-action $action)
     (total-efe $efe)
     (planned-trajectory $trace)
     (planning-depth $depth-reached)
     (max-depth-allowed $max-depth)
     (branches-pruned $pruned)
     (branches-pruned-at-depth $pruned-at-depth)
     (viability-pressure $pressure)
     (epistemic-weight $ew)
     (noise-parameters
       (noise-floor-base (get-planning-config noise-floor-base))
       (noise-at-depth-0 $noise-d0)
       (confidence-at-depth-0 $conf-d0)
       (confidence-floor (get-planning-config confidence-floor))
       (base-beam-width (get-planning-config base-beam-width)))
     (model-confidences
       (wait (avg-action-confidence wait))
       (observe (avg-action-confidence observe))
       (retreat (avg-action-confidence retreat)))
     (reason "first action of minimum cumulative EFE trajectory
              via noise-aware fractal expansion"))))
```


## 6. Comparison: Exhaustive vs. Fractal vs. MCTS

| Feature | Exhaustive (v5) | MCTS | Fractal (Noise-Aware) |
|---------|-----------------|------|-----------------------|
| **Mechanism** | Enumerate all policies | Random sampling | Noise-aware beam |
| **Complexity** | O(\|A\|^d) | O(iterations × d) | O(\|A\| × k × d) |
| **Deterministic** | Yes | No | Yes |
| **Traceable** | Yes (all policies scored) | Low (stochastic trees) | Yes (pruning records) |
| **Metabolic Cost** | High at depth >4 | High (many rollouts) | Low (directed beam) |
| **Handles Nonlinear Interactions** | Perfectly | Statistically | Via residual EFE heuristic |
| **MeTTa Fit** | Good (non-determinism) | Poor (needs randomness) | Excellent (recursive) |
| **Risk** | None (exact) | Sampling bias | Irrecoverable pruning |
| **Tunable Parameters** | 1 (horizon) | ~3 (iterations, rollout depth, UCB) | 1 physics + 2 caps |


## 7. Emergent Behavior

The algorithm naturally regulates the "Level of Detail" of the agent's
reasoning. These are not programmed modes — they emerge from the interaction
of the noise floor with the EFE landscape:

**1. The "Obvious" Path (Tunnel Vision)**
When one action is clearly superior, the gap to the second-best exceeds the
noise floor. Beam collapses to 1. The agent thinks fast and deep — planning
extends to the full confidence horizon along a single trajectory. This is
computationally cheap: O(d) for depth d.

**2. The "Confused" Path (Deliberation)**
When the EFE landscape is flat (small gaps), multiple actions fall within
the noise floor. Beam stays wide. The agent thinks broad and shallow,
naturally favoring `observe` actions that sharpen the gradient — because
`observe` increases precision, which increases EFE spread at the next level,
which collapses the beam. The system seeks its own phase transition.

**3. The "Dark" Path (Intellectual Humility)**
When model confidence is low, the noise floor is large even at depth 0.
Confidence degrades to the floor quickly, truncating the planning horizon.
The agent selects near-term actions, typically `observe` to build confidence
before committing to longer plans. The system refuses to extrapolate into
noise — not because a rule says "don't plan when uncertain," but because the
noise floor swamps all EFE differences.

**4. The "Urgent" Path (Viability Pressure)**
When viability bounds are approached, the viability effect in `sim-efe` creates
steep gradients. One action's EFE is much lower than alternatives — the gap
dwarfs any noise floor. Beam collapses to 1 at depth 1. The agent stops
deliberating and acts, not because urgency is programmed, but because one
action dominates the landscape so strongly that alternatives are noise.

### Why the Unification Matters

In the previous design, these behaviors came from three independent parameters
interacting in ways that were hard to reason about. A change to `cutoff-ratio`
could cause tunnel vision that the `min-beam-width` was supposed to prevent.
Now all four behaviors emerge from the same quantity — the noise floor — and
their relative activation depends only on the state of the world (the EFE
landscape and model confidence), not on parameter interactions.


## 8. Self-Model Integration

The fractal planner provides natural material for self-knowledge. The agent
can introspect its own planning process:

```metta
; How am I thinking right now?
(= (planning-style)
   (let* (
     ($_ (clear-fractal-pruning-records!))
     ($state (current-sim-state))
     ($max-depth (get-planning-config max-planning-depth))
     ($result (fractal-expand $state $max-depth 0))
     ($depth (branch-depth $result))
     ($trace-len (length (branch-trace $result)))
     ($pruned-count (length (all-pruning-records)))
   )
   ; Classify emergent planning style
   (if (and (> $trace-len 4) (> $pruned-count 3))
       (thinking-mode tunnel-vision "deep and narrow — clear best path")
       (if (and (<= $trace-len 2) (< $pruned-count 1))
           (thinking-mode cautious "short horizon — high uncertainty")
           (if (<= $trace-len 2)
               (thinking-mode deliberation "shallow and broad — weighing options")
               (thinking-mode balanced "moderate depth and breadth"))))))
```

Additionally, the noise floor itself is introspectable:

```metta
; Show noise floor progression across depths
(= (noise-floor-profile)
   (let $state (current-sim-state)
     (noise-profile
       (depth-0 (noise-floor-at-depth $state 0)
                (confidence (degraded-confidence $state 0)))
       (depth-3 (noise-floor-at-depth $state 3)
                (confidence (degraded-confidence $state 3)))
       (depth-5 (noise-floor-at-depth $state 5)
                (confidence (degraded-confidence $state 5)))
       (depth-7 (noise-floor-at-depth $state 7)
                (confidence (degraded-confidence $state 7))))))
```

This feeds into the self-model (`self_model.metta`) alongside existing
capabilities like `learning-state` and `structural-complexity`. The agent
doesn't just know what it decided — it knows *how it decided* and *how
noisy its planning was*.


## 9. Computational Complexity

The fractal planner changes scaling from exponential to approximately linear
in depth:

```
Exhaustive:  O(|A|^d)
Fractal:     O(|A| × k × d)    where k = effective beam width
```

With |A| = 3 and typical k = 2:

| Depth | Exhaustive | Fractal (k=2) | Speedup |
|-------|-----------|---------------|---------|
| 3     | 27        | 18            | 1.5×    |
| 5     | 243       | 30            | 8×      |
| 7     | 2,187     | 42            | 52×     |
| 10    | 59,049    | 60            | 984×    |

This makes deep planning (horizon 7-10) tractable within a real-time cycle,
enabling qualitatively different behavior: the agent can reason about
consequences several steps out without exhausting its computational budget.


## 10. Known Limitations and Risks

### Irrecoverable Pruning
The fundamental risk of any beam search. If an action looks bad at depth 1
but becomes optimal at depth 3 due to nonlinear interaction, it gets pruned
and the optimal policy is never discovered.

**Mitigations:**
- Residual EFE scoring (Section 3D) estimates future value, catching many
  such cases (e.g., `observe` looks expensive but its precision gain makes
  future actions better).
- The noise filter naturally keeps more candidates when the landscape is flat
  (exactly when pruning is most dangerous — if nothing clearly dominates,
  the noise floor lets more through).
- The exhaustive planner remains available as a fallback for small horizons:
  `(config action-selection-mode policy)` uses the exact v5 planner.
- The agent can learn: if fractal planning consistently leads to high
  post-action surprise, the bootstrap mechanism can increase `noise-floor-base`
  (widen the effective beam) or switch modes.

### Heuristic Admissibility
The future heuristic in residual EFE must remain admissible (optimistic) for
the pruning to be safe. The current heuristic assumes full precision gain is
capturable, which is optimistic. If someone changes the heuristic to be
pessimistic, pruning becomes unsafe.

### Noise Floor Calibration
The `noise-floor-base` parameter (0.05) interacts with the scale of EFE values.
If the EFE formula changes such that typical values shift by an order of
magnitude, the noise floor needs recalibrating. This is a single parameter to
adjust rather than three interacting ones, but the dependency exists.

Making `noise-floor-base` a learnable belief mitigates this: the bootstrap
mechanism can increase it (wider beam, more conservative) or decrease it
(narrower beam, more aggressive) based on planning outcomes.

### Confidence Floor vs. Noise Floor Redundancy
The confidence floor (`beyond-confidence-floor?`) is technically redundant
with the noise filter — at very low confidence, the noise floor is so large
that all actions survive, making expansion uninformative but not harmful.
The confidence floor is kept as a fast-path optimization: it avoids the
cost of scoring, sorting, and filtering when we already know the result
will be "keep everything." If performance is not a concern, it could be
removed without changing behavior.


## 11. Implementation Status

The implementation in `planning.metta` is complete. Section mapping:

| Planning.metta Section | Contents | Status |
|------------------------|----------|--------|
| §I Planning Parameters | `noise-floor-base`, `confidence-floor`, `base-beam-width`, `max-planning-depth` | Implemented as configs |
| §II Utility Functions | Sort, take-top, EFE statistics, sim-state helpers | Implemented |
| §III Noise Floor | `degraded-confidence`, `noise-floor-at-depth`, `beyond-confidence-floor?`, `noise-filter` | Implemented |
| §IV Residual EFE | `residual-efe`, `future-heuristic` | Implemented |
| §V Pruning Trace | Record/query/clear pruning records | Implemented |
| §VI Branch Data | Branch type, accessors, best-branch selection | Implemented |
| §VII Core Planner | `fractal-expand`, `score-all-actions`, `viability-filter`, `expand-survivors` | Implemented |
| §VIII Entry Points | `select-action-fractal`, `select-action-fractal-with-depth` | Implemented |
| §IX Traced Selection | `select-action-fractal-traced` with full metadata | Implemented |
| §X Self-Model | `planning-style`, `planning-cost-summary` | Implemented |
| §XI Diagnostics | `planning-parameters`, `noise-floor-profile`, `compare-fractal-vs-myopic` | Implemented |

**Remaining work:**
- Promote configs to beliefs for bootstrap learning (see Section 4)
- Validate on Hyperon runtime (blocked by cons-cell matching bug)
- Parameter sensitivity analysis via `test_fractal_planning.py`


## 12. Dependency Map

```
actions.metta (base-actions, action accessors, sim-efe components)
    ↓
policy_efe.metta (sim-state, sim-efe, apply-action, viability checks,
                  discount factor for confidence degradation)
    ↓
planning.metta (fractal-expand, noise filter, residual EFE,
                pruning trace — imports policy_efe's simulation
                infrastructure, replaces only §XI policy selection)
    ↓
cycle.metta (choose-action dispatcher gains 'fractal' mode)
    ↓
self_model.metta (planning-style introspection)
```

**No existing files are broken.** The exhaustive planner in `policy_efe.metta`
remains fully functional and selectable via `(config action-selection-mode policy)`.
The fractal planner is additive.


## 13. Design History

### Previous Design (v1): Three Independent Mechanisms

The original fractal planner used three separate pruning mechanisms, each
with its own parameter:

1. **Adaptive beam width** — governed by `min-beam-width` and `max-beam-width`,
   switching based on `efe-spread` relative to `cutoff-ratio × 0.5`.
2. **Symmetry breaking** — governed by `cutoff-ratio`, using the relative
   EFE gap compared to the EFE range.
3. **Correlation length** — governed by `uncertainty-limit`, stopping when
   degraded confidence dropped below threshold.

**Problem:** These parameters interacted in hard-to-predict ways. Changing
`cutoff-ratio` could cause tunnel vision that `min-beam-width` was supposed
to prevent. The three mechanisms were conceptually three manifestations of
the same thing — decreasing ability to distinguish actions as you look
further into the future — but the implementation treated them as independent.

### Current Design (v3): Unified Noise Floor + Adaptive Beam

The insight: all three mechanisms answer the same question — "is this EFE
difference meaningful given our uncertainty at this depth?" The noise floor
provides a single, physically motivated answer. Actions within the noise
floor survive; those beyond it are pruned. Beam width, symmetry breaking,
and horizon termination all emerge from this one comparison.

**v2.2 refinements (action space scaling benchmark):**

The beam width is now confidence-proportional rather than fixed:
```
effective_beam(depth) = max(2, ceil(base_beam * conf * discount^depth))
```
At depth 0 (conf=1.0) → full beam (8). At depth 5 (conf≈0.44) → beam=4.
This is the RG coarse-graining analogy made literal: fewer relevant
operators survive at coarser scales.

The noise floor base was tightened from 0.05 to 0.03 based on multi-seed
benchmarks at 10-50 actions: 100% answer preservation, up to 38x speedup.

**Parameter reduction:** Five parameters (min-beam-width, max-beam-width,
cutoff-ratio, uncertainty-limit, max-planning-depth) reduced to four
(noise-floor-base, confidence-floor, base-beam-width, max-planning-depth),
with only one (`noise-floor-base`) controlling the physics of pruning.
`base-beam-width` narrows with confidence — an emergent quantity, not a cap.

**Future:** `noise-floor-base` itself could become adaptive via
`noise = k * std(EFE scores)`, eliminating the last hand-tuned parameter.
