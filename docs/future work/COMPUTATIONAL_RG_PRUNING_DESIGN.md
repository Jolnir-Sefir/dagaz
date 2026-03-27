# Computational RG Pruning — Adaptive Cycle Budget Allocation via Noise-Floor Coarse-Graining

## The Problem (The Computational Wall)

Project Dagaz's fractal planner applies Renormalization Group-inspired coarse-graining to the *temporal* decision tree: integrate out trajectories indistinguishable from noise at each depth level. This yields 52× reduction in planning evaluations (depth 7, beam 2) with correct action selection across all test scenarios.

The cognitive cycle itself has no equivalent optimization. Every cycle, the system performs:

- **EFE computation**: O(|A| × |O|) — every action scored against every observable
- **Belief updates**: O(|O|) — every observable updated from prediction errors
- **Structure learning**: O(|O|²) — all observable pairs checked for co-occurring errors (mitigated by LSH at scale)
- **Metabolic step**: O(|atoms|) — every learned atom pays rent, earns rewards

For small-scale deployments (~20 observables, ~5 actions, ~100 state atoms), this is manageable under PyPy (~50ms). But cost scales sharply: industrial monitoring with hundreds of sensors, ecological networks with thousands of species interactions, urban traffic systems with tens of thousands of road segments, or any spatially-indexed environment where zone-keyed observables multiply the state space — the per-cycle cost becomes prohibitive.

The key observation: **most of this computation is wasted.** In any given cycle, the majority of observables have low prediction error (the model is locally accurate). Their EFE contributions are near-zero. Their belief updates are negligible. Their structure learning pairs are uninformative. The system spends metabolic compute on regions of state space where nothing interesting is happening.

This is the exact same waste the fractal planner eliminates in the temporal domain. The principle extends: **integrate out irrelevant degrees of freedom, whether they are future timesteps or current observables.**


## The Physicist's Question

**What is the invariant?**

**Computation is metabolic work. Allocate it proportional to expected information gain.** The same principle that drives the fractal planner's noise floor drives cycle-level pruning. An observable at precision 0.95 with prediction error 0.01 is — for the purpose of discriminating between actions — indistinguishable from an observable with zero error. Evaluating its EFE contribution across all actions is wasted work, just as expanding a clearly dominated trajectory is wasted work in planning.

The invariant remains: minimize EFE. The optimization is: don't compute EFE terms that are provably below the discrimination threshold.

**What are the true degrees of freedom?**

One primary dimensionless quantity governs the pruning:

1. `σ_compute` — the computational noise floor. Observable-action pairs whose EFE contribution is below this threshold are integrated out.

This is analogous to `noise-floor-base` in the fractal planner. It derives from the same source: belief precision. An observable with precision $p$ and prediction error $e$ contributes to EFE discrimination between actions proportional to $|e| \times (1 - p) \times w_i$, where $w_i$ is the observable's importance weight. When this quantity is below $\sigma_{compute}$, the observable cannot meaningfully change which action wins.

A secondary quantity controls structure learning pruning:

2. `σ_structural` — the structural noise floor. Observable pairs whose combined prediction error is below this threshold are excluded from the Hebbian co-occurrence check.

**What are the symmetries?**

**Temporal pruning and spatial pruning are the same operation on different axes.** The fractal planner integrates out future timesteps where confidence has degraded below the noise floor. Computational RG pruning integrates out current observables where prediction error has settled below the noise floor. Both produce the "effective theory" relevant to the current decision — the minimal representation needed to select the right action.

| Fractal Planner | Computational RG Pruning |
|----------------|--------------------------|
| Prunes across depth (time) | Prunes across observables (state) |
| Noise floor from degraded confidence | Noise floor from low prediction error |
| Beam width emerges from EFE landscape | Active observable set emerges from error landscape |
| Symmetry breaking: one action dominates | Satiation: one region is fully predicted |
| Correlation length: all paths converge | Stability: observable hasn't changed in N cycles |

**What is conserved?**

The selected action must be identical (or nearly so) to the action the unpruned cycle would select. This is the correctness constraint. Specifically:

- **EFE ordering is preserved.** If action $a$ has lower EFE than action $b$ in the full computation, it must also have lower EFE in the pruned computation. Pruning can only remove terms that are common across actions (and thus cancel in comparison) or that are below the discrimination threshold.
- **Metabolic fairness is preserved.** Pruned observables still pay rent and earn rewards — the pruning applies to *cycle computation*, not to metabolic accounting. An observable that is pruned from EFE because it's well-predicted should still earn its metabolic reward for being well-predicted.

**What emerges from simple rules?**

- **Attention.** The system allocates compute to surprised observables — the ones with high prediction error. This is not programmed attention; it emerges from the noise-floor filter. The "attentional spotlight" is the set of observables above the computational noise floor.
- **Habituation.** Stable, well-predicted observables are pruned from expensive computations. The system stops "noticing" them. If they later spike (environmental change), they re-enter the active set.
- **Vigilance under threat.** When viability pressure is high, the noise floor should tighten (more observables are decision-relevant because survival depends on precise state knowledge). The active set expands. This mirrors biological hypervigilance.
- **Computational relaxation.** In safe, predictable environments, most observables are pruned. The cycle becomes very fast. This is the computational analogue of the Default Mode Network transition: when there's nothing to attend to externally, compute budget is freed for internal operations.


## Architecture

### The Three-Tier Pruning Hierarchy

#### Tier 1: Observable Triage (Before EFE)

At the start of each cycle, after prediction errors are computed, classify observables into three sets:

```metta
; Compute the decision relevance of an observable
(= (decision-relevance $obs)
   (let* (
     ($error (abs (get-prediction-error $obs)))
     ($prec  (get-belief-precision $obs))
     ($imp   (get-importance $obs))
     ; How much can this observable shift EFE between actions?
     ($max-delta-across-actions (max-efe-sensitivity $obs))
   )
   (* $error (- 1.0 $prec) $imp $max-delta-across-actions)))

; Triage into active/dormant/frozen
; Active: full EFE, belief update, structure learning
; Dormant: belief update only (maintain precision decay), skip EFE detail
; Frozen: skip entirely this cycle (stable for N cycles)
(= (triage-observable $obs $cycle)
   (let* (
     ($relevance (decision-relevance $obs))
     ($sigma (get-config computational-noise-floor))
     ($cycles-stable (cycles-since-last-surprise $obs $cycle))
     ($freeze-threshold (get-config stability-freeze-cycles))
   )
   (if (> $relevance $sigma)
       active
   (if (> $cycles-stable $freeze-threshold)
       frozen
       dormant))))
```

The `max-efe-sensitivity` term is critical. It measures the maximum difference this observable can make to EFE *across the action set*. An observable where all actions have identical predicted effects contributes nothing to discrimination regardless of its error. This is the spatial analogue of the planning noise floor: if the EFE landscape is flat along this dimension, integrating it out is lossless.

#### Tier 2: EFE Partial Evaluation (During Action Scoring)

Even among active observables, early termination is possible:

```metta
; Pruned EFE: stop accumulating terms when winner is clear
(= (compute-efe-pruned $action $active-observables)
   (let* (
     ; Sort by decision relevance (highest first)
     ($sorted (sort-by-relevance $active-observables))
     ; Accumulate EFE terms, tracking running bounds
     ($result (accumulate-efe-terms $action $sorted 0.0))
   )
   $result))

; If partial EFE already exceeds the current best by more than
; the maximum possible contribution of remaining observables,
; this action cannot win — stop computing.
(= (accumulate-efe-terms $action () $running-total) $running-total)
(= (accumulate-efe-terms $action ($obs . $rest) $running-total)
   (let* (
     ($term (efe-term $action $obs))
     ($new-total (+ $running-total $term))
     ($remaining-bound (sum-max-contributions $rest))
   )
   ; Alpha-beta style cutoff
   (if (action-already-dominated? $new-total $remaining-bound)
       $new-total  ; early exit — this action can't win
       (accumulate-efe-terms $action $rest $new-total))))
```

This is branch-and-bound applied to the observable sum. Sorting by relevance (highest first) ensures the most discriminative terms are evaluated first, maximizing the chance of early cutoff.

#### Tier 3: Structure Learning Sparsification (During Hebbian Check)

Structure learning's pair-wise check is the O(|O|²) bottleneck at scale. The RG pruning is:

```metta
; Only check pairs where BOTH observables had recent surprise
(= (structure-learning-step-rg! $cycle $errors)
   (let* (
     ($surprised (filter-surprised $errors
                   (get-config structural-surprise-threshold)))
     ; Only check pairs within the surprised set
     ; Plus: pairs where one surprised observable is adjacent
     ; to a known causal link (propagation check)
     ($candidates (union
       (all-pairs $surprised)
       (adjacent-to-surprised $surprised)))
   )
   (check-pairs-selective! $candidates $cycle)))
```

The LSH optimization in `test_lsh_hebbian.py` already achieves 19.4× pair reduction at 1,000 observables via locality-sensitive hashing. This Tier 3 pruning is complementary: LSH reduces pairs by spatial locality; RG pruning reduces pairs by temporal relevance (recent surprise). Applied together, the effective pair count is:

```
effective_pairs ≈ |surprised|² × LSH_reduction_factor
```

For 1,000 observables with 5% surprised and 19.4× LSH reduction: ~130 pairs instead of 500,000.


### Modified Files

| File | Addition | Purpose |
|------|----------|---------|
| `cycle.metta` | ~80 lines | Observable triage step, pruned EFE dispatch, dormant/frozen handling |
| `actions.metta` | ~60 lines | `compute-efe-pruned`, `max-efe-sensitivity`, branch-and-bound accumulator |
| `structure_learning.metta` | ~40 lines | Selective pair checking, surprise-filtered candidate generation |
| `foundations.metta` | ~15 lines | New config parameters for computational noise floors and freeze threshold |
| `self_model.metta` | ~20 lines | Active-set size, computational savings ratio, attention distribution as self-observables |

### Untouched Files

All existing EFE formulas, belief update math, metabolic accounting, safety architecture, and grounding systems remain unchanged. The pruning is applied *around* these computations, not *within* them. A pruned cycle produces the same result as a full cycle within the noise floor tolerance.


## The Noise Floor Derivation

The computational noise floor should not be a free parameter. It should derive from the same belief precision that drives the planning noise floor, maintaining the single-physics principle:

```metta
; Computational noise floor: derived from system state
(= (compute-sigma-compute)
   (let* (
     ($avg-prec (average-belief-precision))
     ($base (get-config rg-base-noise))
     ; Higher average precision → tighter noise floor
     ; (the system is confident, so small errors matter more)
     ; Lower average precision → wider noise floor
     ; (the system is uncertain, so only large errors matter)
   )
   (/ $base (max $avg-prec 0.01))))
```

This creates an adaptive regime:

| System State | Avg Precision | σ_compute | Active Set | Behavior |
|-------------|---------------|-----------|------------|----------|
| Early exploration | 0.2 | 0.25 | ~80% of observables | Almost everything is surprising |
| Stable model | 0.7 | 0.07 | ~20% of observables | Focused attention on anomalies |
| Expert knowledge | 0.9 | 0.06 | ~10% of observables | Very efficient, fast cycles |
| Novel disruption | drops to 0.4 | 0.13 | ~50% of observables | Attention broadens automatically |

The "novel disruption" row is the critical safety property: when the environment changes, precision drops, the noise floor widens, and more observables become active. The system cannot "sleep through" a crisis because the pruning mechanism itself is driven by the same prediction errors that signal the crisis.


## Viability-Coupled Noise Floor

Under viability pressure, the computational noise floor must tighten:

```metta
; Tighten noise floor when survival is at stake
(= (compute-sigma-compute-viability)
   (let* (
     ($base-sigma (compute-sigma-compute))
     ($vp (max-viability-pressure))
     ; viability pressure in [0, 1]: 0 = safe, 1 = critical
     ; Tighten by up to 4× under maximum pressure
     ($tightening (+ 1.0 (* 3.0 $vp)))
   )
   (/ $base-sigma $tightening)))
```

Under maximum viability pressure, the noise floor drops to 25% of its relaxed value. Nearly all observables become active. This is computational hypervigilance: when survival is threatened, the system cannot afford to ignore *anything*. The metabolic cost of the wider computation is justified by the existential stakes — the same tradeoff biological organisms make under threat.


## Correctness Guarantee

The pruning is only valid if it preserves EFE ordering between actions. The formal condition:

```
For all actions a, b:
  EFE_full(a) < EFE_full(b)  ⟹  EFE_pruned(a) < EFE_pruned(b)
```

This holds when the pruned terms satisfy:

```
|Σ_pruned_obs [efe_term(a, obs) - efe_term(b, obs)]| < σ_compute
```

That is, the total EFE difference attributable to pruned observables is smaller than the noise floor. Since we prune observables whose individual contributions are below `σ_compute / |O|`, and these terms partially cancel across actions (both actions predict similar effects on well-known observables), the aggregate difference is bounded.

**When the guarantee fails:** If two actions have nearly identical EFE (difference < σ_compute), the pruned cycle may select a different action than the full cycle. This is acceptable by the same argument the fractal planner uses: actions within the noise floor are indistinguishable. Selecting either one is equally valid. The noise floor defines the system's computational precision — discriminating beyond it would be false precision.

**Validation strategy:** Run paired cycles (full and pruned) on existing benchmarks. Verify that action selection agrees everywhere the EFE gap exceeds 2 × σ_compute, and that disagreements occur only within the noise floor.


## Performance Projections

### Small-Scale Deployment (Baseline)

20 observables, 5 actions, single-environment monitoring:

| Metric | Full Cycle | RG-Pruned Cycle | Savings |
|--------|-----------|-----------------|---------|
| Observables evaluated | 20 | ~6 (30% active) | 70% |
| EFE terms computed | 100 (5×20) | ~30 + early exits | 70%+ |
| Structure pairs checked | 190 | ~15 (surprised pairs only) | 92% |
| Estimated cycle time (PyPy) | 50ms | ~20ms | 60% |

Marginal benefit at this scale. PyPy already provides ample headroom. The optimization becomes essential as observable counts grow — particularly in spatially-indexed environments where zone-keyed beliefs multiply the state space (e.g. 5 zones × 10 observables = 50 zone-keyed beliefs competing for EFE attention).

### Industrial Monitoring (Target Use Case)

1,000 sensors, 50 actions, 5% in anomalous state at any time:

| Metric | Full Cycle | RG-Pruned Cycle | Savings |
|--------|-----------|-----------------|---------|
| Observables evaluated | 1,000 | ~50 active + 100 dormant | 85% |
| EFE terms computed | 50,000 | ~2,500 + branch-and-bound | 95% |
| Structure pairs (with LSH) | ~25,000 | ~130 (surprised × LSH) | 99.5% |

### Ecological Network (Stress Test)

10,000 species interactions, 200 management actions:

| Metric | Full Cycle | RG-Pruned Cycle | Savings |
|--------|-----------|-----------------|---------|
| EFE terms | 2,000,000 | ~20,000 (1% active × branch-and-bound) | 99% |
| Structure pairs (with LSH) | ~2,500,000 | ~2,500 | 99.9% |

The savings compound across tiers. Tier 1 (triage) reduces the observable set. Tier 2 (branch-and-bound) early-exits within the reduced set. Tier 3 (sparse structure learning) operates on the intersection of surprised observables and LSH buckets.


## Emergent Phenomena

### Attention as Coarse-Graining

The active observable set *is* the system's attentional focus. It is not selected by a dedicated attention mechanism — it emerges from the noise-floor filter applied to prediction errors. This is the same design philosophy as the fractal planner: beam width emerges from the EFE landscape, not from a beam-width parameter. Attention emerges from the error landscape, not from an attention module.

The self-model can observe its own attention distribution:

```metta
(= (attention-distribution)
   (let* (
     ($active (count-active-observables))
     ($total (count-all-observables))
     ($focus (/ $active $total))
   )
   (attention-state $active $total $focus)))
```

An attention-focus of 0.05 (5% of observables active) indicates a highly focused system. An attention-focus of 0.80 indicates a system in crisis or early exploration. The transition between these regimes is not switched — it slides continuously with the noise floor.

### Computational Metabolism

The cycle's compute cost becomes a measurable metabolic quantity. The self-model can track it:

```metta
(= (computational-cost-this-cycle)
   (let* (
     ($efe-evals (count-efe-evaluations))
     ($struct-pairs (count-structure-pairs-checked))
     ($total (+ $efe-evals $struct-pairs))
   )
   (compute-cost $total)))
```

This enables a second-order optimization: if the system notices that its computational cost is consistently high (many observables active for many cycles), it can allocate cognitive budget toward structure learning in the active region — discovering a new causal link would *reduce* prediction errors, which would *reduce* the active set, which would *reduce* computational cost. The system can reason about its own computational efficiency through the same EFE framework that governs everything else.


## When to Implement

**Not yet for small-scale deployments.** PyPy provides 40× speedup on the unmodified evaluator, bringing cycle times to ~50ms against a 500ms tick budget. At ~20 observables, there is 10× headroom. However, spatially-indexed environments (zone-keyed beliefs, multi-region monitoring) can push effective observable counts past 50 even at modest physical scale, closing this margin rapidly.

**When any of these conditions are met:**

1. Observable count exceeds ~200 (spatial scaling in complex environments)
2. Cycle time exceeds 50% of tick budget under PyPy (measured, not projected)
3. A deployment target cannot run PyPy (embedded systems, WASM)
4. Structure learning pair count exceeds LSH's ability to manage it

**The implementation order should be:** Tier 3 (structure learning sparsification) first, because it has the worst scaling (quadratic) and the cleanest correctness guarantee (only surprised observables can form new causal links anyway). Tier 1 (observable triage) second, because it feeds into both EFE and structure learning. Tier 2 (branch-and-bound EFE) last, because it has the most complex correctness argument and the least impact at moderate observable counts.


## Relationship to Other Future Work

**Default Mode Network.** When the RG pruning reduces the active set to near-zero (everything is well-predicted), the freed computational budget is exactly what the DMN design document proposes to use for offline consolidation and deep abstraction search. Computational RG pruning provides the *mechanism* by which the DMN transition occurs: the system doesn't need to "decide" to daydream — it runs out of external observables to attend to, and the compute budget is automatically available for internal operations.

**Epistemic Credit Market.** The chain credit mechanism requires traversing structural dependency graphs. In a large system, this traversal is itself expensive. Tier 1 triage could inform which dependency chains to evaluate: only chains terminating in active (surprised) observables need credit/debit propagation this cycle. Dormant chains can defer their accounting.

**Theory of Mind.** Partner modeling doubles the effective observable space (the system tracks both world-state and partner-state observables). Computational RG pruning is essential for making this tractable — in most conversational turns, the partner's emotional state is stable and can be pruned from the EFE computation.


## Design Principles Compliance

| Principle | Status |
|-----------|--------|
| **Bottom-Up** | No enumerated attention targets. The active set emerges from the noise floor applied to prediction errors. |
| **Symbolic** | All triage, pruning, and bound computation is in MeTTa. No statistical attention model. |
| **Transparent** | The active/dormant/frozen classification is queryable. Every pruning decision is traceable. |
| **Emergent** | Attention, habituation, vigilance, and computational relaxation emerge from one noise-floor parameter. |
| **Honest** | The system reports its computational precision. Actions selected within the noise floor carry appropriate epistemic status. |

---

*This document describes a planned optimization for environments where observable counts make full-cycle computation intractable. It is architecturally designed but not yet implemented. Small-scale deployments run on the unmodified evaluator under PyPy, validating the cognitive dynamics before introducing computational approximations. The three-tier pruning hierarchy is designed to be introduced incrementally — each tier is independently valuable and does not require the others.*
