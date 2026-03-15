# Epistemic Credit Market — Deep Abstraction via Metabolic Dynamics

## The Problem

The current metabolic economy (`atom_lifecycle.metta`) treats all learned atoms equally: every atom pays a flat rent (`c_rent = 0.02`), and every atom earns a flat reward (`c_reward = 0.05`) when it correctly predicts an observation.

This works flawlessly for shallow, immediately testable structure (e.g., Phase 1 causal links like `smoke → fire`). However, as the system ascends the abstraction hierarchy via Phase 2 (Latents), Phase 4 (Concept Blends), and Phase 1.5 (Deduction), two problems compound:

1. **The rent problem.** The temporal distance between a concept's creation and its empirical validation grows with depth. Under a flat metabolic rate, deep abstractions die of metabolic starvation before they ever have the chance to participate in a reward-yielding prediction.

2. **The reward problem.** Even if rent is reduced, deep atoms have no direct mechanism to *earn* energy. A D=4 concept blend doesn't predict observations — it supports the reasoning chains that support the predictions. The current system credits only the terminal predictor (D=1), not the structural scaffolding that enabled the prediction.

Highly abstract concepts (analogous to "calculus", "justice", or "string theory") require massive structural scaffolding. If we simply lower the global `c_rent` to save them, we break the pruning mechanism, filling the Atomspace with spurious noise. If we only modulate rent without addressing reward, we've merely converted a quick death into a slow one.

## The Physicist's Question

**What is the invariant?**

**Collateralized Epistemic Risk with Bidirectional Chain Credit.** The system can only "afford" to extend the metabolic runway of an abstract concept if the foundational concepts it builds upon are highly stable. The concept can only become self-sustaining if the predictive chains it supports actually succeed. And the concept is *actively punished* if the chains it supports generate persistent failure — the metabolic economy tracks not just income but also liability.

The invariant remains: *pay rent or die*. But we introduce three structural dimensions:

- **Rent side:** *Who subsidizes the rent, and what is the collateral?* If a concept is built on top of highly predictive, maximum-energy foundational atoms, the foundation's health reduces the concept's rent. If the foundation's predictive power fails, the subsidy drops and the abstraction starves.

- **Reward side:** *Who earned the prediction, and who enabled it?* When a terminal atom correctly predicts an observation, a fraction of the reward propagates backward through the structural dependency graph. Deep atoms earn energy proportional to the *cumulative predictive success of their dependents*. Conversely, when descendants *fail*, a fraction of the penalty propagates backward — deep atoms that generate persistent failure actively bleed energy beyond rent.

- **Affect coupling:** *How surprising was the resolution?* Reward is not flat — it scales with the chronic prediction error that the successful prediction resolves. Cracking a long-standing mystery earns an order of magnitude more energy than confirming an already-known pattern. This couples the metabolic economy to the affect system: the Fristonian "emotion as first derivative of free energy" becomes a metabolic signal.

**What are the true degrees of freedom?**

Two primary dimensionless quantities govern the economy, with two secondary quantities introduced by the credit market:

1. `c_reward / c_rent` — the base metabolic ratio (existing; transition at ≈1.0)
2. `δ` — the chain credit decay factor (new; governs how far reward/penalty propagates)
3. `c_failure / c_reward` — the failure-to-success penalty ratio (new; governs how harshly failure is penalized relative to success; must be < 1)
4. `α` — the epiphany scaling coefficient (new; governs how much chronic error amplifies reward)

The system's behavior across the abstraction hierarchy is determined by their interaction. Parameters 3 and 4 are second-order — they modulate the dynamics established by 1 and 2.

**What is conserved?**

Total reward is *not* conserved — chain credit creates new energy (the terminal atom keeps its full reward; parents receive additional fractional credit). Conversely, chain debit destroys energy beyond rent (parents lose energy when descendants fail). This is deliberate: the metabolic economy is a *dissipative system*, not an accounting ledger. Energy enters through successful predictions, exits through rent and failure penalties, and the equilibrium Atomspace size emerges from the balance. The conservation constraint is on *rent* — every atom always pays some minimum, so total energy dissipation is bounded below by `N_atoms × rent_floor`.

## Architecture

### New / Modified Files

| File | Addition | Purpose |
|------|----------|---------|
| `atom_lifecycle.metta` | ~200 lines | Structural depth, effective rent, bidirectional chain credit, epiphany bonus, extended gestation (Section VIII). |
| `foundations.metta` | ~20 lines | New configuration parameters for the credit market. |
| `structure_learning.metta` | ~15 lines | Wire chain credit/debit into existing reward pathway; chronic error query. |
| `self_model.metta` | ~25 lines | Track "epistemic debt", "paradigm stability", and "allostatic load" as self-observables. |

### Untouched Files
All reasoning, grounding, and EFE modules remain untouched. The change is purely internal to how metabolic energy is deducted and credited during the `metabolic-step!`.


## The Six Mechanisms

### 1. Structural Depth ($D$)

Every learned atom has a structural depth, representing its distance from raw sensory observables.

*   Observable / Phase 1 Link: $D = 1$
*   Latent Variable (Phase 2): $D = 1 + \max(D_{members})$
*   Deduced Link (Phase 1.5): $D = 1 + \max(D_{source_1}, D_{source_2})$
*   Concept Blend (Phase 4): $D = 1 + \max(D_{parent_A}, D_{parent_B})$

**DAG Assumption.** Structural depth is defined recursively via `max(D_parents)`. This requires acyclic structural dependencies. Mutual causation (A→B and B→A) can exist in passive models, but *structural parentage* (which atom was built from which) is always a DAG — a deduced link is built from its premises, a latent is built from its members, a blend is built from its parents. If a cycle were ever introduced (e.g., a latent variable whose member is a concept that was itself built from that latent), depth would be undefined. The system should assert acyclicity at creation time.

Structural depth is computed once at atom creation and stored as metadata:

```metta
(structural-depth $type $name $depth)
```

It never changes after creation. This is $O(1)$ per query.


### 2. The Mortgage (Extended Gestation)

When a high-level atom is born, it receives an extended grace period (gestation) proportional to its depth. This acts as a metabolic "loan," giving it time to integrate into the causal graph before rent is due.

```metta
; Effective gestation scales logarithmically with depth
(= (effective-gestation $atom-type $atom-name)
   (let* (
     ($base (get-config default-gestation-period))
     ($depth (get-structural-depth $atom-type $atom-name))
   )
   (* $base (+ 1.0 (floor-log2 $depth)))))
```

For D=1 atoms, this returns the existing gestation period unchanged: `3 × (1 + floor_log2(1)) = 3 × 1 = 3`. The existing behavior is a special case. At D=8: `3 × (1 + 3) = 12 cycles`.


### 3. The Subsidy (Rent Discount)

Once gestation ends, the atom must pay rent. However, its rent is discounted based on the *current metabolic energy* of its structural parents (its one-hop structural dependencies — not the full ancestor chain).

```metta
; R_effective = R_base - (γ × average_parent_energy)
(= (compute-effective-rent $atom-type $atom-name)
   (let* (
     ($base-rent (get-config metabolic-rate))
     ($parents (get-structural-parents $atom-type $atom-name))
     ($avg-energy (average-energy $parents))
     ($gamma (get-config epistemic-subsidy-weight))    ; e.g., 0.008
     ($discount (* $gamma $avg-energy))
     ($rent-floor (get-config minimum-metabolic-rate)) ; e.g., 0.002
   )
   (max $rent-floor (- $base-rent $discount))))
```

**Crucial Constraint:** Rent must **never** reach zero or go negative. Epistemic inflation (where useless concepts live forever) is prevented by the `rent-floor`. Every concept, no matter how well-subsidized, bleeds a tiny amount of energy and must eventually prove useful.

**Bounds analysis:** With `γ = 0.008` and maximum parent energy 2.0, the maximum discount is `0.008 × 2.0 = 0.016`, reducing effective rent from 0.02 to 0.004. The floor clamps at 0.002. So effective rent ranges over `[0.002, 0.02]` — a 10× range, controlled entirely by foundation health.

**Note on the "collateral" metaphor.** The foundation's energy is *read* to compute the subsidy but not *debited*. No energy transfers from parent to child. This is closer to a credit rating than true collateral — the parent's health determines the child's borrowing terms, but supporting deep dependents doesn't directly weaken the parent. A true collateral system (where parents pay a tax per dependent) would create pressure against over-abstraction. We chose the simpler mechanism because the rent floor already bounds growth, and adding a dependent-tax would create a second interaction term in the sensitivity analysis. The extension is available if empirical testing shows over-abstraction.


### 4. Chain Credit (Reward Propagation — Success)

This is the mechanism that makes deep abstraction *self-sustaining* rather than merely long-lived. It is one direction of a bidirectional propagation system; Mechanism 5 handles the failure direction.

When a D=1 atom earns reward $R$ from a successful prediction, a fraction propagates backward through the structural dependency graph:

$$R_{parent} = \delta^{\Delta D} \times R$$

where $\delta \in (0, 1)$ is the chain credit decay factor and $\Delta D$ is the depth gap between the rewarded atom and the ancestor.

```metta
; Propagate reward up the structural dependency graph
(= (propagate-chain-credit! $atom-type $atom-name $reward)
   (let* (
     ($delta (get-config chain-credit-decay))       ; e.g., 0.3
     ($parents (get-structural-parents $atom-type $atom-name))
   )
   (propagate-to-parents! $parents $reward $delta)))

(= (propagate-to-parents! () $reward $delta) done)
(= (propagate-to-parents! (($p-type $p-name) . $rest) $reward $delta)
   (let $credit (* $delta $reward)
     (if (> $credit (get-config chain-credit-floor))  ; Don't propagate dust
       (sequential
         (boost-lifecycle-energy-by! $p-type $p-name $credit)
         ; Recurse: parent propagates to grandparent at δ² attenuation
         (propagate-chain-credit! $p-type $p-name $credit)
         (propagate-to-parents! $rest $reward $delta))
       (propagate-to-parents! $rest $reward $delta))))
```

**Concrete example.** With `δ = 0.3` and `c_reward = 0.05`:

| Depth | Per-chain credit | With 1 descendant | With 20 descendants |
|-------|-----------------|-------------------|---------------------|
| D=1   | 0.0500          | 0.0500/cycle      | 1.000/cycle         |
| D=2   | 0.0150          | 0.0150/cycle      | 0.300/cycle         |
| D=3   | 0.0045          | 0.0045/cycle      | 0.090/cycle         |
| D=4   | 0.00135         | 0.00135/cycle     | 0.027/cycle         |
| D=5   | 0.000405        | 0.000405/cycle    | 0.0081/cycle        |

Against the subsidized rent floor of 0.002/cycle, a D=4 atom with 20 active D=1 descendants earning reward every cycle nets `0.027 - 0.002 = +0.025/cycle` — robustly self-sustaining. A D=4 atom with a single intermittent descendant nets `0.5 × 0.00135 - 0.002 = -0.001325/cycle` — slow death.

**The selection criterion shifts with depth.** Shallow atoms survive by predicting. Deep atoms survive by being *structurally important* — by being hubs that support many successful prediction chains. Structural importance is not declared; it is *measured* by the accumulated predictive success of dependents. "Calculus" doesn't predict that the ball lands at 4.2 meters, but the differential equation that does predict it depends on calculus, and so do 500 other predictions across domains.

**Chain credit is non-conserving.** The terminal atom keeps its full reward; parents receive additional credit. Total energy in the system increases when predictions succeed. This is balanced by the universal rent drain, which always removes energy. The equilibrium size of the Atomspace is determined by the balance between total chain-credit income and total rent expenditure.

**Propagation cutoff.** The `chain-credit-floor` parameter (e.g., 0.0001) prevents infinite recursion and negligible updates. At `δ = 0.3`, credit drops below 0.0001 at depth gap ~7, providing a natural horizon.


### 5. Chain Debit (Failure Propagation — The Allostatic Load)

Chain credit (Mechanism 4) propagates reward upward on success. But the current system has an asymmetry: failure is just silence. A deep concept that generates 100 descendant predictions, 90 of which fail, pays no metabolic cost for those failures — the failing descendants simply don't send chain credit. The deep concept survives on the 10% that happen to be right. This is the "superstitious index fund" problem: a bloated, mostly-wrong concept that parasitically survives on a minority of accidental successes.

Biologically, maintaining a theory that constantly generates failures creates stress — allostatic load. In Fristonian terms, persistent prediction errors are not neutral; they represent chronic free energy that the system is failing to minimize. The metabolic economy should reflect this.

**The fix:** propagate failure through the same chain credit infrastructure, as *negative* chain credit (chain debit):

```metta
; On failed prediction, propagate penalty upward
(= (propagate-chain-debit! $atom-type $atom-name $penalty)
   (let* (
     ($delta (get-config chain-credit-decay))
     ($parents (get-structural-parents $atom-type $atom-name))
   )
   (propagate-debit-to-parents! $parents $penalty $delta)))

(= (propagate-debit-to-parents! () $penalty $delta) done)
(= (propagate-debit-to-parents! (($p-type $p-name) . $rest) $penalty $delta)
   (let $debit (* $delta $penalty)
     (if (> $debit (get-config chain-credit-floor))
       (sequential
         (drain-lifecycle-energy-by! $p-type $p-name $debit)
         (propagate-chain-debit! $p-type $p-name $debit)
         (propagate-debit-to-parents! $rest $penalty $delta))
       (propagate-debit-to-parents! $rest $penalty $delta))))
```

**Critical asymmetry:** The failure penalty `c_failure` must be significantly *less* than `c_reward`. Failure should hurt, but not as much as success helps. If `c_failure ≥ c_reward`, the system is penalized into paralysis — any concept whose descendants ever fail would be drained faster than it's filled, and only trivially safe (never-wrong) concepts would survive. We set `c_failure = 0.2 × c_reward = 0.01`.

**Concrete example — the superstition test.** A D=4 concept with 100 descendants, 90 failing and 10 succeeding. With `δ = 0.3`, `c_reward = 0.05`, `c_failure = 0.01`:

```
Chain credit income:  10 × δ³ × 0.05  = +0.0135/cycle
Chain debit penalty:  90 × δ³ × 0.01  = -0.0243/cycle
Net chain flow:                         -0.0108/cycle
Plus rent:                              -0.002 to -0.02/cycle
Total:                                  bleeding from both sides
```

Compare with a concept where 90 succeed and 10 fail:

```
Chain credit income:  90 × δ³ × 0.05  = +0.1215/cycle
Chain debit penalty:  10 × δ³ × 0.01  = -0.0027/cycle
Net chain flow:                         +0.1188/cycle
```

The breakeven success rate (where chain credit exactly offsets chain debit) is:

```
success_rate × c_reward = (1 - success_rate) × c_failure
success_rate = c_failure / (c_reward + c_failure) = 0.01 / 0.06 ≈ 17%
```

At the default parameters, a concept needs its descendants to be right more than ~17% of the time to be net-positive on chain flow alone. Below that, it's actively punished. Above that, it earns. This threshold is governed by the `c_failure / c_reward` ratio and can be tuned without touching the base metabolic parameters.

**Thermodynamic interpretation.** Chain credit counts the work output; chain debit counts the waste heat. The current design (Mechanism 4 alone) is like a heat engine that only measures useful work — it can't distinguish a 90%-efficient engine from a 10%-efficient one that happens to produce the same absolute output. Chain debit closes the thermodynamic accounting: net chain flow = work output - waste heat. Only efficient concepts (high success rate) survive.

**Interaction with affect.** The total chain debit flowing upward to a concept is a direct analog of *allostatic load* — the chronic stress of maintaining a failing world model. This quantity is already tracked (as a side effect of the debit propagation) and can feed into `self_model.metta` as a self-observable. High allostatic load across the entire structural graph corresponds to chronic negative valence and high arousal — the system is *frustrated*, and EFE-driven action selection will favor exploration (seeking new structure to reduce the chronic error).


### 6. The Epiphany Bonus (Affect-Coupled Reward)

In Fristonian terms, emotion (affect) is the *first derivative* of free energy. When prediction errors drop rapidly because a better model was found, the system experiences relief — a positive valence spike. The current metabolic economy ignores this: a correct prediction earns a flat `c_reward = 0.05` whether it resolved a massive chronic mystery or confirmed an already-known pattern.

The epiphany bonus scales reward by the *chronic prediction error history* of the observable being predicted. Cracking a long-standing anomaly earns far more than confirming a well-understood pattern.

```metta
; Reward scaled by how surprising the resolution is
(= (compute-epiphany-reward $observable $prediction-error)
   (let* (
     ($base-reward (get-config metabolic-boost))
     ($alpha (get-config epiphany-scaling))            ; e.g., 2.0
     ($chronic (get-chronic-error $observable))         ; EMA of recent errors
     ($relief (max 0.0 (- $chronic (abs $prediction-error))))
     ($bonus (* $alpha $relief))
   )
   (min (get-config epiphany-reward-cap)               ; e.g., 0.25
        (+ $base-reward $bonus))))

; Chronic error: exponential moving average from error trace buffer
(= (get-chronic-error $observable)
   (let* (
     ($traces (collapse (match &self
       (error-trace $observable $error $surprise $time)
       (abs $error))))
     ($n (length $traces))
   )
   (if (== $n 0) 0.0
     (ema $traces (get-config chronic-error-decay)))))  ; e.g., 0.9
```

**The selection pressure is on *what problem you solve*, not *how novel you are*.** A new concept that correctly predicts something the system was already predicting fine earns base reward. A new concept that cracks a long-standing anomaly — where `chronic_error` has been high for many cycles — earns the full epiphany bonus.

**Concrete example.** Observable $X$ has had chronic prediction error of 0.8 for 50 cycles. A newly invented concept correctly predicts $X$ with error 0.05:

```
relief = 0.8 - 0.05 = 0.75
bonus  = 2.0 × 0.75 = 1.5
reward = min(0.25, 0.05 + 1.5) = 0.25 (capped)
```

This is 5× the base reward. The concept receives a massive one-time energy injection. If it resolves chronic errors across multiple observables, the injections compound across consecutive cycles.

**The reward cap is essential.** Without it, a concept resolving a catastrophic error (chronic_error = 3.0) would earn `0.05 + 2.0 × 3.0 = 6.05`, instantly hitting the energy cap and possibly creating numerical instability in batch propagation. The cap (`epiphany-reward-cap = 0.25`) bounds the maximum per-event reward to 5× base, which is enough to create strong selection pressure without destabilizing the economy.

**Interaction with chain credit.** The epiphany bonus amplifies the *base reward* before chain credit propagation. When a D=1 atom earns an epiphany bonus of 0.25, its D=2 parent receives `δ × 0.25 = 0.075` instead of `δ × 0.05 = 0.015`. The insight "propagates upward with amplification" — the structural scaffolding that enabled the breakthrough benefits proportionally. This creates a strong selection pressure favoring abstract concepts that generate *surprising* predictions over those that merely confirm known patterns.

**Convergence with affect.** The epiphany bonus and the affect system are tracking the *same underlying event* through two different subsystems. The valence spike in `affect.metta` (derived from the rate of change of prediction errors) and the metabolic energy spike (from the epiphany bonus) both fire when chronic error drops sharply. This is convergent, not coincidental — the affect system reads the epistemic state, and the metabolic system rewards the same state change. They reinforce without redundancy: affect drives action selection (EFE uses valence/arousal), while the epiphany bonus drives structural selection (metabolic energy determines what concepts survive).


## Emergent Parsimony (Occam's Razor Without a Module)

A natural question: should the system explicitly reward *compression* — discovering that ten complex rules can be replaced by one simple principle? In active inference, free energy = accuracy minus complexity. The credit market optimizes for accuracy; does it need an explicit complexity penalty?

The answer is that parsimony pressure already *emerges* from the six mechanisms without an explicit Occam module:

When a new, simpler concept makes an intermediate node redundant, the redundant node stops earning reward (its predictions are now made better by the new concept's descendants). It continues paying rent. It dies. The energy it consumed — via rent subsidy, chain credit — is now freed in the sense that the system's total energy drain decreased. Meanwhile, the new concept earns its own energy through the epiphany bonus (if it resolved chronic errors) and chain credit from its own descendants.

The compression benefit is thermodynamic: a simpler structure has fewer atoms paying rent, so the same total predictive income supports higher per-atom energy. Reducing the number of components while maintaining output increases energy density per component. No explicit transfer mechanism is needed; the metabolic economy naturally selects for parsimony through the elimination of overhead.

If an explicit parsimony signal is desired (beyond this indirect thermodynamic pressure), it can be implemented as a light self-model observable: a small boost to the `structural-health` belief when total atom count decreases while total prediction accuracy is maintained or improved. This is a system-level signal, not an atom-level transfer, and it stays within the existing self-model architecture. We note this as an optional enhancement rather than a core mechanism, since the thermodynamic argument is sufficient and adding an explicit module would violate the principle of not scripting behaviors that can emerge.


## Three Regimes of Survival

Bidirectional chain credit creates a natural taxonomy of how atoms at different depths sustain themselves:

### Regime 1: Empirical (D=1–2)
Atoms earn reward directly from successful predictions (amplified by the epiphany bonus when resolving chronic errors). Chain credit from dependents is a minor supplement; chain debit from failing dependents is a minor drag. This is the existing metabolic economy with affect-coupled reward — the credit market's structural mechanisms are largely invisible at this depth.

### Regime 2: Structural (D=3–4)
Direct prediction is rare or impossible. Survival depends on *net* chain flow: chain credit from successful descendants minus chain debit from failing descendants. These atoms function as *useful scaffolding* — they survive if and only if their descendants are right more often than they're wrong (above the ~17% breakeven at default parameters). A D=4 concept with many accurate descendants thrives; one with mostly-failing descendants is actively punished. Parasitic "superstitious" concepts that survive on minority successes are eliminated by the debit mechanism.

### Regime 3: Cultural (D=5+)
Chain credit from empirical descendants is negligible (`δ^4 = 0.0081` per chain), and chain debit is similarly attenuated. These atoms survive primarily through the existing *communication reward* channel in `grounding_hypotheses.metta`: when the system uses a deep concept in self-explanation and the conversational partner understands, that's reward. When a concept is acquired via conversational grounding and subsequently used in successful reasoning, that's reward.

This maps directly onto human cognition. Shallow concepts survive by predicting. Mid-level concepts survive by being useful scaffolding. Deep concepts survive by being *culturally transmitted* — taught, discussed, shared. A concept so deep that it can't earn chain credit *and* can't earn conversational reward is a concept that dies. This is arguably correct: a concept that neither predicts nor communicates is genuinely useless.

**The "string theory" test.** In the current physics community, string theory survives because: (a) it connects to successful lower-level predictions (quantum field theory, which it extends), giving chain credit; and (b) it's intensely communicated and taught, giving cultural reward. If it lost *both* — stopped connecting to any successful prediction chain *and* stopped being discussed — it would indeed be pruned from the scientific knowledge base. The metabolic metaphor is more apt than it first appears.


## Emergent Behavior: The Paradigm Shift (Kuhnian Collapse and Reconstruction)

Because all six mechanisms are dynamically calculated each cycle based on the *current* state of the system, the architecture naturally replicates the full arc of scientific paradigm shifts — not just the collapse, but the reconstruction:

### Phase I: Tower Building
The system discovers a reliable foundation (e.g., Newtonian physics). The foundational atoms reach maximum energy (2.0). Their descendants predict successfully across many domains. Chain credit flows upward; chain debit is minimal (high success rate). Deep concepts thrive on both sides of the ledger.

### Phase II: Abstraction Boom
Deep concepts (D=5, D=6) are generated on top of this foundation. The foundation is at max energy → massive rent discounts. The descendants predict successfully → chain credit flows upward. The deep concepts accumulate energy rapidly.

### Phase III: The Anomaly
A new, persistent observation contradicts the foundation. Prediction errors spike for the D=1 atoms. The chronic error EMA for the affected observables begins climbing.

### Phase IV: The Triple Collapse
Three mechanisms fire simultaneously against the old tower:

1. **Rent subsidy collapse.** The foundational atoms fail to predict, so their energy drops. The rent subsidy for deep concepts vanishes → effective rent snaps to base rate.
2. **Chain credit collapse.** The D=1 descendants stop earning reward → no chain credit propagates upward.
3. **Chain debit surge.** The D=1 descendants are now *failing* predictions → chain debit propagates upward, actively draining the deep concepts.

The tower loses rent protection, income, and is actively penalized — a triple squeeze. High-D concepts at typical energy (0.5) die in ~15 cycles under the combined pressure of base rent (0.02) plus chain debit (~0.01) = 0.03/cycle effective drain.

### Phase V: The Epiphany
Meanwhile, hypothesis generators (induction, abduction, blending) are active. A new concept is created that correctly predicts the anomalous observable — the one with a chronic error EMA of 0.8 built up over the previous 50 cycles.

The epiphany bonus fires: reward = `min(0.25, 0.05 + 2.0 × 0.75) = 0.25`, a 5× amplification. The new concept rockets toward the energy cap. Chain credit from this amplified reward propagates upward to *its* structural parents at 5× the normal rate.

### Phase VI: The New Tower
The new concept's structural parents — the beginnings of a new theoretical framework — receive amplified chain credit. They rapidly accumulate energy, begin subsidizing *their* dependents, and create the conditions for a new abstraction boom. The affect system registers the resolution: chronic high arousal and negative valence (frustration/confusion during the anomaly) shift to positive valence spike (relief/euphoria) as the chronic errors resolve.

**Result:** A complete cognitive paradigm shift — the old conceptual tower is demolished by triple metabolic pressure while a new one is rapidly erected by amplified reward. The system didn't just replace one fact; it experienced the cognitive equivalent of frustration, confusion, insight, and relief, all emerging from the metabolic equations. The timeline compresses compared to mechanisms 1–4 alone: the old tower dies faster (chain debit accelerates collapse) and the new tower grows faster (epiphany bonus accelerates establishment).

### The Astrology → Astronomy Scenario

To make this concrete with a complete example:

1. **The Trap.** The system invents "Astrology" (D=4). It has large fan-out: 100 descendants predicting celestial observations. 10% of predictions succeed by coincidence.
2. **The Slow Bleed.** Chain debit from 90 failing descendants: `90 × δ³ × 0.01 = 0.0243/cycle`. Chain credit from 10 successes: `10 × δ³ × 0.05 = 0.0135/cycle`. Net: `-0.0108/cycle` plus rent. Astrology is slowly dying, but the minority successes keep it alive longer than a concept with zero successes. Affect registers chronic frustration (persistent prediction errors across many observables).
3. **The Leap.** Hypothesis generation creates "Astronomy" (D=4) — a concept whose descendants correctly predict planetary motion.
4. **The Epiphany.** The observables that Astronomy predicts have had chronic errors of ~0.6 for many cycles (from Astrology's failures). The epiphany bonus fires: Astronomy's descendants earn 3–5× base reward. Amplified chain credit floods upward. Astronomy reaches the energy cap within ~10 cycles.
5. **The Death of Dogma.** Astrology, already bleeding from chain debit and now stripped of even its 10% success (Astronomy's descendants predict those observables better), faces maximum rent + maximum chain debit + zero income. It dies in ~15 cycles.
6. **The New Paradigm.** Astronomy, at max energy, subsidizes a new theoretical tower. The affect system registers the shift: valence spikes positive, arousal drops from chronic high to moderate. The system has gone from confused to confident.


## Interaction with Existing Subsystems

### Grounding Hypotheses

Grounding hypotheses (`grounding_hypotheses.metta`) already have their own metabolic parameters (`grounding-metabolic-rate: 0.02`, `grounding-metabolic-boost: 0.08`). A grounding hypothesis for a deep atom faces the same temporal distance problem as the atom itself — the grounding may not be tested (via communication) for many cycles.

**Decision:** Grounding hypotheses inherit the structural depth of the atom they ground. A grounding hypothesis for a D=4 concept has D=4 for gestation and subsidy purposes. This is justified because the grounding's testability is directly coupled to the concept's integration into active reasoning — if the concept is too deep to predict, its grounding is too deep to test.

Chain credit does *not* propagate through grounding hypotheses (they are semantic annotations, not structural dependencies). Their reward comes through the existing communication and self-model channels.

### The Dimensionless Ratios

The existing sensitivity analysis shows the system's health is governed by `c_reward / c_rent ≈ 2.5` (median healthy). The credit market introduces three additional dimensionless quantities:

| Quantity | Default | Controls |
|----------|---------|----------|
| `c_reward / c_rent` | 2.5 | Base metabolic viability (existing) |
| `δ` | 0.3 | Chain propagation depth |
| `c_failure / c_reward` | 0.2 | Failure penalty severity; sets breakeven success rate at ~17% |
| `α × typical_chronic_error / c_reward` | ~28 | Epiphany amplification relative to base (capped at 5×) |

With depth-varying rent and bidirectional chain credit, the effective metabolic ratio becomes atom-specific:

| Atom type | Effective ratio |
|-----------|----------------|
| D=1, no subsidy, no chronic error | `0.05 / 0.02 = 2.5` (unchanged) |
| D=1, high chronic error resolved | `0.25 / 0.02 = 12.5` (epiphany) |
| D=2, full subsidy | `0.05 / 0.004 = 12.5` |
| D=4, 100 descendants, 80% success | `net_chain_flow / 0.002` (see below) |

For the D=4 case: `income = 80 × δ³ × 0.05 = 0.108`, `debit = 20 × δ³ × 0.01 = 0.0054`, `net = 0.1026`, ratio = `0.1026 / 0.002 = 51.3` — very healthy. At 20% success: `income = 0.0027`, `debit = 0.0216`, `net = -0.0189` — dying regardless of rent.

The existing 67% healthy basin result applies unchanged to D=1 atoms. The sensitivity analysis should be extended to sweep `(c_rent, c_reward, δ, c_failure)` to characterize the joint healthy basin for the full depth hierarchy (Test 15).

### Existing Gestation

The current system uses `(config gestation-period 3)` in `structure_learning.metta`. The `effective-gestation` function replaces this for lifecycle-managed atoms. For D=1 atoms: `3 × (1 + floor_log2(1)) = 3 × 1 = 3` — identical to existing behavior. The existing gestation path for Phase 1 passive models (in `structure_learning.metta`) is not modified. The extended gestation applies only to Phase 2+ atoms managed by `atom_lifecycle.metta`.


## Cycle Integration

The changes integrate into the existing `metabolic-step!` in `atom_lifecycle.metta` and the reward pathway in `structure_learning.metta`:

```metta
; === RENT: replace flat drain with depth-modulated drain ===

; OLD:
(= (drain-lifecycle-energy! $type $name)
   (let $current (get-lifecycle-energy $type $name)
     (set-lifecycle-energy! $type $name
       (- $current (get-config metabolic-rate)))))

; NEW:
(= (drain-lifecycle-energy! $type $name)
   (let* (
     ($current (get-lifecycle-energy $type $name))
     ($effective-rent (compute-effective-rent $type $name))
   )
   (set-lifecycle-energy! $type $name
     (- $current $effective-rent))))


; === REWARD: epiphany bonus + bidirectional chain propagation ===

; In structure_learning.metta, replace reward-passive-model! call:
(= (reward-with-credit-market! $cause $effect $actual-error)
   (let $abs-error (abs $actual-error)
     (if (< $abs-error (get-config surprise-threshold))
       ; === SUCCESS PATH: epiphany bonus + upward chain credit ===
       (let* (
         ($epiphany-reward (compute-epiphany-reward $effect $actual-error))
         ($boost (boost-pm-energy! $cause $effect $epiphany-reward))
       )
       (propagate-chain-credit! passive-model
         (pm-name $cause $effect)
         $epiphany-reward))
       ; === FAILURE PATH: upward chain debit ===
       (propagate-chain-debit! passive-model
         (pm-name $cause $effect)
         (get-config chain-failure-penalty)))))
```

The integration points are minimal:
- `drain-lifecycle-energy!` is modified in-place (rent modulation)
- `reward-passive-model!` is extended, not replaced (epiphany bonus wraps existing boost)
- Chain credit/debit propagation is new code called from the existing reward pathway
- All other lifecycle logic (gestation checks, pruning, promotion) remains identical


## Configuration Parameters

```metta
; In foundations.metta — Epistemic Credit Market
;
; These interact with existing metabolic parameters:
;   metabolic-rate       = 0.02  (base rent, unchanged)
;   metabolic-boost      = 0.05  (base reward, unchanged)
;   metabolic-energy-cap = 2.0   (max energy, unchanged)
;   gestation-period     = 3     (base grace period, unchanged)

; --- Mechanisms 1-3: Depth, Gestation, Subsidy ---
(config epistemic-subsidy-weight 0.008)    ; γ: parent energy → rent discount
(config minimum-metabolic-rate 0.002)      ; Absolute rent floor (prevents immortality)

; --- Mechanism 4: Chain Credit ---
(config chain-credit-decay 0.3)            ; δ: reward attenuation per depth level
(config chain-credit-floor 0.0001)         ; Minimum credit to propagate (prevents dust)

; --- Mechanism 5: Chain Debit ---
(config chain-failure-penalty 0.01)        ; c_failure: penalty per failed descendant prediction
                                           ; Set to 0.2 × c_reward. Breakeven success rate: ~17%

; --- Mechanism 6: Epiphany Bonus ---
(config epiphany-scaling 2.0)              ; α: chronic error → reward multiplier
(config epiphany-reward-cap 0.25)          ; Max per-event reward (5× base)
(config chronic-error-decay 0.9)           ; EMA decay for chronic error tracking
```

**Parameter interactions:**

- Maximum rent discount: `γ × energy_cap = 0.008 × 2.0 = 0.016`
- Minimum effective rent: `max(rent_floor, base_rent - max_discount) = max(0.002, 0.004) = 0.004` (rent floor only binds when parents are at cap AND γ > 0.009)
- Chain credit horizon: `log(credit_floor / base_reward) / log(δ) = log(0.0001 / 0.05) / log(0.3) ≈ 5.2` — credit propagates effectively to depth ~5
- Breakeven descendant success rate: `c_failure / (c_reward + c_failure) = 0.01 / 0.06 ≈ 17%`
- Maximum epiphany reward: `min(epiphany_cap, base_reward + α × max_chronic_error)` — capped at 0.25 regardless of chronic error magnitude
- Slowest death (max subsidy, no reward, no debit): `energy_cap / rent_floor = 2.0 / 0.002 = 1000 cycles`
- Fastest death (no subsidy, active debit, initial energy): depends on fan-out, but worst case with 100 failing descendants: `1.0 / (0.02 + 100 × δ³ × 0.01) ≈ 1.0 / 0.29 ≈ 3.4 cycles`
- Fastest establishment (epiphany bonus every cycle): `energy_cap / epiphany_cap = 2.0 / 0.25 = 8 cycles` to reach energy cap


## Validation Strategy (Python Benchmarks)

### `test_epistemic_credit.py`

**Test 1: Depth Computation.** Construct a hierarchy: D=1 link → D=2 latent → D=3 deduced link → D=4 blend. Verify depths are correctly computed and stored. Verify DAG assertion catches a cyclic dependency.

**Test 2: Gestation Scaling.** Verify that D=1 atoms receive gestation=3, D=2 atoms receive gestation=3, D=4 atoms receive gestation=9, D=8 atoms receive gestation=12.

**Test 3: Rent Modulation.** Seed a D=3 concept with parents at varying energy levels (0.0, 0.5, 1.0, 2.0). Verify effective rent matches the formula. Verify rent never drops below the floor.

**Test 4: Chain Credit Propagation.** Construct a D=1 → D=2 → D=3 chain. Trigger a reward event at D=1. Verify D=2 receives `δ × R`, D=3 receives `δ² × R`. Verify propagation stops when credit < `chain-credit-floor`.

**Test 5: Chain Debit Propagation.** Same chain as Test 4. Trigger a *failure* event at D=1. Verify D=2 receives `-δ × c_failure`, D=3 receives `-δ² × c_failure`. Verify that chain debit reduces energy but does not push below zero (clamped).

**Test 6: Epiphany Bonus — Chronic Error Scaling.** Seed an observable with 50 cycles of high prediction error (chronic error EMA ≈ 0.7). Create a new concept that predicts it correctly. Verify the reward is `min(cap, base + α × relief)` — significantly above base reward. Then seed an observable with 50 cycles of *low* prediction error. Predict it correctly. Verify the reward is approximately base (no bonus for confirming known patterns).

**Test 7: Epiphany Bonus — Reward Cap.** Seed an observable with extreme chronic error (EMA ≈ 2.0). Verify the reward is capped at `epiphany-reward-cap`, not the uncapped formula result.

**Test 8: Regime 1 — Empirical Survival.** Seed a D=1 atom with regular successful predictions. Verify behavior is identical to the existing system (credit market is invisible at D=1, except for minor epiphany bonus if resolving novel errors).

**Test 9: Regime 2 — Structural Survival.** Seed a D=4 concept that never directly predicts, but whose D=1 descendants predict successfully every cycle. Verify the D=4 concept reaches positive energy equilibrium (chain credit > effective rent). Then verify: a D=4 concept with only 1 intermittent descendant slowly dies (chain credit < rent floor).

**Test 10: Regime 2 — Superstition Death.** The critical test for chain debit. Seed a D=4 concept with 100 descendants, 90 failing and 10 succeeding. Verify the concept bleeds energy (net chain flow is negative) and dies. Compare against an identical setup *without* chain debit (Mechanism 5 disabled): verify the concept survives on its 10% success rate. This demonstrates that chain debit is necessary to kill superstitious index funds.

**Test 11: Regime 3 — Cultural Survival.** Seed a D=6 concept. Verify that chain credit alone is insufficient for survival. Inject communication reward events. Verify survival only when communication reward supplements chain credit.

**Test 12: The Rent Floor.** Seed a D=4 concept with a healthy foundation (full subsidy) but zero descendants (no chain credit, no chain debit) and no communication events. Verify it eventually dies due to the rent floor. Duration: `initial_energy / rent_floor = 1.0 / 0.002 = 500 cycles`.

**Test 13: The Full Paradigm Shift.** The integration test for all six mechanisms. Seed a D=5 conceptual tower ("Astrology") with mixed prediction success (10% accurate). Run 50 cycles:
- Verify Astrology is slowly bleeding from chain debit (failing descendants)
- Verify chronic error EMAs are building for the affected observables
- At cycle 51, introduce a new D=4 concept ("Astronomy") that correctly predicts the chronic-error observables
- Verify the epiphany bonus fires: Astronomy earns 3-5× base reward per correct prediction
- Verify amplified chain credit propagates upward through Astronomy's structural parents
- Verify Astrology collapses within ~15 cycles (triple squeeze: rent + debit + loss of credit)
- Verify Astronomy reaches energy cap within ~10 cycles
- Verify total tower replacement completes within ~30 cycles of the introduction

**Test 14: Affect Convergence.** During the paradigm shift (Test 13), track both the affect system (valence, arousal) and the metabolic system (energy levels, chain flows). Verify that:
- During the "frustration" phase (Astrology dominant, chronic errors high): negative valence, high arousal, net negative metabolic flow
- During the "epiphany" phase (Astronomy introduced, errors resolve): positive valence spike, arousal drop, net positive metabolic flow
- The two systems track the same transition through different representations

**Test 15: Sensitivity Extension.** Sweep `(c_rent, c_reward, δ, c_failure)` over a coarse grid. For each quadruple, run the Regime 2 test (D=4 hub with 10 descendants, 70% success rate). Map the healthy basin. Verify that the healthy region is connected and that `δ ∈ [0.15, 0.5]`, `c_failure/c_reward ∈ [0.1, 0.4]` produces reasonable behavior across base metabolic parameter ranges.


## Known Limitations & Risks

1. **Computational overhead of bidirectional propagation.** Both chain credit and chain debit propagate through the dependency graph — potentially twice per reward event (once on success, once on failure). This is $O(V + E)$ per event per direction. *Mitigation:* Batch propagation — accumulate all success and failure events in a cycle, compute net credit/debit per parent, then apply once. The `chain-credit-floor` limits depth naturally. In practice, the graph is shallow (most atoms are D=1–3) and sparse.

2. **Credit bubbles.** If `γ` (subsidy weight) is too high relative to the rent floor, or `δ` (chain credit) is too high, the system will tolerate too much noise. Chain debit mitigates this (Mechanism 5 actively punishes noisy concepts), but if `c_failure` is too low relative to `c_reward`, the superstition index fund problem returns at reduced severity. *Mitigation:* The sensitivity sweep (Test 15) characterizes the safe region across all four parameters simultaneously.

3. **Non-conservation of energy.** Chain credit creates energy; chain debit destroys it (beyond rent). If a highly connected hub has many descendants all predicting successfully, the hub accumulates energy quickly. *Mitigation:* The existing `metabolic-energy-cap` (2.0) bounds any single atom's energy. Total system energy is bounded by `N_atoms × energy_cap`. The generation cap (5 new atoms/cycle) bounds N_atoms growth.

4. **Depth as proxy for abstraction.** Structural depth measures graph distance from observables, which is a reasonable but imperfect proxy for conceptual abstraction. A long deductive chain of shallow empirical links (A→B→C→D→E, all directly testable) would have high D despite being empirically grounded at every step. *Mitigation:* This is conservative — such atoms would receive extended gestation and rent subsidies they don't strictly need, but chain credit would also flow abundantly, so the subsidy is redundant rather than harmful.

5. **Interaction with meta-parameter bootstrap.** The credit market introduces four new parameters (`γ`, `δ`, `c_failure`, `α`) alongside the rent floor and caps. The expanded parameter space may slow bootstrap convergence. *Mitigation:* These are second-order parameters. The bootstrap can start with fixed credit market parameters and extend to learning them once the base metabolic ratio has converged. The epiphany parameters (`α`, cap) are particularly safe to fix because they affect rare events, not steady-state dynamics.

6. **Fan-out amplification (mitigated by chain debit).** A concept that is a structural parent of many atoms receives chain credit from all successful descendants. In the original design (Mechanism 4 only), this created a risk of parasitic survival. Chain debit (Mechanism 5) directly addresses this: a parent with many descendants receives credit from successes *and* debit from failures, so net chain flow reflects actual predictive accuracy across the full fan-out.

7. **Epiphany bonus gaming.** A concept that alternates between predicting and not predicting an observable with chronic error could repeatedly trigger the epiphany bonus. *Mitigation:* The chronic error EMA decays toward zero as the observable is successfully predicted — after the first epiphany, the chronic error drops, and subsequent correct predictions earn diminishing bonuses. The bonus is self-extinguishing: the better the concept predicts, the less chronic error remains, the smaller the bonus becomes. Steady-state reward converges to base rate.

8. **Chain debit and exploratory hypotheses.** New hypotheses generated by abduction or induction will often fail initially (they're hypotheses, not established knowledge). Chain debit from their early failures could punish their structural parents before the hypotheses have had time to prove themselves. *Mitigation:* Gestation applies — new atoms don't generate reward or failure signals during their grace period. Chain debit only propagates from *post-gestation* atoms that fail. This preserves the exploration-exploitation balance: the system isn't punished for generating hypotheses, only for maintaining ones that keep failing after the grace period.
