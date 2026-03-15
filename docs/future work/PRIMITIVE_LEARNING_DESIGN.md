# Primitive Learning — Design Document

## The Problem

The 53 semantic primitives in `semantic_primitives.metta` are hand-curated termination points for grounding chains. They represent the system's "childhood education" — the irreducible concepts the architecture presumes before any experience.

This set is currently frozen at deployment. The system can learn new *grounded* concepts (via structure learning Phases 1–6), but it cannot learn new *primitives*. Every learned concept must ultimately decompose into the original 53. If a domain requires a concept that resists decomposition into the existing set, the system hits a hard ceiling: grounding-completeness drops, the four hypothesis generators cycle without progress, and the concept remains semantically opaque despite being metabolically healthy.

This is a representational bottleneck. For domain-general cognition, the primitive set must be extensible through experience.

## What Makes a Concept Primitive

The current primitives share a structural property: **decomposing them into other concepts doesn't reduce them — it merely translates them.** You can define `near` as "small distance," but `distance` is just `near`/`far` repackaged as a noun. The decomposition is lateral, not downward. Primitives are the **fixed points of semantic decomposition**.

The existing set clusters into five categories by grounding type:

| Type | Role | Examples | Count |
|------|------|----------|-------|
| Axiomatic | Structural vocabulary for building grounding chains | `causes`, `is-a`, `existence`, `same` | 33 |
| Perceptual | Irreducible sensory modalities | `see`, `hear`, `touch` | 3 |
| Experiential | Irreducible subjective states | `feel`, `pleasure`, `pain` | 3 |
| Derived | Defined via other primitives but atomic in usage | `away`, `kind-of`, `need`, `cannot`, `believe` | 7 |

**Key observation:** The axiomatic primitives are not just termination points — they are the *relations* that grounding chains are built from. `causes`, `enables`, `is-a`, `part-of` appear as the connective tissue in `obs-grounding` and `action-grounding` atoms. A new primitive that serves as relational vocabulary is categorically different from one that serves only as a terminus.

## Detection: When Does the System Need a New Primitive?

The architecture already has the signals. A new primitive is needed when:

### Signal 1: Persistent Grounding Failure

All four grounding hypothesis generators have been tried and failed for a metabolically healthy concept:

- **Inherited** — no shared primitives among member observables
- **Behavioral** — causal effects don't map to existing primitives
- **Compositional** — cannot decompose into primitive combinations
- **Conversational** — partner's label itself can't be grounded in existing primitives

The concept pays metabolic rent (it predicts successfully), but `grounding-completeness` cannot improve. This is the primary trigger.

```metta
; Detection criterion: metabolically healthy + grounding-stuck
(= (primitive-candidate? $concept)
   (let* (
     ($energy (get-grounding-energy $concept))
     ($age (grounding-age $concept))
     ($attempts (grounding-attempts-all-generators $concept))
     ($successes (grounding-successes $concept))
   )
   (and (> $energy (get-config grounding-healthy-threshold))   ; Paying rent
        (> $age (get-config primitive-min-age))                ; Not just new
        (> $attempts (get-config primitive-min-attempts))      ; Generators tried
        (< $successes 1))))                                    ; None succeeded
```

### Signal 2: Circularity in Decomposition

The compositional generator produces grounding chains that loop: concept A grounds via B, B grounds via A. Or the chain is purely lateral — every decomposition substitutes synonyms without reducing semantic depth.

Circularity detection requires tracking the grounding chain during traversal:

```metta
; Detect circular grounding (chain returns to starting concept)
(= (grounding-circular? $concept)
   (grounding-circular-with-visited? $concept ($concept)))

(= (grounding-circular-with-visited? $concept $visited)
   (let $targets (collapse (match &self (obs-grounding $concept $rel $t) $t))
     (if (== $targets ())
         False
         (any-in-visited-or-recurse? $targets $visited))))
```

### Signal 3: Irreducible Predictive Power

The concept carries information that no combination of existing primitives captures. This is testable: temporarily suppress the concept and measure whether prediction error increases in ways not compensable by existing primitives.

This is the strongest criterion but the most expensive to evaluate. It should be a confirmation step after Signals 1 and 2, not a primary trigger.

### Signal 4: Relational Load-Bearing (Strongest)

The concept appears as a *relation* in grounding chains for other concepts, not just as a terminus. If other concepts are being grounded *via* this concept — using it in the relation slot of `obs-grounding` atoms — it is structurally load-bearing. The grounding machinery needs it as vocabulary.

```metta
; Check if concept is used relationally by other groundings
(= (used-as-relation? $concept)
   (let $uses (collapse (match &self (obs-grounding $other $concept $target) $other))
     (> (length $uses) 0)))
```

## Phase 7: Primitive Promotion

### Position in the Lifecycle

| Phase | Trigger | Product |
|-------|---------|---------|
| 1 | Co-occurring prediction errors | Passive-model (causal link) |
| 1.5 | Two stable links A→B, B→C | Deduced link A→C |
| 2 | Dense causal cluster | Latent variable (hub node) |
| 3 | Latent variable survives | Promoted observable |
| 4 | Policy repeated ≥ threshold | Chunked action |
| 5 | Persistent unexplained error | Perceptual field |
| 6 | Learned atom without grounding | Grounding hypothesis |
| **7** | **Persistent grounding failure + criteria met** | **Learned primitive** |

Phase 7 sits at the top of the lifecycle because it is the rarest and most consequential transition. A concept must have survived all earlier phases and exhausted the grounding machinery before primitive promotion is considered.

### Promotion Criteria

A concept is promoted to primitive status when ALL of the following hold:

1. **Metabolically healthy** — energy above threshold for ≥ N cycles
2. **Grounding-exhausted** — all four generators tried ≥ M times with zero successful groundings
3. **Non-circular** — no valid decomposition chain exists (all attempted chains are circular or lateral)
4. **Irreducibly predictive** — suppression test shows prediction gap not compensable by existing primitives
5. **Optionally relational** — if the concept is used as a relation in other groundings, promotion priority increases

### Promotion Protocol

```metta
; Phase 7: Promote concept to learned primitive
(= (promote-to-primitive! $concept $cycle)
   (if (not (safe-to-mutate? (primitive (concept $concept) empirical)))
       (promotion-blocked $concept stratum-violation)
       (sequential
         ; Register as primitive with new grounding type
         (add-atom &ontology (primitive (concept $concept) empirical))
         
         ; Record origin for transparency
         (add-atom &state (atom-origin $concept phase-7
                    (primitive-promotion $cycle)))
         
         ; Initialize lifecycle energy
         (set-lifecycle-energy! primitive $concept
           (get-config primitive-initial-energy))
         
         ; Register in learned stratum (NOT constitutional)
         (add-atom &state (atom-stratum (primitive (concept $concept) empirical) learned))
         
         ; Trigger grounding re-evaluation for ungrounded concepts
         (trigger-grounding-reevaluation! $concept)
         
         (primitive-promoted $concept $cycle))))
```

### The Cascade: Unlocking Downstream Grounding

A new primitive doesn't just terminate its own chain. It potentially unlocks grounding for every concept that was waiting for a concept like it. When a primitive is promoted, all currently-ungrounded concepts should be re-evaluated:

```metta
; After promoting a new primitive, retry grounding for stuck concepts
(= (trigger-grounding-reevaluation! $new-primitive)
   (let $ungrounded (collapse (match &self (grounding-hyp $target $gen $cyc)
                                (if (not (has-successful-grounding? $target))
                                    $target
                                    ())))
     (map-reeval! $ungrounded $new-primitive)))

(= (map-reeval! () $p) done)
(= (map-reeval! ($target . $rest) $p)
   (sequential
     (reset-grounding-attempts! $target)
     (map-reeval! $rest $p)))
```

This cascade parallels how Phase 3 promotion (latent variable → observable) triggers new belief and action model creation. The system's representational vocabulary expands, and downstream systems adapt.

## Stratum Assignment: Learned, Not Constitutional

The existing 53 primitives are effectively constitutional — they are the cognitive physics of the system, the "childhood education." But a primitive discovered at runtime is epistemically different. It was learned, not innate.

| Primitive Type | Stratum | Rationale |
|----------------|---------|-----------|
| Original 53 | Constitutional | Presumed prior education; unreachable by internal process |
| Runtime-discovered | Learned | Protected by metabolic health, not immutability; revisable |

Placing learned primitives in the learned stratum means they remain subject to metabolic pressure. If a learned primitive stops being useful — its predictive power declines, or a better decomposition is later discovered — it can die like any other learned atom. This preserves the "pay rent or die" invariant across all representational levels.

A learned primitive that survives long enough and proves deeply load-bearing could be considered for goal-stratum promotion via Tier 3 protocol (authenticated operator decision). But this is an operational choice, not an architectural necessity.

## New Grounding Type: `empirical`

The existing grounding types are:

| Type | Meaning |
|------|---------|
| `axiomatic` | Cannot be further reduced |
| `perceptual` | Grounded in sensory observation |
| `experiential` | Grounded in subjective state |
| `derived` | Defined via other primitives |

We add:

| Type | Meaning |
|------|---------|
| `empirical` | Discovered through experience; resists decomposition into existing primitives |

This preserves transparency about epistemic status. Any query that traverses a grounding chain can distinguish between "this bottoms out at an axiom the system was born with" and "this bottoms out at a concept the system discovered it couldn't decompose further." The distinction matters for self-knowledge and for communication — the system can honestly report that a grounding chain terminates at a learned primitive, not a foundational axiom.

```metta
; New grounding type declaration
(: empirical GroundingType)

; Query functions work unchanged — is-primitive? already checks
; any grounding type. New function for epistemic transparency:
(= (is-learned-primitive? $sym)
   (case (match &self (primitive (concept $sym) empirical) True)
     ((True True)
      (() False))))

; Count learned vs innate primitives (self-knowledge)
(= (learned-primitive-count)
   (length (primitives-of-type empirical)))
```

## Contrast and Entailment Discovery

New primitives should participate in the contrast and entailment networks. Two approaches:

### Inherited Relations

If the concept was previously (unsuccessfully) grounded via concepts that have known contrast/entailment relations, those relations transfer as hypotheses:

```metta
; If $concept was behaviorally linked to "danger" and danger contrasts "safety",
; hypothesize that $concept contrasts whatever is anti-correlated with it
(= (hypothesize-contrast! $new-prim)
   (let $anti (find-anti-correlated-concept $new-prim)
     (if (not (== $anti ()))
         (sequential
           (add-atom &ontology (contrast (concept $new-prim) (concept $anti)))
           (add-atom &state (contrast-hyp $new-prim $anti inferred)))
         no-contrast-found)))
```

### Conversational Acquisition

The system can ask its partner: "What is the opposite of X?" This falls naturally out of the existing conversational grounding generator — extending it to contrast/entailment queries alongside meaning queries.

### Metabolic Selection

Contrast and entailment hypotheses are subject to the same metabolic selection as all other hypotheses. If a hypothesized contrast enables correct reasoning (predictions that use the contrast relation succeed), it earns energy. Otherwise it dies.

## Primitive Demotion

If a learned primitive's metabolic energy drops to zero, it should be demoted — removed from the primitive registry and returned to ordinary learned-concept status. This triggers:

1. Remove `(primitive (concept $concept) empirical)` from ontology
2. All grounding chains that terminated at this concept become ungrounded
3. Downstream concepts that were grounded via this primitive have their `grounding-completeness` recalculated
4. Those concepts re-enter the grounding hypothesis cycle

Demotion is the metabolic system working as designed. The concept was *hypothesized* to be primitive. If it stops earning rent, the hypothesis was wrong.

```metta
(= (demote-primitive! $concept)
   (if (is-learned-primitive? $concept)
       (sequential
         (remove-atom &ontology (primitive (concept $concept) empirical))
         ; Remove any contrast/entailment hypotheses
         (remove-contrast-hyps! $concept)
         (remove-entailment-hyps! $concept)
         ; Invalidate downstream groundings
         (invalidate-groundings-via! $concept)
         ; Log for transparency
         (add-atom &state (atom-origin $concept phase-7-demoted
                    (primitive-demotion (current-cycle))))
         (primitive-demoted $concept))
       (not-a-learned-primitive $concept)))
```

## Scale Expectations

The Natural Semantic Metalanguage (NSM) tradition, which the current primitive set most closely resembles, identifies approximately 60–65 semantic primes across human languages. The current set of 53 is in the right order of magnitude.

For a system operating in rich, open-ended domains, learned primitives might expand the set to 100–200. Beyond that, proliferation suggests the promotion criteria are too loose — concepts are being called primitive when they're actually decomposable.

A monitoring function for self-knowledge:

```metta
(= (primitive-health-report)
   (let* (
     ($innate (length (collapse (match &self (primitive (concept $p) axiomatic) $p))))
     ($perceptual (length (primitives-of-type perceptual)))
     ($experiential (length (primitives-of-type experiential)))
     ($derived (length (primitives-of-type derived)))
     ($empirical (length (primitives-of-type empirical)))
     ($total (+ $innate (+ $perceptual (+ $experiential (+ $derived $empirical)))))
   )
   (primitive-report $innate $perceptual $experiential $derived $empirical $total)))
```

## Files

### New File

| File | Est. Lines | Purpose |
|------|-----------|---------|
| `primitive_learning.metta` | ~400 | Phase 7 detection, promotion, demotion, cascade |

### Modified Files (additions)

| File | Addition | Purpose |
|------|----------|---------|
| `foundations.metta` | ~20 lines | New configs: `primitive-min-age`, `primitive-min-attempts`, `primitive-initial-energy` |
| `semantic_primitives.metta` | ~15 lines | `empirical` grounding type, `is-learned-primitive?`, count function |
| `safety.metta` | ~5 lines | Register learned primitives as learned stratum |
| `atom_lifecycle.metta` | ~30 lines | Phase 7 trigger check in lifecycle step |
| `cycle.metta` | ~15 lines | Wire Phase 7 into cognitive cycle (rare reflex) |
| `self_model.metta` | ~25 lines | Primitive health report, learned-primitive-count observable |
| `grounding_hypotheses.metta` | ~20 lines | Cascade: re-evaluate ungrounded on new primitive |

### Untouched Files

All existing grounding traversal in `semantic_grounding.metta` and `action_grounding.metta` works unchanged — `is-primitive?` already checks any grounding type, so learned primitives are immediately recognized as valid chain termini.

## Design Principles Compliance

| Principle | How This Design Complies |
|-----------|--------------------------|
| **Bottom-Up** | No enumeration of what concepts "should" be primitive. Promotion emerges from metabolic dynamics and grounding failure signals. |
| **Symbolic** | All detection, promotion, and demotion in MeTTa. No statistical primitive discovery. |
| **Transparent** | Every learned primitive tagged with `empirical` grounding type and `atom-origin` trace. Distinguishable from innate primitives in any query. |
| **Emergent** | Primitive status is not declared — it is the system's conclusion that a concept resists decomposition. |
| **Honest** | Grounding chains that terminate at learned primitives carry their epistemic status. The system knows the difference between "I was born knowing this" and "I discovered I can't decompose this further." |

## Open Questions

1. **Perceptual primitives from new sensors.** Phase 5 creates new perceptual fields (sensors). Should these automatically generate corresponding perceptual-type primitives? A new sensor modality is, almost by definition, a new irreducible channel. This might be a special case of Phase 7 that should fire immediately rather than waiting for grounding failure.

2. **Cross-agent primitive alignment.** If two Dagaz agents learn different empirical primitives, their grounding chains are incommensurable. Is there a protocol for negotiating shared primitive sets? The conversational grounding generator is the natural pathway, but it currently assumes primitives are shared.

3. **The suppression test cost.** Signal 3 (irreducible predictive power) requires temporarily suppressing a concept and measuring prediction degradation. In a live system, this is expensive and potentially disruptive. Can we approximate this test without actually suppressing — perhaps by measuring the mutual information between the concept's predictions and those of the nearest primitive cluster?

4. **Primitive set coherence.** The current 53 primitives have hand-curated contrast and entailment networks that are internally consistent. Learned primitives with hypothesized relations may introduce inconsistencies (e.g., a contrast that contradicts an entailment chain). Should there be a coherence check before promotion, or should metabolic pressure handle inconsistencies after the fact?

5. **The "right" size of the primitive set.** Is there a principled way to determine whether the primitive set is too small (concepts persistently fail to ground) vs. too large (primitives overlap and some are redundant)? A compression metric — the ratio of grounded concepts to primitives — might serve as a self-monitoring signal.
