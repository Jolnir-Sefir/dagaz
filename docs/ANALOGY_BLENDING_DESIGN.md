# Analogical Reasoning & Concept Blending — Design Document

## The Problem

The system can discover causal structure (induction), extend it
(deduction), and invert it (abduction). It can invent latent variables,
promote observables, and chunk actions. But it cannot do two things
that are central to flexible cognition:

1. **Analogical reasoning**: Recognizing that two domains share
   relational structure, and transferring knowledge across them.
   "This new situation is *like* that old situation, so what worked
   there might work here."

2. **Concept blending**: Combining two existing concepts to produce a
   novel third concept that inherits structure from both parents while
   potentially having emergent properties. "What if we had something
   that was like fire but also like water?"

Both are missing because the system has no mechanism to compare the
*relational structure* of two atoms and act on the comparison.

## The Physicist's Question

**What is the invariant?**

For analogy: **relational isomorphism**. An analogy preserves the
*relations* between entities while substituting the entities themselves.
If fire→smoke is like flood→mud, the invariant is the "causes observable
evidence" relation, not the specific entities. This is the same principle
as a symmetry transformation in physics — the labels change, the
structure doesn't.

For blending: **selective inheritance under metabolic pressure**. A
blend is a hypothesis that combines structure from two parents. The
invariant is that the blend's survival depends on the same criterion
as everything else: does it predict?

**What are the true degrees of freedom?**

Not the entities. The relational skeleton. Two concepts that look
completely different (fire vs. disease) can be analogous because they
share the same relational DOF: "hidden cause → multiple observable
symptoms → detection via investigation."

## Architecture

### New File

| File | Lines | Purpose |
|------|-------|---------|
| `analogy_blending.metta` | ~600 | Structural signatures, analogy detection, concept blending, metabolic management |

### Modified Files

| File | Addition | Purpose |
|------|----------|---------|
| `foundations.metta` | ~20 lines | New types + config |
| `safety.metta` | ~5 lines | Learned stratum for new atom types |
| `cycle.metta` | ~15 lines | Wire into cognitive cycle |

### Untouched Files

All existing modules. Analogy and blending consume existing structure
(obs-grounding, concept-ground, passive-model) but don't modify it.
They produce new atoms using standard schemas.


## Key Concept: Structural Signatures

The foundation of both analogy and blending is the **structural
signature** — a normalized description of an atom's relational
neighborhood. Two atoms are structurally similar if their signatures
overlap.

A signature consists of:

```
Signature(X) = {
  ; Semantic relations (what it IS)
  (obs-grounding X $rel $target)  → ($rel, type-of($target))
  (concept-ground X $rel $target) → ($rel, type-of($target))

  ; Causal relations (what it DOES)
  outgoing: passive-model(X, $effect, ...) → (causes, type-of($effect))
  incoming: passive-model($cause, X, ...) → (caused-by, type-of($cause))

  ; Topological features
  causal-out-degree: count of outgoing passive models
  causal-in-degree: count of incoming passive models
  cluster-membership: latent variables containing X
}
```

The critical abstraction: targets are replaced by their *types* (via
grounding), not their identities. This is what makes the comparison
relational rather than featural. "Causes danger" and "causes injury"
are different features but the same relational skeleton: `(causes,
threat-type)`.

### Type Normalization

To compare targets, we normalize to the *shallowest grounding type*:

```
normalize(target) =
  if primitive(target)       → target itself
  if observable(target)      → first is-a from obs-grounding
  if concept(target)         → first is-a from concept-ground
  if ungrounded(target)      → unknown
```

This is deliberately shallow. Deep normalization (chasing grounding
chains to primitives) would make everything look the same — most
things eventually ground to `thing` or `property`. One-level
normalization preserves enough distinction to be useful.


## Analogical Reasoning

### What An Analogy IS (in this system)

An analogy is a *mapping* between two atoms' relational neighborhoods:

```metta
(analogy-map $source $target $cycle)
(analogy-correspondence $source $target $src-rel $src-norm $tgt-rel $tgt-norm)
(analogy-score $source $target $similarity)
(analogy-energy $source $target $energy)
```

The mapping records: which relations in the source correspond to which
relations in the target, and how similar the overall structures are.

### Similarity Computation

Given two structural signatures S_A and S_B, similarity is:

```
sim(A, B) = |matched-relations| / max(|S_A|, |S_B|)
```

A relation `(rel, type)` in S_A matches `(rel, type')` in S_B if:
- `rel == rel'` (same relation), AND
- `type == type'` OR `type` and `type'` share a common ancestor, OR
  one of them is `unknown`

This is a Jaccard-like coefficient over typed relation sets.

### Trigger Conditions

Analogy detection is a REFLEX, not selected by EFE. It fires when:

1. A new learned atom enters the system (latent variable created,
   observable promoted, concept blended) — compare it to all existing
   atoms of comparable type.

2. The system encounters persistent prediction failure in a domain —
   maybe knowledge from an analogous domain could help.

Both triggers are rate-limited by the analogy budget (max per cycle).

Condition 1 is the primary trigger: when something new appears, check
if something old is like it.

### Analogical Transfer

When an analogy with sufficient similarity is found, the system can
transfer knowledge across the mapping:

**Causal transfer**: If source has `passive-model(source, E, ...)` and
the analogy maps source→target but there is no `passive-model(target,
E', ...)` for the corresponding E', hypothesize one. The transferred
model enters the standard metabolic regime — it must pay rent and
predict.

**Grounding transfer**: If source has `obs-grounding(source, rel, X)`
and target lacks grounding for the corresponding relation, hypothesize
it. This connects to the grounding_hypotheses module: analogical
transfer is a fifth grounding generator.

**Action transfer**: If the system knows that `action-model(observe,
source, ...)` is effective, and target is analogous to source, seed
a weak action model for observing target.

All transfers are hypotheses at low confidence/weight, entering the
metabolic economy like any other hypothesis.

### Metabolic Management

Analogy mappings pay metabolic rent. They earn energy when:
- A transferred hypothesis (causal, grounding, or action) makes a
  correct prediction → the analogy that generated it gets rewarded
- The mapping is used in self-explanation ("this is like that because...")
  and communication succeeds

They die when:
- Transferred hypotheses consistently fail
- The mapping goes unused for many cycles

This naturally handles the "false analogy" problem: a structurally
attractive but empirically wrong analogy produces transferred hypotheses
that fail to predict, which die, which stops rewarding the mapping,
which dies.


## Concept Blending

### What A Blend IS (in this system)

A concept blend is a *new atom* that inherits relational structure
from two parent concepts:

```metta
(concept-blend $blend-name $parent-a $parent-b $cycle)
(concept-ground $blend-name $rel $target)  ; inherited/projected
(blend-origin $blend-name $rel $target $source-parent)  ; audit trail
(blend-energy $blend-name $energy)
```

The blend uses the standard `concept-ground` schema, so the existing
grounding chain traversal works on blended concepts with zero changes.

### Trigger Conditions

Blending fires when:

1. Two concepts have structural similarity above the blend threshold
   (higher than the analogy threshold — blending requires stronger
   correspondence than mere analogy).

2. The two concepts are *active* — they both have current beliefs with
   non-trivial precision, or they've recently participated in
   prediction. This prevents blending dormant concepts.

3. Neither concept is already a blend of the other (no recursive
   self-blending).

### Blending Algorithm

Given parents A and B with structural signatures S_A and S_B:

**Step 1: Generic Space** — Identify shared relations:
```
Generic = { (rel, type) | (rel, type) ∈ S_A AND (rel, type) ∈ S_B }
```

**Step 2: Full Inheritance** — The blend inherits ALL relations from
both parents, at reduced weight:
```
For each (rel, target) in S_A:
  add (concept-ground blend rel target) with audit (from parent-a)
For each (rel, target) in S_B not already present:
  add (concept-ground blend rel target) with audit (from parent-b)
```

**Step 3: Conflict Resolution** — When both parents have the same
relation but different targets (e.g., A `causes danger`, B `causes
growth`), both are inherited. The system doesn't pick one — metabolic
selection does. Whichever target predicts better survives.

**Step 4: Emergent Structure** — Check if the combined relations
produce new entailments or contrasts via the semantic network:
- If blend has both `(causes X)` and `(prevents X)`, flag as
  `(blend-tension $blend X)` — internal contradiction that may be
  productive (fire-water → steam: causes heat AND causes wetness)
- If blend has `(is-a A)` and `(is-a B)` where A and B are contrasts,
  flag as `(blend-novel $blend A B)` — the blend is genuinely novel

**Step 5: Metabolic Initialization** — The blend enters the economy
with standard initial energy and gestation period.

### Naming

Blends start unnamed — they're structural atoms, not symbols.
The conversational acquisition mechanism from `grounding_hypotheses`
can name them: the blend creates a grounding request, the orchestrator
asks the LLM "what do you call something that is [blend properties]?",
and the label is injected back.

### Self-Disassembly

If a blend's metabolic energy drops below a threshold (dying), it
can be decomposed: its relations are checked against each parent,
and any transferred causal hypotheses that originated from the blend
are marked for accelerated metabolic drain. The blend dies cleanly
without leaving orphaned hypotheses.


## Interaction Between Analogy and Blending

Analogy and blending are related but distinct:

| | Analogy | Blending |
|---|---------|---------|
| **Produces** | Mapping between existing atoms | New atom |
| **Modifies** | Nothing (transfers are separate hypotheses) | Creates concept-ground atoms |
| **Trigger** | New atom enters system | Two similar active concepts |
| **Similarity threshold** | Lower (0.4 default) | Higher (0.6 default) |
| **Use case** | Knowledge transfer, explanation | Novel concept creation |

The pipeline can chain: analogy detects similarity → if similarity
is very high, blending is triggered → the blend becomes a new concept
that may itself be analogized to other concepts.

Blending can also run *without* analogy when two concepts are brought
together by the conversational partner ("what if we combined X and Y?").


## Configuration

```metta
; Structural signature computation
(config sig-normalization-depth 1)       ; How deep to normalize targets

; Analogical reasoning
(config analogy-similarity-threshold 0.4)  ; Min similarity for mapping
(config analogy-transfer-weight 0.3)       ; Transferred hypotheses start weak
(config analogy-transfer-precision 0.15)   ; Low confidence in transfers
(config analogy-initial-energy 0.5)        ; Mappings start modest
(config analogy-metabolic-rate 0.03)       ; Rent per cycle
(config analogy-metabolic-boost 0.1)       ; Reward for successful transfer
(config analogy-max-per-cycle 2)           ; Budget cap

; Concept blending
(config blend-similarity-threshold 0.6)    ; Higher than analogy threshold
(config blend-initial-energy 0.8)          ; Blends start stronger (more invested)
(config blend-metabolic-rate 0.03)         ; Same rent as analogies
(config blend-metabolic-boost 0.1)         ; Reward for successful prediction
(config blend-max-per-cycle 1)             ; Very conservative — blends are expensive
(config blend-gestation-period 5)          ; Longer grace period (needs time to prove)
```


## Cycle Integration

The analogy/blending step runs AFTER the grounding step and BEFORE
action selection:

```
... grounding step ...
→ ANALOGY/BLENDING STEP (new):
    1. Compute structural signatures for recently created atoms
    2. Compare against existing atoms (analogy detection)
    3. For strong analogies: attempt knowledge transfer
    4. For very strong analogies between active concepts: attempt blend
    5. Metabolic management (drain + prune analogies and blends)
→ action selection ...
```

This ordering ensures that learned atoms have had a chance to get
grounded (from the grounding step) before being compared structurally.


## Validation Strategy

### Unit Tests (Python benchmark)

1. **Signature computation**: Verify structural signatures correctly
   capture relational neighborhoods.

2. **Similarity detection**: Two manually constructed concepts with
   known shared structure should produce expected similarity scores.

3. **Analogical transfer**: Set up source domain with known causal
   structure, create analogous target domain missing one link, verify
   transfer hypothesizes the missing link.

4. **False analogy death**: Create a structurally similar but causally
   wrong analogy, verify transferred hypotheses die metabolically
   within ~30 cycles.

5. **Concept blend creation**: Two concepts with high similarity
   should produce a blend inheriting relations from both.

6. **Blend tension detection**: Blend two concepts with contradictory
   relations, verify tension is flagged.

7. **Blend metabolic death**: Create a blend that doesn't predict,
   verify it dies and disassembles cleanly.

### Integration Tests

8. **Analogy-driven exploration**: New domain with high structural
   similarity to known domain. Verify the system transfers causal
   models and that EFE guides investigation of transferred hypotheses.

9. **Blend naming**: Create a blend, verify grounding request is
   generated, simulate conversational acquisition of a label.


## Known Limitations

1. **Shallow comparison**: Signatures compare at depth 1, so two
   concepts with isomorphic deep structure but different surface
   relations won't be detected as analogous. Deeper comparison is
   combinatorially expensive. The system discovers deep analogies
   gradually as structure learning fills in intermediate links.

2. **No cross-domain mapping persistence**: The analogy-map atoms
   record correspondences, but the system can't reason *about* the
   mapping itself (e.g., "the mapping between fire and disease breaks
   down for contagion"). Meta-analogical reasoning would require a
   second-order representation.

3. **Blend expressiveness**: Blends can only combine concepts that
   exist as grounded atoms. "What if we had an animal that could fly
   and swim?" requires both "flying animal" and "swimming animal" to
   be represented first. The system can't blend from raw natural
   language — that's the orchestrator's job.

4. **Scale**: Pairwise comparison of all concepts is O(n²). With a
   small concept vocabulary (< 100 active atoms), this is fine. At
   scale, the system would need locality-sensitive hashing on
   structural signatures. Not implemented.

5. **No analogical reasoning about *relations* themselves**: The
   system can map "A causes B" to "C causes D" but can't reason about
   "the causal relation in domain 1 is weaker than in domain 2."
   Analogies are structural, not quantitative.

6. **Blending is symmetric**: The system treats both parents equally.
   Asymmetric blends ("mostly fire with a little water") would require
   a weighting mechanism on parent contributions. Metabolic selection
   approximates this over time (one parent's contributions survive
   more), but the initial blend is symmetric.
