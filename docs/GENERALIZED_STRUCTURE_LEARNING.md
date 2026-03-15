# Generalized Structure Learning â€” Implementation Guide

## Thesis

Three AGI critiques (representational capacity, compositional generalization, bootstrapping) are one problem: the system cannot create new representational atoms from experience. The solution generalizes structure learning's metabolic selection to ALL atom types.

**One rule:** All representational atoms â€” observables, actions, causal links, latent variables, perceptual fields â€” are structurally equal. Created by surprise, selected by prediction, killed by metabolic pressure.

## Files

### New File
| File | Lines | Purpose |
|------|-------|---------|
| `atom_lifecycle.metta` | ~480 | Unified promotion/demotion engine for Phases 3-5 |

### Modified Files (additions)
| File | Addition | Purpose |
|------|----------|---------|
| `foundations.metta` | ~70 lines | New types + config for Phases 3-5 |
| `safety.metta` | ~30 lines | Learned strata for new atom types |
| `structure_learning.metta` | ~200 lines | Phase 3-5 triggers + coherence reward fix |
| `actions.metta` | ~60 lines | Dynamic action enumeration for Phase 4 |
| `cycle.metta` | ~120 lines | Wire Phases 3-5 into cognitive cycle |
| `self_model.metta` | ~170 lines | Lifecycle self-knowledge queries |
| `orchestrator.py` | ~200 lines | Dynamic perceptual parsing for Phase 5 |
| `loader.metta` | ~40 lines | Import atom_lifecycle + new entry points |

### Untouched Files (10)
`beliefs.metta`, `affect.metta`, `policy_efe.metta`, `action_learning.metta`, `semantic_primitives.metta`, `dimensional_primitives.metta`, `semantic_grounding.metta`, `action_grounding.metta`, `conversation_model.metta`, `planning.metta`

## Merge Order

### Phase 1: `foundations.metta`
Insert after existing Structure Learning types (Section I) and configs (Section II):
- New types: `PromotedObservable`, `ChunkedAction`, `PerceptualField`, `PolicyTrace`, `AtomStatus`
- New configs: `promotion-*`, `chunk-*`, `percept-*`, `coherence-reward-threshold`

### Phase 2: `safety.metta`
Insert after existing learned stratum entries in Section I:
- All Phase 3-5 atom types registered as `learned` stratum
- Includes audit trail types

### Phase 3: `atom_lifecycle.metta` (NEW FILE)
Place alongside `structure_learning.metta`. Dependencies: `foundations`, `beliefs`, `safety`.
- Section I: Lifecycle energy management
- Section II: Phase 3 â€” Observable promotion
- Section III: Phase 3 â€” Observable demotion
- Section IV: Phase 4 â€” Action chunking (reification)
- Section V: Phase 4 â€” Action retirement
- Section VI: Phase 5 â€” Perceptual expansion
- Section VII: Phase 5 â€” Perceptual deactivation
- Section VIII: Unified metabolic step (drain + prune all types)
- Section IX: Lifecycle reward
- Section X: Accessors and diagnostics

### Phase 4: `structure_learning.metta`
1. Add `!(import! &self atom_lifecycle)` at top
2. Replace `reward-latent-variable!` with `reward-latent-variable-coherent!` (Section IX) â€” coherence check prevents spurious promotion
3. Add Sections XII-XIV (Phase 3-5 trigger functions)
4. Replace `structure-learning-step!` with `structure-learning-step-v2!` (calls all 5 phases)
5. Replace `reward-structure-for!` with `reward-structure-for-v2!` (includes coherence + lifecycle reward)

### Phase 5: `actions.metta`
Add dynamic action enumeration after Section XI:
- `all-available-actions` queries atomspace instead of hardcoded list
- `compute-all-efes` works with variable-length action lists
- `find-min-efe-dynamic` handles arbitrary number of actions
- `select-action-dynamic` replaces `select-action-myopic`

### Phase 6: `cycle.metta`
1. Add `!(import! &self atom_lifecycle)`
2. Replace `cognitive-cycle!` with `cognitive-cycle-v2!` â€” passes errors + policy to structure learning
3. Replace `reward-all-structure!` with `reward-all-structure-v2!`
4. Replace `get-state` with `get-state-v2` â€” includes lifecycle summary
5. Replace `init!` with `init-v2!` â€” includes lifecycle initialization
6. Replace `explain` with `explain-v2`

### Phase 7: `self_model.metta`
Add Sections XIII-XVII after existing Section XII:
- XIII: Invented observables (Phase 3 introspection)
- XIV: Acquired skills (Phase 4 introspection)
- XV: Discovered senses (Phase 5 introspection)
- XVI: Unified vocabulary growth
- XVII: Extended structural complexity

### Phase 8: `orchestrator.py`
1. Add `dynamic_fields: dict` to `ParseResult`
2. Extend `Parser` with perceptual field awareness
3. Make `build_prompt()` query active fields and append extraction lines
4. Make `extract()` handle dynamic fields via generated regex
5. Add `inject_dynamic_observations()` call after static injection
6. Add `perceptual_fields` list to simulation mode

### Phase 9: `loader.metta`
Insert `!(import! &self core/atom_lifecycle)` between structure_learning and self_model imports. Add new entry points to quick reference.

## Architecture Diagram (After)

```
COGNITIVE CYCLE (v4)
    â”‚
    â”œâ”€ Existing: beliefs â†’ errors â†’ affect â†’ action selection â†’ learning
    â”‚
    â””â”€ Extended: structure learning v2
         â”‚
         â”œâ”€ Phase 1: Correlated errors â†’ passive model
         â”œâ”€ Phase 2: Dense cluster â†’ latent variable  
         â”‚
         â”œâ”€ Phase 3: Successful latent â†’ PROMOTED OBSERVABLE â”€â”€â†’ beliefs
         â”‚     (atom_lifecycle.metta: promote-to-observable!)      actions
         â”‚                                                          EFE
         â”‚
         â”œâ”€ Phase 4: Repeated policy â†’ CHUNKED ACTION â”€â”€â†’ any-action
         â”‚     (atom_lifecycle.metta: reify-chunk!)           action-cost
         â”‚                                                    action-model
         â”‚
         â”œâ”€ Phase 5: Unexplained error â†’ PERCEPTUAL FIELD â”€â”€â†’ orchestrator
         â”‚     (atom_lifecycle.metta: activate-perceptual-field!)  parser
         â”‚                                                         prompt
         â”‚
         â””â”€ ALL TYPES: metabolic-step! (pay rent or die)
               â”‚
               â””â”€ THE INVARIANT
```

## Key Design Decisions

### Why `atom_lifecycle.metta` is separate from `structure_learning.metta`
Promotion touches beliefs, action models, safety strata, and self-model. Putting it in `structure_learning.metta` would create circular dependencies. The lifecycle module sits between structure learning (which discovers) and the core (which uses).

### Why coherence check matters
Without it, latent variables earn metabolic energy during calm periods (small errors â†’ "correct" predictions) even when members aren't correlated. This leads to false promotion. The fix: latent variables earn rent only when member errors move in the same direction â€” the pattern the latent is supposed to explain.

### Why `any-action` works for Phase 4
MeTTa's nondeterministic evaluation means `(any-action)` returns ALL matching definitions. Adding `(= (any-action) $chunk)` at runtime makes the chunk immediately available to `policy_efe.metta`'s enumeration. Removing it on retirement makes it immediately unavailable. No code changes needed in `policy_efe.metta`.

### Why Phase 5 signals rather than directly creates
The MeTTa core detects it needs new sensors but can't know what the LLM can extract. The orchestrator (which constructs prompts) proposes candidate fields. This respects the architecture's boundary: MeTTa reasons symbolically, the orchestrator handles NL.

## Validation

The `generalized_structure_learning.py` benchmark (11/11 tests) validates all three phases in Python simulation. Key results:
- Phase 3: Latent promoted â†’ has beliefs + action models; spurious latent correctly dies
- Phase 4: Repeated policy chunked â†’ discounted cost, aggregate models; unused chunk dies
- Phase 5: Hidden variable discovered from persistent unexplained error
- Metabolic invariant: ALL five atom types die when unrewarded (uniform selection)
- Symmetry: promotion and demotion are fully reversible
- Conservation: total structure bounded by finite metabolic budget

## Extension: Phase 6 — Grounding of Learned Structure

Phases 3-5 create new representational atoms that participate in the cognitive loop (beliefs, actions, sensors). But these atoms have no **semantic grounding** — the system cannot explain what they mean, reason about them compositionally, or communicate about them to a conversational partner.

Phase 6 closes this gap. Four hypothesis generators produce standard grounding atoms (`obs-grounding`, `action-grounding`) for learned structure, under the same metabolic selection as all other atoms:

| Generator | Covers | Mechanism |
|-----------|--------|-----------|
| Inherited | Latent variables (Phase 2) | Intersection of member observables' existing groundings |
| Behavioral | Any learned atom with causal models | Ground by effects on already-grounded observables |
| Compositional | Chunked actions (Phase 4) | Sequential composition of component action groundings |
| Conversational | Any persistent grounding gap | Ask conversational partner via orchestrator |

The critical design insight: hypothesized groundings use the **same atom schemas** as declared groundings. The entire existing traversal machinery (`ground-term`, `bind-observable`, `what-am-i-doing`) works unchanged on learned atoms.

A new self-model observable, `grounding-completeness`, tracks the fraction of active learned atoms that are semantically grounded. This creates continuous EFE pressure toward self-understanding.

**Design document:** `GROUNDING_LEARNED_STRUCTURE.md`
**Implementation:** `grounding_hypotheses.metta` (~910 lines)
