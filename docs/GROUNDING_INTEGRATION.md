# Semantic Grounding Integration — Architecture Notes

## What This Module Does

`semantic_grounding.metta` connects three previously isolated subsystems:

```
BEFORE:                              AFTER:

┌──────────────┐                     ┌──────────────┐
│  Semantic     │  (island)          │  Semantic     │
│  Primitives   │                    │  Primitives   │──────────┐
└──────────────┘                     └──────────────┘          │
                                                                │ (grounds)
┌──────────────┐                     ┌──────────────┐          │
│  Dimensional  │  (island)          │  Dimensional  │──────┐   │
│  Primitives   │                    │  Primitives   │      │   │
└──────────────┘                     └──────────────┘      │   │
                                                            │   │
┌──────────────┐                     ┌──────────────┐      │   │
│  Beliefs +   │  (opaque tokens)    │  Beliefs +   │◄─────┴───┘
│  Actions     │                     │  Actions     │
└──────────────┘                     └──────────────┘
                                       ↕
                                     ┌──────────────────┐
                                     │ semantic_grounding│
                                     │  • obs-grounding  │
                                     │  • action-grounding│
                                     │  • qualify-belief  │
                                     │  • self-narrative  │
                                     │  • preferences!    │
                                     └──────────────────┘
```

## What It Provides

### 1. Observable Grounding (what ARE beliefs about?)
```metta
(obs-grounding power-level is-a quantity)
(obs-grounding power-level belongs-to self)
(obs-grounding power-level enables do)
```
Before: power-level was a token. Now: it's "a quantity the agent has that enables doing."

### 2. Action Grounding (what ARE actions?)
```metta
(action-grounding retreat causes (away threat))
(action-grounding retreat causes (less threat))
(action-grounding retreat requires power-level)
```
Before: retreat was a token. Now: it's "moving away from danger, which requires energy."

### 3. Belief Qualification (numeric → qualitative)
```metta
!(qualify-belief power-level)
→ (qualified power-level low 0.7 higher-better)
```
0.22 becomes "low" on the power scale. The system can reason qualitatively.

### 4. Domain Preferences & Viability Bounds (previously MISSING)
```metta
(preference power-level 0.9 1.0)
(viability-bound power-level 0.15 1.0)
```
These were referenced by actions.metta and policy_efe.metta but never declared for domain observables. Now they exist — and they're semantically grounded:
```metta
(pref-grounding power-level want (because (enables do)))
(viability-grounding power-level need (because (necessary (can do))))
```

### 5. Self-Narrative
```metta
!(what-am-i-doing)
→ (self-narrative
    (doing retreat)
    (which-is ((is-a do) (causes (away threat)) ...))
    (feeling (affect not-good moderate))
    (because-i-believe ((drives threat-level (error 0.35)) ...))
    (my-situation ((qualified power-level low ...) ...)))
```

### 6. Semantic Self-Queries
The system can ask about its own behavior through grounding chains:
- "Am I acting out of fear?" → check if action causes (away threat) and affect has negative valence
- "Am I acting to learn?" → check if action causes know
- "What's most at risk?" → find observable closest to viability bound

## Loader Integration

Add to `loader.metta` after semantic_primitives and before conversation_model:

```metta
; LAYER 2.5: GROUNDING INTEGRATION
; Connects primitives to operational system
!(import! &self core/semantic_grounding)
```

Updated architecture diagram:
```
LAYER 1: COGNITIVE CORE
  foundations → beliefs → affect → actions → cycle → policy_efe

LAYER 2: KNOWLEDGE MODULES  
  semantic_primitives → dimensional_primitives → self_model

LAYER 2.5: GROUNDING INTEGRATION  ← NEW
  semantic_grounding (connects layers 1 and 2)
  Also declares: domain preferences, viability bounds

LAYER 3: INTERACTION MODULE
  conversation_model
```

## Cycle Integration (Optional)

For logging/introspection, add to cognitive-cycle! in cycle.metta:

```metta
; After action selection, before execution:
($narrative (semantic-snapshot!))
```

This doesn't affect behavior (EFE is unchanged). It adds a structured record of what the system "knows" it's doing at each step.

## Dependencies

| References FROM semantic_grounding | Defined IN |
|---|---|
| `get-current-action` | actions.metta |
| `compute-valence`, `compute-arousal` | affect.metta |
| `all-prediction-errors` | beliefs.metta |
| `has-belief?`, `get-belief-value`, `get-belief-precision` | beliefs.metta |
| `max-scale-rank`, `scale-value` | dimensional_primitives.metta |
| `primitive` atoms | semantic_primitives.metta |
| `abs`, `min`, `max`, `clamp` | foundations.metta |

| Provides TO other modules | Used BY |
|---|---|
| `(preference $obs $pref $imp)` | actions.metta, policy_efe.metta |
| `(viability-bound $obs $min $max)` | actions.metta, policy_efe.metta |
| `(obs-dimension $obs ...)` | dimensional reasoning |
| `(what-am-i-doing)` | cycle.metta (optional), self_model |
| `(qualify-belief $obs)` | conversation_model, self_model |

## Design Decisions

**Q: Why declare grounding chains instead of learning them?**
Same reason as semantic primitives: grounding must terminate somewhere. The primitive set is curated. The grounding chains from domain concepts to primitives are also curated. What's *computed* is the traversal and the self-narrative — not the chains themselves.

**Q: Why not just have the LLM generate explanations?**
The LLM could produce "I'm retreating because threat is high." But that's a language trick. The *system* wouldn't know it's retreating — only the LLM wrapper would describe it. The grounding chains exist in the atomspace, traversable by the same pattern matching that drives behavior. The system can query its own narrative and reason over the results.

**Q: Does this change behavior?**
No. EFE computation is identical. What changes is that the system has a *semantic model* of what its actions and beliefs mean, queryable through the same machinery it uses for everything else.

## Extension: Grounding of Learned Structure

The grounding system described above handles **declared** observables and actions — atoms that exist in the deployment manifest with hand-authored grounding chains.

When structure learning invents new atoms (latent variables from Phase 2, promoted observables from Phase 3, chunked actions from Phase 4), those atoms are invisible to the grounding system. They predict well and pay metabolic rent, but they cannot be explained, reasoned about compositionally, or communicated.

`grounding_hypotheses.metta` closes this gap with four hypothesis generators:

| Generator | Trigger | Mechanism |
|-----------|---------|-----------|
| **Inherited** | New latent variable | Intersection of member observables' groundings |
| **Behavioral** | Learned atom with stable causal models | Ground by effects on already-grounded observables |
| **Compositional** | New chunked action | Sequential composition of component action groundings |
| **Conversational** | Persistent grounding gap + active conversation | Ask partner via orchestrator, parse label into grounding atoms |

All four generators produce atoms using the **same schemas** as declared groundings (`obs-grounding`, `action-grounding`, `obs-semantic`). The existing traversal machinery (`ground-term`, `bind-observable`, `what-am-i-doing`, `full-self-understanding`) works unchanged on learned groundings.

Selection is metabolic: hypothesized groundings pay rent and earn energy from successful communication and reasoning. Wrong groundings die.

A self-model observable, `grounding-completeness`, tracks what fraction of active learned atoms are semantically grounded. This creates continuous EFE pressure toward self-understanding: low completeness = high prediction error on self-model = info-gain for clarification actions. The system spontaneously seeks to understand its own concepts.

**Design document:** `GROUNDING_LEARNED_STRUCTURE.md`
**Implementation:** `grounding_hypotheses.metta`

**Q: Why not just extend the declared grounding to cover learned atoms?**
Because the whole point of structure learning is that the system invents concepts we didn't anticipate. We can't declare grounding chains for concepts that don't exist yet. The generators hypothesize groundings; metabolic selection validates them. Same pattern as the Peircean triad: multiple generators, one selection mechanism.
