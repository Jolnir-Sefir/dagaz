# Project Dagaz — Architecture

## Overview

A pure MeTTa implementation of Active Inference for domain-general cognition. The core invariant is **minimize Expected Free Energy (EFE)** — this single principle drives perception, action, learning, self-knowledge, and conversation. No mode switches, no scripted behaviors. 22 modules, ~12,200 lines of MeTTa, with a pure Python MeTTa evaluator for end-to-end execution and Python reference implementations validating every major subsystem.

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Bottom-Up** | No enumerated modes. Behavior emerges from EFE landscape. |
| **Symbolic** | All reasoning in MeTTa. LLM only at NL boundaries. |
| **Transparent** | Every inference traceable. Query any state. |
| **Emergent** | Capabilities derived from action statistics, not declared. |
| **Honest** | Can only claim what is grounded. "I don't know" from query failure. |
| **Minimal LLM** | LLM for NL→structure and structure→NL only. |

## Module Map

### Layer 1: Cognitive Core

The main cognitive loop and its dependencies.

| Module | Lines | Role |
|--------|-------|------|
| `core/foundations.metta` | ~430 | Types, utilities, configuration (all phases, grounding hypothesis types) |
| `core/beliefs.metta` | ~240 | World model, prediction error, Bayesian update |
| `core/affect.metta` | ~300 | Valence/arousal/dominance (DERIVED from errors, never stored) |
| `core/actions.metta` | ~580 | EFE computation, single-step action selection, viability |
| `core/action_learning.metta` | ~320 | Learn action models from experience (prediction → update) |
| `core/cycle.metta` | ~580 | Main cognitive loop (v1 single-step, v2 with lifecycle + grounding reflex) |
| `core/policy_efe.metta` | ~400 | Multi-step EFE planning (defeats temporal myopia) |
| `core/planning.metta` | ~720 | Fractal planning via RG-flow-inspired adaptive beam search |
| `core/safety.metta` | ~440 | Stratified immutability, mutation guards, constitutional watchdog, grounding strata |

### Layer 2: Structure Learning & Reasoning

Discovers causal structure from prediction errors. The Peircean triad.

| Module | Lines | Role |
|--------|-------|------|
| `core/structure_learning.metta` | ~1270 | Induction (Hebbian suspicion → causal links), deduction (transitive closure), hub detection (latent variables), Phases 1-5 triggers |
| `core/abduction.metta` | ~510 | Abductive inference (observed effects + known structure → hidden cause hypotheses) |
| `core/atom_lifecycle.metta` | ~780 | Metabolic selection engine: promotion (Phase 3), chunking (Phase 4), perceptual expansion (Phase 5) |
| `core/grounding_hypotheses.metta` | ~910 | Semantic grounding of learned structure: four generators (inherited, behavioral, compositional, conversational) under metabolic selection |

### Layer 3: Knowledge & Grounding

Semantic foundation and grounding chains.

| Module | Lines | Role |
|--------|-------|------|
| `core/semantic_primitives.metta` | ~300 | 53 primitives, 12 contrast pairs, 5 entailment relations |
| `core/dimensional_primitives.metta` | ~310 | Ordinal scales, comparison, dimensional mapping |
| `core/semantic_grounding.metta` | ~720 | Observable → primitive grounding chains, self-narrative, belief qualification |
| `core/action_grounding.metta` | ~590 | Action → primitive grounding chains, action description generation |
| `core/analogy_blending.metta` | ~890 | Cross-domain structural mapping and conceptual blending |

### Layer 4: Self-Model & Interaction

Meta-cognition, perception, and conversation.

| Module | Lines | Role |
|--------|-------|------|
| `core/self_model.metta` | ~780 | Meta-cognition, emergent capabilities/limitations, structural self-knowledge, grounding introspection |
| `core/conversation_model.metta` | ~530 | Active Inference over dialogue (partner model, discourse tracking) |
| `core/perception.metta` | ~580 | Observation processing, LLM parse integration |
| `core/proprioception.metta` | ~350 | Verbalization fidelity tracking |

### Runtime & Orchestration

| File | Lines | Role |
|------|-------|------|
| `dagaz_runtime.py` | ~400 | Pure Python MeTTa evaluator. Tokenizer, parser, unifier, function dispatch, space management. No domain-specific logic. |
| `orchestrator.py` | ~450 | Stateless LLM orchestrator. Perception (NL→observations), cognition (MeTTa cycle), verbalization (intent→NL). Epistemic firewall. |
| `core/loader.metta` | ~230 | MeTTa-native module loading, initialization, quick reference |

## Key Integration Points

### 1. Everything is Observables + Beliefs + EFE

The same machinery handles physical world, self-knowledge, conversation, and grounding completeness:

```metta
(belief terrain-roughness 0.3 0.7)        ; Physical world
(belief self-competence 0.7 0.5)           ; Self-knowledge
(belief partner-comprehension 0.6 0.5)     ; Conversation
(belief grounding-completeness 0.85 0.5)   ; Meta-cognitive
```

All feed into the same EFE computation. No special cases.

### 2. Actions Compete Uniformly

Domain actions and conversational actions use identical EFE:

```metta
(action-model observe terrain-roughness 0.0 0.08)   ; Physical
(action-model clarify partner-comprehension 0.15 0.08) ; Conversational

(= (compute-efe $action)
   (+ (- (+ $expected-error $cost) $info-gain) $viability-penalty))
```

### 3. Capabilities Emerge from Statistics

No declared capabilities. They emerge from action model performance:

```metta
(action-stats observe 7 10 0.15)  ; 7 successes, 10 attempts, 0.15 avg error
(= (can-do? $action)
   (let* (($sr (success-rate $action)) ($err (avg-error $action)))
   (and (> $sr 0.6) (< $err 0.3))))
```

### 4. Three Hypothesis Generators, One Selection Mechanism

The Peircean reasoning triad — induction, deduction, abduction — are all hypothesis generators producing passive-model atoms. The metabolic system selects: correct predictions earn energy, wrong predictions drain it. Same invariant for all three inference types.

This pattern repeats for grounding hypotheses: four generators (inherited, behavioral, compositional, conversational) produce `obs-grounding` atoms under the same metabolic selection.

### 5. Grounding Chains

Concepts are grounded if they trace to semantic primitives. Both declared and learned groundings use the same schema — the traversal machinery works on both without modification:

```metta
(primitive causes axiomatic)
(obs-grounding threat-level causes fear)       ; Declared
(obs-grounding $L_1 causes danger)             ; Learned (behavioral generator)
```

### 6. Self-Understanding as Observable

The agent tracks its own grounding completeness as a belief. Low completeness creates EFE pressure toward clarification — the system spontaneously seeks to understand its own learned concepts:

```metta
(belief grounding-completeness 0.7 0.5)
(preference grounding-completeness 1.0 0.6)
; Low completeness → high prediction error → info-gain for asking partner
```

## EFE Formula

```
EFE(action) = Expected_Error + Cost - Info_Gain + Viability_Penalty

Where:
- Expected_Error: Weighted prediction error after action
- Cost: Resource/effort cost of action
- Info_Gain: Precision improvement (epistemic value)
- Viability_Penalty: Bonus/penalty near viability bounds
```

Lower EFE = better action. Selection is argmin over available actions.

## Data Flow

```
                    ┌──────────────┐
     Observations → │   BELIEFS    │ → Prediction Errors
                    └──────────────┘
                           ↓
                    ┌──────────────┐
Prediction Errors → │    AFFECT    │ (computed, not stored)
                    └──────────────┘
                           ↓
                    ┌──────────────┐
  Errors + Models → │  STRUCTURE   │ → Causal links, latent variables
                    │   LEARNING   │   (Peircean triad + metabolism)
                    └──────────────┘
                           ↓
                    ┌──────────────┐
    Learned atoms → │  GROUNDING   │ → Semantic grounding hypotheses
                    │  HYPOTHESES  │   (4 generators + metabolism)
                    └──────────────┘
                           ↓
                    ┌──────────────┐
 Affect + Beliefs → │     EFE      │ → Action Selection
                    └──────────────┘
                           ↓
                    ┌──────────────┐
           Action → │   EXECUTE    │ → World Change
                    └──────────────┘
                           ↓
                    ┌──────────────┐
         Outcomes → │   LEARNING   │ → Update Beliefs + Action Models + Stats
                    └──────────────┘
```

## Cognitive Cycle (v2)

The full cycle in `cognitive-cycle-v2!`:

1. **Learn** from previous action (compare snapshot to current observations)
2. **Predict** — compute all prediction errors (beliefs vs observations)
3. **Surprise** — compute surprise per observable, record error traces
4. **Affect** — derive valence/arousal/dominance from errors (computed, not stored)
5. **Select action** — argmin EFE over available actions (policy, fractal, or single-step)
6. **Structure learning** (reflex) — update suspicion links, check Phase 1-5 triggers, metabolic pruning
7. **Abduction** (reflex) — hypothesize hidden causes from surprising effects + known structure
8. **Grounding** (reflex) — run four generators, metabolic management, update grounding-completeness
9. **Analogy & blending** (reflex) — structural comparison, knowledge transfer, concept merging
10. **Snapshot** beliefs + record action for next cycle's learning
11. **Update beliefs** — Bayesian update from prediction errors (adaptive learning rate)
12. **Reward structure** — metabolic energy to atoms that predicted correctly
13. **Execute** action (gated by pause check)
14. **Watchdog** — verify constitutional atom integrity

Steps 6-9 are reflexes — they fire when conditions are met, not selected by EFE. This prevents the system from having to "choose" to learn.

## Safety Architecture

Three-stratum immutability:

| Stratum | Mutability | Contents |
|---------|-----------|----------|
| Constitutional | Immutable | Viability bounds, safety axioms, type definitions |
| Goal | Authenticated protocol only | Task preferences, operational targets |
| Learned | Freely modifiable | Causal links, latent variables, grounding hypotheses |

Post-cycle watchdog counts constitutional atoms and halts the system if any have been added or removed. Input channel separation prevents peer communications from entering the corrigibility pathway. Mutation guards (`safe-to-remove?`, `safe-to-mutate?`) check per-atom stratum annotations before any modification — protection is logical (stratum-based), not physical (space-based).

## Structure Learning Phases

| Phase | Trigger | Product | Selection |
|-------|---------|---------|-----------|
| 1 | Co-occurring prediction errors | Passive-model (causal link) | Metabolic |
| 1.5 | Two stable links A→B, B→C | Deduced link A→C | Metabolic |
| 2 | Dense causal cluster (degree ≥ 3) | Latent variable (hub node) | Metabolic |
| 3 | Latent variable survives long enough | Promoted observable (full belief + EFE) | Metabolic |
| 4 | Policy repeated ≥ threshold times | Chunked action (named composite) | Metabolic |
| 5 | Persistent unexplained error | Perceptual field (new sensor) | Metabolic |
| 6 | Learned atom exists without grounding | Grounding hypothesis (semantic) | Metabolic |

One invariant across all phases: **pay rent or die.**

## Fractal Planning

Adaptive beam search inspired by Renormalization Group flow:

- Coarse-grain low-probability trajectories at each depth level
- Symmetry-breaking detection: when one action dominates, stop expanding alternatives
- Correlation-length cutoff: when all paths converge to same outcome, truncate depth
- Complexity: O(|A| × k × d) vs O(|A|^d) for exhaustive search
- Validated: 52× reduction in evaluations (depth 7, beam 2)

## Key Emergent Behaviors

These behaviors were NOT programmed. They emerge from the architecture:

- **Exploration under uncertainty**: High prediction error → high info-gain for observe → EFE selects observe
- **Retreat under threat**: Low viability → viability bonus for retreat → EFE selects retreat
- **Sherlock Holmes effect**: Abduction hypothesizes hidden cause → low-precision belief → high info-gain for confirming observation → system spontaneously investigates
- **Metabolic death**: Wrong hypotheses drain energy from failed predictions → die within ~30 cycles
- **Spontaneous self-inquiry**: Ungrounded learned concepts → low grounding-completeness → info-gain for clarification → system asks partner what its concepts mean

## Runtime Status

### Tier 1: MeTTa Source (Canonical)

22 modules written against correct MeTTa syntax. Native Hyperon execution is blocked by two bugs in 0.2.10:

1. **Cons-cell pattern matching** (`($head . $rest)`) does not reduce, breaking all recursive list functions
2. **Trie index crash** when a single GroundingSpace accumulates atoms from 16+ modules (see `docs/TRIE_CRASH_FINDINGS.md`)

The canonical MeTTa files are unpatched — the bugs are in the runtime, not the architecture.

### Tier 2: Executable Python Specifications

1:1 logic translations encoding identical formulas, thresholds, and metabolic dynamics. All subsystems validated: Peircean triad (7/7), fractal planning (8/8), deduction (6/6), abduction (6/6), EFE action selection, viability, grounding, scaling to 50 actions and 1,000 observables. Zero external dependencies.

### Tier 3: Python MeTTa Evaluator + Orchestrator

`dagaz_runtime.py` (~400 lines) loads all 22 MeTTa modules and executes them through generic pattern matching. The evaluator implements only universal MeTTa primitives — no domain-specific logic. Validated capabilities on the canonical source:

- **Module loading**: 1,664 code atoms, 107 state atoms, 143 ontology atoms across three spaces
- **EFE computation**: Produces correct numerical results; `select-action-myopic` returns the right action under uncertainty
- **Affect**: Valence, arousal, dominance derived from prediction errors
- **Cognitive cycle**: Full `cognitive-cycle!` executes and returns structured results (total error, affect state, action, structure learning state)
- **Verbalization intent**: `package-verbalization-intent` produces structured tuples for the orchestrator
- **Conversation state**: `conversation-state-description` and `check_knowledge` queries functional

`orchestrator.py` (~450 lines) connects the evaluator to a local LLM (Llama 3.2 3B via Ollama):

- **Perception**: LLM parses user text into structured observations (open-vocabulary utterance types, not enums)
- **Epistemic injection**: Observations routed into the MeTTa state space via `perceive-utterance!` and `inject-observation!`
- **Cognition**: MeTTa cognitive cycle runs; structured intent extracted
- **Epistemic firewall**: When the core selects `wait`, no LLM generation is triggered
- **Knowledge grounding**: Metagraph queried before verbalization; unknown topics explicitly labeled
- **Verbalization**: LLM translates the core's structured intent into natural language, constrained to claim only what the metagraph can ground
- **Hardware sensors**: Direct injection pathway bypasses LLM parsing for high-precision sensor data

### Known Tier 3 Limitations

- `match` returns only the first result (MeTTa `match` should be nondeterministic). `collapse(match ...)` iterates correctly.
- Multi-cycle operation requires the watchdog constitutional count to match the post-initialization atom state. Boot-sequence ordering is sensitive to duplicate viability-bounds from multiple modules.
- Policy-level planning (`core/policy_efe.metta`) is bypassed in the orchestrator; action selection uses single-step EFE.
