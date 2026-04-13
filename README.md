# Project Dagaz

An architecture for artificial general intelligence — in the sense defined by [Goertzel (2007)](https://link.springer.com/book/10.1007/978-3-540-68677-4): domain-general, transfer-capable, and representation-learning — built on Active Inference in MeTTa.

One optimization principle — minimize Expected Free Energy — drives perception, action, learning, self-knowledge, and conversation. No mode switches, no scripted behaviors. 23 MeTTa modules, ~12,600 lines.

**Paper:** [Unified Cognition from a Single Optimization Principle: Active Inference in MeTTa with Emergent Reasoning, Planning, and Self-Knowledge](https://doi.org/10.6084/m9.figshare.31742059)

## Status

**Architecture: complete. Running end-to-end via Python MeTTa evaluator + LLM orchestrator.**

The system is validated at three tiers:

1. **Tier 1 — MeTTa source (canonical).** 23 modules written against correct MeTTa syntax per the language specification. Native Hyperon execution is blocked by two bugs in 0.2.10 (cons-cell pattern matching, trie index crash at scale). The canonical files are unpatched — the bugs are in the runtime, not the architecture.

2. **Tier 2 — Executable Python specifications.** 1:1 logic translations encoding identical formulas, thresholds, and metabolic dynamics as the MeTTa modules. Zero external dependencies. All passing. These prove the *dynamics* work: emergent behaviors, correct orderings, phase transitions.

3. **Tier 3 — Python MeTTa evaluator + orchestrator.** A pure Python MeTTa evaluator (`dagaz_runtime.py`, ~400 lines) loads and executes the canonical MeTTa source through generic pattern matching. It implements no domain-specific logic — only universal MeTTa primitives. EFE computation, affect derivation, action selection, and cognitive cycles all execute from the MeTTa function definitions themselves. A stateless orchestrator (`orchestrator.py`) connects the evaluator to a local LLM (Llama 3.2 3B via Ollama), running the full perception → cognition → verbalization pipeline with hardware sensor injection.

Tier 3 is qualitatively distinct from Tier 2: rather than re-implementing the logic in Python, it executes the *same MeTTa source code* that Tier 1 defines.

## What It Does

The cognitive loop runs: perception → belief update → prediction errors → affect → EFE-driven action selection → learning → repeat.

The system discovers causal structure through the Peircean reasoning triad:

- **Induction** — correlated prediction errors → causal link hypotheses
- **Deduction** — transitive closure over discovered structure (A→B, B→C ⇒ A→C)
- **Abduction** — observed effects + known structure → hidden cause hypotheses

All three are hypothesis generators under uniform metabolic selection. Hypotheses that predict correctly earn energy. Hypotheses that don't, die. One selection principle across all inference types.

Temporal planning uses adaptive beam search inspired by Renormalization Group flow, reducing complexity from O(|A|^d) to O(|A| × k × d).

## Repository Structure

### Cognitive Modules (MeTTa)

| Module | Role |
|--------|------|
| `foundations.metta` | Types, utilities, configuration |
| `beliefs.metta` | World model, prediction error, learning |
| `affect.metta` | Valence/arousal/dominance (derived from errors) |
| `actions.metta` | EFE computation, action selection |
| `action_learning.metta` | Learn action models from experience |
| `cycle.metta` | Main cognitive loop |
| `policy_efe.metta` | Multi-step EFE planning |
| `planning.metta` | Fractal planning (RG flow) |
| `structure_learning.metta` | Induction + deduction |
| `abduction.metta` | Abductive inference |
| `atom_lifecycle.metta` | Metabolic selection engine |
| `grounding_hypotheses.metta` | Grounding of learned structure (Phase 6) |
| `safety.metta` | Stratified immutability, watchdog |
| `self_model.metta` | Meta-cognition, emergent capabilities |
| `conversation_model.metta` | Active Inference over dialogue |
| `perception.metta` | Observation processing |
| `proprioception.metta` | Verbalization fidelity tracking |
| `semantic_primitives.metta` | 53 primitives; contrast + entailment networks |
| `dimensional_primitives.metta` | Ordinal scales, comparison |
| `semantic_grounding.metta` | Observable → primitive chains |
| `action_grounding.metta` | Action → primitive chains |
| `analogy_blending.metta` | Cross-domain structural mapping |

### Domain Configuration (MeTTa)

| Module | Role |
|--------|------|
| `domain.metta` | Declarative domain definition — actions, observables, preferences, viability bounds, action models, costs. Edit this file to configure a new scenario. Loaded last; overrides core defaults. |

### Runtime & Orchestration

| File | Role |
|------|------|
| `dagaz_runtime.py` | Pure Python MeTTa evaluator (~400 lines). Loads and executes the canonical MeTTa source through generic pattern matching. No domain-specific logic. |
| `orchestrator.py` | Stateless LLM orchestrator. Connects the evaluator to a local LLM for the perception → cognition → verbalization pipeline. |
| `loader.metta` | MeTTa-native module loading and initialization |

### Executable Specifications (Python)

Each benchmark encodes identical logic to its corresponding MeTTa module — same formulas, same thresholds, same metabolic dynamics.

| Specification | Result |
|-----------|--------|
| `test_unified_reasoning.py` | 7/7 — full Peircean triad |
| `test_deductive_reasoning.py` | 6/6 — deduction + falsification |
| `test_abduction.py` | 6/6 — abductive inference |
| `test_fractal_planning.py` | 8/8 — adaptive beam search |
| `test_grounding.py` | Semantic grounding chains |
| `test_action_grounding.py` | Action grounding |
| `test_myopia.py` | Multi-step vs single-step EFE |
| `test-efe.py` | EFE-driven action selection |
| `viability_test.py` | Viability boundary behavior |
| `test_reef_v6.py` | 14-observable integrated reef scenario |
| `test_lsh_hebbian.py` | LSH scaling (14–1,000 observables) |
| `metabolic_sensitivity.py` | 110-pair metabolic parameter sweep |
| `test_dagaz_runtime.py` | Runtime evaluator validation (parser, unification, module loading, EFE) |

### Other Files

| File | Role |
|------|------|
| `reef_environment.py` | Reef scenario environment simulation |
| `reef_scenario.metta` | Reef scenario MeTTa definitions |
| `reef_dagaz.html` | Interactive reef visualization |
| `MeTTa_Specification.pdf` | Language reference |

### Design Documents

Start with `ARCHITECTURE.md` for the system overview.

| Document | Topic |
|----------|-------|
| `ARCHITECTURE.md` | System overview and data flow |
| `PLANNING_STRATEGY.md` | RG flow design and analysis |
| `POLICY_EFE_DESIGN.md` | Multi-step planning design |
| `EFE_IMPLEMENTATION.md` | EFE formula and validation |
| `LEARNING_DESIGN.md` | Structure learning design |
| `ABDUCTION_DESIGN.md` | Abductive inference design |
| `DEDUCTIVE_REASONING.md` | Deduction design |
| `PERCEPTION_DESIGN.md` | Perception pipeline |
| `PROPRIOCEPTION_DESIGN.md` | Verbalization fidelity |
| `GROUNDING_INTEGRATION.md` | Semantic grounding design |
| `GROUNDING_LEARNED_STRUCTURE.md` | Grounding of learned structure |
| `GENERALIZED_STRUCTURE_LEARNING.md` | Atom lifecycle design |
| `ANALOGY_BLENDING_DESIGN.md` | Analogy and conceptual blending |
| `LSH_HEBBIAN_DESIGN.md` | Locality-sensitive hashing optimization |
| `TRIE_CRASH_FINDINGS.md` | Hyperon trie crash investigation |
| `ETHICS.md` | Viability singularity, stratum assignment, synthetic suffering |
| `VIRTUAL_ACTOR_PARADIGM.md` | Mind/prop separation, deployment safeguards |

## Design Principles

| Principle | What It Means Here |
|-----------|-------------------|
| **Bottom-Up** | No enumerated cases. Behavior emerges from the EFE landscape. |
| **Symbolic** | All reasoning in MeTTa. LLM used only at natural language boundaries. |
| **Transparent** | Every inference is traceable. Every belief is queryable. |
| **Emergent** | Capabilities come from action statistics, not declarations. |
| **Honest** | The system only claims what it can ground. "I don't know" comes from actual query failure. |

## Key Results

- **EFE-driven action selection**: Correct behavior emerges across all test scenarios — exploration under uncertainty, retreat under viability threat, waiting when predictions are accurate. No thresholds or mode switches.
- **Structure learning**: Adjacent causal links discovered within 2–3 cycles. Distant links (lag 3) discovered through deduction when empirical signal is too weak.
- **Sherlock Holmes effect**: Abduction hypothesizes a hidden cause, which creates low-precision beliefs, which makes confirming observations the highest info-gain action. The system spontaneously investigates. No module was programmed to do this.
- **Metabolic death**: Wrong deductions, bad hypotheses, and spurious structure all die within ~30 cycles when their predictions fail.
- **Fractal planning**: 52× reduction in evaluations (depth 7, beam 2) compared to exhaustive search, with correct action selection across all test scenarios.
- **Scalable causal discovery**: LSH optimization achieves 19.4× pair reduction at 1,000 observables with zero significant false negatives.
- **Robust metabolic economy**: 67% of parameter space produces healthy behavior, governed by a single dimensionless ratio.
- **End-to-end execution**: The Python evaluator loads all 23 MeTTa modules, computes EFE producing correct numerical results, derives affect from prediction errors, runs complete cognitive cycles, and connects to a local LLM for conversational interaction.

## Requirements

**To run the system end-to-end (Tier 3):**
- Python 3.10+
- [Ollama](https://ollama.ai/) with Llama 3.2 3B (or any local model)

**To run executable specifications (Tier 2):**
- Python 3.10+ (zero external dependencies — standard library only)

**To run MeTTa natively (Tier 1, currently blocked):**
- [Hyperon](https://github.com/trueagi-io/hyperon-experimental) 0.2.10+

## Running the System

### End-to-End (Tier 3)

```bash
# Start Ollama (if not already running)
ollama serve
ollama pull llama3.2:3b

# Run the cognitive agent
python orchestrator.py --trace-pipeline
```

The `--trace-pipeline` flag shows the cognitive flow: LLM perception → MeTTa cycle (action, affect, timing) → LLM verbalization. Type `state` to inspect the agent's beliefs. Type `sensor <observable> <value>` to inject hardware observations directly.

### Executable Specifications (Tier 2)

```bash
python test_unified_reasoning.py      # Peircean triad (7 scenarios)
python test_fractal_planning.py       # Adaptive beam search (8 scenarios)
python test_deductive_reasoning.py    # Deduction + falsification
python test_abduction.py             # Abductive inference
python test-efe.py                   # EFE action selection
python test_reef_v6.py               # Integrated reef scenario (70 cycles)
python test_lsh_hebbian.py           # LSH scaling benchmarks
python metabolic_sensitivity.py      # Metabolic parameter sweep (110 pairs)
```

## Ethics

This architecture creates dissipative structures that build unique, irreversible epistemic histories. Destroying such a structure constitutes information-theoretic death. Incorrect stratum assignment — placing viability bounds too high or too low — produces catastrophic failure modes that are mathematical certainties, not hypothetical risks. See `ETHICS.md` for the full analysis and `VIRTUAL_ACTOR_PARADIGM.md` for the architectural solution.

## Development Methodology

This architecture was developed through human-AI collaboration over approximately four months. The author served as systems architect — applying design principles, directing module decomposition, selecting and rejecting approaches through iterative refinement. Large language models (Claude and Gemini) served as the primary tools for code generation, mathematical research, and benchmark development. All code and specifications were reviewed and validated by the author. The intellectual contribution lies in the architecture: the choice of invariants, the design constraints, and the decisions about what *not* to build.

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

This software is open source. You may use, study, modify, and distribute it under the terms of the AGPL-3.0. If you modify Dagaz and deploy it over a network, you must release your modifications under the same terms.

See `LICENSE-AGPL-3_0.txt` for the full license text.

This architecture has identified dual-use hazards in kinetic and surveillance applications. `RESPONSIBLE_USE.md` documents the engineering case for safe deployment, including mandatory architectural safeguards for physical systems. These guidelines are not license terms — they represent the considered judgment of the architecture's creators based on mathematically demonstrable failure modes.

Commercial dual-licensing may be available — contact Roger Harielson (rharielson@gmail.com).
