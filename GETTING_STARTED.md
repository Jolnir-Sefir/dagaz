# Getting Started

> **Coming from LangChain, AutoGPT, CrewAI, or similar frameworks?** MeTTa and Active Inference will look unfamiliar. Start with [`ONBOARDING_FOR_BUILDERS.md`](docs/ONBOARDING_FOR_BUILDERS.md) — it maps every concept to Python equivalents and gives you a guided path into the codebase. Then come back here to run things.

## Quick Start — Run the Agent (5 minutes)

The fastest way to see Dagaz in action is to run the full cognitive pipeline: a local LLM handles perception and verbalization, while the symbolic MeTTa core — loaded and executed by a pure Python evaluator — handles all reasoning, planning, and action selection.

```bash
git clone <repository-url>
cd dagaz

# Install and start Ollama (https://ollama.ai)
ollama pull llama3.2:3b

# Run the agent
python orchestrator.py --trace-pipeline
```

Type a message. The trace shows the full pipeline: LLM parse → MeTTa cognitive cycle (action, affect, timing) → LLM verbalization. Type `state` to inspect beliefs. Type `sensor threat-level 0.9` to inject a hardware threat and watch the agent's behavior shift.

No external Python dependencies required.

## Quick Start — Run the Specifications (2 minutes)

The executable specifications validate the architecture's core claims independently of the runtime. Each encodes the same logic as its corresponding MeTTa module — same formulas, same thresholds, same metabolic dynamics. All are self-contained Python with zero dependencies.

```bash
# Main benchmarks
python test_unified_reasoning.py      # Peircean triad (7 scenarios)
python test_fractal_planning.py       # Adaptive beam search (8 scenarios)
```

If both print green results, everything is working.

## What to Run

### Tier 3: End-to-End System

| Command | What it does |
|---------|-------------|
| `python orchestrator.py --trace-pipeline` | Full agent: perception → cognition → verbalization via local LLM |
| `python orchestrator.py` | Same, without pipeline trace |

### Tier 2: Executable Specifications

| Specification | What it validates | Time |
|-----------|--------------|------|
| `test_unified_reasoning.py` | Induction, deduction, abduction, Sherlock Holmes effect, metabolic death | ~2s |
| `test_fractal_planning.py` | Adaptive beam search, depth scaling, confidence degradation | ~1s |
| `test_deductive_reasoning.py` | Transitive closure, falsification of wrong deductions | ~1s |
| `test_abduction.py` | Hidden cause inference, metabolic selection of hypotheses | ~1s |
| `test-efe.py` | EFE-driven action selection (explore, exploit, retreat) | ~1s |
| `test_reef_v6.py` | 14-observable reef scenario, 70 cycles, 5 ecological phases | ~3s |
| `test_lsh_hebbian.py` | LSH scaling from 14 to 1,000 observables | ~5s |
| `metabolic_sensitivity.py` | 110-pair parameter sweep of metabolic economy | ~10s |

Run everything:

```bash
python test_unified_reasoning.py
python test_fractal_planning.py
python test_deductive_reasoning.py
python test_abduction.py
python test-efe.py
python test_reef_v6.py
python test_lsh_hebbian.py
python metabolic_sensitivity.py
```

### Tier 3: Runtime Evaluator Tests

```bash
python test_dagaz_runtime.py          # Parser, unification, module loading, EFE computation
```

This validates the Python MeTTa evaluator itself — that it correctly parses MeTTa syntax, resolves function definitions, routes atoms to the right spaces, and computes EFE from the canonical source.

## What to Read

If you want to understand the architecture, start here:

1. **`ARCHITECTURE.md`** — System overview. Start here.
2. **`dagaz_paper_v6.tex`** — The whitepaper. Full technical presentation.
3. **`ETHICS.md`** — Dual-use analysis and safety considerations.

If you want to **configure a new scenario**, start with **`core/domain.metta`** — it's a self-documenting template with six sections (actions, observables, preferences, viability bounds, action models, costs). Edit the declarations, and the cognitive core adapts automatically.

If you want to go deeper into a specific subsystem, each has a design document and a corresponding MeTTa implementation. The mapping is in the README.

## Three-Tier Validation

The system is validated at three levels:

**Tier 1 — MeTTa source (canonical).** The 23 `.metta` files are the architecture. They define types, functions, spaces, and dynamics in MeTTa, a symbolic meta-language for the OpenCog Hyperon framework. Native Hyperon execution is currently blocked by two runtime bugs (cons-cell pattern matching, trie index crash). The canonical files are unpatched — the bugs are in the runtime, not the architecture. See `TRIE_CRASH_FINDINGS.md` for the investigation.

**Tier 2 — Executable Python specifications.** 1:1 logic translations of the MeTTa modules into self-contained Python. Same formulas, same thresholds, same metabolic dynamics, different host language. These prove the dynamics work: every emergent behavior, every phase transition, every scaling curve. The relationship is analogous to a physicist's equations validated by simulation.

**Tier 3 — Python MeTTa evaluator + orchestrator.** `dagaz_runtime.py` is a ~750-line pure Python MeTTa interpreter that loads and executes the canonical MeTTa source through generic tokenization, parsing, unification, and function dispatch. It implements no domain-specific logic — only universal MeTTa primitives (`let`, `let*`, `if`, `case`, `match`, `collapse`, arithmetic, logic, `add-atom`/`remove-atom`, cons-cell construction). Evaluation is nondeterministic — `match` iterates all bindings, `collapse` gathers results, and function dispatch explores all matching definitions. The cognitive behavior emerges from evaluating the MeTTa function definitions themselves. `orchestrator.py` connects this evaluator to a local language model, running the full perception → cognition → verbalization pipeline.

Tier 3 is qualitatively distinct from Tier 2: rather than re-implementing the logic in Python, it executes the *same MeTTa source code* that Tier 1 defines. This partially closes the gap between "the dynamics work" (Tier 2) and "the MeTTa source is correct" (Tier 1).

## About the MeTTa Source

The `.metta` files are the canonical architecture — 23 modules, ~12,600 lines. They are written against correct MeTTa syntax per the language specification.

If you want to explore the MeTTa source, `loader.metta` shows the module dependency order, and `cycle.metta` is the main cognitive loop that ties everything together.

## Project Structure

```
dagaz/
├── core/
│   ├── *.metta           # 22 cognitive modules (the architecture)
│   └── domain.metta      # Domain configuration — edit this for your scenario
├── benchmarks/
│   └── test_*.py          # Executable specifications (Tier 2)
├── dagaz_runtime.py       # Pure Python MeTTa evaluator (Tier 3)
├── orchestrator.py        # Stateless LLM orchestrator (Tier 3)
├── docs/
│   └── *_DESIGN.md        # Design documents for each subsystem
├── ARCHITECTURE.md        # System overview
├── dagaz_paper.tex        # Whitepaper
├── reef_environment.py    # Reef scenario environment
├── reef_dagaz.html        # Interactive reef visualization (open in browser)
└── LICENSE*               # AGPL-3.0
```

## Questions and Contributions

This is an early-stage research project. If you find something interesting, confusing, or wrong, open an issue. Contributions that help resolve the Hyperon runtime blockers are especially welcome — as are new domain scenarios that test the architecture's generality claims.
