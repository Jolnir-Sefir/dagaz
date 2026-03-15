# Dagaz for Agentic AI Builders

### A guide for Python/JS developers coming from LangChain, AutoGPT, CrewAI, or similar frameworks.

---

## What You Already Know (and What's Different Here)

If you build with LangChain, AutoGPT, or Devin, your mental model of an "agent" probably looks something like this:

```python
# The agentic AI pattern you know
while not done:
    context = memory.retrieve(query)
    plan = llm.generate(system_prompt + context + user_input)
    action = tool_router.pick(plan)
    result = tool.execute(action)
    memory.store(result)
```

The LLM is the brain. Tools are the hands. Memory is the filing cabinet. Prompts are the personality.

**Dagaz inverts this.** The LLM is *not* the brain. The LLM is the *sensory nerve* — it translates raw text into structured observations on the way in, and translates structured intentions into natural language on the way out. All reasoning, planning, and decision-making happens in a symbolic core that has never seen a token in its life.

```
               YOUR FRAMEWORK                          DAGAZ
          ┌─────────────────────┐            ┌─────────────────────────┐
          │                     │            │                         │
 input ──→│   LLM (reasoning)   │──→ output  │  LLM ──→ Symbols ──→ LLM│
          │                     │            │  (ear)  (brain)  (mouth)│
          └─────────────────────┘            └─────────────────────────┘
```

Why do this? Because LLMs hallucinate, can't show their work, and have no real model of what they know vs. don't know. Dagaz can trace every inference, query any belief, and its "I don't know" comes from an actual failed lookup — not a prompt that says "be honest."

---

## The 60-Second Version

Dagaz is a cognitive architecture built on one equation:

```
Pick the action that minimizes Expected Free Energy (EFE).

EFE(action) = How_Wrong_I'll_Be + Cost - What_I'll_Learn + Am_I_About_To_Die
```

That's it. Exploration, retreat, investigation, curiosity, self-preservation — they all fall out of this one equation with different inputs. No if/else chains. No mode switches. No "if confused, ask for clarification" rules.

The entire system is 22 modules written in **MeTTa** (a symbolic language you've never seen), running on a **pure Python evaluator** you can actually read. The LLM touches exactly two points: parsing text into observations, and converting structured intents back into text.

---

## MeTTa in 10 Minutes (for Python Developers)

MeTTa will look alien at first. It's an S-expression language (like Lisp) designed for symbolic AI. Here's the Rosetta Stone.

### Data: It's Just Tuples

```python
# Python — you'd use a dict or dataclass
belief = {"observable": "temperature", "value": 0.7, "precision": 0.5}
```

```metta
; MeTTa — it's a tuple (called an "atom") in a space (called an "atomspace")
(belief temperature 0.7 0.5)
```

No keys. Position matters. `(belief temperature 0.7 0.5)` means: the agent believes temperature is 0.7 with precision (confidence) 0.5.

These atoms live in **spaces** — think of a space as an in-memory database table:

```metta
; Add an atom to the state space (like INSERT)
!(add-atom &state (belief temperature 0.7 0.5))

; Query the state space (like SELECT ... WHERE)
(match &state (belief temperature $val $prec) $val)
; → Returns 0.7. $val and $prec are pattern variables (like SQL wildcards).

; Remove an atom (like DELETE)
!(remove-atom &state (belief temperature 0.7 0.5))
```

`&state` is a space name. `&self` is where code lives. `&ontology` holds static knowledge. Three spaces, that's the whole database.

### Functions: Pattern-Matched Definitions

```python
# Python
def get_belief_value(obs):
    return state.query(f"belief {obs}").value
```

```metta
; MeTTa — "=" means "this expression reduces to that expression"
(= (get-belief-value $obs)
   (match &state (belief $obs $val $prec) $val))
```

When the evaluator sees `(get-belief-value temperature)`, it:
1. Matches `$obs` → `temperature`
2. Searches `&state` for `(belief temperature $val $prec)`
3. Returns whatever `$val` is bound to

### Control Flow: let, let*, if, sequential

```python
# Python
error = abs(observation - belief)
weighted = error * precision * importance
```

```metta
; MeTTa — let* chains bindings (like a pipeline)
(let* (
  ($error (abs (- $obs-val $belief-val)))
  ($weighted (* (* $error $precision) $importance))
)
$weighted)
```

`let*` is sequential binding — each line can use variables from the lines above. `let` (without the star) binds one variable.

Side effects use `sequential`:

```metta
(sequential
  (remove-atom &state (belief temperature 0.7 0.5))   ; delete old
  (add-atom &state (belief temperature 0.8 0.6))       ; insert new
  (belief-updated temperature))                         ; return value
```

### Types: Declarations, Not Enforcement

```metta
(: Observable Type)           ; "Observable" is a type
(: Precision Type)            ; "Precision" is a type
(: Belief Type)               ; "Belief" is a type
(: Action Type)               ; "Action" is a type
(: wait Action)               ; "wait" is an Action
(: observe Action)            ; "observe" is an Action
(: retreat Action)            ; "retreat" is an Action
(: investigate (-> Observable Action))  ; "investigate" takes an Observable, returns an Action
```

These are declarations — they tell you what things are. MeTTa doesn't enforce them like TypeScript would. Think of them as documentation that lives in the same space as the data.

### The `!` Operator: "Do This Now"

```metta
; Without ! — this is a definition (stored, not executed)
(= (my-function $x) (* $x 2))

; With ! — this is an imperative command (executed immediately)
!(add-atom &state (belief temperature 0.5 0.3))
```

`!` means "evaluate this right now." Definitions without `!` are stored as reduction rules.

### The `collapse` Trick: Collecting Multiple Results

MeTTa's `match` can return multiple results (it's nondeterministic). `collapse` gathers them into a list:

```metta
; Get ALL beliefs (not just the first one)
(= (all-beliefs)
   (collapse (match &state (belief $o $v $p) (belief $o $v $p))))
```

This is like Python's `list(filter(...))` — it collects all matches into a single list you can iterate over.

---

## How the Cognitive Loop Maps to What You Know

Here's the Dagaz cognitive cycle, mapped to concepts you already understand:

| Dagaz Step | Your Framework Equivalent | What Actually Happens |
|------------|--------------------------|----------------------|
| **Perception** | `llm.parse(user_input)` | LLM converts text → structured observations (the ONLY place an LLM touches) |
| **Belief Update** | `memory.store(result)` | Bayesian update: new observations shift beliefs proportional to precision |
| **Prediction Error** | *(no equivalent)* | Compare every belief to every observation. The gap is "surprise." |
| **Affect** | *(no equivalent)* | Valence/arousal/dominance COMPUTED from prediction errors. Not a prompt. Not stored. |
| **Action Selection** | `tool_router.pick(plan)` | argmin EFE over all available actions. No LLM involved. |
| **Structure Learning** | *(no equivalent)* | Correlated surprises → causal link hypotheses → metabolic selection |
| **Abduction** | *(no equivalent)* | Observed effects + known structure → hidden cause hypotheses |
| **Grounding** | *(no equivalent)* | Link learned concepts to semantic primitives so they MEAN something |
| **Verbalization** | `llm.generate(response)` | Structured intent → natural language (the other place an LLM touches) |

The steps marked "no equivalent" are the ones that make Dagaz different from a prompt-engineered agent. Structure learning discovers causal relationships the system was never told about. Abduction hypothesizes hidden causes. Grounding ensures the system can explain what its concepts mean.

---

## The EFE Equation (the Only Equation You Need)

Every action the agent considers gets scored by this:

```
EFE(action) = Expected_Error + Cost - Info_Gain + Viability_Penalty
```

**Lower is better.** The agent picks the action with the lowest EFE. Here's what each term does:

```python
# Pseudocode — this is what the MeTTa computes
def compute_efe(action):
    expected_error = 0
    info_gain = 0
    
    for obs in all_observables:
        # What will my belief look like after this action?
        predicted_value = belief[obs].value + action_model[action][obs].delta_value
        predicted_precision = belief[obs].precision + action_model[action][obs].delta_precision
        
        # How far is that from what I want?
        deviation = abs(predicted_value - preference[obs].target)
        expected_error += deviation * predicted_precision * preference[obs].weight
        
        # How much will I learn? (precision gain = info gain)
        info_gain += action_model[action][obs].delta_precision * 0.5
    
    cost = action_costs[action]
    
    # Am I near a survival boundary?
    viability_penalty = compute_viability_effect(action)
    
    return expected_error + cost - info_gain + viability_penalty
```

**Why this works without if/else:**

| Situation | What dominates EFE | Agent does |
|-----------|-------------------|------------|
| High uncertainty, no threat | Info_gain term is large → observe wins | Explores |
| Predictions are accurate | Expected_error is low → wait wins | Waits |
| Near viability boundary | Viability_penalty dominates → retreat wins | Retreats |
| Surprise + known structure | Abduction creates low-precision belief → investigate wins | Investigates |

No one programmed "explore when uncertain." It falls out of the math.

---

## Where to Start Hacking

### Tier 1: Poke the Running System (5 min)

```bash
# Start the agent
python orchestrator.py --trace-pipeline

# Talk to it — watch the pipeline trace
> Hello, I'm exploring a coral reef.

# Inject a threat — watch behavior shift
> sensor threat-level 0.85

# Inspect beliefs
> state
```

The trace shows you: LLM parse result → MeTTa cycle output (selected action, affect state, timing) → LLM verbalization. This is the fastest way to see the architecture in action.

### Tier 2: Run the Python Specifications (2 min)

These are standalone Python files that encode the same logic as the MeTTa modules. They're the best way to understand what the math actually does, in a language you can read:

```bash
python benchmarks_test-efe.py               # EFE action selection — the core loop
python benchmarks_test_unified_reasoning.py  # Full Peircean triad — induction, deduction, abduction
python benchmarks_test_fractal_planning.py   # Adaptive beam search
```

Each test prints what's happening. Read the code alongside — it's commented for understanding, not just testing.

### Tier 3: Read the MeTTa (with training wheels)

Start with these files in this order:

1. **`core_beliefs.metta`** (~240 lines) — The simplest module. Belief accessors, mutation, prediction error. Every pattern you need to read MeTTa is here.

2. **`core_actions.metta`** (~580 lines) — The EFE computation. This is the heart of the system. The Python pseudocode above maps directly to the `compute-efe` function here.

3. **`core_cycle.metta`** (~580 lines) — The main loop. Once you can read beliefs and actions, the cycle is just sequencing them.

4. **`core_structure_learning.metta`** (~1270 lines) — The most complex module. Read this when you want to understand how causal discovery works.

### Tier 4: Modify Something

Here are starter projects ordered by difficulty:

**Easy — Add a new observable:**
Add a new belief, observation, action model, and preference to the reef scenario. Watch how the agent's behavior changes. No MeTTa knowledge needed — just add atoms to the state space via the orchestrator's sensor injection.

**Medium — Improve the LLM parse boundary:**
`orchestrator.py` lines ~130–200 contain the LLM perception prompt. The parse extracts utterance type, topics, and six numerical estimates. Currently uses Llama 3.2 3B. Swap in a better model, improve the prompt, add structured output parsing. Pure Python — no MeTTa involved.

**Medium — Add a new action type:**
Define a new action with its action-model (predicted effects on observables), cost, and optionally a grounding chain. The EFE machinery will automatically evaluate it alongside existing actions. Look at how `retreat` is defined in `core_actions.metta` and follow the pattern.

**Hard — Connect a real environment:**
`orchestrator.py` has a `sensor` command for injecting hardware observations. Build a bridge to a real sensor (weather API, system metrics, IoT device). The observations flow through the same pathway as LLM-parsed text.

**Hard — Improve the evaluator:**
`dagaz_runtime.py` is a 400-line MeTTa interpreter. The biggest known limitation: `match` returns only the first result instead of being nondeterministic. Making `collapse(match ...)` return all results correctly would unlock more of the architecture's potential. This is a pure computer science problem — pattern matching and unification.

---

## Key Concepts Translated

| Active Inference Term | Your Framework Equivalent | What It Actually Means |
|----------------------|--------------------------|----------------------|
| **Belief** | Memory/state entry | A value + confidence estimate for something observable |
| **Precision** | *(no equivalent)* | How confident the agent is in a belief. High precision = hard to change. Inverse variance. |
| **Prediction Error** | *(no equivalent)* | The gap between what the agent expected and what it observed. Drives all learning. |
| **Expected Free Energy** | Reward function / utility | Score for each action. But it includes curiosity (info gain) natively — no exploration hack needed. |
| **Action Model** | Tool description | "If I do X, observable Y will change by δ with precision change ε" |
| **Viability Bound** | *(no equivalent)* | Hard survival boundary. Not a preference — a structural limit. Cross it and cognition breaks down. |
| **Metabolic Energy** | *(no equivalent)* | Knowledge atoms must "pay rent" by making correct predictions or they die. Prevents knowledge bloat. |
| **Passive Model** | *(no equivalent)* | "When X happens, Y follows" — world dynamics, discovered by the agent, used for prediction only. |
| **Grounding Chain** | *(no equivalent)* | A concept traces through semantic primitives to something meaningful. "power-level" → "a quantity the agent has that enables doing." |
| **Stratum** | Permission level | Constitutional (immutable physics) → Goal (modifiable preferences) → Learned (free to change). The safety architecture. |

---

## The Three Spaces

MeTTa uses three separate storage areas. Think of them as three databases:

| Space | Analogy | Contents | Mutability |
|-------|---------|----------|------------|
| `&self` | Code repository | Function definitions, type declarations, config. ~3,000 atoms. | Static after boot. Never modified at runtime. |
| `&state` | Working memory / Redis | Beliefs, observations, action models, suspicion links, metabolic energy. ~50-200 atoms. | Fully dynamic. add-atom/remove-atom every cycle. |
| `&ontology` | Reference database | Semantic primitives, scales, grounding chains. ~328 atoms. | Static. Reference data. |

When you see `(match &state ...)` — it's querying working memory.
When you see `(match &self ...)` — it's looking up a function definition.
When you see `(match &ontology ...)` — it's looking up semantic knowledge.

---

## Architecture at a Glance

```
orchestrator.py                    dagaz_runtime.py
┌─────────────────┐               ┌──────────────────────────────────┐
│                  │               │  Pure Python MeTTa Evaluator     │
│  LLM (Ollama)   │               │                                  │
│  ┌───────────┐  │  observations │  ┌────────────────────────────┐  │
│  │ Parse     │──┼──────────────→│  │ 22 MeTTa Modules           │  │
│  │ (NL→struct)│ │               │  │                            │  │
│  └───────────┘  │               │  │  beliefs → errors → affect │  │
│                  │               │  │       ↓                    │  │
│  ┌───────────┐  │  intent       │  │  EFE → action selection    │  │
│  │ Verbalize │←─┼──────────────←│  │       ↓                    │  │
│  │ (struct→NL)│ │               │  │  structure learning        │  │
│  └───────────┘  │               │  │  abduction                 │  │
│                  │               │  │  grounding                 │  │
│  Epistemic      │               │  └────────────────────────────┘  │
│  Firewall:      │               │                                  │
│  If action=wait │               │  &self (code) ~3000 atoms        │
│  → no LLM call  │               │  &state (dynamic) ~50-200 atoms  │
│                  │               │  &ontology (static) ~328 atoms   │
└─────────────────┘               └──────────────────────────────────┘
```

The **epistemic firewall** is worth noting: when the MeTTa core selects `wait` (meaning "I have nothing to say"), the orchestrator does NOT call the LLM to generate a response. Silence is an action, not a failure.

---

## Why Not Just Use the LLM for Everything?

You might be thinking: "I can do all of this with a good system prompt and some tool calls." Here's the honest comparison:

| Property | LLM-Native Agent | Dagaz |
|----------|------------------|-------|
| **Can it explain why it chose an action?** | "Based on my analysis..." (generated post-hoc) | Yes — full EFE trace, each term decomposed |
| **Does it know what it doesn't know?** | Only if prompted to say "I don't know" | Yes — low precision beliefs are queryable |
| **Can it discover causal structure?** | Only if it was in the training data | Yes — from correlated prediction errors |
| **Can it learn from a single observation?** | No (needs fine-tuning or RAG) | Yes — Bayesian update adjusts beliefs immediately |
| **Can external code verify its beliefs?** | No (beliefs are implicit in weights) | Yes — every belief is an atom in a queryable space |
| **Token cost per decision** | Thousands–millions of tokens | Zero tokens (symbolic computation) |
| **Failure mode** | Hallucination (confident nonsense) | Thrashing near viability bounds (detectable, haltable) |

The trade-off: Dagaz can't handle open-ended natural language reasoning the way an LLM can. It handles structured domains where you want traceable, auditable decisions. The LLM handles the messy boundary between human language and structured observations.

---

## Common Gotchas

**"Why does `match` only return one result?"**
Known limitation of the Python evaluator. MeTTa's `match` is supposed to be nondeterministic (return ALL matches). The evaluator returns the first. Use `collapse(match ...)` to get all results as a list. Fixing this properly is an open contribution opportunity.

**"Where is the training data?"**
There is none. Dagaz doesn't learn from a dataset. It learns from live observations during operation. Beliefs start at low precision and update from prediction errors. Structure learning discovers causal links from correlated surprises. This is online learning, not batch training.

**"How do I add domain knowledge?"**
Add atoms to the state space. Beliefs, preferences, action models, viability bounds — these are all atoms you can add at initialization or inject at runtime. Look at the reef scenario (`environment_reef_scenario_metta.txt`) for a worked example.

**"Can I use GPT-4 / Claude / Gemini instead of Llama?"**
Yes — the orchestrator's LLM calls go through a simple HTTP interface. Swap `llm_url` and `llm_model` in the Config dataclass. The MeTTa core doesn't know or care which LLM is doing the perception/verbalization.

**"What if I just want the EFE action selection without all the structure learning?"**
You can. Run with `(config action-selection-mode single)` in `cycle.metta` to use myopic (single-step) EFE. This skips policy planning and structure learning reflexes. It's the simplest mode — good for understanding the core loop before adding complexity.

**"The Hyperon runtime doesn't work?"**
Correct — native MeTTa execution on Hyperon 0.2.10 is blocked by two bugs (cons-cell pattern matching, trie index crash). The Python evaluator is the workaround. The MeTTa source files are the canonical architecture; the evaluator executes them. If Hyperon fixes these bugs, the same `.metta` files run natively.

---

## File Map (What to Read When)

| I want to... | Read this |
|--------------|-----------|
| See the agent run | `python orchestrator.py --trace-pipeline` |
| Understand the math in Python | `benchmarks_test-efe.py` |
| Read the simplest MeTTa module | `core_beliefs.metta` |
| Understand the main loop | `core_cycle.metta` |
| Understand action selection | `core_actions.metta` |
| Understand how the LLM connects | `orchestrator.py` |
| Understand the MeTTa evaluator | `dagaz_runtime.py` |
| Understand the full architecture | `docs_ARCHITECTURE.md` |
| Understand causal discovery | `core_structure_learning.metta` + `docs_LEARNING_DESIGN.md` |
| Understand the safety system | `core_safety.metta` + `docs_ETHICS.md` |
| See a complete scenario | `environment_reef_scenario_metta.txt` + `environment_test_reef_v6.py` |
| Understand why certain design choices were made | `docs_PLANNING_STRATEGY.md` (honest about analogy limits) |

---

## Contributing

The biggest leverage points for contributors right now:

1. **Evaluator improvement** — Make `dagaz_runtime.py` handle nondeterministic match correctly. This is a well-defined CS problem: pattern matching with unification over a set of atoms, returning all solutions.

2. **LLM parse quality** — The perception boundary (`orchestrator.py`) is the weakest link. Better prompts, better models, structured output schemas, multi-shot parsing — all would improve the entire system.

3. **New environments** — The reef scenario is the main worked example. Building new scenarios (robot navigation, dialogue systems, game NPCs) would test the architecture's domain-generality claims.

4. **Hyperon bug fixes** — If you're familiar with the Hyperon/OpenCog codebase, the two blocking bugs (cons-cell pattern matching, trie crash at ~1,664 atoms) are documented in the repo. Fixing either would unlock native MeTTa execution.

5. **Visualization** — `environment_reef_dagaz.html` is a start, but real-time visualization of the EFE landscape, belief states, and causal graph during operation would make the system dramatically more understandable.

---

*This document assumes familiarity with Python and agentic AI frameworks. For the full technical specification, start with `docs_ARCHITECTURE.md`. For the mathematical foundations, see the paper.*
