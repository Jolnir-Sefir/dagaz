# Perception Layer — Neural/Symbolic Boundary Design

## What This Module Does

`perception.metta` + `orchestrator.py` implement the boundary between raw
input (natural language, sensor readings) and the symbolic cognitive core.
The LLM is a **sensor**, not a reasoner. It produces observations with
precision values. The cognitive core doesn't know or care what produced them.

```
BEFORE:                                AFTER:

  "I don't get it"                       "I don't get it"
        │                                       │
        ▼                                       ▼
  ┌──────────────┐                      ┌───────────────┐
  │  ??? magic   │                      │  llama.cpp    │
  │  hand-waving │                      │  (3B model)   │
  └──────────────┘                      └───────┬───────┘
        │                                       │ structured parse
        ▼                                       ▼
  process-utterance!                    ┌───────────────┐
  (type, content, topics, time)         │  orchestrator  │
  manually constructed                  │  (Python glue) │
                                        └───────┬───────┘
                                                │ (obs, val, prec, time) × 5
                                                ▼
                                        ┌───────────────┐
                                        │ perception.metta│
                                        │ perceive-      │
                                        │ utterance!     │
                                        └───────┬───────┘
                                                │
                                                ▼
                                        inject-observation!
                                        (same as any sensor)
```

## The Core Insight

A 3B LLM estimating partner-comprehension from text is analogous to a cheap
IR sensor estimating distance. Both are noisy. Both produce observations.
The precision-weighted belief update handles both correctly. The cognitive
core self-corrects from bad parses the same way it self-corrects from noisy
sensor readings: through prediction error.

Misparses don't corrupt the system. They produce false beliefs. False beliefs
generate prediction errors when subsequent observations contradict them.
Prediction errors drive belief updates. The system self-corrects. This is
Active Inference doing what it does — no special error-recovery module needed.


## Architecture

### Components

| Component | Language | Role | State |
|-----------|----------|------|-------|
| `perception.metta` | MeTTa | Observation contracts, precision tables, topic mapping | Atomspace |
| `orchestrator.py` | Python | Prompt construction, LLM calls, response parsing | Stateless |
| llama.cpp | C++ | LLM inference server | Model weights |
| MeTTa runtime | MeTTa | Cognitive core (beliefs, EFE, actions) | Atomspace |

### Data Flow — One Turn

```
1. User types: "I don't really understand the power levels"

2. Orchestrator builds parse prompt:
   ┌─────────────────────────────────────────────────────┐
   │ FORMAT:                                              │
   │ TYPE: query|request|assertion|greeting|...           │
   │ TOPICS: comma,separated,keywords                    │
   │ COMPREHENSION: 0.0-1.0                              │
   │ COHERENCE: 0.0-1.0                                  │
   │ RAPPORT: 0.0-1.0                                    │
   │ PROGRESS: 0.0-1.0                                   │
   │ URGENCY: 0.0-1.0                                    │
   │ ...                                                  │
   │ User message: I don't really understand the power... │
   └─────────────────────────────────────────────────────┘

3. LLM returns:
   TYPE: query
   TOPICS: power,understanding
   COMPREHENSION: 0.3
   COHERENCE: 0.7
   RAPPORT: 0.5
   PROGRESS: 0.3
   URGENCY: 0.2
   SUMMARY: User is confused about power levels.

4. Orchestrator extracts fields via regex → ParseResult

5. MeTTa perceive-utterance! injects 6 observations:
   (observation partner-comprehension  0.3  0.30 42)  ← LLM precision
   (observation discourse-coherence    0.7  0.40 42)
   (observation rapport                0.5  0.25 42)
   (observation goal-progress          0.3  0.30 42)
   (observation partner-predictability 0.7  0.35 42)  ← computed
   (observation unresolved-salience    0.0  0.45 42)  ← 0: all topics resolved

6. Cognitive cycle runs:
   - Prediction errors computed (comprehension 0.3 vs belief 0.6 = big error)
   - EFE landscape shifts: clarify gets low EFE
   - Action selected: clarify

7. MeTTa packages verbalization intent:
   (verbalization-intent
     (action clarify)
     (affect (valence -0.1) (arousal 0.5))
     (drivers ((driver partner-comprehension 0.3 0.3)))
     ...)

8. Orchestrator builds verbalization prompt:
   ┌─────────────────────────────────────────────────┐
   │ ACTION: clarify                                  │
   │ TARGET: something the user seems confused about  │
   │         (regarding partner-comprehension)        │
   │ REASON: partner-comprehension is low             │
   │ TONE: concerned and attentive                    │
   └─────────────────────────────────────────────────┘

9. LLM returns: "Let me explain the power levels more clearly..."

10. Response displayed to user.
```


## Precision Assignment — The Key Design Decision

The entire integration rests on one principle: **precision encodes
trust**. Different sources get different precision values. The cognitive
core treats all observations identically — it just sees numbers.

### Precision by Source

| Source | Observable | Precision | Rationale |
|--------|-----------|-----------|-----------|
| LLM (3B) | partner-comprehension | 0.30 | Theory-of-mind inference is hard |
| LLM (3B) | discourse-coherence | 0.40 | Surface-level detectable |
| LLM (3B) | rapport | 0.25 | Affect reading from text is hardest |
| LLM (3B) | goal-progress | 0.30 | Requires deep context understanding |
| LLM (3B) | partner-predictability | 0.35 | Computed from partner-goal model |
| LLM (3B) | unresolved-salience | 0.45 | LLM good at urgency/salience detection |
| Simulation | terrain-roughness | 0.90 | Environment provides exact values |
| Simulation | power-level | 0.95 | Internal state, high fidelity |
| Simulation | threat-level | 0.80 | Computed, slightly uncertain |
| Failed parse | any | 0.10 | Uninformative — barely moves beliefs |

### What This Means in Practice

At precision 0.30, a single LLM observation barely moves a belief with
precision 0.50. The update magnitude is roughly:

```
update = learning_rate × (obs_precision / (belief_precision + obs_precision)) × error
       = 0.12 × (0.30 / 0.80) × error
       = 0.045 × error
```

Even a large error (0.5) only shifts the belief by ~0.02 per turn. It takes
consistent signal across multiple turns to meaningfully change the system's
model. This is the correct behavior for a noisy sensor.

A simulation sensor at precision 0.90 against the same belief:

```
update = 0.12 × (0.90 / 1.40) × error = 0.077 × error
```

Nearly double the influence per observation. Hardware sensors matter more
than LLM guesses — as they should.


## Parse Failure Handling

The 3B model will sometimes produce malformed output. The design handles
this through **graceful degradation**, not error recovery.

| Failure Mode | Handling | Effect on System |
|-------------|----------|-----------------|
| Field missing | Default value (0.5) at precision 0.10 | Nearly invisible to beliefs |
| Urgency missing | Default 0.0 at precision 0.10 | Conservative: assume not urgent |
| Value out of range | Clamped to [0, 1] | Bounded, can't produce extreme obs |
| Unrecognized type | Falls back to `unknown` | System treats as moderate surprise |
| Complete LLM failure | All defaults at 0.10 precision | System gets no info this turn |
| Unknown topics | Dropped from resolved list, counted for salience | Triggers unresolved-salience if urgent |

The worst case — total parse failure — is equivalent to a sensor blackout.
The system has no new information. Beliefs drift slightly toward priors.
The cognitive cycle still runs. It just selects actions based on existing
beliefs. The next turn with a successful parse corrects course.

This is the same as a human having a momentary lapse in attention. You miss
what someone said. Your worldview doesn't collapse. You just operate on stale
information until the next input arrives.


## Unresolved Salience — "Something I Don't Understand"

### The Blind Spot

Unknown topics are silently dropped from observation injection. This is
correct for grounding (don't claim to understand ungrounded tokens) but
creates a dangerous failure mode:

```
User: "There is a fire!"
System: "fire" not in topic-map → no observation → acts as if nothing said
```

### The Fix

A meta-observable: `unresolved-salience`. It measures "how much salient
content arrived that I couldn't map to known observables." This is NOT
grounding the unknown token — "fire" still doesn't become an observable.
It registers the *gap* between what was said and what was understood.

Two inputs combine multiplicatively:

```
unresolved_salience = urgency × (unresolved_topics / total_topics)
```

| Scenario | Urgency | Topics | Unresolved | Salience |
|----------|---------|--------|-----------|----------|
| "What's the weather?" | 0.2 | weather | 1/1 | 0.20 |
| "There's a FIRE!" | 0.9 | fire | 1/1 | **0.90** |
| "The terrain is rough" | 0.3 | terrain | 0/1 | 0.00 |
| "Help! The quantum flux!" | 0.8 | help, quantum, flux | 2/3 | **0.53** |
| "How's the battery?" | 0.2 | battery | 0/1 | 0.00 |

The multiplicative structure is important. Low urgency with unknown topics
(casual mention of something outside domain) produces low salience. High
urgency with known topics (excited about something we understand) produces
zero salience. Only **urgent unknowns** register.

### How It Enters EFE

```metta
; Preference: we WANT unresolved-salience to be LOW
(preference unresolved-salience 0.1 0.8)

; Action models: what helps?
(action-model request-clarification unresolved-salience -0.2 0.06)
(action-model query-partner         unresolved-salience -0.1 0.08)
(action-model assert                unresolved-salience  0.05 0.01)
```

When "fire" arrives with high urgency:
1. `unresolved-salience` observation jumps to ~0.9
2. Preference gap: 0.9 - 0.1 = 0.8 (large error)
3. `request-clarification` has lowest EFE (model says it reduces salience by 0.2)
4. System asks: "I'm not sure I understand. Can you explain what you mean?"

No scripted "if unknown topic and urgent, ask." The response **emerges
from EFE**, using the same computation that drives every other action
selection. The system asks because asking is the lowest-energy response
to uncertainty — Active Inference doing what it does.

### Urgency Default

When the LLM fails to parse urgency, it defaults to **0.0** (not 0.5).
This is deliberately conservative: a false negative (missing real urgency)
is safer than a false positive (phantom urgency causing the system to
interrupt normal flow with clarification requests). The low fallback
precision (0.10) further attenuates any influence.

### Diagnostic Queries

```metta
; What topics have I failed to understand recently?
!(recent-unresolved-topics)
→ ((unresolved "fire" (urgency 0.9) (at 42))
   (unresolved "quantum" (urgency 0.3) (at 38)))

; Current state of unresolved-salience belief
!(current-unresolved-salience)
→ (unresolved-salience-state (value 0.72) (precision 0.55)
    (interpretation "something important I don't understand"))
```

The `unresolved-topic-log` also provides a natural path for **topic-map
expansion**: recurring unresolved topics with high urgency are candidates
for new mappings in future deployments.


## Topic → Observable Resolution

NL topic words must map to known observables. This mapping is
**deployment-specific configuration**, like action models and viability bounds.

```metta
; Conversational
(topic-map "confused"    partner-comprehension)
(topic-map "off-topic"   discourse-coherence)
(topic-map "thanks"      rapport)

; Domain (Mars rover example)
(topic-map "battery"     power-level)
(topic-map "terrain"     terrain-roughness)
(topic-map "danger"      threat-level)
```

Unknown topics are **silently dropped**. The system cannot ground them —
it honestly doesn't know what they refer to. If "quantum" isn't in the
topic map, the user mentioning quantum produces no observation. The system
doesn't pretend to understand.

This respects the Honest principle: the system only updates beliefs about
things it has observables for. No hallucinated understanding.


## Verbalization — Outbound Contract

The cognitive core decides **what** to say. The LLM decides **how** to say it.

```
COGNITIVE CORE                         LLM
─────────────                         ───
Action: clarify                       "Let me explain that
Target: partner-comprehension          more clearly. The power
Drivers: comprehension low             levels indicate how much
Tone: concerned + attentive            energy the rover has..."
```

The verbalization prompt is **constrained**:
- Action name → what kind of speech act
- Target → what to address
- Drivers → why (from prediction errors)
- Tone → how (from affect state)
- Topics → about what

The LLM's job is sentence construction within these constraints. Content
decisions have already been made by EFE.

### Fallback Responses

If the LLM fails to verbalize, template fallbacks exist for every
action type:

| Action | Fallback |
|--------|----------|
| clarify | "Let me try to explain that more clearly." |
| elaborate | "Let me expand on that point." |
| confirm | "Just to make sure we're on the same page —" |
| acknowledge | "I understand." |
| query-partner | "Could you tell me more about that?" |
| request-clarification | "I'm not sure I follow. Could you clarify?" |

These are stilted but functional. The system never goes silent.


## Orchestrator Design

### Why Python?

The orchestrator is glue code. It does I/O, prompt construction, HTTP
calls, and regex parsing. None of this is reasoning. Python is the right
tool for glue.

### Why Stateless?

All cognitive state lives in the MeTTa atomspace. The orchestrator
maintains no beliefs, no history, no models. It constructs prompts from
the current MeTTa state, calls the LLM, parses the response, and passes
structured data back to MeTTa.

If the orchestrator crashes and restarts, the MeTTa state is intact.
If you swap one orchestrator for another, the system behaves identically.
The orchestrator is a replaceable I/O adapter.

### Dual Mode: Simulation vs Hyperon

The `MeTTaInterface` class runs in two modes:

| Mode | Backend | Use Case |
|------|---------|----------|
| `simulation` | Python dict mirroring atomspace | Development, testing, no Hyperon needed |
| `hyperon` | Real Hyperon MeTTa runtime | Production, full cognitive core |

Simulation mode implements precision-weighted belief updates and simplified
EFE computation in Python, matching the MeTTa logic. This allows testing
the full orchestrator → parse → cycle → verbalize loop without a working
Hyperon installation.


## Integration with Existing Modules

### Loader Addition

```metta
; In loader.metta — after conversation_model, before initialization
!(import! &self core/perception)
```

### Updated Architecture Diagram

```
LAYER 1: COGNITIVE CORE
  foundations → beliefs → affect → actions → cycle → policy_efe

LAYER 2: KNOWLEDGE MODULES
  semantic_primitives → dimensional_primitives → self_model

LAYER 2.5: GROUNDING INTEGRATION
  semantic_grounding → action_grounding

LAYER 3: INTERACTION MODULE
  conversation_model

LAYER 4: PERCEPTION BOUNDARY  ← NEW
  perception.metta      (inbound: NL/sensor → observations)
  proprioception.metta   (outbound: intent vs output → fidelity observation)
  orchestrator.py        (Python-side glue)
  llama.cpp              (LLM inference server)
```

### Dependencies

| perception.metta References | Defined In |
|---|---|
| `inject-observation!` | cycle.metta |
| `process-utterance!` | conversation_model.metta |
| `advance-turn!` | conversation_model.metta |
| `all-prediction-errors` | beliefs.metta |
| `get-affect-state`, `compute-valence`, `compute-arousal` | affect.metta |
| `conversation-state-description`, `current-topics` | conversation_model.metta |
| `get-config`, `clamp` | foundations.metta |

| perception.metta Provides | Used By |
|---|---|
| `perceive-utterance!` | orchestrator.py |
| `perceive-sensor!` | orchestrator.py (2D environment) |
| `package-verbalization-intent` | orchestrator.py |
| `perception-precision` atoms | self-model introspection |


## Running the System

### Prerequisites

```bash
# 1. Get llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# 2. Get a 3B model (Llama 3.2 3B Instruct recommended)
# Download GGUF from huggingface

# 3. Start the server
./llama-server -m llama-3.2-3b-instruct-q4_k_m.gguf --port 8080
```

### Run Chat

```bash
# With LLM
python orchestrator.py --llama-url http://localhost:8080

# Without LLM (template responses, for testing cycle logic)
python orchestrator.py --no-llm

# With real MeTTa backend (requires Hyperon)
python orchestrator.py --mode hyperon

# Verbose logging
python orchestrator.py --log-level DEBUG
```

### Interactive Commands

| Command | Effect |
|---------|--------|
| `state` | Print current beliefs (value + precision + visual bar) |
| `trace` | Print EFE scores for all actions (simulation mode) |
| `quit` | Exit |


## Path to 2D Environment

When the simulation environment is added, it connects through the same
perception layer with no architectural changes:

```
2D Environment                    Chat Interface
     │                                 │
     ▼                                 ▼
perceive-sensor!                  perceive-utterance!
(direct, high precision)          (LLM-derived, low precision)
     │                                 │
     └──────────── both ──────────────┘
                    │
          inject-observation!
          (cognitive core doesn't care about source)
                    │
            Cognitive Cycle (EFE)
                    │
        ┌───────────┴───────────┐
   Domain action            Conversational action
   (move sprite)            (verbalize via LLM)
```

Sensor configuration for the environment:

```metta
; Map simulation sensors to observables
(sensor-config position-x    terrain-roughness  0.0 100.0)
(sensor-config battery-gauge  power-level        0.0 5.0)
(sensor-config threat-radar   threat-level       0.0 1.0)

; These get simulation precision (0.85-0.95)
; vs LLM observations at 0.25-0.40
```

Domain actions (`move-forward`, `retreat`, `observe`) go to the simulation
engine. Conversational actions (`clarify`, `assert`) go to the LLM for
verbalization. Both are selected by the same EFE computation. The agent
might alternate between moving its sprite and explaining what it's doing —
and the choice emerges from the EFE landscape, not from scripted mode
switching.


## Fine-Tuning Path

Once the system accumulates conversation logs, training signal exists:

1. The LLM produces a comprehension estimate of 0.8
2. The system acts on this (doesn't clarify)
3. Next turn, the partner says "I still don't understand"
4. Prediction error spikes on partner-comprehension
5. This is a supervision signal: the LLM's estimate was wrong

By collecting (parse, subsequent_prediction_errors) pairs, you can
fine-tune the 3B model to produce better-calibrated estimates. The
cognitive core's error signal teaches the perception layer.

This is the bootstrap: the system's own surprise becomes training data
for its sensors.


## Design Decisions

### Q: Why not have the LLM produce MeTTa atoms directly?

A 3B model can't reliably generate valid MeTTa syntax. The rigid
line-per-field format with regex extraction is more robust. The
orchestrator translates between the two representations.

### Q: Why not use the LLM for reasoning too?

The architecture principle: "Symbolic Over Statistical — for reasoning,
use symbols." The LLM does perception (parsing) and production
(verbalization). Reasoning happens in MeTTa where it's transparent,
traceable, and queryable.

### Q: Why not a dedicated perception-error model?

Discussed and rejected. Misparses produce false beliefs. False beliefs
produce prediction errors. Prediction errors drive correction. The core
loop already handles this. Adding a perception-error model would be a
parallel mechanism for something the invariant (EFE minimization) already
addresses.

### Q: Why such low precision for the LLM?

A 3B model guessing "how confused is this person?" from text is genuinely
uncertain. 0.30 precision means "I have some information but I'm not
confident." This is honest. If we fine-tune the model and it gets better,
we raise the precision. The system's self-correction rate improves, but
the architecture doesn't change.

### Q: What about the verbalization fidelity problem?

The LLM might generate text that adds claims or changes epistemic status.
This is addressed by `proprioception.metta` — the system re-parses its own
output, compares it against the intent, and injects the difference as a
`verbalization-fidelity` observation. For claim-bearing actions, an assertion
gate blocks unfaithful output. See `PROPRIOCEPTION_DESIGN.md` for the full
architecture.

### Q: Why not add a special handler for unknown urgent topics?

The temptation is to add `if unknown_topic and urgent: ask_for_clarification`.
That's scripted behavior — it violates the bottom-up principle. Instead,
`unresolved-salience` is a regular observable with beliefs, preferences, and
action models. EFE selects clarification because the action models say it
reduces salience, not because of an if-statement. The system doesn't "know"
it should ask about things it doesn't understand — that behavior *emerges*
from the same optimization that drives every other action. This is the
architecture working as designed: one invariant (minimize EFE), uniform
treatment of all observables, emergent behavior.


## Files

| File | Lines | Role |
|------|-------|------|
| `perception.metta` | ~580 | MeTTa-side contracts, precision, topic mapping, unresolved-salience |
| `proprioception.metta` | ~260 | Motor control: fidelity observable, assertion gate, action models |
| `orchestrator.py` | ~1400 | Python glue: LLM client, parser, verbalizer, proprioceptive check, main loop |
| `PERCEPTION_DESIGN.md` | this file | Perception architecture documentation |
| `PROPRIOCEPTION_DESIGN.md` | companion | Proprioception architecture documentation |
