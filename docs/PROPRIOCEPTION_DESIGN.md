# Proprioception â€” Motor Control for Verbalization

## What This Module Does

`proprioception.metta` + extensions to `orchestrator.py` implement a
feedback loop that lets the cognitive core monitor the fidelity of its
own verbalized output. The system re-parses its generated text, compares
it against the intended action, and injects the difference as a
prediction error â€” the same way it handles every other signal.

```
BEFORE:                                AFTER:

  Cognitive Core                        Cognitive Core
       â”‚                                     â”‚
  intent (clarify,                      intent (clarify,
    concerned,                            concerned,
    partner-comp)                         partner-comp)
       â”‚                                     â”‚
       â–¼                                     â–¼
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚ LLM      â”‚                          â”‚ LLM      â”‚
  â”‚ verbalizeâ”‚                          â”‚ verbalizeâ”‚
  â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜                          â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜
       â”‚                                     â”‚
       â–¼                                     â”œâ”€â”€â†’ response text
  response text                              â”‚
  (hope for the best)                        â–¼
                                        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                        â”‚ Re-parse       â”‚ â† same parser
                                        â”‚ (LLM call #3)  â”‚
                                        â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
                                                â”‚
                                                â–¼
                                        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                        â”‚ Compare intent â”‚
                                        â”‚ vs re-parse    â”‚
                                        â”‚ (structural)   â”‚
                                        â””â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”˜
                                            â”‚       â”‚
                                    Layer 1 â”‚       â”‚ Layer 2
                                            â”‚       â”‚
                                            â–¼       â–¼
                              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                              â”‚ Fidelity obs â”‚  â”‚ Assertion    â”‚
                              â”‚ â†’ inject     â”‚  â”‚ Gate         â”‚
                              â”‚ â†’ core adaptsâ”‚  â”‚ â†’ block/retryâ”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```


## The Biological Analogy

In biological motor control, proprioception isn't a verification gate that
checks before an action commits. It's **concurrent feedback**. Your motor
cortex issues a command AND simultaneously predicts the sensory consequence.
Proprioceptors report the actual consequence. The prediction error drives
real-time correction â€” but the action is already in flight. You don't freeze
your arm mid-reach to verify it's going the right direction; you correct
continuously.

The cognitive core issues a verbalization intent ("clarify about power-level,
concerned tone"). That intent is a **prediction** about what the output text
will contain. The re-parse of the generated text reports what it **actually**
contains. The difference is prediction error â€” just like everything else.

A verification gate ("check-then-send") would be an assembly line quality
inspector. It works, but it's architecturally foreign to a system built on
continuous prediction error. This design stays within the invariant.


## Two-Layer Architecture

### Layer 1: Proprioceptive Observable (every turn)

Re-parse the generated text. Compare against intent. Compute a
`verbalization-fidelity` score (0â€“1). Inject as observation. The
cognitive core treats it like any other observable: precision-weighted
belief update, feeds into EFE, drives action selection.

**Emergent motor adaptation**: If verbalization is persistently unreliable,
the system experiences high prediction error on `verbalization-fidelity`.
This drives up arousal (something is wrong), lowers dominance (I'm not
in control of my output), and shifts the EFE landscape toward actions
whose models predict higher fidelity. An `acknowledge` is much harder to
botch than an `elaborate`. The system naturally gravitates toward simpler
communicative actions. No scripted simplification rule â€” it emerges from
the same EFE computation that drives every other decision.

### Layer 2: Assertion Gate (claim-bearing actions only)

For actions that make claims about the world (`assert`, `elaborate`), low
fidelity isn't just a learning signal â€” it's a **safety concern**. The LLM
might generate text asserting something the system doesn't believe. This
extends `safety.metta`'s assertion consistency check from structural content
to the actual generated NL text.

The gate:
1. Only activates for claim-bearing actions (`assert`, `elaborate`)
2. Checks composite fidelity against threshold (0.4)
3. If below threshold: block, retry once with tighter prompt
4. If retry also fails: fall back to template response
5. Template responses have fidelity 1.0 by definition (stilted but faithful)

This is the pain reflex â€” you pull your hand back before the cortex
processes what happened. Fast, unconditional, limited scope.


## Fidelity Computation

Three structural dimensions, compared at the field level (coarse, robust
to parser noise):

### 1. Action Type Fidelity (weight: 0.45)

Did the speech act match the intent?

| Intended | Re-parsed | Score | Why |
|----------|-----------|-------|-----|
| clarify | assertion | 1.0 | Clarifications are assertions |
| clarify | query | 0.6 | Partial â€” still interactive |
| query-partner | query | 1.0 | Exact match |
| assert | query | 0.2 | Bad: intended to tell, asked instead |
| acknowledge | greeting | 1.0 | Brief responses match |

The matching is deliberately coarse. Multiple re-parse types are acceptable
for each intended action. Only gross mismatches (intended assertion, got
question) register as errors.

### 2. Topic Fidelity (weight: 0.30)

Did we address the right things? Jaccard similarity between intended
topics (from drivers + conversation topics) and actual topics (from
re-parse).

```
intended: {partner-comprehension, power-level}
actual:   {power, understanding}      â†’ resolves to same observables
overlap:  2/2 = 1.0                   â†’ perfect fidelity

intended: {partner-comprehension}
actual:   {weather, sports}           â†’ no overlap
overlap:  0/3 = 0.0                   â†’ complete drift
```

### 3. Tone Fidelity (weight: 0.25)

Was the affect appropriate? Checks for gross valence/arousal mismatches:

| Check | Penalty | When |
|-------|---------|------|
| Intended concern, sounds cheery | -0.4 | valence < -0.2 but rapport > 0.8 |
| Intended warmth, sounds cold | -0.3 | valence > 0.2 but rapport < 0.3 |
| High engagement, unclear output | -0.3 | arousal > 0.5 but comprehension < 0.3 |

Tone checking is inherently fuzzy. We only penalize gross mismatches.
The system can't distinguish "slightly warmer than intended" from parser
noise, and shouldn't try.

### Composite Score

```
fidelity = 0.45 Ã— action_type + 0.30 Ã— topic + 0.25 Ã— tone
```

These weights are declared in MeTTa and readable by the core:
```metta
(fidelity-weight action-type-fidelity 0.45)
(fidelity-weight topic-fidelity 0.30)
(fidelity-weight tone-fidelity 0.25)
```


## How It Enters the Cognitive Core

### Belief and Preference

```metta
(belief verbalization-fidelity 0.7 0.4)      ; Prior: works okay
(preference verbalization-fidelity 0.9 0.7)  ; Want high fidelity
(viability-bound verbalization-fidelity 0.25 1.0)  ; Can't converse if broken
```

### Action Models â€” Verbalization Difficulty

Some actions are harder to verbalize. A 3B model can nail "I understand"
but might botch a nuanced explanation.

```metta
; Easy â€” simple speech acts
(action-model acknowledge      verbalization-fidelity  0.05 0.02)
(action-model confirm          verbalization-fidelity  0.03 0.02)
(action-model conclude         verbalization-fidelity  0.02 0.02)

; Moderate â€” structured but focused
(action-model clarify          verbalization-fidelity  0.0  0.03)
(action-model query-partner    verbalization-fidelity  0.02 0.03)
(action-model request-clarif.  verbalization-fidelity  0.02 0.02)

; Hard â€” complex content, risk of drift
(action-model elaborate        verbalization-fidelity -0.05 0.04)
(action-model assert           verbalization-fidelity -0.08 0.05)
(action-model redirect         verbalization-fidelity -0.03 0.03)
```

### Emergent Behavior

When `verbalization-fidelity` is persistently low:

1. Belief drops: (0.7 â†’ 0.4 â†’ 0.3)
2. Preference gap grows: (0.9 - 0.3 = 0.6)
3. Actions with positive fidelity effect get lower EFE
4. `acknowledge` (fidelity +0.05) becomes more attractive than `elaborate` (fidelity -0.05)
5. System simplifies its communicative behavior

This is the same mechanism as a person who, struggling with a foreign
language, switches from complex explanations to simple confirmations.
Not because they decided "I should use simpler sentences" but because
simpler sentences are the path of least resistance through the EFE
landscape. Emergent, not scripted.


## Precision Rationale

```
(perception-precision proprioception verbalization-fidelity 0.50)
```

Higher than most LLM-derived observables (0.25â€“0.40) because we're comparing
structured fields against each other, not inferring unobservable mental states.
But still LLM-derived (the re-parse IS an LLM call) so not as reliable as
direct sensors (0.85â€“0.95).

The 0.50 precision means proprioceptive feedback has moderate influence per
turn â€” more than a single conversational estimate but less than a direct
sensor. Over 3â€“5 turns of consistently low fidelity, the belief moves
meaningfully. This is the right timescale for motor adaptation.


## The Assertion Gate in Detail

### When Does It Fire?

```
action selected = "assert"
        â”‚
        â–¼
    Is action claim-bearing?  â”€â”€noâ”€â”€â†’  Layer 1 only (inject fidelity)
        â”‚yes
        â–¼
    composite fidelity â‰¥ 0.4?  â”€â”€yesâ”€â”€â†’  Pass (send response)
        â”‚no
        â–¼
    BLOCK. Build retry prompt
        â”‚
        â–¼
    Retry with tighter constraints
    (lower temperature, explicit corrections)
        â”‚
        â–¼
    Re-check retry. Pass?  â”€â”€yesâ”€â”€â†’  Send retry
        â”‚no
        â–¼
    Fall back to template response
    (fidelity = 1.0)
```

### Retry Prompt Construction

The retry prompt adds explicit corrections based on which dimensions failed:

| Failed Dimension | Correction Added |
|-----------------|-----------------|
| action_type_fidelity < 0.5 | "Make a clear statement, do not ask a question." |
| topic_fidelity < 0.5 | "Focus specifically on: {driver topics}." |
| tone_fidelity < 0.5 | "Use a concerned and careful tone." |

Maximum one retry. The cost is one extra LLM call (re-parse of retry) for
a total of 5 LLM calls in the worst case (parse input, verbalize, re-parse,
retry, re-parse retry) vs 3 in the normal case (parse, verbalize, re-parse).

### Why Templates Are Fidelity 1.0

Template responses ("Let me try to explain that more clearly.") are
constructed by the orchestrator, not the LLM. They match the intent by
definition â€” they're just not very natural. Injecting fidelity 1.0 for
templates means the proprioceptive loop doesn't penalize the system for
using its safety fallback.

This prevents a pathological loop: low fidelity â†’ template â†’ high fidelity
â†’ try LLM again â†’ low fidelity â†’ template â†’ ... The system should learn
that its verbalization is unreliable (from the turns where the LLM was
used), not that templates fix everything.


## Data Flow â€” Complete Turn

```
1. User: "I don't get the power levels"

2. Parse (LLM call #1):
   TYPE: query, TOPICS: power,understanding, COMP: 0.3, URGENCY: 0.2

3. Inject 6 observations â†’ cognitive cycle â†’ action: clarify

4. Package intent:
   {action: clarify, drivers: [{obs: partner-comprehension, err: 0.3}],
    valence: -0.1, arousal: 0.5, topics: [power-level]}

5. Verbalize (LLM call #2):
   "Let me explain the power levels more clearly. The power level
    shows how much energy the rover has available for operations."

6. Re-parse (LLM call #3):
   TYPE: assertion, TOPICS: power,energy, COMP: 0.7, URGENCY: 0.1

7. Fidelity computation:
   action_type: clarify â†’ expected {assertion,emotional}, got assertion â†’ 1.0
   topic:       intended {partner-comprehension,power-level}
                actual {power,energy} â†’ overlaps on power-level â†’ 0.7
   tone:        intended valence=-0.1, arousal=0.5
                reparsed rapport=0.5, comp=0.7 â†’ no gross mismatch â†’ 1.0

   composite = 0.45Ã—1.0 + 0.30Ã—0.7 + 0.25Ã—1.0 = 0.91

8. Gate check: clarify is not claim-bearing â†’ gate-pass

9. Inject proprioceptive observation:
   (observation verbalization-fidelity 0.91 0.50 42)

10. Send response to user.
```


## Integration

### Module Dependencies

```
proprioception.metta reads from:
  foundations.metta     â€” types, config, utilities
  beliefs.metta         â€” inject-observation!, get-belief-*
  perception.metta      â€” get-perception-precision, field-or-default
  safety.metta          â€” assertion-action? (extended, not replaced)

proprioception.metta provides:
  verbalization-fidelity     â€” observable, belief, preference, viability bound
  action models              â€” fidelity effects for all conversational actions
  perceive-proprioceptive!   â€” observation injection entry point
  assertion-gate-check       â€” gate pass/block decision
  diagnose-fidelity-failure  â€” correction hints for retry prompt

orchestrator.py extensions:
  ProprioceptiveCheck class  â€” re-parse, compare, compute fidelity
  process_turn Step 5-6      â€” proprioceptive check + assertion gate
  inject_proprioception      â€” MeTTaInterface method
```

### Loader Addition

```metta
; In loader.metta â€” after perception, before initialization
!(import! &self core/proprioception)
```

### Architecture Layer

```
LAYER 4: PERCEPTION BOUNDARY
  perception.metta        â€” inbound: NL/sensor â†’ observations
  proprioception.metta    â€” outbound: intent vs output â†’ fidelity observation
  orchestrator.py         â€” glue: prompt construction, LLM calls, comparison
  llama.cpp               â€” LLM inference server
```


## Cost Analysis

| Scenario | LLM Calls | When |
|----------|-----------|------|
| Normal turn | 3 | Parse + verbalize + re-parse |
| Gate blocks, retry passes | 5 | + retry verbalize + re-parse retry |
| Gate blocks, retry fails | 5 | Same, but template used |
| LLM down entirely | 0 | Template all the way |

The third LLM call (re-parse) is the proprioceptive cost. It uses the same
parse prompt template at low temperature, so it's fast and deterministic.
On a 3B model via llama.cpp, this adds ~100â€“200ms per turn. Acceptable for
a conversational agent.


## Design Decisions

### Q: Why not just gate every response (not just assertions)?

The gate is expensive (retry = 2 more LLM calls) and heavy-handed. For
non-assertive actions (acknowledge, confirm, query), a slightly off-tone
or off-topic response is suboptimal but not dangerous. The fidelity
observation (Layer 1) handles it through gradual adaptation. Reserve the
gate for safety-critical cases where wrong output could cause harm.

### Q: Won't the re-parse be noisy since the parser is noisy?

Yes. That's why comparison is **structural and coarse**. We check "is the
re-parse in the right category?" not "did it match exactly?" The soft-match
scoring gives partial credit. A perfect fidelity score means "nothing is
grossly wrong," not "the output is exactly what was intended." This is
the right level of confidence for a noisy sensor.

### Q: Could proprioceptive feedback create oscillations?

Potentially: low fidelity â†’ simplify actions â†’ high fidelity â†’ try complex
actions â†’ low fidelity â†’ ... This is actually healthy exploration, same as
a system learning the boundary of its capabilities. The precision-weighted
belief update provides natural damping. As the belief precision increases
(more data), the oscillation amplitude decreases and the system converges
on the action complexity level its LLM can faithfully execute.

### Q: What about the latency cost?

One extra LLM call per turn (~100â€“200ms on 3B). For a conversational agent
with human response times, this is negligible. For the 2D environment with
rapid cycles, proprioception could be made **intermittent** â€” check every
Nth verbalization, not every one. The fidelity belief persists between
checks, so the EFE influence continues even during unchecked turns.

### Q: How does this interact with unresolved-salience?

They're independent observables that both feed EFE. Unresolved-salience
measures "I don't understand the input." Verbalization-fidelity measures
"I can't control my output." A system could have both problems simultaneously
(doesn't understand AND can't express itself), which would produce
compounding prediction errors and strong pressure toward the simplest
possible response: `request-clarification` (addresses both â€” asks for help
AND is easy to verbalize). Again, emergent, not scripted.


## Fine-Tuning Path

Proprioceptive logs create training data for the LLM:

```
(proprioceptive-log clarify 0.91 gate-pass 42)
(proprioceptive-log assert  0.31 gate-block 43)
(proprioceptive-log assert  0.72 gate-pass 44)  â† after retry
```

The (intent, generated_text, fidelity_score) triples can fine-tune the
verbalization prompt or the model itself. Low-fidelity examples are
negative training signal; high-fidelity examples are positive.

This closes the same bootstrap loop as perception fine-tuning: the system's
own prediction errors become supervision signal for its neural components.


## Files

| File | Lines | Role |
|------|-------|------|
| `proprioception.metta` | ~260 | MeTTa-side: observable, beliefs, action models, gate, diagnostics |
| `orchestrator.py` | extended | Python-side: ProprioceptiveCheck class, retry logic, fidelity injection |
| `PROPRIOCEPTION_DESIGN.md` | this file | Architecture documentation |
