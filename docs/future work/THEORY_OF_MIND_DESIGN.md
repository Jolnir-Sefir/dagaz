# Theory of Mind Extension — Design Document
### Recursive Partner Modeling via Active Inference
**Companion to `conversation_model.metta` — extends 0th-level ToM to recursive belief modeling.**

---

## Motivation

The current conversation model (`conversation_model.metta` v2) treats the conversational partner as a weather system — five scalar observables (comprehension, coherence, rapport, predictability, goal-progress) maintained as beliefs and kept in preferred ranges through EFE-driven action selection. This is 0th-level Theory of Mind: the agent tracks *that* the conversation is going well or poorly, but has no model of *what the partner believes*, *what the partner wants*, or *what the partner expects the agent to do*.

This limitation blocks several conversational capabilities that require modeling the partner's internal state:

| Capability | What It Requires | Current Status |
|-----------|-----------------|----------------|
| Reference resolution | Tracking what the partner is attending to | No partner attention model |
| Implicature | Modeling partner's surprise at agent's utterances | No model of partner expectations |
| Targeted repair | Knowing *what* the partner misunderstood | Only aggregate `partner-comprehension` |
| Topic management | Inferring partner's goals and interests | No partner goal model |
| Turn-taking | Predicting when partner expects a response | No partner expectation model |

The central claim of this document is that **Theory of Mind is not a separate module — it is more observables in the same EFE framework.** The partner's beliefs are observables. The partner's goals are observables. The partner's expectations are observables. All enter the same prediction-error machinery. No new invariant is needed.

---

## Part I: The Current Architecture

### What Exists (Level 0)

The agent maintains beliefs about five conversational dimensions:

```metta
(belief partner-predictability 0.5 0.01)
(belief partner-comprehension 0.6 0.01)
(belief discourse-coherence 0.7 0.01)
(belief goal-progress 0.3 0.01)
(belief rapport 0.5 0.01)
```

These are *summary statistics* about the conversation. They support action selection ("comprehension is low → clarify has low EFE → clarify is selected") but carry no information about the partner's specific beliefs, goals, or reasoning.

### What's Missing

The agent cannot answer any of these questions:
- What does the partner believe about terrain-roughness?
- Does the partner know that power is low?
- What is the partner trying to achieve in this conversation?
- Is the partner expecting me to elaborate or change topic?
- Did the partner understand my last assertion correctly?

These are all Theory of Mind questions. Answering them requires modeling the partner as an agent with beliefs, goals, and expectations — not just as a source of conversational signals.

---

## Part II: The Design — Partner Model as Observables

### Principle: Same Schema, New Domain

A partner-model belief uses the identical `(belief $obs $value $precision)` schema. The observable names are prefixed to distinguish them from the agent's own beliefs:

```metta
;; Agent's own belief about terrain
(belief terrain-roughness 0.3 0.7)

;; Agent's belief about what the PARTNER believes about terrain
(belief pm-terrain-roughness 0.5 0.2)
```

The `pm-` prefix denotes "partner model." The low precision (0.2) encodes "I'm not very confident about what the partner thinks." This low precision creates info-gain for actions that would reveal the partner's belief — like asking them.

This is Level 1 Theory of Mind: the agent has beliefs about the partner's beliefs.

### Why This Works Within the Architecture

No new machinery is required because:

1. **EFE computation** already handles arbitrary observables. Adding `pm-X` observables changes nothing about the EFE formula.
2. **Prediction error** works identically. If the agent predicts `pm-terrain-roughness ≈ 0.3` but the partner's response is consistent with them believing `0.7`, that mismatch is a prediction error that drives belief update.
3. **Action models** already encode how actions affect observables. We add models for how conversational actions affect partner beliefs: `(action-model assert pm-terrain-roughness 0.1 0.06 0.3)` — asserting about terrain moves the partner's belief toward the agent's.
4. **Metabolic selection** already handles hypothesis proliferation. Partner-model observables that don't reduce prediction error die.
5. **Structure learning** already discovers correlated prediction errors. If `pm-terrain-roughness` and `pm-threat-level` co-vary, the system discovers that relationship without being told.

### Level 1 Partner Model Schema

```metta
;; ─── Partner Belief Observables ───
;; Schema: (belief pm-<observable> <estimated-value> <precision>)
;; These are the agent's beliefs about what the partner believes.

(belief pm-terrain-roughness 0.5 0.1)    ;; low precision = don't know what they think
(belief pm-threat-level 0.3 0.1)
(belief pm-power-level 0.5 0.1)

;; ─── Partner Goal Observable ───
;; What is the partner trying to achieve?
;; Value encodes estimated goal-state; precision encodes confidence.
(belief pm-goal-state 0.3 0.1)

;; ─── Partner Expectation Observable ───
;; What does the partner expect the agent to do next?
;; Updated from conversational context (topic, last action, discourse state).
(belief pm-expected-action 0.5 0.1)

;; ─── Partner Attention Observable ───
;; What is the partner currently attending to?
;; Salience = recency × precision of most recently updated pm-X observable.
(belief pm-current-focus 0.5 0.1)
```

### Action Models for Partner Beliefs

Conversational actions now have dual effects: effects on the conversation (existing) and effects on the partner model (new).

```metta
;; ─── Assert: moves partner belief toward agent's belief ───
;; The partner hears an assertion and (ideally) updates their belief.
(action-model assert pm-terrain-roughness 0.1 0.06 0.3)
;; value-delta: positive (partner's belief moves toward agent's)
;; prec-delta: positive (agent gains confidence about what partner now believes)

;; ─── Clarify: doesn't change partner belief, reveals it ───
(action-model clarify pm-terrain-roughness 0.0 0.10 0.3)
;; value-delta: zero (clarifying reveals, doesn't change)
;; prec-delta: high (agent learns what partner actually thinks)

;; ─── Query-partner: explicitly asks what partner believes ───
(action-model query-partner pm-terrain-roughness 0.0 0.15 0.3)
;; Highest precision gain — directly asks the partner

;; ─── Confirm: tests whether partner's belief matches agent's ───
(action-model confirm pm-terrain-roughness 0.0 0.12 0.3)
;; Similar to query but more targeted — checks a specific belief

;; ─── Elaborate: strengthens partner's belief, risks overload ───
(action-model elaborate pm-terrain-roughness 0.05 0.04 0.3)
;; Small value change, small precision gain, but may reduce pm-comprehension
```

### Deriving Existing Observables from Partner Model

The current summary statistics become *derived* quantities rather than primitives:

```metta
;; partner-comprehension becomes the AGGREGATE prediction error
;; across all pm-X observables — "how well does my model of
;; them match their actual responses?"
(= (derived-partner-comprehension)
   (let* (
     ($pm-obs (all-partner-model-observables))
     ($errors (map-pm-prediction-errors $pm-obs))
     ($mean-error (mean $errors))
   )
   (- 1.0 $mean-error)))

;; partner-predictability becomes "how well do my models of
;; their beliefs predict what they say next?"
(= (derived-partner-predictability)
   (let* (
     ($pm-obs (all-partner-model-observables))
     ($prec-sum (sum-pm-precisions $pm-obs))
     ($n (length $pm-obs))
   )
   (/ $prec-sum (max 1 $n))))
```

This is the right direction: summary statistics emerge from detailed models, not the other way around. The transition can be gradual — both primitive and derived versions can coexist during development, with the derived versions replacing the primitives as the partner model matures.

---

## Part III: Emergent Conversational Capabilities

### 1. Reference Resolution

**Problem:** The partner says "it's getting worse." What does "it" refer to?

**Current system:** No mechanism. The LLM at the perception boundary must resolve all references.

**With partner model:** "It" refers to whichever `pm-X` observable has the highest current salience. Salience is a function of recency (recently updated) and precision (confidently modeled). If the last few turns have been about `pm-terrain-roughness`, that observable has the highest salience, and "it" resolves there.

```metta
;; Salience of a partner-model observable
;; Higher = more likely referent of ambiguous pronouns
(= (pm-salience $pm-obs)
   (let* (
     ($prec (match &state (belief $pm-obs $_ $p) $p))
     ($recency (pm-recency $pm-obs))
   )
   (* $prec $recency)))

;; Most salient = most likely referent
(= (likely-referent)
   (argmax-salience (all-partner-model-observables)))
```

No special reference resolution module. Salience is already implicit in the belief dynamics. The extension just makes it queryable.

### 2. Implicature (Gricean Pragmatics)

**Problem:** The agent asserts something the partner already believes. Why? The assertion must carry meaning beyond its literal content.

**With partner model:** The agent can detect when an assertion would be *uninformative* to the partner:

```metta
;; Pre-assertion check: would this be news to the partner?
(= (assertion-informativeness $obs)
   (let* (
     ($my-belief (match &state (belief $obs $v $_) $v))
     ($pm-belief (match &state (belief pm-$obs $v $_) $v))
     ($pm-prec (match &state (belief pm-$obs $_ $p) $p))
   )
   ;; If partner already believes what I'd assert (and I'm confident they do),
   ;; the assertion carries low literal informativeness.
   ;; High pm-prec + small belief gap = low informativeness.
   (* $pm-prec (abs (- $my-belief $pm-belief)))))
```

Low informativeness for a literal assertion creates EFE pressure *against* asserting — which is correct. If the agent asserts anyway (because of other EFE pressures), the partner's prediction error ("why did they tell me something I already know?") signals that the assertion carried non-literal meaning. The system detects this through the mismatch between predicted and actual partner response.

**Important note on scope:** Full Gricean pragmatics (quantity, quality, relation, manner) is a research-level problem. The above mechanism captures one specific implicature pattern — asserting known information to convey something beyond the literal. Other patterns will require further development. This is a starting point, not a complete theory.

### 3. Targeted Repair

**Problem:** The partner misunderstands something. Currently, the agent knows "comprehension is low" but not *what* was misunderstood.

**With partner model:** The agent can identify specific belief mismatches:

```metta
;; Detect specific misunderstanding
;; Compare: what I asserted vs what the partner seems to believe now
(= (detect-misunderstanding $obs)
   (let* (
     ($my-belief (match &state (belief $obs $v $_) $v))
     ($pm-belief (match &state (belief pm-$obs $v $_) $v))
     ($pm-prec (match &state (belief pm-$obs $_ $p) $p))
     ($gap (abs (- $my-belief $pm-belief)))
   )
   ;; Misunderstanding = high-confidence partner belief that differs from mine
   ;; after I've asserted about this topic
   (if (and (> $pm-prec 0.3) (> $gap 0.2))
       (misunderstanding $obs (gap $gap) (pm-belief $pm-belief))
       (no-misunderstanding $obs))))

;; Find all current misunderstandings
(= (all-misunderstandings)
   (collapse
     (match &state (belief pm-$obs $_ $_)
       (let $result (detect-misunderstanding $obs)
         (case $result
           (((misunderstanding $_ $_ $_) $result)
            (_ (empty))))))))
```

This transforms repair from "say it again, differently" to "correct *this specific* belief mismatch." The EFE computation naturally selects the repair target: the misunderstanding with the highest prediction error gets the most EFE pressure.

### 4. Topic Management via Goal Inference

**Problem:** What does the partner want to talk about? When should the agent change topic?

**With partner model:** The partner's goals are observables with prediction errors:

```metta
;; Partner goal observables
;; These track what the partner is trying to achieve in the conversation.
(belief pm-interested-in-terrain 0.5 0.1)
(belief pm-interested-in-safety 0.5 0.1)
(belief pm-wants-advice 0.5 0.1)

;; Preferences: the agent prefers to address the partner's interests
(preference pm-interested-in-terrain 0.7 0.3)
(preference pm-interested-in-safety 0.7 0.3)
(preference pm-wants-advice 0.7 0.3)

;; When the agent misreads the partner's interest (high prediction error
;; on pm-interested-in-X), the EFE naturally favors actions that would
;; reveal the partner's actual interest — query-partner, request-clarification.
;; No topic management logic needed.
```

### 5. Turn-Taking

**Problem:** When should the agent speak vs wait for the partner?

**With partner model:** The partner's expectation of a response is an observable:

```metta
(belief pm-expects-response 0.5 0.2)

;; High pm-expects-response + high precision = the partner is waiting for us
;; Low pm-expects-response + high precision = the partner wants to keep talking
;; Low precision = we don't know → info-gain for observing (listening)
```

This integrates with the existing EFE: when `pm-expects-response` is high and the agent has something to say, the EFE for assert/elaborate/clarify drops. When it's low, the EFE for `acknowledge` or `wait` drops instead.

---

## Part IV: Recursive Depth

### Level 2: I Believe the Partner Believes I Believe X

Level 2 ToM is needed for specific pragmatic phenomena:

- **Common ground:** Knowing that the partner knows you know something changes how you communicate about it. You can use shorthand, ellipsis, indirect reference.
- **Strategic communication:** "If I tell them X, they'll realize I know Y" — reasoning about the partner's reasoning about your reasoning.
- **Deception detection:** "They said X, but if they knew I believe Y, they would have said Z instead" — detecting inconsistency between expected and actual partner communication strategy.

```metta
;; Level 2: agent's belief about what the partner believes the agent believes
(belief pm2-terrain-roughness 0.5 0.1)

;; Interpretation: "I think they think I think terrain-roughness is 0.5"
;; If the agent actually believes terrain-roughness is 0.3, and the agent
;; believes the partner knows this, then pm2-terrain-roughness should be
;; close to 0.3. A large gap between pm2-X and the agent's own belief
;; about X signals a breakdown in common ground.
```

### Emergent Recursion Depth

**Do not prescribe recursion depth.** Let the metabolic system decide.

The architecture already has the mechanism: structure learning creates hypotheses, metabolic selection retains the useful ones. Apply the same principle to ToM depth:

1. Level 1 (`pm-X`) observables are created when the conversation model is active. They start with metabolic energy from the initial allocation.
2. If Level 1 observables consistently reduce prediction error on partner utterances, they survive and earn energy.
3. Level 2 (`pm2-X`) observables are created only if Level 1 observables show persistent prediction errors that Level 2 could explain. This is the same trigger as Phase 2 structure learning (dense error cluster → latent variable hypothesis).
4. Level 3+ follows the same pattern. In practice, the metabolic cost of maintaining Level 3 observables will exceed their prediction-error reduction in most conversations, and they will die.

```metta
;; Metabolic cost scales with recursion depth
;; (This makes deep recursion expensive — it must earn its keep)
(= (pm-metabolic-cost $level)
   (* (get-config base-pm-cost) (pow (get-config pm-depth-penalty) $level)))

;; Level 1: cost = base × 1.5^1 = 1.5× base
;; Level 2: cost = base × 1.5^2 = 2.25× base
;; Level 3: cost = base × 1.5^3 = 3.375× base
;; Level 3 must be 3.375× as useful as base to survive — unlikely in most contexts.
```

The expected result: Level 1 survives in most conversations (knowing what the partner believes is almost always useful). Level 2 survives in adversarial, pedagogical, or collaborative reasoning contexts (where mutual modeling of each other's knowledge matters). Level 3+ dies in most cases but may survive in sustained strategic interaction.

---

## Part V: The Observation Problem — How Partner Beliefs Get Updated

This is the hardest design challenge. The agent does not observe the partner's beliefs directly. It observes utterances and must infer beliefs from them. Three pathways are available, in order of architectural preference:

### Pathway 1: Prediction-Error Inference (Symbolic, Preferred)

The agent predicts what the partner will say, given the agent's model of the partner's beliefs. The mismatch between predicted and actual utterance updates the partner model.

```
Agent believes: pm-terrain-roughness = 0.3 (partner thinks terrain is smooth)
Agent predicts: if partner believes terrain is smooth, they'll respond calmly to a navigation plan
Partner responds: expresses concern about terrain difficulty
Prediction error: partner's response is inconsistent with pm-terrain-roughness = 0.3
Update: pm-terrain-roughness increases (partner thinks terrain is rougher than we assumed)
```

This requires **partner action models** — models of how the partner's beliefs relate to their observable behavior. These are analogous to the agent's own action models but applied to the partner:

```metta
;; Partner behavior model: if the partner believes X, they're likely to do Y
;; Schema: (partner-model $pm-obs $predicted-behavior-feature $sensitivity)
;;
;; "If the partner believes terrain is rough, they're more likely to
;; express concern (high urgency in their utterances)"
(partner-model pm-terrain-roughness utterance-urgency 0.6)

;; "If the partner believes threat is high, they're more likely to
;; ask about safety (high threat in their utterances)"
(partner-model pm-threat-level utterance-threat 0.7)
```

These partner action models are themselves subject to metabolic selection. They're hypotheses about how the partner's beliefs relate to their behavior. Correct predictions earn energy; incorrect predictions drain it.

**Key advantage:** This pathway is fully symbolic, traceable, and operates within the existing prediction-error framework. No new machinery.

**Key challenge:** The partner's utterances are parsed by the LLM into structured observations. The quality of partner-model inference is bounded by the quality of LLM perception. If the LLM can't distinguish "expressing concern" from "asking a question," the partner behavior models can't discriminate between those states.

### Pathway 2: Explicit Partner Statements (Direct Observation)

Sometimes the partner explicitly states their beliefs or goals: "I think the terrain is very rough" or "I want to know about safety." These can be parsed into direct partner-model observations:

```metta
;; When perception detects an explicit belief statement from the partner,
;; inject it as a high-precision partner-model observation.
(= (inject-partner-belief-statement! $obs $value $time)
   (inject-observation! pm-$obs $value 0.7 $time))
   ;; High precision (0.7) because the partner explicitly stated this
```

This pathway is easy to implement and produces high-quality observations. It doesn't cover cases where partner beliefs are implicit, but it handles the common case of explicit statements well.

### Pathway 3: LLM-Assisted Inference (Statistical, Constrained)

For cases where the partner's beliefs are implicit and the symbolic pathway lacks sufficient signal, the LLM at the perception boundary can generate partner-model observations. This is already happening implicitly — the LLM generates `partner-comprehension` estimates. The extension is to generate more specific estimates:

```
LLM prompt (at perception boundary):
"Given the partner's utterance and the conversation history,
estimate what the partner likely believes about: [list of active pm-X observables]"
```

**Critical constraint:** These LLM-derived partner-model observations must carry LOW precision, just as current LLM-derived observations do. The perception module already enforces this:

```metta
;; LLM-inferred partner beliefs get low precision
(perception-precision llm-parse pm-terrain-roughness 0.20)
(perception-precision llm-parse pm-threat-level 0.20)
(perception-precision llm-parse pm-goal-state 0.15)
;; Even lower than standard LLM perception (0.25-0.40)
;; because inferring partner beliefs is harder than surface classification
```

**Architectural concern:** This pathway pushes more cognitive work into the LLM, which conflicts with the "symbolic over statistical" and "minimal LLM" principles. It should be the fallback, not the primary mechanism. The goal is for Pathway 1 (prediction-error inference) to handle most cases, with Pathway 2 (explicit statements) providing calibration, and Pathway 3 (LLM inference) filling gaps at low precision.

### Recommended Priority

| Pathway | Priority | Precision | Principle Compliance |
|---------|----------|-----------|---------------------|
| 1. Prediction-error inference | Primary | Medium (varies) | Full — symbolic, traceable, emergent |
| 2. Explicit partner statements | Supplementary | High (0.7) | Full — direct observation |
| 3. LLM-assisted inference | Fallback | Low (0.15–0.20) | Partial — statistical, at boundary |

---

## Part VI: The Primitive Set Question

The current semantic primitive set (53 primitives, 12 contrast pairs) grounds the agent's understanding of its own concepts. Partner modeling introduces a question: **can the same primitives ground the partner's beliefs?**

The answer should be yes, because the primitives describe structural relationships (causes, enables, is-a, part-of) that are agent-independent. The grounding chain for `pm-terrain-roughness` is the same as for `terrain-roughness` — it's the same concept, just attributed to a different agent.

However, partner modeling may require primitives that the current set lacks:

| Missing Primitive | Needed For |
|-------------------|-----------|
| `intends` | Modeling partner goals and plans |
| `expects` | Modeling partner expectations |
| `attends-to` | Modeling partner attention/focus |
| `values` | Modeling partner preferences |
| `assumes` | Modeling partner presuppositions |

These are all *intentional* primitives — they describe mental states. The current primitive set is oriented toward physical and causal relationships. Extending it to cover intentional relationships is necessary for ToM.

**Design recommendation:** Add these as a new contrast cluster within the existing primitive framework, not as a separate system:

```metta
;; Intentional primitives (new cluster)
(primitive intends intentional)
(primitive expects intentional)
(primitive attends-to intentional)
(primitive values intentional)
(primitive assumes intentional)

;; Contrast pairs
(contrast intends fears)          ;; approach vs avoidance
(contrast expects surprised-by)   ;; predicted vs unpredicted
(contrast attends-to ignores)     ;; focused vs unfocused
(contrast values rejects)         ;; preferred vs dispreferred

;; Entailment
(entails intends expects)         ;; intending X entails expecting X (weakly)
(entails attends-to expects)      ;; attending to X entails expecting something about X
```

---

## Part VII: Open Research Questions

### Q1: How should partner-model observables be initialized?

Two options:

**Option A: Projection (assume partner is like self).** Initialize `pm-X` to match the agent's own belief about X. This is the "naive realism" default — assume others see the world as you do. It produces systematic errors (the "curse of knowledge") but provides a reasonable starting point.

**Option B: Uninformative prior.** Initialize `pm-X` to moderate value (0.5) with very low precision (0.05). This says "I don't know what they think." It's more honest but slower to converge.

**Recommendation:** Option A with low precision. The agent starts by assuming the partner's beliefs are similar to its own, but with very low confidence in that assumption. The projection provides a starting point; the low precision ensures it gets overwritten quickly by actual observation.

```metta
(= (init-partner-model! $obs)
   (let* (
     ($my-val (match &state (belief $obs $v $_) $v))
     ($init-prec 0.08)  ;; Very low — projection is a weak prior
   )
   (add-atom &state (belief pm-$obs $my-val $init-prec))))
```

### Q2: When should partner-model observables be created?

Creating `pm-X` for every domain observable is wasteful. The trigger should be conversational relevance:

- **On mention:** When the partner says something about X, create `pm-X` if it doesn't exist. The partner's utterance is direct evidence that X is in their model.
- **On assertion:** When the agent asserts about X, create `pm-X` to track whether the partner received the information.
- **On prediction error:** When a high prediction error on an existing `pm-X` observable suggests a related `pm-Y` should be tracked.

All three triggers are consistent with the existing architecture's "create on demand, prune metabolically" pattern.

### Q3: How do partner models interact with multi-party conversation?

The current design assumes a single conversational partner. Extension to multi-party conversation requires per-partner model spaces:

```metta
;; Per-partner model spaces
(belief pm-alice-terrain-roughness 0.3 0.2)
(belief pm-bob-terrain-roughness 0.7 0.1)
```

This scales linearly in the number of partners but combinatorially in the number of observables per partner. The metabolic system handles this: in a three-party conversation about terrain, the system might maintain `pm-alice-terrain-roughness` and `pm-bob-terrain-roughness` but let `pm-alice-threat-level` die if Alice hasn't discussed threats.

**Note:** Multi-party conversation introduces additional phenomena (common ground management across multiple partners, audience design, side conversations) that are beyond the scope of this document.

### Q4: How does partner model persistence work across conversations?

If the agent converses with the same partner again, should it retain the partner model from the previous conversation? This is a form of episodic memory for social relationships.

**Architectural fit:** The partner model is just beliefs. If the system has persistent memory across sessions (not currently implemented), partner-model beliefs persist with decayed precision (reflecting uncertainty about whether the partner's beliefs have changed since last contact).

**Concern:** Stale partner models can produce worse predictions than fresh uninformative priors. A partner who has learned new information between conversations will be systematically mismatch against the stale model. The precision decay must be aggressive enough to prevent overconfidence in outdated partner models.

### Q5: What is the relationship between partner modeling and abduction?

The abduction module already hypothesizes hidden causes from observed effects. A partner model is, in a sense, a systematic application of abduction: the partner's beliefs are hidden causes of their observed utterances.

**Potential unification:** Rather than building partner modeling as a separate system, it could be framed as a specialized abduction domain. The abduction module hypothesizes `pm-X` observables as hidden causes when partner utterances are surprising, and the metabolic system selects the hypotheses that predict correctly.

This unification is theoretically attractive (same machinery, same invariant) but practically challenging (abduction currently operates on domain observables, not on attributed mental states). It deserves investigation as a future extension.

### Q6: How do we validate this?

**Minimum viable tests (Tier 2 — Python specifications):**

| Test | What It Validates |
|------|------------------|
| Partner belief tracking | `pm-X` observables update correctly from simulated partner utterances |
| Targeted repair | Agent identifies *which* belief is misunderstood, not just *that* there's a misunderstanding |
| Reference resolution | Salience-based referent selection matches expected referent in test scenarios |
| Assertion informativeness | Agent avoids asserting what the partner already knows |
| Metabolic depth | Level 2 `pm2-X` observables survive in strategic scenarios, die in simple ones |
| Projection initialization | Agent's initial partner model (projected from self) converges to correct values within N turns |

**Stretch tests:**

| Test | What It Validates |
|------|------------------|
| Implicature detection | Agent detects that a literal assertion carries non-literal meaning |
| Goal inference | Agent correctly infers partner's conversational goal from indirect evidence |
| Common ground tracking | Agent adjusts communication strategy based on shared vs unshared knowledge |

---

## Part VIII: Implementation Roadmap

### Phase 1: Foundation (Minimal Partner Model)

**Goal:** Add Level 1 partner-model observables to the existing conversation model.

**Changes:**
- Add `pm-X` belief schema to `conversation_model.metta`
- Add conversational action models for partner beliefs (assert → pm-X, clarify → pm-X, etc.)
- Add explicit partner statement pathway to `perception.metta`
- Add partner model initialization function (projection with low precision)
- Derive `partner-comprehension` from aggregate `pm-X` prediction errors (keep original as fallback during transition)

**Validation:** Tier 2 Python spec proving partner belief tracking and targeted repair.

**Lines of code estimate:** ~200 lines of MeTTa (extensions to conversation_model.metta and perception.metta), ~150 lines of Python test spec.

### Phase 2: Prediction-Error Inference

**Goal:** Implement Pathway 1 — the agent predicts partner behavior from its partner model and updates the model from prediction errors.

**Changes:**
- Add partner action models (partner-model schema)
- Add prediction-error computation for partner behavior (predicted utterance features vs actual)
- Integrate partner-model update into cognitive cycle (step between belief update and structure learning)
- Add metabolic management for partner-model observables

**Validation:** Tier 2 Python spec proving prediction-error-driven partner model convergence.

**Lines of code estimate:** ~350 lines of MeTTa, ~200 lines of Python test spec.

### Phase 3: Pragmatic Capabilities

**Goal:** Demonstrate emergent reference resolution, implicature detection, and topic management.

**Changes:**
- Add salience computation for reference resolution
- Add assertion-informativeness check
- Add partner goal observables and interest tracking
- Add intentional primitives to semantic_primitives.metta

**Validation:** Tier 2 Python specs for reference resolution, assertion filtering, and goal inference.

**Lines of code estimate:** ~250 lines of MeTTa, ~300 lines of Python test spec.

### Phase 4: Recursive Depth

**Goal:** Enable metabolically controlled recursion to Level 2+.

**Changes:**
- Add `pm2-X` observable creation triggers (from Level 1 prediction errors)
- Add depth-scaled metabolic costs
- Add common ground tracking (agent's belief about partner's belief about agent's belief ≈ agent's actual belief)

**Validation:** Tier 2 Python spec proving Level 2 survives in strategic scenarios and dies in simple ones.

**Lines of code estimate:** ~200 lines of MeTTa, ~200 lines of Python test spec.

### Phase 5: Integration and Orchestrator Extension

**Goal:** Connect partner modeling to the full pipeline including LLM perception and verbalization.

**Changes:**
- Extend orchestrator perception prompts to generate `pm-X` observations
- Extend verbalization to account for partner model (don't say what they already know)
- Add Pathway 3 (LLM-assisted inference) as low-precision fallback
- End-to-end testing with the Tier 3 system

**Lines of code estimate:** ~100 lines of MeTTa, ~150 lines of Python orchestrator extensions.

---

## Part IX: Risks and Mitigations

### Risk 1: Combinatorial Explosion

**Problem:** N domain observables × K partners × L recursion levels = N×K×L partner-model observables.

**Mitigation:** Metabolic selection is the primary defense. Only observables that reduce prediction error survive. The system should never maintain more partner-model observables than it can metabolically support. The `pm-metabolic-cost` scaling with recursion depth provides additional pressure against deep, wide models.

**Monitoring:** Track the ratio of partner-model observables to domain observables. If it exceeds 3:1, the metabolic parameters may need tightening.

### Risk 2: LLM Perception Bottleneck

**Problem:** Partner-model inference quality is bounded by LLM perception quality. A 3B model's ability to infer partner beliefs from utterances is limited.

**Mitigation:** Design Pathway 1 (prediction-error inference) to be the primary mechanism, reducing dependence on LLM perception. Use Pathway 2 (explicit statements) for calibration. Reserve Pathway 3 (LLM inference) for gaps, with very low precision.

**Long-term:** As the LLM component is upgraded (or fine-tuned on conversational inference), Pathway 3 precision can increase. The architecture accommodates this through the existing perception-precision mechanism.

### Risk 3: Stale Partner Models

**Problem:** Partner-model beliefs that are accurate at time T may be wrong at time T+N if the partner has learned or changed.

**Mitigation:** Precision decay over conversational turns. Partner-model beliefs lose precision when not recently confirmed, which increases their info-gain and creates EFE pressure to re-verify them.

```metta
;; Precision decay for partner-model beliefs
;; Applied each cycle when the pm-X observable is not directly observed
(= (decay-pm-precision! $pm-obs)
   (match &state (belief $pm-obs $val $prec)
     (let $new-prec (* $prec (get-config pm-precision-decay-rate))
       (sequential
         (remove-atom &state (belief $pm-obs $val $prec))
         (add-atom &state (belief $pm-obs $val $new-prec))))))
```

### Risk 4: Over-Modeling Simple Conversations

**Problem:** A simple factual exchange ("What is the terrain roughness?" / "0.3") doesn't need Level 1 ToM, let alone Level 2.

**Mitigation:** The metabolic system handles this automatically. In simple conversations, partner-model observables don't reduce prediction error enough to justify their metabolic cost, and they die. The system only maintains as much partner model as the conversation requires.

### Risk 5: Principle Violations

**Problem:** Building an elaborate ToM module risks violating "bottom-up over top-down" and "emergent over scripted."

**Mitigation:** This document specifies *schema and dynamics*, not *behaviors*. The partner model uses the same belief/prediction-error/EFE machinery as everything else. No conversational behavior is scripted. Targeted repair, reference resolution, and implicature detection all emerge from the same minimization principle. If they don't emerge, the architecture has a gap — but the gap should be filled by adjusting the dynamics, not by adding special-case logic.

---

## Appendix A: Relationship to Existing Modules

| Module | How ToM Extension Interacts |
|--------|-----------------------------|
| `beliefs.metta` | pm-X observables use identical belief update machinery |
| `actions.metta` | pm-X observables enter EFE computation without modification |
| `conversation_model.metta` | Primary extension target; summary statistics become derived |
| `perception.metta` | Extended to inject pm-X observations from partner utterances |
| `self_model.metta` | Agent can introspect its own partner model ("what do I think they think?") |
| `structure_learning.metta` | May discover correlations between pm-X observables |
| `abduction.metta` | Partner beliefs as hidden causes is a natural abduction domain |
| `grounding_hypotheses.metta` | pm-X observables need grounding — what do partner beliefs *mean*? |
| `semantic_primitives.metta` | Extended with intentional primitives (intends, expects, etc.) |
| `atom_lifecycle.metta` | Metabolic selection governs pm-X observable survival |
| `safety.metta` | No constitutional changes. pm-X are learned stratum. |

## Appendix B: Literature Connections

This design draws on several established frameworks. Key references for future research:

- **Active Inference and social cognition:** Friston & Frith (2015) — "A Duet for One." Models social interaction as mutual Active Inference between agents. The partner model proposed here is a simplified version of their interacting generative models.
- **Recursive Bayesian ToM:** Baker, Saxe, & Tenenbaum (2011) — "Bayesian Theory of Mind." Hierarchical Bayesian models for inferring beliefs and desires from observed actions. The recursive depth mechanism proposed here follows this tradition.
- **Gricean pragmatics formalization:** Goodman & Frank (2016) — "Pragmatic Language Interpretation as Probabilistic Inference." The Rational Speech Act (RSA) framework formalizes implicature as recursive reasoning about speaker and listener models. The assertion-informativeness mechanism is a simplified EFE-based analog.
- **Relevance Theory:** Sperber & Wilson (1986). An alternative to Gricean maxims that frames communication as the search for optimal relevance — maximizing cognitive effect relative to processing effort. This maps naturally onto the EFE framework: info-gain is cognitive effect, and action cost is processing effort.
- **Common Ground:** Clark (1996) — "Using Language." Detailed analysis of how conversational partners build and maintain shared knowledge. The common ground tracking proposed for Phase 4 would benefit from Clark's taxonomy of grounding criteria.
- **Predictive Processing and social cognition:** Kube, Schmalbach, Radke et al. (2020). Reviews how predictive processing (the broader framework containing Active Inference) accounts for social cognitive phenomena. Provides theoretical grounding for the prediction-error partner model approach.
