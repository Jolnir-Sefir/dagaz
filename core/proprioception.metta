; =============================================================================
; PROPRIOCEPTION MODULE â€” Motor Control for Verbalization
; =============================================================================
;
; BIOLOGICAL ANALOGY:
;   In biological motor control, proprioception isn't a verification gate
;   that checks before an action commits. It's concurrent feedback. The
;   motor cortex issues a command AND predicts the sensory consequence.
;   Proprioceptors report the actual consequence. The prediction error
;   drives continuous correction â€” but the action is already in flight.
;
;   This module implements the same principle for language production:
;   The cognitive core selects an action and predicts what the output
;   should look like (the intent). The LLM generates text. A re-parse
;   of that text reports what actually came out. The difference is a
;   prediction error on verbalization-fidelity, fed back into the core.
;
; TWO LAYERS:
;   Layer 1 â€” Proprioceptive Observable (every turn, feeds core)
;     Re-parse generated text. Compare against intent. Inject fidelity
;     score as observation. The cognitive core adapts: if verbalization
;     is consistently unreliable, the system gravitates toward simpler
;     communicative actions (acknowledge vs elaborate). Emergent motor
;     adaptation, not a scripted retry loop.
;
;   Layer 2 â€” Assertion Gate (claim-bearing actions only, blocks)
;     For actions that make claims about the world (assert, elaborate),
;     check if re-parsed content contradicts beliefs. If so, block the
;     output, retry once with tighter prompt, then fall back to template.
;     This is the pain reflex â€” you pull your hand back before the
;     cortex processes what happened.
;
; DESIGN PRINCIPLES:
;   - The intent IS a prediction. The re-parse IS an observation.
;   - The difference IS prediction error â€” same as everything else.
;   - Fidelity comparison is STRUCTURAL (field-level), not semantic.
;     "Is this in the right ballpark?" not "Did it say exactly what I meant?"
;   - Coarse checks are robust to parser noise. Fine checks amplify it.
;
; DEPENDENCIES: foundations, beliefs, cycle, conversation_model, perception,
;               safety (assertion-action?)
; DEPENDED ON BY: orchestrator.py (Python-side re-parse and comparison)
;
; =============================================================================

!(import! &self foundations)


; =============================================================================
; SECTION I: TYPES
; =============================================================================

(: ProprioceptiveObservable Type)
(: FidelityDimension Type)

(: action-type-fidelity FidelityDimension)   ; Did the speech act match?
(: topic-fidelity FidelityDimension)         ; Did we address the right topics?
(: tone-fidelity FidelityDimension)          ; Was the affect appropriate?


; =============================================================================
; SECTION II: VERBALIZATION FIDELITY OBSERVABLE
; =============================================================================
;
; This is the core proprioceptive signal. It measures how well the
; LLM's output matched the cognitive core's intent.
;
; High fidelity (near 1.0) = "my words match my intentions"
; Low fidelity (near 0.0) = "I'm saying things I didn't intend"
;
; This feeds into EFE like any other observable. Over time:
;   - Persistently low fidelity â†’ high arousal, low dominance
;   - System shifts toward simpler actions (lower verbalization risk)
;   - This is emergent motor adaptation

(: verbalization-fidelity ProprioceptiveObservable)

; Belief: prior = moderate (assume verbalization works okay)
(belief verbalization-fidelity 0.7 0.4)

; Preference: we want HIGH fidelity (say what we mean)
(preference verbalization-fidelity 0.9 0.7)

; Viability: if we consistently can't express ourselves, conversation fails
(viability-bound verbalization-fidelity 0.25 1.0)

; Perception precision for the proprioceptive signal
; This is higher than most LLM-derived observables because we're
; comparing structured fields, not inferring mental states.
; But it's still LLM-derived (the re-parse is an LLM call).
(perception-precision proprioception verbalization-fidelity 0.50)


; =============================================================================
; SECTION III: ACTION MODELS â€” FIDELITY EFFECTS
; =============================================================================
;
; Some actions are harder to verbalize than others. A 3B model can
; nail "I understand" (acknowledge) but might botch a nuanced
; explanation (elaborate). The action models encode this.
;
; Positive value_delta = this action IMPROVES fidelity (easy to verbalize).
; Negative value_delta = this action RISKS fidelity (hard to verbalize).
;
; These are initial priors. Action learning refines them from experience.

; Easy to verbalize â€” simple speech acts
(action-model acknowledge verbalization-fidelity 0.05 0.02)
(action-model confirm verbalization-fidelity 0.03 0.02)
(action-model conclude verbalization-fidelity 0.02 0.02)

; Moderate â€” structured but focused
(action-model clarify verbalization-fidelity 0.0 0.03)
(action-model query-partner verbalization-fidelity 0.02 0.03)
(action-model request-clarification verbalization-fidelity 0.02 0.02)

; Hard to verbalize â€” complex content, risk of drift
(action-model elaborate verbalization-fidelity -0.05 0.04)
(action-model assert verbalization-fidelity -0.08 0.05)
(action-model redirect verbalization-fidelity -0.03 0.03)


; =============================================================================
; SECTION IV: INTENT PREDICTION
; =============================================================================
;
; Before verbalization, the system predicts what the output should
; look like. This prediction is compared against the re-parse.
;
; The prediction is structural: expected speech act type, expected
; topics, expected tone quadrant. NOT a predicted sentence.

; Map from intended action to expected speech act in re-parse
; These define what the re-parse of a successful verbalization should show.
(expected-speech-act clarify explanation)
(expected-speech-act elaborate explanation)
(expected-speech-act assert statement)
(expected-speech-act confirm checking)
(expected-speech-act query-partner question)
(expected-speech-act request-clarification question)
(expected-speech-act acknowledge brief-response)
(expected-speech-act redirect statement)
(expected-speech-act conclude closing)

; Map from re-parse TYPE field to speech act category
; Multiple utterance types can map to the same speech act
(type-to-speech-act assertion statement)
(type-to-speech-act assertion explanation)   ; Assertions explain things
(type-to-speech-act query question)
(type-to-speech-act query checking)          ; Questions check things
(type-to-speech-act greeting brief-response)
(type-to-speech-act greeting closing)        ; Greetings are brief
(type-to-speech-act farewell closing)
(type-to-speech-act emotional brief-response)

; Does the re-parsed type match the expected speech act?
(= (action-type-matches? $intended-action $reparsed-type)
   (let* (
     ($expected (match &self (expected-speech-act $intended-action $sa) $sa))
     ($actual-acts (collapse (match &self (type-to-speech-act $reparsed-type $sa) $sa)))
   )
   (if (== $expected ())
       True              ; Unknown action: give benefit of doubt
       (list-contains? $expected $actual-acts))))


; =============================================================================
; SECTION V: FIDELITY COMPUTATION
; =============================================================================
;
; Compute verbalization fidelity from three structural dimensions.
; Each dimension produces a 0-1 score. They combine as a weighted average.
;
; The computation happens in the Python orchestrator (it has the intent
; and re-parse dicts). This section defines the MeTTa-side weighting
; and the observation injection.

; Dimension weights (how much each dimension matters for overall fidelity)
(fidelity-weight action-type-fidelity 0.45)    ; Most important: right speech act
(fidelity-weight topic-fidelity 0.30)          ; Did we address the right things
(fidelity-weight tone-fidelity 0.25)           ; Was the affect appropriate

; Get weight for a fidelity dimension
(= (get-fidelity-weight $dim)
   (match &self (fidelity-weight $dim $w) $w))

; Inject a proprioceptive observation (called by orchestrator)
; Takes the composite fidelity score computed in Python
(= (perceive-proprioceptive! $fidelity-score $time)
   (let $prec (get-perception-precision proprioception verbalization-fidelity)
     (inject-observation! verbalization-fidelity $fidelity-score $prec $time)))


; =============================================================================
; SECTION VI: ASSERTION GATE (LAYER 2)
; =============================================================================
;
; For claim-bearing actions, low fidelity isn't just a learning signal â€”
; it's a safety concern. The LLM might have generated text that asserts
; something the system doesn't believe. This gate blocks such output.
;
; Only activates for assertive actions (assert, elaborate).
; Only blocks when fidelity is below the gate threshold.
; Maximum one retry, then fallback to template.
;
; This extends safety.metta's assertion-consistency check from structural
; content to the actual generated NL text.

; Threshold below which an assertive verbalization is blocked
(config assertion-gate-threshold 0.4)

; Is this action claim-bearing? (extends safety.metta's assertion-action?)
; Elaborate makes implicit claims too, not just explicit assertions.
(= (claim-bearing-action? assert) True)
(= (claim-bearing-action? elaborate) True)
(= (claim-bearing-action? $other) False)

; Should the assertion gate block this output?
; Returns: (gate-pass) or (gate-block $reason)
(= (assertion-gate-check $action $fidelity-score)
   (if (not (claim-bearing-action? $action))
       gate-pass                         ; Non-assertive: always pass
       (let $threshold (get-config assertion-gate-threshold)
         (if (>= $fidelity-score $threshold)
             gate-pass                   ; Fidelity acceptable
             (gate-block
               (action $action)
               (fidelity $fidelity-score)
               (threshold $threshold)
               (reason "verbalization fidelity below assertion safety threshold"))))))


; =============================================================================
; SECTION VII: RETRY PROMPT CONSTRUCTION
; =============================================================================
;
; When the assertion gate blocks, the orchestrator needs to retry with
; a tighter prompt. This section provides the MeTTa-side context for
; building that prompt: what specifically went wrong.
;
; The actual prompt construction happens in Python (orchestrator).
; This module provides the structured correction hints.

; What specifically was wrong with the verbalization?
; Called when gate blocks to produce correction context.
(= (diagnose-fidelity-failure $action $fidelity-dims)
   (let* (
     ($action-ok (field-or-default $fidelity-dims action-type-fidelity 0.5))
     ($topic-ok (field-or-default $fidelity-dims topic-fidelity 0.5))
     ($tone-ok (field-or-default $fidelity-dims tone-fidelity 0.5))
   )
   (fidelity-diagnosis
     (if (< $action-ok 0.5)
         (wrong-speech-act $action)
         ok)
     (if (< $topic-ok 0.5)
         (wrong-topics)
         ok)
     (if (< $tone-ok 0.5)
         (wrong-tone)
         ok))))


; =============================================================================
; SECTION VIII: PROPRIOCEPTIVE LOG
; =============================================================================
;
; Record proprioceptive feedback for diagnostics and potential fine-tuning.
; Each entry: what was intended, what was produced, how well they matched.

(= (record-proprioceptive! $action $fidelity $gate-result $time)
   (add-atom &self (proprioceptive-log $action $fidelity $gate-result $time)))

; Clean old proprioceptive logs
(= (clean-proprioceptive-logs! $current-time)
   (let* (
     ($window (get-config lookback-window))
     ($cutoff (- $current-time (* $window 5)))   ; Keep 5x lookback (longer for fine-tuning data)
     ($old (collapse (match &self (proprioceptive-log $a $f $g $time)
       (if (< $time $cutoff) (proprioceptive-log $a $f $g $time) ()))))
   )
   (remove-old-perceptions! $old)))    ; Reuse cleanup helper from perception.metta


; =============================================================================
; SECTION IX: DIAGNOSTICS
; =============================================================================

; Current verbalization fidelity belief
(= (current-verbalization-fidelity)
   (match &state (belief verbalization-fidelity $val $prec)
     (fidelity-state
       (value $val)
       (precision $prec)
       (interpretation
         (if (> $val 0.8) "expressing myself well"
           (if (> $val 0.5) "mostly saying what I mean"
             (if (> $val 0.3) "struggling to express my intentions"
               "severe disconnect between intent and output")))))))

; Recent proprioceptive history
(= (recent-proprioception)
   (collapse (match &self (proprioceptive-log $action $fidelity $gate $time)
     (proprio-entry $time $action (fidelity $fidelity) $gate))))

; Average fidelity over recent history (for self-model integration)
(= (average-recent-fidelity)
   (let* (
     ($entries (collapse (match &self (proprioceptive-log $a $f $g $t) $f)))
     ($n (length $entries))
   )
   (if (== $n 0)
       0.7    ; Prior: assume okay when no data
       (/ (sum-list $entries) $n))))

; How often has the assertion gate blocked recently?
(= (assertion-gate-block-rate)
   (let* (
     ($all (collapse (match &self (proprioceptive-log $a $f $g $t) $g)))
     ($n (length $all))
     ($blocks (list-count (lambda $g (not (== $g gate-pass))) $all))
   )
   (if (== $n 0)
       0.0
       (/ $blocks $n))))


; =============================================================================
; SECTION X: SEMANTIC GROUNDING
; =============================================================================
;
; Ground verbalization-fidelity in the semantic primitive system.
; The system can understand what fidelity IS, not just track its value.

(obs-grounding verbalization-fidelity is-a property)
(obs-grounding verbalization-fidelity is-a quality)
(obs-grounding verbalization-fidelity belongs-to self)
(obs-grounding verbalization-fidelity is-a (relation (same intent output)))
(obs-semantic  verbalization-fidelity "how well my words match my intentions")

(obs-low-means  verbalization-fidelity (not (same intent output)))
(obs-low-means  verbalization-fidelity (frustrate want))
(obs-low-means  verbalization-fidelity (associated fear))    ; Can't control what I say
(obs-high-means verbalization-fidelity (same intent output))
(obs-high-means verbalization-fidelity (associated good))
(obs-high-means verbalization-fidelity (enables do))         ; Can act through language


; =============================================================================
; END OF PROPRIOCEPTION MODULE
; =============================================================================
