; =============================================================================
; CONVERSATION MODEL (v2) - Active Inference Based
; =============================================================================
;
; DESIGN PRINCIPLES APPLIED:
;   - Conversation is mutual prediction between agents
;   - NO enumerated states - state emerges from observables
;   - Conversational actions selected via EFE like any other
;   - Partner model is a belief structure, updated through error
;   - Understanding = accurate prediction of partner's behavior
;
; KEY INSIGHT: A conversation IS an Active Inference process.
; Each party predicts the other; errors drive adaptation.
; "Understanding" means your model of them predicts their responses.
;
; =============================================================================


; =============================================================================
; SECTION I: CONVERSATIONAL OBSERVABLES
; =============================================================================

; These feed directly into the cognitive cycle.
; High error on these Ã¢â€ â€™ poor conversation Ã¢â€ â€™ adaptive action

(: ConversationalObservable Type)

; Did partner respond as predicted?
(: partner-predictability ConversationalObservable)

; Does partner seem to understand us?
(: partner-comprehension ConversationalObservable)

; Are we staying coherent/on-topic?
(: discourse-coherence ConversationalObservable)

; Is the exchange progressing toward goals?
(: goal-progress ConversationalObservable)

; Social/rapport dimension
(: rapport ConversationalObservable)


; =============================================================================
; SECTION II: CONVERSATIONAL BELIEFS
; =============================================================================

; Beliefs about the conversation state
; These are UPDATED through observation, not declared
; BF: precision starts near-zero. The agent starts knowing nothing about
; these dimensions. Precision ratchets up from observation exposure.

(belief partner-predictability 0.5 0.01)   ; Prior: moderate predictability
(belief partner-comprehension 0.6 0.01)    ; Prior: probably understand
(belief discourse-coherence 0.7 0.01)      ; Prior: usually coherent
(belief goal-progress 0.3 0.01)            ; Prior: early in exchange
(belief rapport 0.5 0.01)                  ; Prior: neutral rapport


; =============================================================================
; SECTION III: CONVERSATIONAL PREFERENCES
; =============================================================================

; What makes a good conversation? These drive EFE.

(preference partner-predictability 0.7 0.6)   ; Want predictable exchanges
(preference partner-comprehension 0.9 1.0)    ; CRITICAL: they must understand
(preference discourse-coherence 0.8 0.8)      ; Want coherent discourse
(preference goal-progress 0.8 0.7)            ; Want to make progress
(preference rapport 0.7 0.5)                  ; Nice to have good rapport


; =============================================================================
; SECTION IV: CONVERSATIONAL VIABILITY
; =============================================================================

; Hard constraints - violate these and conversation fails

(viability-bound partner-comprehension 0.2 1.0)  ; If they don't understand, abort
(viability-bound discourse-coherence 0.15 1.0)   ; Incoherence kills exchange
(viability-bound rapport 0.1 1.0)                ; Can't continue if hostile


; =============================================================================
; SECTION V: CONVERSATIONAL ACTIONS
; =============================================================================

; Actions available during conversation
; These compete via EFE with domain actions

(: ConversationalAction Type)

(: assert ConversationalAction)           ; Make a claim
(: query-partner ConversationalAction)    ; Ask partner something
(: clarify ConversationalAction)          ; Clarify previous utterance
(: confirm ConversationalAction)          ; Confirm understanding
(: request-clarification ConversationalAction)  ; Ask partner to clarify
(: acknowledge ConversationalAction)      ; Simple acknowledgment
(: elaborate ConversationalAction)        ; Expand on previous point
(: redirect ConversationalAction)         ; Change topic
(: conclude ConversationalAction)         ; Signal conversation end


; =============================================================================
; SECTION VI: ACTION MODELS FOR CONVERSATION
; =============================================================================
;
; Schema: (action-model <action> <observable> <value-delta> <prec-delta> <confidence>)
;
; IMPORTANT: Must use 5-field schema to match actions.metta v4 accessors.
; Without the confidence field, get-model-confidence defaults to 0.1 (the
; "unknown model" fallback), which cripples the effective deltas and makes
; all conversation actions look equally impotent.
;
; value-delta models the action's effect on the WORLD (observable changes).
; In predict-single-error (v4.1), this delta is applied to the predicted
; future observation, not to the belief.

; How do conversational actions affect conversational observables?

; assert: might help progress, risks comprehension if complex
(action-model assert partner-comprehension -0.05 0.02 0.3)
(action-model assert goal-progress 0.1 0.05 0.3)
(action-model assert discourse-coherence 0.02 0.03 0.3)

; query-partner: high info gain, tests predictability
(action-model query-partner partner-predictability 0.0 0.1 0.3)
(action-model query-partner goal-progress 0.05 0.04 0.3)
(action-model query-partner rapport 0.02 0.02 0.3)

; clarify: improves comprehension, may slow progress
(action-model clarify partner-comprehension 0.15 0.08 0.3)
(action-model clarify goal-progress -0.02 0.02 0.3)
(action-model clarify discourse-coherence 0.05 0.04 0.3)

; confirm: verifies comprehension, builds rapport
(action-model confirm partner-comprehension 0.05 0.06 0.3)
(action-model confirm rapport 0.05 0.03 0.3)
(action-model confirm partner-predictability 0.05 0.05 0.3)

; request-clarification: reveals own confusion, high info gain
(action-model request-clarification partner-predictability 0.0 0.12 0.3)
(action-model request-clarification discourse-coherence 0.1 0.06 0.3)
(action-model request-clarification goal-progress -0.03 0.02 0.3)

; acknowledge: minimal change, maintains rapport
(action-model acknowledge rapport 0.03 0.02 0.3)
(action-model acknowledge partner-comprehension 0.02 0.01 0.3)
(action-model acknowledge goal-progress 0.01 0.01 0.3)

; elaborate: deepens understanding, risks overload
(action-model elaborate partner-comprehension 0.08 0.05 0.3)
(action-model elaborate discourse-coherence -0.03 0.03 0.3)
(action-model elaborate goal-progress 0.05 0.04 0.3)

; redirect: disrupts coherence, enables new direction
(action-model redirect discourse-coherence -0.15 0.05 0.3)
(action-model redirect goal-progress -0.1 0.08 0.3)
(action-model redirect partner-predictability -0.1 0.06 0.3)

; conclude: signals end, confirms mutual state
(action-model conclude goal-progress 0.2 0.1 0.3)
(action-model conclude discourse-coherence 0.1 0.05 0.3)


; =============================================================================
; SECTION VII: ACTION COSTS
; =============================================================================
; Schema: (action-cost <action> <cost> <confidence>)
; Must use 3-field schema to match actions.metta v4 get-action-cost accessor.
;
; CALIBRATION NOTE: Conversational actions have similar costs because they
; are all verbal acts taking comparable effort/time. Large cost differentials
; (e.g. acknowledge=0.01 vs clarify=0.04) cause the EFE to always select
; the cheapest action when observation precision is low (LLM parser â‰ˆ 0.30),
; because precision-weighted error differences between actions are smaller
; than cost differences. Costs here are within a 2Ã— range.

(action-cost assert 0.02 0.5)
(action-cost query-partner 0.02 0.5)
(action-cost clarify 0.02 0.5)
(action-cost confirm 0.02 0.5)
(action-cost request-clarification 0.02 0.5)
(action-cost acknowledge 0.02 0.5)
(action-cost elaborate 0.03 0.5)
(action-cost redirect 0.03 0.5)
(action-cost conclude 0.02 0.5)


; =============================================================================
; SECTION VIII: PARTNER MODEL
; =============================================================================

; Our beliefs about the conversation partner
; This IS the model that generates predictions

(: Partner Entity)
(isa Partner Agent)

; Belief about partner's knowledge (what they know)
; Schema: (partner-belief <topic> <estimated-knowledge> <our-confidence>)
(= (partner-knows-about? $topic)
   (case (match &state (partner-belief $topic $knowledge $conf) ($knowledge $conf))
     ((($k $c) (> $k 0.4))  ; They probably know about it
      (() False))))

; Belief about partner's goals
; Schema: (partner-goal-belief <goal-type> <confidence>)
(partner-goal-belief information-seeking 0.5)
(partner-goal-belief task-completion 0.3)
(partner-goal-belief exploration 0.2)

; Update partner model from their utterance
(= (update-partner-model! $utterance-type $topics $time)
   (sequential
     (update-partner-knowledge! $topics)
     (update-partner-goals! $utterance-type)
     (partner-model-updated $time)))

(= (update-partner-knowledge! ()) done)
(= (update-partner-knowledge! ($topic . $rest))
   (sequential
     ; If they mention it, they probably know about it
     (let $old (match &state (partner-belief $topic $k $c) ($k $c))
       (if (== $old ())
           (add-atom &state (partner-belief $topic 0.6 0.4))
           (let ($k $c) $old
             (sequential
               (remove-atom &state (partner-belief $topic $k $c))
               (add-atom &state (partner-belief $topic (min 0.95 (+ $k 0.1)) (min 0.9 (+ $c 0.05))))))))
     (update-partner-knowledge! $rest)))

(= (update-partner-goals! query)
   (boost-partner-goal! information-seeking 0.1))
(= (update-partner-goals! request)
   (boost-partner-goal! task-completion 0.15))
(= (update-partner-goals! assertion)
   (boost-partner-goal! exploration 0.05))
(= (update-partner-goals! $_) done)

(= (boost-partner-goal! $goal $amount)
   (match &state (partner-goal-belief $goal $conf)
     (sequential
       (remove-atom &state (partner-goal-belief $goal $conf))
       (add-atom &state (partner-goal-belief $goal (min 0.95 (+ $conf $amount)))))))


; =============================================================================
; SECTION IX: UTTERANCE PROCESSING
; =============================================================================

; Process incoming utterance from partner
; Returns observations that feed into cognitive cycle

(= (process-utterance! $utterance-type $content $topics $time)
   (let* (
     ; Predict what we expected vs what we got
     ($prediction-error (compute-utterance-surprise $utterance-type))
     
     ; Update partner model
     ($update (update-partner-model! $utterance-type $topics $time))
     
     ; Generate observations
     ($obs1 (inject-conv-observation! partner-predictability 
              (- 1.0 $prediction-error) 0.7 $time))
     ($obs2 (inject-conv-observation! discourse-coherence
              (compute-coherence $topics) 0.6 $time))
   )
   (utterance-processed $time)))

; How surprised are we by this utterance type?
(= (compute-utterance-surprise $type)
   (match &state (partner-goal-belief $goal $conf)
     (if (matches-goal? $type $goal)
         (- 1.0 $conf)  ; Low surprise if matches expected goal
         0.5)))         ; Moderate surprise otherwise

(= (matches-goal? query information-seeking) True)
(= (matches-goal? request task-completion) True)
(= (matches-goal? assertion exploration) True)
(= (matches-goal? $_ $_) False)

; How coherent is this with current discourse?
(= (compute-coherence $topics)
   (let $current (current-topics)
     (if (== $current ())
         0.7  ; Nothing established yet, assume okay
         (topic-overlap $topics $current))))

(= (topic-overlap () $_) 0.5)
(= (topic-overlap ($t . $rest) $current)
   (if (member? $t $current)
       (+ 0.3 (* 0.7 (topic-overlap $rest $current)))
       (* 0.8 (topic-overlap $rest $current))))

(= (member? $x ()) False)
(= (member? $x ($x . $_)) True)
(= (member? $x ($_ . $rest)) (member? $x $rest))

; Inject conversational observation
(= (inject-conv-observation! $obs $val $prec $time)
   (sequential
     (clear-conv-observation! $obs)
     (add-atom &state (observation $obs $val $prec $time))
     observed))

(= (clear-conv-observation! $obs)
   (let $old (collapse (match &state (observation $obs $v $p $t) (observation $obs $v $p $t)))
     (remove-list $old)))

(= (remove-list ()) done)
(= (remove-list ($a . $rest)) 
   (sequential (remove-atom &state $a) (remove-list $rest)))


; =============================================================================
; SECTION X: TOPIC TRACKING
; =============================================================================

; Current topics under discussion
(current-topics ())

(= (add-topic! $topic)
   (match &state (current-topics $topics)
     (sequential
       (remove-atom &state (current-topics $topics))
       (add-atom &state (current-topics ($topic . $topics))))))

(= (current-topics)
   (match &state (current-topics $t) $t))

(= (clear-topics!)
   (match &state (current-topics $t)
     (sequential
       (remove-atom &state (current-topics $t))
       (add-atom &state (current-topics ())))))


; =============================================================================
; SECTION XI: TURN MANAGEMENT
; =============================================================================

(conversation-turn 0)

(= (current-turn)
   (match &state (conversation-turn $n) $n))

(= (advance-turn!)
   (let $old (current-turn)
     (sequential
       (remove-atom &state (conversation-turn $old))
       (add-atom &state (conversation-turn (+ $old 1)))
       (+ $old 1))))


; =============================================================================
; SECTION XII: CONVERSATIONAL STATE (EMERGENT)
; =============================================================================

; NO enumerated states. Instead, state is characterized by
; the current values of conversational observables.

; Describe current conversation state (computed, not stored)
(= (conversation-state-description)
   (let* (
     ($comp (match &state (belief partner-comprehension $v $_) $v))
     ($coh (match &state (belief discourse-coherence $v $_) $v))
     ($prog (match &state (belief goal-progress $v $_) $v))
     ($rap (match &state (belief rapport $v $_) $v))
   )
   (conv-state
     (comprehension $comp)
     (coherence $coh)
     (progress $prog)
     (rapport $rap))))

; Is conversation going well?
; DERIVED from viability bounds - healthy means comfortably above bounds
(= (conversation-healthy?)
   (let* (
     ($comp (match &state (belief partner-comprehension $v $_) $v))
     ($coh (match &state (belief discourse-coherence $v $_) $v))
     ($rap (match &state (belief rapport $v $_) $v))
     ; Get viability bounds
     ($comp-bound (match &state (viability-bound partner-comprehension $min $_) $min))
     ($coh-bound (match &state (viability-bound discourse-coherence $min $_) $min))
     ($rap-bound (match &state (viability-bound rapport $min $_) $min))
     ; Healthy = at least 2x the viability bound (comfortable margin)
   )
   (and (> $comp (* 2.0 $comp-bound)) 
        (and (> $coh (* 2.0 $coh-bound)) 
             (> $rap (* 2.0 $rap-bound))))))

; Is conversation in trouble?
; DERIVED from viability bounds - troubled means approaching bounds
(= (conversation-troubled?)
   (let* (
     ($comp (match &state (belief partner-comprehension $v $_) $v))
     ($coh (match &state (belief discourse-coherence $v $_) $v))
     ; Get viability bounds
     ($comp-bound (match &state (viability-bound partner-comprehension $min $_) $min))
     ($coh-bound (match &state (viability-bound discourse-coherence $min $_) $min))
     ; Troubled = within 1.5x of viability bound (danger zone)
   )
   (or (< $comp (* 1.5 $comp-bound)) 
       (< $coh (* 1.5 $coh-bound)))))

; Diagnostic: What aspect has largest gap from preference?
; (For introspection only - EFE handles action selection)
(= (most-problematic-aspect)
   (let* (
     ($comp-val (match &state (belief partner-comprehension $v $_) $v))
     ($comp-pref (match &state (preference partner-comprehension $p $_) $p))
     ($coh-val (match &state (belief discourse-coherence $v $_) $v))
     ($coh-pref (match &state (preference discourse-coherence $p $_) $p))
     ($prog-val (match &state (belief goal-progress $v $_) $v))
     ($prog-pref (match &state (preference goal-progress $p $_) $p))
     ($comp-gap (- $comp-pref $comp-val))
     ($coh-gap (- $coh-pref $coh-val))
     ($prog-gap (- $prog-pref $prog-val))
   )
   (if (and (> $comp-gap $coh-gap) (> $comp-gap $prog-gap))
       (diagnostic partner-comprehension $comp-gap)
       (if (> $coh-gap $prog-gap)
           (diagnostic discourse-coherence $coh-gap)
           (diagnostic goal-progress $prog-gap)))))


; =============================================================================
; SECTION XIII: ACTION SELECTION (Via EFE, Not Scripted)
; =============================================================================

; NOTE: Action selection is handled by the core EFE machinery.
; 
; DO NOT add "if troubled, do clarify" logic here. That's scripted behavior.
; 
; Instead, the action models capture this via predict-single-error (v4.1):
;   (action-model clarify partner-comprehension 0.15 0.08 0.3)
; 
; When partner-comprehension observation is low:
;   - clarify's value-delta (+0.15) predicts a HIGHER future observation
;   - predict-single-error computes |predicted_future_obs - current_belief|
;   - If belief is already higher than obs, clarify moves predicted obs
;     TOWARD the belief, REDUCING predicted error
;   - EFE(clarify) becomes lower than alternatives
;   - clarify is selected AUTOMATICALLY
;
; This is emergence, not scripting.

; Diagnostic only: what aspect has largest preference gap?
; (Use for introspection/logging, NOT action selection)
(= (largest-preference-gap)
   (let* (
     ($comp-val (match &state (belief partner-comprehension $v $_) $v))
     ($comp-pref (match &state (preference partner-comprehension $p $_) $p))
     ($coh-val (match &state (belief discourse-coherence $v $_) $v))
     ($coh-pref (match &state (preference discourse-coherence $p $_) $p))
     ($prog-val (match &state (belief goal-progress $v $_) $v))
     ($prog-pref (match &state (preference goal-progress $p $_) $p))
     ($comp-gap (- $comp-pref $comp-val))
     ($coh-gap (- $coh-pref $coh-val))
     ($prog-gap (- $prog-pref $prog-val))
   )
   (if (and (> $comp-gap $coh-gap) (> $comp-gap $prog-gap))
       (gap partner-comprehension $comp-gap)
       (if (> $coh-gap $prog-gap)
           (gap discourse-coherence $coh-gap)
           (gap goal-progress $prog-gap)))))


; =============================================================================
; SECTION XIV: INTEGRATION WITH CORE CYCLE
; =============================================================================

; Extend base-actions to include conversational actions when in conversation
(= (conversational-actions)
   (assert query-partner clarify confirm request-clarification 
    acknowledge elaborate redirect conclude))

; Check if we're in a conversation
(= (in-conversation?)
   (> (current-turn) 0))

; Full action set when in conversation
(= (available-actions)
   (if (in-conversation?)
       (concat (base-actions) (conversational-actions))
       (base-actions)))


; =============================================================================
; SECTION XV: CONVERSATION LIFECYCLE
; =============================================================================

; Start conversation
(= (begin-conversation!)
   (sequential
     (clear-topics!)
     (reset-turn-counter!)
     (reset-conversational-beliefs!)
     (conversation-begun)))

(= (reset-turn-counter!)
   (match &state (conversation-turn $n)
     (sequential
       (remove-atom &state (conversation-turn $n))
       (add-atom &state (conversation-turn 0)))))

(= (reset-conversational-beliefs!)
   (sequential
     (set-belief! partner-predictability 0.5 0.5)
     (set-belief! partner-comprehension 0.6 0.5)
     (set-belief! discourse-coherence 0.7 0.5)
     (set-belief! goal-progress 0.3 0.4)
     (set-belief! rapport 0.5 0.5)
     beliefs-reset))

; End conversation
(= (end-conversation!)
   (sequential
     (clear-topics!)
     (conversation-ended (current-turn))))


; =============================================================================
; END OF CONVERSATION MODEL (v2)
; =============================================================================
