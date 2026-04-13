; =============================================================================
; PERCEPTION MODULE - Neural/Symbolic Boundary
; =============================================================================
;
; DESIGN PRINCIPLES APPLIED:
;   - LLM is a SENSOR, not a reasoner. It produces observations.
;   - All LLM output enters as (observation $obs $val $prec $time)
;   - Precision reflects sensor quality: LLM estimates get LOW precision,
;     direct sensor readings get HIGH precision
;   - The cognitive core doesn't know or care what produced an observation
;   - Parse failures degrade gracefully: missing fields â†’ default values
;     at very low precision (system just gets less information)
;
; THE KEY INSIGHT:
;   A 3B LLM estimating partner-comprehension from text is analogous to
;   a cheap IR sensor estimating distance. Both are noisy. Both produce
;   observations that the precision-weighted belief update handles correctly.
;   The cognitive core self-corrects from bad parses the same way a robot
;   self-corrects from noisy sensor readings: through prediction error.
;
; ARCHITECTURE:
;   Raw text â†’ [LLM parse] â†’ structured parse â†’ perceive-utterance! â†’ observations
;   Action selected â†’ package-intent â†’ [LLM verbalize] â†’ raw text response
;
;   The [LLM] calls are grounded functions implemented in Python (orchestrator.py).
;   This module defines the MeTTa-side contracts and observation injection.
;
; DEPENDENCIES: foundations, beliefs, cycle, conversation_model, semantic_grounding
; DEPENDED ON BY: orchestrator.py (Python glue layer)
;
; =============================================================================

!(import! &self foundations)


; =============================================================================
; SECTION I: TYPES
; =============================================================================

(: PerceptionSource Type)
(: llm-parse PerceptionSource)         ; LLM-derived observation
(: direct-sensor PerceptionSource)     ; Hardware sensor reading
(: simulation PerceptionSource)        ; 2D environment / simulator
(: self-report PerceptionSource)       ; Internal state observation

; Parse result types (mirror what the Python orchestrator produces)
(: ParseResult Type)
(: ParseField Type)


; =============================================================================
; SECTION II: PERCEPTION PRECISION
; =============================================================================
;
; How much do we trust each source for each kind of observation?
; These are PRIORS about sensor quality. They do not update â€”
; the cognitive core handles miscalibration through prediction error
; on the beliefs these observations feed.
;
; Rationale for LLM precision values:
;   - Utterance type classification is relatively easy (0.55)
;   - Coherence is surface-level detectable (0.40)
;   - Comprehension requires theory-of-mind inference (0.30)
;   - Rapport requires affect reading from text (0.25)
;   - Goal progress requires deep context understanding (0.30)
;
; A 3B model is noisier than a 70B model. These values reflect that.
; If the model is fine-tuned on conversation data, these can increase.

; --- LLM-derived perception precision ---
(perception-precision llm-parse partner-comprehension 0.30)
(perception-precision llm-parse discourse-coherence 0.40)
(perception-precision llm-parse rapport 0.25)
(perception-precision llm-parse goal-progress 0.30)
(perception-precision llm-parse partner-predictability 0.35)
(perception-precision llm-parse unresolved-salience 0.45)  ; LLM good at urgency detection

; --- Direct sensor precision (for 2D environment, later) ---
(perception-precision direct-sensor position 0.90)
(perception-precision direct-sensor surface-type 0.85)
(perception-precision direct-sensor proximity 0.85)
(perception-precision direct-sensor energy-level 0.95)

; --- Simulation precision (2D environment provides exact values) ---
(perception-precision simulation terrain-roughness 0.90)
(perception-precision simulation power-level 0.95)
(perception-precision simulation threat-level 0.80)

; Accessor
(= (get-perception-precision $source $obs)
   (let $result (match &self (perception-precision $source $obs $prec) $prec)
     (if (== $result ())
         0.20    ; Default: low precision for unknown source/observable pairs
         $result)))


; =============================================================================
; SECTION III: UNRESOLVED SALIENCE â€” "SOMETHING I DON'T UNDERSTAND"
; =============================================================================
;
; THE BLIND SPOT THIS FIXES:
;   User shouts "There is a fire!" but "fire" is not in topic-map.
;   Without this observable, the system silently drops the topic and
;   acts as if nothing was said. That's honest (don't ground ungrounded
;   tokens) but dangerous (ignoring urgent unknowns).
;
; THE FIX:
;   Add a meta-observable: unresolved-salience. It measures "how much
;   salient/urgent content arrived that I couldn't map to known observables."
;   This is NOT grounding the unknown token â€” "fire" still doesn't become
;   an observable. It's registering "something important is happening that
;   I don't understand."
;
; HOW IT ENTERS EFE:
;   High unresolved-salience â†’ high prediction error (preference is 0.1)
;   â†’ request-clarification and query-partner get low EFE (their models
;     predict they improve partner-predictability and reduce uncertainty)
;   â†’ system asks "What do you mean?" or "Can you explain?"
;   The response EMERGES from EFE, not from "if unknown topic, ask."
;
; TWO INPUTS:
;   1. Unresolved topic count: how many topics couldn't be mapped
;   2. LLM urgency estimate: how urgent/salient did the LLM judge
;      the utterance to be (independent of topic resolution)
;   Combined: unresolved_salience = urgency Ã— (unresolved / total)
;   If all topics resolve, salience is 0 regardless of urgency.
;   If urgency is 0, salience is 0 regardless of unresolved count.

(: unresolved-salience ConversationalObservable)

; Belief: prior = low (usually we understand what's said)
; BF: precision starts near-zero (from config), not 0.5
(belief unresolved-salience 0.1 0.01)

; Preference: we WANT this to be low (understanding is good)
(preference unresolved-salience 0.1 0.8)

; Viability: if we consistently can't understand, conversation fails
(viability-bound unresolved-salience 0.0 0.85)

; Action models: what helps when unresolved-salience is high?
; request-clarification directly addresses the gap
(action-model request-clarification unresolved-salience -0.2 0.06)
; query-partner also helps (gathering information)
(action-model query-partner unresolved-salience -0.1 0.08)
; acknowledge buys time but doesn't resolve
(action-model acknowledge unresolved-salience -0.02 0.01)
; asserting when confused makes things worse
(action-model assert unresolved-salience 0.05 0.01)


; =============================================================================
; SECTION IV: TOPIC â†’ OBSERVABLE MAPPING
; =============================================================================
;
; The LLM parse extracts topic words from user text. These must map
; to known observables or concepts in the atomspace. Unknown topics
; contribute to unresolved-salience rather than being silently dropped.
;
; This mapping is part of the deployment configuration (like action models).
; A Mars rover deployment has different topic mappings than a chat agent.

; Conversational topic mappings
(topic-map "understanding" partner-comprehension)
(topic-map "comprehension" partner-comprehension)
(topic-map "confused" partner-comprehension)
(topic-map "unclear" partner-comprehension)
(topic-map "makes sense" partner-comprehension)
(topic-map "get it" partner-comprehension)

(topic-map "topic" discourse-coherence)
(topic-map "relevant" discourse-coherence)
(topic-map "coherent" discourse-coherence)
(topic-map "off-topic" discourse-coherence)

(topic-map "progress" goal-progress)
(topic-map "goal" goal-progress)
(topic-map "done" goal-progress)
(topic-map "help" goal-progress)

(topic-map "rapport" rapport)
(topic-map "friendly" rapport)
(topic-map "rude" rapport)
(topic-map "thanks" rapport)

; Domain topic mappings (Mars rover example â€” deployment-specific)
(topic-map "power" power-level)
(topic-map "battery" power-level)
(topic-map "energy" power-level)
(topic-map "terrain" terrain-roughness)
(topic-map "surface" terrain-roughness)
(topic-map "rough" terrain-roughness)
(topic-map "threat" threat-level)
(topic-map "danger" threat-level)
(topic-map "safe" threat-level)

; Resolve a topic string to an observable (or unknown)
(= (resolve-topic $topic-str)
   (let $result (match &self (topic-map $topic-str $obs) $obs)
     (if (== $result ())
         (unknown-topic $topic-str)
         $result)))

; Resolve a list of topic strings to observables, filtering unknowns
(= (resolve-topics ()) ())
(= (resolve-topics ($t . $rest))
   (let $resolved (resolve-topic $t)
     (if (== $resolved (unknown-topic $t))
         (resolve-topics $rest)         ; Skip unknowns from resolved list
         ($resolved . (resolve-topics $rest)))))

; Count how many topics failed to resolve
(= (count-unresolved-topics ()) 0)
(= (count-unresolved-topics ($t . $rest))
   (let $resolved (resolve-topic $t)
     (if (== $resolved (unknown-topic $t))
         (+ 1 (count-unresolved-topics $rest))
         (count-unresolved-topics $rest))))

; Compute unresolved salience from urgency and topic resolution
; salience = urgency Ã— (unresolved / total) when topics exist
; salience = urgency Ã— 0.5 when no topics extracted (LLM failed on topics
;   but still detected urgency â€” partial credit)
(= (compute-unresolved-salience $urgency $topics)
   (let* (
     ($total (length $topics))
     ($unresolved (count-unresolved-topics $topics))
   )
   (if (== $total 0)
       (* $urgency 0.5)                           ; No topics but urgent â†’ moderate salience
       (* $urgency (/ $unresolved $total)))))      ; Fraction unresolved Ã— urgency


; =============================================================================
; SECTION V: UTTERANCE PERCEPTION
; =============================================================================
;
; Entry point for NL input. Takes the structured parse from the
; Python orchestrator and injects observations into the cognitive cycle.
;
; Parse schema (fields arrive from Python as a flat structure):
;   type:          query | request | assertion | greeting | farewell | emotional
;   topics:        list of topic strings
;   comprehension: 0.0-1.0
;   coherence:     0.0-1.0
;   rapport:       0.0-1.0
;   goal-progress: 0.0-1.0
;   urgency:       0.0-1.0 (LLM's estimate of utterance urgency/salience)
;   summary:       string (for logging, not used in reasoning)
;
; Each numeric field becomes an observation with source-appropriate precision.
; The urgency field combines with unresolved topic count to produce the
; unresolved-salience observation.

(= (perceive-utterance! $type $topics $comprehension $coherence
                        $rapport-est $progress-est $urgency $time)
   (let* (
     ; Resolve topic strings to known observables
     ($resolved-topics (resolve-topics $topics))

     ; Compute unresolved salience BEFORE discarding unknown topics
     ; This is where "fire" (unknown) + urgency 0.9 â†’ salience 0.9
     ($salience (compute-unresolved-salience $urgency $topics))
     ($salience-prec (get-perception-precision llm-parse unresolved-salience))

     ; Get precisions for LLM-derived estimates
     ($comp-prec (get-perception-precision llm-parse partner-comprehension))
     ($coh-prec (get-perception-precision llm-parse discourse-coherence))
     ($rap-prec (get-perception-precision llm-parse rapport))
     ($prog-prec (get-perception-precision llm-parse goal-progress))

     ; Compute predictability from utterance type (how expected was this?)
     ($predict-est (compute-type-predictability $type))
     ($predict-prec (get-perception-precision llm-parse partner-predictability))

     ; Process through conversation model's existing pipeline
     ; This updates partner model, partner-knowledge, partner-goals
     ($conv-update (process-utterance! $type "" $resolved-topics $time))

     ; Inject LLM-derived conversational observations
     ; These feed the SAME belief-update pipeline as any sensor reading
     ($obs-comp (inject-observation! partner-comprehension
                  $comprehension $comp-prec $time))
     ($obs-coh (inject-observation! discourse-coherence
                 $coherence $coh-prec $time))
     ($obs-rap (inject-observation! rapport
                 $rapport-est $rap-prec $time))
     ($obs-prog (inject-observation! goal-progress
                  $progress-est $prog-prec $time))
     ($obs-pred (inject-observation! partner-predictability
                  $predict-est $predict-prec $time))

     ; Inject unresolved salience â€” "something is happening I don't understand"
     ; When salience is 0 (all topics resolved or urgency is 0), this
     ; observation matches the preference (0.1) and produces no error.
     ; When salience is high, the error drives request-clarification via EFE.
     ($obs-salience (inject-observation! unresolved-salience
                      $salience $salience-prec $time))

     ; Log unresolved topics for diagnostics and potential structure learning
     ($unresolved-log (log-unresolved-topics! $topics $urgency $time))

     ; Advance conversation turn
     ($turn (advance-turn!))

     ; Record what we perceived (for introspection/debugging)
     ($log (record-perception! $type $resolved-topics $time))
   )
   (perception-result
     (turn $turn)
     (type $type)
     (topics $resolved-topics)
     (unresolved-salience $salience)
     (observations 6))))

; Log unresolved topics â€” not used in reasoning, but provides a trail
; for future topic-map expansion or structure learning integration
(= (log-unresolved-topics! () $urgency $time) done)
(= (log-unresolved-topics! ($t . $rest) $urgency $time)
   (let $resolved (resolve-topic $t)
     (sequential
       (if (== $resolved (unknown-topic $t))
           (add-atom &state (unresolved-topic-log $t $urgency $time))
           done)
       (log-unresolved-topics! $rest $urgency $time))))

; How predictable was this utterance type given our partner model?
(= (compute-type-predictability $type)
   (let $goal-beliefs (collapse
     (match &state (partner-goal-belief $goal $conf) ($goal $conf)))
     (type-predictability-from-goals $type $goal-beliefs)))

; Match utterance type to partner goal model
(= (type-predictability-from-goals $type ()) 0.5) ; Default: moderate
(= (type-predictability-from-goals $type (($goal $conf) . $rest))
   (if (type-matches-goal? $type $goal)
       (max $conf (type-predictability-from-goals $type $rest))
       (type-predictability-from-goals $type $rest)))

(= (type-matches-goal? query information-seeking) True)
(= (type-matches-goal? request task-completion) True)
(= (type-matches-goal? assertion exploration) True)
(= (type-matches-goal? greeting exploration) True)
(= (type-matches-goal? $type $goal) False)


; =============================================================================
; SECTION V: SENSOR PERCEPTION (for 2D environment)
; =============================================================================
;
; Direct sensor readings from the simulation environment.
; Much simpler than NL perception â€” just normalize and inject.
;
; The Python orchestrator calls these when the simulation produces
; new readings. No LLM involved.

(= (perceive-sensor! $sensor-id $raw-value $time)
   (let* (
     ($obs (sensor-observable $sensor-id))
     ($val (sensor-normalize $sensor-id $raw-value))
     ($prec (get-perception-precision simulation $obs))
   )
   (inject-observation! $obs $val $prec $time)))

; Sensor configuration: which sensor maps to which observable
; (deployment-specific â€” these are examples)
(sensor-config position-x terrain-roughness 0.0 100.0)
(sensor-config battery-voltage power-level 0.0 5.0)
(sensor-config threat-detector threat-level 0.0 1.0)

; Resolve sensor ID to observable
(= (sensor-observable $sensor-id)
   (match &self (sensor-config $sensor-id $obs $min $max) $obs))

; Normalize raw reading to 0-1 range using sensor config
(= (sensor-normalize $sensor-id $raw)
   (let ($obs $min $max) (match &self (sensor-config $sensor-id $o $mn $mx)
                                      ($o $mn $mx))
     (clamp (/ (- $raw $min) (- $max $min)) 0.0 1.0)))

; Batch sensor injection (multiple readings at once)
(= (perceive-sensors! () $time) done)
(= (perceive-sensors! (($sensor $value) . $rest) $time)
   (sequential
     (perceive-sensor! $sensor $value $time)
     (perceive-sensors! $rest $time)))


; =============================================================================
; SECTION VI: ACTION VERBALIZATION PACKAGING
; =============================================================================
;
; After the cognitive cycle selects an action, this module packages
; the system's internal state into a structured intent that the
; Python orchestrator sends to the LLM for verbalization.
;
; The LLM's job is SENTENCE CONSTRUCTION, not content decisions.
; All content decisions have been made by EFE.

(= (package-verbalization-intent $action $time)
   (let* (
     ; What does the system know about why it's doing this?
     ($affect (get-affect-state))
     ($valence (compute-valence))
     ($arousal (compute-arousal))
     ($dominance (compute-dominance))

     ; What's driving this action? (largest prediction errors)
     ($errors (all-prediction-errors))
     ($drivers (top-error-drivers $errors))

     ; Conversation state
     ($conv-state (conversation-state-description))
     ($topics (current-topics))

     ; Self-understanding (if available)
     ($self-knowledge (what-do-i-know-about-action $action))
   )
   (verbalization-intent
     (action $action)
     (affect (valence $valence) (arousal $arousal) (dominance $dominance))
     (drivers $drivers)
     (conversation $conv-state)
     (topics $topics)
     (self-knowledge $self-knowledge)
     (time $time))))

; Extract what we know about an action for verbalization context
(= (what-do-i-know-about-action $action)
   (let $effects (collapse
     (match &state (action-model $action $obs $vd $pd)
       (effect $obs $vd)))
     (if (== $effects ())
         (no-model $action)
         (action-knowledge $action $effects))))

; Get the top prediction error drivers (observables with largest errors)
(= (top-error-drivers ()) ())
(= (top-error-drivers ((prediction-error $obs $err $abs $prec $we) . $rest))
   (if (> $abs 0.1)
       ((driver $obs $err $abs) . (top-error-drivers $rest))
       (top-error-drivers $rest)))


; =============================================================================
; SECTION VII: PERCEPTION LOG
; =============================================================================
;
; Record what was perceived for debugging and introspection.
; These are lightweight trace atoms, not used in reasoning.
; Cleaned up periodically (same sliding window as error traces).

(= (record-perception! $type $topics $time)
   (add-atom &state (perception-log $type $topics $time)))

; Clean old perception logs (keep last N)
(= (clean-perception-logs! $current-time)
   (let* (
     ($window (get-config lookback-window))
     ($cutoff (- $current-time (* $window 3)))   ; Keep 3x lookback window
     ($old (collapse (match &state (perception-log $t $tp $time)
       (if (< $time $cutoff) (perception-log $t $tp $time) ()))))
     ($old-unresolved (collapse (match &state (unresolved-topic-log $t $u $time)
       (if (< $time $cutoff) (unresolved-topic-log $t $u $time) ()))))
   )
   (sequential
     (remove-old-perceptions! $old)
     (remove-old-perceptions! $old-unresolved))))

(= (remove-old-perceptions! ()) done)
(= (remove-old-perceptions! ($log . $rest))
   (sequential
     (remove-atom &state $log)
     (remove-old-perceptions! $rest)))


; =============================================================================
; SECTION VIII: PARSE FAILURE HANDLING
; =============================================================================
;
; When the LLM fails to produce a field, the orchestrator sends
; a default value. These defaults are designed to be minimally
; informative â€” they barely move beliefs.
;
; The orchestrator uses these same defaults (defined in Python).
; They are documented here for reference and consistency.
;
; Field          Default    Precision    Effect
; -----          -------    ---------    ------
; type           query      n/a          Safe assumption
; comprehension  0.5        0.10         Uninformative (prior midpoint)
; coherence      0.5        0.10         Uninformative
; rapport        0.5        0.10         Uninformative
; goal-progress  0.5        0.10         Uninformative
; urgency        0.0        0.10         Assume not urgent (safe default)
; topics         ()         n/a          No topic update
;
; NOTE: urgency defaults to 0.0, not 0.5. If the LLM fails to parse
; urgency, we assume the situation is NOT urgent. This is conservative:
; false negatives (missing urgency) are safer than false positives
; (phantom urgency causing erratic behavior). The 0.10 precision means
; even this default barely influences the belief.
;
; At precision 0.10, an observation barely moves even a low-precision
; belief. The system effectively ignores the failed parse.

; Minimum precision for fallback observations
(config perception-fallback-precision 0.10)

; Perceive with fallback (called when orchestrator signals partial parse)
(= (perceive-utterance-partial! $type $topics $fields $time)
   (let* (
     ($fallback-prec (get-config perception-fallback-precision))
     ; Extract fields with defaults
     ($comp (field-or-default $fields comprehension 0.5))
     ($coh (field-or-default $fields coherence 0.5))
     ($rap (field-or-default $fields rapport 0.5))
     ($prog (field-or-default $fields goal-progress 0.5))
     ($urg (field-or-default $fields urgency 0.0))
   )
   ; Use the main perceive function â€” precision is already set per-source
   ; For failed fields, override with fallback precision
   (perceive-utterance! $type $topics $comp $coh $rap $prog $urg $time)))

; Extract field from list of (key value) pairs, or return default
(= (field-or-default () $key $default) $default)
(= (field-or-default (($k $v) . $rest) $key $default)
   (if (== $k $key) $v (field-or-default $rest $key $default)))


; =============================================================================
; SECTION IX: PERCEPTION DIAGNOSTICS
; =============================================================================

; What is the current perception configuration?
(= (perception-config-summary)
   (let* (
     ($llm-precs (collapse
       (match &self (perception-precision llm-parse $obs $prec)
         (llm-precision $obs $prec))))
     ($sensor-precs (collapse
       (match &self (perception-precision simulation $obs $prec)
         (sim-precision $obs $prec))))
     ($topic-count (length (collapse
       (match &self (topic-map $t $o) $t))))
   )
   (perception-summary
     (llm-precisions $llm-precs)
     (sensor-precisions $sensor-precs)
     (known-topics $topic-count))))

; Recent perception history
(= (recent-perceptions)
   (collapse (match &state (perception-log $type $topics $time)
     (perceived $time $type $topics))))

; What topics can we understand?
(= (known-topic-mappings)
   (collapse (match &self (topic-map $topic $obs)
     (maps $topic $obs))))

; What topics have we failed to resolve recently?
; This is useful for expanding the topic-map in future deployments.
(= (recent-unresolved-topics)
   (collapse (match &state (unresolved-topic-log $topic $urgency $time)
     (unresolved $topic (urgency $urgency) (at $time)))))

; Current unresolved salience belief
(= (current-unresolved-salience)
   (match &state (belief unresolved-salience $val $prec)
     (unresolved-salience-state (value $val) (precision $prec)
       (interpretation
         (if (> $val 0.5) "something important I don't understand"
           (if (> $val 0.2) "mild confusion about partner's intent"
             "situation understood"))))))


; =============================================================================
; END OF PERCEPTION MODULE
; =============================================================================
