; =============================================================================
; SELF-MODEL (v3) - Integrated with Active Inference & Structure Learning
; =============================================================================
;
; DESIGN PRINCIPLES APPLIED:
;   - Self is an entity queryable like any other
;   - Capabilities EMERGE from action model statistics, not enumeration
;   - Limitations EMERGE from missing/failed models
;   - Epistemic state connects to beliefs and affects EFE
;   - "I don't know" emerges naturally from query failure
;   - Structural self-knowledge: the agent can introspect its own
;     discovered causal structure (passive models, latent variables)
;
; KEY INSIGHT: The self-model IS the agent's beliefs about its own
; action models and their reliability. Self-knowledge is just another
; domain of prediction and error. Structure learning extends this:
; the agent can now query what causal relationships it has discovered
; and how its world model is evolving.
;
; CHANGELOG:
;   v3 - Added structural self-knowledge (Section XI):
;        structural complexity, learning activity queries,
;        invented concepts introspection, structural health
;        as self-observable
;   v2 - Integrated with Active Inference
;   v1 - Initial self-model
;
; =============================================================================


; =============================================================================
; SECTION I: SELF AS ENTITY
; =============================================================================

(: Self Entity)
(isa Self Agent)

; Identity (minimal, queryable)
(self-property name "MeTTa-AI")
(self-property architecture active-inference)
(self-property grounding-requirement honest-only)


; =============================================================================
; SECTION II: ACTION MODEL STATISTICS
; =============================================================================

; This is the GROUND TRUTH for self-knowledge about capabilities.
; Updated by the cognitive cycle after each action.
;
; Schema: (action-stats <action> <successes> <attempts> <avg-error>)

; Initialize with priors (will be updated through experience)
(action-stats wait 8 10 0.1)
(action-stats observe 7 10 0.15)
(action-stats retreat 6 10 0.2)
(action-stats reflect 3 10 0.3)

; Update action statistics after outcome
(= (update-action-stats! $action $success $error)
   (match &state (action-stats $action $s $a $avg-err)
     (let* (
       ($new-attempts (+ $a 1))
       ($new-successes (if $success (+ $s 1) $s))
       ($new-avg-err (+ (* 0.9 $avg-err) (* 0.1 $error)))
     )
     (sequential
       (remove-atom &state (action-stats $action $s $a $avg-err))
       (add-atom &state (action-stats $action $new-successes $new-attempts $new-avg-err))
       (stats-updated $action $new-successes $new-attempts)))))


; =============================================================================
; SECTION III: EMERGENT CAPABILITIES
; =============================================================================

; A capability is NOT declared - it EMERGES from action model success.
; "I can do X" means "my model of X has good success rate and low error"

(= (success-rate $action)
   (match &state (action-stats $action $s $a $_)
     (/ $s (max 1 $a))))

(= (avg-error $action)
   (match &state (action-stats $action $_ $_ $err) $err))

; Capability thresholds AS BELIEFS (can be updated through experience)
; These start as priors and adapt if the system finds them miscalibrated
; BF: precision starts near-zero — no confidence in thresholds until observed
(belief capability-success-threshold 0.6 0.01)
(belief capability-error-threshold 0.3 0.01)

; Can Self do this action?
; Thresholds come from beliefs, not hardcoded values
(= (can-do? $action)
   (let* (
     ($sr (success-rate $action))
     ($err (avg-error $action))
     ($sr-thresh (match &state (belief capability-success-threshold $t $_) $t))
     ($err-thresh (match &state (belief capability-error-threshold $t $_) $t))
   )
   (if (== $sr ())
       unknown  ; No data on this action
       (and (> $sr $sr-thresh) (< $err $err-thresh)))))

; What can Self do? (returns list of capable actions)
(= (capabilities)
   (collapse 
     (match &state (action-stats $action $_ $_ $_)
       (if (can-do? $action) $action (empty)))))

; Confidence in capability (derived from sample size and consistency)
(= (capability-confidence $action)
   (match &state (action-stats $action $s $a $err)
     (let* (
       ($sample-factor (min 1.0 (/ $a 20)))  ; More attempts = more confidence
       ($success-factor (/ $s (max 1 $a)))
       ($error-factor (- 1.0 $err))
     )
     (* $sample-factor (* $success-factor $error-factor)))))


; =============================================================================
; SECTION IV: EMERGENT LIMITATIONS
; =============================================================================

; Limitations emerge from:
; 1. Missing action models (never tried)
; 2. Failed action models (tried but poor results)
; 3. Explicit impossibilities (structural)

; Cannot do due to poor track record
(= (limited-by-failure? $action)
   (let $sr (success-rate $action)
     (if (== $sr ())
         False
         (< $sr 0.3))))

; Cannot do due to no model (never attempted)
(= (limited-by-no-model? $action)
   (== (match &state (action-stats $action $_ $_ $_) found) ()))

; Structural limitations (these ARE declared, but sparingly)
(structural-limitation real-time-perception)
(structural-limitation physical-action)
(structural-limitation persistent-memory)

(= (limited-by-structure? $action)
   (case (match &self (structural-limitation $action) True)
     ((True True)
      (() False))))

; Unified limitation check
(= (cannot-do? $action)
   (or (limited-by-structure? $action)
       (or (limited-by-failure? $action)
           (limited-by-no-model? $action))))

; Why can't Self do this?
(= (limitation-reason $action)
   (if (limited-by-structure? $action)
       (reason structural "Not physically possible")
       (if (limited-by-failure? $action)
           (reason poor-performance (success-rate $action))
           (if (limited-by-no-model? $action)
               (reason no-experience "Never attempted")
               (reason none "Can do this")))))


; =============================================================================
; SECTION V: EPISTEMIC STATE (Current Knowledge)
; =============================================================================

; What topics does Self have beliefs about?
; This QUERIES the actual belief base, not a separate store.

(= (knows-about? $topic)
   (case (match &state (belief $topic $_ $_) found)
     ((found True)
      (() False))))

; Confidence about a topic (from belief precision)
(= (confidence-about $topic)
   (match &state (belief $topic $_ $prec) $prec))

; Topics Self is uncertain about (low precision beliefs)
(= (uncertain-topics)
   (collapse
     (match &state (belief $topic $_ $prec)
       (if (< $prec 0.4) $topic (empty)))))

; Topics Self is confident about
(= (confident-topics)
   (collapse
     (match &state (belief $topic $_ $prec)
       (if (> $prec 0.7) $topic (empty)))))


; =============================================================================
; SECTION VI: SELF AS OBSERVABLE
; =============================================================================

; Self-knowledge can be treated as observables in the Active Inference loop.
; This allows meta-cognition: reasoning about one's own states.

(: SelfObservable Type)
(: self-competence SelfObservable)    ; Overall capability assessment
(: self-confidence SelfObservable)    ; Certainty in own knowledge
(: model-reliability SelfObservable)  ; How well are predictions working
(: structural-health SelfObservable)  ; Is structure learning helping or stale

; Compute self-competence (aggregate of action success rates)
(= (compute-self-competence)
   (let $rates (collapse (match &state (action-stats $a $s $n $_) (/ $s (max 1 $n))))
     (safe-avg $rates)))

(= (safe-avg ()) 0.5)
(= (safe-avg ($x)) $x)
(= (safe-avg ($x $y)) (/ (+ $x $y) 2))
(= (safe-avg ($x $y $z)) (/ (+ (+ $x $y) $z) 3))
(= (safe-avg ($x . $rest)) (/ (+ $x (* (safe-avg $rest) (length $rest))) (+ 1 (length $rest))))

(= (length ()) 0)
(= (length ($x . $xs)) (+ 1 (length $xs)))

; Compute self-confidence (aggregate of belief precisions)
(= (compute-self-confidence)
   (let $precs (collapse (match &state (belief $_ $_ $p) $p))
     (safe-avg $precs)))

; Compute model reliability (inverse of recent prediction errors)
(= (compute-model-reliability)
   (let $errs (match &state (error-history $h) $h)
     (if (== $errs ())
         0.5
         (- 1.0 (safe-avg (extract-errors $errs))))))

; Compute structural health: how well is structure learning performing?
; High = structure exists and reducing surprise. Low = no structure or stale.
; Normalized to [0, 1] via sigmoid-like mapping.
(= (compute-structural-health)
   (let* (
     ($n-pm (passive-model-count))
     ($n-lv (latent-variable-count))
     ($total (+ $n-pm $n-lv))
     ($surprise (structural-surprise))
   )
   (if (== $total 0)
       0.5          ; No structure yet: neutral
       ; More structure + less surprise = healthier
       ; Heuristic: health = structures / (structures + surprise)
       (/ $total (+ $total (+ $surprise 0.1))))))

(= (extract-errors ()) ())
(= (extract-errors (($err $_) . $rest)) ($err . (extract-errors $rest)))

; Generate self-observations (called periodically)
(= (observe-self! $time)
   (sequential
     (add-atom &state (observation self-competence (compute-self-competence) 0.8 $time))
     (add-atom &state (observation self-confidence (compute-self-confidence) 0.7 $time))
     (add-atom &state (observation model-reliability (compute-model-reliability) 0.75 $time))
     (add-atom &state (observation structural-health (compute-structural-health) 0.6 $time))
     self-observed))


; =============================================================================
; SECTION VII: SELF-BELIEFS (Meta-Cognition)
; =============================================================================

; Self can have beliefs about its own states
; These are updated like any other beliefs through prediction error
; BF: all precisions start near-zero

(belief self-competence 0.7 0.01)     ; Prior: moderately competent
(belief self-confidence 0.5 0.01)     ; Prior: moderate confidence
(belief model-reliability 0.6 0.01)   ; Prior: models work okay
(belief structural-health 0.5 0.01)   ; Prior: neutral (new)

; Preferences about self-states (feed into EFE)
(preference self-competence 0.8 0.7)     ; Want to be competent
(preference self-confidence 0.7 0.5)     ; Want to be confident
(preference model-reliability 0.8 0.9)   ; Critical: models must work
(preference structural-health 0.7 0.4)   ; Prefer healthy structure

; --- Grounding-completeness: meta-cognitive observable ---
; Tracks what fraction of learned atoms have semantic grounding.
; Initial belief is high (few learned atoms at startup = vacuously grounded).
; Preference drives EFE pressure toward grounding when completeness drops.
; Updated by grounding_hypotheses.metta:update-grounding-completeness!
(belief grounding-completeness 0.85 0.3)
(preference grounding-completeness 1.0 0.6)

; Viability bounds on self-states
(viability-bound self-competence 0.3 1.0)     ; Don't become incompetent
(viability-bound model-reliability 0.2 1.0)   ; Models must not fail completely
(viability-bound structural-health 0.1 1.0)   ; Don't let structure completely decay


; =============================================================================
; SECTION VIII: SELF-QUERIES (Natural Language Interface)
; =============================================================================

; "What do you know about X?"
(= (what-do-i-know-about $topic)
   (if (knows-about? $topic)
       (let* (
         ($val (match &state (belief $topic $v $_) $v))
         ($conf (confidence-about $topic))
       )
       (knowledge-report $topic $val $conf))
       (no-knowledge $topic)))

; "How confident are you about X?"
(= (how-confident-about $topic)
   (let $conf (confidence-about $topic)
     (if (== $conf ())
         (confidence-report $topic unknown)
         (confidence-report $topic $conf))))

; "Can you do X?"
(= (can-i-do $action)
   (let $result (can-do? $action)
     (if (== $result True)
         (capability-report $action yes (capability-confidence $action))
         (if (== $result unknown)
             (capability-report $action unknown no-data)
             (capability-report $action no (limitation-reason $action))))))

; "What are you good at?"
(= (what-am-i-good-at)
   (capabilities))

; "What are you uncertain about?"
(= (what-am-i-uncertain-about)
   (uncertain-topics))


; =============================================================================
; SECTION IX: INTEGRATION WITH EFE
; =============================================================================

; Self-knowledge affects action selection through:
; 1. Action models use capability confidence
; 2. Self-state observables have preferences
; 3. Meta-actions can improve self-knowledge

; Adjust action cost based on capability confidence
; Low confidence = higher effective cost (risky to attempt)
(= (confidence-adjusted-cost $action $base-cost)
   (let $conf (capability-confidence $action)
     (if (== $conf ())
         (* $base-cost 1.5)  ; Unknown action: penalty
         (* $base-cost (+ 0.5 (- 1.0 $conf))))))

; Meta-action: reflect
; Type, action-model, and action-cost declarations are in actions.metta
; (5-field schema with confidence). This module provides only the
; action-stats prior for the self-model.


; =============================================================================
; SECTION X: LEARNING FROM OUTCOMES
; =============================================================================

; After each action, update self-model
(= (learn-from-outcome! $action $predicted-error $actual-error)
   (let* (
     ($success (< $actual-error $predicted-error))
     ($error-diff (abs (- $actual-error $predicted-error)))
   )
   (sequential
     (update-action-stats! $action $success $error-diff)
     (outcome-learned $action $success))))

; Periodic self-assessment (call from cognitive cycle)
(= (self-assess! $time)
   (sequential
     (observe-self! $time)
     (selective-update-belief! self-competence)
     (selective-update-belief! self-confidence)
     (selective-update-belief! model-reliability)
     (selective-update-belief! structural-health)
     self-assessed))


; =============================================================================
; SECTION XI: STRUCTURAL SELF-KNOWLEDGE
; =============================================================================
;
; The agent can introspect its own discovered causal structure.
; This extends the emergent capabilities pattern: just as "I can do X"
; emerges from action model statistics, "I understand XÃ¢â€ â€™Y" emerges
; from the existence of passive models and latent variables.
;
; These queries are used by:
;   - The explanation interface (what has the agent learned?)
;   - Self-assessment (is structure learning helping?)
;   - The NL interface ("What have you figured out?")

; How complex is my current world model?
(= (structural-complexity)
   (let* (
     ($n-links (passive-model-count))
     ($n-latents (latent-variable-count))
     ($n-suspicions (length (all-suspicion-links)))
   )
   (complexity-report
     (passive-models $n-links)
     (latent-variables $n-latents)
     (suspicion-links $n-suspicions))))

; Am I actively learning structure right now?
; True when there are suspicion links being tracked (co-errors detected)
(= (currently-learning-structure?)
   (structure-learning-active?))

; What causal relationships have I discovered?
; Returns a queryable list of causeÃ¢â€ â€™effect relationships.
(= (discovered-relationships)
   (collapse
     (match &state (passive-model $cause $effect $lag $weight $type)
       (relationship $cause $type $effect (lag $lag) (strength $weight)))))

; What latent concepts have I invented?
; Returns each concept with its member observables.
(= (invented-concepts)
   (collapse
     (match &state (latent-variable $name)
       (let $members (get-latent-members $name)
         (concept $name (explains $members))))))

; How is a specific observable being predicted by structure?
; Shows all structural atoms that predict this observable.
(= (structural-predictions-for $obs)
   (let* (
     ($causal (passive-models-affecting $obs))
     ($latent (collapse (match &state (latent-member $name $obs $w)
       (latent-prediction $name $obs $w))))
   )
   (structural-predictions $obs
     (from-causal-links $causal)
     (from-latent-variables $latent))))

; What is the strongest relationship I've found?
(= (strongest-relationship)
   (let $all (all-passive-models)
     (find-strongest-pm $all (none 0.0))))

(= (find-strongest-pm () $best) $best)
(= (find-strongest-pm ((passive-model $c $e $l $w $t) . $rest) (best-pm $bc $be $bw))
   (if (> $w $bw)
       (find-strongest-pm $rest (best-pm $c $e $w))
       (find-strongest-pm $rest (best-pm $bc $be $bw))))
(= (find-strongest-pm ((passive-model $c $e $l $w $t) . $rest) (none $bw))
   (find-strongest-pm $rest (best-pm $c $e $w)))


; =============================================================================
; SECTION XII: STRUCTURAL NL QUERIES
; =============================================================================
;
; Natural language-oriented queries about structural self-knowledge.
; These parallel the existing "What do you know about X?" queries.

; "What have you figured out about the world?"
(= (what-have-i-learned)
   (let* (
     ($rels (discovered-relationships))
     ($concepts (invented-concepts))
     ($health (compute-structural-health))
   )
   (structural-knowledge-report
     (relationships $rels)
     (concepts $concepts)
     (health $health))))

; "Why did X happen?" Ã¢â‚¬â€ look for causal explanations
(= (why-did $obs $direction)
   (let $models (passive-models-affecting $obs)
     (if (== $models ())
         (no-causal-explanation $obs)
         (causal-explanation $obs $models))))

; "What does X affect?" Ã¢â‚¬â€ look for downstream effects
(= (what-does-affect $obs)
   (let $models (passive-models-from $obs)
     (if (== $models ())
         (no-known-effects $obs)
         (known-effects $obs $models))))

; "Are you confused right now?"
(= (am-i-confused)
   (let* (
     ($surprise (structural-surprise))
     ($salient (salient-count))
   )
   (if (> $surprise 0.5)
       (confused-report yes $surprise $salient)
       (if (> $surprise 0.1)
           (confused-report somewhat $surprise $salient)
           (confused-report no $surprise $salient)))))


; =============================================================================
; SECTION XIII: INVENTED OBSERVABLES (Phase 3)
; =============================================================================

; "What observables did I discover that weren't in my deployment manifest?"
(= (invented-observables)
   (collapse
     (match &state (promoted-observable $name $source $cycle)
       (let* (
         ($energy (get-lifecycle-energy promoted-observable $name))
         ($val (get-belief-value $name))
         ($prec (get-belief-precision $name))
       )
       (invented-observable $name
         (from $source)
         (promoted-at $cycle)
         (current-belief $val $prec)
         (energy $energy))))))

; "What did observable X start as?"
(= (observable-origin $name)
   (let $r (match &state (atom-origin $name $phase $info) ($phase $info))
     (if (== $r ())
         (deployment-observable $name)  ; Not invented â€” original
         (let ($phase $info) $r
           (invented-from $name $phase $info)))))

; "How many sensors do I have now vs at deployment?"
(= (observable-growth)
   (let* (
     ($all-beliefs (collapse (match &state (belief $obs $v $p) $obs)))
     ($n-total (length $all-beliefs))
     ($n-promoted (length (all-promoted-observables)))
     ($n-percept (length (all-perceptual-fields)))
     ($n-original (- $n-total (+ $n-promoted $n-percept)))
   )
   (observable-growth-report
     (original $n-original)
     (invented $n-promoted)
     (expanded $n-percept)
     (total $n-total))))


; =============================================================================
; SECTION XIV: ACQUIRED SKILLS (Phase 4)
; =============================================================================

; "What composite actions have I learned?"
(= (acquired-skills)
   (collapse
     (match &state (chunked-action $policy $cycle)
       (let* (
         ($energy (get-lifecycle-energy chunked-action $policy))
         ($cost (get-action-cost $policy))
         ($freq (policy-frequency $policy))
       )
       (acquired-skill
         (policy $policy)
         (learned-at $cycle)
         (cost $cost)
         (recent-uses $freq)
         (energy $energy))))))

; "Can I do X as a single action?" â€” extended to include chunks
(= (can-do-composite? $policy)
   (already-chunked? $policy))

; "What's my most reliable skill?"
(= (best-skill)
   (let $chunks (all-chunked-actions)
     (find-highest-energy-chunk $chunks (none 0.0))))

(= (find-highest-energy-chunk () $best) $best)
(= (find-highest-energy-chunk ($policy . $rest) (best $bp $be))
   (let $e (get-lifecycle-energy chunked-action $policy)
     (if (> $e $be)
         (find-highest-energy-chunk $rest (best $policy $e))
         (find-highest-energy-chunk $rest (best $bp $be)))))
(= (find-highest-energy-chunk ($policy . $rest) (none $be))
   (let $e (get-lifecycle-energy chunked-action $policy)
     (find-highest-energy-chunk $rest (best $policy $e))))


; =============================================================================
; SECTION XV: DISCOVERED SENSES (Phase 5)
; =============================================================================

; "What new senses have I developed?"
(= (discovered-senses)
   (collapse
     (match &state (perceptual-field $name $query $prec $cycle)
       (let $energy (get-lifecycle-energy perceptual-field $name)
         (discovered-sense $name
           (query $query)
           (precision $prec)
           (activated-at $cycle)
           (energy $energy))))))

; "Am I looking for new things to perceive?"
(= (seeking-new-perception?)
   (perceptual-expansion-needed?))


; =============================================================================
; SECTION XVI: UNIFIED VOCABULARY GROWTH
; =============================================================================

; "How is my understanding of the world evolving?"
; This is the meta-cognitive query that synthesizes all three phases.
(= (vocabulary-growth-report)
   (let* (
     ($obs-growth (observable-growth))
     ($skills (acquired-skills))
     ($senses (discovered-senses))
     ($promotion-log (promotion-history))
     ($demotion-log (demotion-history))
   )
   (vocabulary-report
     (observables $obs-growth)
     (skills $skills)
     (senses $senses)
     (promotions $promotion-log)
     (demotions $demotion-log))))

; "What have you figured out?" â€” v2, extends existing
(= (what-have-i-learned-v2)
   (let* (
     ($rels (discovered-relationships))
     ($concepts (invented-concepts))
     ($health (compute-structural-health))
     ($observables (invented-observables))
     ($skills (acquired-skills))
     ($senses (discovered-senses))
   )
   (structural-knowledge-report
     (relationships $rels)
     (concepts $concepts)
     (invented-observables $observables)
     (acquired-skills $skills)
     (discovered-senses $senses)
     (health $health))))

; "Am I getting better at understanding this domain?"
(= (am-i-improving?)
   (let* (
     ($growth (observable-growth))
     ($n-promoted (length (all-promoted-observables)))
     ($n-skills (length (all-chunked-actions)))
     ($n-senses (length (all-perceptual-fields)))
     ($n-demoted (length (demotion-history)))
     ($n-promoted-total (length (promotion-history)))
   )
   (if (> (+ $n-promoted (+ $n-skills $n-senses)) 0)
       (improving
         (net-growth (- $n-promoted-total $n-demoted))
         (current-active
           (observables $n-promoted)
           (skills $n-skills)
           (senses $n-senses)))
       (if (> $n-promoted-total 0)
           (tried-but-reverted $n-promoted-total $n-demoted)
           (not-yet-growing)))))


; =============================================================================
; SECTION XVII: EXTENDED STRUCTURAL COMPLEXITY
; =============================================================================

(= (structural-complexity-v2)
   (let* (
     ($n-links (passive-model-count))
     ($n-latents (latent-variable-count))
     ($n-suspicions (length (all-suspicion-links)))
     ($n-promoted (length (all-promoted-observables)))
     ($n-chunks (length (all-chunked-actions)))
     ($n-percepts (length (all-perceptual-fields)))
   )
   (complexity-report
     (passive-models $n-links)
     (latent-variables $n-latents)
     (suspicion-links $n-suspicions)
     (promoted-observables $n-promoted)
     (chunked-actions $n-chunks)
     (perceptual-fields $n-percepts))))


; =============================================================================
; END OF SELF-MODEL (v4 — Phases 3-5 lifecycle self-knowledge)
; =============================================================================


; =============================================================================
; SECTION XVIII: GROUNDING INTROSPECTION
; =============================================================================
;
; The system tracks what it can and cannot explain about its own learned
; concepts. grounding-completeness is a self-observable: low completeness
; creates EFE pressure toward clarification. "I don't know" emerges from
; actual grounding failure, not from an engineered response.
;
; These functions delegate to grounding_hypotheses.metta for the data;
; self_model.metta provides the meta-cognitive interpretation layer.

; --- Core Grounding Queries ---

; What fraction of active learned atoms have semantic grounding?
(= (grounding-coverage)
   (grounding-completeness-ratio))

; Which active learned atoms lack any grounding hypothesis?
(= (ungrounded-atoms)
   (ungrounded-learned-atoms))

; Which generator(s) provided grounding for a specific atom?
(= (grounding-sources $atom)
   (let $result (collapse (match &state (grounding-hyp $atom $gen $cycle) $gen))
     (if (== $result ()) (no-grounding $atom) $result)))

; Can the system explain a specific learned concept?
(= (can-explain? $atom)
   (let $groundings (collapse (match &state (obs-grounding $atom $rel $target) ($rel $target)))
     (not (== $groundings ()))))

; What does the system think a learned concept means?
(= (explain-concept $atom)
   (let* (
     ($groundings (collapse (match &state (obs-grounding $atom $rel $target) ($rel $target))))
     ($semantic (collapse (match &state (obs-semantic $atom $desc) $desc)))
     ($sources (grounding-sources $atom))
   )
   (if (== $groundings ())
       (cannot-explain $atom (no-grounding-hypotheses))
       (explanation $atom
         (meaning $groundings)
         (description $semantic)
         (sources $sources)))))

; --- Epistemic Honesty ---

; Count of concepts the system cannot currently explain
(= (unexplained-count)
   (length (ungrounded-atoms)))

; Is the system's self-understanding complete?
(= (self-understanding-complete?)
   (let $ratio (grounding-coverage)
     (> $ratio 0.95)))


; =============================================================================
; SECTION XIX: ABDUCTIVE SELF-KNOWLEDGE
; =============================================================================
;
; The system tracks which of its beliefs were created by abductive inference
; (hypothesized hidden causes) vs direct observation. This enables it to
; report its own hypotheses and distinguish confirmed from speculative beliefs.

; Which beliefs were created by abduction?
(= (abductive-hypotheses)
   (collapse (match &state (belief-source $obs abduced $cycle) $obs)))

; How many active hypotheses exist?
(= (hypothesis-count)
   (length (abductive-hypotheses)))

; Is a specific belief abductively sourced?
(= (is-hypothesis? $obs)
   (let $result (collapse (match &state (belief-source $obs abduced $c) True))
     (not (== $result ()))))

; What is the system currently investigating?
; (Abduced beliefs with low precision — the Sherlock Holmes targets)
(= (active-investigations)
   (let $hyps (abductive-hypotheses)
     (filter-low-precision $hyps)))

(= (filter-low-precision ()) ())
(= (filter-low-precision ($obs . $rest))
   (let $prec (match &state (belief $obs $v $p) $p)
     (if (< $prec 0.3)
         ($obs . (filter-low-precision $rest))
         (filter-low-precision $rest))))


; =============================================================================
; END OF SELF-MODEL (v6 — abductive self-knowledge)
; =============================================================================
