; =============================================================================
; METTA COGNITIVE CORE - COGNITIVE CYCLE
; =============================================================================
;
; The main cognitive loop. One version. No v1/v2 split.
;
; PIPELINE:
;   1. Learn from previous action (action_learning)
;   2. Compute prediction errors & surprises
;   3. Derive affect (valence/arousal/dominance)
;   4. Run reasoning triad (abduction, grounding, analogy)
;   5. Update beliefs
;   6. Select action (dispatched to fractal/myopic/policy)
;   7. Structure learning (Phases 1-5)
;   8. Snapshot state for next cycle's learning
;   9. Reward structure, execute action, increment cycle, watchdog
;
; Version: 5.0.0 — version collapse (METTA_CLEANUP.md)
; =============================================================================

!(import! &self foundations)
!(import! &self affect)
!(import! &self beliefs)
!(import! &self actions)
!(import! &self action_learning)
!(import! &self policy_efe)
!(import! &self safety)
!(import! &self structure_learning)
!(import! &self atom_lifecycle)


; =============================================================================
; ACTION SELECTION MODE
; =============================================================================

(config action-selection-mode fractal)

(= (choose-action)
   (let $mode (get-config action-selection-mode)
     (if (== $mode single)
         (select-action-myopic)        
         (if (== $mode fractal)
             (select-action-fractal)   
             (select-action)))))       


; =============================================================================
; SECTION I: OBSERVATION INTERFACE
; =============================================================================

(= (inject-observation! $obs $val $prec $time)
   (sequential
     (clear-observation! $obs)
     (add-atom &state (observation $obs $val $prec $time))
     (ratchet-precision! $obs)                  
     (observation-injected $obs $val $prec $time)))

(= (clear-observation! $obs)
   (let $old (collapse (match &state (observation $obs $v $p $t) 
                                    (observation $obs $v $p $t)))
     (remove-atoms $old)))

(= (remove-atoms ()) done)
(= (remove-atoms ($a . $rest)) 
   (sequential (remove-atom &state $a) (remove-atoms $rest)))


; =============================================================================
; SECTION II: METABOLIC CAPITAL
; =============================================================================

(= (init-metabolic-capital!)
   (sequential
     (let $old (collapse (match &state (metabolic-capital $c) (metabolic-capital $c)))
       (remove-atom-list-safe! $old))
     (add-atom &state (metabolic-capital (get-config initial-metabolic-capital)))
     capital-initialized))

(= (earn-capital! $amount)
   (match &state (metabolic-capital $current)
     (let $new (min (+ $current $amount) (get-config capital-cap))
       (sequential
         (remove-atom &state (metabolic-capital $current))
         (add-atom &state (metabolic-capital $new))
         (capital-earned $amount $current $new)))))

(= (can-afford-abduction?)
   (match &state (metabolic-capital $current)
     (>= $current (get-config abduction-capital-cost))))

(= (spend-abduction-capital!)
   (match &state (metabolic-capital $current)
     (let $new (- $current (get-config abduction-capital-cost))
       (sequential
         (remove-atom &state (metabolic-capital $current))
         (add-atom &state (metabolic-capital $new))
         (capital-spent abduction $current $new)))))

(= (get-metabolic-capital)
   (match &state (metabolic-capital $c) $c))


; =============================================================================
; SECTION III: THE COGNITIVE CYCLE
; =============================================================================

(= (cognitive-cycle!)
   (if (not (system-operational?))
       (cycle-aborted (get-system-status))
   (let* (
     ($learning (learn-from-last-action!))
     ($errors (all-prediction-errors))
     ($total-error (sum-weighted-errors $errors))
     ($timestamp (get-cycle-count))
     ($surprises (compute-all-surprises))
     ($traces (record-all-error-traces! $surprises $timestamp))
     ($cleanup (clear-old-traces! $timestamp))
     ($record (record-error! $total-error $timestamp))

     ($affect (get-affect-state))
     ($urgency (compute-urgency))
     ($confidence (compute-confidence))

     ($abduction (abductive-step! $timestamp))
     ($grounding (grounding-step! $timestamp))
     ($analogy (analogy-blending-step! $timestamp))

     ; UPDATE BELIEFS
     ($belief-learning (selective-update-all-beliefs!))

     ; SELECT ACTION
     ($action (choose-action))
     
     ($selected-policy ($action))
     ($structural (structure-learning-step-v2! $timestamp $errors $selected-policy))

     ($snapshot (snapshot-pre-action-state! $action))
     ($structural-reward (reward-all-structure! $errors $timestamp))

     ($exec (if (system-paused?)
                (action-paused $action)
                (execute-action! $action)))

     ($count (increment-cycle-count!))
     ($watchdog (verify-immutability!))
     ($lifecycle (lifecycle-atom-count))
   )
   (cycle-result
     (cycle $count) (total-error $total-error) (affect $affect)
     (urgency $urgency) (confidence $confidence) (action $action)
     (model-learning $learning) (structure $structural)
     (abduction $abduction) (grounding $grounding)
     (analogy $analogy) (lifecycle $lifecycle) (watchdog $watchdog)))))


; =============================================================================
; SECTION III-B: SLIM COGNITIVE CYCLE (game hot path)
; =============================================================================
;
; Keeps: action learning, belief update, affect, error recording, watchdog.
; Skips: action selection (Python planner), structure learning (Python
; Perception), abduction/grounding/analogy (evaluator limitations),
; snapshot (Python server does this after planner runs).
;
; Target: <20ms. The full cycle is ~55ms with redundant subsystems.

(= (cognitive-cycle-slim!)
   (if (not (system-operational?))
       (cycle-aborted (get-system-status))
   (let* (
     ; 1. Learn from what happened last cycle
     ($learning (learn-from-last-action!))

     ; 2. Prediction errors (needed for affect + error history)
     ($errors (all-prediction-errors))
     ($total-error (sum-weighted-errors $errors))
     ($timestamp (get-cycle-count))
     ($record (record-error! $total-error $timestamp))

     ; 3. Affect (dashboard needs this)
     ($affect (get-affect-state))

     ; 4. Belief update (the essential step — Bayesian update from observations)
     ($belief-learning (selective-update-all-beliefs!))

     ; 5. Bookkeeping
     ($count (increment-cycle-count!))
     ($watchdog (verify-immutability!))
   )
   (slim-cycle-result
     (cycle $count) (total-error $total-error)
     (affect $affect) (model-learning $learning)
     (watchdog $watchdog)))))


; =============================================================================
; SECTION IV: INITIALIZATION
; =============================================================================

(= (init!)
   (sequential
     (reset-cycle-count!)
     (clear-error-history!)
     (clear-snapshot!)
     (init-metabolic-capital!)  
     (initialize-watchdog!)
     (init-lifecycle!)
     initialized))

(= (init-lifecycle!)
   (sequential
     (let $old-traces (collapse (match &state (policy-trace $c $p)
                                             (policy-trace $c $p)))
       (remove-atom-list-safe! $old-traces))
     (let $old-unexp (collapse (match &state (unexplained-error-ratio $c $r)
                                            (unexplained-error-ratio $c $r)))
       (remove-atom-list-safe! $old-unexp))
     (let $old-flag (collapse (match &state (perceptual-expansion-needed $v)
                                           (perceptual-expansion-needed $v)))
       (remove-atom-list-safe! $old-flag))
     lifecycle-initialized))

(= (reset-cycle-count!)
   (sequential
     (remove-atom &state (cycle-count (get-cycle-count)))
     (add-atom &state (cycle-count 0))
     done))

(= (clear-error-history!)
   (sequential
     (remove-atom &state (error-history (get-error-history)))
     (add-atom &state (error-history ()))
     done))

!(add-atom &state (cycle-count 0))

(= (get-cycle-count)
   (match &state (cycle-count $c) $c))

(= (increment-cycle-count!)
   (let $old (get-cycle-count)
     (sequential
       (remove-atom &state (cycle-count $old))
       (add-atom &state (cycle-count (+ $old 1)))
       (+ $old 1))))


; =============================================================================
; SECTION V: STATE QUERY
; =============================================================================

(= (get-state)
   (state
     (cycle (get-cycle-count))
     (affect (get-affect-state))
     (urgency (compute-urgency))
     (confidence (compute-confidence))
     (curiosity (compute-curiosity))
     (action (get-current-action))
     (viability (viability-pressure))
     (epistemic-weight (epistemic-weight))
     (model-confidence (overall-model-confidence))
     (metabolic-capital (get-metabolic-capital))  
     (structure (structure-summary))
     (lifecycle (lifecycle-summary))
     (structural-surprise (structural-surprise))
     (perceptual-expansion (perceptual-expansion-needed?))
     (safety (get-system-status))))


; =============================================================================
; SECTION VI: EXPLANATION INTERFACE
; =============================================================================

(= (explain)
   (let* (
     ($v (compute-valence))
     ($a (compute-arousal))
     ($d (compute-dominance))
     ($action (choose-action))
     ($pressure (get-viability-pressure))
     ($ew (epistemic-weight))
     ($mc (overall-model-confidence))
     ($struct (structure-summary))
     ($life (lifecycle-atom-count))
   )
   (explanation
     (status $v)
     (activation $a)
     (confidence $d)
     (doing $action)
     (because (explain-action $action $pressure $ew $mc))
     (structure $struct)
     (lifecycle $life))))

(= (explain-action $action $pressure $ew $mc)
   (if (> $pressure 0.3)
       (action-reason $action "viability pressure" (pressure $pressure))
       (if (> $ew 0.5)
           (action-reason $action "information gathering" 
             (epistemic-weight $ew) (model-confidence $mc))
           (action-reason $action "lowest expected error"
             (epistemic-weight $ew)))))


; =============================================================================
; SECTION VII: VIABILITY PRESSURE (delegated to actions module)
; =============================================================================

(= (viability-pressure) (get-viability-pressure))


; =============================================================================
; SECTION VIII: STRUCTURAL REWARD
; =============================================================================

(= (reward-all-structure! $errors $current-cycle)
   (reward-structure-from-errors! $errors $current-cycle $errors))

(= (reward-structure-from-errors! () $cycle $all-errors) done)
(= (reward-structure-from-errors!
     ((prediction-error $obs $error $abs-err $prec $weighted) . $rest)
     $cycle $all-errors)
   (sequential
     (reward-structure-for-v2! $obs $error $cycle $all-errors)
     (reward-structure-from-errors! $rest $cycle $all-errors)))


; =============================================================================
; END OF COGNITIVE CYCLE (v5.0.0)
; =============================================================================
