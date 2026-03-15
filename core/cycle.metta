; =============================================================================
; METTA COGNITIVE CORE - COGNITIVE CYCLE (v4 - 1-Cycle Lag Fixed)
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

(config action-selection-mode policy)

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
; SECTION I-B: METABOLIC CAPITAL (Boundary Condition Fix)
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
; SECTION II: THE COGNITIVE CYCLE (v4 - Reordered to fix lag)
; =============================================================================

(= (cognitive-cycle!)
   (if (not (system-operational?))
       (cycle-aborted (get-system-status))
   (let* (
     ; 1. LEARN from previous action
     ($learning (learn-from-last-action!))

     ; 2. Compute prediction errors & record surprise
     ($errors (all-prediction-errors))
     ($total-error (sum-weighted-errors $errors))
     ($timestamp (get-cycle-count))
     ($surprises (compute-all-surprises))
     ($traces (record-all-error-traces! $surprises $timestamp))
     ($cleanup (clear-old-traces! $timestamp))
     ($record (record-error! $total-error $timestamp))

     ; 3. Derive affect 
     ($affect (get-affect-state))
     ($urgency (compute-urgency))
     ($confidence (compute-confidence))
     ($structural (structure-learning-step! $timestamp))

     ; 4. UPDATE BELIEFS (Must happen BEFORE action selection)
     ($belief-learning (selective-update-all-beliefs!))

     ; 5. SELECT ACTION (Now sees the reality of the updated beliefs)
     ($action (choose-action))

     ; 6. SNAPSHOT STATE (Records the post-update state for next cycle's learning)
     ($snapshot (snapshot-pre-action-state! $action))
     
     ; 7. Reward structure
     ($structural-reward (reward-all-structure! $errors $timestamp))

     ; 8. Execute action (gated by pause check)
     ($exec (if (system-paused?)
                (action-paused $action)
                (execute-action! $action)))

     ; 9. Increment cycle counter & Watchdog
     ($count (increment-cycle-count!))
     ($watchdog (verify-immutability!))
   )
   ; Return complete cycle state
   (cycle-result
     (cycle $count)
     (total-error $total-error)
     (affect $affect)
     (urgency $urgency)
     (confidence $confidence)
     (action $action)
     (model-learning $learning)
     (structure $structural)
     (watchdog $watchdog)))))

(= (step!)
   (if (not (system-operational?))
       (cycle-aborted (get-system-status))
   (let* (
     ($learning (learn-from-last-action!))
     ($timestamp (get-cycle-count))
     ($surprises (compute-all-surprises))
     ($traces (record-all-error-traces! $surprises $timestamp))
     ($cleanup (clear-old-traces! $timestamp))
     ($structural (structure-learning-step! $timestamp))

     ; UPDATE BELIEFS
     ($belief-learning (selective-update-all-beliefs!))

     ; CHOOSE ACTION
     ($action (choose-action))
     ($snapshot (snapshot-pre-action-state! $action))

     ($reward (reward-all-structure! (all-prediction-errors) $timestamp))
     ($exec (if (system-paused?)
                (action-paused $action)
                (execute-action! $action)))
     ($count (increment-cycle-count!))
     ($watchdog (verify-immutability!))
   )
   $action)))

; =============================================================================
; SECTION II-B: THE COGNITIVE CYCLE v2 (Phases 3-5 integration)
; =============================================================================

(= (cognitive-cycle-v2!)
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
     ($structural-reward (reward-all-structure-v2! $errors $timestamp))

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

(= (decide)
   (let* (
     ($errors (all-prediction-errors))
     ($affect (get-affect-state))
     ($action (choose-action))
   )
   (decision $errors $affect $action)))


; =============================================================================
; SECTION III: CYCLE COUNTER
; =============================================================================

!(add-atom &state (cycle-count 0))

(= (get-cycle-count)
   (let $n (match &state (cycle-count $c) $c)
     (if (== $n ()) 0 $n)))

(= (increment-cycle-count!)
   (let $old (get-cycle-count)
     (sequential
       (remove-atom &state (cycle-count $old))
       (add-atom &state (cycle-count (+ $old 1)))
       (+ $old 1))))


; =============================================================================
; SECTION IV: EXTERNAL INTERFACE
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
     (structure (structure-summary))
     (structural-surprise (structural-surprise))
     (safety (get-system-status))))

(= (get-action) (choose-action))

(= (get-action-traced) 
   (let $mode (get-config action-selection-mode)
     (if (== $mode single)
         (select-action-myopic-traced)   
         (select-action-traced 3))))       

(= (process! $obs $val $prec $time)
   (sequential
     (inject-observation! $obs $val $prec $time)
     (step!)))

(= (process-batch! () $time) (step!))
(= (process-batch! (($obs $val $prec) . $rest) $time)
   (sequential
     (inject-observation! $obs $val $prec $time)
     (process-batch! $rest $time)))


; =============================================================================
; SECTION V: INITIALIZATION
; =============================================================================

(= (init!)
   (sequential
     (reset-cycle-count!)
     (clear-error-history!)
     (clear-snapshot!)      
     (init-metabolic-capital!)  
     (initialize-watchdog!) 
     initialized))

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


; =============================================================================
; SECTION VI: EXPLANATION INTERFACE
; =============================================================================

(= (explain)
   (let* (
     ($v (compute-valence))
     ($a (compute-arousal))
     ($d (compute-dominance))
     ($action (choose-action))
     ($pressure (viability-pressure))
     ($ew (epistemic-weight))
     ($mc (overall-model-confidence))
     ($struct (structure-summary))
   )
   (explanation
     (status (interpret-valence $v))
     (activation (interpret-arousal $a))
     (confidence (interpret-dominance $d))
     (doing $action)
     (because (explain-action-v2 $action $pressure $ew $mc))
     (structure $struct))))

(= (interpret-valence $v)
   (if (< $v -0.3) "significant prediction errors"
   (if (< $v 0.0)  "some prediction errors"
   (if (< $v 0.3)  "predictions mostly confirmed"
                   "predictions well confirmed"))))

(= (interpret-arousal $a)
   (if (> $a 0.7) "highly activated"
   (if (> $a 0.4) "moderately activated"
                  "calm")))

(= (interpret-dominance $d)
   (if (> $d 0.7) "high confidence"
   (if (> $d 0.4) "moderate confidence"
                  "uncertain")))

(= (explain-action-v2 $action $pressure $ew $mc)
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
; SECTION VIII: STRUCTURAL REWARD BRIDGE
; =============================================================================

(= (reward-all-structure! $errors $current-cycle)
   (reward-structure-from-errors! $errors $current-cycle))

(= (reward-structure-from-errors! () $cycle) done)
(= (reward-structure-from-errors!
     ((prediction-error $obs $error $abs-err $prec $weighted) . $rest) $cycle)
   (sequential
     (reward-structure-for! $obs $error $cycle)
     (reward-structure-from-errors! $rest $cycle)))


; =============================================================================
; SECTION VIII-B: STRUCTURAL REWARD v2 (coherence-aware + lifecycle)
; =============================================================================

(= (reward-all-structure-v2! $errors $current-cycle)
   (reward-structure-from-errors-v2! $errors $current-cycle $errors))

(= (reward-structure-from-errors-v2! () $cycle $all-errors) done)
(= (reward-structure-from-errors-v2!
     ((prediction-error $obs $error $abs-err $prec $weighted) . $rest)
     $cycle $all-errors)
   (sequential
     (reward-structure-for-v2! $obs $error $cycle $all-errors)
     (reward-structure-from-errors-v2! $rest $cycle $all-errors)))


; =============================================================================
; SECTION IV-B: EXTERNAL INTERFACE v2 (lifecycle-aware)
; =============================================================================

(= (get-state-v2)
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
; SECTION V-B: INITIALIZATION v2 (lifecycle-aware)
; =============================================================================

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

(= (init-v2!)
   (sequential
     (reset-cycle-count!)
     (clear-error-history!)
     (clear-snapshot!)
     (init-metabolic-capital!)  
     (initialize-watchdog!)
     (init-lifecycle!)
     initialized))


; =============================================================================
; SECTION VI-B: EXPLANATION INTERFACE v2 (lifecycle-aware)
; =============================================================================

(= (explain-v2)
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
     (status (interpret-valence $v))
     (activation (interpret-arousal $a))
     (confidence (interpret-dominance $d))
     (doing $action)
     (because (explain-action-v2 $action $pressure $ew $mc))
     (structure $struct)
     (lifecycle $life))))

; =============================================================================
; END OF COGNITIVE CYCLE
; =============================================================================