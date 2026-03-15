; =============================================================================
; METTA COGNITIVE CORE - ACTION MODEL LEARNING (v4 - Dual Space Fix)
; =============================================================================

; =============================================================================
; SECTION I: CONFIGURATION
; =============================================================================

(config model-learning-rate 0.15)
(config model-good-prediction-threshold 0.05)


; =============================================================================
; SECTION II: PRE-ACTION SNAPSHOT
; =============================================================================

(= (snapshot-pre-action-state! $action)
   (sequential
     (clear-snapshot!)
     (record-last-action! $action)
     (store-belief-snapshot!)
     snapshot-taken))

(= (record-last-action! $action)
   (let $old (collapse (match &self (last-action-taken $a) (last-action-taken $a)))
     (sequential
       (remove-list! $old)
       (add-atom &self (last-action-taken $action))
       done)))

(= (store-belief-snapshot!)
   ; FIX: Beliefs live in &state
   (let $beliefs (collapse (match &state (belief $o $v $p) ($o $v $p)))
     (store-snapshot-list $beliefs)))

(= (store-snapshot-list ()) done)
(= (store-snapshot-list (($obs $val $prec) . $rest))
   (sequential
     (add-atom &self (pre-action-belief $obs $val $prec))
     (store-snapshot-list $rest)))

(= (clear-snapshot!)
   (sequential
     (let $old-beliefs (collapse (match &self (pre-action-belief $o $v $p)
                                              (pre-action-belief $o $v $p)))
       (remove-list! $old-beliefs))
     (let $old-action (collapse (match &self (last-action-taken $a)
                                             (last-action-taken $a)))
       (remove-list! $old-action))
     cleared))

(= (remove-list! ()) done)
(= (remove-list! ($a . $rest))
   (sequential (remove-atom &self $a) (remove-list! $rest)))


; =============================================================================
; SECTION III: LEARN FROM OUTCOME
; =============================================================================

(= (learn-from-last-action!)
   (let $action-result (collapse (match &self (last-action-taken $a) $a))
     (if (== $action-result ())
         no-previous-action  
         (let $action (list-head-safe $action-result)
           (sequential
             (learn-action-models! $action)
             (clear-snapshot!)
             (action-models-learned $action))))))

(= (list-head-safe ($x . $rest)) $x)
(= (list-head-safe ()) none)

(= (learn-action-models! $action)
   (let $snapshots (collapse (match &self (pre-action-belief $obs $old-v $old-p)
                                          ($obs $old-v $old-p)))
     (learn-from-snapshots $action $snapshots)))

(= (learn-from-snapshots $action ()) done)
(= (learn-from-snapshots $action (($obs $old-v $old-p) . $rest))
   (sequential
     (learn-single-model! $action $obs $old-v $old-p)
     (learn-from-snapshots $action $rest)))

(= (learn-single-model! $action $obs $old-val $old-prec)
   ; FIX: Observations live in &state
   (let $obs-result (collapse (match &state (observation $obs $new-obs-val $obs-prec $t)
                                           ($new-obs-val $obs-prec)))
     (if (== $obs-result ())
         no-observation  
         (let ($actual-obs-val $actual-obs-prec) (list-head-safe $obs-result)
           (let* (
             ($actual-vd (- $actual-obs-val $old-val))
             ($current-prec (get-current-prec $obs $old-prec))
             ($actual-pd (- $current-prec $old-prec))
           )
           (update-action-model! $action $obs $actual-vd $actual-pd))))))

(= (get-current-prec $obs $default)
   ; FIX: Beliefs live in &state
   (let $r (match &state (belief $obs $v $p) $p)
     (if (== $r ()) $default $r)))


; =============================================================================
; SECTION IV: ACTION MODEL UPDATE RULE
; =============================================================================

(= (update-action-model! $action $obs $actual-vd $actual-pd)
   ; FIX: Leverage the space-agnostic accessors from actions.metta
   (let* (
     ($pred-vd (get-raw-value-delta $action $obs))
     ($pred-pd (get-raw-prec-delta $action $obs))
     ($conf (get-model-confidence $action $obs))
   )
   (if (== $conf 0.1) ; 0.1 is the fallback for an unknown model
       (sequential
         (add-atom &state (action-model $action $obs $actual-vd $actual-pd 0.15))
         (model-discovered $action $obs $actual-vd $actual-pd))
       (let* (
         ($vd-error (- $actual-vd $pred-vd))
         ($pd-error (- $actual-pd $pred-pd))
         ($base-lr (get-config model-learning-rate))
         ($lr (* $base-lr (- 1.1 $conf)))
         ($new-vd (+ $pred-vd (* $lr $vd-error)))
         ($new-pd (+ $pred-pd (* $lr $pd-error)))
         ($total-model-error (+ (abs $vd-error) (abs $pd-error)))
         ($new-conf (update-model-confidence $conf $total-model-error))
       )
       (sequential
         ; Clean up old models from BOTH spaces to prevent phantom duplicates
         (remove-atom &state (action-model $action $obs $pred-vd $pred-pd $conf))
         (remove-atom &state (action-model $action $obs $pred-vd $pred-pd $conf))
         ; Save newly learned model to &state
         (add-atom &state (action-model $action $obs $new-vd $new-pd $new-conf))
         (model-updated $action $obs (predicted $pred-vd $pred-pd $conf) (actual $actual-vd $actual-pd) (updated $new-vd $new-pd $new-conf)))))))

(= (update-model-confidence $old-conf $model-error)
   (let* (
     ($threshold (get-config model-good-prediction-threshold))
     ($delta (if (< $model-error $threshold)
                 0.02                               
                 (* -0.1 (- $model-error $threshold)))) 
     ($floor (get-config precision-floor))
     ($ceiling (get-config precision-ceiling))
   )
   (clamp (+ $old-conf $delta) $floor $ceiling)))


; =============================================================================
; SECTION V: MODEL DISCOVERY
; =============================================================================

(= (has-action-model? $action $obs)
   (let* (
     ($r1 (collapse (match &state (action-model $action $obs $vd $pd $c) found)))
     ($r2 (collapse (match &state (action-model $action $obs $vd $pd $c) found)))
   )
   (if (and (== $r1 ()) (== $r2 ())) False True)))


; =============================================================================
; SECTION VI: LEARNING DIAGNOSTICS
; =============================================================================

(= (all-model-confidences)
   (collapse (match &state (action-model $a $o $vd $pd $c)
     (model-conf $a $o $c))))

(= (overall-model-confidence)
   (let $confs (collapse (match &state (action-model $a $o $vd $pd $c) $c))
     (safe-avg-conf $confs)))

(= (safe-avg-conf ()) 0.0)
(= (safe-avg-conf ($x . $rest))
   (/ (+ $x (sum-conf $rest)) (+ 1 (count-conf $rest))))
(= (sum-conf ()) 0.0)
(= (sum-conf ($x . $rest)) (+ $x (sum-conf $rest)))
(= (count-conf ()) 0)
(= (count-conf ($x . $rest)) (+ 1 (count-conf $rest)))

(= (least-confident-models)
   (collapse (match &state (action-model $a $o $vd $pd $c)
     (if (< $c 0.4) (uncertain-model $a $o $c) (empty)))))

(= (most-learned-models)
   (collapse (match &state (action-model $a $o $vd $pd $c)
     (if (> $c 0.7) (learned-model $a $o $vd $pd $c) (empty)))))

(= (action-model-report $action)
   (collapse (match &state (action-model $action $obs $vd $pd $c)
     (model $obs (value-delta $vd) (prec-delta $pd) (confidence $c)))))


; =============================================================================
; SECTION VII: COST LEARNING (optional extension)
; =============================================================================

(= (update-action-cost! $action $observed-cost)
   (let $old-cost-result (collapse (match &state (action-cost $action $predicted-cost $conf) 
                                         ($predicted-cost $conf)))
     (if (not (== $old-cost-result ()))
         (let ($predicted-cost $conf) (list-head-safe $old-cost-result)
           (let* (
             ($error (- $observed-cost $predicted-cost))
             ($lr (* (get-config model-learning-rate) (- 1.0 $conf)))
             ($new-cost (+ $predicted-cost (* $lr $error)))
             ($new-conf (update-model-confidence $conf (abs $error)))
           )
           (sequential
             (remove-atom &state (action-cost $action $predicted-cost $conf))
             (add-atom &state (action-cost $action $new-cost $new-conf))
             (cost-updated $action $new-cost $new-conf))))
         no-cost-model)))


; =============================================================================
; END OF ACTION MODEL LEARNING
; =============================================================================