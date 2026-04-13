; =============================================================================
; METTA COGNITIVE CORE - POLICY-BASED EFE (v5.2)
; =============================================================================

; =============================================================================
; SECTION I: NON-DETERMINISTIC ACTION GENERATOR
; =============================================================================
;
; any-action is a DYNAMIC REGISTRY for nondeterministic policy generation.
; Initial entries are derived from base-actions (defined in actions.metta).
; Chunked actions are added/removed dynamically by atom_lifecycle.metta.
;
; LOAD ORDER DEPENDENCY: actions.metta must load before this module
; so that (base-actions) resolves.

; Register each element of a list as an any-action definition
(= (register-actions-for-policy! ()) done)
(= (register-actions-for-policy! ($a . $rest))
   (sequential
     done
     (register-actions-for-policy! $rest)))

; Populate initial any-action entries from the canonical action list
!(register-actions-for-policy! (base-actions))

; =============================================================================
; SECTION II: POLICY GENERATION
; =============================================================================

(: Policy Type)
(: Nil Policy)
(: Cons (-> Action Policy Policy))

(= (gen-policy 0) Nil)
(= (gen-policy $n)
   (Cons (any-action) (gen-policy (- $n 1))))

(= (policy-length Nil) 0)
(= (policy-length (Cons $a $rest)) (+ 1 (policy-length $rest)))

(= (policy-head (Cons $a $rest)) $a)
(= (policy-tail (Cons $a $rest)) $rest)

; =============================================================================
; SECTION III: SIMULATED BELIEF STATE
; =============================================================================

(: SimState Type)
(: sim-state (-> SimBeliefs SimState))

(: SimBeliefs Type)
(: SNil SimBeliefs)
(: SCons (-> Observable Float Float SimBeliefs SimBeliefs))

(= (current-sim-state)
   (sim-state (beliefs-to-sim)))

(= (beliefs-to-sim)
   ; FIX: Beliefs live in &state
   (let $bs (collapse (match &state (belief $o $v $p) ($o $v $p)))
     (list-to-sim $bs)))

(= (list-to-sim ()) SNil)
(= (list-to-sim (($o $v $p) . $rest))
   (SCons $o $v $p (list-to-sim $rest)))

; =============================================================================
; SECTION IV: SIMULATED STATE EVOLUTION
; =============================================================================

(= (apply-action $action (sim-state $beliefs))
   (sim-state (apply-to-beliefs $action $beliefs)))

(= (apply-to-beliefs $action SNil) SNil)
(= (apply-to-beliefs $action (SCons $o $v $p $rest))
   (let* (
     ($vd (get-value-delta $action $o))
     ($pd (get-prec-delta $action $o))
     ($new-v (+ $v $vd))
     ($new-p (clamp (+ $p $pd) 0.05 0.95))
   )
   (SCons $o $new-v $new-p (apply-to-beliefs $action $rest))))

; =============================================================================
; SECTION V: SIMULATED EFE
; =============================================================================

(= (sim-efe $action (sim-state $beliefs))
   (let* (
     ($error-sum (sim-error-sum $action $beliefs))
     ($cost (get-action-cost $action))
     ($obs-ig (sim-obs-info-gain $action $beliefs))
     ($model-ig (compute-model-info-gain $action)) 
     ($total-ig (+ $obs-ig $model-ig))
     ($ew (epistemic-weight))                        
     ($v-effect (sim-viability-effect $action $beliefs))
     ($efe (+ (- (+ $error-sum $cost) (* $ew $total-ig)) $v-effect))
   )
   $efe))

; =============================================================================
; SECTION VI: SIMULATED ERROR (confidence-weighted)
; =============================================================================

(= (sim-error-sum $action SNil) 0.0)
(= (sim-error-sum $action (SCons $o $v $p $rest))
   (+ (sim-single-error $action $o $v $p)
      (sim-error-sum $action $rest)))

(= (sim-single-error $action $obs $bval $bprec)
   (let* (
     ($conf (get-model-confidence $action $obs))
     ($raw-vd (get-raw-value-delta $action $obs))
     ($pdelta (get-prec-delta $action $obs))

     ($new-bprec (clamp (+ $bprec $pdelta) 0.1 1.0))
     ($pref-val (get-preferred-value $obs $bval))

     ($predicted-val-right (+ $bval $raw-vd))
     ($error-right (abs (- $predicted-val-right $pref-val)))  
     ($weighted-right (* $error-right $new-bprec))

     ($error-wrong (abs (- $bval $pref-val)))              
     ($weighted-wrong (* $error-wrong $bprec))
   )
   (+ (* $conf $weighted-right) (* (- 1.0 $conf) $weighted-wrong))))

; =============================================================================
; SECTION VII: SIMULATED INFORMATION GAIN
; =============================================================================

(= (sim-obs-info-gain $action SNil) 0.0)
(= (sim-obs-info-gain $action (SCons $o $v $p $rest))
   (let* (
     ($pdelta (get-raw-prec-delta $action $o))
     ($uncertainty (- 1.0 $p))
     ($effective-gain (* (positive-part $pdelta) $uncertainty))
   )
   (+ $effective-gain (sim-obs-info-gain $action $rest))))

; =============================================================================
; SECTION VIII: SIMULATED VIABILITY EFFECT
; =============================================================================

(= (sim-viability-effect $action $beliefs)
   (let* (
     ($b1 (collapse (match &state (viability-bound $obs $min $max) ($obs $min $max))))
     ($b2 (collapse (match &state (viability-bound $obs $min $max) ($obs $min $max))))
     ($bounds (list-append $b1 $b2))
   )
   (fold-sim-viability $action $beliefs $bounds)))

(= (fold-sim-viability $action $beliefs ()) 0.0)
(= (fold-sim-viability $action $beliefs (($obs $min $max) . $rest))
   (+ (single-sim-viability $action $beliefs $obs $min $max)
      (fold-sim-viability $action $beliefs $rest)))

(= (single-sim-viability $action $beliefs $obs $min $max)
   (let* (
     ($bval (sim-belief-value $beliefs $obs))
     ($bprec (sim-belief-precision $beliefs $obs))
     ($current-pressure (compute-boundary-pressure $obs $bval $bprec $min $max))

     ($vdelta (get-value-delta $action $obs))
     ($new-val (+ $bval $vdelta))
     ($pdelta (get-prec-delta $action $obs))
     ($new-prec (clamp (+ $bprec $pdelta) 0.01 1.0))
     ($new-pressure (compute-boundary-pressure $obs $new-val $new-prec $min $max))

     ($delta-pressure (- $new-pressure $current-pressure))
     ($scale (max $current-pressure $new-pressure))

     ($conf (get-model-confidence $action $obs))
     ($uncertainty-risk (* $scale (* (- 1.0 $conf) 0.1)))
   )
   (+ (* $scale $delta-pressure) $uncertainty-risk)))

(= (sim-belief-value SNil $obs) 0.5)  
(= (sim-belief-value (SCons $o $v $p $rest) $obs)
   (if (== $o $obs)
       $v
       (sim-belief-value $rest $obs)))

(= (sim-belief-precision SNil $obs) 0.5)
(= (sim-belief-precision (SCons $o $v $p $rest) $obs)
   (if (== $o $obs)
       $p
       (sim-belief-precision $rest $obs)))

; =============================================================================
; SECTION IX: CUMULATIVE POLICY EFE
; =============================================================================

(= discount 0.9)  

(= (policy-efe Nil $state) 0.0)
(= (policy-efe (Cons $action $rest) $state)
   (let* (
     ($step-efe (sim-efe $action $state))
     ($next-state (apply-action $action $state))
     ($future-efe (policy-efe $rest $next-state))
   )
   (+ $step-efe (* discount $future-efe))))

; =============================================================================
; SECTION X: VIABILITY PRUNING
; =============================================================================

(= (violates-viability? (sim-state $beliefs))
   (check-sim-viability $beliefs))

(= (check-sim-viability SNil) False)
(= (check-sim-viability (SCons $o $v $p $rest))
   (let $violation (check-bound $o $v)
     (if $violation True (check-sim-viability $rest))))

(= (check-bound $o $v)
   (let* (
     ($r1 (match &state (viability-bound $o $min $max) (or (< $v $min) (> $v $max))))
     ($r2 (match &state (viability-bound $o $min $max) (or (< $v $min) (> $v $max))))
   )
   (if (not (== $r1 ())) $r1
       (if (not (== $r2 ())) $r2 False))))

(= (policy-efe-safe Nil $state) 0.0)
(= (policy-efe-safe (Cons $action $rest) $state)
   (let $next-state (apply-action $action $state)
     (if (violates-viability? $next-state)
         999.0  
         (let* (
           ($step-efe (sim-efe $action $state))
           ($future-efe (policy-efe-safe $rest $next-state))
         )
         (+ $step-efe (* discount $future-efe))))))

; =============================================================================
; SECTION XI: POLICY SELECTION
; =============================================================================

(= (evaluate-policy $horizon)
   (let* (
     ($policy (gen-policy $horizon))
     ($state (current-sim-state))
     ($efe (policy-efe-safe $policy $state)) 
   )
   (policy-eval $policy $efe)))

(= (all-policy-evals $horizon)
   (collapse (evaluate-policy $horizon)))

(= (best-of-evals ((policy-eval $p $e))) (policy-eval $p $e))
(= (best-of-evals ((policy-eval $p1 $e1) . $rest))
   (let (policy-eval $p2 $e2) (best-of-evals $rest)
     (if (< $e1 $e2)
         (policy-eval $p1 $e1)
         (policy-eval $p2 $e2))))

(= (select-best-policy $horizon)
   (let $evals (all-policy-evals $horizon)
     (best-of-evals $evals)))

(= (select-action-with-horizon $horizon)
   (let (policy-eval $policy $efe) (select-best-policy $horizon)
     (policy-head $policy)))

(= (select-action) (select-action-with-horizon 3))

; =============================================================================
; SECTION XII: TRACED SELECTION
; =============================================================================

(= (select-action-traced $horizon)
   (let* (
     ($evals (all-policy-evals $horizon))
     ((policy-eval $best-policy $best-efe) (best-of-evals $evals))
     ($first-action (policy-head $best-policy))
     ($pressure (get-viability-pressure))
     ($ew (epistemic-weight))
   )
   (trace
     (horizon $horizon)
     (policies-evaluated (length-evals $evals))
     (best-policy $best-policy)
     (best-efe $best-efe)
     (selected-action $first-action)
     (viability-pressure $pressure)
     (epistemic-weight $ew)
     (model-confidences (build-model-confidences (base-actions)))
     (reason "first action of minimum cumulative EFE policy"))))

(= (length-evals ()) 0)
(= (length-evals ($x . $rest)) (+ 1 (length-evals $rest)))

; =============================================================================
; END OF POLICY-BASED EFE (v5.2)
; =============================================================================
