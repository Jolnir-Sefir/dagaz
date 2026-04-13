; =============================================================================
; METTA COGNITIVE CORE - ACTIONS v4.3 (Integrated Space & Physics Fixes)
; =============================================================================
;
; INVARIANT: select action minimizing Expected Free Energy
;   EFE(a) = confidence_weighted_error(a) + cost(a) 
;            - epistemic_weight × (obs_info_gain(a) + model_info_gain(a))
;            + viability_effect(a)
;
; =============================================================================

; =============================================================================
; SECTION I: ACTION TYPES
; =============================================================================

(: Action Type)
(: wait Action)
(: observe Action)
(: retreat Action)
(: reflect Action)
(: investigate (-> Observable Action))
(: ground-concept (-> Observable Action))


; =============================================================================
; SECTION II: ACTION OUTCOME MODELS (with confidence)
; =============================================================================
;
; Schema: (action-model <action> <observable> <value-delta> <prec-delta> <confidence>)

; --- WAIT: Minimal change, slight uncertainty growth ---
!(add-atom &state (action-model wait power-level -0.005 -0.01 0.3))
!(add-atom &state (action-model wait terrain-roughness 0.0 -0.02 0.3))
!(add-atom &state (action-model wait ambient-temperature 0.0 -0.02 0.3))
!(add-atom &state (action-model wait threat-level 0.0 -0.01 0.3))

; --- OBSERVE: No value change, significant precision gain ---
!(add-atom &state (action-model observe power-level 0.0 0.05 0.3))
!(add-atom &state (action-model observe terrain-roughness 0.0 0.08 0.3))
!(add-atom &state (action-model observe ambient-temperature 0.0 0.08 0.3))
!(add-atom &state (action-model observe threat-level 0.0 0.06 0.3))

; --- RETREAT: Move toward safer conditions ---
!(add-atom &state (action-model retreat power-level -0.03 0.02 0.3))
!(add-atom &state (action-model retreat terrain-roughness -0.1 0.05 0.3))
!(add-atom &state (action-model retreat threat-level -0.2 0.04 0.3))
!(add-atom &state (action-model retreat ambient-temperature 0.0 0.02 0.3))

; --- REFLECT: Internal introspection — precision gain on self-model ---
; External precision decays (attention directed inward), slight power drain.
; Precision gains on self-competence and grounding-completeness are
; comparable to observe's gains on terrain (0.08). This ensures reflect
; wins when self-model uncertainty is the dominant error source, but
; yields to observe/retreat when environmental uncertainty or viability
; pressure is high.
!(add-atom &state (action-model reflect power-level -0.01 -0.02 0.3))
!(add-atom &state (action-model reflect terrain-roughness 0.0 -0.02 0.3))
!(add-atom &state (action-model reflect ambient-temperature 0.0 -0.02 0.3))
!(add-atom &state (action-model reflect threat-level 0.0 -0.02 0.3))
!(add-atom &state (action-model reflect self-competence 0.01 0.08 0.3))
!(add-atom &state (action-model reflect grounding-completeness 0.01 0.08 0.3))


; =============================================================================
; SECTION III: ACTION COSTS
; =============================================================================

!(add-atom &state (action-cost wait 0.01 0.5))
!(add-atom &state (action-cost observe 0.05 0.5))
!(add-atom &state (action-cost retreat 0.12 0.5))
!(add-atom &state (action-cost reflect 0.02 0.5))


; =============================================================================
; SECTION IV: DUAL-SPACE MODEL ACCESSORS (confidence-weighted)
; =============================================================================

; Raw model values (before confidence weighting)
(= (get-raw-value-delta $action $obs)
   (let $r1 (match &state (action-model $action $obs $vd $pd $c) $vd)
     (if (== $r1 ())
         (let $r2 (match &state (action-model $action $obs $vd $pd $c) $vd)
           (if (== $r2 ()) 0.0 $r2))
         $r1)))

(= (get-raw-prec-delta $action $obs)
   (let $r1 (match &state (action-model $action $obs $vd $pd $c) $pd)
     (if (== $r1 ())
         (let $r2 (match &state (action-model $action $obs $vd $pd $c) $pd)
           (if (== $r2 ()) 0.0 $r2))
         $r1)))

(= (get-model-confidence $action $obs)
   (let $r1 (match &state (action-model $action $obs $vd $pd $c) $c)
     (if (== $r1 ())
         (let $r2 (match &state (action-model $action $obs $vd $pd $c) $c)
           (if (== $r2 ()) 0.1 $r2))
         $r1)))

(= (get-action-cost $action)
   (let $r1 (match &state (action-cost $action $cost $conf) $cost)
     (if (== $r1 ())
         (let $r2 (match &state (action-cost $action $cost $conf) $cost)
           (if (== $r2 ()) 0.05 $r2))
         $r1)))

; Effective deltas: raw × confidence
(= (get-value-delta $action $obs)
   (* (get-model-confidence $action $obs) (get-raw-value-delta $action $obs)))

(= (get-prec-delta $action $obs)
   (* (get-model-confidence $action $obs) (get-raw-prec-delta $action $obs)))

; Special handling for investigate
(= (get-raw-prec-delta (investigate $target) $obs)
   (if (== $target $obs) 0.15 -0.01))
(= (get-raw-value-delta (investigate $target) $obs) 0.0)
(= (get-model-confidence (investigate $target) $obs) 0.3)

; Special handling for ground-concept
; Targeted grounding: high precision gain on grounding-completeness,
; moderate value improvement (grounding work produces understanding).
; Slight gain on target itself (testing and refining its meaning).
(= (get-raw-prec-delta (ground-concept $target) $obs)
   (if (== $obs grounding-completeness) 0.15
     (if (== $target $obs) 0.05 -0.01)))
(= (get-raw-value-delta (ground-concept $target) $obs)
   (if (== $obs grounding-completeness) 0.05
     (if (== $target $obs) 0.02 0.0)))
(= (get-model-confidence (ground-concept $target) $obs) 0.3)
(= (get-action-cost (ground-concept $target)) 0.03)

; Observation accessors
(= (get-obs-value $obs)
   (let $r (match &state (observation $obs $val $prec $t) $val)
     (if (== $r ()) 0.5 $r)))

(= (get-obs-precision $obs)
   (let $r (match &state (observation $obs $val $prec $t) $prec)
     (if (== $r ()) 0.5 $r)))

(= (get-belief-value-safe $obs)
   (let $r (match &state (belief $obs $val $prec) $val)
     (if (== $r ()) 0.5 $r)))

; Preference accessor (Dual-space)
(= (get-preferred-value $obs $fallback)
   (let $r1 (match &state (preference $obs $pv $pw) $pv)
     (if (== $r1 ())
         (let $r2 (match &state (preference $obs $pv $pw) $pv)
           (if (== $r2 ()) $fallback $r2))
         $r1)))


; =============================================================================
; SECTION V: EXPECTED FREE ENERGY COMPUTATION
; =============================================================================

(= (compute-efe $action)
   (let* (
     ($expected-error (predict-total-error $action))
     ($cost (get-action-cost $action))
     ($obs-info-gain (compute-obs-info-gain $action))
     ($model-info-gain (compute-model-info-gain $action))
     ($total-info-gain (+ $obs-info-gain $model-info-gain))
     ($ew (epistemic-weight))
     ($v-effect (viability-effect $action))
     ($efe (+ (- (+ $expected-error $cost) (* $ew $total-info-gain)) $v-effect))
   )
   $efe))


; =============================================================================
; SECTION VI: PREDICTED ERROR (confidence-weighted, recursive)
; =============================================================================

(= (predict-total-error $action)
   (predict-error-sum $action (get-belief-list)))

(= (get-belief-list)
   (collapse (match &state (belief $obs $val $prec) ($obs $val $prec))))

(= (predict-error-sum $action ()) 0.0)
(= (predict-error-sum $action (($obs $bval $bprec) . $rest))
   (+ (predict-single-error $action $obs $bval $bprec)
      (predict-error-sum $action $rest)))

; Single observable: expected error under model uncertainty.
(= (predict-single-error $action $obs $bval $bprec)
   (let* (
     ($conf (get-model-confidence $action $obs))
     ($raw-vd (get-raw-value-delta $action $obs))
     ($pdelta (get-prec-delta $action $obs))         

     ($new-bprec (clamp (+ $bprec $pdelta) 0.1 1.0))
     ($pref-val (get-preferred-value $obs $bval))

     ; Case 1: Model is correct (prob = conf). Predict from BELIEF.
     ($predicted-val-right (+ $bval $raw-vd))
     ($error-right (abs (- $predicted-val-right $pref-val)))  
     ($weighted-right (* $error-right $new-bprec))

     ; Case 2: Model is wrong (prob = 1-conf)
     ($error-wrong (abs (- $bval $pref-val)))              
     ($weighted-wrong (* $error-wrong $bprec))
   )
   (+ (* $conf $weighted-right) (* (- 1.0 $conf) $weighted-wrong))))


; =============================================================================
; SECTION VII: INFORMATION GAIN (observable + model)
; =============================================================================

(= (compute-obs-info-gain $action)
   (obs-info-gain-sum $action (get-belief-list)))

(= (obs-info-gain-sum $action ()) 0.0)
(= (obs-info-gain-sum $action (($obs $bval $bprec) . $rest))
   (let* (
     ($pdelta (get-raw-prec-delta $action $obs))
     ($uncertainty (- 1.0 $bprec))
     ; Info-gain asymmetry:
     ;   Positive delta-prec: scaled by (1 - precision) — diminishing returns
     ;     at high precision. An agent that already knows something well gains
     ;     little from knowing it better.
     ;   Negative delta-prec: applies at full rate — precision decay is not
     ;     attenuated. You can't ignore the cost of losing knowledge just
     ;     because you currently have it.
     ($effective-gain (if (> $pdelta 0.0)
                         (* $pdelta $uncertainty)
                         $pdelta))
   )
   (+ $effective-gain (obs-info-gain-sum $action $rest))))

(= (positive-part $x) (if (> $x 0.0) $x 0.0))

(= (compute-model-info-gain $action)
   (model-info-gain-sum $action (get-belief-list)))

(= (model-info-gain-sum $action ()) 0.0)
(= (model-info-gain-sum $action (($obs $bval $bprec) . $rest))
   (let $conf (get-model-confidence $action $obs)
     (+ (* 0.1 (- 1.0 $conf))
        (model-info-gain-sum $action $rest))))


; =============================================================================
; SECTION VIII: ADAPTIVE EPISTEMIC WEIGHT
; =============================================================================

(= (epistemic-weight)
   (let* (
     ($pressure (get-viability-pressure))
     ($safety (- 1.0 $pressure))           
   )
   (max 0.05 $safety)))

(= (safe-avg-prec ()) 0.5)
(= (safe-avg-prec ($x)) $x)
(= (safe-avg-prec ($x . $rest))
   (let* (
     ($n (+ 1 (count-list $rest)))
     ($sum (+ $x (sum-list-r $rest)))
   )
   (/ $sum $n)))

(= (count-list ()) 0)
(= (count-list ($x . $rest)) (+ 1 (count-list $rest)))

(= (sum-list-r ()) 0.0)
(= (sum-list-r ($x . $rest)) (+ $x (sum-list-r $rest)))


; =============================================================================
; SECTION IX: PRINCIPLED VIABILITY EFFECT (Dual-Space)
; =============================================================================

(= (viability-effect $action)
   (let* (
     ($b1 (collapse (match &state (viability-bound $obs $min $max) ($obs $min $max))))
     ($b2 (collapse (match &state (viability-bound $obs $min $max) ($obs $min $max))))
   )
   (fold-viability-effects $action (list-append $b1 $b2))))

(= (fold-viability-effects $action ()) 0.0)
(= (fold-viability-effects $action (($obs $min $max) . $rest))
   (+ (single-viability-effect $action $obs $min $max)
      (fold-viability-effects $action $rest)))

(= (single-viability-effect $action $obs $min $max)
   (let* (
     ($bval (get-belief-value-safe $obs))
     ($bprec (get-belief-precision-safe $obs))
     ($current-pressure (compute-boundary-pressure $obs $bval $bprec $min $max))

     ($vdelta (get-value-delta $action $obs))
     ($new-val (+ $bval $vdelta))
     ; After action, precision may change — use post-action precision estimate
     ($pdelta (get-prec-delta $action $obs))
     ($new-prec (clamp (+ $bprec $pdelta) 0.01 1.0))
     ($new-pressure (compute-boundary-pressure $obs $new-val $new-prec $min $max))

     ($delta-pressure (- $new-pressure $current-pressure))
     ($scale (max $current-pressure $new-pressure))

     ($conf (get-model-confidence $action $obs))
     ($uncertainty-risk (* $scale (* (- 1.0 $conf) 0.1)))
   )
   (+ (* $scale $delta-pressure) $uncertainty-risk)))


; =============================================================================
; SECTION X: VIABILITY PRESSURE (shared utility)
; =============================================================================

; Belief precision accessor with safe fallback
(= (get-belief-precision-safe $obs)
   (let $r (match &state (belief $obs $val $prec) $prec)
     (if (== $r ()) 0.5 $r)))

; Viability pressure derived from physical proximity to the boundary.
;
; INVARIANT: Precision AMPLIFIES existential risk. An agent that knows
; with certainty it is near death should be MORE alarmed, not less.
;
; The danger zone is a fraction of the viable range where pressure
; activates. Within this zone, pressure scales linearly from 0 to 1,
; multiplied by precision — the more certain the agent is about its
; proximity to the boundary, the stronger the signal.
;
; This is the biological analogy: pain is the highest-precision signal
; in the nervous system. Certainty about damage produces urgency.
;
(= (compute-boundary-pressure $obs $val $prec $min $max)
   (let* (
     ($range (- $max $min))
     ($margin-lo (- $val $min))
     ($margin-hi (- $max $val))
     ($margin (min $margin-lo $margin-hi))
     ($normalized (if (> $range 0.001) (/ $margin $range) 1.0))
     ($danger-zone (get-config viability-danger-zone))
   )
   (if (< $normalized 0.0)
       1.0
       (if (< $normalized $danger-zone)
           (* $prec (- 1.0 (/ $normalized $danger-zone)))
           0.0))))

(= (get-viability-pressure)
   (let* (
     ($p1 (collapse
            (match &state (viability-bound $obs $min $max)
              (match &state (belief $obs $val $prec)
                (compute-boundary-pressure $obs $val $prec $min $max)))))
     ($p2 (collapse
            (match &state (viability-bound $obs $min $max)
              (match &state (belief $obs $val $prec)
                (compute-boundary-pressure $obs $val $prec $min $max)))))
   )
   (max-pressure-from (list-append $p1 $p2))))

(= (max-pressure-from ()) 0.0)
(= (max-pressure-from ($p . $rest))
   (max $p (max-pressure-from $rest)))


; =============================================================================
; SECTION XI: ACTION SELECTION
; =============================================================================

; base-actions is the single source of truth for the default action set.
; compute-efe-list and find-min-efe-dynamic (Section XII) handle any length.
; base-actions reads from &state, set by the server during init.
; This eliminates the boot-time &self mutation hack.
(= (base-actions)
   (match &state (active-action-set $actions) $actions))

(= (select-action-myopic)
   (let* (
     ($efes (compute-efe-list (base-actions)))
     ($best (find-min-efe-dynamic $efes))
   )
   $best))

; =============================================================================
; SECTION XII: TRACED SELECTION & DYNAMIC SUPPORT
; =============================================================================

(= (select-action-myopic-traced)
   (let* (
     ($efes (compute-efe-list (base-actions)))
     ($best (find-min-efe-dynamic $efes))
     ($pressure (get-viability-pressure))
     ($ew (epistemic-weight))
   )
   (selection-trace
     (efe-values $efes)
     (selected $best)
     (viability-pressure $pressure)
     (epistemic-weight $ew)
     (model-confidences (build-model-confidences (base-actions))))))

(= (avg-action-confidence $action)
   (let $confs (collapse (match &state (action-model $action $obs $vd $pd $c) $c))
     (safe-avg-prec $confs)))

; Build model-confidence list from base-actions (for trace output)
(= (build-model-confidences ()) ())
(= (build-model-confidences ($a . $rest))
   (($a (avg-action-confidence $a)) . (build-model-confidences $rest)))

(= (all-available-actions)
   (let* (
     ($raw1 (collapse (match &state (action-model $a $obs $vd $pd $c) $a)))
     ($raw2 (collapse (match &state (action-model $a $obs $vd $pd $c) $a)))
   )
   (deduplicate (list-append $raw1 $raw2))))

(= (deduplicate ()) ())
(= (deduplicate ($x . $xs))
   (if (list-contains? $x $xs)
       (deduplicate $xs)
       ($x . (deduplicate $xs))))

(= (compute-all-efes)
   (let $actions (all-available-actions)
     (compute-efe-list $actions)))

(= (compute-efe-list ()) ())
(= (compute-efe-list ($action . $rest))
   (let $efe (compute-efe $action)
     (($action $efe) . (compute-efe-list $rest))))

(= (find-min-efe-dynamic ()) wait)  
(= (find-min-efe-dynamic (($action $efe))) $action)  
(= (find-min-efe-dynamic (($a1 $e1) ($a2 $e2) . $rest))
   (if (< $e1 $e2)
       (find-min-efe-dynamic (($a1 $e1) . $rest))
       (find-min-efe-dynamic (($a2 $e2) . $rest))))

(= (select-action-dynamic)
   (let* (
     ($efes (compute-all-efes))
     ($best (find-min-efe-dynamic $efes))
   )
   $best))

(= (select-action-dynamic-traced)
   (let* (
     ($efes (compute-all-efes))
     ($best (find-min-efe-dynamic $efes))
     ($pressure (get-viability-pressure))
     ($ew (epistemic-weight))
   )
   (selection-trace
     (efe-values $efes)
     (selected $best)
     (viability-pressure $pressure)
     (epistemic-weight $ew)
     (available-actions (all-available-actions)))))

; =============================================================================
; SECTION XIII: ACTION EXECUTION
; =============================================================================

(= (execute-action! $action)
   (sequential
     (clear-current-action!)
     (add-atom &state (current-action $action))
     done))

(= (clear-current-action!)
   (let $old (collapse (match &state (current-action $a) (current-action $a)))
     (remove-all-atoms $old)))

(= (remove-all-atoms ()) done)
(= (remove-all-atoms ($a . $rest))
   (sequential (remove-atom &state $a) (remove-all-atoms $rest)))

(= (get-current-action)
   (match &state (current-action $a) $a))

; =============================================================================
; SECTION XIV: DIAGNOSTICS
; =============================================================================

(= (remaining-hand-tuned)
   (hand-tuned-report
     (model-info-gain-scale 0.1 "relative value of model vs obs learning")
     (epistemic-floor 0.05 "minimum info value")
     (viability-uncertainty-risk 0.1 "risk aversion scaling near bounds")
     (initial-model-confidence 0.3 "prior strength on action effects")
     (note "All other values either learned from experience or derived from state")))

; =============================================================================
; END OF ACTIONS v4.3
; =============================================================================
