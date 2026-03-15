; =============================================================================
; METTA COGNITIVE CORE - BELIEFS AND LEARNING
; =============================================================================
;
; Beliefs represent the system's model of the world.
; Learning updates beliefs when observations surprise us.
;
; Schema: (belief <observable> <value> <precision>)
;
; Precision is inverse variance - how confident we are in the belief.
; High precision beliefs are harder to change.
; Low precision beliefs update easily from observations.
;
; =============================================================================

!(import! &self foundations)


; =============================================================================
; SECTION I: BELIEF ACCESSORS
; =============================================================================

; Get belief value for an observable
(= (get-belief-value $obs)
   (match &state (belief $obs $val $prec) $val))

; Get belief precision for an observable
(= (get-belief-precision $obs)
   (match &state (belief $obs $val $prec) $prec))

; Get full belief
(= (get-belief $obs)
   (match &state (belief $obs $val $prec) (belief $obs $val $prec)))

; Get all beliefs
(= (all-beliefs)
   (collapse (match &state (belief $o $v $p) (belief $o $v $p))))

; Check if belief exists
(= (has-belief? $obs)
   (let $result (collapse (match &state (belief $obs $v $p) True))
     (if (== $result ()) False True)))


; =============================================================================
; SECTION II: BELIEF MUTATION
; =============================================================================

; Set belief (create or update)
(= (set-belief! $obs $val $prec)
   (if (has-belief? $obs)
       ; Update existing
       (let ($old-val $old-prec) (match &state (belief $obs $ov $op) ($ov $op))
         (sequential
           (remove-atom &state (belief $obs $old-val $old-prec))
           (add-atom &state (belief $obs $val $prec))
           (belief-set $obs $val $prec)))
       ; Create new
       (sequential
         (add-atom &state (belief $obs $val $prec))
         (belief-created $obs $val $prec))))

; Remove belief
(= (remove-belief! $obs)
   (let $belief (match &state (belief $obs $v $p) (belief $obs $v $p))
     (sequential
       (remove-atom &state $belief)
       (belief-removed $obs))))


; =============================================================================
; SECTION III: PREDICTION ERROR COMPUTATION
; =============================================================================

; Core Active Inference: Compare beliefs to observations.
; Error = observation - belief, weighted by combined precision.

; Compute prediction error for a single observable
(= (compute-prediction-error $obs)
   (match &state (belief $obs $bval $bprec)
     (match &state (observation $obs $oval $oprec $time)
       (let* (
         ($error (- $oval $bval))
         ($abs-error (abs $error))
         ($combined-prec (* $bprec $oprec))
         ($weighted-error (* $abs-error $combined-prec))
       )
       (prediction-error $obs $error $abs-error $combined-prec $weighted-error)))))

; Compute all prediction errors
(= (all-prediction-errors)
   (collapse
     (match &state (belief $obs $bv $bp)
       (match &state (observation $obs $ov $op $t)
         (let* (
           ($error (- $ov $bv))
           ($abs-error (abs $error))
           ($combined-prec (* $bp $op))
           ($weighted-error (* $abs-error $combined-prec))
         )
         (prediction-error $obs $error $abs-error $combined-prec $weighted-error))))))

; Get observables where the Bayesian update would be non-trivial.
; Same criterion as should-learn? — based on Kalman update magnitude.
; (Currently unused — retained for diagnostic/introspection queries.)
(= (surprising-observables)
   (collapse
     (match &state (belief $obs $bv $bp)
       (match &state (observation $obs $ov $op $t)
         (let* (
           ($error (abs (- $ov $bv)))
           ($prec-sum (+ $bp $op))
           ($obs-weight (/ $op $prec-sum))
           ($update-magnitude (* $obs-weight $error))
         )
         (if (> $update-magnitude 0.01)
             $obs
             (empty)))))))


; =============================================================================
; SECTION III-B: PREDICTION CORRECTNESS (for metabolic rewards)
; =============================================================================
;
; TWO QUESTIONS, TWO METRICS:
;
; Q1: "Should I update my belief?" (should-learn?, above)
;     Metric: Kalman update magnitude. "Would the posterior shift meaningfully?"
;     Gate on the update itself, not the error.
;
; Q2: "Was my prediction correct?" (is-prediction-correct?, below)
;     Metric: error < uncertainty. "Was the error within my uncertainty band?"
;     Used by metabolic rewards in structure_learning and atom_lifecycle.
;     An uncertain agent's rough prediction counts as correct — this prevents
;     structure from starving during early learning when everything is noisy.
;
; These MUST be different. Q1 with (1 - precision) blocks all learning at low
; precision. Q2 with Kalman magnitude would make reward depend on observation
; precision (a sensor property, not a prediction property).

(= (is-prediction-correct? $obs $error)
   (match &state (belief $obs $val $prec)
     (< (abs $error) (max (- 1.0 $prec) 0.01))))

; Aggregate: mean uncertainty across all beliefs.
; Used by reward-chunk! where the threshold applies to total error.
(= (mean-belief-uncertainty)
   (let* (
     ($uncertainties (collapse (match &state (belief $obs $val $prec) (- 1.0 $prec))))
     ($n (length-list $uncertainties))
   )
   (if (== $n 0) 0.5
       (/ (sum-list $uncertainties) $n))))

(= (length-list ()) 0)
(= (length-list ($x . $rest)) (+ 1 (length-list $rest)))


; =============================================================================
; SECTION IV: BAYESIAN-ISH BELIEF UPDATE
; =============================================================================
;
; Update belief toward observation, weighted by relative precision.
; Higher precision observations have more influence.
;

(= (update-belief-from-observation! $obs)
   (match &state (belief $obs $bval $bprec)
     (match &state (observation $obs $oval $oprec $time)
       (let* (
         ($error (- $oval $bval))
         ($abs-error (abs $error))
         
         ; Precision-weighted update (True Bayesian Kalman filter)
         ($prec-sum (+ $bprec $oprec))
         ($obs-weight (/ $oprec $prec-sum))
         
         ; FIX: Removed the artificial * $lr throttle! 
         ; High precision sensors now update beliefs immediately.
         ($update-magnitude (* $obs-weight $error))
         ($new-val (+ $bval $update-magnitude))
         
         ; Update precision based on error
         ($new-prec (update-precision $bprec $abs-error))
       )
       ; Apply update
       (sequential
         (remove-atom &state (belief $obs $bval $bprec))
         (add-atom &state (belief $obs $new-val $new-prec))
         (belief-updated $obs $bval $new-val $bprec $new-prec))))))

; Update all beliefs that have observations
(= (update-all-beliefs!)
   (let $obs-list (collapse (match &state (observation $o $v $p $t) $o))
     (map-update $obs-list)))

(= (map-update ()) done)
(= (map-update ($o . $rest))
   (sequential
     (update-belief-from-observation! $o)
     (map-update $rest)))


; =============================================================================
; SECTION V: SELECTIVE LEARNING
; =============================================================================

; Only update beliefs when the Bayesian update would be non-trivial.
; This prevents noisy observations from degrading good beliefs.
;
; INVARIANT: The gate is the update magnitude itself — the same Kalman
; weighting the actual update uses. "Should I learn?" IS "would the
; posterior shift meaningfully?"
;
;   update_magnitude = |(oprec / (bprec + oprec)) × error|
;
; This automatically handles all cases:
;   - Uncertain belief + reliable sensor → big update → learn
;   - Confident belief + noisy sensor → tiny update → skip
;   - Confident belief + reliable sensor + small error → moderate update → learn
;
; The noise floor (0.01) is a numerical epsilon, not a hand-tuned threshold.

(= (should-learn? $obs)
   (match &state (belief $obs $bval $bprec)
     (match &state (observation $obs $oval $oprec $time)
       (let* (
         ($error (abs (- $oval $bval)))
         ($prec-sum (+ $bprec $oprec))
         ($obs-weight (/ $oprec $prec-sum))
         ($update-magnitude (* $obs-weight $error))
       )
       (> $update-magnitude 0.01)))))

; Selective update - only learn from surprising observations
(= (selective-update-belief! $obs)
   (if (should-learn? $obs)
       (update-belief-from-observation! $obs)
       (no-update $obs)))

; Update all beliefs selectively
(= (selective-update-all-beliefs!)
   (let $obs-list (collapse (match &state (observation $o $v $p $t) $o))
     (map-selective-update $obs-list)))

(= (map-selective-update ()) done)
(= (map-selective-update ($o . $rest))
   (sequential
     (selective-update-belief! $o)
     (map-selective-update $rest)))


; =============================================================================
; SECTION VI: PRECISION DYNAMICS
; =============================================================================

; Precision changes based on prediction success:
; - Confirmed predictions -> increase precision (more confident)
; - Surprising observations -> decrease precision (less confident)

(= (boost-precision! $obs $amount)
   (match &state (belief $obs $val $prec)
     (let $new-prec (clamp (+ $prec $amount) 
                           (get-config precision-floor)
                           (get-config precision-ceiling))
       (sequential
         (remove-atom &state (belief $obs $val $prec))
         (add-atom &state (belief $obs $val $new-prec))
         (precision-boosted $obs $prec $new-prec)))))

(= (reduce-precision! $obs $amount)
   (boost-precision! $obs (- 0.0 $amount)))


; =============================================================================
; SECTION VII: BELIEF INITIALIZATION
; =============================================================================

; Initialize belief with default (near-zero) precision
(= (init-belief! $obs $val)
   (set-belief! $obs $val (get-config initial-belief-precision)))

; Initialize belief with specified precision
(= (init-belief-with-precision! $obs $val $prec)
   (set-belief! $obs $val $prec))

; Bulk initialize from list
(= (init-beliefs! ()) done)
(= (init-beliefs! ((belief $o $v $p) . $rest))
   (sequential
     (set-belief! $o $v $p)
     (init-beliefs! $rest)))


; =============================================================================
; SECTION VII-A: OBSERVATION-DRIVEN PRECISION RATCHET
; =============================================================================
;
; Each time an observable is sampled, its precision increases by a fixed
; amount independent of prediction accuracy. This solves the bootstrap
; problem: precision reflects "how much data the agent has about this
; variable" rather than "how well it predicted last time."

(= (ratchet-precision! $obs)
   (match &state (belief $obs $val $prec)
     (let $new-prec (clamp (+ $prec (get-config precision-ratchet-rate))
                           (get-config precision-floor)
                           (get-config precision-ceiling))
       (sequential
         (remove-atom &state (belief $obs $val $prec))
         (add-atom &state (belief $obs $val $new-prec))
         (precision-ratcheted $obs $prec $new-prec)))))


; =============================================================================
; END OF BELIEFS AND LEARNING
; =============================================================================