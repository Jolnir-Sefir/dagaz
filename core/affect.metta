; =============================================================================
; METTA COGNITIVE CORE - AFFECT (REVISED)
; =============================================================================
;
; PRINCIPLE FIX: Affect is a DERIVED VIEW, not stored state.
;
; Affect is computed on-demand from the current error landscape.
; There is no (affect valence X) atom to mutate.
; Affect is what the error pattern LOOKS LIKE, not a separate thing.
;
; This eliminates:
;   - The ability to set affect arbitrarily
;   - Affect drifting from actual system state
;   - Separate affect update rules
;
; =============================================================================

!(import! &self foundations)


; =============================================================================
; SECTION I: ERROR ACCUMULATOR
; =============================================================================

; We DO store a smoothed error history for temporal dynamics.
; But this is an observation about errors, not affect itself.

; Error history (ring buffer of recent weighted errors)
; (error-history ())
; (error-history-max-length 10)

; Add error to history
(= (record-error! $weighted-error $timestamp)
   (let* (
     ($current (match &state (error-history $h) $h))
     ($max-len (match &state (error-history-max-length $n) $n))
     ($new-entry ($weighted-error $timestamp))
     ($updated (cons-limited $new-entry $current $max-len))
   )
   (sequential
     (remove-atom &state (error-history $current))
     (add-atom &state (error-history $updated))
     recorded)))

(= (cons-limited $x $xs $max)
   (take $max ($x . $xs)))

(= (take 0 $xs) ())
(= (take $n ()) ())
(= (take $n ($x . $xs)) ($x . (take (- $n 1) $xs)))

; Get error history
(= (get-error-history)
   (match &state (error-history $h) $h))


; =============================================================================
; SECTION II: AFFECT AS DERIVED VIEW
; =============================================================================

; Valence: How well are predictions being confirmed?
; Computed from total weighted error (more error = more negative)
; SIMPLIFIED: Linear mapping, clamped

(= (compute-valence)
   (let $total (total-weighted-error)
     ; Linear map: 0 error = +0.5, 1.0 error = -0.5
     (clamp (- 0.5 $total) -1.0 1.0)))

; Helper to compute total error (handles MeTTa list format)
(= (total-weighted-error)
   (fold-errors (all-prediction-errors) 0.0))

; Sum weighted errors from prediction-error list
; MeTTa lists are (a b c), not (a . (b . c))
(= (sum-errors ()) 0.0)
(= (sum-errors ((prediction-error $o $e $ae $p $we))) $we)
(= (sum-errors ((prediction-error $o1 $e1 $ae1 $p1 $we1) 
                (prediction-error $o2 $e2 $ae2 $p2 $we2)))
   (+ $we1 $we2))
(= (sum-errors ((prediction-error $o1 $e1 $ae1 $p1 $we1) 
                (prediction-error $o2 $e2 $ae2 $p2 $we2)
                (prediction-error $o3 $e3 $ae3 $p3 $we3)))
   (+ (+ $we1 $we2) $we3))
(= (sum-errors ((prediction-error $o1 $e1 $ae1 $p1 $we1) 
                (prediction-error $o2 $e2 $ae2 $p2 $we2)
                (prediction-error $o3 $e3 $ae3 $p3 $we3)
                (prediction-error $o4 $e4 $ae4 $p4 $we4)))
   (+ (+ (+ $we1 $we2) $we3) $we4))

(= (fold-errors () $acc) $acc)
(= (fold-errors ((prediction-error $o $e $ae $p $we) . $rest) $acc)
   (fold-errors $rest (+ $acc $we)))

; Arousal: How surprising/volatile is the current situation?
; Computed from max error magnitude
; SIMPLIFIED: Direct mapping

(= (compute-arousal)
   (let $max-err (max-current-error)
     (clamp $max-err 0.0 1.0)))

; Helper to get max error (handles MeTTa list format)
(= (max-current-error)
   (fold-max-error (all-prediction-errors) 0.0))

; Max error from prediction-error list
(= (max-err ()) 0.0)
(= (max-err ((prediction-error $o $e $ae $p $we))) $ae)
(= (max-err ((prediction-error $o1 $e1 $ae1 $p1 $we1) 
             (prediction-error $o2 $e2 $ae2 $p2 $we2)))
   (max $ae1 $ae2))
(= (max-err ((prediction-error $o1 $e1 $ae1 $p1 $we1) 
             (prediction-error $o2 $e2 $ae2 $p2 $we2)
             (prediction-error $o3 $e3 $ae3 $p3 $we3)))
   (max (max $ae1 $ae2) $ae3))
(= (max-err ((prediction-error $o1 $e1 $ae1 $p1 $we1) 
             (prediction-error $o2 $e2 $ae2 $p2 $we2)
             (prediction-error $o3 $e3 $ae3 $p3 $we3)
             (prediction-error $o4 $e4 $ae4 $p4 $we4)))
   (max (max (max $ae1 $ae2) $ae3) $ae4))

(= (fold-max-error () $acc) $acc)
(= (fold-max-error ((prediction-error $o $e $ae $p $we) . $rest) $acc)
   (fold-max-error $rest (max $ae $acc)))

; Dominance: How confident are we in our beliefs?
; Computed from average belief precision
; SIMPLIFIED: Direct average

(= (compute-dominance)
   (avg-belief-precision))

; Helper to compute average precision
(= (avg-belief-precision)
   (let $precs (collapse (match &state (belief $o $v $p) $p))
   (if (== $precs ())
      0.5
     (average $precs))))

; (= (compute-avg ()) 0.5)  ; Default if no beliefs
; (= (compute-avg ($x)) $x) ; Single element
; (= (compute-avg ($x $y)) (/ (+ $x $y) 2))
; (= (compute-avg ($x $y $z)) (/ (+ (+ $x $y) $z) 3))
; (= (compute-avg ($a $b $c $d)) (/ (+ (+ (+ $a $b) $c) $d) 4))
; (= (compute-avg ($a $b $c $d $e)) (/ (+ (+ (+ (+ $a $b) $c) $d) $e) 5))
; Add more cases as needed, or use a different approach

(= (fold-sum () $acc) $acc)
(= (fold-sum ($x . $rest) $acc) (fold-sum $rest (+ $acc $x)))

; --- Helpers ---

(= (sum-weighted-errors ()) 0.0)
(= (sum-weighted-errors ((prediction-error $o $e $ae $p $we) . $rest))
   (+ $we (sum-weighted-errors $rest)))

(= (max-error-magnitude ()) 0.0)
(= (max-error-magnitude ((prediction-error $o $e $ae $p $we) . $rest))
   (max $ae (max-error-magnitude $rest)))

(= (all-belief-precisions)
   (collapse (match &state (belief $o $v $p) $p)))

; (= (safe-average () $default) $default)
; (= (safe-average $xs $default)
;   (/ (sum-list $xs) (max 1 (length-list $xs))))

; (= (sum-list ()) 0.0)
; (= (sum-list ($x . $xs)) (+ $x (sum-list $xs)))

; (= (length-list ()) 0)
; (= (length-list ($x . $xs)) (+ 1 (length-list $xs)))

(= (error-variance ()) 0.0)
(= (error-variance $history)
   (let* (
     ($vals (map-first $history))
     ($mean (safe-average $vals 0.0))
     ($sq-diffs (map-sq-diff $vals $mean))
   )
   (safe-average $sq-diffs 0.0)))

(= (map-first ()) ())
(= (map-first (($v $t) . $rest)) ($v . (map-first $rest)))

(= (map-sq-diff () $mean) ())
(= (map-sq-diff ($x . $xs) $mean) 
   ((square (- $x $mean)) . (map-sq-diff $xs $mean)))

(= (square $x) (* $x $x))

(= (sqrt $x) (pow $x 0.5))

; Sigmoid approximation
(= (sigmoid $x)
   (/ 1.0 (+ 1.0 (exp (- 0.0 $x)))))

; Exp approximation (for small values)
(= (exp $x)
   (if (> $x 2.0) 7.389
     (if (< $x -2.0) 0.135
       (+ 1.0 $x (* 0.5 $x $x)))))  ; Taylor series


; =============================================================================
; SECTION III: UNIFIED AFFECT ACCESSOR
; =============================================================================

; Get current affect state (always computed fresh)
(= (get-affect-state)
   (affect-state
     (compute-valence)
     (compute-arousal)
     (compute-dominance)))

; Individual dimension accessors
(= (get-affect valence) (compute-valence))
(= (get-affect arousal) (compute-arousal))
(= (get-affect dominance) (compute-dominance))

; NO set-affect! or delta-affect! functions
; Affect cannot be set directly - it reflects the error landscape


; =============================================================================
; SECTION IV: AFFECT-DERIVED QUANTITIES
; =============================================================================

; These replace the old "affect modes" with continuous quantities.
; No thresholds, no discrete categories.

; --- Urgency: How much does the system need to act? ---
; High arousal + negative valence = high urgency
(= (compute-urgency)
   (let* (
     ($a (compute-arousal))
     ($v (compute-valence))
     ; Urgency = arousal * (1 - normalized_valence)
     ($v-factor (- 0.5 (* 0.5 $v)))  ; Maps [-1,1] to [1,0]
   )
   (* $a $v-factor)))

; --- Confidence: Can we trust our decisions? ---
; High dominance + low arousal = confident
(= (compute-confidence)
   (let* (
     ($d (compute-dominance))
     ($a (compute-arousal))
   )
   (* $d (- 1.0 $a))))

; --- Curiosity: Should we explore? ---
; Positive valence + low arousal + low dominance = curious
(= (compute-curiosity)
   (let* (
     ($v (compute-valence))
     ($a (compute-arousal))
     ($d (compute-dominance))
     ; Curious when calm, things are OK, but we're uncertain
     ($happy (+ 0.5 (* 0.5 $v)))  ; Maps [-1,1] to [0,1]
   )
   (* (* $happy (- 1.0 $a)) (- 1.0 $d))))


; =============================================================================
; SECTION V: AFFECT FOR ACTION SELECTION
; =============================================================================

; Instead of modes, provide continuous modulation parameters

; Risk tolerance: How much risk should we accept?
; Low when arousal high or valence negative
(= (risk-tolerance)
   (let* (
     ($v (compute-valence))
     ($a (compute-arousal))
     ($raw (* (+ 0.5 (* 0.3 $v)) (- 1.0 (* 0.5 $a))))
   )
   (clamp $raw 0.1 0.9)))

; Exploration rate: How much should we try new things?
; High when curious, low when urgent
(= (exploration-rate)
   (let* (
     ($curiosity (compute-curiosity))
     ($urgency (compute-urgency))
     ($raw (- $curiosity $urgency))
   )
   (clamp (+ 0.5 $raw) 0.0 1.0)))

; Action temperature: How stochastic should selection be?
; High arousal = more deterministic (focus)
(= (action-temperature)
   (let $a (compute-arousal)
     (+ 0.1 (* 0.4 (- 1.0 $a)))))


; =============================================================================
; END OF AFFECT (REVISED)
; =============================================================================
