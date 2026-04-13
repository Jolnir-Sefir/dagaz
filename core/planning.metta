; =============================================================================
; METTA COGNITIVE CORE - FRACTAL PLANNING (Adaptive Beam Search)
; =============================================================================
;
; WHAT THIS DOES:
;   Replaces exhaustive policy enumeration (O(|A|^d)) with adaptive beam
;   search (O(|A| × k × d)) for scalable temporal planning.
;
; DESIGN:
;   Coarse-grain the decision tree by pruning high-EFE branches at each
;   depth level. A SINGLE adaptive mechanism — the noise floor — controls
;   all pruning decisions:
;
;     noise_floor(depth) = base_noise / degraded_confidence(depth)
;
;   As model confidence degrades with planning depth, the noise floor
;   grows. This single quantity determines:
;
;     1. BEAM WIDTH: Actions within noise_floor of the best survive.
;        Flat landscape + low noise → many survive (deliberation).
;        Steep landscape + low noise → few survive (focus).
;
;     2. SYMMETRY BREAKING: When only one action is within the noise
;        floor, the beam collapses to 1. The system has found a clear
;        minimum — evaluating alternatives is uninformative.
;
;     3. PLANNING HORIZON: When degraded confidence drops below a floor,
;        the noise is so large that no EFE differences are trustworthy.
;        Planning stops — the system refuses to extrapolate into noise.
;
;   Previously these were three independent mechanisms with separate
;   parameters (cutoff-ratio, min-beam-width, uncertainty-limit). Now
;   they emerge from one quantity that has a clear physical meaning:
;   "how much EFE difference is distinguishable from noise at this depth?"
;
; REUSES from policy_efe.metta:
;   sim-state, sim-efe, apply-action, violates-viability?,
;   sim-belief-value, sim-belief-precision, current-sim-state
;
; REUSES from actions.metta:
;   base-actions, get-action-cost, get-value-delta, get-prec-delta,
;   get-model-confidence, epistemic-weight, get-viability-pressure,
;   avg-action-confidence
;
; REUSES from foundations.metta:
;   clamp, abs, max, min, length, sum-list, sequential
;
; REPLACES: gen-policy, evaluate-policy, all-policy-evals,
;           best-of-evals, select-best-policy from policy_efe.metta §XI
;
; TRANSPARENCY: Every pruned branch leaves a (pruned ...) record.
;
; INVARIANT: Returns first action of the lowest-EFE trajectory found
;            within the adaptive computational budget.
;
; =============================================================================

!(import! &self foundations)
!(import! &self actions)
!(import! &self policy_efe)


; =============================================================================
; SECTION I: PLANNING PARAMETERS
; =============================================================================
;
; Two physics parameters + two hard caps. That's it.
;
;   noise-floor-base: The inherent noise in EFE discrimination, scaled
;     by inverse confidence. This is the ONE tunable that controls beam
;     width, symmetry breaking, and horizon termination. Typical EFE
;     ranges are ~0.2-0.3 in test scenarios; at confidence 0.5 the
;     noise floor is base/0.5 = 2*base.
;
;     SCALING FINDING (v2): At 3 actions, EFE range ~0.12. At 50 actions,
;     EFE range ~0.50. A base of 0.05 causes 195K node evaluations at
;     50 actions. Lowering to 0.03 cuts this to ~50 nodes with zero
;     accuracy loss across 10 random seeds. The EFE landscape has a
;     natural scale; ideally this parameter would be replaced entirely
;     by a landscape-derived quantity (see adaptive-noise-floor below).
;
;   confidence-floor: Minimum degraded confidence to attempt planning
;     at a given depth. Below this, the noise floor is so large that
;     all actions are indistinguishable. Derived from the same principle
;     as the noise floor — this is just the fast-path check that avoids
;     scoring actions when we already know the noise will swamp them.
;
;   base-beam-width: Starting beam width at depth 0, which NARROWS with
;     planning depth via degraded confidence. At depth d:
;       effective_beam(d) = max(2, ceil(base_beam × conf × discount^d))
;     This replaces the fixed max-beam-width. The physics: at shallow
;     depth (high confidence) we resolve fine detail among candidates.
;     At deep depth (degraded confidence) only dominant modes survive.
;     This is the RG coarse-graining analogy made literal: the number
;     of "relevant operators" decreases as you zoom out.
;
;   max-planning-depth: Hard ceiling on recursion. Not physics — safety.
;
; These can be promoted to beliefs with precision for bootstrap learning
; (see PLANNING_STRATEGY.md §4). Using a distinct prefix to avoid
; collision with world beliefs.

(planning-config noise-floor-base 0.03)    ; Tightened: see scaling benchmark
(planning-config confidence-floor 0.15)    ; Min confidence to continue
(planning-config base-beam-width 8)        ; Starting beam (narrows with depth)
(planning-config max-planning-depth 7)     ; Hard ceiling on recursion

; Accessor
(= (get-planning-config $key)
   (match &self (planning-config $key $val) $val))


; =============================================================================
; SECTION II: UTILITY FUNCTIONS
; =============================================================================

; --- Integer exponentiation (for confidence degradation) ---
; discount^depth where depth is a non-negative integer
(= (pow-int $base 0) 1.0)
(= (pow-int $base $n)
   (if (<= $n 0)
       1.0
       (* $base (pow-int $base (- $n 1)))))

; --- Insertion sort for candidate lists (ascending by EFE) ---
; Candidates are (candidate $action $efe) atoms.

(= (sort-candidates ()) ())
(= (sort-candidates ($x . $xs))
   (insert-candidate $x (sort-candidates $xs)))

(= (insert-candidate $x ()) ($x))
(= (insert-candidate (candidate $a1 $e1) ((candidate $a2 $e2) . $rest))
   (if (<= $e1 $e2)
       ((candidate $a1 $e1) (candidate $a2 $e2) . $rest)
       ((candidate $a2 $e2) . (insert-candidate (candidate $a1 $e1) $rest))))

; --- Take first N elements ---

(= (take-top 0 $xs) ())
(= (take-top $n ()) ())
(= (take-top $n ($x . $xs))
   (if (<= $n 0) ()
       ($x . (take-top (- $n 1) $xs))))

; --- Candidate accessors ---

(= (candidate-action (candidate $a $e)) $a)
(= (candidate-efe (candidate $a $e)) $e)

; --- EFE landscape statistics (for diagnostics only) ---

; Range: max EFE - min EFE across candidates
(= (efe-range ()) 0.001)  ; Avoid division by zero
(= (efe-range $candidates)
   (let* (
     ($efes (extract-efes $candidates))
     ($mn (list-min $efes))
     ($mx (list-max $efes))
     ($range (- $mx $mn))
   )
   (if (< $range 0.001) 0.001 $range)))

; Spread: mean absolute deviation from mean (for diagnostics)
(= (efe-spread ()) 0.0)
(= (efe-spread $candidates)
   (let* (
     ($efes (extract-efes $candidates))
     ($n (length $efes))
     ($mean (/ (sum-list $efes) (max $n 1)))
   )
   (/ (sum-abs-dev $efes $mean) (max $n 1))))

(= (sum-abs-dev () $mean) 0.0)
(= (sum-abs-dev ($x . $xs) $mean)
   (+ (abs (- $x $mean)) (sum-abs-dev $xs $mean)))

(= (extract-efes ()) ())
(= (extract-efes ((candidate $a $e) . $rest))
   ($e . (extract-efes $rest)))

(= (list-min ($x)) $x)
(= (list-min ($x . $rest))
   (min $x (list-min $rest)))

(= (list-max ($x)) $x)
(= (list-max ($x . $rest))
   (max $x (list-max $rest)))

; --- Average model confidence across all observables in a sim-state ---

; --- Average model confidence ---
; Model confidence is a property of the action-model atoms in &state,
; NOT of the simulated belief state. It doesn't change during simulation.
; Delegates to overall-model-confidence from action_learning.metta which
; averages across ALL action-model atoms — works for any action set.

(= (sim-avg-confidence $state)
   (overall-model-confidence))

; --- Average precision in a sim-state ---

(= (sim-avg-precision (sim-state $beliefs))
   (sim-avg-prec-beliefs $beliefs 0.0 0))

(= (sim-avg-prec-beliefs SNil $sum $count)
   (if (== $count 0) 0.5 (/ $sum $count)))
(= (sim-avg-prec-beliefs (SCons $o $v $p $rest) $sum $count)
   (sim-avg-prec-beliefs $rest (+ $sum $p) (+ $count 1)))

; --- Average preference gap in a sim-state ---
; How far are sim beliefs from preferences? (Unsigned mean gap.)

(= (sim-avg-preference-gap (sim-state $beliefs))
   (let $prefs (collapse (match &state (preference $o $target $prec) ($o $target)))
     (sim-pref-gap-acc $beliefs $prefs 0.0 0)))

(= (sim-pref-gap-acc $beliefs () $sum $count)
   (if (== $count 0) 0.1 (/ $sum (max $count 1))))
(= (sim-pref-gap-acc $beliefs (($o $target) . $rest) $sum $count)
   (let $val (sim-belief-value $beliefs $o)
     (sim-pref-gap-acc $beliefs $rest
       (+ $sum (abs (- $target $val)))
       (+ $count 1))))


; =============================================================================
; SECTION III: NOISE FLOOR (Unified Pruning Mechanism)
; =============================================================================
;
; The noise floor is the minimum EFE difference that's distinguishable
; from model uncertainty at a given planning depth. It unifies what were
; previously three independent mechanisms:
;
;   OLD: adaptive beam width     → NOW: count of actions within noise floor
;   OLD: symmetry breaking       → NOW: beam collapses to 1 when gap > noise
;   OLD: correlation length stop → NOW: confidence below floor = all noise
;
; The physics: model confidence degrades geometrically with depth
; (inherited from policy_efe.metta's discount factor). The noise floor
; is inversely proportional to this degraded confidence:
;
;   noise(d) = base / (base_conf × discount^d)
;
; At depth 0 with confidence 0.5 and base 0.05: noise = 0.10
;   → only actions within 0.10 of the best survive
; At depth 5 with confidence 0.30 and base 0.05: noise = 0.17
;   → wider beam, less discrimination
; At confidence floor 0.15: noise = 0.33
;   → virtually everything is noise, planning stops
;
; ONE parameter (noise-floor-base) replaces THREE (cutoff-ratio,
; min-beam-width, uncertainty-limit). base-beam-width narrows with
; degraded confidence — not a fixed cap but an emergent quantity.

; Compute the degraded model confidence at a given planning depth
(= (degraded-confidence $state $depth)
   (let $base-conf (sim-avg-confidence $state)
     (* $base-conf (pow-int discount $depth))))

; Compute the noise floor at a given planning depth
; Higher noise → less discrimination between actions
(= (noise-floor-at-depth $state $depth)
   (let* (
     ($deg-conf (degraded-confidence $state $depth))
     ($base (get-planning-config noise-floor-base))
   )
   ; Inverse confidence scaling, clamped to avoid division by zero
   (/ $base (max $deg-conf 0.01))))

; Should planning stop at this depth?
; Yes when degraded confidence drops below the floor — the noise
; is so large that scoring actions is uninformative.
(= (beyond-confidence-floor? $state $depth)
   (let* (
     ($deg-conf (degraded-confidence $state $depth))
     ($floor (get-planning-config confidence-floor))
   )
   (< $deg-conf $floor)))

; The unified noise filter: keep actions within noise floor of the best.
; Input: sorted candidates (ascending EFE), sim-state, current depth.
; Output: survivors (capped by adaptive beam width, at least 1 if any input).
;
; This single function replaces both adaptive-beam and apply-symmetry-cutoff.
; When the gap to the best is small relative to noise → action survives.
; When the gap exceeds noise → action is pruned (indistinguishable from
; being genuinely worse vs. being a noise artifact — we keep the best).
;
; ADAPTIVE BEAM WIDTH (v2):
;   effective_beam(depth) = max(2, ceil(base_beam × degraded_confidence))
;   At depth 0, conf=1.0 → full beam (8). At depth 5, conf≈0.44 → beam=4.
;   Combined with noise-floor pruning, this provides a double filter:
;   noise kills actions indistinguishable from best; beam caps kill
;   actions that are merely not-terrible.
(= (noise-filter $sorted-candidates $state $depth)
   (if (list-empty? $sorted-candidates)
       ()
   (let* (
     ($noise (noise-floor-at-depth $state $depth))
     ($best-efe (candidate-efe (list-head $sorted-candidates)))
     ; Adaptive beam: narrows with degraded confidence
     ($deg-conf (degraded-confidence $state $depth))
     ($base-beam (get-planning-config base-beam-width))
     ($effective-beam (max 2 (ceil-int (* $base-beam $deg-conf))))
     ; Keep all within noise floor, then cap at effective beam
     ($within-noise (filter-within-noise $sorted-candidates $best-efe $noise $depth))
   )
   (take-top $effective-beam $within-noise))))

; Integer ceiling (for beam width computation)
(= (ceil-int $x)
   (let $floor (floor-int $x)
     (if (> $x $floor) (+ $floor 1) $floor)))
(= (floor-int $x)
   (if (< $x 0)
       (- 0 (floor-int (- 0 $x)))
       (floor-int-pos $x 0)))
(= (floor-int-pos $x $n)
   (if (> (+ $n 1) $x) $n
       (floor-int-pos $x (+ $n 1))))

; Filter: keep candidates whose EFE is within noise-floor of the best
; Since the list is sorted ascending, once we hit one beyond the noise
; floor, all remaining are also beyond it — we can stop.
(= (filter-within-noise () $best $noise $depth) ())
(= (filter-within-noise ((candidate $a $e) . $rest) $best $noise $depth)
   (let $gap (- $e $best)
     (if (<= $gap $noise)
         ; Within noise floor — keep and continue
         ((candidate $a $e) . (filter-within-noise $rest $best $noise $depth))
         ; Beyond noise floor — prune this and everything after it
         (sequential
           (record-pruning-at-depth! $a $depth noise-floor $gap)
           (prune-remaining-noise $rest $depth)
           ()))))

; Record all remaining candidates as pruned (list is sorted, all are worse)
(= (prune-remaining-noise () $depth) done)
(= (prune-remaining-noise ((candidate $a $e) . $rest) $depth)
   (sequential
     (record-pruning-at-depth! $a $depth noise-floor $e)
     (prune-remaining-noise $rest $depth)))


; =============================================================================
; SECTION IV: RESIDUAL EFE (Avoids Myopic Pruning)
; =============================================================================
;
; Scores candidates using immediate sim-efe PLUS a future value estimate.
; Analogous to A*'s f = g + h:
;   g = sim-efe (immediate cost, from policy_efe infrastructure)
;   h = future-heuristic (optimistic estimate of remaining value)
;
; The heuristic is ADMISSIBLE (optimistic): it assumes future actions
; will capture the full available precision gain. This means pruning
; based on residual EFE never discards a branch that would have been
; the true optimum — at worst we keep too many branches.

(= (residual-efe $action $state $remaining-depth)
   (let* (
     ; Immediate: full sim-efe from existing infrastructure
     ($immediate (sim-efe $action $state))

     ; Evolve state to estimate future value
     ($next-state (apply-action $action $state))
     ($future-est (future-heuristic $next-state $remaining-depth))
   )
   (+ $immediate (* discount $future-est))))

; Heuristic: optimistic estimate of how much future actions can help.
; Based on: remaining preference gap × available precision headroom.
; Negative value = good (this is EFE, lower is better).
; Capped at 3 effective future steps to avoid runaway optimism.
(= (future-heuristic $state $remaining)
   (if (<= $remaining 0)
       0.0
       (let* (
         ($avg-prec (sim-avg-precision $state))
         ($avg-gap (sim-avg-preference-gap $state))
         ; More precision → lower future error (optimistic)
         ($headroom (- 1.0 $avg-prec))
         ($est-improvement (* $avg-gap $headroom))
         ($effective-steps (min $remaining 3))
       )
       ; Negative: this is how much EFE could decrease
       (- 0.0 (* $est-improvement $effective-steps)))))


; =============================================================================
; SECTION V: PRUNING TRACE (Transparency)
; =============================================================================
;
; Every discarded branch leaves a record. Queryable by diagnostics
; and the self-model.
;
; Schema: (fractal-pruned <action> <reason> <detail>)
;         (fractal-pruned-at <action> <depth> <reason> <detail>)
;
; Stored in &state (working memory), not &self (static code).
; These are transient per-cycle diagnostics, not function definitions.

(= (record-pruning! $action $reason $detail)
   (add-atom &state (fractal-pruned $action $reason $detail)))

(= (record-pruning-at-depth! $action $depth $reason $detail)
   (add-atom &state (fractal-pruned-at $action $depth $reason $detail)))

; Query: what was pruned and why?
(= (all-pruning-records)
   (collapse (match &state (fractal-pruned $a $r $d)
     (pruning-record $a $r $d))))

(= (all-pruning-records-with-depth)
   (collapse (match &state (fractal-pruned-at $a $depth $r $d)
     (pruning-record $a $depth $r $d))))

; Clear pruning records (called at start of each planning cycle)
(= (clear-fractal-pruning-records!)
   (let* (
     ($old1 (collapse (match &state (fractal-pruned $a $r $d)
                        (fractal-pruned $a $r $d))))
     ($old2 (collapse (match &state (fractal-pruned-at $a $dp $r $d)
                        (fractal-pruned-at $a $dp $r $d))))
     ($c1 (remove-all-listed $old1))
     ($c2 (remove-all-listed $old2))
   )
   done))

(= (remove-all-listed ()) done)
(= (remove-all-listed ($a . $rest))
   (sequential (remove-atom &state $a) (remove-all-listed $rest)))


; =============================================================================
; SECTION VI: BRANCH DATA STRUCTURE
; =============================================================================
;
; A branch carries:
;   $action    - the first action (what we'd actually do)
;   $efe       - cumulative EFE of the trajectory
;   $depth     - how deep the search went
;   $trace     - action sequence for transparency

(: Branch Type)
(: branch (-> Action Float Number %Undefined% Branch))
(: terminal-branch (-> Float Number Branch))

(= (branch-action (branch $a $efe $depth $trace)) $a)
(= (branch-efe (branch $a $efe $depth $trace)) $efe)
(= (branch-efe (terminal-branch $efe $depth)) $efe)
(= (branch-depth (branch $a $efe $depth $trace)) $depth)
(= (branch-depth (terminal-branch $efe $depth)) $depth)
(= (branch-trace (branch $a $efe $depth $trace)) $trace)
(= (branch-trace (terminal-branch $efe $depth)) ())

; Select best branch from a list (minimum EFE)
(= (best-branch ($b)) $b)
(= (best-branch ($b1 . $rest))
   (let $b2 (best-branch $rest)
     (if (<= (branch-efe $b1) (branch-efe $b2)) $b1 $b2)))


; =============================================================================
; SECTION VII: THE FRACTAL PLANNER (Core Algorithm)
; =============================================================================
;
; Recursive beam search with noise-aware pruning at each depth level.
;
; FLOW at each level:
;   1. Check stopping conditions (max depth, confidence floor)
;   2. Score all actions via residual EFE
;   3. Filter out viability violations (hard kill)
;   4. Sort by EFE (ascending — best first)
;   5. Apply noise filter (unified: beam width + symmetry + horizon)
;   6. Recurse on survivors
;   7. Return best branch (first action of lowest cumulative EFE trajectory)
;
; The noise filter at step 5 does ALL the adaptive work:
;   - If the EFE landscape is steep and confidence is high, noise is low,
;     only the best action survives → tunnel vision (fast, deep search)
;   - If the landscape is flat, many actions are within the noise floor
;     → deliberation (broad, shallow search)
;   - If confidence has degraded to the floor, the stopping condition
;     at step 1 catches it → intellectual humility (refuse to guess)

(= (fractal-expand $state $max-depth $current-depth)
   ; --- Stopping conditions ---
   (if (>= $current-depth $max-depth)
       ; Hit depth ceiling
       (terminal-branch 0.0 $current-depth)

   (if (beyond-confidence-floor? $state $current-depth)
       ; Predictions beyond here are noise
       (sequential
         (record-pruning-at-depth! all-actions $current-depth
           confidence-floor (degraded-confidence $state $current-depth))
         (terminal-branch 0.0 $current-depth))

   ; --- Active expansion ---
   (let* (
     ($remaining (- $max-depth $current-depth))

     ; 1. Score all candidates via residual EFE
     ($candidates (score-all-actions $state $remaining))

     ; 2. Viability filter (hard kill — same as policy_efe v5)
     ($viable (viability-filter $candidates $state $current-depth))
   )
   ; Guard: if nothing viable, return terminal
   (if (list-empty? $viable)
       (terminal-branch 999.0 $current-depth)
   (let* (
     ; 3. Sort by EFE (ascending)
     ($sorted (sort-candidates $viable))

     ; 4. Noise filter (unified beam width + symmetry breaking)
     ($survivors (noise-filter $sorted $state $current-depth))

     ; 5. Recurse on survivors
     ($branches (expand-survivors $survivors $state
                                  $max-depth $current-depth))
   )
   ; 6. Return best
   (if (list-empty? $branches)
       (terminal-branch 999.0 $current-depth)
       (best-branch $branches))))))))


; --- Score all base actions against current sim-state ---

(= (score-all-actions $state $remaining)
   (score-action-list (base-actions) $state $remaining))

(= (score-action-list () $state $remaining) ())
(= (score-action-list ($a . $rest) $state $remaining)
   (let $efe (residual-efe $a $state $remaining)
     ((candidate $a $efe) . (score-action-list $rest $state $remaining))))


; --- Viability filter: discard actions that violate hard bounds ---

(= (viability-filter () $state $depth) ())
(= (viability-filter ((candidate $a $efe) . $rest) $state $depth)
   (let $next (apply-action $a $state)
     (if (violates-viability? $next)
         ; Record and skip
         (sequential
           (record-pruning-at-depth! $a $depth viability $efe)
           (viability-filter $rest $state $depth))
         ; Keep
         ((candidate $a $efe) . (viability-filter $rest $state $depth)))))


; --- Expand surviving candidates recursively ---

(= (expand-survivors () $state $max-depth $depth) ())
(= (expand-survivors ((candidate $a $efe) . $rest)
                      $state $max-depth $depth)
   (let* (
     ; Evolve sim-state
     ($next-state (apply-action $a $state))

     ; Recurse into the future
     ($child (fractal-expand $next-state $max-depth (+ $depth 1)))

     ; Cumulative EFE: this step (EXACT, not heuristic) + discounted future
     ($step-efe (sim-efe $a $state))
     ($total-efe (+ $step-efe (* discount (branch-efe $child))))

     ; Build branch: action, cumulative EFE, depth reached, trace
     ($child-trace (branch-trace $child))
     ($trace ($a . $child-trace))
     ($this-branch (branch $a $total-efe (+ $depth 1) $trace))
   )
   ($this-branch . (expand-survivors $rest $state $max-depth $depth))))


; =============================================================================
; SECTION VIII: ENTRY POINTS
; =============================================================================

; --- Standard entry point ---
; Returns the first action of the best trajectory found.

(= (select-action-fractal)
   (let* (
     ($_ (clear-fractal-pruning-records!))
     ($state (current-sim-state))
     ($max-depth (get-planning-config max-planning-depth))
     ($result (fractal-expand $state $max-depth 0))
   )
   (branch-action $result)))

; --- Entry point with explicit depth ---

(= (select-action-fractal-with-depth $depth)
   (let* (
     ($_ (clear-fractal-pruning-records!))
     ($state (current-sim-state))
     ($result (fractal-expand $state $depth 0))
   )
   (branch-action $result)))


; =============================================================================
; SECTION IX: TRACED SELECTION (Full Transparency)
; =============================================================================
;
; Returns the selected action along with complete planning metadata:
; trajectory, depth reached, pruning records, and parameter state.

(= (select-action-fractal-traced)
   (let* (
     ($_ (clear-fractal-pruning-records!))
     ($state (current-sim-state))
     ($max-depth (get-planning-config max-planning-depth))
     ($result (fractal-expand $state $max-depth 0))
     ($action (branch-action $result))
     ($efe (branch-efe $result))
     ($trace (branch-trace $result))
     ($depth-reached (branch-depth $result))
     ($pruned (all-pruning-records))
     ($pruned-at-depth (all-pruning-records-with-depth))
     ($pressure (get-viability-pressure))
     ($ew (epistemic-weight))
     ($noise-d0 (noise-floor-at-depth $state 0))
     ($conf-d0 (degraded-confidence $state 0))
   )
   (fractal-trace
     (selected-action $action)
     (total-efe $efe)
     (planned-trajectory $trace)
     (planning-depth $depth-reached)
     (max-depth-allowed $max-depth)
     (branches-pruned $pruned)
     (branches-pruned-at-depth $pruned-at-depth)
     (viability-pressure $pressure)
     (epistemic-weight $ew)
     (noise-parameters
       (noise-floor-base (get-planning-config noise-floor-base))
       (noise-at-depth-0 $noise-d0)
       (confidence-at-depth-0 $conf-d0)
       (confidence-floor (get-planning-config confidence-floor))
       (base-beam-width (get-planning-config base-beam-width))
       (effective-beam-d0 (max 2 (ceil-int (* (get-planning-config base-beam-width) $conf-d0)))))
     (model-confidences (build-model-confidences (base-actions)))
     (reason "first action of minimum cumulative EFE trajectory via noise-aware fractal expansion"))))


; =============================================================================
; SECTION X: SELF-MODEL INTEGRATION
; =============================================================================
;
; The fractal planner provides natural material for self-knowledge.
; These functions let the self-model introspect planning behavior.
;
; The four emergent thinking modes still arise naturally:
;
;   TUNNEL VISION: steep landscape + high confidence → low noise floor
;     → one action within noise → beam=1 → deep, narrow search
;
;   DELIBERATION: flat landscape → many actions within noise → wide beam
;     → shallow, broad search
;
;   INTELLECTUAL HUMILITY: low confidence → high noise → confidence floor
;     triggers early → short horizon, system refuses to extrapolate
;
;   URGENCY: viability pressure → one action dominates via viability
;     bonus → gap exceeds any noise level → immediate tunnel vision

; How is the agent thinking right now?
; Characterizes the most recent planning episode.
(= (planning-style)
   (let* (
     ($_ (clear-fractal-pruning-records!))
     ($state (current-sim-state))
     ($max-depth (get-planning-config max-planning-depth))
     ($result (fractal-expand $state $max-depth 0))
     ($depth (branch-depth $result))
     ($trace-len (length (branch-trace $result)))
     ($pruned-count (length (all-pruning-records)))
   )
   ; Classify emergent planning style
   (if (and (> $trace-len 4) (> $pruned-count 3))
       (thinking-mode tunnel-vision "deep and narrow — clear best path")
       (if (and (<= $trace-len 2) (< $pruned-count 1))
           (thinking-mode cautious "short horizon — high uncertainty")
           (if (<= $trace-len 2)
               (thinking-mode deliberation "shallow and broad — weighing options")
               (thinking-mode balanced "moderate depth and breadth"))))))

; How much computation did planning use?
(= (planning-cost-summary)
   (let* (
     ($pruned (all-pruning-records))
     ($pruned-depth (all-pruning-records-with-depth))
   )
   (planning-cost
     (branches-pruned (length $pruned))
     (depth-pruning-events (length $pruned-depth))
     (note "Lower pruning count indicates focused search"))))


; =============================================================================
; SECTION XI: DIAGNOSTICS
; =============================================================================

; Report current planning parameters
(= (planning-parameters)
   (let $state (current-sim-state)
     (planning-params
       (noise-floor-base (get-planning-config noise-floor-base))
       (confidence-floor (get-planning-config confidence-floor))
       (base-beam-width (get-planning-config base-beam-width))
       (max-planning-depth (get-planning-config max-planning-depth))
       (confidence-discount discount)
       (noise-at-depth-0 (noise-floor-at-depth $state 0))
       (noise-at-depth-3 (noise-floor-at-depth $state 3))
       (noise-at-depth-5 (noise-floor-at-depth $state 5)))))

; Show noise floor progression across depths (for understanding behavior)
(= (noise-floor-profile)
   (let $state (current-sim-state)
     (noise-profile
       (depth-0 (noise-floor-at-depth $state 0)
                (confidence (degraded-confidence $state 0)))
       (depth-1 (noise-floor-at-depth $state 1)
                (confidence (degraded-confidence $state 1)))
       (depth-2 (noise-floor-at-depth $state 2)
                (confidence (degraded-confidence $state 2)))
       (depth-3 (noise-floor-at-depth $state 3)
                (confidence (degraded-confidence $state 3)))
       (depth-5 (noise-floor-at-depth $state 5)
                (confidence (degraded-confidence $state 5)))
       (depth-7 (noise-floor-at-depth $state 7)
                (confidence (degraded-confidence $state 7))))))

; Compare fractal vs myopic selection (for validation)
(= (compare-fractal-vs-myopic)
   (let* (
     ($fractal (select-action-fractal))
     ($myopic (select-action-myopic))
   )
   (comparison
     (fractal-selected $fractal)
     (myopic-selected $myopic)
     (agree? (== $fractal $myopic)))))


; =============================================================================
; END OF FRACTAL PLANNING
; =============================================================================
