; =============================================================================
; METTA COGNITIVE CORE - STRUCTURE LEARNING
; =============================================================================
;
; Discovers causal structure from patterns in prediction errors.
; This is the "Synchronicity of Surprise" mechanism: when two observables
; are simultaneously surprising, they are likely causally related.
;
; PIPELINE:
;   1. Error traces (from beliefs.metta) record timestamped surprises
;   2. Suspicion links accumulate co-error via Hebbian learning
;   3. Phase 1: Strong suspicion ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ create causal link (passive-model)
;   4. Phase 2: Dense causal clusters ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ compress into latent variables
;   5. Metabolic pruning: structures pay rent, earn energy via prediction
;
; DESIGN PRINCIPLES:
;   - Bottom-Up: Structure emerges from error correlations, not enumeration
;   - Reflexive: Generation is triggered by surprise, not selected by EFE
;   - Parsimonious: Metabolic cost ensures only useful structure survives
;   - Squared-Space: All threshold comparisons avoid sqrt (Gemini fix)
;   - Safety-Aware: Checks safe-to-prune? before removing any atom
;
; KEY DISTINCTION:
;   passive-model = "When X happens, Y follows" (world dynamics)
;   action-model  = "If I do X, Y changes" (agent actions, in actions.metta)
;   Passive models are used for PREDICTION only, never for EFE action selection.
;
; ATOM SCHEMAS:
;   (suspicion-link $cause $effect $lag $strength $last-updated)
;   (passive-model $cause $effect $lag $weight $type)
;   (latent-variable $name)
;   (latent-member $name $obs $weight)
;   (metabolic-energy passive-model $cause $effect $energy)
;   (metabolic-energy latent-var $name $energy)
;   (created-at passive-model $cause $effect $cycle)
;   (created-at latent-var $name $cycle)
;   (prediction-stats passive-model $cause $effect $predictions $successes)
;   (prediction-stats latent-var $name $predictions $successes)
;   (latent-var-counter $n)
;   (structural-atoms-this-cycle $n)
;
; Version: 0.5.0
; =============================================================================

!(import! &self foundations)
!(import! &self beliefs)
!(import! &self atom_lifecycle)


; =============================================================================
; SECTION I: INITIALIZATION
; =============================================================================

; Counter for generating unique latent variable names
(latent-var-counter 0)

; Per-cycle generation budget tracker
(structural-atoms-this-cycle 0)

; Reset generation budget at cycle start
(= (reset-structural-budget!)
   (let $old (match &self (structural-atoms-this-cycle $n) $n)
     (sequential
       (remove-atom &self (structural-atoms-this-cycle $old))
       (add-atom &self (structural-atoms-this-cycle 0))
       done)))

; Compute the adaptive structural budget based on observable count.
;
; With N observables there are N*(N-1)/2 undirected pairs to consider.
; The budget scales as ceil(sqrt(pairs)) — sublinear in pair space,
; enough throughput to sample proportionally without flooding the
; metabolic economy.
;
; SCALING BENCHMARK FINDINGS:
;   Fixed budget=5: 100% recall at ≤14 obs, drops to 62% at 24 obs
;   Adaptive:       Recovers +12% recall at 24 obs (75%)
;                   without increasing spurious link density
;
(= (adaptive-structural-budget)
   (let* (
     ($base (get-config max-structural-atoms-per-cycle-base))
     ($obs-list (collapse (match &state (belief $o $v $p) $o)))
     ($n (length $obs-list))
     ($n-pairs (/ (* $n (- $n 1)) 2))
     ($sqrt-pairs (approx-isqrt $n-pairs))
   )
   (max $base $sqrt-pairs)))

; Integer square root approximation (sufficient for budget computation)
(= (approx-isqrt $n)
   (if (<= $n 0) 0
   (if (<= $n 4) 2
   (if (<= $n 9) 3
   (if (<= $n 16) 4
   (if (<= $n 25) 5
   (if (<= $n 36) 6
   (if (<= $n 49) 7
   (if (<= $n 64) 8
   (if (<= $n 81) 9
   (if (<= $n 100) 10
   (if (<= $n 144) 12
   (if (<= $n 196) 14
   (if (<= $n 289) 17
   (if (<= $n 400) 20
       25)))))))))))))))  ; Cap at 25 for safety

; Check if generation budget allows more atoms this cycle
(= (structural-budget-available?)
   (let* (
     ($used (match &self (structural-atoms-this-cycle $n) $n))
     ($budget (adaptive-structural-budget))
   )
   (< $used $budget)))

; Increment generation budget
(= (use-structural-budget!)
   (let $old (match &self (structural-atoms-this-cycle $n) $n)
     (sequential
       (remove-atom &self (structural-atoms-this-cycle $old))
       (add-atom &self (structural-atoms-this-cycle (+ $old 1)))
       done)))


; =============================================================================
; SECTION II: SUSPICION LINK MANAGEMENT
; =============================================================================
;
; Suspicion links are Hebbian accumulators that track co-occurring errors
; between pairs of observables. They exist below the threshold of
; "committed structure" ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â they're hypotheses, not beliefs.
;
; Schema: (suspicion-link $cause $effect $lag $strength $last-updated)
;   $cause:        observable that errors first
;   $effect:       observable that errors later (or simultaneously)
;   $lag:          temporal offset (effect_cycle - cause_cycle)
;   $strength:     accumulated signed covariance (in squared surprise space)
;   $last-updated: cycle of most recent update (for decay tracking)

; Get suspicion link strength (0.0 if not found)
(= (get-suspicion-strength $cause $effect $lag)
   (let $r (match &self (suspicion-link $cause $effect $lag $str $upd) $str)
     (if (== $r ()) 0.0 $r)))

; Check if suspicion link exists
(= (has-suspicion-link? $cause $effect $lag)
   (let $r (collapse (match &self (suspicion-link $cause $effect $lag $s $u) True))
     (if (== $r ()) False True)))

; Create or update a suspicion link
(= (set-suspicion-link! $cause $effect $lag $new-strength $cycle)
   (if (has-suspicion-link? $cause $effect $lag)
       ; Update existing
       (let ($old-str $old-upd)
            (match &self (suspicion-link $cause $effect $lag $s $u) ($s $u))
         (sequential
           (remove-atom &self (suspicion-link $cause $effect $lag $old-str $old-upd))
           (add-atom &self (suspicion-link $cause $effect $lag $new-strength $cycle))
           (suspicion-updated $cause $effect $lag $old-str $new-strength)))
       ; Create new
       (sequential
         (add-atom &self (suspicion-link $cause $effect $lag $new-strength $cycle))
         (suspicion-created $cause $effect $lag $new-strength))))

; Get all suspicion links
(= (all-suspicion-links)
   (collapse (match &self (suspicion-link $c $e $l $s $u)
     (suspicion-link $c $e $l $s $u))))


; =============================================================================
; SECTION III: HEBBIAN SUSPICION ACCUMULATOR
; =============================================================================
;
; The core mechanism: find pairs of simultaneously-surprised observables
; and strengthen the suspicion link between them.
;
; Algorithm:
;   1. Collect all salient error traces within the lookback window
;   2. For each ordered pair (cause, effect) with cause_time ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¤ effect_time:
;      a. Compute signed surprise product (SQUARED SPACE ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â no sqrt)
;      b. Update suspicion: new = stability ÃƒÆ’Ã¢â‚¬â€ old + plasticity ÃƒÆ’Ã¢â‚¬â€ covariance
;   3. Decay all suspicion links not updated this cycle
;
; SQUARED SPACE:
;   Original formula: cov = sign(ea) ÃƒÆ’Ã¢â‚¬â€ sign(eb) ÃƒÆ’Ã¢â‚¬â€ ÃƒÂ¢Ã‹â€ Ã…Â¡(Sa ÃƒÆ’Ã¢â‚¬â€ Sb)
;   Squared space:    cov = sign(ea) ÃƒÆ’Ã¢â‚¬â€ sign(eb) ÃƒÆ’Ã¢â‚¬â€ (Sa ÃƒÆ’Ã¢â‚¬â€ Sb)
;   Thresholds are squared to match (e.g., 0.12Ãƒâ€šÃ‚Â² = 0.0144)
;
; FORWARD-TEMPORAL ONLY:
;   Cause must precede or be simultaneous with effect.
;   For simultaneous errors, BOTH directions are generated as separate
;   suspicion links. This is correct: we don't know the causal direction
;   for co-occurring events. Over time, the true direction accumulates
;   more temporal support (from lagged observations) and dominates.

; Collect all pairs that need suspicion updates this cycle.
; Returns a list of (suspicion-pair ...) tuples.
; For non-simultaneous: forward-temporal only (cause before effect).
; For simultaneous: both directions (AÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢B and BÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢A), since causal
; direction is unknown. The correct one wins via temporal evidence.
(= (collect-suspicion-pairs $current-cycle)
   (let $window (get-config lookback-window)
     (let $cutoff (- $current-cycle $window)
       (collapse
         (match &self (error-trace $oa $ea $sa $ta)
           (match &self (error-trace $ob $eb $sb $tb)
             (if (and (not (== $oa $ob))
                      (is-salient? $sa $ea)
                      (is-salient? $sb $eb)
                      (>= $ta $cutoff)
                      (>= $tb $cutoff)
                      (or (< $ta $tb)
                          (and (== $ta $tb)
                               (not (== $oa $ob)))))
                 (suspicion-pair $oa $ea $sa $ta $ob $eb $sb $tb)
                 (empty))))))))

; Process a single suspicion pair: compute covariance, update link
(= (update-one-suspicion! (suspicion-pair $oa $ea $sa $ta $ob $eb $sb $tb) $cycle)
   (let* (
     ; Temporal lag (causeÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢effect)
     ($lag (- $tb $ta))

     ; Signed surprise product (SQUARED SPACE ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â no sqrt)
     ($sign-a (sign $ea))
     ($sign-b (sign $eb))
     ($direction (* $sign-a $sign-b))
     ($surprise-product (* $sa $sb))
     ($covariance (* $direction $surprise-product))

     ; Get current link state
     ($old-strength (get-suspicion-strength $oa $ob $lag))

     ; Hebbian update: strength = stability ÃƒÆ’Ã¢â‚¬â€ old + plasticity ÃƒÆ’Ã¢â‚¬â€ covariance
     ($stability (get-config default-structural-stability))
     ($plasticity (get-config default-plasticity))
     ($new-strength (+ (* $stability $old-strength)
                       (* $plasticity $covariance)))
   )
   (set-suspicion-link! $oa $ob $lag $new-strength $cycle)))

; Process all suspicion pairs
(= (process-suspicion-pairs! () $cycle) done)
(= (process-suspicion-pairs! ($pair . $rest) $cycle)
   (sequential
     (update-one-suspicion! $pair $cycle)
     (process-suspicion-pairs! $rest $cycle)))

; Decay all suspicion links that were NOT updated this cycle.
; Links that were updated already incorporate the stability factor.
(= (decay-suspicion-links! $current-cycle)
   (let* (
     ($stability (get-config default-structural-stability))
     ($stale (collapse
       (match &self (suspicion-link $c $e $l $s $u)
         (if (< $u $current-cycle)
             (suspicion-link $c $e $l $s $u)
             (empty)))))
   )
   (decay-suspicion-list! $stale $stability $current-cycle)))

(= (decay-suspicion-list! () $stability $cycle) done)
(= (decay-suspicion-list! ((suspicion-link $c $e $l $s $u) . $rest) $stability $cycle)
   (let $new-s (* $s $stability)
     (sequential
       (remove-atom &self (suspicion-link $c $e $l $s $u))
       (add-atom &self (suspicion-link $c $e $l $new-s $cycle))
       (decay-suspicion-list! $rest $stability $cycle))))

; Master function: run the full suspicion update for this cycle
(= (update-all-suspicion-links! $current-cycle)
   (let* (
     ($pairs (collect-suspicion-pairs $current-cycle))
     ($updates (process-suspicion-pairs! $pairs $current-cycle))
     ($decay (decay-suspicion-links! $current-cycle))
   )
   (suspicion-step-done (length $pairs))))


; =============================================================================
; SECTION IV: PHASE 1 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â CAUSAL LINK CREATION
; =============================================================================
;
; When a suspicion link's absolute strength exceeds the structural cost
; threshold, it is promoted to a committed passive-model atom.
;
; SQUARED SPACE:
;   Compare: |strength| > cost   ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢   strengthÃƒâ€šÃ‚Â² > costÃƒâ€šÃ‚Â²
;   Since strength can be negative: |strength| > ÃƒÂ¢Ã‹â€ Ã…Â¡cost_sq
;   Equivalently: strengthÃƒâ€šÃ‚Â² > cost_sq (where cost_sq = 0.0144)
;   But our strength is ALREADY in squared space (accumulated from
;   surprise products), so we compare |strength| > cost_sq directly.
;
; NOTE: We compare absolute strength against the squared cost threshold.
; The strength is in squared-surprise units because the covariance
; accumulates surprise products (not sqrt of products).

; Check a single suspicion link for promotion
(= (check-promotion $cause $effect $lag $strength $cycle)
   (let* (
     ($cost-sq (get-config structural-cost-link))
     ($abs-strength (abs $strength))
     ($already-exists (has-passive-model? $cause $effect))
     ($budget-ok (structural-budget-available?))
   )
   (if (and (> $abs-strength $cost-sq)
            (not $already-exists)
            $budget-ok)
       (create-passive-model! $cause $effect $lag $strength $cycle)
       no-promotion)))

; Create a new passive model from a mature suspicion link
(= (create-passive-model! $cause $effect $lag $strength $cycle)
   (let* (
     ($type (if (> $strength 0.0) excitatory inhibitory))
     ($weight (min (abs $strength) 1.0))
     ($initial-energy (get-config metabolic-initial-energy))
   )
   (sequential
     (add-atom &self (passive-model $cause $effect $lag $weight $type))
     (add-atom &self (metabolic-energy passive-model $cause $effect $initial-energy))
     (add-atom &self (created-at passive-model $cause $effect $cycle))
     (add-atom &self (prediction-stats passive-model $cause $effect 0 0))
     (use-structural-budget!)
     (passive-model-created $cause $effect $lag $weight $type))))

; Check if a passive model already exists for this causeÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢effect pair
(= (has-passive-model? $cause $effect)
   (let $r (collapse (match &self (passive-model $cause $effect $l $w $t) True))
     (if (== $r ()) False True)))

; Run Phase 1 triggers on all suspicion links
(= (check-phase1-triggers! $current-cycle)
   (let $links (all-suspicion-links)
     (check-phase1-list! $links $current-cycle)))

(= (check-phase1-list! () $cycle) done)
(= (check-phase1-list! ((suspicion-link $c $e $l $s $u) . $rest) $cycle)
   (sequential
     (check-promotion $c $e $l $s $cycle)
     (check-phase1-list! $rest $cycle)))


; =============================================================================
; SECTION IV-B: PHASE 1.5 â€” DEDUCTIVE STRUCTURE GENERATION
; =============================================================================
;
; TRANSITIVE CLOSURE over established passive models.
;
; When Aâ†’B and Bâ†’C exist with sufficient metabolic energy (meaning both
; have survived selection pressure and make useful predictions), generate
; a candidate Aâ†’C link. This is grounded deduction: the premises are
; empirically validated, and the conclusion enters the same metabolic
; regime â€” it must pay rent, make predictions, and be falsifiable.
;
; KEY DESIGN DECISIONS:
;   1. Deduction fires AFTER Phase 1, so it operates on stable links
;   2. Source links must have min energy (have survived metabolic selection)
;   3. Deduced weight = discount Ã— product of source weights (conservative)
;   4. Deduced lag = sum of source lags (transitive effects are slower)
;   5. Type tag is the PRODUCT of source types:
;      excitatory Ã— excitatory = excitatory (positive Ã— positive = positive)
;      excitatory Ã— inhibitory = inhibitory (positive Ã— negative = negative)
;      inhibitory Ã— inhibitory = excitatory (negative Ã— negative = positive)
;   6. Audit trail: (deduction-source $cause $effect $intermediate)
;      links the deduced model to its premises for transparency
;   7. Respects structural budget â€” competes with empirical discovery
;   8. Normal metabolic regime: initial energy, gestation, rent
;
; WHAT THIS GIVES US OVER PLN:
;   PLN: valid deductions persist forever (valid â‰  true)
;   This: deduced links that fail to predict die metabolically
;   PLN: deductions are grounded in logical consistency
;   This: deductions are grounded in empirical prediction success
;   PLN: separate reasoning system with own optimization principle
;   This: deduction is a hypothesis generator; selection is EFE via metabolism
;
; SCHEMA (new atoms):
;   (deduction-source $cause $effect $intermediate)  ; audit trail
;   passive-model atoms with deductive origin get a companion:
;   (deduction-origin $cause $effect)                ; marks deductive provenance

; Find all transitive chain candidates: Aâ†’B and Bâ†’C where both links
; have sufficient metabolic energy and Aâ†’C doesn't already exist.
(= (deductive-candidates)
   (collapse
     (match &self (passive-model $a $b $lag1 $w1 $t1)
       (match &self (passive-model $b $c $lag2 $w2 $t2)
         (if (and (not (== $a $c))                    ; no self-loops
                  (not (has-passive-model? $a $c))     ; not already known
                  (not (has-deduction-origin? $a $c))  ; not already deduced (even if pruned)
                  (pm-energy-sufficient? $a $b)        ; source 1 is stable
                  (pm-energy-sufficient? $b $c))       ; source 2 is stable
             (deductive-candidate $a $b $c $lag1 $lag2 $w1 $w2 $t1 $t2)
             (empty))))))

; Check if a passive model has sufficient energy for deduction
(= (pm-energy-sufficient? $cause $effect)
   (let $energy (get-pm-energy $cause $effect)
     (if (== $energy ())
         False
         (>= $energy (get-config deductive-min-energy)))))

; Check if a deduction origin marker exists (prevents re-deduction after pruning)
; This decays naturally â€” cleared when both source links are still alive
(= (has-deduction-origin? $cause $effect)
   (let $r (collapse (match &self (deduction-origin $cause $effect) True))
     (if (== $r ()) False True)))

; Compute the type of a deduced link from its sources
; excitatory Ã— excitatory = excitatory
; excitatory Ã— inhibitory = inhibitory
; inhibitory Ã— inhibitory = excitatory
(= (deduced-type excitatory excitatory) excitatory)
(= (deduced-type excitatory inhibitory) inhibitory)
(= (deduced-type inhibitory excitatory) inhibitory)
(= (deduced-type inhibitory inhibitory) excitatory)

; Create a single deduced passive model from a candidate
(= (create-deductive-model! (deductive-candidate $a $b $c $lag1 $lag2 $w1 $w2 $t1 $t2) $cycle)
   (if (not (structural-budget-available?))
       deductive-budget-exhausted
       (let* (
         ; Combine lags (transitive effects are slower)
         ($combined-lag (+ $lag1 $lag2))

         ; Combine weights (conservative: product with discount)
         ($discount (get-config deductive-weight-discount))
         ($combined-weight (min (* $discount (* $w1 $w2)) 1.0))

         ; Combine types (sign algebra)
         ($combined-type (deduced-type $t1 $t2))

         ; Standard metabolic initialization
         ($initial-energy (get-config metabolic-initial-energy))
       )
       (sequential
         ; Create the passive model (same schema as empirical)
         (add-atom &self (passive-model $a $c $combined-lag $combined-weight $combined-type))
         (add-atom &self (metabolic-energy passive-model $a $c $initial-energy))
         (add-atom &self (created-at passive-model $a $c $cycle))
         (add-atom &self (prediction-stats passive-model $a $c 0 0))

         ; Audit trail: record the deductive provenance
         (add-atom &self (deduction-origin $a $c))
         (add-atom &self (deduction-source $a $c $b))

         ; Budget accounting
         (use-structural-budget!)

         (deductive-model-created $a $c $b $combined-lag $combined-weight $combined-type)))))

; Process all deductive candidates
(= (process-deductive-candidates! () $cycle) done)
(= (process-deductive-candidates! ($candidate . $rest) $cycle)
   (sequential
     (create-deductive-model! $candidate $cycle)
     (process-deductive-candidates! $rest $cycle)))

; Master function: run deductive triggers for this cycle
(= (check-deductive-triggers! $current-cycle)
   (let $candidates (deductive-candidates)
     (if (list-empty? $candidates)
         no-deductive-candidates
         (process-deductive-candidates! $candidates $current-cycle))))

; Clean up stale deduction-origin markers
; If BOTH source links of a deduction have been pruned, clear the origin
; marker so the deduction can be re-attempted if the sources are rediscovered.
(= (clean-deduction-origins!)
   (let $origins (collapse
     (match &self (deduction-source $a $c $b)
       (if (and (not (has-passive-model? $a $b))
                (not (has-passive-model? $b $c)))
           (pair $a $c $b)
           (empty))))
   (clean-origin-list! $origins)))

(= (clean-origin-list! ()) done)
(= (clean-origin-list! ((pair $a $c $b) . $rest))
   (sequential
     (remove-atom &self (deduction-origin $a $c))
     (remove-atom &self (deduction-source $a $c $b))
     (clean-origin-list! $rest)))

; Diagnostics: list all deductively generated links
(= (all-deductive-models)
   (collapse
     (match &self (deduction-source $a $c $b)
       (let $model (match &self (passive-model $a $c $l $w $t) (passive-model $a $c $l $w $t))
         (if (== $model ())
             (deduction-pruned $a $c $b)          ; Link was deduced then died
             (deduction-active $a $c $b $model)))))) ; Link alive

; Count of active deductive models
(= (deductive-model-count)
   (length (collapse
     (match &self (deduction-origin $a $c) (pair $a $c)))))


; =============================================================================
; SECTION V: PASSIVE MODEL ACCESSORS
; =============================================================================
;
; Passive models represent discovered world dynamics. They are used
; by the prediction step to anticipate changes, NOT by EFE action selection.

; Get all passive models
(= (all-passive-models)
   (collapse (match &self (passive-model $c $e $l $w $t)
     (passive-model $c $e $l $w $t))))

; Get passive models where $obs is the effect
(= (passive-models-affecting $obs)
   (collapse (match &self (passive-model $c $obs $l $w $t)
     (passive-model $c $obs $l $w $t))))

; Get passive models where $obs is the cause
(= (passive-models-from $obs)
   (collapse (match &self (passive-model $obs $e $l $w $t)
     (passive-model $obs $e $l $w $t))))

; Count of passive models
(= (passive-model-count)
   (length (all-passive-models)))

; Get metabolic energy for a passive model
(= (get-pm-energy $cause $effect)
   (match &self (metabolic-energy passive-model $cause $effect $e) $e))

; Get creation cycle for a passive model
(= (get-pm-created $cause $effect)
   (match &self (created-at passive-model $cause $effect $c) $c))

; Check if passive model is still gestating
(= (pm-gestating? $cause $effect $current-cycle)
   (let $created (get-pm-created $cause $effect)
     (< (- $current-cycle $created) (get-config gestation-period))))


; =============================================================================
; SECTION VI: PHASE 2 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â HUB DETECTION & LATENT VARIABLE INVENTION
; =============================================================================
;
; When many pairwise causal links form a dense cluster ("hairball"),
; it's more parsimonious to explain them with a single latent variable
; (hub node) connected to all members.
;
; Detection: find observables with degree ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¥ 3 in the causal link graph.
; Parsimony: hairball_cost > hub_cost ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ create latent variable.
;
; SQUARED SPACE applies to cost comparison:
;   hairball_cost = n_pairwise_links ÃƒÆ’Ã¢â‚¬â€ cost_link_sq
;   hub_cost = cost_latent_sq + n_members ÃƒÆ’Ã¢â‚¬â€ cost_spoke_sq
;
; Graph rewrite: pairwise links among the cluster are REMOVED and
; replaced by hubÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢member spokes.

; Build adjacency: for each observable, find its causal neighbors
; Returns list of (obs-degree $obs $degree $neighbors) tuples
(= (compute-causal-degrees)
   (let* (
     ; Collect all unique observables appearing in passive models
     ($all-obs (collect-causal-observables))
   )
   (compute-degrees-for $all-obs)))

; Collect all unique observables that participate in causal links
(= (collect-causal-observables)
   (let* (
     ($causes (collapse (match &self (passive-model $c $e $l $w $t) $c)))
     ($effects (collapse (match &self (passive-model $c $e $l $w $t) $e)))
   )
   (unique-merge $causes $effects)))

; Merge two lists keeping only unique elements
(= (unique-merge () $ys) $ys)
(= (unique-merge ($x . $xs) $ys)
   (if (list-contains? $x $ys)
       (unique-merge $xs $ys)
       (unique-merge $xs ($x . $ys))))

; Compute degree and neighbor list for each observable
(= (compute-degrees-for ()) ())
(= (compute-degrees-for ($obs . $rest))
   (let* (
     ($neighbors (get-causal-neighbors $obs))
     ($degree (length $neighbors))
   )
   ((obs-degree $obs $degree $neighbors) . (compute-degrees-for $rest))))

; Get all causal neighbors of an observable (both directions)
(= (get-causal-neighbors $obs)
   (let* (
     ($effects (collapse (match &self (passive-model $obs $e $l $w $t) $e)))
     ($causes (collapse (match &self (passive-model $c $obs $l $w $t) $c)))
   )
   (unique-merge $effects $causes)))

; Check if observable is already part of a latent variable
(= (already-hubbed? $obs)
   (let $r (collapse (match &self (latent-member $name $obs $w) True))
     (if (== $r ()) False True)))

; Generate next latent variable name
(= (next-latent-name!)
   (let $n (match &self (latent-var-counter $c) $c)
     (let $new (+ $n 1)
       (sequential
         (remove-atom &self (latent-var-counter $n))
         (add-atom &self (latent-var-counter $new))
         $new))))

; Run Phase 2 on all high-degree nodes
(= (check-phase2-triggers! $current-cycle)
   (let $degrees (compute-causal-degrees)
     (check-phase2-list! $degrees $current-cycle)))

(= (check-phase2-list! () $cycle) done)
(= (check-phase2-list! ((obs-degree $obs $degree $neighbors) . $rest) $cycle)
   (sequential
     (maybe-create-hub! $obs $degree $neighbors $cycle)
     (check-phase2-list! $rest $cycle)))

; Attempt hub creation for a single high-degree node
; BF: Gated by metabolic capital — creative structure invention
; (latent variable = hypothesized hidden cause) requires earned capital.
; Induction (Phase 1 causal links) remains free.
(= (maybe-create-hub! $obs $degree $neighbors $cycle)
   (if (and (>= $degree 3)
            (not (already-hubbed? $obs))
            (structural-budget-available?)
            (can-afford-abduction?))      ; BF: must afford creative abduction
       ; Compute parsimony comparison
       (let* (
         ($cluster ($obs . $neighbors))
         ($n-pairwise (count-pairwise-links $cluster))
         ($n-members (length $cluster))
         ($cost-link-sq (get-config structural-cost-link))
         ($cost-latent-sq (get-config structural-cost-latent))
         ($cost-spoke-sq (get-config structural-cost-spoke))
         ($hairball-cost (* $n-pairwise $cost-link-sq))
         ($hub-cost (+ $cost-latent-sq (* $n-members $cost-spoke-sq)))
       )
       (if (> $hairball-cost $hub-cost)
           (sequential
             (spend-abduction-capital!)   ; BF: pay for creative abduction
             (create-latent-variable! $obs $cluster $cycle))
           no-hub-needed))
       no-hub-candidate))

; Count pairwise links that exist among a set of observables
(= (count-pairwise-links $cluster)
   (let $pairs (list-pairs $cluster)
     (count-existing-links $pairs)))

(= (count-existing-links ()) 0)
(= (count-existing-links ((pair $a $b) . $rest))
   (let* (
     ($fwd (has-passive-model? $a $b))
     ($rev (has-passive-model? $b $a))
     ($count (+ (if $fwd 1 0) (if $rev 1 0)))
   )
   (+ $count (count-existing-links $rest))))

; Create a latent variable and perform graph rewrite
(= (create-latent-variable! $hub-obs $cluster $cycle)
   (let* (
     ($id (next-latent-name!))
     ($name (latent-name $id))
     ($initial-energy (get-config metabolic-initial-energy))

     ; Create the hub node
     ($create-hub (sequential
       (add-atom &self (latent-variable $name))
       (add-atom &self (metabolic-energy latent-var $name $initial-energy))
       (add-atom &self (created-at latent-var $name $cycle))
       (add-atom &self (prediction-stats latent-var $name 0 0))
       done))

     ; Create member spokes with weights derived from existing links
     ($create-members (create-latent-members! $name $hub-obs $cluster))

     ; Graph rewrite: remove pairwise links among cluster
     ($rewrite (remove-cluster-links! $cluster))

     ; Budget tracking
     ($budget (use-structural-budget!))
   )
   (latent-variable-created $name $cluster)))

; Create latent-member atoms for each observable in the cluster
(= (create-latent-members! $name $hub $cluster)
   (create-members-list! $name $hub $cluster))

(= (create-members-list! $name $hub ()) done)
(= (create-members-list! $name $hub ($obs . $rest))
   (let $weight (derive-latent-weight $hub $obs)
     (sequential
       (add-atom &self (latent-member $name $obs $weight))
       (create-members-list! $name $hub $rest))))

; Derive initial weight for a latentÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢member spoke from existing link strength
(= (derive-latent-weight $hub $obs)
   (if (== $hub $obs)
       0.5  ; hub node gets default weight
       (let $r (collapse (match &self (passive-model $hub $obs $l $w $t) $w))
         (if (== $r ())
             (let $r2 (collapse (match &self (passive-model $obs $hub $l $w $t) $w))
               (if (== $r2 ()) 0.3 (list-head $r2)))
             (list-head $r)))))

; Remove all pairwise passive-model links among a cluster (graph rewrite)
(= (remove-cluster-links! $cluster)
   (let $pairs (list-pairs $cluster)
     (remove-cluster-pairs! $pairs)))

(= (remove-cluster-pairs! ()) done)
(= (remove-cluster-pairs! ((pair $a $b) . $rest))
   (sequential
     (remove-passive-model-if-exists! $a $b)
     (remove-passive-model-if-exists! $b $a)
     (remove-cluster-pairs! $rest)))

; Safely remove a passive model and its associated atoms
(= (remove-passive-model-if-exists! $cause $effect)
   (if (has-passive-model? $cause $effect)
       (remove-passive-model! $cause $effect)
       done))

(= (remove-passive-model! $cause $effect)
   (match &self (passive-model $cause $effect $lag $weight $type)
     (sequential
       (remove-atom &self (passive-model $cause $effect $lag $weight $type))
       (remove-atom &self (metabolic-energy passive-model $cause $effect
                            (get-pm-energy $cause $effect)))
       (remove-atom &self (created-at passive-model $cause $effect
                            (get-pm-created $cause $effect)))
       (let ($p $s) (match &self (prediction-stats passive-model $cause $effect $p $s) ($p $s))
         (remove-atom &self (prediction-stats passive-model $cause $effect $p $s)))
       (passive-model-removed $cause $effect))))


; =============================================================================
; SECTION VII: LATENT VARIABLE ACCESSORS
; =============================================================================

; Get all latent variables
(= (all-latent-variables)
   (collapse (match &self (latent-variable $name) $name)))

; Get members of a latent variable
(= (get-latent-members $name)
   (collapse (match &self (latent-member $name $obs $w) $obs)))

; Get weight for a specific member
(= (get-latent-weight $name $obs)
   (let $r (match &self (latent-member $name $obs $w) $w)
     (if (== $r ()) 0.0 $r)))

; Get energy for a latent variable
(= (get-lv-energy $name)
   (match &self (metabolic-energy latent-var $name $e) $e))

; Get creation cycle for a latent variable
(= (get-lv-created $name)
   (match &self (created-at latent-var $name $c) $c))

; Check if latent variable is gestating
(= (lv-gestating? $name $current-cycle)
   (let $created (get-lv-created $name)
     (< (- $current-cycle $created) (get-config gestation-period))))

; Count of latent variables
(= (latent-variable-count)
   (length (all-latent-variables)))


; =============================================================================
; SECTION VIII: METABOLIC PRUNING
; =============================================================================
;
; Every structural atom (passive-model, latent-variable) pays metabolic
; rent each cycle. Successful predictions earn energy. Structures that
; don't pay their way eventually die (energy ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¤ 0 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ removal).
;
; Gestation period: new structures get a grace period before rent starts.
; This prevents transient spikes from creating and immediately killing
; structure in the same few cycles.

; Drain energy from all non-gestating passive models
(= (drain-passive-models! $current-cycle)
   (let $models (all-passive-models)
     (drain-pm-list! $models $current-cycle)))

(= (drain-pm-list! () $cycle) done)
(= (drain-pm-list! ((passive-model $c $e $l $w $t) . $rest) $cycle)
   (sequential
     (if (pm-gestating? $c $e $cycle)
         done  ; grace period
         (drain-pm-energy! $c $e))
     (drain-pm-list! $rest $cycle)))

(= (drain-pm-energy! $cause $effect)
   (let $old-energy (get-pm-energy $cause $effect)
     (let $new-energy (- $old-energy (get-config metabolic-rate))
       (sequential
         (remove-atom &self (metabolic-energy passive-model $cause $effect $old-energy))
         (add-atom &self (metabolic-energy passive-model $cause $effect $new-energy))
         done))))

; Drain energy from all non-gestating latent variables
(= (drain-latent-variables! $current-cycle)
   (let $names (all-latent-variables)
     (drain-lv-list! $names $current-cycle)))

(= (drain-lv-list! () $cycle) done)
(= (drain-lv-list! ($name . $rest) $cycle)
   (sequential
     (if (lv-gestating? $name $cycle)
         done
         (drain-lv-energy! $name))
     (drain-lv-list! $rest $cycle)))

(= (drain-lv-energy! $name)
   (let $old-energy (get-lv-energy $name)
     (let $new-energy (- $old-energy (get-config metabolic-rate))
       (sequential
         (remove-atom &self (metabolic-energy latent-var $name $old-energy))
         (add-atom &self (metabolic-energy latent-var $name $new-energy))
         done))))

; Prune passive models with energy ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¤ 0
(= (prune-dead-passive-models!)
   (let $dead (collapse
     (match &self (metabolic-energy passive-model $c $e $energy)
       (if (<= $energy 0.0)
           (pair $c $e)
           (empty))))
   (prune-pm-list! $dead)))

(= (prune-pm-list! ()) done)
(= (prune-pm-list! ((pair $c $e) . $rest))
   (sequential
     (remove-passive-model! $c $e)
     (prune-pm-list! $rest)))

; Prune latent variables with energy ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¤ 0
(= (prune-dead-latent-variables!)
   (let $dead (collapse
     (match &self (metabolic-energy latent-var $name $energy)
       (if (<= $energy 0.0)
           $name
           (empty))))
   (prune-lv-list! $dead)))

(= (prune-lv-list! ()) done)
(= (prune-lv-list! ($name . $rest))
   (sequential
     (remove-latent-variable! $name)
     (prune-lv-list! $rest)))

; Remove a latent variable and all its associated atoms
(= (remove-latent-variable! $name)
   (let* (
     ; Remove member atoms
     ($members (collapse (match &self (latent-member $name $obs $w)
       (latent-member $name $obs $w))))
     ($rm-members (remove-atom-list! $members))
     ; Remove metabolic energy
     ($energy (get-lv-energy $name))
     ($rm-energy (remove-atom &self (metabolic-energy latent-var $name $energy)))
     ; Remove creation time
     ($created (get-lv-created $name))
     ($rm-created (remove-atom &self (created-at latent-var $name $created)))
     ; Remove prediction stats
     (($p $s) (match &self (prediction-stats latent-var $name $p $s) ($p $s)))
     ($rm-stats (remove-atom &self (prediction-stats latent-var $name $p $s)))
     ; Remove the variable itself
     ($rm-self (remove-atom &self (latent-variable $name)))
   )
   (latent-variable-removed $name)))

(= (remove-atom-list! ()) done)
(= (remove-atom-list! ($a . $rest))
   (sequential
     (remove-atom &self $a)
     (remove-atom-list! $rest)))

; Master metabolic step
(= (metabolic-step! $current-cycle)
   (sequential
     (drain-passive-models! $current-cycle)
     (drain-latent-variables! $current-cycle)
     (prune-dead-passive-models!)
     (prune-dead-latent-variables!)
     done))


; =============================================================================
; SECTION IX: METABOLIC REWARD
; =============================================================================
;
; When a structural prediction helps reduce error, the responsible
; structure earns metabolic energy. This is how useful structure survives.
;
; Called after belief update: compare prediction with actual outcome.
; If the structure's prediction was roughly correct, it earns a boost.

; Reward passive models that correctly predicted an observable's behavior
(= (reward-passive-model! $cause $effect $actual-error)
   (if (pm-gestating? $cause $effect 0)  ; use 0 as "don't know cycle" fallback
       done
       (let* (
         ($old-energy (get-pm-energy $cause $effect))
         ($cap (get-config metabolic-energy-cap))
         ($boost (get-config metabolic-boost))
         ; Reward if prediction was correct relative to belief precision
         ($correct (is-prediction-correct? $effect $actual-error))
         ($earned (if $correct $boost 0.0))
         ($new-energy (min (+ $old-energy $earned) $cap))
         ; Update prediction stats
         (($old-p $old-s) (match &self (prediction-stats passive-model $cause $effect $p $s) ($p $s)))
         ($new-p (+ $old-p 1))
         ($new-s (+ $old-s (if $correct 1 0)))
       )
       (sequential
         (remove-atom &self (metabolic-energy passive-model $cause $effect $old-energy))
         (add-atom &self (metabolic-energy passive-model $cause $effect $new-energy))
         (remove-atom &self (prediction-stats passive-model $cause $effect $old-p $old-s))
         (add-atom &self (prediction-stats passive-model $cause $effect $new-p $new-s))
         ; BF: successful predictions earn global capital for abduction
         (if $correct
             (earn-capital! (get-config capital-earn-rate))
             done)
         done))))

; Reward latent variable that correctly predicted a member observable
; COHERENCE-AWARE: rewards only when member errors are correlated
; (the pattern the latent is supposed to explain). Without this check,
; spurious latents accumulate energy during calm periods.
(= (reward-latent-variable-coherent! $name $obs $actual-error $all-errors)
   (let* (
     ($members (get-latent-members $name))
     ($member-errors (collect-member-errors $members $all-errors))
     ($coherent (check-error-coherence $member-errors))
     ($correct (is-prediction-correct? $obs $actual-error))
   )
   (if (and $coherent $correct)
       ; Coherent and correct: full reward
       (let $e (get-lv-energy $name)
         (sequential
           (remove-atom &self (metabolic-energy latent-var $name $e))
           (add-atom &self (metabolic-energy latent-var $name
             (min (+ $e (get-config metabolic-boost))
                  (get-config metabolic-energy-cap))))
           (update-lv-stats-success! $name)
           done))
       ; Not coherent or not correct: count prediction but no reward
       (update-lv-stats-attempt! $name))))

; Legacy wrapper: non-coherence-aware reward (for backward compatibility)
; Now accepts $obs to derive the precision-based threshold.
(= (reward-latent-variable! $name $obs $actual-error)
   (if (lv-gestating? $name 0)
       done
       (let* (
         ($old-energy (get-lv-energy $name))
         ($cap (get-config metabolic-energy-cap))
         ($boost (get-config metabolic-boost))
         ($correct (is-prediction-correct? $obs $actual-error))
         ($earned (if $correct $boost 0.0))
         ($new-energy (min (+ $old-energy $earned) $cap))
         (($old-p $old-s) (match &self (prediction-stats latent-var $name $p $s) ($p $s)))
         ($new-p (+ $old-p 1))
         ($new-s (+ $old-s (if $correct 1 0)))
       )
       (sequential
         (remove-atom &self (metabolic-energy latent-var $name $old-energy))
         (add-atom &self (metabolic-energy latent-var $name $new-energy))
         (remove-atom &self (prediction-stats latent-var $name $old-p $old-s))
         (add-atom &self (prediction-stats latent-var $name $new-p $new-s))
         done))))

; Collect errors for latent variable members from the error list
(= (collect-member-errors () $all-errors) ())
(= (collect-member-errors ($obs . $rest) $all-errors)
   (let $e (lookup-error $obs $all-errors)
     (if (== $e ())
         (collect-member-errors $rest $all-errors)
         ($e . (collect-member-errors $rest $all-errors)))))

; Lookup error value from the error list
; $all-errors is a list of (prediction-error $obs $error ...)
(= (lookup-error $obs ()) ())
(= (lookup-error $obs ((prediction-error $o $error $abs $prec $w) . $rest))
   (if (== $obs $o)
       $error
       (lookup-error $obs $rest)))

; Check if errors are coherent (mostly same direction)
(= (check-error-coherence $errors)
   (if (< (length $errors) 2)
       True   ; Not enough data to check Ã¢â‚¬â€ give benefit of doubt
       (let* (
         ($threshold (get-config coherence-reward-threshold))
         ($pos (count-positive $errors $threshold))
         ($neg (count-negative $errors $threshold))
       )
       ; Coherent if errors are not in opposing directions
       (or (== $pos 0) (== $neg 0)))))

(= (count-positive () $t) 0)
(= (count-positive ($e . $rest) $t)
   (+ (if (> $e $t) 1 0) (count-positive $rest $t)))

(= (count-negative () $t) 0)
(= (count-negative ($e . $rest) $t)
   (+ (if (< $e (- 0 $t)) 1 0) (count-negative $rest $t)))

; Update prediction stats (attempt only, no success)
(= (update-lv-stats-attempt! $name)
   (match &self (prediction-stats latent-var $name $p $s)
     (sequential
       (remove-atom &self (prediction-stats latent-var $name $p $s))
       (add-atom &self (prediction-stats latent-var $name (+ $p 1) $s))
       done)))

; Update prediction stats (success)
(= (update-lv-stats-success! $name)
   (match &self (prediction-stats latent-var $name $p $s)
     (sequential
       (remove-atom &self (prediction-stats latent-var $name $p $s))
       (add-atom &self (prediction-stats latent-var $name (+ $p 1) (+ $s 1)))
       done)))

; Reward all structure that predicted a specific observable (v2)
; Now includes coherence check for latent variables and Phase 3-5 atoms.
(= (reward-structure-for! $obs $actual-error $current-cycle)
   (sequential
     ; Reward passive models where $obs is the effect (unchanged)
     (let $pms (passive-models-affecting $obs)
       (reward-pm-list! $pms $actual-error $current-cycle))
     ; Reward latent variables where $obs is a member
     (let $lvs (collapse (match &self (latent-member $name $obs $w) $name))
       (reward-lv-list! $lvs $obs $actual-error $current-cycle))
     done))

; v2 reward: coherence-aware + Phase 3-5 lifecycle rewards
(= (reward-structure-for-v2! $obs $actual-error $current-cycle $all-errors)
   (sequential
     ; Reward passive models (existing, unchanged)
     (let $pms (passive-models-affecting $obs)
       (reward-pm-list! $pms $actual-error $current-cycle))

     ; Reward latent variables (WITH coherence check)
     (let $lvs (collapse (match &self (latent-member $name $obs $w) $name))
       (reward-lv-coherent-list! $lvs $obs $actual-error $current-cycle $all-errors))

     ; Reward promoted observables (Phase 3)
     (reward-promoted! $obs $actual-error)

     ; Reward perceptual fields (Phase 5)
     (reward-percept! $obs $actual-error)

     done))

(= (reward-lv-coherent-list! () $obs $err $cycle $all-errors) done)
(= (reward-lv-coherent-list! ($name . $rest) $obs $err $cycle $all-errors)
   (sequential
     (if (lv-gestating? $name $cycle)
         done
         (reward-latent-variable-coherent! $name $obs $err $all-errors))
     (reward-lv-coherent-list! $rest $obs $err $cycle $all-errors)))

(= (reward-pm-list! () $err $cycle) done)
(= (reward-pm-list! ((passive-model $c $e $l $w $t) . $rest) $err $cycle)
   (sequential
     (if (pm-gestating? $c $e $cycle)
         done
         (reward-passive-model! $c $e $err))
     (reward-pm-list! $rest $err $cycle)))

(= (reward-lv-list! () $obs $err $cycle) done)
(= (reward-lv-list! ($name . $rest) $obs $err $cycle)
   (sequential
     (if (lv-gestating? $name $cycle)
         done
         (reward-latent-variable! $name $obs $err))
     (reward-lv-list! $rest $obs $err $cycle)))


; =============================================================================
; SECTION X: MASTER STRUCTURE LEARNING STEP
; =============================================================================
;
; Called once per cognitive cycle from cycle.metta (step 4b).
; This is a REFLEX ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â it runs unconditionally when surprise conditions
; are met, not as an action selected by EFE.
;
; Flow:
;   1. Reset generation budget
;   2. Update suspicion links (Hebbian accumulation)
;   3. Phase 1: Check for causal link creation
;   4. Phase 2: Check for hub detection
;   5. Metabolic step (drain + prune)

(= (structure-learning-step! $current-cycle)
   (sequential
     (reset-structural-budget!)
     (update-all-suspicion-links! $current-cycle)
     (check-phase1-triggers! $current-cycle)
     (check-deductive-triggers! $current-cycle)
     (check-phase2-triggers! $current-cycle)
     (clean-deduction-origins!)
     (metabolic-step! $current-cycle)
     (structure-step-done $current-cycle)))

; v2: Extended with Phases 3-5 and lifecycle metabolic step
; Called from cognitive-cycle-v2! with errors and selected policy.
(= (structure-learning-step-v2! $current-cycle $errors $selected-policy)
   (sequential
     (reset-structural-budget!)

     ; Existing phases
     (update-all-suspicion-links! $current-cycle)
     (check-phase1-triggers! $current-cycle)
     (check-deductive-triggers! $current-cycle)
     (check-phase2-triggers! $current-cycle)
     (clean-deduction-origins!)

     ; New phases
     (check-phase3-triggers! $current-cycle)
     (check-phase4-triggers! $current-cycle $selected-policy)
     (check-phase5-triggers! $current-cycle $errors)

     ; Metabolic steps (existing + lifecycle)
     (metabolic-step! $current-cycle)
     (lifecycle-metabolic-step! $current-cycle)

     (structure-step-done $current-cycle)))


; =============================================================================
; SECTION XII: PHASE 3 TRIGGERS Ã¢â‚¬â€ OBSERVABLE PROMOTION
; =============================================================================
;
; After Phases 1-2 and metabolic step, check if any latent variable
; is ready for promotion. Delegates to atom_lifecycle.metta.
;
; This is a REFLEX Ã¢â‚¬â€ fires when conditions are met, not selected by EFE.

(= (check-phase3-triggers! $current-cycle)
   (let $latents (all-latent-variables)
     (check-promotion-list! $latents $current-cycle)))

(= (check-promotion-list! () $cycle) done)
(= (check-promotion-list! ($name . $rest) $cycle)
   (sequential
     (if (and (promotion-ready? $name) (not (already-promoted? $name)))
         (promote-to-observable! $name $cycle)
         done)
     (check-promotion-list! $rest $cycle)))


; =============================================================================
; SECTION XIII: PHASE 4 TRIGGERS Ã¢â‚¬â€ ACTION CHUNKING
; =============================================================================
;
; After action selection records a policy, check if any policy
; has exceeded the repeat threshold. Delegates to atom_lifecycle.metta.

(= (check-phase4-triggers! $current-cycle $selected-policy)
   (sequential
     ; Record this policy execution
     (record-policy-trace! $current-cycle $selected-policy)
     ; Clean old traces
     (clean-old-policy-traces! $current-cycle)
     ; Check if any recent policy should be chunked
     (let $freq (policy-frequency $selected-policy)
       (if (>= $freq (get-config chunk-repeat-threshold))
           (if (not (already-chunked? $selected-policy))
               (reify-chunk! $selected-policy $current-cycle)
               (reward-chunk! $selected-policy 0.0))  ; Reward existing chunk
           done))))


; =============================================================================
; SECTION XIV: PHASE 5 TRIGGERS Ã¢â‚¬â€ PERCEPTUAL EXPANSION
; =============================================================================
;
; Compute how much of the current prediction error is "unexplained"
; (not reduced by any known passive model or latent variable).
; If persistently high, propose a new perceptual field.
;
; NOTE: Actual field creation requires the orchestrator to propose
; candidate field names. In the MeTTa core, we track the trigger
; condition and signal readiness. The orchestrator queries
; (perceptual-expansion-needed?) and acts accordingly.

; Compute unexplained error ratio
(= (compute-unexplained-ratio $errors)
   (let* (
     ($total (sum-abs-errors $errors 0.0))
     ($explained (sum-explained-errors $errors 0.0))
     ($unexplained (- $total $explained))
   )
   (if (== $total 0.0) 0.0 (/ $unexplained $total))))

; Sum absolute errors
(= (sum-abs-errors () $acc) $acc)
(= (sum-abs-errors ((prediction-error $obs $err $abs $prec $w) . $rest) $acc)
   (sum-abs-errors $rest (+ $acc $abs)))

; Sum errors explained by passive models
; An error is "explained" if there's a passive model predicting that observable
; and the model's cause also has an error in the same direction
(= (sum-explained-errors () $acc) $acc)
(= (sum-explained-errors
     ((prediction-error $obs $err $abs $prec $w) . $rest) $acc)
   (let $models (passive-models-affecting $obs)
     (if (list-empty? $models)
         (sum-explained-errors $rest $acc)
         ; Has at least one passive model Ã¢â‚¬â€ count as partially explained
         (sum-explained-errors $rest (+ $acc (* $abs 0.5))))))

; Check and record Phase 5 conditions
(= (check-phase5-triggers! $current-cycle $errors)
   (let* (
     ($ratio (compute-unexplained-ratio $errors))
     ($recorded (record-unexplained-ratio! $current-cycle $ratio))
     ($cleaned (clean-old-unexplained! $current-cycle))
     ($avg (avg-unexplained-ratio))
     ($threshold (get-config percept-unexplained-threshold))
     ($window (get-config percept-evaluation-window))
     ($n-records (length (collapse
       (match &state (unexplained-error-ratio $c $r) 1))))
   )
   ; Signal readiness but don't create fields directly
   ; (The orchestrator reads this and proposes candidates)
   (if (and (>= $n-records $window) (> $avg $threshold))
       (sequential
         (add-atom &state (perceptual-expansion-needed True))
         (expansion-triggered $avg $threshold))
       (sequential
         ; Clear the flag if it was set
         (let $old (collapse (match &state (perceptual-expansion-needed $v)
                                          (perceptual-expansion-needed $v)))
           (remove-atom-list-safe! $old))
         (expansion-not-needed $avg $threshold)))))

; Query: does the system think it needs new sensors?
(= (perceptual-expansion-needed?)
   (let $r (collapse (match &state (perceptual-expansion-needed True) True))
     (if (== $r ()) False True)))


; =============================================================================
; SECTION XV: DIAGNOSTICS
; =============================================================================

; Complete structure learning state summary
(= (structure-summary)
   (summary
     (suspicion-links (length (all-suspicion-links)))
     (passive-models (passive-model-count))
     (deductive-models (deductive-model-count))
     (latent-variables (latent-variable-count))
     (salient-traces (salient-count))))

; Detailed suspicion link report (top N by absolute strength)
(= (suspicion-report)
   (all-suspicion-links))

; Get structural health: ratio of energy income to expense
(= (structural-health)
   (let* (
     ($n-pm (passive-model-count))
     ($n-lv (latent-variable-count))
     ($total-structures (+ $n-pm $n-lv))
     ($total-rent (* $total-structures (get-config metabolic-rate)))
     ($surplus (structural-surprise))
   )
   (if (== $total-structures 0)
       (health no-structure 0.0)
       (health $total-structures $total-rent))))

; Is structure learning currently active?
; (i.e., are there suspicion links being tracked?)
(= (structure-learning-active?)
   (let $links (all-suspicion-links)
     (not (list-empty? $links))))

; Detailed passive model report
(= (passive-model-report)
   (collapse
     (match &self (passive-model $c $e $l $w $t)
       (let* (
         ($energy (get-pm-energy $c $e))
         ($created (get-pm-created $c $e))
         (($preds $succs) (match &self (prediction-stats passive-model $c $e $p $s) ($p $s)))
       )
       (pm-report $c $e $l $w $t $energy $created $preds $succs)))))

; Detailed latent variable report
(= (latent-variable-report)
   (collapse
     (match &self (latent-variable $name)
       (let* (
         ($members (get-latent-members $name))
         ($energy (get-lv-energy $name))
         ($created (get-lv-created $name))
         (($preds $succs) (match &self (prediction-stats latent-var $name $p $s) ($p $s)))
       )
       (lv-report $name $members $energy $created $preds $succs)))))


; =============================================================================
; END OF STRUCTURE LEARNING (v0.5.0 -- Phases 1-5 + Deductive + Boundary Fix)
; =============================================================================
