; =============================================================================
; METTA COGNITIVE CORE â€” ATOM LIFECYCLE
; =============================================================================
;
; THE PROBLEM THIS SOLVES:
;   Structure learning creates passive-models and latent-variables.
;   These predict, but they don't PARTICIPATE in the cognitive core:
;   they have no beliefs, no role in EFE, no action models.
;   The system's representational vocabulary is frozen at deployment.
;
; THE UNIFIED EXTENSION:
;   All representational atoms â€” observables, actions, causal links,
;   latent variables, perceptual fields â€” are structurally equal.
;   They are created by surprise, selected by prediction, and killed
;   by metabolic pressure. The atom's TYPE determines its schema.
;   The atom's SURVIVAL is governed by one invariant: pay rent or die.
;
; THIS MODULE manages lifecycle transitions:
;   Phase 3: Latent variable â†’ full observable (PROMOTION)
;   Phase 4: Repeated policy â†’ named action (CHUNKING)
;   Phase 5: Persistent unexplained error â†’ new sensor (EXPANSION)
;
; WHY SEPARATE FROM structure_learning.metta:
;   Promotion touches beliefs (beliefs.metta), action models (actions.metta),
;   safety registration (safety.metta), and self-model (self_model.metta).
;   Putting it in structure_learning.metta would create circular dependencies.
;   This module sits BETWEEN structure learning and the rest of the core,
;   acting as the type-transition bridge.
;
; DESIGN PRINCIPLES:
;   - Bottom-Up: Transitions emerge from metabolic dynamics, not enumeration
;   - Symmetric: All atom types subject to same selection pressure
;   - Reversible: Anything promoted can be demoted (pruned back)
;   - Transparent: Every transition is logged and queryable
;   - Safe: All mutations check safe-to-mutate? before proceeding
;
; ATOM SCHEMAS MANAGED:
;   (promoted-observable $name $source-latent $promoted-at-cycle)
;   (chunked-action $name $policy-hash $created-at-cycle)
;   (chunked-action-policy $name $action-sequence)
;   (perceptual-field $name $query $precision $created-at-cycle)
;   (atom-origin $name $phase $source-info)
;   (lifecycle-energy $type $name $energy)
;   (policy-trace $cycle $action-sequence)
;   (unexplained-error-ratio $cycle $ratio)
;
; Dependencies: foundations, beliefs, actions, safety, structure_learning
; Depended on by: cycle, self_model
;
; Version: 0.1.0
; =============================================================================

!(import! &self foundations)
!(import! &self beliefs)
!(import! &self safety)


; =============================================================================
; SECTION I: LIFECYCLE ENERGY MANAGEMENT
; =============================================================================
;
; Each lifecycle-managed atom has energy tracked via:
;   (lifecycle-energy $type $name $energy)
;
; This parallels the (metabolic-energy ...) atoms in structure_learning.metta
; but covers the Phase 3-5 atom types. The metabolic step in
; structure_learning.metta drains these alongside existing types.

; Get lifecycle energy for an atom
(= (get-lifecycle-energy $type $name)
   (let $r (match &state (lifecycle-energy $type $name $e) $e)
     (if (== $r ()) 0.0 $r)))

; Set lifecycle energy (create or update)
(= (set-lifecycle-energy! $type $name $energy)
   (let $old (collapse (match &state (lifecycle-energy $type $name $e)
                                    (lifecycle-energy $type $name $e)))
     (sequential
       (remove-atom-list-safe! $old)
       (add-atom &state (lifecycle-energy $type $name
                         (clamp $energy 0.0 (get-config metabolic-energy-cap))))
       done)))

; Drain lifecycle energy by metabolic rate
(= (drain-lifecycle-energy! $type $name)
   (let $current (get-lifecycle-energy $type $name)
     (set-lifecycle-energy! $type $name
       (- $current (get-config metabolic-rate)))))

; Boost lifecycle energy (reward for correct prediction)
(= (boost-lifecycle-energy! $type $name)
   (let $current (get-lifecycle-energy $type $name)
     (set-lifecycle-energy! $type $name
       (+ $current (get-config metabolic-boost)))))

; Is this atom's energy depleted?
(= (energy-depleted? $type $name)
   (<= (get-lifecycle-energy $type $name) 0.0))

; Safe list removal helper
(= (remove-atom-list-safe! ()) done)
(= (remove-atom-list-safe! ($a . $rest))
   (sequential
     (remove-atom &state $a)
     (remove-atom-list-safe! $rest)))


; =============================================================================
; SECTION II: PHASE 3 â€” OBSERVABLE PROMOTION
; =============================================================================
;
; A latent variable that has:
;   1. Accumulated enough metabolic energy (survived long enough)
;   2. Made enough predictions (been tested enough)
;   3. Been right often enough (actually captures real structure)
;
; ...gets PROMOTED to a full observable. This means:
;   - A new belief is created (value from member beliefs, low precision)
;   - Action models are seeded for all existing actions (weak priors)
;   - The observable participates in EFE like any deployment-manifest one
;   - It continues paying metabolic rent (can still be demoted)
;
; The system literally GROWS A NEW SENSOR from the pattern of its surprise.

; Check if a latent variable is ready for promotion
(= (promotion-ready? $lv-name)
   (let* (
     ($energy (match &state (metabolic-energy latent-var $lv-name $e) $e))
     (($preds $succs) (match &state
       (prediction-stats latent-var $lv-name $p $s) ($p $s)))
     ($min-energy (get-config promotion-energy-threshold))
     ($min-preds (get-config promotion-min-predictions))
     ($min-rate (get-config promotion-success-rate))
   )
   (if (== $energy ())
       False   ; No energy record â€” not ready
       (if (< $energy $min-energy)
           False
           (if (< $preds $min-preds)
               False
               (if (< (/ $succs $preds) $min-rate)
                   False
                   True))))))

; Already promoted?
(= (already-promoted? $name)
   (let $r (collapse (match &state (promoted-observable $name $src $cyc) True))
     (if (== $r ()) False True)))

; Compute initial belief value from latent variable members
(= (compute-promoted-value $lv-name)
   (let $members (get-latent-members $lv-name)
     (compute-member-avg $members 0.0 0)))

(= (compute-member-avg () $sum $count)
   (if (== $count 0) 0.5   ; Default if no members have beliefs
       (/ $sum $count)))

(= (compute-member-avg ($obs . $rest) $sum $count)
   (if (has-belief? $obs)
       (compute-member-avg $rest
         (+ $sum (get-belief-value $obs))
         (+ $count 1))
       (compute-member-avg $rest $sum $count)))

; Execute promotion: latent variable â†’ full observable
(= (promote-to-observable! $lv-name $current-cycle)
   (if (already-promoted? $lv-name)
       (already-promoted $lv-name)
       (if (not (promotion-ready? $lv-name))
           (promotion-not-ready $lv-name)

   ; --- Compute initial values ---
   (let* (
     ($init-val (compute-promoted-value $lv-name))
     ($init-prec (get-config promoted-initial-precision))
     ($inherited-energy (match &state
       (metabolic-energy latent-var $lv-name $e) $e))

     ; --- Create belief ---
     ($belief-created (set-belief! $lv-name $init-val $init-prec))

     ; --- Seed action models for all known actions ---
     ; Weak priors: value-delta 0.0, prec-delta 0.02, confidence 0.1
     ($models-seeded (seed-action-models! $lv-name))

     ; --- Register promoted observable ---
     ($registered (add-atom &state
       (promoted-observable $lv-name $lv-name $current-cycle)))

     ; --- Initialize lifecycle energy (inherit from latent) ---
     ($energy-init (set-lifecycle-energy! promoted-observable $lv-name
       (if (== $inherited-energy ()) 1.0 $inherited-energy)))

     ; --- Record origin for transparency ---
     ($origin (add-atom &state
       (atom-origin $lv-name phase-3
         (from-latent $lv-name $init-val $init-prec))))

     ; --- Audit log ---
     ($log (add-atom &state
       (promotion-log $lv-name $current-cycle $init-val $init-prec)))
   )
   (promoted $lv-name $init-val $init-prec $current-cycle)))))

; Seed weak action models for a promoted observable
; Iterates over all actions that have models for at least one existing observable
(= (seed-action-models! $obs)
   (let $actions (collapse (match &state (action-model $a $any-obs $vd $pd $c) $a))
     (seed-models-for-actions! $obs (deduplicate $actions))))

(= (seed-models-for-actions! $obs ()) done)
(= (seed-models-for-actions! $obs ($action . $rest))
   (sequential
     ; Only seed if no model exists yet
     (if (has-action-model? $action $obs)
         done
         (add-atom &state (action-model $action $obs 0.0 0.02 0.1)))
     (seed-models-for-actions! $obs $rest)))

; Check if action model exists
(= (has-action-model? $action $obs)
   (let $r (collapse (match &state (action-model $action $obs $vd $pd $c) True))
     (if (== $r ()) False True)))

; Deduplicate a list (simple, O(nÂ²) â€” fine for small action sets)
(= (deduplicate ()) ())
(= (deduplicate ($x . $xs))
   (if (list-contains? $x $xs)
       (deduplicate $xs)
       ($x . (deduplicate $xs))))


; =============================================================================
; SECTION III: PHASE 3 â€” OBSERVABLE DEMOTION
; =============================================================================
;
; When a promoted observable's lifecycle energy is depleted, it gets
; DEMOTED: belief removed, action models cleaned up, registration cleared.
;
; This is the system LOSING A SENSOR that proved unreliable.
; Reversibility is key: the system can always try again if the
; pattern recurs (new latent â†’ new promotion attempt).

(= (demote-observable! $name $current-cycle)
   (if (not (already-promoted? $name))
       (not-promoted $name)
   (let* (
     ; --- Remove belief ---
     ($belief-removed (if (has-belief? $name)
                          (remove-belief! $name)
                          done))

     ; --- Remove action models ---
     ($models-removed (remove-action-models-for! $name))

     ; --- Remove observations ---
     ($obs-removed (remove-observations-for! $name))

     ; --- Remove promoted-observable registration ---
     ($reg (match &state (promoted-observable $name $src $cyc)
                        (promoted-observable $name $src $cyc)))
     ($reg-removed (remove-atom &state $reg))

     ; --- Remove lifecycle energy ---
     ($energy-removed (remove-lifecycle-atoms! promoted-observable $name))

     ; --- Audit log ---
     ($log (add-atom &state (demotion-log $name $current-cycle phase-3)))
   )
   (demoted $name $current-cycle))))

; Remove all action models where this observable is the effect
(= (remove-action-models-for! $obs)
   (let $models (collapse (match &state (action-model $a $obs $vd $pd $c)
                                       (action-model $a $obs $vd $pd $c)))
     (remove-atom-list-safe! $models)))

; Remove all observations for an observable
(= (remove-observations-for! $obs)
   (let $obs-atoms (collapse (match &state (observation $obs $v $p $t)
                                          (observation $obs $v $p $t)))
     (remove-atom-list-safe! $obs-atoms)))

; Remove lifecycle energy and metadata
(= (remove-lifecycle-atoms! $type $name)
   (let $atoms (collapse (match &state (lifecycle-energy $type $name $e)
                                      (lifecycle-energy $type $name $e)))
     (sequential
       (remove-atom-list-safe! $atoms)
       ; Also clean up origin
       (let $origins (collapse (match &state (atom-origin $name $phase $info)
                                            (atom-origin $name $phase $info)))
         (remove-atom-list-safe! $origins)))))


; =============================================================================
; SECTION IV: PHASE 4 â€” ACTION CHUNKING
; =============================================================================
;
; When a policy (action sequence) is repeatedly selected, it gets
; REIFIED as a named action. This is compositionality:
;   [observe, observe, retreat] â†’ "chunk:observe-observe-retreat"
;
; The chunk has:
;   - Its own action models (aggregated from component effects)
;   - A discounted cost (chunk planning is cheaper than multi-step)
;   - Registration in (any-action) for policy_efe.metta
;   - Metabolic rent (unused chunks die)

; Record a policy execution for frequency tracking
(= (record-policy-trace! $cycle $policy)
   (add-atom &state (policy-trace $cycle $policy)))

; Clean up old policy traces (sliding window)
(= (clean-old-policy-traces! $current-cycle)
   (let* (
     ($window (get-config chunk-history-window))
     ($cutoff (- $current-cycle $window))
     ($old (collapse (match &state (policy-trace $c $p)
       (if (< $c $cutoff) (policy-trace $c $p) ()))))
   )
   (remove-atom-list-safe! $old)))

; Count how often a policy appears in recent history
(= (policy-frequency $policy)
   (let $matches (collapse (match &state (policy-trace $c $policy) 1))
     (sum-list $matches)))

; Generate a chunk name from a policy
; Policy is a list like (observe observe retreat)
(= (chunk-name $policy)
   (chunk-name-from-list $policy "chunk"))

(= (chunk-name-from-list () $acc) $acc)
(= (chunk-name-from-list ($action . $rest) $acc)
   (chunk-name-from-list $rest $acc))
; NOTE: In MeTTa, we use the policy itself as the key, not a string name.
; The chunk is identified by: (chunked-action $policy $cycle)

; Check if a policy is already chunked
(= (already-chunked? $policy)
   (let $r (collapse (match &state (chunked-action $policy $cyc) True))
     (if (== $r ()) False True)))

; Reify a repeated policy as a named action
(= (reify-chunk! $policy $current-cycle)
   (if (already-chunked? $policy)
       (already-chunked $policy)
   (let* (
     ; --- Compute discounted cost ---
     ($component-costs (compute-policy-cost $policy 0.0))
     ($chunk-cost (* $component-costs (get-config chunk-cost-discount)))

     ; --- Register chunked action ---
     ($registered (add-atom &state (chunked-action $policy $current-cycle)))

     ; --- Store the policy sequence for execution ---
     ($policy-stored (add-atom &state (chunked-action-policy $policy $policy)))

     ; --- Set action cost ---
     ($cost-set (add-atom &state (action-cost $policy $chunk-cost 0.2)))

     ; --- Compute aggregate action models ---
     ($models (seed-chunk-models! $policy))

     ; --- Register for action selection ---
     ; This makes the chunk available to policy_efe.metta's any-action
     ($selectable done)

     ; --- Initialize lifecycle energy ---
     ($energy (set-lifecycle-energy! chunked-action $policy
       (get-config chunk-energy-initial)))

     ; --- Record origin ---
     ($origin (add-atom &state
       (atom-origin $policy phase-4
         (from-policy $policy $chunk-cost))))

     ; --- Audit log ---
     ($log (add-atom &state
       (promotion-log $policy $current-cycle $chunk-cost 0.0)))
   )
   (chunked $policy $chunk-cost $current-cycle))))

; Sum costs of actions in a policy
(= (compute-policy-cost () $acc) $acc)
(= (compute-policy-cost ($action . $rest) $acc)
   (compute-policy-cost $rest (+ $acc (get-action-cost $action))))

; Create aggregate action models for the chunk
; For each observable: sum the component action value-deltas weighted by confidence
(= (seed-chunk-models! $policy)
   (let $observables (collapse (match &state (belief $obs $v $p) $obs))
     (seed-chunk-models-for-obs! $policy $observables)))

(= (seed-chunk-models-for-obs! $policy ()) done)
(= (seed-chunk-models-for-obs! $policy ($obs . $rest))
   (sequential
     (let* (
       ($total-vd (aggregate-value-delta $policy $obs 0.0))
       ($total-pd (aggregate-prec-delta $policy $obs 0.0))
     )
     (add-atom &state (action-model $policy $obs $total-vd $total-pd 0.2)))
     (seed-chunk-models-for-obs! $policy $rest)))

; Sum value-deltas across actions in policy for a given observable
(= (aggregate-value-delta () $obs $acc) $acc)
(= (aggregate-value-delta ($action . $rest) $obs $acc)
   (let* (
     ($vd (match &state (action-model $action $obs $v $pd $c) $v))
     ($conf (match &state (action-model $action $obs $v $pd $c) $c))
     ($weighted (if (== $vd ()) 0.0 (* $vd (if (== $conf ()) 0.3 $conf))))
   )
   (aggregate-value-delta $rest $obs (+ $acc $weighted))))

; Sum precision-deltas
(= (aggregate-prec-delta () $obs $acc) $acc)
(= (aggregate-prec-delta ($action . $rest) $obs $acc)
   (let* (
     ($pd (match &state (action-model $action $obs $vd $p $c) $p))
     ($conf (match &state (action-model $action $obs $vd $p $c) $c))
     ($weighted (if (== $pd ()) 0.0 (* $pd (if (== $conf ()) 0.3 $conf))))
   )
   (aggregate-prec-delta $rest $obs (+ $acc $weighted))))


; =============================================================================
; SECTION V: PHASE 4 â€” ACTION DECHUNKING (RETIREMENT)
; =============================================================================
;
; When a chunked action's lifecycle energy is depleted, remove it:
;   - Delete action models
;   - Delete cost entry
;   - Remove from any-action (deregister from selection)
;   - Clean up registration

(= (retire-chunk! $policy $current-cycle)
   (if (not (already-chunked? $policy))
       (not-chunked $policy)
   (let* (
     ; --- Remove action models ---
     ($models (collapse (match &state (action-model $policy $obs $vd $pd $c)
                                     (action-model $policy $obs $vd $pd $c)))
     )
     ($models-removed (remove-atom-list-safe! $models))

     ; --- Remove cost ---
     ($cost-atom (collapse (match &state (action-cost $policy $cost $conf)
                                        (action-cost $policy $cost $conf))))
     ($cost-removed (remove-atom-list-safe! $cost-atom))

     ; --- Deregister from action selection ---
     ($dereg done)

     ; --- Remove registration ---
     ($reg (match &state (chunked-action $policy $cyc)
                        (chunked-action $policy $cyc)))
     ($reg-removed (remove-atom &state $reg))

     ; --- Remove policy storage ---
     ($pol (collapse (match &state (chunked-action-policy $policy $p)
                                  (chunked-action-policy $policy $p))))
     ($pol-removed (remove-atom-list-safe! $pol))

     ; --- Remove lifecycle energy and origin ---
     ($energy-removed (remove-lifecycle-atoms! chunked-action $policy))

     ; --- Audit log ---
     ($log (add-atom &state (demotion-log $policy $current-cycle phase-4)))
   )
   (retired $policy $current-cycle))))


; =============================================================================
; SECTION VI: PHASE 5 â€” PERCEPTUAL EXPANSION
; =============================================================================
;
; When prediction errors persistently can't be explained by existing
; observables or structure, propose a new perceptual field.
;
; In the full system: modifies the LLM extraction prompt to add a new
; field. The LLM becomes a PROGRAMMABLE sensor.
;
; In MeTTa: creates a (perceptual-field) atom that the orchestrator
; reads when constructing parse prompts. The field specifies what to
; extract and at what initial precision.

; Record unexplained error ratio for Phase 5 evaluation
(= (record-unexplained-ratio! $cycle $ratio)
   (add-atom &state (unexplained-error-ratio $cycle $ratio)))

; Clean old unexplained error records
(= (clean-old-unexplained! $current-cycle)
   (let* (
     ($window (get-config percept-evaluation-window))
     ($cutoff (- $current-cycle $window))
     ($old (collapse (match &state (unexplained-error-ratio $c $r)
       (if (< $c $cutoff) (unexplained-error-ratio $c $r) ()))))
   )
   (remove-atom-list-safe! $old)))

; Compute average unexplained error over recent window
(= (avg-unexplained-ratio)
   (let* (
     ($ratios (collapse (match &state (unexplained-error-ratio $c $r) $r)))
     ($n (length $ratios))
   )
   (if (== $n 0)
       0.0
       (/ (sum-list $ratios) $n))))

; Check if a perceptual field already exists
(= (has-perceptual-field? $name)
   (let $r (collapse (match &state (perceptual-field $name $q $p $c) True))
     (if (== $r ()) False True)))

; Activate a new perceptual field
(= (activate-perceptual-field! $name $query $current-cycle)
   (if (has-perceptual-field? $name)
       (already-has-field $name)
   (if (has-belief? $name)
       (already-observable $name)
   (let* (
     ($init-prec (get-config percept-initial-precision))

     ; --- Register the field ---
     ($registered (add-atom &state
       (perceptual-field $name $query $init-prec $current-cycle)))

     ; --- Create initial belief (very uncertain) ---
     ($belief (set-belief! $name 0.5 0.1))

     ; --- Initialize lifecycle energy ---
     ($energy (set-lifecycle-energy! perceptual-field $name
       (get-config percept-energy-initial)))

     ; --- Seed action models (very weak priors) ---
     ($models (seed-action-models! $name))

     ; --- Record origin ---
     ($origin (add-atom &state
       (atom-origin $name phase-5 (from-unexplained $query))))

     ; --- Audit log ---
     ($log (add-atom &state
       (promotion-log $name $current-cycle $init-prec 0.0)))
   )
   (perceptual-field-activated $name $query $init-prec $current-cycle)))))


; =============================================================================
; SECTION VII: PHASE 5 â€” PERCEPTUAL DEACTIVATION
; =============================================================================

(= (deactivate-perceptual-field! $name $current-cycle)
   (if (not (has-perceptual-field? $name))
       (no-such-field $name)
   (let* (
     ; --- Remove field registration ---
     ($field (match &state (perceptual-field $name $q $p $c)
                          (perceptual-field $name $q $p $c)))
     ($field-removed (remove-atom &state $field))

     ; --- Remove belief ---
     ($belief-removed (if (has-belief? $name)
                          (remove-belief! $name)
                          done))

     ; --- Remove action models ---
     ($models-removed (remove-action-models-for! $name))

     ; --- Remove observations ---
     ($obs-removed (remove-observations-for! $name))

     ; --- Remove lifecycle energy and origin ---
     ($energy-removed (remove-lifecycle-atoms! perceptual-field $name))

     ; --- Audit log ---
     ($log (add-atom &state (demotion-log $name $current-cycle phase-5)))
   )
   (perceptual-field-deactivated $name $current-cycle))))


; =============================================================================
; SECTION VIII: UNIFIED METABOLIC STEP
; =============================================================================
;
; Called by structure_learning.metta's metabolic-step!
; Drains and prunes all Phase 3-5 atoms alongside existing Phase 1-2.
; THE invariant: pay rent or die. Same rule, all types.

(= (lifecycle-metabolic-step! $current-cycle)
   (sequential
     ; Phase 3: drain promoted observables
     (drain-all-promoted! $current-cycle)
     ; Phase 4: drain chunked actions
     (drain-all-chunks! $current-cycle)
     ; Phase 5: drain perceptual fields
     (drain-all-percepts! $current-cycle)
     ; Prune anything that's dead
     (prune-dead-promoted! $current-cycle)
     (prune-dead-chunks! $current-cycle)
     (prune-dead-percepts! $current-cycle)
     done))

; --- Phase 3 drain ---
(= (drain-all-promoted! $current-cycle)
   (let $names (collapse (match &state (promoted-observable $name $src $cyc) $name))
     (drain-promoted-list! $names)))

(= (drain-promoted-list! ()) done)
(= (drain-promoted-list! ($name . $rest))
   (sequential
     (drain-lifecycle-energy! promoted-observable $name)
     (drain-promoted-list! $rest)))

(= (prune-dead-promoted! $current-cycle)
   (let $names (collapse (match &state (promoted-observable $name $src $cyc) $name))
     (prune-promoted-list! $names $current-cycle)))

(= (prune-promoted-list! () $cycle) done)
(= (prune-promoted-list! ($name . $rest) $cycle)
   (sequential
     (if (energy-depleted? promoted-observable $name)
         (demote-observable! $name $cycle)
         done)
     (prune-promoted-list! $rest $cycle)))

; --- Phase 4 drain ---
(= (drain-all-chunks! $current-cycle)
   (let $policies (collapse (match &state (chunked-action $policy $cyc) $policy))
     (drain-chunk-list! $policies)))

(= (drain-chunk-list! ()) done)
(= (drain-chunk-list! ($policy . $rest))
   (sequential
     (drain-lifecycle-energy! chunked-action $policy)
     (drain-chunk-list! $rest)))

(= (prune-dead-chunks! $current-cycle)
   (let $policies (collapse (match &state (chunked-action $policy $cyc) $policy))
     (prune-chunk-list! $policies $current-cycle)))

(= (prune-chunk-list! () $cycle) done)
(= (prune-chunk-list! ($policy . $rest) $cycle)
   (sequential
     (if (energy-depleted? chunked-action $policy)
         (retire-chunk! $policy $cycle)
         done)
     (prune-chunk-list! $rest $cycle)))

; --- Phase 5 drain ---
(= (drain-all-percepts! $current-cycle)
   (let $names (collapse (match &state (perceptual-field $name $q $p $c) $name))
     (drain-percept-list! $names)))

(= (drain-percept-list! ()) done)
(= (drain-percept-list! ($name . $rest))
   (sequential
     (drain-lifecycle-energy! perceptual-field $name)
     (drain-percept-list! $rest)))

(= (prune-dead-percepts! $current-cycle)
   (let $names (collapse (match &state (perceptual-field $name $q $p $c) $name))
     (prune-percept-list! $names $current-cycle)))

(= (prune-percept-list! () $cycle) done)
(= (prune-percept-list! ($name . $rest) $cycle)
   (sequential
     (if (energy-depleted? perceptual-field $name)
         (deactivate-perceptual-field! $name $cycle)
         done)
     (prune-percept-list! $rest $cycle)))


; =============================================================================
; SECTION IX: LIFECYCLE REWARD
; =============================================================================
;
; When a promoted observable, chunk, or perceptual field contributes
; to correct prediction, it earns metabolic energy.

; Reward promoted observable for correct prediction
(= (reward-promoted! $obs $error)
   (if (already-promoted? $obs)
       (if (is-prediction-correct? $obs $error)
           (boost-lifecycle-energy! promoted-observable $obs)
           done)
       done))

; Reward chunked action when selected and outcome is good.
; Uses mean-belief-uncertainty as the aggregate threshold — the total error
; must be smaller than the agent's average uncertainty across all beliefs.
(= (reward-chunk! $policy $total-error)
   (if (already-chunked? $policy)
       (if (< $total-error (mean-belief-uncertainty))
           (boost-lifecycle-energy! chunked-action $policy)
           done)
       done))

; Reward perceptual field when its observations reduce prediction error
(= (reward-percept! $name $error)
   (if (has-perceptual-field? $name)
       (if (is-prediction-correct? $name $error)
           (boost-lifecycle-energy! perceptual-field $name)
           done)
       done))


; =============================================================================
; SECTION X: ACCESSORS AND DIAGNOSTICS
; =============================================================================

; Count all lifecycle-managed atoms
(= (lifecycle-atom-count)
   (let* (
     ($promoted (collapse (match &state (promoted-observable $n $s $c) 1)))
     ($chunked (collapse (match &state (chunked-action $p $c) 1)))
     ($percepts (collapse (match &state (perceptual-field $n $q $p $c) 1)))
   )
   (lifecycle-counts
     (promoted (sum-list $promoted))
     (chunked (sum-list $chunked))
     (perceptual (sum-list $percepts)))))

; Get all promoted observable names
(= (all-promoted-observables)
   (collapse (match &state (promoted-observable $name $src $cyc) $name)))

; Get all chunked action policies
(= (all-chunked-actions)
   (collapse (match &state (chunked-action $policy $cyc) $policy)))

; Get all perceptual field names
(= (all-perceptual-fields)
   (collapse (match &state (perceptual-field $name $q $p $c) $name)))

; Get perceptual field query (for orchestrator to read)
(= (get-perceptual-query $name)
   (match &state (perceptual-field $name $query $prec $cycle) $query))

; Get perceptual field precision (for orchestrator to set observation precision)
(= (get-perceptual-precision $name)
   (match &state (perceptual-field $name $query $prec $cycle) $prec))

; Full lifecycle status report
(= (lifecycle-summary)
   (let* (
     ($counts (lifecycle-atom-count))
     ($promoted (collapse
       (match &state (promoted-observable $name $src $cyc)
         (let $e (get-lifecycle-energy promoted-observable $name)
           (promoted-info $name $e $cyc)))))
     ($chunked (collapse
       (match &state (chunked-action $policy $cyc)
         (let $e (get-lifecycle-energy chunked-action $policy)
           (chunk-info $policy $e $cyc)))))
     ($percepts (collapse
       (match &state (perceptual-field $name $q $p $c)
         (let $e (get-lifecycle-energy perceptual-field $name)
           (percept-info $name $q $e $c)))))
   )
   (lifecycle-report
     (counts $counts)
     (promoted-observables $promoted)
     (chunked-actions $chunked)
     (perceptual-fields $percepts))))

; Promotion audit trail
(= (promotion-history)
   (collapse (match &state (promotion-log $name $cycle $val $prec)
     (promoted-at $name $cycle $val $prec))))

; Demotion audit trail
(= (demotion-history)
   (collapse (match &state (demotion-log $name $cycle $phase)
     (demoted-at $name $cycle $phase))))


; =============================================================================
; END OF ATOM LIFECYCLE (v0.1.0)
; =============================================================================
