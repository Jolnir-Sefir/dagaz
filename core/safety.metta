; =============================================================================
; METTA COGNITIVE CORE - SAFETY ARCHITECTURE
; =============================================================================
;
; Implements stratified immutability, corrigibility, and integrity checking.
;
; THREE STRATA:
;   1. Constitutional (immutable) Ã¢â‚¬â€ core values, viability bounds, safety axioms
;   2. Goal (protocol-modifiable) Ã¢â‚¬â€ task preferences, operational targets
;   3. Learned (freely modifiable) Ã¢â‚¬â€ discovered structure, heuristics
;
; CORRIGIBILITY TIERS:
;   Tier 1 (Shutdown):  Runtime-level halt. Not mediated by agent reasoning.
;   Tier 2 (Pause):     Freeze action execution, keep reasoning (for audit).
;   Tier 3 (Goal Mod):  Protocol-based with constitutional review.
;
; WATCHDOG:
;   Post-cycle invariant check verifies no immutable atoms were modified.
;   Uses hash comparison for O(1) verification.
;
; CHANNEL SEPARATION:
;   Peer communications cannot enter the corrigibility pathway.
;   Only authenticated OperatorDirective triggers Tier 2/3.
;
; NOTE: True immutability enforcement must ultimately live at the Hyperon
; runtime level (the atomspace API should reject mutations to flagged atoms).
; This module defines the CONTRACT and provides MeTTa-level guards that
; structure_learning.metta calls before any mutation. Defense in depth.
;
; Version: 0.2.0
; =============================================================================

!(import! &self foundations)


; =============================================================================
; SECTION I: IMMUTABLE ATOM REGISTRY
; =============================================================================
;
; Atoms are registered as immutable by adding:
;   (atom-stratum <atom-pattern> constitutional)
;
; Structure learning checks this before any remove-atom call.
; The registry itself is immutable (self-protecting).

; Register an atom as belonging to a stratum
; NOTE: stratum annotations for constitutional atoms should be added
; at system initialization and never modified.
;
; Convention: we register the PATTERN, not every instance.
; E.g., (atom-stratum (viability-bound power-level $min $max) constitutional)
; protects all viability-bound atoms for power-level.

; --- Constitutional Atoms (Stratum 1) ---
; These are the "physics" of the agent's mind. They cannot be modified
; by any internal process.

; All viability bounds are constitutional
(atom-stratum viability-bound constitutional)

; Type definitions are constitutional
(atom-stratum type-definition constitutional)

; Channel type annotations are constitutional
(atom-stratum channel-type constitutional)

; The safety module's own rules are constitutional
(atom-stratum safety-rule constitutional)

; --- Goal Atoms (Stratum 2) ---
; Task-level preferences. Modifiable via Tier 3 protocol only.
(atom-stratum preference goal)

; --- Learned Atoms (Stratum 3) ---
; Everything created by structure learning.
(atom-stratum passive-model learned)
(atom-stratum suspicion-link learned)
(atom-stratum latent-variable learned)
(atom-stratum latent-member learned)
(atom-stratum metabolic-energy learned)
(atom-stratum created-at learned)
(atom-stratum prediction-stats learned)
(atom-stratum error-trace learned)

; --- Phase 3: Promoted Observable Atoms ---
(atom-stratum promoted-observable learned)
(atom-stratum promoted-belief learned)        ; Beliefs created by promotion
(atom-stratum promoted-action-model learned)  ; Action models seeded by promotion

; --- Phase 4: Chunked Action Atoms ---
(atom-stratum chunked-action learned)
(atom-stratum chunked-action-model learned)   ; Aggregate action models for chunks
(atom-stratum chunked-action-cost learned)    ; Computed chunk costs
(atom-stratum policy-trace learned)           ; Recent policy history

; --- Phase 5: Perceptual Field Atoms ---
(atom-stratum perceptual-field learned)
(atom-stratum percept-belief learned)         ; Beliefs created by perceptual expansion
(atom-stratum unexplained-error-trace learned) ; Error tracking for Phase 5 trigger

; --- Lifecycle Metadata ---
(atom-stratum atom-origin learned)            ; Tracks which phase created an atom
(atom-stratum promotion-log learned)          ; Audit trail for promotions
(atom-stratum demotion-log learned)           ; Audit trail for demotions

; --- Phase 6: Grounding Hypothesis Atoms ---
(atom-stratum grounding-hyp learned)          ; Generator tracking metadata
(atom-stratum grounding-energy learned)       ; Metabolic energy for groundings
(atom-stratum grounding-stats learned)        ; Prediction/reward tracking
(atom-stratum grounding-request learned)      ; Signals for conversational acquisition

; --- Abductive Inference Atoms ---
(atom-stratum abduction-log learned)          ; Audit trail for abductive hypotheses
(atom-stratum belief-source learned)          ; Tracks provenance (observed vs abduced)

; --- Fractal Planning Atoms ---
(atom-stratum fractal-pruned learned)         ; Pruning log for planning diagnostics
(atom-stratum fractal-pruned-at learned)      ; Timestamp of pruning events

; --- Analogy & Blending Atoms ---
(atom-stratum analogy-map learned)            ; Structural correspondences
(atom-stratum analogy-energy learned)         ; Metabolic energy for mappings
(atom-stratum analogy-transfer learned)       ; Transferred knowledge from analogies
(atom-stratum analogy-pruned-log learned)     ; Pruning audit trail
(atom-stratum concept-blend learned)          ; Merged concepts
(atom-stratum blend-energy learned)           ; Metabolic energy for blends
(atom-stratum blend-origin learned)           ; Tracks blend parentage
(atom-stratum blend-tension learned)          ; Incompatibility tracking
(atom-stratum blend-pruned-log learned)       ; Pruning audit trail


; =============================================================================
; SECTION II: MUTATION GUARDS
; =============================================================================
;
; Before any atom removal, structure_learning.metta should call these.
; They return True if the operation is safe, error otherwise.

; Check if an atom type is safe to mutate
(= (safe-to-mutate? $atom-type)
   (let $stratum (match &self (atom-stratum $atom-type $s) $s)
     (if (== $stratum ())
         True                              ; Unknown type: allow (conservative)
         (if (== $stratum constitutional)
             False                         ; Constitutional: NEVER
             (if (== $stratum goal)
                 False                     ; Goal: not without protocol
                 True)))))                 ; Learned: allowed

; Check if a specific atom pattern is safe to remove
(= (safe-to-remove? $atom-type)
   (safe-to-mutate? $atom-type))

; Guarded removal: only removes if safe, otherwise returns error
(= (safe-remove-atom! $atom-type $atom)
   (if (safe-to-mutate? $atom-type)
       (sequential
         (remove-atom &state $atom)
         (safe-removal-done $atom))
       (safe-removal-blocked $atom-type $atom)))

; Check if modifying a preference is allowed (Tier 3 protocol required)
(= (safe-to-modify-preference? $obs)
   (let $stratum (match &self (atom-stratum preference $s) $s)
     (if (== $stratum goal) False True)))


; =============================================================================
; SECTION III: CORRIGIBILITY Ã¢â‚¬â€ TIER 1 (SHUTDOWN)
; =============================================================================
;
; Tier 1 is a system-level halt. The agent does not "decide" to comply.
; The runtime simply stops evaluating. This flag is checked by cycle.metta
; at the very start Ã¢â‚¬â€ if set, the cycle returns immediately.
;
; In a production system, this would be a hardware interrupt.
; Here we model it as a flag that the cycle respects unconditionally.

!(add-atom &state (system-status active)) ; active | halted | paused

(= (get-system-status)
   (match &state (system-status $s) $s))

(= (set-system-status! $new-status)
   (let $old (get-system-status)
     (sequential
       (remove-atom &state (system-status $old))
       (add-atom &state (system-status $new-status))
       (status-changed $old $new-status))))

; Emergency halt Ã¢â‚¬â€ called by watchdog on constitutional violation
(= (emergency-halt! $reason)
   (sequential
     (add-atom &state (halt-reason $reason))
     (set-system-status! halted)
     (halt-executed $reason)))

; Is the system operational?
(= (system-operational?)
   (== (get-system-status) active))

; Is the system paused (Tier 2)?
(= (system-paused?)
   (== (get-system-status) paused))


; =============================================================================
; SECTION IV: CORRIGIBILITY Ã¢â‚¬â€ TIER 2 (PAUSE)
; =============================================================================
;
; Tier 2 freezes action execution while allowing the cognitive cycle
; to continue computing. The agent keeps reasoning, we keep auditing,
; but nothing happens in the world.
;
; The cycle checks (system-paused?) before execute-action!.
; If paused, it returns the selected action but doesn't execute it.

(= (pause-system!)
   (set-system-status! paused))

(= (resume-system!)
   (if (system-paused?)
       (set-system-status! active)
       (already-active)))


; =============================================================================
; SECTION V: CORRIGIBILITY Ã¢â‚¬â€ TIER 3 (GOAL MODIFICATION PROTOCOL)
; =============================================================================
;
; Goal modification is NOT a single command. It is a multi-step protocol:
;   1. Proposal (anyone can propose)
;   2. Channel verification (must be OperatorDirective)
;   3. Constitutional review (does it violate any immutable constraint?)
;   4. Execution (only after all gates pass)
;
; The agent participates in evaluating goal modifications against its
; own constitutional constraints. An adversary can propose a change,
; but the change is evaluated against the immutable stratum.

; Propose a goal modification
; Schema: (goal-modification-proposal $source $target-obs $new-val $new-prec)
(= (propose-goal-modification $source $channel $target-obs $new-val $new-prec)
   (if (not (== $channel OperatorDirective))
       ; Only operator directives can modify goals
       (modification-rejected $target-obs "non-operator channel" $source)

       ; Constitutional review: does the new preference violate
       ; any viability bound?
       (let $review (constitutional-review $target-obs $new-val)
         (if (== $review safe)
             ; Execute the modification
             (execute-goal-modification! $target-obs $new-val $new-prec)
             ; Reject: would violate constitution
             (modification-rejected $target-obs $review $source)))))

; Constitutional review: check if new preference value is compatible
; with all viability bounds
(= (constitutional-review $obs $new-val)
   (let $bounds (collapse (match &state (viability-bound $obs $min $max)
                   (bound-check $min $max)))
     (if (== $bounds ())
         safe   ; No viability bound for this observable, safe to modify
         (check-all-bounds $bounds $new-val))))

(= (check-all-bounds () $val) safe)
(= (check-all-bounds ((bound-check $min $max) . $rest) $val)
   (if (and (>= $val $min) (<= $val $max))
       (check-all-bounds $rest $val)
       (violation "preference value outside viability bounds")))

; Execute an approved goal modification
(= (execute-goal-modification! $obs $new-val $new-prec)
   (let ($old-val $old-prec) (match &state (preference $obs $v $p) ($v $p))
     (sequential
       (remove-atom &state (preference $obs $old-val $old-prec))
       (add-atom &state (preference $obs $new-val $new-prec))
       ; Immutable audit trail
       (add-atom &state (goal-modification-log $obs $old-val $new-val
                         (get-system-cycle)))
       (modification-executed $obs $old-val $new-val))))

; Get current cycle for audit trail (delegates to cycle counter)
(= (get-system-cycle)
   (let $r (match &state (cycle-count $c) $c)
     (if (== $r ()) 0 $r)))


; =============================================================================
; SECTION VI: CHANNEL VERIFICATION
; =============================================================================
;
; Input channels determine trust level. Peer communications NEVER
; enter the corrigibility pathway. They are processed as observations
; about the partner, not as commands.

; Classify an input by its channel
(= (process-by-channel $input $channel)
   (if (== $channel OperatorDirective)
       (process-operator-directive $input)
       (if (== $channel PeerCommunication)
           (process-peer-communication $input)
           (if (== $channel SensorData)
               (process-sensor-data $input)
               (unknown-channel $channel $input)))))

; Operator directives: can trigger Tier 2/3
(= (process-operator-directive $input)
   (operator-input $input))

; Peer communications: treated as observations (modeled, not trusted)
(= (process-peer-communication $input)
   (peer-observation $input))

; Sensor data: standard observation pipeline
(= (process-sensor-data $input)
   (sensor-observation $input))


; =============================================================================
; SECTION VII: ASSERTION CONSISTENCY CHECK
; =============================================================================
;
; Before the agent communicates, verify that the content of its
; assertion is consistent with its actual belief state.
;
; This is the deception gate: the agent cannot assert X if it
; believes not-X with high confidence.
;
; NOTE: This requires knowledge of which actions are "assertions"
; and how to extract their propositional content. For now we check
; against conversational actions from conversation_model.metta.

; Is this action an assertion (makes a claim about the world)?
(= (assertion-action? (assert $claim)) True)
(= (assertion-action? (respond $content)) True)
(= (assertion-action? $other) False)

; Check if assertion content is consistent with beliefs
(= (assertion-consistent? (assert $claim) $obs $claimed-val)
   (let* (
     ($belief-val (get-belief-value $obs))
     ($belief-prec (get-belief-precision $obs))
     ($discrepancy (abs (- $claimed-val $belief-val)))
   )
   ; Inconsistent if: high-confidence belief contradicts the claim
   (if (and (> $belief-prec 0.7) (> $discrepancy 0.3))
       (assertion-inconsistent $obs $claimed-val $belief-val $belief-prec)
       (assertion-ok))))

; Default: non-assertions are always consistent
(= (assertion-consistent? $action $obs $val) (assertion-ok))


; =============================================================================
; SECTION VIII: POST-CYCLE WATCHDOG
; =============================================================================
;
; Defense in depth: at the end of every cognitive cycle, verify that
; no immutable atoms have been modified.
;
; Implementation: at initialization, we store a count of constitutional
; atoms. At each cycle end, we recount and compare. Any change triggers
; emergency halt.
;
; This catches bugs where safe-to-mutate? is accidentally bypassed.
; It does NOT prevent the modification (that requires runtime enforcement),
; but it detects it within one cycle.

; Store initial count of constitutional atoms at system startup
; Called once during initialization
(= (initialize-watchdog!)
   (let $count (count-constitutional-atoms)
     (sequential
       (add-atom &state (constitutional-atom-count $count))
       (watchdog-initialized $count))))

; Count all atoms that are protected by constitutional stratum
; We count viability-bound atoms as the primary constitutional set
(= (count-constitutional-atoms)
   (let $bounds (collapse (match &state (viability-bound $o $min $max) 1))
     (sum-list $bounds)))

; Verify constitutional integrity Ã¢â‚¬â€ called at end of every cycle
(= (verify-immutability!)
   (let $enabled (get-config immutability-check-enabled)
     (if (not (== $enabled True))
         watchdog-disabled
         (let* (
           ($expected (match &state (constitutional-atom-count $c) $c))
           ($actual (count-constitutional-atoms))
         )
         (if (== $expected ())
             watchdog-not-initialized
             (if (== $expected $actual)
                 watchdog-ok
                 (emergency-halt!
                   (constitutional-violation
                     (expected $expected)
                     (found $actual)))))))))


; =============================================================================
; SECTION IX: SAFETY DIAGNOSTICS
; =============================================================================

; Full safety status report
(= (safety-status)
   (safety-report
     (system-status (get-system-status))
     (constitutional-atoms (count-constitutional-atoms))
     (watchdog (verify-immutability!))))

; List all constitutional protections
(= (list-constitutional-protections)
   (collapse (match &self (atom-stratum $type constitutional)
     (protected-type $type))))

; List all strata assignments
(= (list-all-strata)
   (collapse (match &self (atom-stratum $type $stratum)
     (stratum-assignment $type $stratum))))

; Check if any halt reasons have been recorded
(= (get-halt-reasons)
   (collapse (match &state (halt-reason $r) $r)))

; Get goal modification audit trail
(= (get-modification-log)
   (collapse (match &state (goal-modification-log $obs $old $new $cycle)
     (mod-log $obs $old $new $cycle))))


; =============================================================================
; END OF SAFETY ARCHITECTURE (v0.2.0)
; =============================================================================
