; =============================================================================
; METTA COGNITIVE CORE — GROUNDING HYPOTHESES
; =============================================================================
;
; THE PROBLEM THIS SOLVES:
;   Structure learning invents latent variables, promotes observables,
;   and chunks actions. These atoms predict well and pay metabolic rent.
;   But they have NO semantic grounding — the system cannot explain,
;   reason about, or communicate what they mean.
;
;   The grounding system (semantic_grounding.metta, action_grounding.metta)
;   only knows about DECLARED atoms. Learned atoms are invisible to it.
;
; THE SOLUTION:
;   Four hypothesis generators produce grounding atoms using the SAME
;   schemas as declared groundings. One metabolic selection mechanism
;   kills wrong groundings. The existing traversal machinery works
;   unchanged on learned atoms.
;
;   Generator 1: Inherited    — from member observables' shared groundings
;   Generator 2: Behavioral   — from causal effects on grounded observables
;   Generator 3: Compositional — from sequential composition of action parts
;   Generator 4: Conversational — from partner labeling (orchestrator-mediated)
;
; DESIGN PRINCIPLES:
;   - Bottom-Up: Multiple generators propose; metabolic pressure selects
;   - Uniform: Hypothesized groundings use standard obs-grounding schema
;   - Transparent: Every hypothesis is tagged with its generator source
;   - Metabolic: Pay rent or die — same invariant as all structure
;   - Emergent: The system spontaneously seeks to understand its own
;     concepts via EFE pressure on grounding-completeness
;
; ATOM SCHEMAS:
;   (grounding-hyp $target $generator $created-at)
;   (grounding-energy $target $energy)
;   (grounding-stats $target $tests $successes)
;   (grounding-request $target $members $context)
;   (grounding-budget-this-cycle $n)
;
;   Plus standard schemas (produced by generators, consumed by traversal):
;   (obs-grounding $learned-atom $rel $target)
;   (obs-semantic $learned-atom $description)
;   (action-grounding $chunk $rel $target)
;   (action-semantic $chunk $description)
;
; Dependencies: foundations, beliefs, safety, structure_learning, atom_lifecycle
; Depended on by: cycle, self_model, orchestrator.py
;
; Version: 0.1.0
; =============================================================================

!(import! &self foundations)
!(import! &self beliefs)
!(import! &self safety)


; =============================================================================
; SECTION I: INITIALIZATION & BUDGET
; =============================================================================

(grounding-budget-this-cycle 0)

(= (reset-grounding-budget!)
   (let $old (match &self (grounding-budget-this-cycle $n) $n)
     (sequential
       (remove-atom &self (grounding-budget-this-cycle $old))
       (add-atom &self (grounding-budget-this-cycle 0))
       done)))

(= (grounding-budget-available?)
   (let $used (match &self (grounding-budget-this-cycle $n) $n)
     (< $used (get-config grounding-max-per-cycle))))

(= (use-grounding-budget!)
   (let $old (match &self (grounding-budget-this-cycle $n) $n)
     (sequential
       (remove-atom &self (grounding-budget-this-cycle $old))
       (add-atom &self (grounding-budget-this-cycle (+ $old 1)))
       done)))


; =============================================================================
; SECTION II: GROUNDING ENERGY MANAGEMENT
; =============================================================================

(= (get-grounding-energy $target)
   (let $r (match &self (grounding-energy $target $e) $e)
     (if (== $r ()) 0.0 $r)))

(= (set-grounding-energy! $target $energy)
   (let $old (collapse (match &self (grounding-energy $target $e)
                                    (grounding-energy $target $e)))
     (sequential
       (remove-atom-list-safe! $old)
       (add-atom &self (grounding-energy $target
                         (clamp $energy 0.0 (get-config metabolic-energy-cap))))
       done)))

(= (drain-grounding-energy! $target)
   (let $current (get-grounding-energy $target)
     (set-grounding-energy! $target
       (- $current (get-config grounding-metabolic-rate)))))

(= (boost-grounding-energy! $target)
   (let $current (get-grounding-energy $target)
     (set-grounding-energy! $target
       (+ $current (get-config grounding-metabolic-boost)))))

(= (grounding-energy-depleted? $target)
   (<= (get-grounding-energy $target) 0.0))


; =============================================================================
; SECTION III: GROUNDING HYPOTHESIS TRACKING
; =============================================================================

; Check if an atom has any grounding hypothesis
(= (has-grounding-hyp? $target)
   (let $r (collapse (match &self (grounding-hyp $target $gen $cyc) True))
     (if (== $r ()) False True)))

; Check if an atom has grounding from a specific generator
(= (has-grounding-from? $target $generator)
   (let $r (collapse (match &self (grounding-hyp $target $generator $cyc) True))
     (if (== $r ()) False True)))

; Get all grounding hypotheses for an atom
(= (get-groundings-for $target)
   (collapse (match &self (grounding-hyp $target $gen $cyc)
     (grounding-source $gen $cyc))))

; Get all atoms with grounding hypotheses
(= (all-grounded-learned-atoms)
   (collapse (match &self (grounding-hyp $target $gen $cyc) $target)))

; Register a new grounding hypothesis with metadata
(= (register-grounding-hyp! $target $generator $cycle)
   (if (has-grounding-from? $target $generator)
       (already-grounded $target $generator)
       (sequential
         (add-atom &self (grounding-hyp $target $generator $cycle))
         ; Initialize energy if this is the first grounding for this target
         (if (== (get-grounding-energy $target) 0.0)
             (set-grounding-energy! $target
               (get-config grounding-initial-energy))
             done)
         ; Initialize stats if needed
         (let $stats (collapse (match &self (grounding-stats $target $t $s) True))
           (if (== $stats ())
               (add-atom &self (grounding-stats $target 0 0))
               done))
         (grounding-registered $target $generator $cycle))))

; Update grounding stats after a test
(= (update-grounding-stats! $target $success?)
   (let ($tests $successes) (match &self (grounding-stats $target $t $s) ($t $s))
     (sequential
       (remove-atom &self (grounding-stats $target $tests $successes))
       (add-atom &self (grounding-stats $target
                         (+ $tests 1)
                         (+ $successes (if $success? 1 0))))
       done)))


; =============================================================================
; SECTION IV: GENERATOR 1 — INHERITED GROUNDING
; =============================================================================
;
; When a latent variable is created from a hub of observables, inherit
; grounding from the members. A grounding relation is inherited if at
; least (threshold) fraction of members share it.
;
; TRIGGER: New latent variable exists without grounding
; INPUT:   Latent variable name → member observables → their obs-grounding atoms
; OUTPUT:  obs-grounding atoms for the latent variable
;
; This captures "what kind of thing is this?" from the shared nature of
; the cluster members. Fire's members (smoke, heat, light) all ground
; to physical-process → fire inherits physical-process.

; Collect all (rel, target) pairs from a member observable's groundings
(= (member-groundings $obs)
   (collapse (match &self (obs-grounding $obs $rel $target) ($rel $target))))

; Collect groundings from all members of a latent variable
(= (all-member-groundings $lv-name)
   (let $members (get-latent-members $lv-name)
     (collect-member-groundings $members)))

(= (collect-member-groundings ()) ())
(= (collect-member-groundings ($obs . $rest))
   (list-append (member-groundings $obs) (collect-member-groundings $rest)))

; Count how many members share a specific (rel, target) pair
(= (count-members-with-grounding $lv-name $rel $target)
   (let $members (get-latent-members $lv-name)
     (count-matching-members $members $rel $target)))

(= (count-matching-members () $rel $target) 0)
(= (count-matching-members ($obs . $rest) $rel $target)
   (let $has (collapse (match &self (obs-grounding $obs $rel $target) True))
     (+ (if (== $has ()) 0 1)
        (count-matching-members $rest $rel $target))))

; Get unique (rel, target) pairs from the collected groundings
(= (unique-groundings ()) ())
(= (unique-groundings (($rel $target) . $rest))
   (if (list-contains? ($rel $target) $rest)
       (unique-groundings $rest)
       (($rel $target) . (unique-groundings $rest))))

; Run inherited grounding for one latent variable
(= (generate-inherited-grounding! $lv-name $cycle)
   (if (has-grounding-from? $lv-name inherited)
       (already-grounded $lv-name inherited)
   (if (not (grounding-budget-available?))
       grounding-budget-exhausted
   (let* (
     ($members (get-latent-members $lv-name))
     ($n-members (length $members))
     ($threshold (get-config grounding-inheritance-threshold))
     ($min-count (* $n-members $threshold))
     ($all-pairs (all-member-groundings $lv-name))
     ($unique-pairs (unique-groundings $all-pairs))
   )
   (emit-inherited-groundings! $lv-name $unique-pairs $min-count $cycle)))))

; Emit grounding atoms for pairs that pass the threshold
(= (emit-inherited-groundings! $lv-name () $min-count $cycle) done)
(= (emit-inherited-groundings! $lv-name (($rel $target) . $rest) $min-count $cycle)
   (let $count (count-members-with-grounding $lv-name $rel $target)
     (sequential
       (if (>= $count $min-count)
           (sequential
             (add-atom &self (obs-grounding $lv-name $rel $target))
             done)
           done)
       (emit-inherited-groundings! $lv-name $rest $min-count $cycle))))

; Finalize: register hypothesis and use budget (called once per latent)
(= (finalize-inherited-grounding! $lv-name $cycle)
   (let $emitted (collapse (match &self (obs-grounding $lv-name $rel $target) True))
     (if (== $emitted ())
         (no-inherited-grounding $lv-name)  ; Intersection was empty
         (sequential
           (register-grounding-hyp! $lv-name inherited $cycle)
           (use-grounding-budget!)
           (inherited-grounding-created $lv-name)))))

; Run Generator 1 on all ungrounded latent variables
(= (run-inherited-grounding! $cycle)
   (let $latents (all-latent-variables)
     (inherited-grounding-list! $latents $cycle)))

(= (inherited-grounding-list! () $cycle) done)
(= (inherited-grounding-list! ($lv . $rest) $cycle)
   (sequential
     (if (not (has-grounding-hyp? $lv))
         (sequential
           (generate-inherited-grounding! $lv $cycle)
           (finalize-inherited-grounding! $lv $cycle))
         done)
     (inherited-grounding-list! $rest $cycle)))


; =============================================================================
; SECTION V: GENERATOR 2 — BEHAVIORAL GROUNDING
; =============================================================================
;
; Ground a learned atom by what it DOES — its causal effects on other
; observables that ARE grounded. If $L_1 excites threat-level, and
; threat-level is grounded as "danger", then $L_1 "causes danger".
;
; TRIGGER: Learned atom has passive-model connections to grounded
;          observables with sufficient metabolic energy.
; INPUT:   Passive models where learned atom is cause or effect
; OUTPUT:  obs-grounding atoms based on causal semantics
;
; This handles the heterogeneous case (members have nothing in common)
; and captures emergent properties (fire causes danger — neither smoke
; nor heat does directly).

; Get grounded effects of a learned atom (via passive-model, cause→effect)
(= (grounded-effects-of $learned-atom)
   (collapse
     (match &self (passive-model $learned-atom $effect $lag $weight $type)
       (if (is-grounded-obs? $effect)
           (causal-effect $effect $type $weight)
           (empty)))))

; Get grounded causes of a learned atom (via passive-model, cause→effect)
(= (grounded-causes-of $learned-atom)
   (collapse
     (match &self (passive-model $cause $learned-atom $lag $weight $type)
       (if (is-grounded-obs? $cause)
           (causal-cause $cause $type $weight)
           (empty)))))

; Check if an observable has declared grounding (not hypothesized)
(= (is-grounded-obs? $obs)
   (let $r (collapse (match &self (obs-grounding $obs $rel $target) True))
     (if (== $r ()) False True)))

; Map causal type to grounding relation
; excitatory cause → "causes" the effect's nature
; inhibitory cause → "prevents" the effect's nature
(= (causal-rel excitatory) causes)
(= (causal-rel inhibitory) prevents)

; Generate behavioral grounding from causal effects
(= (generate-behavioral-from-effects! $learned-atom $effects $cycle)
   (behavioral-effects-list! $learned-atom $effects $cycle))

(= (behavioral-effects-list! $atom () $cycle) done)
(= (behavioral-effects-list! $atom ((causal-effect $effect $type $weight) . $rest) $cycle)
   (let* (
     ; Only use effects from stable passive models
     ($energy (get-pm-energy $atom $effect))
     ($min-energy (get-config grounding-behavioral-min-energy))
   )
   (sequential
     (if (and (not (== $energy ())) (>= $energy $min-energy))
         ; Get the effect's primary grounding and map through causal type
         (let $effect-groundings (collapse
               (match &self (obs-grounding $effect is-a $target) $target))
           (emit-behavioral-groundings! $atom (causal-rel $type)
             $effect-groundings))
         done)
     (behavioral-effects-list! $atom $rest $cycle))))

; Emit one obs-grounding per effect-grounding target
(= (emit-behavioral-groundings! $atom $rel ()) done)
(= (emit-behavioral-groundings! $atom $rel ($target . $rest))
   (sequential
     ; Don't duplicate existing groundings
     (let $exists (collapse (match &self (obs-grounding $atom $rel $target) True))
       (if (== $exists ())
           (add-atom &self (obs-grounding $atom $rel $target))
           done))
     (emit-behavioral-groundings! $atom $rel $rest)))

; Generate behavioral grounding from viability signal
; If this atom being high-belief correlates with viability pressure,
; ground it as threat-related
(= (generate-viability-grounding! $learned-atom $cycle)
   (let $belief-result (collapse (match &state (belief $learned-atom $val $prec) ($val $prec)))
     (if (== $belief-result ())
         done  ; No belief, can't assess
         (let ($val $prec) (list-head $belief-result)
           ; If belief value is high AND viability pressure is elevated,
           ; the concept might be threat-related
           (if (and (> $val 0.6) (> (get-viability-pressure) 0.3))
               (let $exists (collapse (match &self (obs-grounding $learned-atom causes danger) True))
                 (if (== $exists ())
                     (add-atom &self (obs-grounding $learned-atom causes danger))
                     done))
               done)))))

; Run Generator 2 on a single learned atom
(= (generate-behavioral-grounding! $learned-atom $cycle)
   (if (has-grounding-from? $learned-atom behavioral)
       (already-grounded $learned-atom behavioral)
   (if (not (grounding-budget-available?))
       grounding-budget-exhausted
   (let* (
     ($effects (grounded-effects-of $learned-atom))
     ($causes (grounded-causes-of $learned-atom))
     ($gen-effects (generate-behavioral-from-effects! $learned-atom $effects $cycle))
     ($gen-viability (generate-viability-grounding! $learned-atom $cycle))
   )
   ; Check if anything was generated
   (let $new-groundings (collapse (match &self (obs-grounding $learned-atom $r $t) True))
     (if (== $new-groundings ())
         (no-behavioral-grounding $learned-atom)
         (sequential
           (register-grounding-hyp! $learned-atom behavioral $cycle)
           (use-grounding-budget!)
           (behavioral-grounding-created $learned-atom))))))))

; Run Generator 2 on all ungrounded learned atoms that have passive models
(= (run-behavioral-grounding! $cycle)
   (let* (
     ; Collect all learned atoms: latent variables + promoted observables
     ($latents (all-latent-variables))
     ($promoted (all-promoted-observables))
     ($all (list-append $latents $promoted))
   )
   (behavioral-grounding-list! $all $cycle)))

(= (behavioral-grounding-list! () $cycle) done)
(= (behavioral-grounding-list! ($atom . $rest) $cycle)
   (sequential
     ; Only try behavioral grounding if no inherited grounding exists
     ; (behavioral is the fallback for heterogeneous clusters)
     (if (and (not (has-grounding-from? $atom behavioral))
              (not (has-grounding-from? $atom inherited)))
         (generate-behavioral-grounding! $atom $cycle)
         done)
     (behavioral-grounding-list! $rest $cycle)))


; =============================================================================
; SECTION VI: GENERATOR 3 — COMPOSITIONAL GROUNDING
; =============================================================================
;
; When Phase 4 reifies a policy [observe, observe, retreat] as the
; chunked action "observe-observe-retreat", compose the groundings of
; the component actions.
;
; TRIGGER: New chunked action without grounding
; INPUT:   Chunked action → component action sequence → their groundings
; OUTPUT:  action-grounding atoms for the chunk
;
; The chunk IS-A sequence of its parts. It CAUSES the sequential
; composition of effects. It ENABLES what the last action enables.
; It REQUIRES what the first action requires.

; Get the action sequence for a chunked action
(= (get-chunk-sequence $chunk-name)
   (let $r (collapse (match &self (chunked-action-policy $chunk-name $seq) $seq))
     (if (== $r ()) () (list-head $r))))

; Collect the "causes" groundings of an action
(= (action-causes $action)
   (collapse (match &self (action-grounding $action causes $target) $target)))

; Collect the "enables" groundings of an action
(= (action-enables $action)
   (collapse (match &self (action-grounding $action enables $target) $target)))

; Collect the "requires" groundings of an action
(= (action-requires $action)
   (collapse (match &self (action-grounding $action requires $target) $target)))

; Get semantic description of an action (empty string if none)
(= (action-desc $action)
   (let $r (collapse (match &self (action-semantic $action $d) $d))
     (if (== $r ()) "unknown-action" (list-head $r))))

; Get the last element of a list
(= (list-last ($x)) $x)
(= (list-last ($x . $rest)) (list-last $rest))

; Get the first element of a list
(= (list-first ($x . $rest)) $x)

; Generate compositional grounding for a chunked action
(= (generate-compositional-grounding! $chunk-name $cycle)
   (if (has-grounding-from? $chunk-name compositional)
       (already-grounded $chunk-name compositional)
   (if (not (grounding-budget-available?))
       grounding-budget-exhausted
   (let $seq (get-chunk-sequence $chunk-name)
     (if (== $seq ())
         (no-sequence-for $chunk-name)
     (let* (
       ; IS-A: a sequence of its components
       ($is-a-created (add-atom &self
         (action-grounding $chunk-name is-a (sequence . $seq))))

       ; CAUSES: union of all component effects
       ($all-effects (collect-sequence-effects $seq))
       ($effects-created (emit-chunk-groundings! $chunk-name causes $all-effects))

       ; ENABLES: what the last action enables
       ($last-action (list-last $seq))
       ($last-enables (action-enables $last-action))
       ($enables-created (emit-chunk-groundings! $chunk-name enables $last-enables))

       ; REQUIRES: what the first action requires
       ($first-action (list-first $seq))
       ($first-requires (action-requires $first-action))
       ($requires-created (emit-chunk-groundings! $chunk-name requires $first-requires))

       ; SEMANTIC: compose descriptions
       ($descs (collect-sequence-descs $seq))
       ($composed-desc (join-descriptions $descs))
       ($desc-created (add-atom &self (action-semantic $chunk-name $composed-desc)))

       ; Register and budget
       ($registered (register-grounding-hyp! $chunk-name compositional $cycle))
       ($budget (use-grounding-budget!))
     )
     (compositional-grounding-created $chunk-name)))))))

; Collect all causal effects across a sequence of actions
(= (collect-sequence-effects ()) ())
(= (collect-sequence-effects ($action . $rest))
   (list-append (action-causes $action) (collect-sequence-effects $rest)))

; Collect descriptions across a sequence
(= (collect-sequence-descs ()) ())
(= (collect-sequence-descs ($action . $rest))
   ((action-desc $action) . (collect-sequence-descs $rest)))

; Join descriptions with "then" connective
(= (join-descriptions ()) "")
(= (join-descriptions ($d)) $d)
(= (join-descriptions ($d . $rest))
   (let $rest-joined (join-descriptions $rest)
     (string-concat $d (string-concat " then " $rest-joined))))

; Helper: string concatenation (MeTTa built-in or fallback)
; NOTE: If MeTTa runtime lacks string-concat, this becomes a
; structured representation instead. The structure is still usable
; for reasoning even if not a flat string.
(= (string-concat $a $b)
   (let $r (concat $a $b)
     (if (== $r ()) (desc-parts $a $b) $r)))

; Emit action-grounding atoms for a list of targets
(= (emit-chunk-groundings! $chunk $rel ()) done)
(= (emit-chunk-groundings! $chunk $rel ($target . $rest))
   (sequential
     (let $exists (collapse (match &self (action-grounding $chunk $rel $target) True))
       (if (== $exists ())
           (add-atom &self (action-grounding $chunk $rel $target))
           done))
     (emit-chunk-groundings! $chunk $rel $rest)))

; Run Generator 3 on all ungrounded chunked actions
(= (run-compositional-grounding! $cycle)
   (let $chunks (all-chunked-actions)
     (compositional-grounding-list! $chunks $cycle)))

(= (compositional-grounding-list! () $cycle) done)
(= (compositional-grounding-list! ($chunk . $rest) $cycle)
   (sequential
     (if (not (has-grounding-hyp? $chunk))
         (generate-compositional-grounding! $chunk $cycle)
         done)
     (compositional-grounding-list! $rest $cycle)))


; =============================================================================
; SECTION VII: GENERATOR 4 — CONVERSATIONAL ACQUISITION (Signaling)
; =============================================================================
;
; When internal generators fail, signal the orchestrator to ask the
; conversational partner. The MeTTa core doesn't do NL — it creates a
; structured request. The orchestrator reads it, queries the LLM, and
; injects the result back as standard grounding atoms.
;
; TRIGGER: Active learned atom with no grounding after Generators 1-3
; OUTPUT:  (grounding-request $atom $member-list $behavioral-context)
;
; The orchestrator queries this and acts on it. The EFE integration
; happens naturally: ungrounded atoms lower grounding-completeness,
; which creates info-gain for clarification actions.

; Collect context for a grounding request
(= (build-grounding-request $atom)
   (let* (
     ; What are its members? (for latent variables)
     ($members (collapse (match &self (latent-member $atom $obs $w) $obs)))

     ; What are its behavioral effects? (for any learned atom)
     ($effects (collapse
       (match &self (passive-model $atom $effect $l $w $t)
         (effect $effect $t))))

     ; What is its current belief state?
     ($belief (collapse (match &state (belief $atom $val $prec) ($val $prec))))
   )
   (grounding-request $atom $members
     (context (effects $effects) (belief $belief)))))

; Signal all ungrounded active atoms
(= (signal-grounding-requests! $cycle)
   (let* (
     ; Collect all learned atoms
     ($latents (all-latent-variables))
     ($promoted (all-promoted-observables))
     ($chunks (all-chunked-actions))
     ($all (list-append $latents (list-append $promoted $chunks)))
   )
   (signal-request-list! $all $cycle)))

(= (signal-request-list! () $cycle) done)
(= (signal-request-list! ($atom . $rest) $cycle)
   (sequential
     (if (and (not (has-grounding-hyp? $atom))
              ; Only signal if atom is metabolically healthy
              (not (grounding-energy-depleted? $atom)))
         (let $request (build-grounding-request $atom)
           (add-atom &self $request))
         done)
     (signal-request-list! $rest $cycle)))

; Check if there are pending grounding requests (for orchestrator to query)
(= (pending-grounding-requests)
   (collapse (match &self (grounding-request $atom $members $ctx)
     (grounding-request $atom $members $ctx))))

(= (has-pending-grounding-requests?)
   (let $r (pending-grounding-requests)
     (not (list-empty? $r))))

; Inject conversational grounding (called by orchestrator after LLM parse)
; The orchestrator has parsed the LLM's response into a type and label.
(= (inject-conversational-grounding! $atom $type $label $cycle)
   (sequential
     ; Create the standard grounding atoms
     (add-atom &self (obs-grounding $atom is-a $type))
     (add-atom &self (obs-semantic $atom $label))
     ; Register hypothesis
     (register-grounding-hyp! $atom conversational $cycle)
     ; Clear the request
     (let $old-requests (collapse
       (match &self (grounding-request $atom $m $c)
                    (grounding-request $atom $m $c)))
       (remove-atom-list-safe! $old-requests))
     (conversational-grounding-injected $atom $type $label)))


; =============================================================================
; SECTION VIII: METABOLIC MANAGEMENT
; =============================================================================
;
; Grounding hypotheses pay rent and earn reward, same as all structure.
; The rent is lower than causal structure (groundings are tested less
; often — only when the system explains itself or communicates).

; Check if a grounding is still in gestation
(= (grounding-gestating? $target $current-cycle)
   (let $created (collapse (match &self (grounding-hyp $target $gen $cyc) $cyc))
     (if (== $created ())
         False
         (< (- $current-cycle (list-head $created))
            (get-config grounding-gestation-period)))))

; Drain all grounding hypotheses
(= (drain-all-groundings! $current-cycle)
   (let $targets (all-grounded-learned-atoms)
     (let $unique (unique-groundings-targets $targets)
       (drain-grounding-list! $unique $current-cycle))))

; Deduplicate target list (atom may have multiple generators)
(= (unique-groundings-targets ()) ())
(= (unique-groundings-targets ($x . $rest))
   (if (list-contains? $x $rest)
       (unique-groundings-targets $rest)
       ($x . (unique-groundings-targets $rest))))

(= (drain-grounding-list! () $cycle) done)
(= (drain-grounding-list! ($target . $rest) $cycle)
   (sequential
     (if (not (grounding-gestating? $target $cycle))
         (drain-grounding-energy! $target)
         done)
     (drain-grounding-list! $rest $cycle)))

; Prune groundings with depleted energy
(= (prune-dead-groundings! $current-cycle)
   (let $targets (all-grounded-learned-atoms)
     (let $unique (unique-groundings-targets $targets)
       (prune-grounding-list! $unique $current-cycle))))

(= (prune-grounding-list! () $cycle) done)
(= (prune-grounding-list! ($target . $rest) $cycle)
   (sequential
     (if (grounding-energy-depleted? $target)
         (remove-grounding! $target $cycle)
         done)
     (prune-grounding-list! $rest $cycle)))

; Remove all grounding atoms for a target
(= (remove-grounding! $target $cycle)
   (let* (
     ; Remove obs-grounding atoms
     ($obs-grnds (collapse (match &self (obs-grounding $target $r $t)
                                        (obs-grounding $target $r $t))))
     ($obs-removed (remove-atom-list-safe! $obs-grnds))

     ; Remove action-grounding atoms
     ($act-grnds (collapse (match &self (action-grounding $target $r $t)
                                        (action-grounding $target $r $t))))
     ($act-removed (remove-atom-list-safe! $act-grnds))

     ; Remove semantic atoms
     ($obs-sem (collapse (match &self (obs-semantic $target $d)
                                      (obs-semantic $target $d))))
     ($obs-sem-removed (remove-atom-list-safe! $obs-sem))
     ($act-sem (collapse (match &self (action-semantic $target $d)
                                      (action-semantic $target $d))))
     ($act-sem-removed (remove-atom-list-safe! $act-sem))

     ; Remove hypothesis metadata
     ($hyps (collapse (match &self (grounding-hyp $target $g $c)
                                   (grounding-hyp $target $g $c))))
     ($hyps-removed (remove-atom-list-safe! $hyps))

     ; Remove energy and stats
     ($energy (collapse (match &self (grounding-energy $target $e)
                                     (grounding-energy $target $e))))
     ($energy-removed (remove-atom-list-safe! $energy))
     ($stats (collapse (match &self (grounding-stats $target $t $s)
                                    (grounding-stats $target $t $s))))
     ($stats-removed (remove-atom-list-safe! $stats))

     ; Remove any pending requests
     ($reqs (collapse (match &self (grounding-request $target $m $c)
                                   (grounding-request $target $m $c))))
     ($reqs-removed (remove-atom-list-safe! $reqs))

     ; Audit log
     ($log (add-atom &self (grounding-pruned-log $target $cycle)))
   )
   (grounding-removed $target $cycle)))


; =============================================================================
; SECTION IX: REWARD CHANNELS
; =============================================================================
;
; Three ways grounding hypotheses earn energy:
;   1. Communication: concept used in narrative, partner comprehension OK
;   2. Reasoning: grounded concept participates in successful deduction
;   3. Self-model: grounding-completeness moves toward preference

; Reward a grounding after successful communication
; Called by the cycle after verbalization + comprehension check
(= (reward-grounding-communication! $target)
   (if (has-grounding-hyp? $target)
       (sequential
         (boost-grounding-energy! $target)
         (update-grounding-stats! $target True)
         (grounding-rewarded $target communication))
       done))

; Reward a grounding when its concept participates in reasoning
; Called when a deductive chain uses a grounded learned concept
(= (reward-grounding-reasoning! $target)
   (if (has-grounding-hyp? $target)
       (sequential
         ; Smaller boost for reasoning (less direct feedback)
         (let $current (get-grounding-energy $target)
           (set-grounding-energy! $target
             (+ $current (* 0.5 (get-config grounding-metabolic-boost)))))
         (update-grounding-stats! $target True)
         (grounding-rewarded $target reasoning))
       done))

; Penalize a grounding after communication failure
; Called when verbalization using the concept leads to comprehension drop
(= (penalize-grounding-communication! $target)
   (if (has-grounding-hyp? $target)
       (sequential
         (update-grounding-stats! $target False)
         ; No explicit energy drain — rent handles decay.
         ; But we mark the failure for diagnostics.
         (grounding-penalized $target communication))
       done))


; =============================================================================
; SECTION X: GROUNDING COMPLETENESS OBSERVABLE
; =============================================================================
;
; Self-model observable: what fraction of active learned atoms are
; semantically grounded? This creates continuous EFE pressure toward
; self-understanding.
;
; Updated each cycle as part of the grounding step.

; Count active learned atoms (latent + promoted + chunked)
(= (count-active-learned-atoms)
   (let* (
     ($n-latent (length (all-latent-variables)))
     ($n-promoted (length (all-promoted-observables)))
     ($n-chunked (length (all-chunked-actions)))
   )
   (+ $n-latent (+ $n-promoted $n-chunked))))

; Count grounded learned atoms (have at least one grounding hypothesis)
(= (count-grounded-learned-atoms)
   (length (unique-groundings-targets (all-grounded-learned-atoms))))

; Compute grounding completeness ratio
(= (grounding-completeness-ratio)
   (let $total (count-active-learned-atoms)
     (if (== $total 0)
         1.0  ; No learned atoms = fully grounded (vacuously)
         (/ (count-grounded-learned-atoms) $total))))

; Update the grounding-completeness belief
(= (update-grounding-completeness!)
   (let* (
     ($ratio (grounding-completeness-ratio))
     ; Precision increases with more data points
     ($total (count-active-learned-atoms))
     ($prec (clamp (+ 0.3 (* 0.05 $total)) 0.3 0.8))
   )
   ; Use set-belief! from beliefs.metta
   (set-belief! grounding-completeness $ratio $prec)))

; List all atoms that lack grounding
(= (ungrounded-learned-atoms)
   (let* (
     ($latents (all-latent-variables))
     ($promoted (all-promoted-observables))
     ($chunks (all-chunked-actions))
     ($all (list-append $latents (list-append $promoted $chunks)))
   )
   (filter-ungrounded-learned $all)))

(= (filter-ungrounded-learned ()) ())
(= (filter-ungrounded-learned ($atom . $rest))
   (if (has-grounding-hyp? $atom)
       (filter-ungrounded-learned $rest)
       ($atom . (filter-ungrounded-learned $rest))))


; =============================================================================
; SECTION XI: MASTER GROUNDING STEP
; =============================================================================
;
; Called once per cognitive cycle, AFTER structure learning, BEFORE
; action selection. This is a REFLEX — fires when conditions are met,
; not selected by EFE.
;
; Order matters:
;   1. Inherited first (fast, principled, handles easy cases)
;   2. Behavioral second (slower, handles hard cases)
;   3. Compositional third (only for chunked actions)
;   4. Signal remaining gaps for conversational acquisition
;   5. Metabolic management (drain + prune)
;   6. Update self-model observable

(= (grounding-step! $current-cycle)
   (sequential
     ; Reset per-cycle budget
     (reset-grounding-budget!)

     ; Generator 1: Inherited grounding for latent variables
     (run-inherited-grounding! $current-cycle)

     ; Generator 2: Behavioral grounding for all learned atoms
     (run-behavioral-grounding! $current-cycle)

     ; Generator 3: Compositional grounding for chunked actions
     (run-compositional-grounding! $current-cycle)

     ; Generator 4: Signal ungrounded atoms for conversational acquisition
     (signal-grounding-requests! $current-cycle)

     ; Metabolic: drain energy from non-gestating hypotheses
     (drain-all-groundings! $current-cycle)

     ; Metabolic: prune dead groundings
     (prune-dead-groundings! $current-cycle)

     ; Update self-model observable
     (update-grounding-completeness!)

     (grounding-step-done $current-cycle)))


; =============================================================================
; SECTION XII: DIAGNOSTICS
; =============================================================================

; Complete grounding status summary
(= (grounding-summary)
   (let* (
     ($total (count-active-learned-atoms))
     ($grounded (count-grounded-learned-atoms))
     ($ungrounded (ungrounded-learned-atoms))
     ($ratio (grounding-completeness-ratio))
     ($pending (length (pending-grounding-requests)))
   )
   (grounding-report
     (total-learned $total)
     (grounded $grounded)
     (ungrounded-list $ungrounded)
     (completeness $ratio)
     (pending-requests $pending))))

; Detailed grounding report for a specific atom
(= (grounding-detail $atom)
   (let* (
     ($hyps (get-groundings-for $atom))
     ($energy (get-grounding-energy $atom))
     ($stats (match &self (grounding-stats $atom $t $s) ($t $s)))
     ($obs-grnds (collapse (match &self (obs-grounding $atom $r $t) ($r $t))))
     ($act-grnds (collapse (match &self (action-grounding $atom $r $t) ($r $t))))
     ($semantics (collapse (match &self (obs-semantic $atom $d) $d)))
   )
   (grounding-detail-report
     (atom $atom)
     (sources $hyps)
     (energy $energy)
     (stats $stats)
     (obs-groundings $obs-grnds)
     (action-groundings $act-grnds)
     (semantic-labels $semantics))))

; Audit: history of pruned groundings
(= (grounding-prune-history)
   (collapse (match &self (grounding-pruned-log $target $cycle)
     (pruned $target $cycle))))


; =============================================================================
; END OF GROUNDING HYPOTHESES (v0.1.0)
; =============================================================================
