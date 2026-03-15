; =============================================================================
; METTA COGNITIVE CORE - ABDUCTIVE INFERENCE (Phase 1.6)
; =============================================================================
;
; Completes the Peircean reasoning triad:
;   Phase 1:   INDUCTION   â€” learn correlations from co-error (structure)
;   Phase 1.5: DEDUCTION   â€” transitive closure over structure (structure)
;   Phase 1.6: ABDUCTION   â€” inverse model activation (STATE from structure)
;
; DESIGN PRINCIPLES:
;   - Bottom-Up: No "if smoke then check fire" rules. The EFE landscape
;     naturally drives verification when hypotheses create low-precision
;     beliefs (high information gain for observations that would confirm).
;   - Reflexive: Fires when surprise conditions are met, like structure
;     learning. Not selected by EFE.
;   - Honest: Hypotheses inject LOW precision beliefs. The system knows
;     they are guesses via (belief-source $obs abduced $cycle) markers.
;     "I think there's fire, but that's a hypothesis" emerges from the
;     epistemic state, not a scripted disclaimer.
;   - Transparent: Every abductive inference is logged with the evidence
;     and model that produced it.
;   - Single-Step: Only immediate causes are hypothesized per cycle.
;     Multi-hop backward reasoning (Aâ†Bâ†C) emerges over multiple cycles
;     as each hypothesis becomes a surprising belief in the next round.
;
; KEY DISTINCTION FROM OTHER PHASES:
;   Induction creates STRUCTURE (passive-model atoms)
;   Deduction extends STRUCTURE (new passive-model from existing ones)
;   Abduction creates STATE (belief updates from existing structure)
;
;   This means abduction uses the causal graph built by Phases 1-2
;   but does not modify it. It runs the graph backward to propagate
;   beliefs from observed effects to unobserved causes.
;
; THE MECHANISM:
;   Forward model: high cause â†’ high effect (excitatory)
;   Abduction:     high effect observed + (causeâ†’effect) â‡’ cause is high
;
;   The hypothesis value is computed by inverting the forward model:
;     hyp_val = clamp(observed_effect / weight, 0, 1)
;   For inhibitory links, the direction flips:
;     high effect + inhibitory(causeâ†’effect) â‡’ cause is LOW
;
;   The hypothesis precision is ALWAYS LOW (configurable, default 0.10).
;   This ensures the hypothesis is a whisper in the belief landscape,
;   not a shout. It takes multiple confirming observations or several
;   cycles of consistent abduction to move the belief meaningfully.
;
; THE PAYOFF ("Sherlock Holmes" Effect):
;   1. Agent sees smoke (high surprise on smoke observable)
;   2. Abduction: passive-model(fireâ†’smoke) â‡’ hypothesize fire=high
;   3. Fire belief now has LOW precision (0.10)
;   4. EFE computation: observe(heat) has massive info gain because
;      fire predicts heat, and confirming fire would resolve the
;      low-precision hypothesis
;   5. Agent chooses to check for heat â€” EMERGENT verification
;
; ATOM SCHEMAS:
;   (belief-source $obs abduced $cycle)     â€” marks abductive origin
;   (belief-source $obs observed $cycle)    â€” marks observational origin
;   (abduction-log $cause $effect $evidence $hyp-val $hyp-prec $cycle)
;
; DEPENDENCIES: foundations, beliefs, structure_learning (for passive-model)
; DEPENDED ON BY: cycle (step 4b-d)
;
; Version: 0.2.0 (boundary condition fix — capital-gated)
; =============================================================================

!(import! &self foundations)
!(import! &self beliefs)


; =============================================================================
; SECTION I: CONFIGURATION
; =============================================================================
;
; Abductive inference parameters. These could become beliefs with
; precision (like structure learning meta-parameters) to enable
; bootstrap tuning, but start as simple configs for clarity.

; Minimum surprise to trigger abduction on an observable.
; Below this, the observation is "expected enough" that we don't
; need to search for hidden causes.
; Note: surprise = 0.5 Ã— precision Ã— errorÂ². With typical precision 0.5
; and error 0.6, surprise = 0.09. The threshold must be below typical
; high-surprise values. Using 0.05 catches genuine surprises without
; firing on noise.
(config abductive-surprise-threshold 0.05)

; Precision assigned to abductive hypotheses.
; LOW by design â€” this is a guess, not a measurement.
; With belief_prec=0.5 and hyp_prec=0.10, the mixing weight is
; 0.10/(0.50+0.10) = 0.167, times learning rate 0.12 = ~2% influence.
; It takes consistent evidence over several cycles to move a belief.
(config abductive-precision 0.10)

; Minimum passive-model weight to consider for abduction.
; Weak causal links (weight < 0.15) are too uncertain to justify
; hypothesizing their causes.
(config abductive-min-link-weight 0.15)

; Minimum metabolic energy for a passive model to be used in abduction.
; Models that are dying (low energy) shouldn't drive hypotheses.
; Uses same threshold as deduction for consistency.
(config abductive-min-energy 0.5)

; Maximum precision of existing cause belief for abduction to fire.
; If we're already confident about the cause (prec > 0.5), abduction
; adds nothing â€” we already know. Save the computation.
(config abductive-max-cause-precision 0.50)

; Maximum number of hypotheses injected per cycle.
; Prevents runaway abduction during high-surprise periods.
; Analogous to structure-generation-cap for structure learning.
(config abductive-budget-per-cycle 5)

; Per-cycle budget tracker
(abductive-hypotheses-this-cycle 0)


; =============================================================================
; SECTION II: BUDGET MANAGEMENT
; =============================================================================

(= (reset-abductive-budget!)
   (let $old (match &self (abductive-hypotheses-this-cycle $n) $n)
     (sequential
       (remove-atom &self (abductive-hypotheses-this-cycle $old))
       (add-atom &self (abductive-hypotheses-this-cycle 0))
       done)))

(= (abductive-budget-available?)
   (let $used (match &self (abductive-hypotheses-this-cycle $n) $n)
     (< $used (get-config abductive-budget-per-cycle))))

(= (use-abductive-budget!)
   (let $old (match &self (abductive-hypotheses-this-cycle $n) $n)
     (sequential
       (remove-atom &self (abductive-hypotheses-this-cycle $old))
       (add-atom &self (abductive-hypotheses-this-cycle (+ $old 1)))
       done)))


; =============================================================================
; SECTION III: EPISTEMIC SOURCE TRACKING
; =============================================================================
;
; Lightweight markers that distinguish HOW a belief was formed.
; These don't affect EFE computation (beliefs are uniform) but enable:
;   (a) Honest self-report: "I believe X, but it's a hypothesis"
;   (b) Observation supersedes hypothesis: when a direct observation
;       arrives for an abduced belief, we upgrade the source marker
;       and let the normal observation pathway take over.
;
; Schema: (belief-source $obs $origin $cycle)
;   $origin: abduced | observed
;   $cycle:  when this determination was made

; Mark a belief as abductively sourced
(= (mark-abduced! $obs $cycle)
   (sequential
     ; Remove any existing source marker
     (clear-belief-source! $obs)
     (add-atom &state (belief-source $obs abduced $cycle))
     done))

; Mark a belief as observationally sourced
(= (mark-observed! $obs $cycle)
   (sequential
     (clear-belief-source! $obs)
     (add-atom &state (belief-source $obs observed $cycle))
     done))

; Clear any existing source marker for an observable
(= (clear-belief-source! $obs)
   (let $old (collapse (match &state (belief-source $obs $origin $cycle)
                                    (belief-source $obs $origin $cycle)))
     (remove-source-list! $old)))

(= (remove-source-list! ()) done)
(= (remove-source-list! ($a . $rest))
   (sequential
     (remove-atom &self $a)
     (remove-source-list! $rest)))

; Query: is this belief abductively sourced?
(= (is-abduced? $obs)
   (let $r (collapse (match &state (belief-source $obs abduced $c) True))
     (if (== $r ()) False True)))

; Query: is this belief observationally sourced?
(= (is-observed? $obs)
   (let $r (collapse (match &state (belief-source $obs observed $c) True))
     (if (== $r ()) False True)))

; Query: get epistemic source for an observable
(= (get-belief-source $obs)
   (let $r (collapse (match &state (belief-source $obs $origin $cycle)
                                  (source $origin $cycle)))
     (if (== $r ()) (source unknown 0) (list-head $r))))


; =============================================================================
; SECTION IV: HYPOTHESIS VALUE COMPUTATION
; =============================================================================
;
; The forward model says: when cause is at level C with weight W,
; effect changes by approximately C * W (for excitatory links).
;
; Inverting: if we observe effect at level E, and the link weight is W,
; the cause level that would produce this effect is approximately E / W.
;
; For inhibitory links, high effect implies LOW cause:
;   excitatory: high effect â†’ hypothesize cause HIGH
;   inhibitory: high effect â†’ hypothesize cause LOW
;
; We also incorporate the sign of the error (not just the observation value).
; Positive error (obs > belief) on the effect means the effect is
; HIGHER than expected. This is what needs explaining.

; Compute the hypothesized value for a cause given an observed effect
; $obs-val:  observed value of the effect
; $bval:     current belief value of the effect (what we expected)
; $weight:   passive-model link weight (0 to 1)
; $type:     excitatory or inhibitory
(= (compute-hypothesis-value $obs-val $bval $weight $type)
   (let* (
     ; The surprise direction: is the effect higher or lower than expected?
     ($error (- $obs-val $bval))

     ; For excitatory links: positive effect error â†’ cause is HIGH
     ; For inhibitory links: positive effect error â†’ cause is LOW
     ($direction (if (== $type excitatory) 1.0 -1.0))

     ; Scale the hypothesis by the evidence strength
     ; Use the observation magnitude divided by weight, clamped
     ; This inverts the forward model: if weight=0.8 and obs=0.9,
     ; the cause that would produce this is approximately 0.9/0.8 â‰ˆ 1.0
     ($raw-hyp (if (> $weight 0.01)
                   (/ $obs-val $weight)
                   1.0))  ; Avoid division by near-zero

     ; Apply direction: excitatory â†’ high, inhibitory â†’ complement
     ($directed-hyp (if (> $direction 0.0)
                        $raw-hyp
                        (- 1.0 $raw-hyp)))
   )
   ; Clamp to valid range
   (clamp $directed-hyp 0.0 1.0)))


; =============================================================================
; SECTION V: ABDUCTIVE CANDIDATE COLLECTION
; =============================================================================
;
; For each surprising observation, find passive models where the
; observation is the EFFECT, and the CAUSE has a belief we could
; update. Filter by link quality, energy, and cause uncertainty.

; Collect all abductive candidates from current error traces.
; Returns list of (abductive-candidate ...) tuples.
(= (collect-abductive-candidates $current-cycle)
   (let $threshold (get-config abductive-surprise-threshold)
     (collapse
       (match &self (error-trace $effect $error $surprise $time)
         (if (and (> $surprise $threshold)
                  (== $time $current-cycle))  ; Only current cycle's traces
             ; Find passive models where this observable is the effect
             (match &self (passive-model $cause $effect $lag $weight $type)
               (let* (
                 ($min-weight (get-config abductive-min-link-weight))
                 ($min-energy (get-config abductive-min-energy))
                 ($max-cause-prec (get-config abductive-max-cause-precision))
                 ($energy (get-pm-energy-safe $cause $effect))
                 ($cause-prec (get-belief-precision-safe $cause))
               )
               (if (and (>= $weight $min-weight)
                        (>= $energy $min-energy)
                        (< $cause-prec $max-cause-prec)
                        (has-belief? $cause))
                   (abductive-candidate $cause $effect $error $surprise
                                        $weight $type $energy)
                   (empty))))
             (empty))))))

; Safe accessor for passive model energy (returns 0.0 if not found)
(= (get-pm-energy-safe $cause $effect)
   (let $r (match &self (metabolic-energy passive-model $cause $effect $e) $e)
     (if (== $r ()) 0.0 $r)))

; Safe accessor for belief precision (returns 1.0 if not found, blocking abduction)
(= (get-belief-precision-safe $obs)
   (let $r (match &state (belief $obs $v $p) $p)
     (if (== $r ()) 1.0 $r)))


; =============================================================================
; SECTION VI: HYPOTHESIS INJECTION
; =============================================================================
;
; The core operation: merge a low-precision hypothesis into an existing
; belief using precision-weighted averaging.
;
; This uses the SAME mathematical form as belief updates from observations:
;   new_val = old_val + lr * (hyp_prec / (belief_prec + hyp_prec)) * (hyp_val - old_val)
;
; With hyp_prec=0.10 and belief_prec=0.50:
;   mixing_weight = 0.10 / 0.60 = 0.167
;   effective_influence = 0.12 * 0.167 â‰ˆ 0.02 (2% per cycle)
;
; This is a whisper, not a shout. Consistent abduction over 5+ cycles
; moves the belief meaningfully. A single abductive inference barely
; registers. This is the right behavior: hypotheses should be tentative.

(= (inject-hypothesis! $cause $hyp-val $hyp-prec $effect $cycle)
   (if (not (abductive-budget-available?))
       abductive-budget-exhausted
       (match &state (belief $cause $bval $bprec)
         (let* (
           ; Precision-weighted merge (same form as observation update)
           ($lr (get-config learning-rate))
           ($prec-sum (+ $bprec $hyp-prec))
           ($hyp-weight (/ $hyp-prec $prec-sum))
           ($error (- $hyp-val $bval))
           ($update (* $lr (* $hyp-weight $error)))
           ($new-val (clamp (+ $bval $update) 0.0 1.0))

           ; Precision: slight DECREASE (abduction adds uncertainty,
           ; not confidence). The hypothesis says "something is going on"
           ; but doesn't pin down the value precisely.
           ($prec-nudge -0.02)
           ($new-prec (clamp (+ $bprec $prec-nudge)
                             (get-config precision-floor)
                             (get-config precision-ceiling)))
         )
         (sequential
           ; Update the belief
           (remove-atom &state (belief $cause $bval $bprec))
           (add-atom &state (belief $cause $new-val $new-prec))

           ; Mark epistemic source (only if not already observed)
           (if (not (is-observed? $cause))
               (mark-abduced! $cause $cycle)
               done)

           ; Log the abductive inference (for transparency/diagnostics)
           (add-atom &self (abduction-log $cause $effect $hyp-val $hyp-prec $cycle))

           ; Budget accounting
           (use-abductive-budget!)

           ; Return trace
           (abduced $cause
             (from-effect $effect)
             (hypothesis $hyp-val)
             (prior $bval)
             (posterior $new-val)
             (precision $bprec $new-prec)))))))


; =============================================================================
; SECTION VII: PROCESS CANDIDATES
; =============================================================================
;
; Process all abductive candidates for the current cycle.
; For each candidate: compute hypothesis value, inject into beliefs.

(= (process-abductive-candidate!
     (abductive-candidate $cause $effect $error $surprise
                          $weight $type $energy) $cycle)
   (let* (
     ; Get current observation and belief for the effect
     ($obs-val (get-obs-value-safe $effect))
     ($bval (get-belief-value $effect))
     ($hyp-prec (get-config abductive-precision))

     ; Compute what the cause should be, given this effect
     ($hyp-val (compute-hypothesis-value $obs-val $bval $weight $type))
   )
   (inject-hypothesis! $cause $hyp-val $hyp-prec $effect $cycle)))

; Safe accessor for observation value (returns 0.5 if not found)
(= (get-obs-value-safe $obs)
   (let $r (match &state (observation $obs $val $prec $time) $val)
     (if (== $r ()) 0.5 $r)))

; Process a list of candidates
(= (process-abductive-candidates! () $cycle) done)
(= (process-abductive-candidates! ($candidate . $rest) $cycle)
   (sequential
     (process-abductive-candidate! $candidate $cycle)
     (process-abductive-candidates! $rest $cycle)))


; =============================================================================
; SECTION VIII: OBSERVATION SUPERSESSION
; =============================================================================
;
; When a direct observation arrives for an observable that was previously
; hypothesized via abduction, the observation SUPERSEDES the hypothesis.
; The belief-source marker upgrades from 'abduced' to 'observed', and
; the normal observation update pathway handles the rest.
;
; This is called from the observation injection pathway (cycle.metta)
; whenever a new observation arrives.

(= (check-observation-supersedes! $obs $cycle)
   (if (is-abduced? $obs)
       (sequential
         (mark-observed! $obs $cycle)
         (superseded $obs abduced observed $cycle))
       done))


; =============================================================================
; SECTION IX: MASTER ABDUCTIVE STEP
; =============================================================================
;
; Called once per cognitive cycle from cycle.metta.
; Placement: AFTER error traces and suspicion updates,
;            BEFORE action selection.
;
; This ensures:
;   - Structure learning sees "honest" errors (before belief adjustment)
;   - Action selection sees enriched beliefs (after hypothesis injection)

(= (abductive-step! $current-cycle)
   (sequential
     ; Reset per-cycle budget
     (reset-abductive-budget!)

     ; BF: Check metabolic capital before attempting any abduction.
     ; Creative abduction (hypothesizing hidden causes from effects)
     ; requires earned capital. This naturally sequences development:
     ; the agent must first master induction (earning capital through
     ; successful predictions) before it can afford to philosophize.
     (if (not (can-afford-abduction?))
         (abduction-deferred insufficient-capital $current-cycle)
         ; Collect candidates: surprising effects with known causes
         (let $candidates (collect-abductive-candidates $current-cycle)
           (if (list-empty? $candidates)
               (abduction-idle $current-cycle)
               ; Process all candidates (capital spent per hypothesis)
               (sequential
                 (spend-abduction-capital!)    ; BF: pay for creative abduction
                 (process-abductive-candidates! $candidates $current-cycle)
                 (abduction-step-done $current-cycle (length $candidates))))))))


; =============================================================================
; SECTION X: DIAGNOSTICS
; =============================================================================

; Summary of abductive activity
(= (abduction-summary)
   (let* (
     ($logs (collapse (match &self (abduction-log $c $e $hv $hp $cy)
       (abduction-entry $c $e $hv $hp $cy))))
     ($abduced-beliefs (collapse (match &state (belief-source $obs abduced $cy) $obs)))
     ($n-logs (length $logs))
     ($n-abduced (length $abduced-beliefs))
   )
   (abduction-state
     (total-abductions $n-logs)
     (currently-abduced $n-abduced)
     (abduced-observables $abduced-beliefs))))

; Recent abductions (current cycle only)
(= (recent-abductions $cycle)
   (collapse (match &self (abduction-log $cause $effect $hyp $prec $cy)
     (if (== $cy $cycle)
         (recent-abduction $cause (from $effect) (hypothesized $hyp))
         (empty)))))

; Is any belief currently abductively sourced?
(= (has-abductive-hypotheses?)
   (let $r (collapse (match &state (belief-source $obs abduced $c) True))
     (if (== $r ()) False True)))

; Count of currently abduced beliefs
(= (abduced-belief-count)
   (length (collapse (match &state (belief-source $obs abduced $c) $obs))))

; Clean old abduction logs (keep last N cycles worth)
(= (clean-abduction-logs! $current-cycle)
   (let* (
     ($window (* 3 (get-config lookback-window)))  ; 3x lookback
     ($cutoff (- $current-cycle $window))
     ($old (collapse (match &self (abduction-log $c $e $hv $hp $cy)
       (if (< $cy $cutoff)
           (abduction-log $c $e $hv $hp $cy)
           (empty)))))
   )
   (remove-old-abduction-logs! $old)))

(= (remove-old-abduction-logs! ()) done)
(= (remove-old-abduction-logs! ($log . $rest))
   (sequential
     (remove-atom &self $log)
     (remove-old-abduction-logs! $rest)))

; Detailed report: for each abduced belief, show evidence chain
(= (abduction-report)
   (collapse
     (match &state (belief-source $obs abduced $cycle)
       (let* (
         ($bval (get-belief-value $obs))
         ($bprec (get-belief-precision $obs))
         ; Find the most recent abduction log for this observable
         ($logs (collapse (match &self (abduction-log $obs $effect $hv $hp $cy)
           (log-entry $effect $hv $cy))))
       )
       (abduction-detail $obs
         (current-belief $bval $bprec)
         (evidence $logs))))))


; =============================================================================
; END OF ABDUCTIVE INFERENCE (Phase 1.6, v0.2.0 — boundary condition fix)
; =============================================================================
