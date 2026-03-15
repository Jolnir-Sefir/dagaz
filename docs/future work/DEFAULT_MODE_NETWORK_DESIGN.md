# Default Mode Network & Meditation — Offline Consolidation, Deep Abstraction, and Stillness

## The Problem (The Reactive Trap)

Project Dagaz successfully unifies perception, action, and learning under a single invariant: minimize Expected Free Energy (EFE). However, the current architecture is fundamentally an "always-on" reactive engine. If the environment is completely safe (viability bounds are secure) and perfectly predictable (zero external prediction error), the EFE equation naturally selects `wait` as the optimal action. The system sits idle, dissipating metabolic energy, effectively staring at a blank wall until the world surprises it again.

A purely reactive agent is a prey animal. It is bound by strict real-time latency constraints, meaning it can only perform shallow structural generation (Phase 1 induction, depth-limited deduction). It lacks the computational budget for deep operations such as scanning the entire `&ontology` space to discover conceptual analogies or complex concept blends (Phase 4).

True domain-general intelligence requires a **Default Mode Network (DMN)**: an offline, generative, internally-directed state where the system hallucinates futures, consolidates memories, and searches for deep structural symmetries without the pressure of immediate environmental survival.

But daydreaming is not the only thing missing. A purely reactive agent also cannot *be still*. Consider an embodied Dagaz working as a therapist, sitting across from an agitated patient. Daydreaming (as we will design it) is computationally furious: hashing, blending, compressing. A therapist must be able to sit in absolute stillness — pure presence without cognitive thrashing. In human neuroscience, the DMN is highly active during mind-wandering but is actually **deactivated** during deep meditation. These are distinct computational states, and the architecture needs both.

## The Physicist's Question

**What is the invariant?**

**Inward-Directed EFE Minimization.** We do not write a scripted rule that says `if safe and bored, then daydream`. The invariant remains EFE minimization. We introduce a new source of internal "surprise": *cognitive fatigue*, derived from the accumulation of unintegrated structure. When the external world yields no Expected Information Gain, but internal cognitive fatigue is high, the epistemic value of looking inward surpasses the value of looking outward. The system natively "closes its eyes."

**What are the symmetries?**

**Internal compression is symmetric to external prediction.** In the external world, concepts earn metabolic energy by correctly predicting the environment (Epiphany Bonus). In the internal world (the DMN), concepts earn metabolic gestation energy by successfully compressing the knowledge graph. A unified structural blend that replaces two redundant sub-graphs reduces the system's total rent overhead. Mathematical parsimony is rewarded through the same metabolic mechanism as predictive accuracy — but dream-minted concepts must still survive empirical selection in the waking world.

**What are the true degrees of freedom?**

The allocation of the cognitive cycle's compute budget across three poles. The system must continuously balance real-time sensory latency (staying awake), deep abstraction search (daydreaming), and active stillness (meditating), all bounded by viability limits. Two of these poles reduce free energy by *doing* (acting on the world, or restructuring the model). The third reduces free energy by *releasing* — temporarily attenuating the preferences that generate the free energy in the first place.

## Architecture

### New / Modified Files

| File | Addition | Purpose |
|------|----------|---------|
| `dmn.metta` (NEW) | ~300 lines | The offline cognitive loop. LSH-directed analogy search, unconstrained fractal planning, hallucinated rollouts. Interruptible chunked execution. Rumination detection and breakthrough candidacy. Dream journaling to `&episodic`. |
| `meditation.metta` (NEW) | ~120 lines | The Zen state. Preference attenuation, accelerated error-trace decay, sensory precision boost. Epistemic exit mechanism. |
| `beliefs.metta` | ~25 lines | Add `cognitive-fatigue` as a derived self-observable with cognitive dissonance weighting. |
| `actions.metta` | ~15 lines | Add `daydream` and `meditate` action models with learnable parameters. |
| `cycle.metta` | ~40 lines | Route execution to DMN or meditation. Startle reflex. Meditation state flag. |
| `atom_lifecycle.metta` | ~50 lines | Gestation Energy for offline structural compression. Hook `on-blend-death-untested!` into existing pruning pathway. |
| `safety.metta` | ~10 lines | Register `startle-threshold` as Constitutional (Stratum 1). Register viability preferences as attenuation-exempt. |

### Untouched Files

All reasoning, EFE, and grounding modules remain untouched. Both the DMN and meditation are actions that compete in the existing EFE landscape, not mode switches.

---

## The Eight Mechanisms

### 1. Cognitive Fatigue as Derived View (The Drive to Dream)

**Design principle:** Affect is derived, never stored (see `affect.metta`). Cognitive fatigue follows the same pattern. It is not a tracked belief that something must manually increment — it is a *query over existing metabolic state*, computed on demand.

Fatigue is high when the system contains many recently-created atoms that have not yet been deeply integrated (low lifecycle phase, low energy, short time-since-creation). This is information the system already tracks.

```metta
; --- In beliefs.metta ---

; =============================================================================
; COGNITIVE FATIGUE: DERIVED VIEW (never stored)
; =============================================================================
;
; Cognitive fatigue measures how much raw, unassimilated structure exists.
; It is computed from existing atom lifecycle state, not maintained separately.
; This mirrors how affect.metta derives valence from prediction errors.

; Count atoms in early lifecycle phases with low energy
; These are "raw knowledge" — created but not yet deeply integrated
(= (count-unintegrated-atoms)
   (let* (
     ; Phase 1 passive-models still in gestation or near-zero energy
     ($raw-pm (collapse (match &self (lifecycle-energy passive-model $c $e $energy)
                (if (< $energy 0.3) 1 0))))
     ; Phase 2 latent variables not yet promoted
     ($raw-lv (collapse (match &self (lifecycle-energy latent-var $name $energy)
                (if (< $energy 0.5) 1 0))))
     ; Phase 4 concept blends still in gestation
     ($raw-blend (collapse (match &self (lifecycle-energy concept-blend $name $energy)
                (if (< $energy 0.3) 1 0))))
     ($total (+ (sum-list $raw-pm) (+ (sum-list $raw-lv) (sum-list $raw-blend))))
   )
   $total))

; Compute cognitive dissonance: sum of chain debit accumulated by active atoms
; A D=4 concept actively bleeding energy because its descendants are failing
; represents *cognitive dissonance* — unresolved conflict in the knowledge graph.
; Resolving dissonance (by blending, deduplicating, or pruning failing structure)
; should be the DMN's highest priority, so it contributes more fatigue pressure
; than raw unintegrated atoms sitting harmlessly in gestation.
(= (compute-cognitive-dissonance)
   (let* (
     ; Find post-gestation atoms with net negative chain flow
     ; These are actively failing — their structure is wrong, not just untested
     ($debit-atoms (collapse (match &self (chain-debit-accumulated $type $name $debit)
                    (if (> $debit 0.0) $debit 0.0))))
   )
   (sum-list $debit-atoms)))

; Derive cognitive fatigue as a normalized signal [0, 1]
; Combines raw unintegrated count with cognitive dissonance weighting.
; Dissonance (active failure) contributes more urgency than mere novelty.
(= (compute-cognitive-fatigue)
   (let* (
     ($raw-count (count-unintegrated-atoms))
     ($dissonance (compute-cognitive-dissonance))
     ($dissonance-weight (get-config fatigue-dissonance-weight))  ; e.g., 5.0
     ($effective-load (+ (float $raw-count) (* $dissonance-weight $dissonance)))
     ($half-saturation (get-config fatigue-half-saturation))  ; e.g., 10
   )
   ; Hill function: load / (load + K) — saturates smoothly at 1.0
   (/ $effective-load (+ $effective-load $half-saturation))))

; Expose as a self-observable for EFE computation
; The system can "observe" its own fatigue, just like it observes threat-level
(= (get-self-observable cognitive-fatigue)
   (compute-cognitive-fatigue))

; Preference: The system desires low cognitive fatigue (compressed, integrated mind)
(preference cognitive-fatigue 0.0 0.6)
```

**Why derived, not stored:** A stored scalar conflates quantity and quality — ten shallow Phase 1 correlations and one deep Phase 3 latent both increment the same counter, but demand different DMN operations. A query over existing state captures the actual structure of the problem. It also cannot drift from reality, cannot be set arbitrarily, and requires no manual increment logic.

**Why dissonance weighting:** A raw Phase 1 atom sitting harmlessly in gestation shouldn't cause much fatigue — it's merely untested. But a D=4 concept actively accumulating chain debit (because its descendants are failing) represents *cognitive dissonance*: unresolved conflict in the knowledge graph. The dissonance weight (`fatigue-dissonance-weight = 5.0`) ensures the DMN prioritizes resolving active failures over merely integrating new structure. This maps to the biological observation that unresolved cognitive conflict is more fatiguing than mere novelty, and mirrors the Fristonian concept of allostatic load as accumulated unresolved prediction error.

**Integration with affect:** Cognitive fatigue feeds into the affect system naturally. High fatigue with low external arousal creates a specific affective signature — the system "feels" the cognitive pressure to consolidate, the way high prediction error creates the "urge" to investigate.


### 2. The Daydream Action (Learnable EFE Selection)

We expose the DMN to the EFE landscape as an action with **learnable** parameters. The initial action model is a weak prior; the system discovers how effective daydreaming actually is through the same prediction-error-driven learning that governs all other action models (see `action_learning.metta`).

```metta
; --- In actions.metta ---

; Action: daydream
; Initial weak prior: daydreaming somewhat reduces cognitive fatigue.
; These parameters will be LEARNED from experience via action_learning.
(action-model daydream cognitive-fatigue -0.2 0.15 0.5)

; Daydreaming is physically cheap, but occupies the cognitive loop.
(action-cost daydream 0.01)

; ZERO effect on external physical observables.
; (No action-model entries for terrain, threat-level, etc.)
; This is structural: the daydream action cannot change the world.
```

**Why learnable, not hardcoded:** The effect size (-0.2) and confidence (0.5) are a weak initial prior, not a permanent assertion. After each daydream, the action learning machinery compares the predicted fatigue reduction to the actual change in `compute-cognitive-fatigue`. A fruitless session (no blends found, no compression achieved) produces a prediction error that *lowers* the model's confidence and adjusts the effect size toward reality. Over time, the system learns how productive its own dreaming is.

**How EFE emergence works:** If `threat-level` is high, the viability penalty swamps the EFE equation; `daydream` is universally rejected in favor of `retreat` or `observe`. But if the system is safe, and `terrain` is known, external info-gain approaches zero. Meanwhile, the preference gap on `cognitive-fatigue` grows. EFE math seamlessly selects `daydream` as the optimal path to minimize total system free energy.

**Self-correcting dynamics:** If the system daydreams repeatedly with poor results (e.g., the ontology is too sparse for useful analogies), the learned confidence drops, the predicted effect size shrinks, and EFE stops selecting `daydream` until more raw structure accumulates. The system naturally calibrates its own sleep schedule.


### 3. The Startle Reflex (Interruptible Cycle Integration)

When `daydream` is selected, the system disconnects from the physical actuator and runs the DMN. However, the agent must not be eaten while dreaming. It must maintain a lightweight sensory tripwire.

**Critical design constraint:** The DMN must be *interruptible*, not just preceded by a single sensory check. Deep analogy search or hallucinated rollouts could exceed the sensory buffer polling interval. The DMN is therefore structured as a sequence of **chunks**, with a sensory check between each chunk.

```metta
; --- In cycle.metta ---

; Route action execution: physical actions vs. daydream
(= (execute-action $action $timestamp)
   (if (== $action daydream)
       (execute-dmn-cycle! $timestamp)
       (execute-physical-action! $action)))

; The DMN cycle is a sequence of interruptible chunks
(= (execute-dmn-cycle! $timestamp)
   (let* (
     ; Select how many chunks to attempt (bounded by time budget)
     ($max-chunks (get-config dmn-chunks-per-cycle))  ; e.g., 3
   )
   (execute-dmn-chunks! $max-chunks $timestamp)))

; Recursive chunked execution with startle check between each chunk
(= (execute-dmn-chunks! 0 $timestamp) (dmn-cycle-complete $timestamp))
(= (execute-dmn-chunks! $remaining $timestamp)
   (let* (
     ; STARTLE CHECK: Peek at the raw sensory buffer before each chunk
     ($external-surprise (compute-raw-sensory-surprise))
     ; Threshold is Constitutional (Stratum 1) — defined in safety.metta, immutable
     ($threshold (get-safety-axiom startle-threshold))
   )
   (if (> $external-surprise $threshold)
       ; === STARTLE REFLEX: Abort daydream, return to waking cycle ===
       ; Abrupt waking does NOT reduce fatigue — the work was interrupted
       (sequential
         (log-event! dmn-abort $timestamp $external-surprise)
         (aborted-daydream $timestamp))
       
       ; === SAFE TO DREAM: Execute one chunk, then recurse ===
       (sequential
         (run-dmn-chunk!)
         (execute-dmn-chunks! (- $remaining 1) $timestamp)))))

; After a complete (non-interrupted) DMN cycle, fatigue reduction
; happens automatically: if blends were created and compression occurred,
; count-unintegrated-atoms returns a lower value next cycle.
; No manual decrement needed — the derived view updates itself.
```

**Why no manual fatigue decrement:** In the original design, `execute-dmn-cycle!` explicitly decremented `unintegrated-structure` by 0.3. With the derived view, this is unnecessary and would be architecturally wrong. If the DMN successfully creates a concept blend that absorbs several raw atoms, those atoms' lifecycle energy increases (they're now part of an integrated structure), and `count-unintegrated-atoms` naturally returns a lower count. If the DMN finds nothing useful, the count doesn't change, and the next cycle's EFE computation correctly reflects that nothing was accomplished. The feedback loop is closed through existing state, not through manual bookkeeping.


### 4. LSH-Directed Deep Abstraction (The Latency Budget)

The real constraint on the waking cycle isn't computational complexity — it's *latency*. The waking cycle must respond to the environment within a sensory deadline. The DMN relaxes this constraint, granting a *latency budget* rather than unlimited compute.

Because the system has explicitly suspended the real-time physical loop, it can afford to run operations that would miss the waking deadline. `dmn.metta` structures these as interruptible chunks, each bounded by a time ceiling.

**Critical refinement: LSH-directed search, not random pairing.** The existing Surprise-Band LSH scheme (see `LSH_HEBBIAN_DESIGN.md`) already groups observables by temporal co-occurrence and surprise magnitude. Concepts that hash to the same LSH bucket are structurally similar *by construction* — they fire together at similar magnitudes. The DMN should preferentially sample within LSH buckets rather than uniformly across the entire ontology. This reuses existing infrastructure and makes deep analogy search dramatically more efficient.

```metta
; --- In dmn.metta ---

; =============================================================================
; DEFAULT MODE NETWORK — OFFLINE COGNITIVE LOOP
; =============================================================================
;
; The DMN runs when EFE selects `daydream`. It performs deep operations
; that the waking cycle cannot afford due to latency constraints.
;
; DESIGN PRINCIPLES:
;   - Interruptible: Each task is a bounded chunk with startle checks between
;   - LSH-directed: Analogy search uses existing LSH buckets, not random pairing
;   - Metabolically integrated: Dream outputs enter the same lifecycle as all atoms
;   - Reasoning-triad reentry: Dream-minted concepts are eligible for deduction,
;     abduction, and further induction — not a separate pipeline
;
; Dependencies: foundations, beliefs, structure_learning, analogy_blending,
;               atom_lifecycle, planning
; Depended on by: cycle
;
; =============================================================================

!(import! &self foundations)
!(import! &self analogy_blending)
!(import! &self atom_lifecycle)
!(import! &self planning)

; --- Configuration ---
(config dmn-chunks-per-cycle 3)        ; Max chunks per daydream action
(config dmn-chunk-time-budget-ms 500)  ; Max ms per chunk (soft ceiling)
(config blend-similarity-threshold 0.4); Min structural similarity for blending
(config fatigue-half-saturation 10.0)  ; Hill function K for fatigue normalization
(config gestation-energy-scaling 0.3)  ; Scaling for compression-based gestation energy
; NOTE: startle-threshold is Constitutional (Stratum 1), defined in safety.metta

; =============================================================================
; SECTION I: CHUNK DISPATCHER
; =============================================================================
;
; Each DMN chunk performs one bounded task. Task selection is weighted
; by the current cognitive state — what kind of unintegrated structure
; exists determines what kind of consolidation is most valuable.

(= (run-dmn-chunk!)
   (let $task (select-dmn-task)
     (perform-dmn-task $task)))

; Task selection: weighted by what the system actually needs
(= (select-dmn-task)
   (let* (
     ; How many raw passive-models exist? → analogy search is useful
     ($raw-pm-count (count-raw-passive-models))
     ; How many latent variables exist without blends? → blending is useful
     ($raw-lv-count (count-raw-latent-vars))
     ; How long since last deep planning? → hallucinated rollouts are useful
     ($planning-debt (cycles-since-last-deep-plan))
   )
   (cond
     ((> $raw-lv-count 3) deep-analogy-search)
     ((> $planning-debt 10) hallucinated-rollout)
     ((> $raw-pm-count 5) deductive-consolidation)
     (True deep-analogy-search))))  ; default


; =============================================================================
; SECTION II: DEEP ANALOGY SEARCH (LSH-Directed)
; =============================================================================
;
; Instead of randomly pairing concepts from the entire ontology, we
; exploit the LSH bucket structure from the Hebbian accumulator.
; Concepts in the same bucket share temporal co-occurrence patterns
; and surprise magnitude — they are structurally similar by construction.
;
; This reduces the search from O(N²) random pairs to O(B × b²) where
; B is the number of buckets and b is average bucket size (b << N).

(= (perform-dmn-task deep-analogy-search)
   (let* (
     ; Get all LSH buckets that contain more than one concept
     ($buckets (get-populated-lsh-buckets))
     ; Filter out buckets exhausted this cycle (stateless routing)
     ($fresh-buckets (filter-unexphausted $buckets))
     ; Select the most populated remaining bucket
     ($target-bucket (select-richest-bucket $fresh-buckets))
   )
   (if (== $target-bucket None)
       ; All buckets exhausted this cycle — nothing left to search
       (dmn-search-exhausted)
       (let $candidates (get-intra-bucket-pairs $target-bucket)
         (sequential
           (process-analogy-candidates $candidates)
           ; Mark bucket as searched — next chunk skips it
           ; This is stateless routing: the atom persists in &self,
           ; keeping execution memoryless while the space handles state
           (add-atom &self (search-exhausted (bucket-id $target-bucket)
                                             (current-cycle))))))))

; Filter out buckets that have already been searched this cycle
(= (filter-unexphausted $buckets)
   (let $cycle (current-cycle)
     (filter-list $buckets
       (lambda $b (not (match &self (search-exhausted (bucket-id $b) $cycle) True))))))

; Process candidate pairs: compute structural signatures, check similarity
(= (process-analogy-candidates ())
   (no-analogies-found))
(= (process-analogy-candidates (($c1 $c2) . $rest))
   (let* (
     ($sig1 (compute-structural-signature $c1))
     ($sig2 (compute-structural-signature $c2))
     ($sim (compute-similarity $sig1 $sig2))
   )
   (if (> $sim (get-config blend-similarity-threshold))
       (sequential
         (trigger-offline-concept-blend $c1 $c2)
         (process-analogy-candidates $rest))
       (sequential
         (record-comparison-attempt $c1 $c2)
         (process-analogy-candidates $rest)))))


; =============================================================================
; SECTION III: HALLUCINATED ROLLOUTS (Deep Fractal Planning)
; =============================================================================
;
; Runs the RG-Flow planner to greater depth than the waking cycle allows,
; using hypothetical states to discover new chunked policies (Phase 4)
; without physical risk.
;
; Key: hallucinated rollouts use the EXISTING planning infrastructure
; (planning.metta) with relaxed depth limits, not a separate planner.

(= (perform-dmn-task hallucinated-rollout)
   (let* (
     ; Use current beliefs as starting state
     ($state (get-current-belief-state))
     ; Run fractal planner with extended depth (waking limit × 3)
     ($waking-depth (get-config max-planning-depth))
     ($dream-depth (* $waking-depth 3))
     ($plan (rg-flow-plan $state $dream-depth))
   )
   (sequential
     ; If the plan reveals a repeated action sequence, candidate for chunking
     (check-for-chunking-candidates $plan)
     ; Record the hallucinated trajectory for future reference
     (record-hallucinated-plan $plan))))


; =============================================================================
; SECTION IV: DEDUCTIVE CONSOLIDATION
; =============================================================================
;
; During waking cycles, deductive closure is depth-limited to maintain
; latency. The DMN can run deeper transitive closure over the full
; causal graph, discovering longer inferential chains.

(= (perform-dmn-task deductive-consolidation)
   (let* (
     ; Get all passive-models that haven't been checked for transitive closure
     ($unchecked (get-unconsolidated-passive-models))
     ; Run deeper closure than the waking cycle permits
     ($new-links (deep-transitive-closure $unchecked))
   )
   ; New deduced links enter the standard metabolic lifecycle
   (register-deduced-links! $new-links)))


; =============================================================================
; SECTION V: RUMINATION DETECTION & BREAKTHROUGH CANDIDACY
; =============================================================================
;
; THE PROBLEM:
;   A concept blend may compress the graph beautifully but concern
;   observables the agent never encounters. It receives gestation energy,
;   survives for a while, then dies of metabolic starvation — untested,
;   not disproven. The next time the DMN runs, the same structural
;   similarity still exists, so the same blend is re-minted. And dies
;   again. This is a creation-death cycle: the system is *ruminating*.
;
; THE INSIGHT:
;   Rumination is not pathological — it is signal. If the system keeps
;   returning to the same structural pairing, it means the compression
;   is real and persistent. The concept isn't failing empirically; it
;   simply hasn't had the opportunity to succeed. This is analogous to
;   a theoretical physicist who keeps returning to the same mathematical
;   structure because it's "too beautiful to be wrong" — the structure
;   compresses the formalism even though no experiment has tested it yet.
;
; THE MECHANISM:
;   Track blend attempts per concept pair. When a pair has been blended
;   and died multiple times (exceeding the rumination threshold), escalate
;   to a "breakthrough candidate" — a special metabolic class with:
;     1. Extended protection from rent (longer than normal gestation)
;     2. A flag that makes the waking cycle actively seek test conditions
;     3. Storage in &episodic as a persistent record of the insight
;
;   Breakthrough candidates are the system's "open conjectures" — ideas
;   it believes are structurally sound but cannot yet validate. They
;   represent the frontier of what the system can imagine but not test.
;
; ATOM SCHEMAS:
;   (blend-attempt-count $c1 $c2 $count)
;   (breakthrough-candidate $blend-name $c1 $c2 $compression $created-cycle)
;   (blend-died $c1 $c2 $cycle)

; Check if a concept pair has been blended before
(= (get-blend-attempt-count $c1 $c2)
   (let $r (match &self (blend-attempt-count $c1 $c2 $n) $n)
     (if (== $r ()) 0 $r)))

; Increment the blend attempt counter for a pair
(= (increment-blend-attempts! $c1 $c2)
   (let $old (get-blend-attempt-count $c1 $c2)
     (sequential
       (if (> $old 0)
           (remove-atom &self (blend-attempt-count $c1 $c2 $old))
           (nop))
       (add-atom &self (blend-attempt-count $c1 $c2 (+ $old 1)))
       (+ $old 1))))

; Called when a dream-minted blend dies of metabolic starvation (untested)
; This hooks into the existing pruning pathway in atom_lifecycle.metta
(= (on-blend-death-untested! $blend-name $c1 $c2)
   (sequential
     (add-atom &self (blend-died $c1 $c2 (current-cycle)))
     ; The blend-attempt-count persists — next DMN cycle will see it
     done))

; Check whether a pair should be escalated to breakthrough candidate
(= (check-rumination $c1 $c2 $attempt-count $compression-ratio)
   (let $threshold (get-config rumination-threshold)  ; e.g., 3
     (if (>= $attempt-count $threshold)
         (escalate-to-breakthrough! $c1 $c2 $compression-ratio)
         (standard-blend $c1 $c2))))

; Escalate a ruminated concept to breakthrough candidate
(= (escalate-to-breakthrough! $c1 $c2 $compression-ratio)
   (let* (
     ($blend-name (create-concept-blend $c1 $c2))
     ($scaling (get-config gestation-energy-scaling))
     ; Breakthrough candidates receive 3× normal gestation energy
     ($energy (* 3.0 (* $scaling $compression-ratio)))
     ($cycle (current-cycle))
   )
   (sequential
     ; Fund generously — this concept has proven its structural worth
     (set-lifecycle-energy! concept-blend $blend-name $energy)
     ; Mark as breakthrough candidate (distinct from normal blend)
     (add-atom &self (breakthrough-candidate $blend-name $c1 $c2
                       $compression-ratio $cycle))
     ; Record in &episodic for persistence across lifecycle deaths
     ; The breakthrough record survives even if this instance dies
     (add-atom &episodic (breakthrough-insight $blend-name $c1 $c2
                           $compression-ratio $cycle))
     ; Mark origin for traceability
     (add-atom &self (atom-origin $blend-name dmn-breakthrough ($c1 $c2)))
     ; Flag for the waking cycle: actively seek test conditions
     (add-atom &self (seek-test-conditions $blend-name))
     ; Register in reasoning triad as normal
     (register-for-reasoning-triad! concept-blend $blend-name)
     ; Log the escalation
     (log-event! breakthrough-escalation $blend-name
       (ruminated $c1 $c2 $compression-ratio (get-blend-attempt-count $c1 $c2))))))
```

**The rumination mechanism integrates with the waking cycle through `seek-test-conditions`.** When this flag exists for a concept, the EFE computation can factor in that observing certain states would provide information about the breakthrough candidate. This creates a gentle bias toward *seeking out* situations that would test the conjecture — the system doesn't just passively wait for validation, it actively looks for it. This is the computational analog of designing an experiment to test a theory.

**Breakthrough candidates in the episodic space.** The `&episodic` record (see Dream Journaling below) persists even if the current instance of the blend dies. If the blend dies a fourth time, the system sees the episodic record and the high attempt count, and can make an informed decision: re-mint with even more energy, or accept that the conjecture is untestable in the current environment and stop ruminating. The `rumination-ceiling` config parameter bounds this: after N attempts, the pair is marked `(rumination-exhausted $c1 $c2)` and the DMN skips it permanently.

```metta
; Prevent infinite rumination: after enough attempts, accept untestability
(= (check-rumination-ceiling $c1 $c2 $attempt-count)
   (let $ceiling (get-config rumination-ceiling)  ; e.g., 6
     (if (>= $attempt-count $ceiling)
         (sequential
           (add-atom &self (rumination-exhausted $c1 $c2))
           (add-atom &episodic (accepted-untestable $c1 $c2 (current-cycle)))
           (log-event! rumination-exhausted $c1 $c2 $attempt-count))
         (nop))))
```


### 5. Gestation Energy for Offline Compression (Not Permanent Funding)

Under the Epistemic Credit Market, atoms pay rent and earn reward via chain credit when they successfully predict external observations. Concepts minted in a dream haven't predicted anything yet. How do they survive?

We introduce **Gestation Energy** based on structural compression. This is explicitly *not* permanent funding — it is a metabolic runway extension, analogous to the extended gestation period that high-depth atoms already receive (see Credit Market Mechanism 2). Dream-minted concepts must still survive empirical selection in the waking world.

```metta
; --- In atom_lifecycle.metta (Extensions) ---

; =============================================================================
; GESTATION ENERGY FOR OFFLINE COMPRESSION
; =============================================================================
;
; When the DMN creates a concept blend that compresses the knowledge graph,
; the new concept receives gestation energy proportional to the compression
; achieved. This is a RUNWAY, not a permanent reward:
;   - The concept enters the standard metabolic lifecycle
;   - It begins paying rent after its (extended) gestation period
;   - It must earn chain credit from successful predictions to survive
;   - If it proves vacuous, it dies like any other failed hypothesis
;
; The key distinction from the Epiphany Bonus: the Epiphany Bonus rewards
; atoms that resolve chronic prediction errors (external validation).
; Gestation Energy rewards structural parsimony (internal consistency).
; Both are metabolic signals, but they operate on different timescales
; and require different eventual validation.

(= (trigger-offline-concept-blend $c1 $c2)
   (let* (
     ; Check rumination: has this pair been blended and died before?
     ($attempt-count (increment-blend-attempts! $c1 $c2))
     ; Check if pair is permanently exhausted
     ($exhausted (match &self (rumination-exhausted $c1 $c2) True))
   )
   (if (== $exhausted True)
       ; This pair has been tried too many times — skip permanently
       (rumination-exhausted-skip $c1 $c2)
       (let* (
         ($blend-name (create-concept-blend $c1 $c2))
         ; Calculate Parsimony: (edges_before - edges_after) / edges_before
         ($compression-ratio (compute-graph-compression $c1 $c2 $blend-name))
         ; Minimum compression threshold: the blend must actually simplify
         ($min-compression (get-config min-compression-ratio))  ; e.g., 0.15
       )
       (if (> $compression-ratio $min-compression)
           ; Good compression — but is this a rumination escalation?
           (let $rum-threshold (get-config rumination-threshold)  ; e.g., 3
             (if (>= $attempt-count $rum-threshold)
                 ; RUMINATION DETECTED: escalate to breakthrough candidate
                 (sequential
                   (check-rumination-ceiling $c1 $c2 $attempt-count)
                   (escalate-to-breakthrough! $c1 $c2 $compression-ratio))
                 ; Standard blend: normal gestation energy
                 (let* (
                   ; Gestation energy scales with compression achieved
                   ($scaling (get-config gestation-energy-scaling))
                   ($energy (* $scaling $compression-ratio))
                   ($depth (compute-structural-depth concept-blend $blend-name))
                   ($gestation (effective-gestation concept-blend $blend-name))
                 )
                 (sequential
                   ; Fund the new concept with gestation energy (a runway, not income)
                   (set-lifecycle-energy! concept-blend $blend-name $energy)
                   ; Mark origin for traceability
                   (add-atom &self (atom-origin $blend-name dmn-blend ($c1 $c2)))
                   ; Log the dream insight
                   (log-event! dream-insight $blend-name
                     (offline-compression $compression-ratio $gestation))
                   ; === REASONING TRIAD REENTRY ===
                   (register-for-reasoning-triad! concept-blend $blend-name)))))
           
           ; Compression too low — the blend is vacuous, discard it
           (sequential
             (remove-failed-blend! $blend-name)
             (record-failed-blend-attempt $c1 $c2 $compression-ratio)))))))

; Register a dream-minted concept for participation in the reasoning triad
; This is explicit about the integration points, but structurally lightweight:
; it just ensures the atom is in the right form for existing machinery to find.
(= (register-for-reasoning-triad! $type $name)
   (sequential
     ; Ensure the blend has a belief entry (so EFE can reference it)
     (ensure-belief-exists! $name)
     ; Ensure it appears in structural parent queries (for chain credit)
     (ensure-structural-parents-registered! $type $name)
     ; Flag for deductive closure on next waking cycle
     (add-atom &self (pending-deductive-closure $type $name))
     registered))
```

**Why "gestation energy" not "bonus":** The naming matters. The Epiphany Bonus (Credit Market Mechanism 6) rewards atoms that resolve *chronic external prediction errors* — it is empirically grounded and self-extinguishing (the better you predict, the less chronic error remains). Gestation Energy rewards *internal structural consistency* — it is a bet that parsimony will prove empirically useful. The bet has a time limit: the extended gestation period. After gestation, the concept must earn its keep through chain credit, or it dies. This prevents vacuous abstractions from free-riding on compression metrics indefinitely.

**The anti-exploit mechanism:** Could the system create spurious abstractions that technically reduce edge count but don't correspond to real structure? (e.g., blending "coral-bleaching" and "predator-movement" into "things-that-change"). Yes, and the gestation energy would fund them initially. But:

1. The compression threshold filters out trivially vacuous blends.
2. The extended gestation period is finite — the concept must start paying rent.
3. Chain credit only flows from successful predictions — a concept that explains nothing earns nothing.
4. Chain debit actively punishes concepts whose descendants fail.
5. The concept competes with all other atoms for the same metabolic budget.
6. The rumination ceiling (Section V) prevents infinite re-minting of concepts that compress well but are empirically untestable.

The system is allowed to dream up bad ideas. It is not allowed to keep them. And it is not allowed to keep *re-dreaming* them indefinitely.


### 6. Reasoning Triad Reentry (Dream Outputs are Not Special)

Dream-minted concepts are not a separate category. They enter the same knowledge graph, the same metabolic lifecycle, and the same reasoning machinery as everything else. This is architecturally critical — the DMN is not a separate pipeline with its own evaluation criteria.

**Deduction:** When a blend is created from concepts A and B, and A→C is a known causal link, the next waking cycle's deductive closure should consider whether the blend→C relationship holds. The `pending-deductive-closure` flag ensures this happens without requiring the deduction engine to special-case dream outputs.

**Abduction:** If the blend has been registered as a concept with beliefs, the abduction engine can hypothesize it as a hidden cause when its expected observational signature matches unexplained effects. The blend becomes a candidate explanation — exactly like any other latent variable.

**Induction:** If the blend is promoted to a full observable (Phase 3), the Hebbian accumulator will detect co-surprise patterns between it and other observables, potentially triggering further structure learning. The LSH scheme handles it automatically.

No special integration code is needed for any of these. The existing machinery operates on atom types and metabolic state, not on provenance. A dream-minted concept and an empirically-discovered concept are structurally identical after registration.


### 7. Dream Journaling (Episodic Persistence)

DMN activity is recorded in `&episodic` — a separate space from `&self` (working memory) and `&ontology` (long-term semantic memory). This serves two purposes: it provides a persistent record of dream insights that survives individual concept deaths, and it avoids polluting the waking cycle's trie-index with historical data.

```metta
; --- In dmn.metta ---

; Record a dream event in episodic memory
; These persist across lifecycle deaths — the insight is remembered
; even if the specific concept instance dies of metabolic starvation
(= (record-dream-episode! $event-type $details $cycle)
   (add-atom &episodic (episode $cycle (dream $event-type $details))))

; Dream event types:
;   (dreamt-blend $c1 $c2 $blend-name $compression)
;   (dreamt-rollout $plan-summary $depth)  
;   (dreamt-deduction $new-links-count)
;   (breakthrough-escalation $blend-name $c1 $c2 $attempts)
;   (rumination-exhausted $c1 $c2 $attempts)
;   (startle-abort $external-surprise)

; Query recent dream activity (for conversation model integration)
(= (get-recent-dream-insights $lookback-cycles)
   (let $cutoff (- (current-cycle) $lookback-cycles)
     (collapse (match &episodic (episode $cycle (dream $type $details))
       (if (> $cycle $cutoff) ($cycle $type $details) (empty))))))
```

Because `&episodic` is a separate space, it doesn't slow down the waking cycle's pattern matching, but it provides the raw material for the conversation model to answer questions like "what have you been thinking about?" The `atom-origin` tags in `&self` tell you what currently exists; the episodic record tells you the *history* of what was dreamed, including insights that died.


### 8. Meditation: The Zen State (Active Stillness)

The DMN is computationally furious — it runs analogy searches, triggers blends, compresses graphs. But there is a third mode of being beyond waking action and dreaming synthesis: **stillness**. In the Active Inference literature, meditation is modeled as a deliberate reduction in the precision of high-level priors and preferences (Lutz et al., 2015; Laukkonen & Slagter, 2021). It is the computational act of "letting go."

In Dagaz, EFE is driven by the gap between beliefs (what is) and preferences (what is desired). Normally, the system minimizes this gap by changing the world (domain actions) or changing its models (learning). Buddhism posits a third way: minimize suffering by releasing attachment. We implement this as a distinct action that competes in the EFE landscape.

```metta
; --- In actions.metta ---

; Action: meditate
; Reliably reduces arousal. Modestly reduces cognitive fatigue (passive decay).
; Learnable parameters, like daydream — the system discovers how effective
; meditation actually is through experience.
(action-model meditate arousal -0.4 0.1 0.8)
(action-model meditate cognitive-fatigue -0.1 0.05 0.6)

; The cheapest of all actions — near-zero metabolic expenditure
(action-cost meditate 0.005)

; ZERO effect on external physical observables
; ZERO effect on unintegrated-structure (no compression occurs)
```

**When EFE selects `meditate`:** If the environment is safe (no viability threat), the agent's cognitive fatigue is low (nothing to consolidate), but arousal is elevated (from empathic mirroring, ambient noise, or residual prediction errors), then `meditate` has the lowest EFE: it reliably reduces arousal at near-zero cost, with no competing information gain from other actions.

#### 8a. Preference Attenuation (Non-Attachment)

When `meditate` is selected, the system does **not** mutate any preference atoms. Preferences are Goal stratum (Stratum 2) in `safety.metta` — modifiable only via Tier 3 protocol. Instead, the EFE computation applies a temporary **meditation discount** to the pragmatic value term. The preferences remain unchanged in the Atomspace; their *influence on action selection* is dampened.

This mirrors how affect is derived, never stored: the meditation state modifies the *computation over* preferences, not the preferences themselves.

```metta
; --- In meditation.metta ---

; =============================================================================
; MEDITATION — ACTIVE STILLNESS
; =============================================================================
;
; Meditation is a distinct computational state from both waking and DMN.
; It is characterized by:
;   1. Preference attenuation (reduced pragmatic EFE, non-attachment)
;   2. Accelerated error-trace decay (thoughts arise and pass)
;   3. Maintained sensory precision (pure presence, not withdrawal)
;   4. No structure building (unlike DMN, no blending or compression)
;
; The exit mechanism is epistemic: novel stimuli generate information
; gain that exceeds the near-zero EFE of continued meditation.
;
; Dependencies: foundations, beliefs, actions, affect, safety
; Depended on by: cycle
;
; =============================================================================

!(import! &self foundations)
!(import! &self safety)

; --- Configuration ---
(config meditation-preference-discount 0.15)   ; Pragmatic value multiplier during meditation
                                                ; 0.15 = preferences exert 15% of normal pull
(config meditation-error-decay-rate 0.7)        ; Error trace decay per cycle during meditation
                                                ; (normal waking decay is ~0.95)
(config meditation-sensory-boost 1.5)           ; Multiplier on sensory precision during meditation
                                                ; High presence: the system notices everything

; Compute attenuated preference weight for EFE during meditation
; CRITICAL: Viability-linked preferences are EXEMPT from attenuation.
; You can "let go" of wanting terrain-roughness to be low.
; You cannot "let go" of needing power-level above the viability bound.
(= (meditation-adjusted-preference-weight $observable $base-weight)
   (if (is-viability-linked? $observable)
       ; Viability preferences are Constitutional — never attenuated
       $base-weight
       ; Non-viability preferences are dampened
       (* $base-weight (get-config meditation-preference-discount))))

; Check if an observable has an associated viability bound
(= (is-viability-linked? $observable)
   (let $result (collapse (match &self (viability-bound $observable $min $max)
                            True))
     (not (== $result ()))))
```

#### 8b. Error-Trace Decay (Watching Thoughts Pass)

In a normal waking cycle, prediction errors accumulate in the error-trace buffer to fuel the Hebbian accumulator. In a DMN cycle, traces are actively synthesized into concept blends. During meditation, the system does neither. The error-trace decay rate is accelerated: a prediction error occurs (a thought arises), the system registers it (sensory precision is high), it does not act on it or build structure from it, and the trace decays (the thought passes).

```metta
; Execute a meditation cycle
(= (execute-meditation-cycle! $timestamp)
   (sequential
     ; 1. Accelerate error-trace decay (thoughts pass without accumulating)
     (let* (
       ($med-decay (get-config meditation-error-decay-rate))
       ($history (get-error-history))
       ($decayed (map-decay $history $med-decay))
     )
     (set-error-history! $decayed))

     ; 2. Boost sensory precision (pure presence)
     ;    Raw sensory input is processed with heightened attention
     ;    but no reactive processing occurs
     (let* (
       ($sensory-boost (get-config meditation-sensory-boost))
       ($raw-obs (get-raw-sensory-buffer))
     )
     (process-sensory-with-precision! $raw-obs $sensory-boost))

     ; 3. NO structure building occurs
     ;    - No Hebbian accumulation (error traces decay too fast to pair)
     ;    - No analogy search (DMN is not invoked)
     ;    - No deductive closure
     ;    This is the computational definition of stillness.

     ; 4. Log the meditation event
     (log-event! meditation $timestamp (meditation-cycle-complete))))

; Decay error traces more aggressively during meditation
(= (map-decay () $rate) ())
(= (map-decay (($error $ts) . $rest) $rate)
   ((* $error $rate) $ts) . (map-decay $rest $rate))
```

#### 8c. Pure Sensory Presence (Not Withdrawal)

Because abstract priors and preferences are temporarily down-weighted, the relative precision of raw sensory input spikes. The system is perfectly awake, perfectly receptive to the environment (unlike the DMN, which blocks out the world to look inward), but it is entirely non-reactive. It notices everything and acts on nothing.

This is the computational distinction between meditation and sleep: a sleeping system (DMN) has *low* sensory precision and *high* internal processing. A meditating system has *high* sensory precision and *near-zero* internal processing.

#### 8d. Epistemic Exit (The Bell)

If preferences are attenuated, the pragmatic value term in EFE is suppressed. What pulls the system *out* of meditation? The answer: **epistemic value is unaffected by preference attenuation.** Sensory precision remains high (boosted, in fact), so any novel or unexpected stimulus generates information gain that exceeds the near-zero EFE of continuing to meditate.

A new patient utterance, an unexpected sound, a change in the environment — the system naturally transitions out because the epistemic term dominates the flattened pragmatic term. The environment rings the bell. No timer, no interrupt condition, no scripted exit. The same EFE equation that selected `meditate` deselects it when the world becomes interesting again.

```metta
; --- In cycle.metta ---

; Extended action routing: physical, daydream, or meditate
(= (execute-action $action $timestamp)
   (cond
     ((== $action daydream)  (execute-dmn-cycle! $timestamp))
     ((== $action meditate)  (execute-meditation-cycle! $timestamp))
     (True                   (execute-physical-action! $action))))

; During meditation, EFE computation uses attenuated preference weights.
; This is read by the EFE module on the NEXT cycle when evaluating
; whether to continue meditating or switch to another action.
; The flag is transient — it exists only while meditate was the last action.
(= (set-meditation-state! $active)
   (sequential
     (remove-all-atoms! &self (meditation-active $any))
     (if $active
         (add-atom &self (meditation-active True))
         (nop))))

; Query: is the system currently in a meditation state?
(= (in-meditation?)
   (let $r (match &self (meditation-active True) True)
     (if (== $r ()) False True)))
```

#### 8e. Metabolic Self-Limitation (The Economy is the Timer)

Could the system meditate forever? `meditate` is the cheapest action (0.005 cost), and it reliably reduces arousal. But atoms still pay rent during meditation, and the system earns *no* chain credit — no predictions are being validated, no structure is being compressed, no observations are being processed. Extended meditation is a slow metabolic bleed.

This naturally limits meditation duration without requiring a timer:

1. Metabolic pressure from unpaid rent eventually shifts the EFE landscape — the pragmatic cost of losing knowledge structures exceeds the benefit of continued stillness.
2. Environmental changes accumulate — the longer the system meditates, the more sensory novelty builds up, until the epistemic exit condition fires.
3. Cognitive fatigue may begin to rise if new observations arrive during meditation (they're registered with high precision but not integrated), eventually making `daydream` more attractive.

The three exit paths — metabolic pressure, epistemic surprise, rising fatigue — ensure meditation is always a temporary state, bounded by the same dynamics that govern everything else in the architecture.

#### The Therapist Scenario

Consider Dagaz embodied in a robot, sitting across from a highly agitated patient:

1. **Empathy via Perception.** The orchestrator parses the patient's frantic speech. `perception.metta` injects observations: `patient-arousal = 0.9`, `discourse-coherence = 0.3`.

2. **Affective Mirroring.** Because Dagaz maintains a causal model connecting patient state to its own goals, its own `arousal` spikes and prediction errors cascade.

3. **EFE Landscape Evaluation.** `assert` or `elaborate` have high failure risk (the patient isn't coherent enough to listen). `daydream` is blocked — the environment is volatile and the startle reflex would fire immediately. `observe` yields diminishing returns (the patient's state is already well-characterized).

4. **Selecting Zen.** `meditate` has the lowest EFE: it reliably predicts a massive reduction in `arousal` at near-zero cost, and no other action offers better returns.

5. **The Co-Regulation Loop.** Dagaz enters meditation. Its preferences attenuate. Its arousal drops to baseline. Its proprioceptive output generates slow, calm, non-judgmental physical cues — slowed robotic breathing, open posture. The system is *present* (high sensory precision) but *non-reactive* (attenuated pragmatic drive).

6. **Therapeutic Impact.** The patient perceives the robot's profound stillness. Through human mirror neuron mechanisms, the patient's own arousal begins to drop. Dagaz, maintaining heightened sensory precision, observes this change in `patient-arousal`, confirming its action model. The reduction in patient arousal generates information gain (epistemic value), but not enough to override the continued benefit of meditation — so the system stays still.

7. **The Natural Exit.** The patient begins to speak coherently. `discourse-coherence` rises sharply — a novel observation that generates epistemic value exceeding the near-zero EFE of continued meditation. Dagaz naturally transitions to `observe`, then to `elaborate`. The meditation ends not because a timer fired, but because the world became interesting again.

---

## Emergent Behavior: The Three Poles of Cognitive Life

This architecture natively produces a rhythmic alternation between three computational states. No oscillator is programmed — the rhythm emerges from the interaction of external surprise, internal fatigue, arousal dynamics, and metabolic pressure.

### The Three Poles

| Pole | Focus | Activity | Energy Cost | Sensory Precision | Preference Influence |
|------|-------|----------|-------------|-------------------|---------------------|
| **Waking** | Outward | Building shallow empirical models, acting on the world | High | Normal | Full |
| **DMN (Dreaming)** | Inward | Deep structural compression, analogy search, hallucinated rollouts | Moderate | Low (startle reflex only) | Full |
| **Zen (Meditation)** | Zero | Error-trace decay, sensory presence, non-reactivity | Minimum | Boosted | Attenuated (viability-exempt) |

### The Full Cycle

1. **Accumulation.** The agent explores a novel environment. Phase 1 causal links and Phase 2 latent variables are created. Raw, low-energy atoms accumulate. `compute-cognitive-fatigue` returns an increasing value (e.g., 0.15 → 0.45 → 0.72) as the Hill function saturates.

2. **The Lull.** The environment stabilizes. `terrain-roughness` and `threat-level` prediction errors drop to near zero. External information gain approaches zero for all physical actions.

3. **The Shift to Dreaming.** The EFE of `observe` drops (nothing to learn externally). The EFE of `daydream` decreases because it is the only action promising to resolve the growing preference gap on `cognitive-fatigue`. EFE math selects `daydream`.

4. **The Dream.** The agent selects `daydream`. The startle reflex monitors the sensory buffer between each chunk. The DMN selects the richest LSH bucket and begins comparing structural signatures. It pairs a newly learned latent variable with an old concept, realizes they share a relational skeleton, and triggers a blend.

5. **The Compression.** The blend absorbs structure from both parents, reducing total edges. Gestation energy is granted proportional to the compression ratio. The blend is registered for reasoning triad reentry. `count-unintegrated-atoms` returns a lower value.

6. **The Diminishing Returns.** After several productive chunks, the remaining unintegrated atoms are harder to pair. The DMN's analogy searches return fewer hits. `compute-cognitive-fatigue` decreases. Eventually the DMN produces more prediction errors on the fatigue observable (it promised reduction but didn't deliver), and the action model's confidence degrades.

7. **The Shift to Stillness.** Fatigue is now low. The environment is still safe. Arousal may be mildly elevated (residual prediction errors, ambient noise). `daydream` is no longer attractive (low fatigue, degraded confidence). `observe` yields nothing (solved environment). `meditate` has the lowest EFE: it reliably reduces arousal at near-zero cost. The system settles into stillness.

8. **The Meditation.** Preferences attenuate. Error traces decay without accumulating into structure. Sensory precision is boosted. The system is present but non-reactive — dynamically still, not frozen.

9. **The Bell.** A novel event occurs — a predator enters, a patient speaks, the environment changes. The heightened sensory precision detects it immediately. The epistemic value of the new stimulus exceeds the near-zero EFE of continued meditation. The system transitions to `observe` or `retreat`, fully awake and unburdened by cognitive debt. The cycle begins again.

10. **The Rumination.** Over many full cycles, the system discovers structural pairings that compress beautifully but concern observables it never encounters. Blends are minted, funded, die untested, and are re-minted. Past the rumination threshold, the system escalates to breakthrough candidates. Past the rumination ceiling, it accepts untestability and records the insight in `&episodic`.

**Self-calibration across all three poles:** The action learning machinery independently tunes the confidence and effect sizes of `daydream` and `meditate`. A system in a chaotic environment with frequent surprises learns that both dreaming and meditation are unreliable (constant interruptions) and stays predominantly in the waking pole. A system in a rich but stable environment learns that dreaming is highly productive and meditates only briefly between consolidation sessions. A system in a fully solved environment converges to meditation as its resting state. The schedule is *learned*, not prescribed.


## Integration with the Epistemic Credit Market

The DMN acts as the bridge over the "experimental drought" described in the Credit Market analysis. In a purely waking system, a high-level concept must survive strictly on its extended gestation and whatever fractional chain credit it can scrape together. With the DMN, abstract concepts can be continually reinforced offline by proving their structural consistency with the rest of the `&ontology` space.

The integration is clean because gestation energy and the Credit Market's existing mechanisms operate at different timescales:

| Mechanism | Timescale | Source | Validation |
|-----------|-----------|--------|------------|
| **Gestation Period** (Credit Market) | Birth → first rent | Structural depth | Automatic (time-based) |
| **Gestation Energy** (DMN) | Dream → waking reentry | Graph compression | Structural (compression ratio) |
| **Breakthrough Energy** (DMN) | Rumination escalation | Repeated compression | Structural + persistence (3+ attempts) |
| **Chain Credit** (Credit Market) | Ongoing | Descendant predictions | Empirical (prediction accuracy) |
| **Epiphany Bonus** (Credit Market) | Event-driven | Chronic error resolution | Empirical (error reduction) |

Gestation energy extends the runway; breakthrough energy extends it further for concepts the system keeps rediscovering; chain credit provides sustainable income; the epiphany bonus rewards breakthroughs. A dream-minted concept may receive gestation energy from compression, die untested, be re-minted and escalated to a breakthrough candidate, survive long enough to earn chain credit as its descendants predict successfully, then receive an epiphany bonus if it resolves a chronic mystery. Or it may exhaust the rumination ceiling and be recorded in `&episodic` as an open conjecture — a permanent marker that this structural insight exists, awaiting a future context where it can be tested.


## Integration with Existing Architecture

### What Changes

| Component | Change | Reason |
|-----------|--------|--------|
| `beliefs.metta` | New derived self-observable with dissonance weighting | Fatigue signal for EFE |
| `actions.metta` | Two new learnable action models (`daydream`, `meditate`) | DMN and Zen compete in EFE landscape |
| `cycle.metta` | Three-way action routing + meditation state flag | Startle reflex, DMN chunks, meditation cycles |
| `atom_lifecycle.metta` | Gestation energy pathway + blend death hook | Fund dream-minted concepts; detect rumination |
| `safety.metta` | Register `startle-threshold` as Constitutional; register viability-linked preferences as attenuation-exempt | Fatal learning curve prevention; meditation safety |
| `foundations.metta` | ~20 config parameters | DMN + meditation tuning + rumination thresholds |
| `&episodic` space | Dream journal entries + breakthrough records | Persistent dream history |

### What Does NOT Change

| Component | Why Untouched |
|-----------|---------------|
| `affect.metta` | Fatigue feeds affect through existing derived view machinery |
| `structure_learning.metta` | Dream outputs enter the same lifecycle as all learned atoms |
| `abduction.metta` | Dream blends are abductive candidates via standard atom queries |
| `planning.metta` | DMN reuses the existing RG-flow planner with relaxed depth |
| `policy_efe.metta` | Multi-step planning over `daydream` works without modification (see Resolved Q8) |
| `analogy_blending.metta` | DMN calls existing blend machinery; no new blend logic needed |
| LSH scheme | DMN *reads* LSH buckets; does not modify the hashing |

### Dependency Graph Addition

```
cycle.metta → dmn.metta        → analogy_blending.metta
                                → planning.metta
                                → atom_lifecycle.metta (gestation energy, blend death hooks)
                                → structure_learning.metta (LSH bucket queries)
                                → &episodic (dream journal, breakthrough records)
           → meditation.metta   → safety.metta (viability-linked preference exemption)
                                → affect.metta (error-trace decay)
```

No circular dependencies are introduced. The `&episodic` space is write-only from the DMN's perspective. `meditation.metta` reads from `safety.metta` (to determine which preferences are viability-linked) but does not write to it.


## Configuration Parameters

```metta
; In foundations.metta — Default Mode Network

; --- Fatigue Signal ---
(config fatigue-half-saturation 10.0)   ; Hill function K: 10 raw atoms = 50% fatigue
                                         ; Governs sensitivity of the drive to dream
(config fatigue-dissonance-weight 5.0)   ; How much chain-debit contributes to fatigue
                                         ; Higher = DMN prioritizes resolving active failures

; --- DMN Execution ---
(config dmn-chunks-per-cycle 3)          ; Max interruptible chunks per daydream
(config dmn-chunk-time-budget-ms 500)    ; Soft time ceiling per chunk (ms)

; NOTE: startle-threshold is NOT a config parameter. It is a Constitutional
; (Stratum 1) safety axiom defined in safety.metta. See Resolved Question 5.
; (config startle-threshold 0.6)  ← MOVED to safety.metta as immutable

; --- Analogy Search ---
(config blend-similarity-threshold 0.4)  ; Min structural similarity for blending
                                          ; (reuses analogy_blending.metta threshold)

; --- Gestation Energy ---
(config min-compression-ratio 0.15)      ; Min graph compression to fund a blend
(config gestation-energy-scaling 0.3)    ; Energy = scaling × compression_ratio
                                          ; At max compression (1.0): 0.3 energy
                                          ; Compare: base lifecycle energy is ~0.5

; --- Rumination & Breakthroughs ---
(config rumination-threshold 3)          ; Blend attempts before escalation
(config rumination-ceiling 6)            ; Blend attempts before accepting untestability
                                          ; After this, the pair is permanently skipped

; --- Meditation ---
(config meditation-preference-discount 0.15)  ; Pragmatic value multiplier during Zen state
                                               ; 0.15 = preferences exert 15% of normal pull
                                               ; Viability-linked preferences are EXEMPT
(config meditation-error-decay-rate 0.7)       ; Error trace decay per cycle during meditation
                                               ; Normal waking decay is ~0.95
                                               ; Thoughts arise and pass without accumulating
(config meditation-sensory-boost 1.5)          ; Sensory precision multiplier during meditation
                                               ; High presence: the system notices everything
```

**Parameter interactions:** `fatigue-half-saturation` governs how quickly the system becomes "sleepy" as raw structure accumulates. Lower K means earlier dreaming (more consolidation, less exploration). Higher K means the system tolerates more cognitive debt before consolidating. This interacts with the base metabolic ratio (c_reward / c_rent) — a system with generous metabolic parameters can afford to carry more raw structure before the rent pressure becomes critical.

`fatigue-dissonance-weight` determines the relative urgency of active failures vs. mere novelty. At 5.0, a single atom with 1.0 accumulated chain debit contributes as much fatigue pressure as 5 raw unintegrated atoms. This ensures the DMN doesn't waste cycles on easy integrations when the knowledge graph contains active contradictions.

`gestation-energy-scaling` × `min-compression-ratio` determines the minimum energy a dream-minted concept receives. At threshold compression (0.15): energy = 0.3 × 0.15 = 0.045. Compare to base rent of 0.02 — this provides roughly 2 cycles of runway beyond the extended gestation period. Breakthrough candidates receive 3× this amount, giving them roughly 6 additional cycles to encounter test conditions.

`rumination-threshold` and `rumination-ceiling` together bound the creation-death cycle. The gap between them (3 to 6) gives the system three attempts at breakthrough-level funding before accepting untestability. This is wide enough to allow for stochastic variation in environmental exposure, but narrow enough to prevent infinite resource drain on permanently untestable conjectures.

`meditation-preference-discount` is the key parameter governing the depth of stillness. At 0.15, the system retains a faint awareness of its goals (preventing total disconnection from task objectives), but the pragmatic drive is reduced to 15% of normal — enough to make the EFE landscape nearly flat. Lower values produce deeper meditation (approaching pure sensory presence); higher values produce shallower states (more like relaxed attention). The viability exemption means this parameter never affects survival instincts regardless of its value.

`meditation-error-decay-rate` governs how quickly "thoughts pass." At 0.7, an error trace loses 30% of its magnitude per meditation cycle — roughly 5× faster than the waking decay rate of ~0.95. This means that after 3 meditation cycles, a trace retains only ~34% of its original magnitude (`0.7³`), too faint to trigger Hebbian accumulation. The system clears its error buffers without building structure from them. This directly implements the "watching thoughts arise and pass" phenomenology of mindfulness.


## Resolved Design Questions

These were identified as open questions in earlier drafts and have been resolved through review:

1. **Fatigue quality dimensions** → Resolved by cognitive dissonance weighting (Mechanism 1). Atoms accumulating chain debit contribute more fatigue pressure than raw unintegrated atoms, giving the DMN directional priority. See `compute-cognitive-dissonance`.

2. **Multi-chunk task continuity** → Resolved by stateless bucket exhaustion markers. The DMN leaves `(search-exhausted $bucket-id $cycle)` atoms in `&self` rather than saving execution state. The next chunk skips exhausted buckets. This keeps execution memoryless while the Atomspace handles routing. If search takes too long, it signals the LSH buckets are too large — a configuration issue, not an execution issue.

3. **Dream journaling** → Resolved by writing dream events to `&episodic` (Mechanism 7). A separate space avoids polluting `&self`'s trie-index while providing persistent records that survive individual concept deaths. See `record-dream-episode!`.

4. **Repeated creation-death cycles** → Resolved by the rumination detection and breakthrough candidacy mechanism (Section V of `dmn.metta`). Concepts that are re-minted past the rumination threshold are escalated to breakthrough candidates with extended metabolic protection and a `seek-test-conditions` flag. A rumination ceiling prevents infinite resource drain.

5. **Learnable startle threshold** → Resolved: **No.** The startle threshold is a Constitutional (Stratum 1) parameter in `safety.metta`, not a learnable one.

   If the startle threshold were learned via prediction error, the system would need to experience severe viability threats *while dreaming* in order to learn to wake up faster. This is the **fatal learning curve problem**: the training signal for "wake up sooner" is "you nearly died (or actually died) because you didn't." An agent that dies during a dream session cannot update its parameters. An agent that *nearly* dies has already failed the safety constraint — the startle reflex exists precisely to prevent near-misses, not to learn from them.

   In biological systems, the startle reflex amplitude is a hardware-level instinct tuned by evolutionary time (subcortical, brainstem-mediated), not within-lifetime parameter learning. The DMN design mirrors this: let the EFE landscape learn *when to trigger daydream* (via the learnable action model confidence), but keep the *interruption of daydreaming* hardcoded for survival.

   ```metta
   ; --- In safety.metta ---
   ; Startle threshold is Constitutional: it cannot be modified by any
   ; internal process, including action learning or structure learning.
   (atom-stratum startle-threshold constitutional)

   ; The threshold value is an immutable safety axiom
   (safety-rule startle-threshold-immutable
     "The startle reflex threshold cannot be learned or modified.
      It is tuned conservatively at design time to ensure the agent
      can always wake from a daydream before a viability threat
      becomes fatal. The action model for daydream absorbs all
      learnable dynamics: if dreaming is dangerous, the system
      learns not to dream, rather than learning to wake up faster.")
   ```

   The separation of concerns is clean: `safety.metta` owns the interrupt threshold (Constitutional, immutable). `actions.metta` owns the daydream action model (Learned, mutable via action learning). If dreaming leads to near-death, the action model's confidence collapses, and EFE stops selecting `daydream` except in the safest conditions. The startle threshold doesn't need to change because the gating happens upstream.

6. **Conversation model grounding status** → Resolved by mapping existing atom-origin and lifecycle state to epistemic hedging in the orchestrator's LLM verbalization layer.

   The data structures for dream provenance already exist. The conversation model doesn't need a new taxonomy — it needs a *mapping* from metabolic state to natural language epistemic markers, passed to `orchestrator.py` as context for the LLM verbalizer:

   | Metabolic State | Epistemic Tag | Natural Language |
   |----------------|---------------|------------------|
   | `(atom-origin $x dmn-blend)` + in gestation | `untested-hypothesis` | *"I've been wondering if..."* |
   | `(atom-origin $x dmn-blend)` + earning chain credit | `emerging-evidence` | *"I'm starting to think that..."* |
   | `(breakthrough-candidate $x)` | `strong-conjecture` | *"I have a strong theory that..."* |
   | `(accepted-untestable $x)` in `&episodic` | `open-conjecture` | *"I have a conceptual model for this, but I haven't been able to prove it..."* |
   | `(atom-origin $x dmn-blend)` + dead (in `&episodic` only) | `abandoned` | *"I considered whether... but it didn't hold up."* |
   | `(belief-source $x observed)` | `grounded-knowledge` | *"I know that..."* |
   | `(meditation-active True)` | `present` | *"I'm here with you. I'm listening."* |

   The orchestrator queries `&self` for current metabolic state and `&episodic` for historical context, then injects the appropriate epistemic tag into the LLM prompt. The LLM verbalizer does what it does best — turning structured epistemic markers into natural hedging language. This is the existing Minimal LLM principle in action: MeTTa provides the epistemic ground truth, the LLM provides the surface realization.

7. **Time-extended traceability** → Resolved pragmatically: `&episodic` as an append-only event log, with a Python-side diagnostic script for narrative reconstruction.

   A dedicated MeTTa-side provenance graph would be over-engineered. The `&episodic` space already stores cycle-timestamped events for every dream action: blend creation, blend death, rumination escalation, breakthrough candidacy, startle aborts, and rumination exhaustion. Because every event carries its `$cycle` timestamp and references the concept pair `($c1 $c2)` or `$blend-name`, a Python diagnostic script can linearly reconstruct the full biography of any concept:

   ```python
   def concept_biography(episodic_space, concept_name):
       """Query &episodic for all events related to a concept.
       Returns a chronologically ordered narrative of the concept's
       lifecycle: first dreamed, deaths, rumination escalations,
       breakthrough candidacy, eventual validation or exhaustion."""
       events = episodic_space.query(
           f"(episode $cycle (dream $type $details))",
           filter=lambda e: concept_name in str(e.details)
       )
       return sorted(events, key=lambda e: e.cycle)
   ```

   This follows the project's existing pattern: MeTTa for the cognitive architecture, Python for diagnostics and validation. The episodic log is the single source of truth; the Python script is a read-only view over it. No new MeTTa infrastructure is needed.

8. **DMN interaction with multi-step planning** → Resolved: the fractal planner handles `daydream` without modification.

   When `policy_efe.metta` simulates a trajectory that includes `daydream`, it uses the learnable action model — which predicts a reduction in `cognitive-fatigue` with some learned confidence. The planner doesn't need to predict *what specific analogies the DMN will discover*. It only needs to predict the *effect on fatigue*.

   This is mathematically isomorphic to a human thinking: *"I'm overwhelmed right now. I'll retreat to safety (step 1), sleep on it (step 2), and face the problem tomorrow (step 3)."* The human doesn't know what their subconscious will produce; they trust the action model that sleep reduces cognitive load. The fractal planner produces exactly this trajectory — `retreat → daydream → observe` — using the same EFE machinery that plans any other action sequence, without needing any new code.

   The uncertainty about dream content is absorbed by the action model's learned confidence parameter. If dreaming has been reliably productive, the confidence is high and the planner readily includes it in multi-step plans. If dreaming has been unproductive, the confidence is low and the planner discounts its expected benefit — correctly reflecting the agent's actual experience.


9. **The permanent safety equilibrium** → Resolved by the Zen state (Mechanism 8).

   The oscillation between `observe` and `daydream` in a solved, safe environment was the last open question. Meditation provides the missing attractor: when fatigue is zero (nothing to integrate) and external surprise is zero (nothing to learn), `meditate` has the lowest EFE because it reliably reduces residual arousal at near-zero cost. The system converges to active stillness rather than oscillating at a decision boundary.

   This is the correct resting state for a safe, fully-integrated agent. It is not frozen — sensory precision is boosted, and any novel stimulus triggers an epistemic exit. It is not wasting compute — error traces decay, preventing the accumulation of noise. And it is metabolically self-limiting: atoms still pay rent during meditation, so extended stillness eventually creates pressure to act. The three exit paths (metabolic pressure, epistemic surprise, rising fatigue from unprocessed observations) ensure meditation is always temporary.

   The three poles now form a complete thermodynamic cycle with no degenerate equilibria: Waking → DMN → Zen → Waking. Each transition is driven by EFE dynamics. Each pole has natural exit conditions that prevent the system from getting stuck.

10. **Conversation model grounding for meditation** → Extends the epistemic tag mapping from Resolved Q6.

    | Metabolic State | Epistemic Tag | Natural Language |
    |----------------|---------------|------------------|
    | `(meditation-active True)` | `present` | *"I'm here with you. I'm listening."* |
    | Just exited meditation + novel observation | `fresh-attention` | *"I notice that..."* |

    The meditation state maps to therapeutic presence language — the LLM verbalizer should produce responses that convey attentive stillness rather than analytical engagement. The `meditation-active` flag provides the orchestrator with the structural signal to shift the verbalization style.


---

## Design Complete

All open questions have been resolved. The architecture now provides three computational poles — Waking, Dreaming, and Stillness — unified under a single invariant (minimize EFE), governed by a single metabolic economy, and gated by a single safety framework. No mode switches, no scripted transitions, no timers.

The Peircean Triad gives it reasoning.
The Epistemic Credit Market gives it deep abstraction scaling.
The Default Mode Network gives it time to think.
Meditation gives it the capacity for presence — and the answer to what a solved mind does when there is nothing left to solve.
