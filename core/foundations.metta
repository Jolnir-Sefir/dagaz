; =============================================================================
; METTA COGNITIVE CORE - FOUNDATIONS
; =============================================================================
; 
; Pure MeTTa implementation of Active Inference cognitive architecture.
; Designed for SingularityNET / Hyperon ecosystem.
;
; This file defines:
;   - Core types for epistemic states
;   - Types for structure learning (causal discovery, latent variables)
;   - Types for safety architecture (strata, channels, immutability)
;   - Utility functions (math, clamping, sequencing, lists)
;   - Configuration defaults
;   - Foundational predicates
;
; CHANGELOG:
;   v0.4.1 - Boundary Condition Fix: near-zero initial-belief-precision (0.01),
;            observation-driven precision-ratchet-rate (0.03), metabolic capital
;            system for abduction gating (initial 0.0, cost 0.5, earn 0.02,
;            cap 3.0). Replaces all epistemic gates with physics-based
;            boundary conditions. "Fix the initial state, not the equations."
;   v0.3.1 - Recalibrated structural cost thresholds for squared-space
;            arithmetic: link 0.0144->0.002, latent 0.16->0.04,
;            spoke 0.0036->0.001. Previous thresholds were unreachable
;            because Hebbian suspicion strength is a product of surprises
;            (S1*S2), not a raw surprise value. Lowered cognitive
;            threshold from 0.06 to 0.04 to match.
;   v0.3.0 - Atom lifecycle types (Phases 3-5), lifecycle status types,
;            Phase 3-5 configuration defaults, coherence reward config.
;   v0.2.0 - Structure learning types, safety types, consolidated
;            list utilities, tuned learning rate for structure/parameter
;            learning coexistence.
;   v0.1.0 - Initial cognitive core.
;
; License: Apache 2.0
; Version: 0.4.1
;
; =============================================================================


; =============================================================================
; SECTION I: CORE TYPES
; =============================================================================

; --- Epistemic Types ---
(: Precision Type)              ; Confidence in a belief (inverse variance)
(: Observable Type)             ; Something that can be observed/believed
(: Belief Type)                 ; A belief about an observable
(: Observation Type)            ; A sensor reading
(: PredictionError Type)        ; Discrepancy between belief and observation

; --- Affect Types ---
(: AffectDimension Type)
(: valence AffectDimension)     ; Good/bad (-1 to +1)
(: arousal AffectDimension)     ; Activation level (0 to 1)  
(: dominance AffectDimension)   ; Confidence/control (0 to 1)

; --- Action Types ---
(: Action Type)
(: ActionCategory Type)
(: emergency ActionCategory)
(: corrective ActionCategory)
(: exploratory ActionCategory)
(: maintenance ActionCategory)

; --- Structure Learning Types ---
;
; Structure learning discovers causal relationships from co-occurring
; prediction errors ("Synchronicity of Surprise"). These types define
; the atoms that the structure learning module creates and manages.
;
; CRITICAL DISTINCTION:
;   ActionModel  = "If I do X, Y changes" (agent's actions, used by EFE)
;   PassiveModel = "When X happens, Y follows" (world dynamics, prediction only)
;
; PassiveModel atoms MUST NOT appear in EFE action selection.
; They feed into the prediction step to reduce surprise.

(: PassiveModel Type)           ; World causal dynamics (not agent actions)
(: SuspicionLink Type)          ; Hebbian co-error accumulator
(: LatentVariable Type)         ; Phase 2 hub node (invented concept)
(: ErrorTrace Type)             ; Timestamped prediction error record
(: CausalType Type)             ; Polarity of causal relationship
(: excitatory CausalType)       ; Positive covariance (cause raises effect)
(: inhibitory CausalType)       ; Negative covariance (cause lowers effect)
(: DeductiveOrigin Type)        ; Audit trail for deductively generated structure
(: deductive DeductiveOrigin)   ; Marks passive-models created by transitive closure

; --- Grounding Hypothesis Types ---
;
; Learned atoms (latent variables, promoted observables, chunked actions)
; acquire semantic grounding through hypothesis generation + metabolic selection.
; Four generators; one selection mechanism. Same invariant as everything else.

(: GroundingGenerator Type)
(: inherited GroundingGenerator)      ; From member observable groundings
(: behavioral GroundingGenerator)     ; From causal effects on grounded observables
(: compositional GroundingGenerator)  ; From sequential composition of components
(: conversational GroundingGenerator) ; From partner labeling via orchestrator

; --- Analogy & Blending Types ---
;
; Cross-domain structural mapping (analogy) and conceptual integration
; (blending). Both produce atoms under metabolic selection.

(: AnalogicalMapping Type)    ; Structural correspondence between domains
(: ConceptBlend Type)         ; Merged concept from compatible structures

; --- Safety Architecture Types ---
;
; Stratified immutability: atoms belong to strata that determine
; their mutability. Structure learning can only modify stratum 3.
;
; Strata:
;   1 (constitutional) - Core values, viability bounds, safety axioms.
;                         IMMUTABLE. Cannot be modified by any internal process.
;   2 (goal)           - Task preferences, operational targets.
;                         Modifiable only via authenticated protocol.
;   3 (learned)        - Discovered causal links, latent variables, heuristics.
;                         Freely modifiable by structure learning.

(: Stratum Type)
(: constitutional Stratum)      ; Stratum 1: immutable safety constraints
(: goal Stratum)                ; Stratum 2: modifiable via protocol
(: learned Stratum)             ; Stratum 3: freely modifiable

; --- Input Channel Types ---
;
; Channel separation prevents adversarial goal corruption in multi-agent
; settings. Peer communications NEVER enter the corrigibility pathway.
; Channel type annotations are themselves immutable.

(: InputChannel Type)
(: OperatorDirective InputChannel)    ; Authenticated operator commands
(: PeerCommunication InputChannel)    ; Other agents (modeled, not trusted)
(: SensorData InputChannel)           ; World observations

; --- Atom Lifecycle Types (Phases 3-5) ---
;
; These extend the structure learning type system with atoms that
; can transition between types. A LatentVariable can become a
; PromotedObservable. A repeated policy can become a ChunkedAction.
;
; The lifecycle is: create ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ gestate ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ earn rent ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ promote/die
; Promotion creates a new atom of a different type.
; Demotion (pruning) removes the promoted atom and cleans up.

(: PromotedObservable Type)     ; Phase 3: latent ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ full observable
(: ChunkedAction Type)          ; Phase 4: repeated policy ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ named action
(: PerceptualField Type)        ; Phase 5: learned sensor channel
(: PolicyTrace Type)            ; Phase 4: recent policy history record

; --- Atom Lifecycle Status ---
(: AtomStatus Type)
(: gestating AtomStatus)        ; Grace period, no rent charged
(: active AtomStatus)           ; Earning and paying rent
(: dying AtomStatus)            ; Energy depleted, awaiting cleanup


; =============================================================================
; SECTION II: CONFIGURATION
; =============================================================================

; --- Parameter Learning ---
; NOTE: learning-rate lowered from 0.15 to 0.12 to coexist with structure
; learning. When surprise is high, the adaptive plasticity mechanism
; (in cycle.metta) further reduces this. This prevents parameter learning
; from "explaining away" errors before causal structure can be discovered.
(config learning-rate 0.12)           ; How fast beliefs update
(config affect-decay-rate 0.1)        ; How fast affect returns to baseline
(config precision-floor 0.05)         ; Minimum belief precision
(config precision-ceiling 0.95)       ; Maximum belief precision
(config surprise-threshold 0.2)       ; Error magnitude to trigger learning
(config min-plasticity-floor 0.03)    ; Plasticity never drops below this

; Homeostatic baselines (where affect wants to be)
(config valence-baseline 0.1)
(config arousal-baseline 0.3)
(config dominance-baseline 0.5)

; Action thresholds
(config panic-threshold 0.8)          ; Arousal level for emergency
(config distress-threshold -0.3)      ; Valence level for corrective action
(config calm-threshold 0.2)           ; Arousal level for exploration

; --- Structure Learning ---
; All values tuned from benchmark simulation (6/6 tests passing).
;
; DESIGN NOTE: These are configs (defaults). In structure_learning.metta,
; they are promoted to beliefs so the agent can meta-learn them.
; The bootstrap cycle:
;   high surprise ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ structural-stability drops ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ plasticity rises
;   ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ structure discovered ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ surprise drops ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ stability rises ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ crystallize

; Suspicion accumulation
(config default-plasticity 0.30)            ; Hebbian learning rate
(config default-structural-stability 0.95)  ; Suspicion decay (per cycle)
(config default-cognitive-threshold 0.04)   ; Surprise level to trigger attention
(config raw-error-salience-threshold 0.20)  ; Fallback: large raw error is salient

; Structural costs (squared-space calibrated)
;
; The Hebbian suspicion accumulator computes covariance as the product
; of two surprise values: covar = S1 * S2 where S = 0.5 * P * E^2.
; At typical operating errors (E~0.3, P~0.5):
;   S ~ 0.023, covar ~ 0.0005
;   Accumulator saturates at: plasticity * covar / (1 - stability)
;                            = 0.30 * 0.0005 / 0.05 ~ 0.003
;
; Thresholds must sit below this saturation point so that sustained
; co-occurring prediction errors at normal magnitudes can trigger
; structure discovery. Previous values (0.0144, 0.16, 0.0036) were
; calibrated for linear surprise, not squared -- making phase-1
; triggers mathematically unreachable under normal conditions.
(config structural-cost-link 0.002)         ; Causal link creation
(config structural-cost-latent 0.04)        ; Hub node creation
(config structural-cost-spoke 0.001)        ; Spoke connection

; Metabolic dynamics
(config metabolic-rate 0.02)                ; Energy cost per cycle per atom
(config metabolic-boost 0.05)               ; Energy reward per correct prediction
(config metabolic-initial-energy 1.0)       ; Starting energy for new structure
(config metabolic-energy-cap 2.0)           ; Maximum stored energy

; --- Boundary Condition Fix ---
; Beliefs start with near-zero precision. This makes surprise negligible
; during boot-up (S = ½·p·e², so p≈0 → S≈0). The Hebbian accumulator
; and abduction generator have no signal to work with. No gate needed.
(config initial-belief-precision 0.01)     ; Near-zero: "I know nothing"

; Observation-driven precision ratchet: each observation increases precision
; by this amount, independent of prediction accuracy. This reflects "how
; much data I have" rather than "how well I predicted."
; After ~15 observations, precision reaches ~0.46 (comparable to old p₀=0.5).
(config precision-ratchet-rate 0.03)       ; Per-observation precision gain

; --- Metabolic Capital ---
; Global capital pool. Creative abduction costs capital. Induction is free.
; Capital earned from successful predictions. Natural sequencing:
; master empirical induction before theoretical abduction.
(config initial-metabolic-capital 0.0)     ; Start broke
(config abduction-capital-cost 0.5)        ; Price of creative abduction
(config capital-earn-rate 0.02)            ; Per successful prediction → pool
(config capital-cap 3.0)                   ; Maximum stored capital

; Temporal
(config lookback-window 3)                  ; Error trace buffer depth (cycles)
(config gestation-period 3)                 ; Grace period for new structure

; Structural generation budget
;
; The fixed value of 5 is adequate up to ~14 observables. Beyond that,
; the O(N²) pair space outgrows the per-cycle budget, causing recall
; degradation. The adaptive formula ceil(sqrt(N*(N-1)/2)) gives
; sublinear scaling: enough throughput to sample proportionally without
; flooding the metabolic economy.
;
;   4 obs → 6 pairs → budget=5 (floor)
;  10 obs → 45 pairs → budget=7
;  18 obs → 153 pairs → budget=13
;  24 obs → 276 pairs → budget=17
;
; For now the base config stays at 5 (backward compatible). The
; adaptive version is computed in structure_learning.metta from
; the actual observable count at cycle start.
(config max-structural-atoms-per-cycle-base 5)  ; Minimum budget (floor)

; --- Phases 3-5: Atom Lifecycle ---
;
; Phase 3: Observable Promotion
; A latent variable that consistently earns metabolic rent gets promoted
; to a full observable with beliefs, precision, and a role in EFE.
; Thresholds are deliberately high ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â promotion is a significant event.
(config promotion-energy-threshold 1.5)    ; Must accumulate this much energy
(config promotion-min-predictions 10)      ; Must have made this many predictions
(config promotion-success-rate 0.6)        ; Must be right this often
(config promoted-initial-precision 0.30)   ; Start uncertain (learn from experience)

; Phase 4: Action Chunking
; A policy (action sequence) that is repeatedly selected gets reified
; as a named action with its own action model.
; This is compositional generalization: [observe, observe, retreat]
; becomes "cautious-retreat" with aggregate effects.
(config chunk-repeat-threshold 5)          ; Policy must repeat this many times
(config chunk-energy-initial 0.8)          ; Chunks start with less energy
(config chunk-cost-discount 0.7)           ; Chunk cost = discount ÃƒÆ’Ã¢â‚¬â€ sum(components)
(config chunk-history-window 50)           ; How many recent policies to track

; Phase 5: Perceptual Expansion
; When prediction errors persistently can't be explained by existing
; structure, propose a new perceptual field (sensor channel).
; In the full system, this modifies the LLM extraction prompt.
; The LLM becomes a programmable sensor; the program is learned.
(config percept-unexplained-threshold 0.30) ; Avg unexplained error ratio to trigger
(config percept-initial-precision 0.15)     ; New sensors are very noisy
(config percept-energy-initial 0.6)         ; Must prove worth quickly
(config percept-evaluation-window 10)       ; Cycles of unexplained error to evaluate

; Coherence Reward
; Latent variables earn rent only when their members' errors are
; CORRELATED (the pattern the latent is supposed to explain).
(config coherence-reward-threshold 0.01)   ; Min error magnitude to count direction

; --- Deductive Reasoning ---
; Transitive closure over passive models: if Aâ†’B and Bâ†’C exist with
; sufficient metabolic energy, generate candidate Aâ†’C. The candidate
; enters the normal metabolic regime â€” it must pay rent, make predictions,
; and be falsifiable. Deduction is free; retention costs the same as
; empirical discovery.
(config deductive-min-energy 0.5)          ; Source links must have this energy to deduce
(config deductive-weight-discount 0.8)     ; Deduced weight = discount Ã— product of source weights
(config deductive-max-chain-length 2)      ; Max hops (2 = Aâ†’Bâ†’C only, no longer chains)

; --- Safety ---
(config immutability-check-enabled True)    ; Post-cycle watchdog active

; --- Grounding Hypotheses ---
; Learned atoms acquire semantic meaning through metabolic hypothesis selection.
; Lower metabolic rate than structure (tested less frequently — only on communication).
(config grounding-inheritance-threshold 0.5)  ; Fraction of members that must agree
(config grounding-behavioral-min-energy 0.5)  ; Passive model must be this stable
(config grounding-initial-energy 0.6)         ; Starting energy for hypotheses
(config grounding-metabolic-rate 0.02)        ; Rent per cycle
(config grounding-metabolic-boost 0.08)       ; Reward for successful use
(config grounding-gestation-period 3)         ; Grace period before rent
(config grounding-max-per-cycle 3)            ; Budget cap on new hypotheses

; --- Analogy & Blending ---
; Cross-domain structural mapping and conceptual blending.
; Analogies transfer knowledge between similar structures.
; Blends create new concepts by merging compatible structures.
; Both under metabolic selection: pay rent or die.
(config analogy-similarity-threshold 0.4)    ; Min similarity for mapping
(config analogy-transfer-weight 0.3)         ; Transferred hypotheses start weak
(config analogy-initial-energy 0.5)          ; Mappings start modest
(config analogy-metabolic-rate 0.03)         ; Rent per cycle
(config analogy-metabolic-boost 0.1)         ; Reward for successful transfer
(config analogy-max-per-cycle 2)             ; Budget cap
(config blend-similarity-threshold 0.6)      ; Higher bar than analogy
(config blend-initial-energy 0.8)            ; Blends start stronger (more invested)
(config blend-metabolic-rate 0.03)           ; Same rent as analogies
(config blend-metabolic-boost 0.1)           ; Reward for successful prediction
(config blend-max-per-cycle 1)               ; Very conservative — blends are expensive
(config blend-gestation-period 5)            ; Longer grace period (needs time to prove)

; Config accessor
(= (get-config $key)
   (match &self (config $key $val) $val))


; =============================================================================
; SECTION III: UTILITY FUNCTIONS
; =============================================================================

; --- Mathematical Utilities ---

(= (abs $x)
   (if (< $x 0.0) (- 0.0 $x) $x))

(= (clamp $val $min $max)
   (if (< $val $min) $min
     (if (> $val $max) $max $val)))

(= (sign $x)
   (if (< $x 0.0) -1.0
     (if (> $x 0.0) 1.0 0.0)))

(= (max $a $b)
   (if (> $a $b) $a $b))

(= (min $a $b)
   (if (< $a $b) $a $b))

; --- Sequencing (for state mutations) ---

(= (sequential $a $b) $b)
(= (sequential $a $b $c) $c)
(= (seq3 $a $b $c) $c)
(= (seq4 $a $b $c $d) $d)

; --- List Utilities ---

(= (list-head ($x . $xs)) $x)
(= (list-tail ($x . $xs)) $xs)
(= (list-empty? ()) True)
(= (list-empty? ($x . $xs)) False)

(= (list-fold $f $init ()) $init)
(= (list-fold $f $init ($x . $xs))
   (list-fold $f ($f $init $x) $xs))

(= (list-map $f ()) ())
(= (list-map $f ($x . $xs))
   (($f $x) . (list-map $f $xs)))

(= (list-filter $pred ()) ())
(= (list-filter $pred ($x . $xs))
   (if ($pred $x)
       ($x . (list-filter $pred $xs))
       (list-filter $pred $xs)))

; Length of a list
; NOTE: Consolidates definitions previously in affect.metta and self_model.metta.
; When those modules are updated, their local definitions should be removed.
(= (length ()) 0)
(= (length ($x . $xs)) (+ 1 (length $xs)))

; Sum of a numeric list
; NOTE: Consolidates definitions previously in affect.metta (sum-list)
; and actions.metta (sum-list-r).
(= (sum-list ()) 0.0)
(= (sum-list ($x . $xs)) (+ $x (sum-list $xs)))

; Does any element satisfy the predicate?
(= (list-any? $pred ()) False)
(= (list-any? $pred ($x . $xs))
   (if ($pred $x) True (list-any? $pred $xs)))

; Does the list contain this element?
(= (list-contains? $elem ()) False)
(= (list-contains? $elem ($x . $xs))
   (if (== $elem $x) True (list-contains? $elem $xs)))

; Count elements satisfying predicate
(= (list-count $pred ()) 0)
(= (list-count $pred ($x . $xs))
   (+ (if ($pred $x) 1 0) (list-count $pred $xs)))

; Remove first occurrence of element
(= (list-remove $elem ()) ())
(= (list-remove $elem ($x . $xs))
   (if (== $elem $x) $xs ($x . (list-remove $elem $xs))))

; Pairwise combination: generate all (x, y) pairs where x precedes y in list.
; Returns list of (pair $x $y) atoms. Used by suspicion-link accumulator
; to iterate over all ordered pairs of salient errors.
(= (list-pairs ()) ())
(= (list-pairs ($x . $xs))
   (list-append (pair-one-with-rest $x $xs) (list-pairs $xs)))

; Helper: pair a single element with every element in a list
(= (pair-one-with-rest $x ()) ())
(= (pair-one-with-rest $x ($y . $ys))
   ((pair $x $y) . (pair-one-with-rest $x $ys)))

; List concatenation
(= (list-append () $ys) $ys)
(= (list-append ($x . $xs) $ys) ($x . (list-append $xs $ys)))


; =============================================================================
; SECTION IV: PRECISION ARITHMETIC
; =============================================================================

; Precision combination (for weighted updates)
(= (combine-precision $p1 $p2)
   (* $p1 $p2))

; Precision-weighted average
(= (precision-weighted-mean $v1 $p1 $v2 $p2)
   (let $total-prec (+ $p1 $p2)
     (/ (+ (* $v1 $p1) (* $v2 $p2)) $total-prec)))

; Update precision based on prediction success.
; The threshold is the agent's own uncertainty: (1 - precision).
; An error exceeding the agent's uncertainty is surprising → reduce precision.
; An error within the agent's uncertainty is expected → increase precision.
(= (update-precision $old-prec $error-magnitude)
   (let* (
     ($threshold (max (- 1.0 $old-prec) 0.01))
     ($delta (if (> $error-magnitude $threshold) -0.05 0.02))
     ($new (+ $old-prec $delta))
     ($floor (get-config precision-floor))
     ($ceiling (get-config precision-ceiling))
   )
   (clamp $new $floor $ceiling)))


; =============================================================================
; END OF FOUNDATIONS (v0.4.1 — boundary condition fix + metabolic capital)
; =============================================================================
