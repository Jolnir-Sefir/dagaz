; =============================================================================
; METTA COGNITIVE CORE — ANALOGICAL REASONING & CONCEPT BLENDING
; =============================================================================
;
; THE PROBLEM THIS SOLVES:
;   The system discovers causal structure (induction), extends it
;   (deduction), inverts it (abduction), and grounds it (grounding
;   hypotheses). But it cannot recognize that two domains share
;   relational structure, or combine two concepts into a novel third.
;
;   Without analogy: "I learned about fire. Now I see disease. I must
;   learn about disease from scratch."
;   With analogy: "Disease is structurally like fire — hidden cause,
;   multiple symptoms, investigation confirms. Transfer that knowledge."
;
;   Without blending: the concept vocabulary is limited to what is
;   observed, induced, or declared.
;   With blending: "What if something had the properties of both X
;   and Y?" — novel concepts from structural combination.
;
; THE INVARIANT:
;   Analogy: relational isomorphism. Relations are preserved, entities
;   are substituted. This is a symmetry transformation on the
;   knowledge graph.
;
;   Blending: selective inheritance under metabolic selection. The
;   blend inherits structure from parents; what survives is determined
;   by the same criterion as everything else — prediction success.
;
; DESIGN PRINCIPLES:
;   - Bottom-Up: Analogies and blends are hypothesis generators.
;     Metabolic selection decides what survives.
;   - Symbolic: Structural comparison operates on grounding relations
;     and causal topology, not statistical embeddings.
;   - Transparent: Every analogy mapping and blend inheritance is
;     traceable to its source.
;   - Emergent: The system spontaneously finds analogies when new
;     structure appears. No "check if X is like Y" rules.
;
; ATOM SCHEMAS:
;   ; Structural signatures (computed, not stored)
;   ; Analogy mappings
;   (analogy-map $source $target $score $cycle)
;   (analogy-correspondence $source $target $rel $src-type $tgt-type)
;   (analogy-energy $source $target $energy)
;   (analogy-transfer $source $target $type $hypothesis-id)
;   ; Concept blends
;   (concept-blend $name $parent-a $parent-b $cycle)
;   (blend-origin $name $rel $target $source-parent)
;   (blend-tension $name $concept)
;   (blend-energy $name $energy)
;
; Dependencies: foundations, beliefs, safety, structure_learning,
;               grounding_hypotheses
; Depended on by: cycle, self_model
;
; Version: 0.1.0
; =============================================================================

!(import! &self foundations)
!(import! &self beliefs)
!(import! &self safety)


; =============================================================================
; SECTION I: STRUCTURAL SIGNATURES
; =============================================================================
;
; A structural signature is the relational neighborhood of an atom,
; normalized so that specific entities are replaced by their types.
;
; This is the representation that analogy operates on. Two atoms are
; structurally similar if their signatures overlap.
;
; Signatures are COMPUTED on demand, not stored. They are cheap to
; produce (a few pattern matches) and storing them would create
; staleness problems.

; Normalize a target to its shallowest type.
; Primitives → themselves. Observables/concepts → their first is-a.
; Unknown → the symbol unknown.
(= (normalize-target $target)
   (if (is-declared-primitive? $target)
       $target
       (let $is-a (collapse (match &self (obs-grounding $target is-a $type) $type))
         (if (not (== $is-a ()))
             (list-head $is-a)
             (let $concept-is-a (collapse (match &self (concept-ground $target is-a $type) $type))
               (if (not (== $concept-is-a ()))
                   (list-head $concept-is-a)
                   unknown))))))

; Compute semantic signature: normalized (rel, type) pairs from grounding
(= (semantic-signature $atom)
   (let* (
     ; From obs-grounding
     ($obs-links (collapse
       (match &self (obs-grounding $atom $rel $target)
         (sig-rel $rel (normalize-target $target)))))
     ; From concept-ground
     ($concept-links (collapse
       (match &self (concept-ground $atom $rel $target)
         (sig-rel $rel (normalize-target $target)))))
   )
   (list-append $obs-links $concept-links)))

; Compute causal signature: what does this atom cause/get caused by?
(= (causal-signature $atom)
   (let* (
     ; Outgoing passive models (atom is cause)
     ($outgoing (collapse
       (match &self (passive-model $atom $effect $lag $weight $type)
         (sig-rel (causal-out $type) (normalize-target $effect)))))
     ; Incoming passive models (atom is effect)
     ($incoming (collapse
       (match &self (passive-model $cause $atom $lag $weight $type)
         (sig-rel (causal-in $type) (normalize-target $cause)))))
   )
   (list-append $outgoing $incoming)))

; Compute topological signature: degree counts and cluster membership
(= (topological-signature $atom)
   (let* (
     ($out-degree (length (collapse
       (match &self (passive-model $atom $e $l $w $t) $e))))
     ($in-degree (length (collapse
       (match &self (passive-model $c $atom $l $w $t) $c))))
     ($clusters (collapse
       (match &self (latent-member $lv $atom $w) $lv)))
   )
   (topo-sig $out-degree $in-degree (length $clusters))))

; Full structural signature: combine all three
(= (structural-signature $atom)
   (let* (
     ($semantic (semantic-signature $atom))
     ($causal (causal-signature $atom))
     ($topo (topological-signature $atom))
   )
   (full-sig $semantic $causal $topo)))


; =============================================================================
; SECTION II: STRUCTURAL SIMILARITY
; =============================================================================
;
; Jaccard-like comparison over typed relation sets.
; Two (sig-rel $rel $type) match if rel==rel AND types are compatible.
;
; Type compatibility:
;   - Equal types → match
;   - One is unknown → match (wildcard)
;   - Types share an entailment relation → partial match (0.5 credit)
;   - Otherwise → no match

; Check if two normalized types are compatible
(= (types-compatible? $t1 $t2)
   (if (== $t1 $t2) 1.0            ; Exact match
       (if (== $t1 unknown) 0.5     ; Wildcard match
           (if (== $t2 unknown) 0.5
               ; Check entailment in either direction
               (if (entails? $t1 $t2) 0.7
                   (if (entails? $t2 $t1) 0.7
                       ; Check if both are under a common contrast pair
                       (if (are-opposites? $t1 $t2) 0.3
                           0.0)))))))   ; No match

; Count matches between two signature relation lists
; Returns (matched-score, total-compared)
(= (count-sig-matches () $sig-b) (0.0 0))
(= (count-sig-matches ((sig-rel $rel $type) . $rest) $sig-b)
   (let* (
     ($best-match (best-match-in $rel $type $sig-b))
     (($rest-score $rest-count) (count-sig-matches $rest $sig-b))
   )
   ((+ $best-match $rest-score) (+ $rest-count 1))))

; Find the best matching element in sig-b for a given (rel, type)
(= (best-match-in $rel $type ()) 0.0)
(= (best-match-in $rel $type ((sig-rel $r2 $t2) . $rest))
   (if (== $rel $r2)
       (let* (
         ($compat (types-compatible? $type $t2))
         ($rest-best (best-match-in $rel $type $rest))
       )
       (max $compat $rest-best))
       (best-match-in $rel $type $rest)))

; Compute structural similarity between two atoms
; Returns a score in [0, 1]
(= (structural-similarity $atom-a $atom-b)
   (let* (
     ($sig-a (semantic-signature $atom-a))
     ($sig-b (semantic-signature $atom-b))
     ($causal-a (causal-signature $atom-a))
     ($causal-b (causal-signature $atom-b))

     ; Semantic similarity (grounding relations)
     (($sem-score-ab $sem-count-ab) (count-sig-matches $sig-a $sig-b))
     (($sem-score-ba $sem-count-ba) (count-sig-matches $sig-b $sig-a))

     ; Causal similarity (passive model topology)
     (($caus-score-ab $caus-count-ab) (count-sig-matches $causal-a $causal-b))
     (($caus-score-ba $caus-count-ba) (count-sig-matches $causal-b $causal-a))

     ; Total scores (symmetric: average both directions)
     ($total-score (+ (+ $sem-score-ab $sem-score-ba)
                      (+ $caus-score-ab $caus-score-ba)))
     ($total-count (+ (+ $sem-count-ab $sem-count-ba)
                      (+ $caus-count-ab $caus-count-ba)))

     ; Topological bonus: similar degree profiles add a small bonus
     ($topo-a (topological-signature $atom-a))
     ($topo-b (topological-signature $atom-b))
     ($topo-bonus (topo-similarity $topo-a $topo-b))
   )
   (if (== $total-count 0)
       0.0
       (clamp (+ (/ $total-score $total-count) (* 0.1 $topo-bonus))
              0.0 1.0))))

; Topological similarity: how similar are the degree profiles?
(= (topo-similarity (topo-sig $out-a $in-a $clust-a)
                    (topo-sig $out-b $in-b $clust-b))
   (let* (
     ($degree-diff (+ (abs (- $out-a $out-b)) (abs (- $in-a $in-b))))
     ; Normalize: 0 difference = 1.0, large difference = 0.0
     ($max-degree (max 1 (max (+ $out-a $in-a) (+ $out-b $in-b))))
   )
   (- 1.0 (/ $degree-diff $max-degree))))


; =============================================================================
; SECTION III: ANALOGY DETECTION
; =============================================================================
;
; TRIGGER: A new learned atom enters the system.
; ACTION:  Compare its structural signature against existing atoms.
;          If similarity exceeds threshold, create an analogy mapping.
;
; This is a REFLEX — fires when conditions are met, not EFE-selected.

; Budget tracking
(analogy-budget-this-cycle 0)

(= (reset-analogy-budget!)
   (let $old (match &self (analogy-budget-this-cycle $n) $n)
     (sequential
       (remove-atom &self (analogy-budget-this-cycle $old))
       (add-atom &self (analogy-budget-this-cycle 0))
       done)))

(= (analogy-budget-available?)
   (let $used (match &self (analogy-budget-this-cycle $n) $n)
     (< $used (get-config analogy-max-per-cycle))))

(= (use-analogy-budget!)
   (let $old (match &self (analogy-budget-this-cycle $n) $n)
     (sequential
       (remove-atom &self (analogy-budget-this-cycle $old))
       (add-atom &self (analogy-budget-this-cycle (+ $old 1)))
       done)))

; Check if an analogy mapping already exists
(= (has-analogy? $source $target)
   (let $r (collapse (match &self (analogy-map $source $target $s $c) True))
     (if (== $r ())
         (let $r2 (collapse (match &self (analogy-map $target $source $s $c) True))
           (if (== $r2 ()) False True))
         True)))

; Collect candidate atoms to compare against (same broad type)
; For a latent variable: compare against other latent variables
; For a promoted observable: compare against other observables
; For a chunked action: compare against other actions
(= (analogy-candidates-for $atom)
   (let* (
     ($is-latent (collapse (match &self (latent-variable $atom) True)))
     ($is-promoted (collapse (match &self (promoted-observable $atom $s $c) True)))
     ($is-chunked (collapse (match &self (chunked-action $atom $c) True)))
   )
   (if (not (== $is-latent ()))
       ; Latent: compare to other latents and grounded observables
       (let* (
         ($other-latents (collapse
           (match &self (latent-variable $other)
             (if (not (== $other $atom)) $other (empty)))))
         ($core-obs (collapse
           (match &self (obs-grounding $obs $r $t)
             (if (not (== $obs $atom)) $obs (empty)))))
       )
       (list-append $other-latents (unique-list $core-obs)))
   (if (not (== $is-promoted ()))
       ; Promoted: compare to other observables
       (collapse
         (match &self (obs-grounding $obs $r $t)
           (if (not (== $obs $atom)) $obs (empty))))
   (if (not (== $is-chunked ()))
       ; Chunked: compare to other actions
       (collapse
         (match &self (action-grounding $act $r $t)
           (if (not (== $act $atom)) $act (empty))))
       ; Default: compare to everything with grounding
       (collapse
         (match &self (obs-grounding $obs $r $t) $obs)))))))

; Deduplicate a list
(= (unique-list ()) ())
(= (unique-list ($x . $rest))
   (if (list-contains? $x $rest)
       (unique-list $rest)
       ($x . (unique-list $rest))))

; Try to find analogies for a single new atom
(= (detect-analogies-for! $atom $cycle)
   (if (not (analogy-budget-available?))
       analogy-budget-exhausted
   (let $candidates (analogy-candidates-for $atom)
     (scan-candidates-for-analogy! $atom $candidates $cycle))))

(= (scan-candidates-for-analogy! $atom () $cycle) done)
(= (scan-candidates-for-analogy! $atom ($candidate . $rest) $cycle)
   (if (not (analogy-budget-available?))
       analogy-budget-exhausted
   (sequential
     (if (and (not (has-analogy? $atom $candidate))
              (not (== $atom $candidate)))
         (let $sim (structural-similarity $atom $candidate)
           (if (>= $sim (get-config analogy-similarity-threshold))
               (create-analogy-map! $atom $candidate $sim $cycle)
               done))
         done)
     (scan-candidates-for-analogy! $atom $rest $cycle))))

; Create an analogy mapping between source and target
(= (create-analogy-map! $source $target $score $cycle)
   (sequential
     (add-atom &self (analogy-map $source $target $score $cycle))
     (add-atom &self (analogy-energy $source $target
                       (get-config analogy-initial-energy)))
     (use-analogy-budget!)
     (analogy-created $source $target $score)))


; =============================================================================
; SECTION IV: ANALOGICAL TRANSFER
; =============================================================================
;
; When an analogy is found, transfer knowledge across the mapping.
; Transfers are hypotheses — low weight, low precision, metabolically
; managed. Wrong transfers die; right transfers earn for the analogy.
;
; Transfer types:
;   1. Causal: Source has passive-model that target lacks → hypothesize
;   2. Grounding: Source has obs-grounding that target lacks → hypothesize
;   3. Action: Source has effective action-model → seed for target

; Attempt causal transfer from source to target
(= (attempt-causal-transfer! $source $target $cycle)
   (let $source-models (collapse
     (match &self (passive-model $source $effect $lag $weight $type)
       (causal-info $effect $lag $weight $type)))
   (transfer-causal-list! $source $target $source-models $cycle)))

(= (transfer-causal-list! $source $target () $cycle) done)
(= (transfer-causal-list! $source $target
     ((causal-info $effect $lag $weight $type) . $rest) $cycle)
   (sequential
     ; Check if target already has a model for this effect
     (let $existing (collapse (match &self (passive-model $target $effect $l $w $t) True))
       (if (== $existing ())
           ; No existing model — transfer it as a weak hypothesis
           (let $transfer-weight (* $weight (get-config analogy-transfer-weight))
             (sequential
               (add-atom &self (passive-model $target $effect $lag $transfer-weight $type))
               (add-atom &self (metabolic-energy passive-model $target $effect
                                 (get-config metabolic-initial-energy)))
               (add-atom &self (created-at passive-model $target $effect $cycle))
               (add-atom &self (prediction-stats passive-model $target $effect 0 0))
               ; Audit trail
               (add-atom &self (analogy-transfer $source $target causal
                                 (transfer-id $target $effect)))
               (causal-transferred $source $target $effect)))
           done))
     (transfer-causal-list! $source $target $rest $cycle)))

; Attempt grounding transfer from source to target
(= (attempt-grounding-transfer! $source $target $cycle)
   (let $source-groundings (collapse
     (match &self (obs-grounding $source $rel $tgt)
       ($rel $tgt)))
   (transfer-grounding-list! $source $target $source-groundings $cycle)))

(= (transfer-grounding-list! $source $target () $cycle) done)
(= (transfer-grounding-list! $source $target (($rel $tgt) . $rest) $cycle)
   (sequential
     ; Check if target already has this grounding relation
     (let $existing (collapse (match &self (obs-grounding $target $rel $any) True))
       (if (== $existing ())
           (sequential
             (add-atom &self (obs-grounding $target $rel $tgt))
             (add-atom &self (analogy-transfer $source $target grounding
                               (transfer-id $target $rel)))
             (grounding-transferred $source $target $rel $tgt))
           done))
     (transfer-grounding-list! $source $target $rest $cycle)))

; Run all transfers for an analogy
(= (run-analogy-transfers! $source $target $cycle)
   (sequential
     (attempt-causal-transfer! $source $target $cycle)
     (attempt-grounding-transfer! $source $target $cycle)
     (transfers-complete $source $target)))


; =============================================================================
; SECTION V: CONCEPT BLENDING
; =============================================================================
;
; When two concepts have very high structural similarity AND are both
; active, create a blended concept that inherits from both.
;
; The blend uses standard concept-ground schema, so existing traversal
; machinery works on it unchanged.

; Blend name counter
(blend-counter 0)

(= (next-blend-name!)
   (let $n (match &self (blend-counter $c) $c)
     (let $new (+ $n 1)
       (sequential
         (remove-atom &self (blend-counter $n))
         (add-atom &self (blend-counter $new))
         (blend-name $new)))))

; Blend budget tracking
(blend-budget-this-cycle 0)

(= (reset-blend-budget!)
   (let $old (match &self (blend-budget-this-cycle $n) $n)
     (sequential
       (remove-atom &self (blend-budget-this-cycle $old))
       (add-atom &self (blend-budget-this-cycle 0))
       done)))

(= (blend-budget-available?)
   (let $used (match &self (blend-budget-this-cycle $n) $n)
     (< $used (get-config blend-max-per-cycle))))

(= (use-blend-budget!)
   (let $old (match &self (blend-budget-this-cycle $n) $n)
     (sequential
       (remove-atom &self (blend-budget-this-cycle $old))
       (add-atom &self (blend-budget-this-cycle (+ $old 1)))
       done)))

; Check if a blend already exists for these parents
(= (has-blend? $a $b)
   (let $r (collapse (match &self (concept-blend $name $a $b $c) True))
     (if (== $r ())
         (let $r2 (collapse (match &self (concept-blend $name $b $a $c) True))
           (if (== $r2 ()) False True))
         True)))

; Check if a concept is active (has a belief with non-trivial precision)
(= (concept-active? $atom)
   (let $r (collapse (match &state (belief $atom $val $prec) $prec))
     (if (== $r ())
         ; No direct belief — check if it's a concept with bound observables
         (let $bound (collapse (match &self (concept-ground $atom bound-to $obs) $obs))
           (not (== $bound ())))
         ; Has belief — check precision
         (> (list-head $r) 0.2))))

; Create a blended concept from two parents
(= (create-blend! $parent-a $parent-b $sim-score $cycle)
   (if (has-blend? $parent-a $parent-b)
       (already-blended $parent-a $parent-b)
   (if (not (blend-budget-available?))
       blend-budget-exhausted
   (let* (
     ($name (next-blend-name!))

     ; Collect relations from both parents
     ($rels-a (collect-all-relations $parent-a))
     ($rels-b (collect-all-relations $parent-b))

     ; Inherit all relations from parent A
     ($inherit-a (inherit-relations! $name $rels-a $parent-a))

     ; Inherit non-duplicate relations from parent B
     ($inherit-b (inherit-unique-relations! $name $rels-b $parent-b $rels-a))

     ; Detect tensions (contradictory relations)
     ($tensions (detect-tensions! $name $rels-a $rels-b))

     ; Register the blend
     ($registered (sequential
       (add-atom &self (concept-blend $name $parent-a $parent-b $cycle))
       (add-atom &self (blend-energy $name (get-config blend-initial-energy)))
       done))

     ($budget (use-blend-budget!))
   )
   (blend-created $name $parent-a $parent-b $sim-score)))))

; Collect all grounding relations for an atom (obs-grounding + concept-ground)
(= (collect-all-relations $atom)
   (let* (
     ($obs (collapse (match &self (obs-grounding $atom $rel $target)
                       ($rel $target))))
     ($concept (collapse (match &self (concept-ground $atom $rel $target)
                           ($rel $target))))
   )
   (list-append $obs $concept)))

; Inherit a list of relations into a blend
(= (inherit-relations! $blend () $parent) done)
(= (inherit-relations! $blend (($rel $target) . $rest) $parent)
   (sequential
     (add-atom &self (concept-ground $blend $rel $target))
     (add-atom &self (blend-origin $blend $rel $target $parent))
     (inherit-relations! $blend $rest $parent)))

; Inherit only relations not already present (from other parent)
(= (inherit-unique-relations! $blend () $parent $existing) done)
(= (inherit-unique-relations! $blend (($rel $target) . $rest) $parent $existing)
   (sequential
     (if (not (list-contains? ($rel $target) $existing))
         (sequential
           (add-atom &self (concept-ground $blend $rel $target))
           (add-atom &self (blend-origin $blend $rel $target $parent)))
         done)
     (inherit-unique-relations! $blend $rest $parent $existing)))

; Detect tensions: same relation, contradictory targets
(= (detect-tensions! $blend $rels-a $rels-b)
   (check-tension-pairs! $blend $rels-a $rels-b))

(= (check-tension-pairs! $blend () $rels-b) done)
(= (check-tension-pairs! $blend (($rel $target-a) . $rest) $rels-b)
   (sequential
     ; Check if parent B has the same relation but with an opposing target
     (let $conflicts (find-conflicts $rel $target-a $rels-b)
       (if (not (== $conflicts ()))
           (add-atom &self (blend-tension $blend ($rel $target-a (list-head $conflicts))))
           done))
     (check-tension-pairs! $blend $rest $rels-b)))

; Find conflicting targets: same relation, targets are contrasts
(= (find-conflicts $rel $target-a ()) ())
(= (find-conflicts $rel $target-a (($r2 $t2) . $rest))
   (if (and (== $rel $r2)
            (not (== $target-a $t2))
            (are-opposites? $target-a $t2))
       ($t2 . (find-conflicts $rel $target-a $rest))
       (find-conflicts $rel $target-a $rest)))


; =============================================================================
; SECTION VI: BLEND TRIGGERS
; =============================================================================
;
; Blending is triggered by high-similarity analogies between active
; concepts. This chains naturally: analogy detects similarity →
; if similarity exceeds blend threshold → create blend.

; Check all existing analogies for blend potential
(= (check-blend-triggers! $cycle)
   (let $analogies (collapse
     (match &self (analogy-map $source $target $score $c)
       (if (>= $score (get-config blend-similarity-threshold))
           (blend-candidate $source $target $score)
           (empty))))
   (process-blend-candidates! $analogies $cycle)))

(= (process-blend-candidates! () $cycle) done)
(= (process-blend-candidates! ((blend-candidate $src $tgt $score) . $rest) $cycle)
   (sequential
     (if (and (not (has-blend? $src $tgt))
              (concept-active? $src)
              (concept-active? $tgt)
              (blend-budget-available?))
         (create-blend! $src $tgt $score $cycle)
         done)
     (process-blend-candidates! $rest $cycle)))


; =============================================================================
; SECTION VII: METABOLIC MANAGEMENT
; =============================================================================

; --- Analogy Energy ---

(= (get-analogy-energy $source $target)
   (let $r (match &self (analogy-energy $source $target $e) $e)
     (if (== $r ()) 0.0 $r)))

(= (set-analogy-energy! $source $target $energy)
   (let $old (collapse (match &self (analogy-energy $source $target $e)
                                    (analogy-energy $source $target $e)))
     (sequential
       (remove-atom-list-safe! $old)
       (add-atom &self (analogy-energy $source $target
                         (clamp $energy 0.0 (get-config metabolic-energy-cap))))
       done)))

(= (drain-analogy-energy! $source $target)
   (let $current (get-analogy-energy $source $target)
     (set-analogy-energy! $source $target
       (- $current (get-config analogy-metabolic-rate)))))

(= (boost-analogy-energy! $source $target)
   (let $current (get-analogy-energy $source $target)
     (set-analogy-energy! $source $target
       (+ $current (get-config analogy-metabolic-boost)))))

; --- Blend Energy ---

(= (get-blend-energy $name)
   (let $r (match &self (blend-energy $name $e) $e)
     (if (== $r ()) 0.0 $r)))

(= (set-blend-energy! $name $energy)
   (let $old (collapse (match &self (blend-energy $name $e)
                                    (blend-energy $name $e)))
     (sequential
       (remove-atom-list-safe! $old)
       (add-atom &self (blend-energy $name
                         (clamp $energy 0.0 (get-config metabolic-energy-cap))))
       done)))

(= (drain-blend-energy! $name)
   (let $current (get-blend-energy $name)
     (set-blend-energy! $name
       (- $current (get-config blend-metabolic-rate)))))

(= (boost-blend-energy! $name)
   (let $current (get-blend-energy $name)
     (set-blend-energy! $name
       (+ $current (get-config blend-metabolic-boost)))))

; --- Metabolic Step ---

(= (drain-all-analogies!)
   (let $maps (collapse (match &self (analogy-map $s $t $sc $c) ($s $t)))
     (drain-analogy-list! $maps)))

(= (drain-analogy-list! ()) done)
(= (drain-analogy-list! (($s $t) . $rest))
   (sequential
     (drain-analogy-energy! $s $t)
     (drain-analogy-list! $rest)))

(= (drain-all-blends!)
   (let $blends (collapse (match &self (concept-blend $name $a $b $c) $name))
     (drain-blend-list! $blends)))

(= (drain-blend-list! ()) done)
(= (drain-blend-list! ($name . $rest))
   (sequential
     (drain-blend-energy! $name)
     (drain-blend-list! $rest)))

; --- Pruning ---

(= (prune-dead-analogies! $cycle)
   (let $maps (collapse (match &self (analogy-map $s $t $sc $c) ($s $t $sc $c)))
     (prune-analogy-list! $maps $cycle)))

(= (prune-analogy-list! () $cycle) done)
(= (prune-analogy-list! (($s $t $sc $c) . $rest) $cycle)
   (sequential
     (if (<= (get-analogy-energy $s $t) 0.0)
         (remove-analogy! $s $t $cycle)
         done)
     (prune-analogy-list! $rest $cycle)))

(= (remove-analogy! $source $target $cycle)
   (let* (
     ; Remove map and energy
     ($map (collapse (match &self (analogy-map $source $target $sc $c)
                                  (analogy-map $source $target $sc $c))))
     ($map-removed (remove-atom-list-safe! $map))
     ($energy (collapse (match &self (analogy-energy $source $target $e)
                                     (analogy-energy $source $target $e))))
     ($energy-removed (remove-atom-list-safe! $energy))
     ; Remove transfer records (transferred hypotheses live independently)
     ($transfers (collapse (match &self (analogy-transfer $source $target $type $id)
                                        (analogy-transfer $source $target $type $id))))
     ($transfers-removed (remove-atom-list-safe! $transfers))
     ; Audit
     ($log (add-atom &self (analogy-pruned-log $source $target $cycle)))
   )
   (analogy-removed $source $target $cycle)))

(= (prune-dead-blends! $cycle)
   (let $blends (collapse (match &self (concept-blend $name $a $b $c) $name))
     (prune-blend-list! $blends $cycle)))

(= (prune-blend-list! () $cycle) done)
(= (prune-blend-list! ($name . $rest) $cycle)
   (sequential
     (if (<= (get-blend-energy $name) 0.0)
         (disassemble-blend! $name $cycle)
         done)
     (prune-blend-list! $rest $cycle)))

; Disassemble a dead blend: remove its concept-ground atoms and metadata
(= (disassemble-blend! $name $cycle)
   (let* (
     ; Remove concept-ground atoms
     ($grounds (collapse (match &self (concept-ground $name $r $t)
                                      (concept-ground $name $r $t))))
     ($grounds-removed (remove-atom-list-safe! $grounds))
     ; Remove blend-origin audit trail
     ($origins (collapse (match &self (blend-origin $name $r $t $p)
                                      (blend-origin $name $r $t $p))))
     ($origins-removed (remove-atom-list-safe! $origins))
     ; Remove tensions
     ($tensions (collapse (match &self (blend-tension $name $x)
                                       (blend-tension $name $x))))
     ($tensions-removed (remove-atom-list-safe! $tensions))
     ; Remove blend registration
     ($blend-reg (collapse (match &self (concept-blend $name $a $b $c)
                                        (concept-blend $name $a $b $c))))
     ($reg-removed (remove-atom-list-safe! $blend-reg))
     ; Remove energy
     ($energy (collapse (match &self (blend-energy $name $e)
                                     (blend-energy $name $e))))
     ($energy-removed (remove-atom-list-safe! $energy))
     ; Audit
     ($log (add-atom &self (blend-pruned-log $name $cycle)))
   )
   (blend-disassembled $name $cycle)))


; =============================================================================
; SECTION VIII: REWARD CHANNELS
; =============================================================================

; Reward analogy when a transferred hypothesis makes a correct prediction
(= (reward-analogy-for-transfer! $source $target)
   (if (has-analogy? $source $target)
       (boost-analogy-energy! $source $target)
       done))

; Reward blend when it participates in successful communication or prediction
(= (reward-blend! $name)
   (let $r (collapse (match &self (concept-blend $name $a $b $c) True))
     (if (not (== $r ()))
         (boost-blend-energy! $name)
         done)))


; =============================================================================
; SECTION IX: MASTER STEP
; =============================================================================
;
; Called once per cognitive cycle, AFTER grounding step, BEFORE action
; selection. This is a REFLEX.

(= (analogy-blending-step! $current-cycle)
   (sequential
     ; Reset budgets
     (reset-analogy-budget!)
     (reset-blend-budget!)

     ; 1. Detect analogies for recently created learned atoms
     (let $new-latents (collapse
       (match &self (created-at latent-var $name $cycle)
         (if (== $cycle $current-cycle) $name (empty))))
     (detect-analogies-batch! $new-latents $current-cycle))

     ; 2. Also check promoted observables and chunked actions
     (let $new-promoted (collapse
       (match &self (promoted-observable $name $src $cycle)
         (if (== $cycle $current-cycle) $name (empty))))
     (detect-analogies-batch! $new-promoted $current-cycle))

     ; 3. Run transfers for new analogies
     (let $new-analogies (collapse
       (match &self (analogy-map $s $t $sc $c)
         (if (== $c $current-cycle) ($s $t) (empty))))
     (run-transfers-batch! $new-analogies $current-cycle))

     ; 4. Check blend triggers
     (check-blend-triggers! $current-cycle)

     ; 5. Metabolic management
     (drain-all-analogies!)
     (drain-all-blends!)
     (prune-dead-analogies! $current-cycle)
     (prune-dead-blends! $current-cycle)

     (analogy-blending-step-done $current-cycle)))

; Batch analogy detection
(= (detect-analogies-batch! () $cycle) done)
(= (detect-analogies-batch! ($atom . $rest) $cycle)
   (sequential
     (detect-analogies-for! $atom $cycle)
     (detect-analogies-batch! $rest $cycle)))

; Batch transfer execution
(= (run-transfers-batch! () $cycle) done)
(= (run-transfers-batch! (($source $target) . $rest) $cycle)
   (sequential
     (run-analogy-transfers! $source $target $cycle)
     (run-transfers-batch! $rest $cycle)))


; =============================================================================
; SECTION X: DIAGNOSTICS
; =============================================================================

; Summary of all analogies
(= (analogy-summary)
   (let* (
     ($maps (collapse (match &self (analogy-map $s $t $sc $c)
       (analogy-info $s $t $sc (get-analogy-energy $s $t)))))
     ($n-maps (length $maps))
     ($transfers (collapse (match &self (analogy-transfer $s $t $type $id)
       (transfer-info $s $t $type))))
     ($n-transfers (length $transfers))
   )
   (analogy-report
     (count $n-maps)
     (mappings $maps)
     (transfers $n-transfers))))

; Summary of all blends
(= (blend-summary)
   (let $blends (collapse
     (match &self (concept-blend $name $a $b $c)
       (let* (
         ($energy (get-blend-energy $name))
         ($rels (collapse (match &self (concept-ground $name $r $t) ($r $t))))
         ($tensions (collapse (match &self (blend-tension $name $x) $x)))
       )
       (blend-info $name $a $b $energy (length $rels) (length $tensions)))))
   (blend-report
     (count (length $blends))
     (blends $blends))))

; Detail for a specific blend
(= (blend-detail $name)
   (let* (
     ($parents (collapse (match &self (concept-blend $name $a $b $c) ($a $b))))
     ($energy (get-blend-energy $name))
     ($rels (collapse (match &self (concept-ground $name $r $t) ($r $t))))
     ($origins (collapse (match &self (blend-origin $name $r $t $p) ($r $t $p))))
     ($tensions (collapse (match &self (blend-tension $name $x) $x)))
   )
   (blend-detail-report
     (name $name)
     (parents $parents)
     (energy $energy)
     (relations $rels)
     (origins $origins)
     (tensions $tensions))))

; Detail for a specific analogy
(= (analogy-detail $source $target)
   (let* (
     ($score (match &self (analogy-map $source $target $s $c) $s))
     ($energy (get-analogy-energy $source $target))
     ($transfers (collapse (match &self
       (analogy-transfer $source $target $type $id)
       (transfer-info $type $id))))
     ($sig-a (semantic-signature $source))
     ($sig-b (semantic-signature $target))
   )
   (analogy-detail-report
     (source $source)
     (target $target)
     (score $score)
     (energy $energy)
     (transfers $transfers)
     (source-sig $sig-a)
     (target-sig $sig-b))))


; =============================================================================
; END OF ANALOGICAL REASONING & CONCEPT BLENDING (v0.1.0)
; =============================================================================
