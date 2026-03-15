; =============================================================================
; DIMENSIONAL PRIMITIVES (v2)
; =============================================================================
;
; DESIGN PRINCIPLES APPLIED:
;   - Dimensions are the structure for ordinal comparison
;   - Minimal entity assignments (examples only, not exhaustive)
;   - Implemented query functions
;   - Connected to belief system for inference
;
; KEY INSIGHT: "Is X bigger than Y?" is an operation on dimensional scales,
; not a memorized fact. Structure enables generalization.
;
; =============================================================================


; =============================================================================
; SECTION I: TYPES
; =============================================================================

(: Dimension Type)
(: Scale Type)
(: ComparisonResult Type)

(: greater ComparisonResult)
(: lesser ComparisonResult)
(: equal-rank ComparisonResult)
(: unknown ComparisonResult)


; =============================================================================
; SECTION II: DIMENSION DEFINITIONS
; =============================================================================

; Each dimension has an associated scale
; Schema: (dimension <name> <scale>)

(dimension physical-size size-scale)
(dimension temporal-rate speed-scale)
(dimension physical-weight weight-scale)
(dimension cognitive-capacity intelligence-scale)
(dimension temporal-age age-scale)
(dimension thermal-intensity temperature-scale)


; =============================================================================
; SECTION III: SCALE DEFINITIONS
; =============================================================================

; Ordinal values with explicit ranks
; Schema: (scale-value <scale> <rank> <label>)
; Lower rank = less of the quality

; Size scale
(scale-value size-scale 1 microscopic)
(scale-value size-scale 2 tiny)
(scale-value size-scale 3 small)
(scale-value size-scale 4 medium)
(scale-value size-scale 5 large)
(scale-value size-scale 6 very-large)
(scale-value size-scale 7 enormous)

; Speed scale
(scale-value speed-scale 1 stationary)
(scale-value speed-scale 2 slow)
(scale-value speed-scale 3 moderate)
(scale-value speed-scale 4 fast)
(scale-value speed-scale 5 very-fast)

; Weight scale
(scale-value weight-scale 1 negligible)
(scale-value weight-scale 2 light)
(scale-value weight-scale 3 medium)
(scale-value weight-scale 4 heavy)
(scale-value weight-scale 5 massive)

; Intelligence scale
(scale-value intelligence-scale 1 minimal)
(scale-value intelligence-scale 2 low)
(scale-value intelligence-scale 3 moderate)
(scale-value intelligence-scale 4 high)
(scale-value intelligence-scale 5 exceptional)


; =============================================================================
; SECTION IV: COMPARATIVE ADJECTIVES
; =============================================================================

; Map linguistic comparatives to dimensional operations
; Schema: (comparative <word> <dimension> <direction>)

(comparative bigger physical-size greater)
(comparative larger physical-size greater)
(comparative smaller physical-size lesser)

(comparative faster temporal-rate greater)
(comparative slower temporal-rate lesser)

(comparative heavier physical-weight greater)
(comparative lighter physical-weight lesser)

(comparative smarter cognitive-capacity greater)
(comparative older temporal-age greater)
(comparative younger temporal-age lesser)

(comparative hotter thermal-intensity greater)
(comparative colder thermal-intensity lesser)


; =============================================================================
; SECTION V: ENTITY DIMENSION VALUES (Minimal Examples)
; =============================================================================

; Schema: (dim-value <entity> <dimension> <label>)
; These are EXAMPLES for testing - real system would learn these

; Size examples
(dim-value Ant physical-size tiny)
(dim-value Cat physical-size medium)
(dim-value Dog physical-size medium)
(dim-value Elephant physical-size very-large)
(dim-value Whale physical-size enormous)

; Speed examples
(dim-value Turtle temporal-rate slow)
(dim-value Human temporal-rate moderate)
(dim-value Cheetah temporal-rate very-fast)

; Intelligence examples
(dim-value Ant cognitive-capacity minimal)
(dim-value Dog cognitive-capacity moderate)
(dim-value Human cognitive-capacity exceptional)


; =============================================================================
; SECTION VI: COMPARISON FUNCTIONS (Implemented)
; =============================================================================

; Get dimension for an entity
(= (get-dimension-value $entity $dimension)
   (match &self (dim-value $entity $dimension $label) $label))

; Get rank of a label on a scale
(= (get-rank $label $scale)
   (match &self (scale-value $scale $rank $label) $rank))

; Get scale for a dimension
(= (get-scale $dimension)
   (match &ontology (dimension $dimension $scale) $scale))

; Compare two entities on a dimension
; Returns: greater, lesser, equal-rank, or unknown
(= (compare-on-dimension $entity1 $entity2 $dimension)
   (let* (
     ($label1 (get-dimension-value $entity1 $dimension))
     ($label2 (get-dimension-value $entity2 $dimension))
   )
   (if (or (== $label1 ()) (== $label2 ()))
       unknown
       (let* (
         ($scale (get-scale $dimension))
         ($rank1 (get-rank $label1 $scale))
         ($rank2 (get-rank $label2 $scale))
       )
       (if (> $rank1 $rank2)
           greater
           (if (< $rank1 $rank2)
               lesser
               equal-rank))))))

; Answer comparative questions: "Is X bigger than Y?"
; Usage: !(comparative-query Dog Ant bigger)
(= (comparative-query $entity1 $entity2 $adjective)
   (match &self (comparative $adjective $dimension $direction)
     (let $result (compare-on-dimension $entity1 $entity2 $dimension)
       (if (== $result $direction)
           True
           (if (== $result unknown)
               unknown
               False)))))


; =============================================================================
; SECTION VII: SUPERLATIVE FUNCTIONS
; =============================================================================

; Find entity with highest/lowest value on dimension among a set
; Usage: !(find-extreme (Ant Cat Elephant) physical-size greater)

(= (find-extreme $entities $dimension $direction)
   (find-extreme-helper $entities $dimension $direction () 0))

(= (find-extreme-helper () $dim $dir $best $best-rank) $best)
(= (find-extreme-helper ($e . $rest) $dim $dir $best $best-rank)
   (let* (
     ($label (get-dimension-value $e $dim))
     ($scale (get-scale $dim))
     ($rank (get-rank $label $scale))
     ($rank-val (if (== $rank ()) 0 $rank))
   )
   (if (== $best ())
       (find-extreme-helper $rest $dim $dir $e $rank-val)
       (if (== $dir greater)
           (if (> $rank-val $best-rank)
               (find-extreme-helper $rest $dim $dir $e $rank-val)
               (find-extreme-helper $rest $dim $dir $best $best-rank))
           (if (< $rank-val $best-rank)
               (find-extreme-helper $rest $dim $dir $e $rank-val)
               (find-extreme-helper $rest $dim $dir $best $best-rank))))))


; =============================================================================
; SECTION VIII: STRUCTURAL CAPABILITIES
; =============================================================================

; Physical structures that enable actions
(enables-structure legs walking)
(enables-structure legs running)
(enables-structure wings flying)
(enables-structure fins swimming)
(enables-structure gills breathing-underwater)

; Structure requirements (necessary conditions)
(requires-structure flying wings)
(requires-structure swimming fins)
(requires-structure breathing-underwater gills)

; Entity structures (minimal examples)
(has-structure Dog legs)
(has-structure Bird wings)
(has-structure Bird legs)
(has-structure Fish fins)
(has-structure Fish gills)
(has-structure Human legs)

; Explicit lacks (for negative inference)
(lacks-structure Dog wings)
(lacks-structure Human wings)
(lacks-structure Human gills)

; Can entity perform action via structure?
(= (can-via-structure? $entity $action)
   (match &self (enables-structure $struct $action)
     (case (match &self (has-structure $entity $struct) True)
       ((True True)
        (() False)))))

; Cannot perform action due to lacking required structure?
(= (cannot-due-to-structure? $entity $action)
   (match &self (requires-structure $action $struct)
     (case (match &self (lacks-structure $entity $struct) True)
       ((True True)
        (() False)))))


; =============================================================================
; SECTION IX: INTEGRATION WITH BELIEFS
; =============================================================================

; Dimensional knowledge can be represented as beliefs
; This allows the Active Inference core to reason about dimensions

; Convert dim-value to belief format
; The "value" is the rank, precision reflects confidence
(= (dimension-as-belief $entity $dimension)
   (let* (
     ($label (get-dimension-value $entity $dimension))
     ($scale (get-scale $dimension))
     ($rank (get-rank $label $scale))
     ($max-rank (max-scale-rank $scale))
     ($normalized (/ $rank $max-rank))
   )
   (belief (dim $entity $dimension) $normalized 0.8)))

; Get max rank in a scale (for normalization)
(= (max-scale-rank $scale)
   (max-of-ranks (collapse (match &self (scale-value $scale $r $_) $r))))

(= (max-of-ranks ()) 1)
(= (max-of-ranks ($r)) $r)
(= (max-of-ranks ($r1 $r2)) (max $r1 $r2))
(= (max-of-ranks ($r1 . $rest)) (max $r1 (max-of-ranks $rest)))


; =============================================================================
; SECTION X: OBSERVABLE INTEGRATION
; =============================================================================

; Dimensional assessments can come from observations
; Schema: (observe-dimension $entity $dimension $label $precision)

(= (observe-dimension! $entity $dimension $label $precision $time)
   (let* (
     ($scale (get-scale $dimension))
     ($rank (get-rank $label $scale))
     ($max-rank (max-scale-rank $scale))
     ($normalized (/ $rank $max-rank))
     ($obs-name (dim-obs $entity $dimension))
   )
   (add-atom &state (observation $obs-name $normalized $precision $time))))


; =============================================================================
; END OF DIMENSIONAL PRIMITIVES
; =============================================================================
