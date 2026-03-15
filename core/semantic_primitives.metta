; =============================================================================
; SEMANTIC PRIMITIVES (v3)
; =============================================================================
;
; DESIGN PRINCIPLES APPLIED:
;   - Primitives are the termination points for grounding chains
;   - Functions are implemented, not stubbed
;   - Minimal but complete
;
; USE/MENTION DISTINCTION:
;   Primitives are stored as (primitive (concept X) <grounding-type>).
;   The (concept ...) wrapper distinguishes MENTIONING a concept (reasoning
;   about conjunction) from USING an operator (computing with 'and').
;   This prevents trie index collisions when primitive names shadow
;   MeTTa builtins (and, or, not) or symbols used in other type contexts.
;
;   The wrapper is internal to the representation. All query functions
;   (is-primitive?, opposite-of, etc.) accept and return bare symbols.
;   Callers never need to know about (concept ...).
;
; NOTE: Hand-curated primitives are acceptable — this is where grounding
; bottoms out. But the USAGE of primitives should be emergent.
;
; CHANGELOG (v2.1 → v3):
;   - All primitives wrapped with (concept ...) for use/mention distinction
;   - Resolves trie index crash when 'and', 'or', 'not' collide with builtins
;   - Query API unchanged: is-primitive?, opposite-of, etc. take bare symbols
;
; CHANGELOG (v2 → v2.1):
;   - Added 7 primitives from v1 needed to make structural relations coherent:
;     cannot, need, believe, kind-of, toward, away, quantity
;   - Section III: Contrast network expanded from 9 → 14 pairs
;   - Section IV: Entailment network expanded from 3 → 5 pairs
;   - Fixed inconsistency: (contrast can cannot) and (entails know believe)
;     were declared but their second terms were not registered as primitives
;
; =============================================================================


; =============================================================================
; SECTION I: TYPES
; =============================================================================

(: SemanticPrimitive Type)
(: GroundingType Type)

; How primitives are justified
(: axiomatic GroundingType)       ; Cannot be further reduced
(: perceptual GroundingType)      ; Grounded in sensory observation
(: experiential GroundingType)    ; Grounded in subjective state
(: derived GroundingType)         ; Defined via other primitives


; =============================================================================
; SECTION II: PRIMITIVE DECLARATIONS
; =============================================================================
; Schema: (primitive (concept <name>) <grounding-type>)
; The (concept ...) wrapper distinguishes mentioning a concept from
; using a same-named operator. Query functions abstract over this.

; --- Ontological (what IS) ---
(primitive (concept existence) axiomatic)
(primitive (concept thing) axiomatic)
(primitive (concept same) axiomatic)
(primitive (concept different) axiomatic)
(primitive (concept part-of) axiomatic)

; --- Logical (reasoning machinery) ---
(primitive (concept true) axiomatic)
(primitive (concept false) axiomatic)
(primitive (concept and) axiomatic)
(primitive (concept or) axiomatic)
(primitive (concept not) axiomatic)
(primitive (concept if-then) axiomatic)

; --- Relational (how things connect) ---
(primitive (concept is-a) axiomatic)
(primitive (concept kind-of) derived)       ; Subtype relation; entails is-a
(primitive (concept has) axiomatic)
(primitive (concept causes) axiomatic)
(primitive (concept enables) axiomatic)
(primitive (concept prevents) axiomatic)

; --- Spatial (where) ---
(primitive (concept place) axiomatic)
(primitive (concept at) axiomatic)
(primitive (concept in) axiomatic)
(primitive (concept near) axiomatic)
(primitive (concept far) axiomatic)
(primitive (concept toward) axiomatic)      ; In the direction of
(primitive (concept away) derived)          ; In the opposite direction; contrast: toward

; --- Temporal (when) ---
(primitive (concept time) axiomatic)
(primitive (concept before) axiomatic)
(primitive (concept after) axiomatic)
(primitive (concept during) axiomatic)
(primitive (concept now) experiential)

; --- Quantity (how much) ---
(primitive (concept one) axiomatic)
(primitive (concept more) axiomatic)
(primitive (concept less) axiomatic)
(primitive (concept equal) axiomatic)
(primitive (concept none) axiomatic)
(primitive (concept quantity) axiomatic)    ; The abstract category of measurable amount

; --- Quality (basic properties) ---
(primitive (concept good) axiomatic)
(primitive (concept bad) axiomatic)
(primitive (concept big) axiomatic)
(primitive (concept small) axiomatic)

; --- Agency (action and intention) ---
(primitive (concept agent) axiomatic)
(primitive (concept do) axiomatic)
(primitive (concept want) axiomatic)
(primitive (concept need) derived)          ; Require; stronger than want; entails want
(primitive (concept can) axiomatic)
(primitive (concept cannot) derived)        ; Lack ability; contrast: can
(primitive (concept know) axiomatic)
(primitive (concept believe) derived)       ; Hold as true; weaker than know; entailed by know

; --- Experiential (subjective states) ---
(primitive (concept feel) experiential)
(primitive (concept pleasure) experiential)
(primitive (concept pain) experiential)

; --- Perceptual (sensory grounding) ---
(primitive (concept see) perceptual)
(primitive (concept hear) perceptual)
(primitive (concept touch) perceptual)


; =============================================================================
; SECTION III: CONTRAST RELATIONS
; =============================================================================
; Opposites — enables negation reasoning and antonym traversal.
; Convention: (contrast (concept A) (concept B)) means A and B are semantic
; opposites. Query functions accept bare symbols.

; Logical
(contrast (concept true) (concept false))

; Identity / Ontological
(contrast (concept same) (concept different))

; Relational
(contrast (concept enables) (concept prevents))
(contrast (concept prevents) (concept enables))

; Spatial
(contrast (concept near) (concept far))
(contrast (concept toward) (concept away))

; Temporal
(contrast (concept before) (concept after))

; Quantity
(contrast (concept more) (concept less))

; Quality
(contrast (concept good) (concept bad))
(contrast (concept big) (concept small))

; Agency
(contrast (concept can) (concept cannot))

; Experiential
(contrast (concept pleasure) (concept pain))


; =============================================================================
; SECTION IV: ENTAILMENT RELATIONS
; =============================================================================
; Meaning postulates: if X is true, then Y must also be true.
; These are not inferences to be learned — they are constitutive of meaning.
; (entails (concept A) (concept B)) = "being A implies being B"

(entails (concept in) (concept at))           ; If in X, then at X
(entails (concept kind-of) (concept is-a))    ; If kind-of X, then is-a X
(entails (concept need) (concept want))       ; If you need X, you want X
(entails (concept know) (concept believe))    ; If you know X, you believe X
(entails (concept causes) (concept enables))  ; If X causes Y, then X enables Y


; =============================================================================
; SECTION V: QUERY FUNCTIONS
; =============================================================================
; All functions accept and return BARE symbols.
; The (concept ...) wrapper is internal to the representation.

; Check if symbol is a primitive
(= (is-primitive? $sym)
   (case (match &self (primitive (concept $sym) $g) $g)
     (($grounding True)
      (() False))))

; Get grounding type of primitive
(= (grounding-type $sym)
   (match &self (primitive (concept $sym) $g) $g))

; Get all primitives (returns bare symbols)
(= (all-primitives)
   (collapse (match &self (primitive (concept $p) $g) $p)))

; Get primitives by grounding type (returns bare symbols)
(= (primitives-of-type $gtype)
   (collapse (match &self (primitive (concept $p) $gtype) $p)))

; Get contrast (opposite) of a primitive (bare symbol in, bare symbol out)
(= (opposite-of $sym)
   (match &ontology (contrast (concept $sym) (concept $opp)) $opp))

; Check if two primitives are opposites (symmetric check)
(= (are-opposites? $a $b)
   (case (match &ontology (contrast (concept $a) (concept $b)) True)
     ((True True)
      (() (case (match &ontology (contrast (concept $b) (concept $a)) True)
            ((True True)
             (() False)))))))

; Check if $a entails $b
(= (entails? $a $b)
   (case (match &ontology (entails (concept $a) (concept $b)) True)
     ((True True)
      (() False))))

; Get everything a concept entails (returns bare symbols)
(= (what-entails $concept)
   (collapse (match &ontology (entails (concept $concept) (concept $e)) $e)))

; Get all contrasting pairs (returns pairs of bare symbols)
(= (all-contrasts)
   (collapse (match &ontology (contrast (concept $a) (concept $b)) ($a $b))))

; Get all entailment pairs (returns pairs of bare symbols)
(= (all-entailments)
   (collapse (match &ontology (entails (concept $a) (concept $b)) ($a $b))))

; Check if primitive is axiomatic
(= (is-axiomatic? $sym)
   (case (match &self (primitive (concept $sym) axiomatic) True)
     ((True True)
      (() False))))


; =============================================================================
; SECTION VI: GROUNDING CHAIN UTILITIES
; =============================================================================

; A concept is grounded if it either:
; 1. Is a primitive, or
; 2. Has a definition path to primitives

(= (is-grounded? $concept)
   (if (is-primitive? $concept)
       True
       (has-grounding-path? $concept)))

; Check for grounding path (assumes (grounds $concept $primitive) atoms)
(= (has-grounding-path? $concept)
   (case (match &self (grounds $concept $target) $target)
     (($target (is-grounded? $target))
      (() False))))

; Grounding depth: primitives = 0, ungrounded = -1
(= (grounding-depth $concept)
   (if (is-primitive? $concept)
       0
       (let $targets (collapse (match &self (grounds $concept $t) $t))
         (if (== $targets ())
             -1
             (+ 1 (max-depth $targets))))))

(= (max-depth ()) 0)
(= (max-depth ($x)) (grounding-depth $x))
(= (max-depth ($x $y)) (max (grounding-depth $x) (grounding-depth $y)))
(= (max-depth ($x . $rest)) (max (grounding-depth $x) (max-depth $rest)))


; =============================================================================
; SECTION VII: INTEGRATION WITH BELIEFS
; =============================================================================
;
; Primitives serve as grounding for beliefs about meaning.
; When the system represents what a word means, the chain terminates here.

; Check if a meaning representation is fully grounded
(= (meaning-grounded? $meaning-expr)
   (all-grounded? (extract-concepts $meaning-expr)))

; Extract concept symbols from an expression (simplified)
(= (extract-concepts $x)
   (if (is-symbol? $x)
       ($x)
       (if (is-expression? $x)
           (flatten-concepts $x)
           ())))

(= (flatten-concepts ()) ())
(= (flatten-concepts ($h . $t))
   (let* (
     ($head-concepts (extract-concepts $h))
     ($tail-concepts (flatten-concepts $t))
   )
   (++ $head-concepts $tail-concepts)))

; Check that all concepts in a list are grounded
(= (all-grounded? ()) True)
(= (all-grounded? ($h . $t))
   (if (is-grounded? $h)
       (all-grounded? $t)
       False))


; =============================================================================
; END OF SEMANTIC PRIMITIVES (v3)
; =============================================================================
