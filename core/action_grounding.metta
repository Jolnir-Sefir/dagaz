; =============================================================================
; ACTION GROUNDING CHAINS
; =============================================================================
;
; THE PROBLEM:
;   (action-grounding retreat causes (away threat))
;   
;   "threat" is an ungrounded token in that expression. The system can
;   collect the fact, but can't decompose it. It doesn't know that
;   "threat" connects to threat-level (an observable), which connects
;   to the current belief (0.65), which qualifies as "dangerous," which
;   causes fear, which is what the agent is feeling.
;
;   The chain is: retreat â†’ away â†’ threat â†’ threat-level â†’ belief â†’ 
;                 qualify â†’ dangerous â†’ causes fear â†’ check affect â†’ yes
;
;   Without this chain, the system is the Chinese Room. With it,
;   every token in the action's grounding resolves to either a primitive
;   (termination point) or a live system state (belief, affect, self).
;
; WHAT THIS ADDS:
;   1. Concept vocabulary: bridge tokens between action-grounding and
;      primitives/observables (threat, safety, danger, movement, etc.)
;   2. Chain traversal: recursive decomposition that BINDS to current state
;   3. Semantic self-queries: "Am I acting from fear?" answered by joining
;      grounding chains with beliefs and affect
;   4. Contrastive explanation: "Why retreat, not observe?" answered by
;      comparing grounded chains
;
; DESIGN:
;   - Concept grounding uses same (concept-ground X rel Y) pattern as
;     obs-grounding and action-grounding. Single pattern for everything.
;   - Traversal terminates at: primitives, bound observables, or self.
;   - State binding happens AT TRAVERSAL TIME, not declaration time.
;     The chain is a template; the grounded chain is an instantiation.
;   - No new mechanisms. Pattern matching and recursion. Same tools
;     the system uses for everything else.
;
; INVARIANT:
;   A term is GROUNDED iff it is:
;     (a) a semantic primitive, OR
;     (b) an observable with a current belief, OR
;     (c) decomposable into terms that are themselves grounded
;   An UNGROUNDED term is a gap in understanding.
;
; =============================================================================


; =============================================================================
; SECTION I: CONCEPT VOCABULARY
; =============================================================================
;
; These are the intermediate concepts that appear in action-grounding
; facts but are neither primitives nor observables. Each one decomposes
; into primitives and/or links to observables.
;
; Schema: (concept-ground <concept> <relation> <target>)
;
; The relation vocabulary is the same as everywhere else: is-a, causes,
; contrast, associated, bound-to, part-of, enables, prevents.
;
; "bound-to" is the key new relation: it connects a concept to the
; observable that carries its current value in the belief system.

; --- threat ---
; "Threat is something in the environment that can cause harm to the agent.
;  Its current magnitude is carried by the threat-level observable."
(concept-ground threat is-a property)
(concept-ground threat causes fear)
(concept-ground threat causes pain)
(concept-ground threat prevents good)
(concept-ground threat bound-to threat-level)

; --- safety ---
; "Safety is the absence of threat. It enables action and is associated
;  with positive affect."
(concept-ground safety contrast threat)
(concept-ground safety is-a property)
(concept-ground safety enables do)
(concept-ground safety associated good)
(concept-ground safety bound-to-inverse threat-level)

; --- danger ---
; "Danger is high threat. It's a qualitative state of the threat concept."
(concept-ground danger is-a (more threat))
(concept-ground danger causes fear)
(concept-ground danger associated (strong feel))
(concept-ground danger bound-to threat-level)

; --- movement ---
; "Movement is change of place by an agent."
(concept-ground movement is-a event)
(concept-ground movement is-a do)
(concept-ground movement causes (different place))
(concept-ground movement requires agent)

; --- energy / power (as concept, not observable) ---
; "Energy is the capacity to do. Its magnitude is power-level."
(concept-ground energy is-a property)
(concept-ground energy belongs-to self)
(concept-ground energy enables do)
(concept-ground energy bound-to power-level)

; --- difficulty ---
; "Difficulty is what makes doing harder. Carried by terrain-roughness."
(concept-ground difficulty is-a property)
(concept-ground difficulty prevents do)
(concept-ground difficulty bound-to terrain-roughness)

; --- comfort ---
; "Comfort is the absence of environmental stress. Related to temperature."
(concept-ground comfort is-a property)
(concept-ground comfort associated good)
(concept-ground comfort bound-to-inverse ambient-temperature)
; (actually middle-better, but conceptually: comfort = not-extreme)

; --- knowledge (as concept) ---
; "Knowledge is what the agent has about the world. Precision is its measure."
(concept-ground knowledge is-a property)
(concept-ground knowledge belongs-to self)
(concept-ground knowledge is-a know)
(concept-ground knowledge associated good)

; --- uncertainty (as concept) ---
; "Uncertainty is lack of knowledge. It's what drives observation."
(concept-ground uncertainty contrast knowledge)
(concept-ground uncertainty is-a (not know))
(concept-ground uncertainty causes fear)
(concept-ground uncertainty associated bad)


; =============================================================================
; SECTION II: TERM STATUS
; =============================================================================
;
; Before traversing, we need to know what kind of thing a term is:
;   - primitive: termination point, fully grounded by definition
;   - observable: binds to a belief, grounded by current state
;   - concept: decomposes via concept-ground links
;   - compound: a structured term like (away threat) that must be
;     decomposed element by element
;   - unknown: no grounding path exists â€” a gap in understanding

; Check if a symbol is a declared primitive
; Matches against (primitive (concept $sym) $g) — the (concept ...) wrapper
; separates mentioning a concept from using a same-named operator.
(= (is-declared-primitive? $sym)
   (let $r (collapse (match &self (primitive (concept $sym) $g) $g))
     (if (== $r ()) False True)))

; Check if a symbol is a known observable
(= (is-observable? $sym)
   (let $r (collapse (match &self (obs-grounding $sym $rel $target) $sym))
     (if (== $r ()) False True)))

; Check if a symbol is a grounded concept
(= (is-concept? $sym)
   (let $r (collapse (match &self (concept-ground $sym $rel $target) $sym))
     (if (== $r ()) False True)))

; Check if a symbol is a known action
(= (is-action? $sym)
   (let $r (collapse (match &self (action-grounding $sym $rel $target) $sym))
     (if (== $r ()) False True)))

; Classify a term
(= (term-status $term)
   (if (is-declared-primitive? $term)
       primitive
       (if (is-observable? $term)
           observable
           (if (is-concept? $term)
               concept
               (if (is-action? $term)
                   action
                   unknown)))))


; =============================================================================
; SECTION III: CHAIN TRAVERSAL WITH STATE BINDING
; =============================================================================
;
; The core mechanism. Given a term, produce its grounding:
;   - Primitive â†’ (grounded <prim>)
;   - Observable â†’ (bound <obs> <value> <precision> <qualitative>)
;   - Concept â†’ (decomposed <concept> (<rel> <grounded-target>) ...)
;   - Compound â†’ (compound (<grounded-part> ...))
;   - Unknown â†’ (ungrounded <term>)
;
; State binding: when traversal hits an observable, it reads the CURRENT
; belief value and qualifies it. This is what makes the chain live â€”
; it reflects the system's actual state at the moment of introspection.

; Ground a single term against current state
(= (ground-term $term)
   (case (term-status $term)
     ((primitive (grounded $term))
      (observable (bind-observable $term))
      (concept (decompose-concept $term))
      (action (grounded-action-ref $term))
      (unknown (ungrounded $term)))))

; Reference to an action (grounded because it has action-grounding)
(= (grounded-action-ref $action)
   (let $desc (collapse (match &self (action-semantic $action $d) $d))
     (action-ref $action $desc)))

; Bind an observable to its current belief state
(= (bind-observable $obs)
   (let $belief-result (collapse (match &state (belief $obs $val $prec) ($val $prec)))
     (if (== $belief-result ())
         ; No belief exists â€” we know OF this observable but have no data
         (bound $obs no-belief no-belief unknown)
         ; Belief exists â€” bind and qualify
         (let ($val $prec) (car-safe $belief-result)
           (let $qual-result (collapse (qualify-belief $obs))
             (if (== $qual-result ())
                 (bound $obs $val $prec unqualified)
                 (let $qual (car-safe $qual-result)
                   (bound $obs $val $prec $qual))))))))

; Safely extract first element from a list
(= (car-safe ($x . $rest)) $x)
(= (car-safe ($x)) $x)
(= (car-safe ()) ())

; Decompose a concept into its grounding links, recursively grounding targets
(= (decompose-concept $concept)
   (let $links (collapse (match &self (concept-ground $concept $rel $target) ($rel $target)))
     (decomposed $concept (ground-links $links))))

; Ground each link target (one level â€” not infinite recursion)
; We ground the TARGET of each link, not the link itself
(= (ground-links ()) ())
(= (ground-links (($rel $target) . $rest))
   ((link $rel (ground-term-shallow $target)) . (ground-links $rest)))

; Shallow grounding: one level of decomposition
; For chain traversal we go one level deep on targets to avoid
; infinite recursion through circular concept references.
; Primitives and observables resolve fully; concepts show their links but
; don't recursively expand.
(= (ground-term-shallow $term)
   (case (term-status $term)
     ((primitive (grounded $term))
      (observable (bind-observable $term))
      (concept (concept-summary $term))
      (action (grounded-action-ref $term))
      (unknown (ungrounded $term)))))

; Concept summary: name + what it's bound to (without full decomposition)
(= (concept-summary $concept)
   (let $bound (collapse (match &self (concept-ground $concept bound-to $obs) $obs))
     (if (== $bound ())
         (concept-ref $concept)
         (concept-ref $concept (binds-to (car-safe $bound))))))


; =============================================================================
; SECTION IV: FULL ACTION GROUNDING CHAIN
; =============================================================================
;
; This is the main entry point. Given an action, produce a fully grounded
; representation that connects the action to primitives, beliefs, and affect.
;
; The output is a structure the system can REASON OVER â€” not a string,
; not a description, but a queryable tree of grounded terms.

(= (ground-action $action)
   (let* (
     ; 1. What IS this action? (nature: is-a links)
     ($nature-links (collapse 
       (match &self (action-grounding $action is-a $target) $target)))
     ($grounded-nature (ground-term-list $nature-links))

     ; 2. What does it CAUSE? (effects)
     ($effect-links (collapse 
       (match &self (action-grounding $action causes $target) $target)))
     ($grounded-effects (ground-term-list $effect-links))

     ; 3. What does it REQUIRE? (prerequisites)
     ($req-links (collapse 
       (match &self (action-grounding $action requires $target) $target)))
     ($grounded-reqs (ground-term-list $req-links))

     ; 4. What does it ENABLE? 
     ($enable-links (collapse 
       (match &self (action-grounding $action enables $target) $target)))
     ($grounded-enables (ground-term-list $enable-links))

     ; 5. Current affect (live binding)
     ($valence (compute-valence))
     ($arousal (compute-arousal))
     ($affect-qual (qualify-affect $valence $arousal))

     ; 6. Semantic description
     ($desc (collapse (match &self (action-semantic $action $d) $d)))
   )
   (action-chain
     (action $action)
     (nature $grounded-nature)
     (causes $grounded-effects)
     (requires $grounded-reqs)
     (enables $grounded-enables)
     (affect $affect-qual)
     (description $desc))))

; Ground a list of terms
(= (ground-term-list ()) ())
(= (ground-term-list ($term . $rest))
   ((ground-term $term) . (ground-term-list $rest)))


; =============================================================================
; SECTION V: SEMANTIC SELF-QUERIES
; =============================================================================
;
; These are the payoff. The system can now ask questions about its own
; actions in semantic terms and get answers by joining grounding chains
; with current state.
;
; Each query is a COMPUTATION, not a lookup. The answer emerges from
; traversing the grounding structure and checking against live state.

; --- "Am I acting from fear?" ---
; Two independent paths to "yes":
;   1. Motivational: action chain involves threat AND threat belief is high
;      (calm retreat from known danger IS fear-motivated)
;   2. Emotional: action chain involves threat AND high arousal
;      (panic response to surprising threat)
; Either path suffices. This distinguishes motivation from emotion.
(= (acting-from-fear?)
   (let* (
     ($action (get-current-action))
     ($effects (collapse (match &self (action-grounding $action causes $e) $e)))
     ($has-threat-link (mentions-threat? $effects))
     ($valence (compute-valence))
     ($arousal (compute-arousal))
     ; Motivational path: check if threat-level belief is high
     ($threat-val (get-belief-value-safe threat-level))
     ($threat-high (> $threat-val 0.5))
     ; Emotional path: high arousal
     ($emotional (> $arousal 0.4))
   )
   (if (and $has-threat-link (or $threat-high $emotional))
       (yes (because
         (action-involves threat)
         (threat-belief $threat-val)
         (feeling (valence $valence) (arousal $arousal))
         (path (if $threat-high 
                   (if $emotional both motivational)
                   emotional))))
       (no (because
         (threat-link $has-threat-link)
         (threat-belief $threat-val)
         (valence $valence)
         (arousal $arousal))))))

; Check if any term in a list mentions threat
(= (mentions-threat? ()) False)
(= (mentions-threat? ($term . $rest))
   (if (term-involves-threat? $term)
       True
       (mentions-threat? $rest)))

; Does this term involve threat? Check recursively for compound terms.
(= (term-involves-threat? threat) True)
(= (term-involves-threat? (away threat)) True)
(= (term-involves-threat? (less threat)) True)
(= (term-involves-threat? (toward threat)) True)
(= (term-involves-threat? (more threat)) True)
; General compound check
(= (term-involves-threat? ($a $b))
   (or (term-involves-threat? $a) (term-involves-threat? $b)))
; Base case for non-matching atoms
(= (term-involves-threat? $other) False)


; --- "Am I acting to gain knowledge?" ---
; True iff: current action is-a sense OR causes know/more-know
(= (acting-to-learn?)
   (let* (
     ($action (get-current-action))
     ($is-sense (collapse (match &self (action-grounding $action is-a sense) True)))
     ($causes-know (collapse (match &self (action-grounding $action causes know) True)))
   )
   (if (or (not (== $is-sense ())) (not (== $causes-know ())))
       (yes (because (action-is sense) (causes knowledge)))
       (no))))


; --- "Am I changing place?" ---
; True iff: current action causes (different place) or causes movement
(= (acting-to-move?)
   (let* (
     ($action (get-current-action))
     ($moves (collapse 
       (match &self (action-grounding $action causes (different place)) True)))
   )
   (if (not (== $moves ()))
       (yes (because (causes (different place))))
       (no))))


; --- "Why this action, not that one?" ---
; Contrastive explanation. Compares grounded chains of two actions.
; The answer highlights: what the chosen action causes that the alternative
; doesn't, and vice versa.
(= (why-not? $chosen $alternative)
   (let* (
     ; What each action causes (semantic level)
     ($chosen-effects (collapse 
       (match &self (action-grounding $chosen causes $e) $e)))
     ($alt-effects (collapse 
       (match &self (action-grounding $alternative causes $e) $e)))
     
     ; What each action IS
     ($chosen-nature (collapse 
       (match &self (action-grounding $chosen is-a $n) $n)))
     ($alt-nature (collapse 
       (match &self (action-grounding $alternative is-a $n) $n)))

     ; EFE values (the numeric justification)
     ($chosen-efe (compute-efe $chosen))
     ($alt-efe (compute-efe $alternative))
     
     ; Current situation context
     ($pressure (get-viability-pressure))
     ($affect-v (compute-valence))
   )
   (contrastive-explanation
     (chose $chosen (efe $chosen-efe)
       (which-is $chosen-nature)
       (causing $chosen-effects))
     (over $alternative (efe $alt-efe)
       (which-is $alt-nature)
       (causing $alt-effects))
     (context 
       (viability-pressure $pressure)
       (valence $affect-v)))))


; --- "What do I need most right now?" ---
; Returns the most critical observable with its semantic grounding:
; not just "power-level is near bound" but "my capacity to act is
; critically low, which means I cannot do things"
(= (what-do-i-need)
   (let* (
     ($critical (most-critical-observable))
   )
   (match $critical (critical $obs $margin)
     (let* (
       ($grounding (what-is-observable $obs))
       ($why (why-care-about $obs))
       ($qual (qualify-belief $obs))
       ($helpful (actions-that-help $obs))
     )
     (need-report
       (most-critical $obs)
       (margin $margin)
       (which-is $grounding)
       (matters-because $why)
       (current-state $qual)
       (can-help-by $helpful))))))


; --- "What do I understand about what I'm doing?" ---
; The comprehensive self-knowledge query. Returns everything the system
; knows about its current action, grounded all the way down.
; This is the full "knowing what retreat means" from the design doc.
(= (full-self-understanding)
   (let* (
     ($action (get-current-action))
     ($chain (ground-action $action))
     ($from-fear (acting-from-fear?))
     ($to-learn (acting-to-learn?))
     ($moving (acting-to-move?))
     ($need (what-do-i-need))
   )
   (self-understanding
     (grounded-action $chain)
     (acting-from-fear $from-fear)
     (acting-to-learn $to-learn)
     (acting-to-move $moving)
     (primary-need $need))))


; =============================================================================
; SECTION VI: GROUNDING COMPLETENESS CHECK
; =============================================================================
;
; Diagnostic: are there any ungrounded terms in the system?
; An ungrounded term is a token that appears in action-grounding or
; concept-ground but has no path to primitives or observables.
; These are gaps in the system's understanding.

; Find all terms used in action groundings
(= (all-grounding-targets)
   (collapse (match &self (action-grounding $a $r $t) $t)))

; Check which targets are ungrounded
(= (find-ungrounded)
   (let $targets (all-grounding-targets)
     (filter-ungrounded $targets)))

(= (filter-ungrounded ()) ())
(= (filter-ungrounded ($term . $rest))
   (let $status (term-status-deep $term)
     (if (== $status ungrounded)
         ($term . (filter-ungrounded $rest))
         (filter-ungrounded $rest))))

; Deep status check for compound terms
(= (term-status-deep $term)
   (if (is-declared-primitive? $term) primitive
     (if (is-observable? $term) observable
       (if (is-concept? $term) concept
         (if (is-action? $term) action
           ; Compound term: check if it's built from known parts
           ; (away threat) is grounded if both away and threat are
           (if (== (term-status-deep-compound $term) grounded)
               compound-grounded
               ungrounded))))))

; For compound terms like (away threat), check each component
(= (term-status-deep-compound ($a $b))
   (if (and (not (== (term-status-deep $a) ungrounded))
            (not (== (term-status-deep $b) ungrounded)))
       grounded
       ungrounded))
(= (term-status-deep-compound ($a $b $c))
   (if (and (not (== (term-status-deep $a) ungrounded))
            (and (not (== (term-status-deep $b) ungrounded))
                 (not (== (term-status-deep $c) ungrounded))))
       grounded
       ungrounded))
; Atoms that aren't compound â€” if we got here, it's ungrounded
(= (term-status-deep-compound $atom) ungrounded)


; =============================================================================
; SECTION VII: INTEGRATION WITH EXISTING what-am-i-doing
; =============================================================================
;
; The existing what-am-i-doing in semantic_grounding.metta collects flat
; facts. This enhanced version grounds them.
;
; We don't replace what-am-i-doing â€” we add a deeper version that
; uses grounding chains. The original is still useful for lightweight
; introspection. This one is for when the system needs to truly
; understand what it's doing.

(= (what-am-i-doing-grounded)
   (let* (
     ($action (get-current-action))
     
     ; Full grounding chain (the new thing)
     ($chain (ground-action $action))
     
     ; Affect with semantic qualification  
     ($valence (compute-valence))
     ($arousal (compute-arousal))
     ($affect (qualify-affect $valence $arousal))
     
     ; What beliefs are driving this? (from semantic_grounding.metta)
     ($drivers (belief-drivers))
     
     ; What's my most critical need?
     ($critical (most-critical-observable))
     
     ; Semantic self-queries
     ($from-fear (acting-from-fear?))
     ($to-learn (acting-to-learn?))
   )
   (grounded-narrative
     (i-am-doing $action)
     (which-means $chain)
     (i-feel $affect)
     (driven-by $drivers)
     (most-critical $critical)
     (from-fear $from-fear)
     (to-learn $to-learn))))


; =============================================================================
; END OF ACTION GROUNDING CHAINS
; =============================================================================
