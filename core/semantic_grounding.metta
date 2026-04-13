; =============================================================================
; SEMANTIC GROUNDING INTEGRATION
; =============================================================================
;
; THE PROBLEM THIS SOLVES:
;   The system selects "retreat" because compute-efe returns a low number.
;   But it doesn't KNOW what retreat means. The symbol could be "action-7"
;   and nothing would change. Beliefs about "power-level" are numbers, but
;   the system doesn't know power-level is a quantity the agent has, that
;   low power prevents action, or that power relates to "can" and "do."
;
;   Semantic primitives exist. Beliefs exist. Actions exist. Dimensions exist.
;   But they're islands. This module is the WIRING.
;
; WHAT THIS ENABLES:
;   1. "What am I doing?" â†’ structured answer through grounding chains
;   2. "Why does this belief matter?" â†’ traced to needs and primitives  
;   3. "What kind of thing is X?" â†’ categorical knowledge about observables
;   4. Dimensional reasoning about beliefs (is my power high or low?)
;   5. Self-narrative: not "action=retreat, EFE=-0.04" but "I am moving
;      away from what I believe is a threat because I believe I cannot
;      handle it, and I feel urgency"
;
; DESIGN PRINCIPLES:
;   - No new primitives. Uses existing ones from semantic_primitives.metta
;   - No new dimensions. Uses existing scales from dimensional_primitives.metta
;   - Grounding chains are DECLARED, not learned (this is where grounding
;     bottoms out â€” same justification as for the primitives themselves)
;   - Query functions COMPUTE meaning by traversal, not lookup
;   - The system reasons over these structures with the same pattern
;     matching it uses for everything else
;
; =============================================================================


; =============================================================================
; SECTION I: OBSERVABLE GROUNDING
; =============================================================================
;
; What ARE the observables? Each observable is connected to semantic
; primitives that define its nature. This is the system knowing what
; its own beliefs are ABOUT.
;
; Schema: (obs-grounding <observable> <relation> <primitive-or-concept>)
;
; These use relations from the primitives: is-a, has, part-of, causes,
; enables, prevents. The targets are primitives or grounded concepts.

; --- power-level ---
; "Power is a quantity that the agent has. It enables doing things.
;  When power is low, doing becomes impossible."
(obs-grounding power-level is-a property)
(obs-grounding power-level is-a quantity)
(obs-grounding power-level belongs-to self)
(obs-grounding power-level enables do)
(obs-grounding power-level enables can)
(obs-semantic  power-level "capacity of the agent to act")

; What happens at extremes?
(obs-low-means  power-level (prevents do))
(obs-low-means  power-level (associated fear))
(obs-high-means power-level (enables do))
(obs-high-means power-level (associated good))

; --- terrain-roughness ---
; "Roughness is a property of the place the agent is at.
;  High roughness makes movement difficult and predictions uncertain."
(obs-grounding terrain-roughness is-a property)
(obs-grounding terrain-roughness is-a quality)
(obs-grounding terrain-roughness belongs-to place)
(obs-grounding terrain-roughness part-of (at self))
(obs-semantic  terrain-roughness "how difficult the environment is to traverse")

(obs-low-means  terrain-roughness (enables do))
(obs-low-means  terrain-roughness (associated good))
(obs-high-means terrain-roughness (prevents do))
(obs-high-means terrain-roughness (associated fear))

; --- threat-level ---
; "Threat is a property of the situation. High threat causes fear
;  and may cause pain. It is something the agent wants to be low."
(obs-grounding threat-level is-a property)
(obs-grounding threat-level is-a quality)
(obs-grounding threat-level causes fear)
(obs-grounding threat-level causes pain)
(obs-semantic  threat-level "presence of danger to the agent")

(obs-low-means  threat-level (associated good))
(obs-low-means  threat-level (enables do))
(obs-high-means threat-level (causes pain))
(obs-high-means threat-level (associated fear))
(obs-high-means threat-level (prevents good))

; --- ambient-temperature ---
; "Temperature is a perceptual quality of the environment."
(obs-grounding ambient-temperature is-a property)
(obs-grounding ambient-temperature is-a quality)
(obs-grounding ambient-temperature belongs-to place)
(obs-semantic  ambient-temperature "thermal condition of the environment")

; Temperature is interesting: both extremes are bad
(obs-low-means  ambient-temperature (associated cold))
(obs-low-means  ambient-temperature (causes pain))
(obs-high-means ambient-temperature (associated hot))
(obs-high-means ambient-temperature (causes pain))


; =============================================================================
; SECTION II: ACTION GROUNDING
; =============================================================================
;
; What ARE the actions? Each action is connected to semantic primitives
; that define its nature. This is the system knowing what it IS DOING,
; not just which symbol was selected.
;
; Schema: (action-grounding <action> <relation> <primitive-or-concept>)

; --- wait ---
; "To wait is to do nothing. The agent stays in place.
;  It requires minimal energy. It does not change the world."
(action-grounding wait is-a do)
(action-grounding wait is-a event)
(action-grounding wait requires agent)
(action-grounding wait associated (not do))    ; Wait is the absence of doing
(action-grounding wait causes (same place))    ; Agent stays where it is
(action-semantic  wait "remaining in place without acting")

; --- observe ---
; "To observe is to sense. The agent looks at the world.
;  It causes knowing more. It uses some energy."
(action-grounding observe is-a do)
(action-grounding observe is-a sense)
(action-grounding observe requires agent)
(action-grounding observe causes know)         ; Observation produces knowledge
(action-grounding observe causes (more know))  ; Specifically: more knowledge
(action-grounding observe enables (less fear)) ; Knowing reduces fear
(action-semantic  observe "actively sensing the environment to reduce uncertainty")

; --- retreat ---
; "To retreat is to move away from threat. The agent changes place.
;  The goal is to be near safe and far from threat."
(action-grounding retreat is-a do)
(action-grounding retreat is-a event)
(action-grounding retreat requires agent)
(action-grounding retreat causes (different place))   ; Agent moves
(action-grounding retreat causes (away threat))       ; Specifically: away from threat
(action-grounding retreat causes (toward good))       ; Toward safety
(action-grounding retreat causes (less threat))       ; Results in less threat
(action-grounding retreat requires power-level)       ; Costs energy
(action-semantic  retreat "moving away from perceived danger toward safety")

; --- investigate (parameterized) ---
; "To investigate is to observe a specific thing. Focused sensing."
(action-grounding investigate is-a do)
(action-grounding investigate is-a sense)
(action-grounding investigate requires agent)
(action-grounding investigate causes know)
(action-grounding investigate is-a (part-of observe)) ; More specific than observe
(action-semantic  investigate "focused observation of a specific aspect")

; --- reflect ---
; "To reflect is to sense inward. The agent examines its own state —
;  its competence, its confidence, and what it does and doesn't understand.
;  It produces self-knowledge. It enables less uncertainty about self."
(action-grounding reflect is-a do)
(action-grounding reflect is-a sense)
(action-grounding reflect requires agent)
(action-grounding reflect causes (know self))
(action-grounding reflect causes (more know))
(action-grounding reflect enables (less uncertainty))
(action-grounding reflect associated (same place))   ; Internal action, no movement
(action-semantic  reflect "examining own state and capabilities to improve self-knowledge")

; --- ground-concept (parameterized) ---
; "To ground a concept is to find its meaning. The agent works to connect
;  a learned symbol to primitives it already understands. It produces
;  understanding of what a thing means."
(action-grounding ground-concept is-a do)
(action-grounding ground-concept is-a sense)
(action-grounding ground-concept requires agent)
(action-grounding ground-concept causes (know (meaning thing)))
(action-grounding ground-concept enables (more know))
(action-grounding ground-concept is-a (part-of reflect)) ; More specific than reflect
(action-semantic  ground-concept "actively working to understand the meaning of a learned concept")


; =============================================================================
; SECTION III: DOMAIN PREFERENCES (previously missing!)
; =============================================================================
;
; Preferences express what the agent WANTS (semantically: agent want X).
; Viability bounds express what the agent NEEDS (semantically: agent need X).
;
; These were referenced by actions.metta and policy_efe.metta but never
; declared for domain observables. The conversation_model and self_model
; declare their own. These belong here because they're part of the semantic
; definition of what these observables MEAN to the agent.
;
; Schema: (preference <observable> <preferred-value> <importance>)
; Schema: (viability-bound <observable> <min> <max>)

; Power: agent wants high power, needs minimum to survive
(preference power-level 0.9 1.0)           ; Want near-full power
(viability-bound power-level 0.15 1.0)     ; Below 0.15 = cannot act

; Terrain: agent prefers smooth terrain, can handle some roughness
(preference terrain-roughness 0.2 0.5)     ; Prefer smooth, moderate importance
(viability-bound terrain-roughness 0.0 0.85) ; Above 0.85 = impassable

; Threat: agent wants zero threat, high importance
(preference threat-level 0.1 0.9)          ; Want minimal threat
(viability-bound threat-level 0.0 0.8)     ; Above 0.8 = lethal

; Temperature: agent wants moderate, bounded both sides
(preference ambient-temperature 0.5 0.4)   ; Prefer moderate
(viability-bound ambient-temperature 0.1 0.9) ; Extremes lethal

; Connect preferences to semantic primitives
; "Why do I want power?" â†’ because power enables do, and I am an agent, and agents do.
(pref-grounding power-level want (because (enables do)))
(pref-grounding threat-level want (because (causes pain)))
(pref-grounding terrain-roughness want (because (enables do)))
(pref-grounding ambient-temperature want (because (prevents pain)))

; Connect viability bounds to semantic primitives
; "Why is this a hard limit?" â†’ because below this, agent cannot exist/act.
(viability-grounding power-level need (because (necessary (can do))))
(viability-grounding threat-level need (because (prevents existence)))
(viability-grounding terrain-roughness need (because (prevents (at place))))
(viability-grounding ambient-temperature need (because (prevents existence)))


; =============================================================================
; SECTION IV: DIMENSIONAL MAPPING
; =============================================================================
;
; Connect observables to the dimensional scale system.
; This enables the system to reason about beliefs in qualitative terms:
; "My power is LOW" rather than "power-level = 0.22"
;
; Schema: (obs-dimension <observable> <dimension> <scale> <polarity>)
;   polarity: higher-better or lower-better (for preference alignment)

; Define scales for agent observables (extending dimensional_primitives.metta)
(dimension agent-power power-scale)
(scale-value power-scale 1 critical)
(scale-value power-scale 2 low)
(scale-value power-scale 3 moderate)
(scale-value power-scale 4 high)
(scale-value power-scale 5 full)

(dimension environmental-threat threat-scale)
(scale-value threat-scale 1 safe)
(scale-value threat-scale 2 mild)
(scale-value threat-scale 3 moderate)
(scale-value threat-scale 4 dangerous)
(scale-value threat-scale 5 lethal)

(dimension surface-roughness roughness-scale)
(scale-value roughness-scale 1 smooth)
(scale-value roughness-scale 2 mild)
(scale-value roughness-scale 3 rough)
(scale-value roughness-scale 4 very-rough)
(scale-value roughness-scale 5 impassable)

(dimension thermal-condition temp-scale)
(scale-value temp-scale 1 freezing)
(scale-value temp-scale 2 cold)
(scale-value temp-scale 3 moderate)
(scale-value temp-scale 4 hot)
(scale-value temp-scale 5 extreme)

; Map observables to dimensions
(obs-dimension power-level agent-power power-scale higher-better)
(obs-dimension threat-level environmental-threat threat-scale lower-better)
(obs-dimension terrain-roughness surface-roughness roughness-scale lower-better)
(obs-dimension ambient-temperature thermal-condition temp-scale middle-better)


; =============================================================================
; SECTION V: BELIEF QUALIFICATION
; =============================================================================
;
; Convert numeric belief values to qualitative labels using dimensional
; scales. This is the bridge from "power-level = 0.22" to "power is LOW."
;
; The system can now reason: "I believe my power is low. Low power
; prevents doing. I am an agent. Agents do. Therefore my situation
; is bad."

; Convert a belief value (0.0-1.0) to a scale label
; Uses even quantile mapping: divides [0,1] into N equal bins
(= (value-to-label $val $scale)
   (let $max-rank (max-scale-rank $scale)
     (let $bin (+ 1 (min (- $max-rank 1) 
                         (floor (* $val $max-rank))))
       (match &self (scale-value $scale $bin $label) $label))))

; floor approximation (MeTTa doesn't have built-in floor)
(= (floor $x)
   (if (< $x 0.0) (- 0 (ceil (abs $x)))
     (let $int (to-int $x) $int)))

; Integer conversion (truncation toward zero)
(= (to-int $x)
   (if (< $x 1.0) 0
     (if (< $x 2.0) 1
       (if (< $x 3.0) 2
         (if (< $x 4.0) 3
           (if (< $x 5.0) 4
             5))))))

; Qualify a specific belief: returns (qualified <obs> <label> <precision>)
(= (qualify-belief $obs)
   (match &state (belief $obs $val $prec)
     (match &self (obs-dimension $obs $dim $scale $polarity)
       (let $label (value-to-label $val $scale)
         (qualified $obs $label $prec $polarity)))))

; Qualify all beliefs: returns list of qualitative descriptions
(= (qualify-all-beliefs)
   (collapse (match &state (belief $obs $val $prec)
     (match &self (obs-dimension $obs $dim $scale $pol)
       (let $label (value-to-label $val $scale)
         (qualified $obs $label $prec $pol))))))

; Is this belief value good or bad relative to preference?
; Returns: (evaluation <obs> <good|bad|neutral> <gap>)
(= (evaluate-belief $obs)
   (match &state (belief $obs $val $prec)
     (match &state (preference $obs $pref $imp)
       (let* (
         ($gap (- $pref $val))
         ($abs-gap (abs $gap))
         ($eval (if (< $abs-gap 0.1) neutral
                  (match &self (obs-dimension $obs $dim $scale $pol)
                    (if (== $pol higher-better)
                        (if (> $gap 0.0) bad good)
                        (if (== $pol lower-better)
                            (if (< $gap 0.0) bad good)
                            ; middle-better: bad if far from pref either way
                            (if (> $abs-gap 0.2) bad neutral))))))
       )
       (evaluation $obs $eval $gap $imp)))))


; =============================================================================
; SECTION VI: GROUNDING CHAIN QUERIES
; =============================================================================
;
; The core traversal functions. These let the system ask:
; "What IS X?" and get an answer in terms of semantic primitives.

; Get all groundings for an observable
(= (what-is-observable $obs)
   (collapse (match &state (obs-grounding $obs $rel $target) ($rel $target))))

; Get semantic description of an observable
(= (describe-observable $obs)
   (match &state (obs-semantic $obs $desc) $desc))

; Get all groundings for an action
(= (what-is-action $action)
   (collapse (match &state (action-grounding $action $rel $target) ($rel $target))))

; Get semantic description of an action
(= (describe-action $action)
   (match &state (action-semantic $action $desc) $desc))

; Why does the agent care about this observable?
(= (why-care-about $obs)
   (let* (
     ($pref-reason (match &self (pref-grounding $obs $rel $reason) ($rel $reason)))
     ($viab-reason (match &self (viability-grounding $obs $rel $reason) ($rel $reason)))
   )
   (care-about $obs
     (want $pref-reason)
     (need $viab-reason))))

; What semantic effects does an action have?
(= (action-effects $action)
   (collapse (match &state (action-grounding $action causes $effect) $effect)))

; What does an action require?
(= (action-requirements $action)
   (collapse (match &state (action-grounding $action requires $req) $req)))

; Is a concept grounded in a specific primitive?
; Recursive traversal: checks direct grounding and indirect chains
(= (grounded-in? $concept $primitive)
   (if (== $concept $primitive)
       True
       (let $targets (collapse (match &state (obs-grounding $concept $rel $target) $target))
         (any-grounded-in? $targets $primitive))))

; is-primitive? may come from v3 semantic_primitives or this fallback
; Fallback: check for (primitive (concept $sym) ...) atom in atomspace
(= (is-primitive-check? $sym)
   (let $r (collapse (match &self (primitive (concept $sym) $g) $g))
     (if (== $r ()) False True)))

(= (any-grounded-in? () $primitive) False)
(= (any-grounded-in? ($t . $rest) $primitive)
   (if (== $t $primitive)
       True
       (if (is-primitive-check? $t)
           (any-grounded-in? $rest $primitive)  ; Primitive: don't recurse deeper
           ; Non-primitive: check if it has grounding links
           (let $sub (collapse (match &state (obs-grounding $t $r $s) $s))
             (if (any-grounded-in? $sub $primitive)
                 True
                 (any-grounded-in? $rest $primitive))))))


; =============================================================================
; SECTION VII: SELF-NARRATIVE
; =============================================================================
;
; THE KEY FUNCTION. This connects action selection to the full semantic
; stack: what am I doing, why, what do I believe, what do I feel, and
; how does it all connect to the primitives?
;
; This is NOT an explanation for humans. It's a structured representation
; the system itself can reason over.

; What am I doing right now, and why?
(= (what-am-i-doing)
   (let* (
     ; What action was selected?
     ($action (get-current-action))
     
     ; What IS this action? (semantic grounding)
     ($action-nature (what-is-action $action))
     
     ; What does it cause?
     ($effects (action-effects $action))
     
     ; What does it require?
     ($requirements (action-requirements $action))
     
     ; What's my current affect? (why do I feel this way?)
     ($valence (compute-valence))
     ($arousal (compute-arousal))
     ($affect-quality (qualify-affect $valence $arousal))
     
     ; What beliefs drove this? (most surprising observables)
     ($drivers (belief-drivers))
     
     ; What are the qualified beliefs?
     ($qualified (qualify-all-beliefs))
   )
   (self-narrative
     (doing $action)
     (which-is $action-nature)
     (causing $effects)
     (requiring $requirements)
     (feeling $affect-quality)
     (because-i-believe $drivers)
     (my-situation $qualified))))

; Qualify affect in semantic terms (not numbers)
(= (qualify-affect $valence $arousal)
   (let* (
     ($v-quality (if (< $valence -0.3) pain
                   (if (< $valence 0.0) (not good)
                     (if (< $valence 0.3) (not bad)
                       good))))
     ($a-quality (if (> $arousal 0.7) (strong feel)
                   (if (> $arousal 0.4) (moderate feel)
                     (weak feel))))
   )
   (affect $v-quality $a-quality)))

; What beliefs are driving current behavior?
; Returns the beliefs with largest weighted errors (most surprising)
(= (belief-drivers)
   (let $errors (all-prediction-errors)
     (top-errors $errors)))

; Extract top prediction errors (simplified: return all with error > threshold)
(= (top-errors ()) ())
(= (top-errors ((prediction-error $obs $err $abs-err $prec $we) . $rest))
   (if (> $abs-err 0.15)
       ((drives $obs (error $abs-err)) . (top-errors $rest))
       (top-errors $rest)))


; =============================================================================
; SECTION VIII: SEMANTIC QUERIES (for self-model)
; =============================================================================
;
; These extend the self-model's query interface with semantic awareness.

; "Can I do X?" with semantic explanation
(= (why-can-i-do $action)
   (let* (
     ($reqs (action-requirements $action))
     ($met (check-requirements $reqs))
   )
   (capability-explanation $action $reqs $met)))

; Check if requirements are met
(= (check-requirements ()) (all-met))
(= (check-requirements ($req . $rest))
   (let $status (requirement-status $req)
     (($req $status) . (check-requirements $rest))))

; Check single requirement
(= (requirement-status agent) met)           ; We ARE an agent
(= (requirement-status power-level)          ; Check if power is adequate
   (let $val (get-belief-value power-level)
     (if (> $val 0.3) met (unmet $val))))
(= (requirement-status $other) unknown)

; "What do I know about X?" with semantic context
(= (knowledge-about $obs)
   (let* (
     ($has-belief (has-belief? $obs))
     ($val (if $has-belief (get-belief-value $obs) unknown))
     ($prec (if $has-belief (get-belief-precision $obs) unknown))
     ($qual (if $has-belief (qualify-belief $obs) unknown))
     ($nature (what-is-observable $obs))
     ($why-matters (why-care-about $obs))
   )
   (knowledge-report
     (about $obs)
     (nature $nature)
     (current-belief $val)
     (confidence $prec)
     (qualitative $qual)
     (matters-because $why-matters))))


; =============================================================================
; SECTION IX: CROSS-MODULE REASONING
; =============================================================================
;
; Semantic grounding enables new kinds of inference that weren't possible
; when modules were isolated.

; "Which of my beliefs are about properties of SELF vs properties of PLACE?"
(= (self-beliefs)
   (collapse (match &state (obs-grounding $obs belongs-to self)
     (match &state (belief $obs $val $prec) (belief $obs $val $prec)))))

(= (place-beliefs)
   (collapse (match &state (obs-grounding $obs belongs-to place)
     (match &state (belief $obs $val $prec) (belief $obs $val $prec)))))

; "Which actions CAUSE the thing I'm trying to avoid?"
; If threat is high and causes pain, find actions that reduce threat
(= (actions-that-reduce $obs)
   (collapse (match &state (action-model $action $obs $vd $pd $conf)
     (if (< $vd 0.0) $action (empty)))))

; "Which actions help with what I NEED most?"
(= (most-needed-action)
   (let* (
     ; Find the observable closest to viability bound
     ($critical (most-critical-observable))
   )
   (match $critical (critical $obs $margin)
     (actions-that-help $obs))))

; Find observable closest to its viability bound
(= (most-critical-observable)
   (let $candidates (collapse 
     (match &state (viability-bound $obs $min $max)
       (match &state (belief $obs $val $prec)
         (let* (
           ($low-margin (- $val $min))
           ($high-margin (- $max $val))
           ($min-margin (min $low-margin $high-margin))
         )
         (critical $obs $min-margin)))))
     (find-min-margin $candidates)))

(= (find-min-margin ((critical $obs $margin))) (critical $obs $margin))
(= (find-min-margin ((critical $o1 $m1) . $rest))
   (let (critical $o2 $m2) (find-min-margin $rest)
     (if (< $m1 $m2)
         (critical $o1 $m1)
         (critical $o2 $m2))))

; "Which actions help (move value TOWARD preference) for this observable?"
(= (actions-that-help $obs)
   (match &state (preference $obs $pref $imp)
     (match &state (belief $obs $val $prec)
       (let $gap (- $pref $val)
         (collapse (match &state (action-model $action $obs $vd $pd $conf)
           (if (> (* $gap $vd) 0.0)    ; vd moves in same direction as gap
               (helps $action $obs (delta $vd) (confidence $conf))
               (empty))))))))


; =============================================================================
; SECTION X: INTEGRATION WITH COGNITIVE CYCLE
; =============================================================================
;
; These functions can be called from cycle.metta to add semantic
; awareness to the cognitive loop without changing the EFE machinery.

; Post-action semantic snapshot (for logging/introspection)
(= (semantic-snapshot!)
   (let* (
     ($narrative (what-am-i-doing))
     ($situation (qualify-all-beliefs))
     ($critical (most-critical-observable))
   )
   (semantic-state $narrative $situation $critical)))

; Rich explanation combining EFE trace with semantic grounding
(= (explain-with-meaning)
   (let* (
     ($action (get-current-action))
     ($efe-explanation (explain))
     ($semantic (what-am-i-doing))
     ($eval (collapse (match &state (belief $obs $val $prec)
       (match &state (preference $obs $pref $imp)
         (evaluate-belief $obs)))))
   )
   (rich-explanation
     (efe $efe-explanation)
     (meaning $semantic)
     (evaluations $eval))))


; =============================================================================
; SECTION XI: CONVERSATIONAL OBSERVABLE GROUNDING
; =============================================================================
;
; The conversation model's observables also need semantic grounding.
; This connects partner-comprehension, rapport, etc. to primitives.

(obs-grounding partner-comprehension is-a property)
(obs-grounding partner-comprehension belongs-to (relation self other))
(obs-grounding partner-comprehension is-a know)
(obs-semantic  partner-comprehension "whether the partner understands us")

(obs-low-means  partner-comprehension (not know))
(obs-low-means  partner-comprehension (frustrate want))
(obs-high-means partner-comprehension (know same))
(obs-high-means partner-comprehension (satisfy want))

(obs-grounding discourse-coherence is-a property)
(obs-grounding discourse-coherence belongs-to (relation self other))
(obs-grounding discourse-coherence is-a (quality good))
(obs-semantic  discourse-coherence "logical consistency of the conversation")

(obs-grounding goal-progress is-a property)
(obs-grounding goal-progress is-a (toward want))
(obs-semantic  goal-progress "progress toward the purpose of the conversation")

(obs-grounding rapport is-a property)
(obs-grounding rapport belongs-to (relation self other))
(obs-grounding rapport is-a (quality good))
(obs-semantic  rapport "quality of interpersonal connection")

; Conversational action grounding
(action-grounding clarify is-a do)
(action-grounding clarify causes know)
(action-grounding clarify causes (more know))
(action-semantic  clarify "restating or explaining to improve understanding")

(action-grounding assert is-a do)
(action-grounding assert causes (know other))
(action-semantic  assert "making a claim or sharing information")

(action-grounding query-partner is-a do)
(action-grounding query-partner is-a sense)
(action-grounding query-partner causes (know self))
(action-semantic  query-partner "asking the partner to provide information")

(action-grounding acknowledge is-a do)
(action-grounding acknowledge causes (know other (know self)))
(action-semantic  acknowledge "signaling that we received and understood")

(action-grounding elaborate is-a do)
(action-grounding elaborate causes (more know))
(action-semantic  elaborate "providing additional detail on a previous point")

(action-grounding redirect is-a do)
(action-grounding redirect causes (different (toward want)))
(action-semantic  redirect "changing the topic or direction of conversation")


; =============================================================================
; SECTION XII: DIMENSIONAL MAPPING FOR CONVERSATION
; =============================================================================

(dimension conversational-understanding understanding-scale)
(scale-value understanding-scale 1 lost)
(scale-value understanding-scale 2 confused)
(scale-value understanding-scale 3 partial)
(scale-value understanding-scale 4 good)
(scale-value understanding-scale 5 excellent)

(dimension conversational-rapport rapport-scale)
(scale-value rapport-scale 1 hostile)
(scale-value rapport-scale 2 cold)
(scale-value rapport-scale 3 neutral)
(scale-value rapport-scale 4 warm)
(scale-value rapport-scale 5 strong)

(obs-dimension partner-comprehension conversational-understanding understanding-scale higher-better)
(obs-dimension rapport conversational-rapport rapport-scale higher-better)


; =============================================================================
; END OF SEMANTIC GROUNDING INTEGRATION
; =============================================================================
