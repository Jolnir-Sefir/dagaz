; =============================================================================
; DOMAIN CONFIGURATION — Define Your AI's World Here
; =============================================================================
;
; This module is the single place to define what your AI agent knows about
; its environment. Edit the declarations below, load this module after
; the core, and call !(init-domain!). The cognitive architecture adapts
; automatically — no changes needed to any core module.
;
; WHAT TO DEFINE:
;   domain-action       — what the agent can do
;   domain-observable    — what the agent can sense (with initial belief)
;   domain-preference    — what the agent wants (target values)
;   domain-viability     — hard survival constraints
;   domain-action-model  — how actions affect observables
;   domain-action-cost   — energy cost of each action
;
; LOAD ORDER: This module must load AFTER all core modules.
; The !(init-domain!) call at the bottom processes the declarations.
;
; FOR THE GAME: dagaz_server.py can override these from the Godot init
; message. This file provides standalone defaults and serves as the
; canonical schema documentation.
;
; =============================================================================


; =============================================================================
; SECTION I: ACTION DECLARATIONS
; =============================================================================
;
; What can the agent do? One (domain-action <name>) per action.
; These become (base-actions) and get registered for policy generation.

(domain-action wait)
(domain-action observe)
(domain-action retreat)


; =============================================================================
; SECTION II: OBSERVABLE DECLARATIONS
; =============================================================================
;
; What can the agent sense? Each observable gets an initial belief.
;   (domain-observable <name> <initial-value> <initial-precision>)
;
; Precision is confidence in the belief (0.0 = total ignorance, 1.0 = certain).
; Start low if the agent should learn from observation.

(domain-observable power-level 0.5 0.3)
(domain-observable terrain-roughness 0.5 0.2)


; =============================================================================
; SECTION III: PREFERENCE DECLARATIONS
; =============================================================================
;
; What does the agent want? Target values with importance weights.
;   (domain-preference <observable> <target-value> <importance>)
;
; The agent minimizes the gap between beliefs and preferences.
; Higher importance = stronger drive to reach the target.

(domain-preference power-level 0.9 1.0)
(domain-preference terrain-roughness 0.3 0.8)


; =============================================================================
; SECTION IV: VIABILITY BOUNDS
; =============================================================================
;
; Hard survival constraints. Cross these and the agent treats it as
; catastrophic — viability pressure dominates action selection.
;   (domain-viability <observable> <min> <max>)
;
; These are NOT preferences. They are structural limits on what the
; agent can survive. Battery below 10% means the drone can't function.

(domain-viability power-level 0.10 1.0)


; =============================================================================
; SECTION V: ACTION MODELS
; =============================================================================
;
; How does each action affect each observable?
;   (domain-action-model <action> <observable> <value-delta> <prec-delta> <confidence>)
;
;   value-delta: expected change in belief value when action is taken
;   prec-delta:  expected change in belief precision (+ = more certain)
;   confidence:  how much the agent trusts this model (0.0-1.0)
;
; Start confidence low (0.3-0.5) to let the agent learn from experience.
; The metabolic selection mechanism will strengthen accurate models and
; kill inaccurate ones.

(domain-action-model wait power-level -0.01 0.0 0.5)
(domain-action-model wait terrain-roughness 0.0 -0.02 0.5)
(domain-action-model observe power-level -0.02 0.0 0.5)
(domain-action-model observe terrain-roughness 0.0 0.12 0.5)
(domain-action-model retreat power-level -0.05 0.0 0.5)
(domain-action-model retreat terrain-roughness -0.10 0.05 0.5)


; =============================================================================
; SECTION VI: ACTION COSTS
; =============================================================================
;
; Energy cost of taking each action.
;   (domain-action-cost <action> <cost> <confidence>)
;
; Cost enters the EFE formula directly — higher cost actions are
; disfavored unless they provide enough information gain or
; viability protection to compensate.

(domain-action-cost wait 0.01 0.5)
(domain-action-cost observe 0.05 0.5)
(domain-action-cost retreat 0.12 0.5)


; =============================================================================
; SECTION VII: DOMAIN INITIALIZATION (Do Not Edit Below This Line)
; =============================================================================
;
; The functions below process the declarations above and configure
; the cognitive core. They run once at boot time.

; --- Collect all declared domain actions ---
(= (domain-actions)
   (collapse (match &self (domain-action $a) $a)))

; --- Rewrite (base-actions) to use domain declarations ---
; Removes the hardcoded default from actions.metta and installs
; a version that reads from domain-action declarations.
(= (install-domain-base-actions!)
   (let $actions (domain-actions)
     (if (list-empty? $actions)
         (domain-error no-actions-declared)
         done)))

; --- Register domain actions for policy generation ---
; Clears any existing (any-action) entries and re-registers
; from the domain's action list.
(= (clear-any-action-registry!) done)
; This is handled Python-side or via the loader since we can't
; easily pattern-match on (= (any-action) $x) from MeTTa.

; --- Initialize beliefs from domain observables ---
(= (init-domain-beliefs! ()) done)
(= (init-domain-beliefs! (($o $v $p) . $rest))
   (sequential
     (add-atom &state (belief $o $v $p))
     (init-domain-beliefs! $rest)))

; --- Initialize preferences ---
(= (init-domain-preferences! ()) done)
(= (init-domain-preferences! (($o $t $i) . $rest))
   (sequential
     (add-atom &state (preference $o $t $i))
     (init-domain-preferences! $rest)))

; --- Initialize viability bounds ---
(= (init-domain-viability! ()) done)
(= (init-domain-viability! (($o $min $max) . $rest))
   (sequential
     (add-atom &state (viability-bound $o $min $max))
     (init-domain-viability! $rest)))

; --- Initialize action models ---
(= (init-domain-models! ()) done)
(= (init-domain-models! (($a $o $vd $pd $c) . $rest))
   (sequential
     (add-atom &state (action-model $a $o $vd $pd $c))
     (init-domain-models! $rest)))

; --- Initialize action costs ---
(= (init-domain-costs! ()) done)
(= (init-domain-costs! (($a $c $conf) . $rest))
   (sequential
     (add-atom &state (action-cost $a $c $conf))
     (init-domain-costs! $rest)))

; --- Master initialization ---
(= (init-domain!)
   (sequential
     ; Clear any existing domain state from core module defaults.
     ; This ensures the domain module is the single source of truth
     ; for beliefs, preferences, viability bounds, action models, and costs.
     (clear-domain-state!)

     ; Beliefs from observables
     (let $obs (collapse (match &self (domain-observable $o $v $p) ($o $v $p)))
       (init-domain-beliefs! $obs))

     ; Preferences
     (let $prefs (collapse (match &self (domain-preference $o $t $i) ($o $t $i)))
       (init-domain-preferences! $prefs))

     ; Viability bounds
     (let $bounds (collapse (match &self (domain-viability $o $min $max) ($o $min $max)))
       (init-domain-viability! $bounds))

     ; Action models
     (let $models (collapse (match &self (domain-action-model $a $o $vd $pd $c) ($a $o $vd $pd $c)))
       (init-domain-models! $models))

     ; Action costs
     (let $costs (collapse (match &self (domain-action-cost $a $c $conf) ($a $c $conf)))
       (init-domain-costs! $costs))

     domain-initialized))

; --- Clear existing domain state ---
; Removes module-loaded defaults so domain declarations are authoritative.
; Uses specific patterns per atom type to ensure exact tuple matching
; for remove-atom (generic cons-cell decomposition changes tuple shape).
(= (clear-domain-state!)
   (sequential
     (let $old (collapse (match &state (belief $o $v $p) (belief $o $v $p)))
       (remove-state-atoms! $old))
     (let $old (collapse (match &state (preference $o $t $i) (preference $o $t $i)))
       (remove-state-atoms! $old))
     (let $old (collapse (match &state (viability-bound $o $min $max) (viability-bound $o $min $max)))
       (remove-state-atoms! $old))
     (let $old (collapse (match &state (action-model $a $o $vd $pd $c) (action-model $a $o $vd $pd $c)))
       (remove-state-atoms! $old))
     (let $old (collapse (match &state (action-cost $a $c $conf) (action-cost $a $c $conf)))
       (remove-state-atoms! $old))
     done))

(= (remove-state-atoms! ()) done)
(= (remove-state-atoms! ($a . $rest))
   (sequential (remove-atom &state $a) (remove-state-atoms! $rest)))


; =============================================================================
; SECTION VIII: AUTO-INITIALIZATION
; =============================================================================
;
; Process domain declarations on load.
; base-actions rewrite and any-action registration are handled by
; the Python loader (dagaz_runtime.py load_dagaz or dagaz_server.py
; init_game_state) since they require &self function definition
; manipulation that MeTTa's add-atom/remove-atom can do but that
; should happen exactly once at boot time.

!(init-domain!)

; =============================================================================
; END OF DOMAIN CONFIGURATION
; =============================================================================
