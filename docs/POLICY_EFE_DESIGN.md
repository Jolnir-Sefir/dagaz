; =============================================================================
; POLICY EFE v5 â€” DESIGN NOTES
; =============================================================================
;
; What changed and why, relative to policy_efe v4.
;
; =============================================================================

; PROBLEM 1: Duplicate State
; --------------------------
; v4 declared its own action-model, action-cost, preference, and viability-bound
; atoms with DIFFERENT names than actions.metta:
;
;   policy_efe v4:   (action-model wait (terrain) 0.0 -0.02)     ; 4-field, tuple name
;   actions.metta v4: (action-model wait terrain-roughness -0.005 -0.01 0.3)  ; 5-field, symbol name
;
; When both loaded into one atomspace, you get contradictory models.
; The pattern matcher returns whichever it finds first, or both non-deterministically.
;
; FIX: policy_efe v5 declares ZERO action models, costs, or type declarations.
; It reads everything through actions.metta's accessors (get-value-delta, etc.).
; Preferences and viability bounds ARE declared here because they're domain config
; that both modules need and don't belong in either.

; PROBLEM 2: No Confidence in Planning
; ------------------------------------
; v4 treated all action model predictions as certain. A model with delta=-0.15
; contributed that full amount at every planning step, even though the agent
; might have zero evidence for it.
;
; FIX: v5 uses the same confidence-weighted accessors as single-step EFE:
;   effective_delta = raw_delta Ã— confidence
; Additionally, confidence DEGRADES with planning depth:
;   planning_confidence = base_confidence Ã— discount^step
; Step 0 (next action): full confidence. Step 2 (three actions out): 0.81Ã— base.
; This correctly captures that deeper predictions are less trustworthy.

; PROBLEM 3: Fixed Epistemic Weight
; ----------------------------------
; v4 used a hardcoded 0.5 coefficient on information gain:
;   EFE = error + cost - 0.5 Ã— info_gain
;
; FIX: v5 computes epistemic weight from the SIMULATED state at each step:
;   sim_ew = max(0.05, safety Ã— uncertainty)
; where safety = 1 - sim_viability_pressure, uncertainty = 1 - sim_avg_precision.
;
; This matters for policy evaluation: a policy (observe, observe, observe)
; sees precision increase at each step, which REDUCES the epistemic weight
; at subsequent steps, correctly capturing diminishing returns.

; PROBLEM 4: Hardcoded Viability Logic  
; -------------------------------------
; v4 viability check used simple bound comparison.
; v4 had no principled viability EFFECT in the EFE formula itself â€” only pruning.
;
; FIX: v5 has BOTH:
;   - Viability pruning: policies crossing hard bounds â†’ EFE = 999 (same as v4)
;   - Viability effect in sim-efe: mirrors v4 actions.metta's principled version
;     based on whether actions move state toward/away from bounds
;     No action-specific hardcoding (no "if retreat then bonus").

; PROBLEM 5: select-action Name Collision
; ----------------------------------------
; v4 defined (= (select-action) (select-action-with-horizon 3))
; actions.metta defines (= (select-action) ...)
; Both loaded â†’ non-deterministic choice between single-step and policy.
;
; FIX: v5 entry point is (select-action-policy). The cycle module has a
; (choose-action) dispatcher controlled by (config action-selection-mode ...).

; PROBLEM 6: Fixed-Arity Lists
; ----------------------------
; v4 had list-to-sim with separate clauses for 1, 2, 3 elements.
; v4 had best-of-evals with separate clauses for 1, 2, N elements.
;
; FIX: v5 uses recursive processing throughout:
;   (= (list-to-sim (($o $v $p) . $rest)) (SCons $o $v $p (list-to-sim $rest)))

; PROBLEM 7: Model Info Gain in Planning
; ---------------------------------------
; In single-step EFE, model info gain creates natural exploration:
; testing uncertain models teaches you about them.
;
; But in a 3-step policy, you only ACTUALLY execute the first action.
; Steps 2 and 3 are hypothetical â€” you don't learn from them.
;
; FIX: v5 counts model info gain ONLY at step 0:
;   (= (sim-model-info-gain $action $state $step)
;      (if (> $step 0) 0.0 (sim-model-info-beliefs $action $beliefs)))
;
; Observable info gain still counts at all steps (precision improvements
; are real within the simulation), but model learning is first-action-only.


; =============================================================================
; MODULE DEPENDENCY GRAPH (v5)
; =============================================================================
;
;   foundations.metta
;       â†“
;   beliefs.metta    â†â”€â”€ observations from environment
;       â†“
;   affect.metta     â†â”€â”€ derived from prediction errors
;       â†“
;   actions.metta    â†â”€â”€ action models (5-field with confidence)
;       â†“                  defines: select-action (single-step)
;   action_learning.metta  â†â”€â”€ updates action models from experience
;       â†“
;   policy_efe.metta â†â”€â”€ reads models via actions.metta accessors
;       â†“                  defines: select-action-policy (multi-step)
;   cycle.metta      â†â”€â”€ dispatches via choose-action
;                          configurable: single vs policy mode
;
; Shared state in atomspace:
;   (action-model $a $o $vd $pd $conf)  â€” declared in actions.metta
;   (preference $o $pref $imp)           â€” declared in policy_efe.metta
;   (viability-bound $o $min $max)       â€” declared in policy_efe.metta
;   (belief $o $val $prec)               â€” declared/updated in beliefs.metta
;   (observation $o $val $prec $t)       â€” injected by cycle.metta


; =============================================================================
; EFE FORMULA COMPARISON
; =============================================================================
;
; SINGLE-STEP (actions.metta):
;   EFE = conf_weighted_error(obs) + cost 
;         - ew Ã— (obs_info_gain + model_info_gain) 
;         + viability_effect
;   where error is measured against actual OBSERVATIONS
;
; POLICY STEP (policy_efe.metta):
;   EFE = conf_weighted_error(pref) + cost
;         - sim_ew Ã— (obs_info_gain + model_info_gain)
;         + sim_viability_effect
;   where error is measured against PREFERENCES (no future observations)
;   and confidence degrades: conf Ã— discount^step
;   and model_info_gain = 0 for step > 0
;
; CUMULATIVE POLICY:
;   Î£_t discount^t Ã— step_efe(t)
;   with viability pruning: if any step violates bounds â†’ 999.0


; =============================================================================
; REMAINING WORK
; =============================================================================
;
; 1. Action set discovery: any-action generator is static (wait/observe/retreat).
;    Should dynamically query available actions from atomspace.
;
; 2. Adaptive horizon: always planning 3 steps ahead. Could adapt based on
;    model confidence â€” plan deeper when models are trusted.
;
; 3. MCTS pruning: 27 policies is fine, but adding a 4th action â†’ 256,
;    5th â†’ 3125. Need branch-and-bound or Monte Carlo tree search.
;
; 4. Preference learning: preferences are static. Agent can't discover
;    new goals or adjust existing ones from experience.
;
; 5. Test on Hyperon: recursive list-to-sim, planning confidence degradation,
;    and the interaction between collapse and non-deterministic gen-policy
;    are the most likely failure points in actual execution.
