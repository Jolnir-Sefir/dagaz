"""
Test EFE-driven action selection — Pure Python executable specification.

Validates that behavior EMERGES from free energy minimization, including
meta-cognitive actions (reflect, ground-concept) competing in the same
EFE landscape as domain actions.

Shared formula (canonical — identical to viability_test.py):
  EFE(action) = Σ_obs [pragmatic_error - info_value + viability_penalty] + cost

  pragmatic_error = |predicted_value - preference| × importance
  info_value:
    if delta_prec > 0:  0.5 × delta_prec × (1 - precision)   [diminishing returns]
    if delta_prec ≤ 0:  0.5 × delta_prec                      [decay at full rate]
  viability_penalty (threshold derived from belief precision):
    uncertainty = max(1 - precision, 0.01)
    safety_ratio = margin / uncertainty
    margin < 0:          10.0                                    [violated]
    safety_ratio < 1.0:  5.0 × uncertainty × (1 - safety_ratio) [approaching]
    safety_ratio ≥ 1.0:  0.0                                    [safe]
  viability_bonus:
    safety_ratio < 1.0 AND action helps:  0.5 × |effect|

No external dependencies. No Hyperon.

Usage: python test-efe.py
"""

import sys
from typing import Dict, List, Tuple


# =============================================================================
# COGNITIVE CORE — Pure Python EFE with viability
# =============================================================================

class EFECore:
    """Minimal EFE computation with viability penalties.
    
    All model parameters, preferences, viability bounds, and costs are
    identical to viability_test.py. The two files are cross-validated:
    this one proves the formula in pure Python, that one proves it
    executes correctly in MeTTa via Hyperon.
    """
    
    ACTIONS = ["wait", "observe", "retreat", "reflect", "ground-concept"]
    OBSERVABLES = ["terrain", "power", "threat", "self-comp", "grounding"]
    
    def __init__(self):
        # Internal metabolic state for dynamic costs
        self.action_model_count = 5.0
        
        # --- Preferences: (value, importance) ---
        self.preferences = {
            "terrain":   (0.3, 0.8),
            "power":     (0.9, 1.0),
            "threat":    (0.1, 0.9),
            "self-comp": (0.8, 0.7),
            "grounding": (1.0, 0.6),
        }
        
        # --- Viability bounds: (threshold, direction) ---
        # direction: "lower" = value must stay above; "upper" = must stay below
        self.viability_bounds = {
            "power":     (0.15, "lower"),
            "threat":    (0.6,  "upper"),
            "self-comp": (0.3,  "lower"),
        }
        
        # --- Action costs (dynamic, scale with model complexity) ---
        self.action_costs = {
            "wait":           0.01,
            "observe":        0.04,
            "retreat":        0.15,
            "reflect":        0.01 * self.action_model_count,   # 0.05
            "ground-concept": 0.012 * self.action_model_count,  # 0.06
        }
        
        # --- Action models: {(action, obs): (delta_mean, delta_precision)} ---
        self.models = {
            # wait: minimal change, slight precision decay
            ("wait", "terrain"):   (0.0,  -0.02),
            ("wait", "power"):     (-0.01, -0.01),
            ("wait", "threat"):    (0.0,  -0.02),
            ("wait", "self-comp"): (0.0,  -0.01),
            ("wait", "grounding"): (0.0,  -0.01),
            
            # observe: external info gain, drains power (sensors cost energy)
            ("observe", "terrain"):   (0.0,  0.08),
            ("observe", "power"):     (-0.02, 0.05),
            ("observe", "threat"):    (0.0,  0.06),
            ("observe", "self-comp"): (0.0,  0.0),
            ("observe", "grounding"): (0.0,  0.0),
            
            # retreat: reduce threat/terrain, RECOVERS power (move to safety)
            ("retreat", "terrain"):   (-0.30, 0.03),
            ("retreat", "power"):     (0.05,  0.02),
            ("retreat", "threat"):    (-0.25, 0.05),
            ("retreat", "self-comp"): (0.0,   0.0),
            ("retreat", "grounding"): (0.0,   0.0),
            
            # reflect: internal info gain, slight self-model improvement,
            # external precision decays (attention directed inward)
            ("reflect", "terrain"):   (0.0,  -0.02),
            ("reflect", "power"):     (-0.01, -0.02),
            ("reflect", "threat"):    (0.0,  -0.02),
            ("reflect", "self-comp"): (0.01,  0.08),
            ("reflect", "grounding"): (0.01,  0.08),
            
            # ground-concept: targeted grounding work, moves grounding
            # toward preference via active integration of learned structure
            ("ground-concept", "terrain"):   (0.0,   -0.02),
            ("ground-concept", "power"):     (-0.01,  -0.02),
            ("ground-concept", "threat"):    (0.0,   -0.02),
            ("ground-concept", "self-comp"): (0.0,    0.0),
            ("ground-concept", "grounding"): (0.05,   0.15),
        }
        
        # Beliefs set per-scenario
        self.beliefs = {}
    
    def set_beliefs(self, belief_list: List[Tuple[str, float, float]]):
        """Set beliefs from (observable, value, precision) tuples."""
        self.beliefs = {obs: (val, prec) for obs, val, prec in belief_list}
    
    def _clamp(self, v: float) -> float:
        return max(0.0, min(1.0, v))
    
    def _efe_obs(self, action: str, obs: str) -> float:
        """EFE contribution from one observable: pragmatic error - info value."""
        bval, bprec = self.beliefs.get(obs, (0.5, 0.5))
        pval, pimp = self.preferences.get(obs, (0.5, 0.5))
        dm, dp = self.models.get((action, obs), (0.0, 0.0))
        
        # Predicted value after action
        pred_val = self._clamp(bval + dm)
        
        # Pragmatic error (NOT precision-weighted — prevents ignorance-seeking)
        deviation = abs(pred_val - pval)
        pragmatic_error = deviation * pimp
        
        # Epistemic value — info-gain asymmetry:
        #   Positive delta-prec scaled by (1-prec): diminishing returns at high precision
        #   Negative delta-prec at full rate: precision decay is not attenuated
        uncertainty = 1.0 - bprec
        if dp > 0:
            scaled_dp = dp * uncertainty
        else:
            scaled_dp = dp  # Negative (decay) applies directly
        info_value = 0.5 * scaled_dp
        
        return pragmatic_error - info_value
    
    def _viability_penalty(self, action: str, obs: str) -> float:
        """Viability penalty for one observable. Supports lower and upper bounds.
        
        Threshold derived from belief precision: safety_ratio = margin / uncertainty.
        When < 1, boundary is within one uncertainty-width → pressure activates.
        """
        if obs not in self.viability_bounds:
            return 0.0
        
        thresh, direction = self.viability_bounds[obs]
        bval, bprec = self.beliefs.get(obs, (0.5, 0.5))
        dm, _ = self.models.get((action, obs), (0.0, 0.0))
        
        pred_val = self._clamp(bval + dm)
        uncertainty = max(1.0 - bprec, 0.01)
        
        # Margin: positive = safe, negative = violated
        if direction == "lower":
            margin = pred_val - thresh
        else:
            margin = thresh - pred_val
        
        safety_ratio = margin / uncertainty
        
        # Penalty curve
        if margin < 0.0:
            penalty = 10.0
        elif safety_ratio < 1.0:
            penalty = 5.0 * uncertainty * (1.0 - safety_ratio)
        else:
            penalty = 0.0
        
        # Bonus for actions that move toward safety when within uncertainty
        bonus = 0.0
        if safety_ratio < 1.0:
            if direction == "lower" and dm > 0.0:
                bonus = 0.5 * dm
            elif direction == "upper" and dm < 0.0:
                bonus = 0.5 * abs(dm)
        
        return penalty - bonus
    
    def compute_efe(self, action: str) -> float:
        """Total EFE for an action across all observables."""
        total = 0.0
        for obs in self.OBSERVABLES:
            total += self._efe_obs(action, obs)
            total += self._viability_penalty(action, obs)
        total += self.action_costs.get(action, 0.05)
        return total
    
    def select_action(self) -> Tuple[str, Dict[str, float]]:
        """Select action with minimum EFE. Returns (action, {action: efe})."""
        efes = {a: self.compute_efe(a) for a in self.ACTIONS}
        best = min(efes, key=efes.get)
        return best, efes


# =============================================================================
# TEST HARNESS
# =============================================================================

def test_scenario(name: str, beliefs: List[Tuple[str, float, float]], 
                  expected: str, allow_alternatives: List[str] = None) -> bool:
    """Run a scenario and check if expected action is selected."""
    core = EFECore()
    core.set_beliefs(beliefs)
    
    selected, efes = core.select_action()
    valid = [expected] + (allow_alternatives or [])
    passed = selected in valid
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")
    print(f"  {'Action':<16} {'EFE':>8}")
    print(f"  {'-'*28}")
    for a in EFECore.ACTIONS:
        marker = " ← SELECTED" if a == selected else ""
        print(f"  {a:<16} {efes[a]:>8.4f}{marker}")
    print(f"\n  Expected: {expected}  |  Got: {selected}  |  {status}")
    return passed


def main():
    print("=" * 70)
    print("EFE-DRIVEN ACTION SELECTION VALIDATION")
    print("Pure Python executable specification — zero dependencies")
    print("=" * 70)
    print("\nKey: NO HEURISTICS, NO THRESHOLDS, NO MODE SWITCHES")
    print("Action emerges from minimizing Expected Free Energy.")
    print("Meta-cognitive actions (reflect, ground-concept) compete")
    print("in the same EFE landscape as domain actions.")
    print("Shared canonical formula with viability_test.py (MeTTa).")
    
    results = []
    
    # ==================================================================
    # DOMAIN ACTION SCENARIOS 
    # ==================================================================
    
    # 1. Nominal: everything near preferences, well-known state.
    #    Self-model precision must be high enough that reflect's info-gain
    #    is outweighed by its cost — this IS epistemic satiation.
    results.append(test_scenario(
        "Nominal — everything near preferences, well-known",
        [("terrain", 0.35, 0.8), ("power", 0.85, 0.9), ("threat", 0.15, 0.7),
         ("self-comp", 0.75, 0.85), ("grounding", 0.95, 0.85)],
        "wait",
        allow_alternatives=["observe"]
    ))
    
    # 2. High threat: viability penalty on threat makes retreat win
    results.append(test_scenario(
        "High threat — viability penalty drives retreat",
        [("terrain", 0.35, 0.8), ("power", 0.85, 0.9), ("threat", 0.7, 0.8),
         ("self-comp", 0.75, 0.7), ("grounding", 0.95, 0.7)],
        "retreat"
    ))
    
    # 3. Uncertain terrain: low precision → high info gain for observe
    results.append(test_scenario(
        "Uncertain terrain — observe for info gain",
        [("terrain", 0.35, 0.3), ("power", 0.85, 0.9), ("threat", 0.15, 0.7),
         ("self-comp", 0.75, 0.7), ("grounding", 0.95, 0.7)],
        "observe"
    ))
    
    # 4. Extreme terrain: far from preference → retreat reduces deviation
    results.append(test_scenario(
        "Extreme terrain — retreat to smoother",
        [("terrain", 0.95, 0.8), ("power", 0.85, 0.9), ("threat", 0.15, 0.7),
         ("self-comp", 0.75, 0.7), ("grounding", 0.95, 0.7)],
        "retreat"
    ))
    
    # 5. Low power: near viability bound → retreat recovers power (+0.05).
    #    Power precision is moderate (0.7): the agent isn't fully sure it's safe.
    #    Uncertainty (0.3) > margin (0.1), so viability pressure activates.
    results.append(test_scenario(
        "Low power — retreat to recover",
        [("terrain", 0.35, 0.8), ("power", 0.25, 0.7), ("threat", 0.15, 0.7),
         ("self-comp", 0.75, 0.7), ("grounding", 0.95, 0.7)],
        "retreat"
    ))
    
    # ==================================================================
    # META-COGNITIVE SCENARIOS
    # ==================================================================
    
    # 6. Uncertain self-model: low precision on self-comp → reflect wins via info gain
    results.append(test_scenario(
        "Uncertain self-model — reflect for self-knowledge",
        [("terrain", 0.35, 0.8), ("power", 0.85, 0.9), ("threat", 0.15, 0.8),
         ("self-comp", 0.5, 0.15), ("grounding", 0.95, 0.7)],
        "reflect"
    ))
    
    # 7. Ungrounded concepts: low precision + low value on grounding → ground-concept
    results.append(test_scenario(
        "Ungrounded concepts — ground-concept for understanding",
        [("terrain", 0.35, 0.8), ("power", 0.85, 0.9), ("threat", 0.15, 0.8),
         ("self-comp", 0.75, 0.7), ("grounding", 0.4, 0.15)],
        "ground-concept"
    ))
    
    # 8. Broad self-uncertainty: both self-comp and grounding uncertain → reflect
    #    (reflect gains on both; ground-concept only gains on grounding)
    results.append(test_scenario(
        "Broad self-uncertainty — reflect over ground-concept",
        [("terrain", 0.35, 0.8), ("power", 0.85, 0.9), ("threat", 0.15, 0.8),
         ("self-comp", 0.5, 0.15), ("grounding", 0.5, 0.15)],
        "reflect"
    ))
    
    # 9. Threat overrides introspection: even with self-uncertainty,
    #    viability penalty on threat dominates
    results.append(test_scenario(
        "Threat overrides self-reflection",
        [("terrain", 0.35, 0.8), ("power", 0.85, 0.9), ("threat", 0.7, 0.8),
         ("self-comp", 0.5, 0.15), ("grounding", 0.5, 0.15)],
        "retreat"
    ))
    
    # 10. Known self-model: high precision on self-comp + grounding → 
    #     reflect/ground-concept lose to cheaper domain actions (epistemic satiation)
    results.append(test_scenario(
        "Known self-model — reflect loses to cheaper wait/observe",
        [("terrain", 0.35, 0.8), ("power", 0.85, 0.9), ("threat", 0.15, 0.7),
         ("self-comp", 0.78, 0.85), ("grounding", 0.96, 0.85)],
        "wait",
        allow_alternatives=["observe"]
    ))
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {sum(results)}/{len(results)} scenarios passed")
    print("=" * 70)
    
    domain_pass = sum(results[:5])
    meta_pass = sum(results[5:])
    print(f"  Domain actions:         {domain_pass}/5")
    print(f"  Meta-cognitive actions: {meta_pass}/5")
    
    if all(results):
        print("\n✓ All scenarios pass.")
        print("  Behavior emerges from EFE minimization — no heuristics.")
        print("  Viability penalties drive protective action near bounds.")
        print("  Meta-cognitive actions compete uniformly — no special cases.")
        print("  Self-inquiry emerges when self-model precision is low.")
        print("  External threats override introspection via viability penalty.")
        print("  Epistemic satiation: high precision kills info-gain term.")
    else:
        print("\n✗ Some scenarios failed — check EFE computation or parameters.")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
