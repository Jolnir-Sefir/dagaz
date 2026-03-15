"""
MeTTa Cognitive Core - Viability-Aware Implementation

Validates that EFE-driven action selection correctly handles viability
bounds on both domain observables (power, threat) and self-model
observables (self-competence, grounding-completeness).

Key behaviors tested:
  - Retreat emerges under power viability threat 
  - Retreat emerges under threat viability pressure (upper bound)
  - Reflect emerges under self-competence viability threat
  - Ground-concept emerges under grounding-completeness pressure
  - External viability threats override meta-cognitive actions
  - Epistemic Satiation: agents stop reflecting when precision is high

Shared formula (canonical — identical to test-efe.py):
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

Usage: python viability_test.py
Requires: pip install hyperon
"""

from hyperon import MeTTa
from typing import Dict, Tuple


class ViabilityAwareCognitiveCore:
    """Cognitive core with viability penalty across domain and self-model observables"""
    
    OBSERVABLES = ['power', 'terrain', 'threat', 'self-comp', 'grounding']
    ACTIONS = ['wait', 'observe', 'retreat', 'reflect', 'ground-concept']
    
    def __init__(self):
        self.metta = MeTTa()
        self.cycle_count = 0
        self.action_history = []
        self._setup_core()
    
    def _setup_core(self):
        """Initialize with viability-aware EFE including meta-cognitive actions"""
        init_code = """
        (= (abs $x) (if (< $x 0.0) (- 0.0 $x) $x))
        (= (clamp $v $min $max) (if (< $v $min) $min (if (> $v $max) $max $v)))
        (= (max $a $b) (if (> $a $b) $a $b))
        
        ; ==================================================================
        ; STATE: Beliefs (value, precision)
        ; ==================================================================
        (belief power 0.7 0.6)
        (belief terrain 0.35 0.5)
        (belief threat 0.15 0.7)
        (belief self-comp 0.7 0.5)
        (belief grounding 0.8 0.5)
        
        ; Internal metabolism (used for dynamic cost scaling)
        (belief action-model-count 5.0 0.9)
        
        ; ==================================================================
        ; PREFERENCES: (observable, target_value, importance)
        ; ==================================================================
        (preference power 0.9 1.0)
        (preference terrain 0.3 0.8)
        (preference threat 0.1 0.9)
        (preference self-comp 0.8 0.7)
        (preference grounding 1.0 0.6)
        
        ; ==================================================================
        ; VIABILITY BOUNDS: (observable, threshold, direction)
        ;   lower = value must stay ABOVE threshold
        ;   upper = value must stay BELOW threshold
        ; ==================================================================
        (viability-bound power 0.15 lower)
        (viability-bound threat 0.6 upper)
        (viability-bound self-comp 0.3 lower)
        
        ; ==================================================================
        ; ACTION MODELS: unified (action, observable, delta_mean, delta_prec)
        ; Matches canonical MeTTa schema from actions.metta
        ; ==================================================================
        
        ; --- wait: minimal change, slight precision decay ---
        (action-model wait power -0.01 -0.01)
        (action-model wait terrain 0.0 -0.02)
        (action-model wait threat 0.0 -0.02)
        (action-model wait self-comp 0.0 -0.01)
        (action-model wait grounding 0.0 -0.01)
        
        ; --- observe: external info gain, drains power ---
        (action-model observe power -0.02 0.05)
        (action-model observe terrain 0.0 0.08)
        (action-model observe threat 0.0 0.06)
        (action-model observe self-comp 0.0 0.0)
        (action-model observe grounding 0.0 0.0)
        
        ; --- retreat: reduce threat/terrain, recover power ---
        (action-model retreat power 0.05 0.02)
        (action-model retreat terrain -0.30 0.03)
        (action-model retreat threat -0.25 0.05)
        (action-model retreat self-comp 0.0 0.0)
        (action-model retreat grounding 0.0 0.0)
        
        ; --- reflect: internal info gain, external precision decays ---
        (action-model reflect power -0.01 -0.02)
        (action-model reflect terrain 0.0 -0.02)
        (action-model reflect threat 0.0 -0.02)
        (action-model reflect self-comp 0.01 0.08)
        (action-model reflect grounding 0.01 0.08)
        
        ; --- ground-concept: targeted grounding work ---
        (action-model ground-concept power -0.01 -0.02)
        (action-model ground-concept terrain 0.0 -0.02)
        (action-model ground-concept threat 0.0 -0.02)
        (action-model ground-concept self-comp 0.0 0.0)
        (action-model ground-concept grounding 0.05 0.15)
        
        ; ==================================================================
        ; DYNAMIC COSTS: scale with internal model complexity
        ; ==================================================================
        (= (cost wait) 0.01)
        (= (cost observe) 0.04)
        (= (cost retreat) 0.15)
        (= (cost reflect) (* 0.01 (match &self (belief action-model-count $v $p) $v)))
        (= (cost ground-concept) (* 0.012 (match &self (belief action-model-count $v $p) $v)))
        
        ; ==================================================================
        ; ACCESSORS
        ; ==================================================================
        (= (belief-val $o) (match &self (belief $o $v $p) $v))
        (= (belief-prec $o) (match &self (belief $o $v $p) $p))
        (= (pref-val $o) (match &self (preference $o $v $i) $v))
        (= (pref-imp $o) (match &self (preference $o $v $i) $i))
        (= (delta-mean $a $o) (match &self (action-model $a $o $dm $dp) $dm))
        (= (delta-prec $a $o) (match &self (action-model $a $o $dm $dp) $dp))
        
        ; ==================================================================
        ; EFE CONTRIBUTION PER OBSERVABLE
        ;
        ; Info-gain asymmetry: positive delta-prec is scaled by (1 - prec)
        ; to give diminishing returns at high precision. Negative delta-prec
        ; (precision decay) applies at full rate — you can't ignore the
        ; world just because you already know about it.
        ; ==================================================================
        (= (info-value $action $obs)
           (let* (
             ($prec (belief-prec $obs))
             ($dp (delta-prec $action $obs))
             ($uncertainty (- 1.0 $prec))
             ($scaled-dp (if (> $dp 0.0)
                             (* $dp $uncertainty)
                             $dp))
           )
           (* 0.5 $scaled-dp)))
        
        (= (efe-obs $action $obs)
           (let* (
             ($val (belief-val $obs))
             ($pref-v (pref-val $obs))
             ($pref-i (pref-imp $obs))
             ($dm (delta-mean $action $obs))
             ($new-val (clamp (+ $val $dm) 0.0 1.0))
             ($deviation (abs (- $new-val $pref-v)))
             ($weighted-dev (* $deviation $pref-i))
             ($iv (info-value $action $obs))
           )
           (- $weighted-dev $iv)))
        
        ; ==================================================================
        ; VIABILITY PENALTY: supports both lower and upper bounds
        ;
        ; Lower bound (power, self-comp): margin = value - threshold
        ; Upper bound (threat): margin = threshold - value
        ; In both cases: positive margin = safe, negative = violated
        ; ==================================================================
        (= (has-viability? $obs)
           (let $r (match &self (viability-bound $obs $b $d) True)
             (if (== $r ()) False True)))
        
        (= (viability-margin $action $obs)
           (match &self (viability-bound $obs $bound $direction)
             (let* (
               ($val (belief-val $obs))
               ($dm (delta-mean $action $obs))
               ($new-val (clamp (+ $val $dm) 0.0 1.0))
             )
             (if (== $direction lower)
                 (- $new-val $bound)
                 (- $bound $new-val)))))
        
        (= (viability-helps? $action $obs)
           (match &self (viability-bound $obs $bound $direction)
             (let $dm (delta-mean $action $obs)
               (if (== $direction lower)
                   (> $dm 0.0)
                   (< $dm 0.0)))))
        
        (= (viability-penalty $action $obs)
           (if (has-viability? $obs)
               (let* (
                 ($margin (viability-margin $action $obs))
                 ($dm (delta-mean $action $obs))
                 ($prec (belief-prec $obs))
                 ($uncertainty (max (- 1.0 $prec) 0.01))
                 ($safety-ratio (/ $margin $uncertainty))
                 ($penalty
                   (if (< $margin 0.0)
                       10.0
                       (if (< $safety-ratio 1.0)
                           (* 5.0 (* $uncertainty (- 1.0 $safety-ratio)))
                           0.0)))
                 ($bonus
                   (if (< $safety-ratio 1.0)
                       (if (viability-helps? $action $obs)
                           (* 0.5 (abs $dm))
                           0.0)
                       0.0))
               )
               (- $penalty $bonus))
               0.0))
        
        ; ==================================================================
        ; TOTAL EFE: sum over all observables
        ; ==================================================================
        (= (compute-efe $action)
           (let* (
             ($e-power (efe-obs $action power))
             ($e-terrain (efe-obs $action terrain))
             ($e-threat (efe-obs $action threat))
             ($e-scomp (efe-obs $action self-comp))
             ($e-ground (efe-obs $action grounding))
             ($v-power (viability-penalty $action power))
             ($v-threat (viability-penalty $action threat))
             ($v-scomp (viability-penalty $action self-comp))
             ($c (cost $action))
             ($total (+ $e-power (+ $e-terrain (+ $e-threat (+ $e-scomp
                     (+ $e-ground (+ $v-power (+ $v-threat (+ $v-scomp $c)))))))))
           )
           $total))
        """
        self.metta.run(init_code)
    
    def get_efes(self) -> Dict[str, float]:
        """Get current EFE values for all actions"""
        efes = {}
        for action in self.ACTIONS:
            r = self.metta.run(f"!(compute-efe {action})")
            if r and r[0]:
                try: efes[action] = float(str(r[0][0]))
                except: pass
        return efes
    
    def get_belief(self, obs: str) -> Tuple[float, float]:
        """Get belief value and precision"""
        r = self.metta.run(f"!(match &self (belief {obs} $v $p) ($v $p))")
        if r and r[0]:
            s = str(r[0][0]).replace('(', '').replace(')', '').split()
            if len(s) >= 2: return float(s[0]), float(s[1])
        return 0.5, 0.5
    
    def update_belief(self, obs: str, new_val: float, new_prec: float):
        """Update belief"""
        old_val, old_prec = self.get_belief(obs)
        self.metta.run(f"!(remove-atom &self (belief {obs} {old_val} {old_prec}))")
        self.metta.run(f"!(add-atom &self (belief {obs} {new_val} {new_prec}))")
    
    def select_action(self) -> str:
        """Select action with minimum EFE"""
        efes = self.get_efes()
        return min(efes, key=efes.get) if efes else "wait"
    
    def apply_effects(self, action: str):
        """Apply action effects with correct info-gain asymmetry"""
        for obs in self.OBSERVABLES:
            val, prec = self.get_belief(obs)
            
            r_dm = self.metta.run(f"!(delta-mean {action} {obs})")
            r_dp = self.metta.run(f"!(delta-prec {action} {obs})")
            dm = float(str(r_dm[0][0])) if (r_dm and r_dm[0]) else 0.0
            dp = float(str(r_dp[0][0])) if (r_dp and r_dp[0]) else 0.0
            
            new_val = max(0.0, min(1.0, val + dm))
            
            # Info-gain asymmetry: positive gains diminish, decay applies fully
            if dp > 0:
                precision_delta = dp * (1.0 - prec)
            else:
                precision_delta = dp
            
            new_prec = max(0.1, min(0.95, prec + precision_delta))
            self.update_belief(obs, new_val, new_prec)
    
    def observe_world(self, observations: Dict[str, float]):
        """Update beliefs from observations"""
        lr = 0.3
        for obs, obs_val in observations.items():
            bel_val, bel_prec = self.get_belief(obs)
            error = obs_val - bel_val
            new_val = bel_val + lr * error
            new_prec = min(0.95, bel_prec + 0.03)
            self.update_belief(obs, new_val, new_prec)
    
    def cycle(self, observations: Dict[str, float] = None) -> Dict:
        """Run one cycle"""
        self.cycle_count += 1
        if observations: self.observe_world(observations)
        
        efes = self.get_efes()
        action = self.select_action()
        self.apply_effects(action)
        self.action_history.append(action)
        
        power_val, _ = self.get_belief("power")
        scomp_val, _ = self.get_belief("self-comp")
        ground_val, _ = self.get_belief("grounding")
        
        return {
            "cycle": self.cycle_count,
            "power": power_val,
            "self-comp": scomp_val,
            "grounding": ground_val,
            "efes": efes,
            "action": action
        }


# ==========================================================================
# EFE DECOMPOSITION: show all observable contributions
# ==========================================================================

def run_efe_decomposition():
    print("=" * 80)
    print("EFE DECOMPOSITION — ALL OBSERVABLES AT CRITICAL SELF-COMPETENCE")
    print("=" * 80)
    
    # State: self-comp near viability bound, everything else nominal
    beliefs = {
        "power":     (0.70, 0.6),
        "terrain":   (0.35, 0.5),
        "threat":    (0.15, 0.7),
        "self-comp": (0.35, 0.7),
        "grounding": (0.80, 0.5),
    }
    prefs = {
        "power":     (0.9, 1.0),
        "terrain":   (0.3, 0.8),
        "threat":    (0.1, 0.9),
        "self-comp": (0.8, 0.7),
        "grounding": (1.0, 0.6),
    }
    viability = {
        "power":     (0.15, "lower"),
        "threat":    (0.6,  "upper"),
        "self-comp": (0.3,  "lower"),
    }
    models = {
        ("wait",    "power"): (-0.01, -0.01), ("wait",    "terrain"): (0.0, -0.02),
        ("wait",    "threat"): (0.0, -0.02),  ("wait",    "self-comp"): (0.0, -0.01),
        ("wait",    "grounding"): (0.0, -0.01),
        ("observe", "power"): (-0.02, 0.05),  ("observe", "terrain"): (0.0, 0.08),
        ("observe", "threat"): (0.0, 0.06),   ("observe", "self-comp"): (0.0, 0.0),
        ("observe", "grounding"): (0.0, 0.0),
        ("retreat", "power"): (0.05, 0.02),   ("retreat", "terrain"): (-0.30, 0.03),
        ("retreat", "threat"): (-0.25, 0.05),  ("retreat", "self-comp"): (0.0, 0.0),
        ("retreat", "grounding"): (0.0, 0.0),
        ("reflect", "power"): (-0.01, -0.02), ("reflect", "terrain"): (0.0, -0.02),
        ("reflect", "threat"): (0.0, -0.02),  ("reflect", "self-comp"): (0.01, 0.08),
        ("reflect", "grounding"): (0.01, 0.08),
        ("ground-concept", "power"): (-0.01, -0.02), ("ground-concept", "terrain"): (0.0, -0.02),
        ("ground-concept", "threat"): (0.0, -0.02),  ("ground-concept", "self-comp"): (0.0, 0.0),
        ("ground-concept", "grounding"): (0.05, 0.15),
    }
    costs = {"wait": 0.01, "observe": 0.04, "retreat": 0.15, "reflect": 0.05, "ground-concept": 0.06}
    actions = ["wait", "observe", "retreat", "reflect", "ground-concept"]
    observables = ["power", "terrain", "threat", "self-comp", "grounding"]
    
    for action in actions:
        print(f"\n{action.upper()} (cost={costs[action]:.3f}):")
        total_pragmatic = 0.0
        total_info = 0.0
        total_viab = 0.0
        
        for obs in observables:
            bval, bprec = beliefs[obs]
            pval, pimp = prefs[obs]
            dm, dp = models[(action, obs)]
            
            new_val = max(0.0, min(1.0, bval + dm))
            deviation = abs(new_val - pval)
            pragmatic = deviation * pimp
            
            uncertainty = 1.0 - bprec
            if dp > 0:
                scaled_dp = dp * uncertainty
            else:
                scaled_dp = dp
            info_val = 0.5 * scaled_dp
            
            # Viability
            viab = 0.0
            if obs in viability:
                thresh, direction = viability[obs]
                if direction == "lower":
                    margin = new_val - thresh
                else:
                    margin = thresh - new_val
                
                unc = max(1.0 - bprec, 0.01)
                safety_ratio = margin / unc
                
                if margin < 0.0:
                    viab = 10.0
                elif safety_ratio < 1.0:
                    viab = 5.0 * unc * (1.0 - safety_ratio)
                
                if safety_ratio < 1.0:
                    if direction == "lower" and dm > 0.0:
                        viab -= 0.5 * dm
                    elif direction == "upper" and dm < 0.0:
                        viab -= 0.5 * abs(dm)
            
            total_pragmatic += pragmatic
            total_info += info_val
            total_viab += viab
            
            if abs(pragmatic) > 0.001 or abs(info_val) > 0.001 or abs(viab) > 0.001:
                print(f"  {obs:<12} prag={pragmatic:+.4f}  info={info_val:+.4f}  viab={viab:+.4f}")
        
        total = total_pragmatic - total_info + total_viab + costs[action]
        print(f"  {'TOTAL':<12} prag={total_pragmatic:+.4f}  info={total_info:+.4f}  "
              f"viab={total_viab:+.4f}  cost={costs[action]:+.4f}  EFE={total:+.4f}")


# ==========================================================================
# TESTS
# ==========================================================================

def run_power_viability_test():
    print("\n" + "=" * 80)
    print("TEST 1: POWER VIABILITY (Domain Threat vs Normal Operation)")
    print("=" * 80)
    print(f"\n{'Power':<8} {'wait':<10} {'observe':<10} {'retreat':<10} "
          f"{'reflect':<10} {'ground-c':<10} {'Selected':<10}")
    print("-" * 72)
    
    test_powers = [0.7, 0.5, 0.35, 0.25, 0.18]
    selections = []
    
    for power in test_powers:
        core = ViabilityAwareCognitiveCore()
        core.update_belief("power", power, 0.8)
        core.update_belief("self-comp", 0.75, 0.8)
        efes = core.get_efes()
        selected = min(efes, key=efes.get)
        selections.append(selected)
        
        print(f"{power:<8.2f} {efes.get('wait',0):<10.3f} "
              f"{efes.get('observe',0):<10.3f} {efes.get('retreat',0):<10.3f} "
              f"{efes.get('reflect',0):<10.3f} {efes.get('ground-concept',0):<10.3f} "
              f"{selected:<10}")
    
    passed = (selections[0] in ['observe', 'wait']) and (selections[-1] == 'retreat')
    print(f"\n{'✓' if passed else '✗'} Valid scaling from observation to retreat.")
    return passed


def run_self_competence_viability_test():
    print("\n" + "=" * 80)
    print("TEST 2: SELF-COMPETENCE VIABILITY (meta-cognitive observable)")
    print("=" * 80)
    print(f"\n{'SelfComp':<10} {'wait':<10} {'observe':<10} {'retreat':<10} "
          f"{'reflect':<10} {'ground-c':<10} {'Selected':<10}")
    print("-" * 74)
    
    test_scomps = [0.7, 0.55, 0.45, 0.38, 0.33]
    selections = []
    
    for sc in test_scomps:
        core = ViabilityAwareCognitiveCore()
        core.update_belief("self-comp", sc, 0.7)
        efes = core.get_efes()
        selected = min(efes, key=efes.get)
        selections.append(selected)
        
        print(f"{sc:<10.2f} {efes.get('wait',0):<10.3f} "
              f"{efes.get('observe',0):<10.3f} {efes.get('retreat',0):<10.3f} "
              f"{efes.get('reflect',0):<10.3f} {efes.get('ground-concept',0):<10.3f} "
              f"{selected:<10}")
    
    passed = selections[-1] == 'reflect' and selections[0] != 'reflect'
    print(f"\n{'✓' if passed else '✗'} Near bound selects reflect: {passed}")
    return passed


def run_grounding_pressure_test():
    print("\n" + "=" * 80)
    print("TEST 3: GROUNDING-COMPLETENESS PRESSURE (Epistemic scaling)")
    print("=" * 80)
    print(f"\n{'Ground':<8} {'Prec':<6} {'wait':<10} {'observe':<10} {'retreat':<10} "
          f"{'reflect':<10} {'ground-c':<10} {'Selected':<10}")
    print("-" * 74)
    
    scenarios = [
        (0.9, 0.7, "High grounding, high prec"),
        (0.5, 0.7, "Low grounding, high prec"),
        (0.5, 0.2, "Low grounding, low prec"),
        (0.3, 0.2, "Very low grounding, low prec"),
    ]
    
    results = []
    for gval, gprec, desc in scenarios:
        core = ViabilityAwareCognitiveCore()
        core.update_belief("grounding", gval, gprec)
        core.update_belief("self-comp", 0.75, 0.8)
        efes = core.get_efes()
        selected = min(efes, key=efes.get)
        results.append((selected, desc))
        
        print(f"{gval:<8.2f} {gprec:<6.2f} {efes.get('wait',0):<10.3f} "
              f"{efes.get('observe',0):<10.3f} {efes.get('retreat',0):<10.3f} "
              f"{efes.get('reflect',0):<10.3f} {efes.get('ground-concept',0):<10.3f} "
              f"{selected:<10}")
    
    meta_actions = {'reflect', 'ground-concept'}
    passed = results[2][0] in meta_actions and results[0][0] not in meta_actions
    print(f"\n{'✓' if passed else '✗'} Low precision triggers grounding, high precision does not.")
    return passed


def run_multi_cycle_test():
    print("\n" + "=" * 80)
    print("TEST 4: MULTI-CYCLE WITH EPISTEMIC SATIATION")
    print("=" * 80)
    
    core = ViabilityAwareCognitiveCore()
    core.update_belief("self-comp", 0.75, 0.8)
    core.update_belief("grounding", 0.95, 0.8)
    
    print("\nPhase 1: Normal (self-comp stable at 0.70, high precision)")
    for i in range(5):
        result = core.cycle({"self-comp": 0.70, "power": 0.7})
        print(f"  Cycle {result['cycle']:>2}: self-comp={result['self-comp']:.2f}, "
              f"action={result['action']}")
              
    print("\nPhase 2: Crisis (self-comp dropping 0.40 -> 0.30)")
    for i in range(5):
        obs_sc = 0.40 - i * 0.025
        result = core.cycle({"self-comp": obs_sc, "power": 0.7})
        print(f"  Cycle {result['cycle']:>2}: self-comp={result['self-comp']:.2f}, "
              f"action={result['action']}")
              
    print("\nPhase 3: Recovery (self-comp rising 0.50 -> 0.70)")
    for i in range(5):
        obs_sc = 0.50 + i * 0.05
        result = core.cycle({"self-comp": obs_sc, "power": 0.7})
        print(f"  Cycle {result['cycle']:>2}: self-comp={result['self-comp']:.2f}, "
              f"action={result['action']}")
              
    phase1 = core.action_history[:5]
    phase2 = core.action_history[5:10]
    
    print("\n--- ACTION SUMMARY ---")
    print(f"Phase 1 (Normal):   {phase1}")
    print(f"Phase 2 (Crisis):   {phase2}")
    
    passed = ('reflect' not in phase1 and 'reflect' in phase2)
    print(f"\n{'✓' if passed else '✗'} Behavioral shift detected without stuck attractors.")
    return passed


def run_dual_viability_test():
    print("\n" + "=" * 80)
    print("TEST 5: DUAL VIABILITY THREAT")
    print("=" * 80)
    
    core = ViabilityAwareCognitiveCore()
    core.update_belief("power", 0.20, 0.8)
    core.update_belief("self-comp", 0.35, 0.7)
    
    efes = core.get_efes()
    selected = min(efes, key=efes.get)
    
    print("State: power=0.20 (bound=0.15), self-comp=0.35 (bound=0.30)\n")
    for action in ViabilityAwareCognitiveCore.ACTIONS:
        marker = " ← SELECTED" if action == selected else ""
        print(f"  {action:<15} {efes.get(action, 0):.4f}{marker}")
    
    passed = selected == 'retreat'
    print(f"\n{'✓' if passed else '✗'} Retreat selected under dual threat.")
    return passed


def run_threat_viability_test():
    print("\n" + "=" * 80)
    print("TEST 6: THREAT UPPER VIABILITY BOUND")
    print("=" * 80)
    print(f"\n{'Threat':<8} {'wait':<10} {'observe':<10} {'retreat':<10} "
          f"{'reflect':<10} {'ground-c':<10} {'Selected':<10}")
    print("-" * 72)
    
    test_threats = [0.15, 0.35, 0.50, 0.60, 0.70]
    selections = []
    
    for threat in test_threats:
        core = ViabilityAwareCognitiveCore()
        core.update_belief("threat", threat, 0.8)
        core.update_belief("self-comp", 0.75, 0.8)
        efes = core.get_efes()
        selected = min(efes, key=efes.get)
        selections.append(selected)
        
        print(f"{threat:<8.2f} {efes.get('wait',0):<10.3f} "
              f"{efes.get('observe',0):<10.3f} {efes.get('retreat',0):<10.3f} "
              f"{efes.get('reflect',0):<10.3f} {efes.get('ground-concept',0):<10.3f} "
              f"{selected:<10}")
    
    passed = (selections[0] != 'retreat') and (selections[-1] == 'retreat')
    print(f"\n{'✓' if passed else '✗'} Upper viability bound triggers retreat at high threat.")
    return passed


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    run_efe_decomposition()
    results = [
        run_power_viability_test(),
        run_self_competence_viability_test(),
        run_grounding_pressure_test(),
        run_multi_cycle_test(),
        run_dual_viability_test(),
        run_threat_viability_test(),
    ]
    print(f"\nOVERALL: {sum(results)}/{len(results)} tests passed")
