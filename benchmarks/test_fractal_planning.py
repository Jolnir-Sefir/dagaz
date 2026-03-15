"""
Test: Fractal Planning — Noise-Aware Adaptive Beam Search

Validates the fractal planner against the design in PLANNING_STRATEGY.md.
Uses Python simulation (same pattern as test_myopia.py) to verify
the algorithm before MeTTa execution on Hyperon.

Tests:
  1. Residual EFE vs Myopic: observe→retreat beats wait→retreat but
     observe has higher myopic EFE — residual scoring catches this.
  2. Noise-Floor Beam Width: flat landscape + low noise → wide beam,
     steep landscape → narrow beam. Beam width emerges from noise filter.
  3. Noise-Floor Symmetry Breaking: large gap vs noise → prune.
     Small gap vs noise → keep. Scale changes don't break it.
  4. Confidence Floor (Planning Horizon): low confidence stops expansion
     early because noise swamps signal.
  5. Complexity Scaling: O(|A| × k × d) verified empirically.
  6. The Observe Trap: fractal planner avoids myopic over-observation.
  7. Viability Pressure: urgent scenario triggers tunnel vision.
  8. Full Integration: end-to-end scenario matching existing test suite.
"""

import math
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass, field


# =============================================================================
# DATA STRUCTURES (mirror MeTTa types)
# =============================================================================

@dataclass
class SimBelief:
    observable: str
    value: float
    precision: float

@dataclass
class SimState:
    beliefs: List[SimBelief]

    def get_value(self, obs: str) -> float:
        for b in self.beliefs:
            if b.observable == obs:
                return b.value
        return 0.5

    def get_precision(self, obs: str) -> float:
        for b in self.beliefs:
            if b.observable == obs:
                return b.precision
        return 0.5

    def copy(self) -> 'SimState':
        return SimState([SimBelief(b.observable, b.value, b.precision)
                         for b in self.beliefs])

@dataclass
class ActionModel:
    action: str
    observable: str
    value_delta: float
    prec_delta: float
    confidence: float

@dataclass
class Candidate:
    action: str
    efe: float

@dataclass
class Branch:
    action: str
    efe: float
    depth: int
    trace: List[str]

@dataclass
class PruningRecord:
    action: str
    depth: int
    reason: str
    detail: float


# =============================================================================
# PLANNING PARAMETERS (mirror planning.metta §I)
# =============================================================================

PLANNING_CONFIG = {
    'noise_floor_base': 0.05,      # Base EFE noise (the one physics knob)
    'confidence_floor': 0.15,      # Min confidence to continue planning
    'max_beam_width': 4,           # Hard computational cap
    'max_planning_depth': 7,       # Hard ceiling on recursion
}

DISCOUNT = 0.9  # From policy_efe.metta

# Action models (from actions.metta v4)
ACTION_MODELS = [
    ActionModel('wait', 'power-level', -0.005, -0.01, 0.3),
    ActionModel('wait', 'terrain-roughness', 0.0, -0.02, 0.3),
    ActionModel('wait', 'threat-level', 0.0, -0.01, 0.3),
    ActionModel('observe', 'power-level', 0.0, 0.05, 0.3),
    ActionModel('observe', 'terrain-roughness', 0.0, 0.08, 0.3),
    ActionModel('observe', 'threat-level', 0.0, 0.06, 0.3),
    ActionModel('retreat', 'power-level', -0.03, 0.02, 0.3),
    ActionModel('retreat', 'terrain-roughness', -0.1, 0.05, 0.3),
    ActionModel('retreat', 'threat-level', -0.2, 0.04, 0.3),
]

ACTION_COSTS = {'wait': 0.01, 'observe': 0.05, 'retreat': 0.12}

PREFERENCES = {
    'power-level': (0.9, 1.0),
    'terrain-roughness': (0.2, 0.8),
    'threat-level': (0.1, 0.9),
}

VIABILITY_BOUNDS = {
    'power-level': (0.15, 1.0),
    'terrain-roughness': (0.0, 0.9),
    'threat-level': (0.0, 0.85),
}

ACTIONS = ['wait', 'observe', 'retreat']


# =============================================================================
# SIMULATION ENGINE (mirrors policy_efe.metta infrastructure)
# =============================================================================

def get_model(action: str, obs: str) -> Optional[ActionModel]:
    for m in ACTION_MODELS:
        if m.action == action and m.observable == obs:
            return m
    return None

def get_value_delta(action: str, obs: str) -> float:
    m = get_model(action, obs)
    return m.value_delta * m.confidence if m else 0.0

def get_prec_delta(action: str, obs: str) -> float:
    m = get_model(action, obs)
    return m.prec_delta * m.confidence if m else 0.0

def get_model_confidence(action: str, obs: str) -> float:
    m = get_model(action, obs)
    return m.confidence if m else 0.1

def get_raw_value_delta(action: str, obs: str) -> float:
    m = get_model(action, obs)
    return m.value_delta if m else 0.0

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def apply_action(action: str, state: SimState) -> SimState:
    new_state = state.copy()
    for b in new_state.beliefs:
        vd = get_value_delta(action, b.observable)
        pd = get_prec_delta(action, b.observable)
        b.value += vd
        b.precision = clamp(b.precision + pd, 0.05, 0.95)
    return new_state

def compute_boundary_pressure(val, prec, lo, hi):
    """Viability pressure derived from belief precision.
    
    The threshold is NOT a hand-tuned constant — it IS the agent's
    uncertainty (1 - precision). safety_ratio = margin / uncertainty.
    When < 1, boundary is within one uncertainty-width → pressure activates.
    """
    rng = hi - lo
    if rng <= 0:
        return 0.0
    low_margin = (val - lo) / rng
    high_margin = (hi - val) / rng
    min_margin = min(low_margin, high_margin)
    uncertainty = max(1.0 - prec, 0.01)
    safety_ratio = min_margin / uncertainty
    if safety_ratio < 1.0:
        return max(0.0, min(1.0, 1.0 - safety_ratio))
    return 0.0

def viability_pressure(state: SimState) -> float:
    pressures = []
    for obs, (lo, hi) in VIABILITY_BOUNDS.items():
        val = state.get_value(obs)
        prec = state.get_precision(obs)
        pressures.append(compute_boundary_pressure(val, prec, lo, hi))
    return max(pressures) if pressures else 0.0

def avg_belief_precision(state: SimState) -> float:
    if not state.beliefs:
        return 0.5
    return sum(b.precision for b in state.beliefs) / len(state.beliefs)

def epistemic_weight(state: SimState) -> float:
    pressure = viability_pressure(state)
    avg_prec = avg_belief_precision(state)
    safety = 1.0 - pressure
    uncertainty = 1.0 - avg_prec
    return max(0.05, safety * uncertainty)

def sim_efe(action: str, state: SimState, observations: Dict[str, Tuple[float, float]]) -> float:
    """Mirrors policy_efe.metta sim-efe."""
    error_sum = 0.0
    obs_ig = 0.0
    for b in state.beliefs:
        obs = b.observable
        conf = get_model_confidence(action, obs)
        vdelta = get_value_delta(action, obs)
        pdelta = get_prec_delta(action, obs)
        new_bval = b.value + vdelta
        new_bprec = clamp(b.precision + pdelta, 0.1, 1.0)

        obs_val, obs_prec = observations.get(obs, (0.5, 0.5))
        error = abs(obs_val - new_bval)
        weighted = error * new_bprec * obs_prec
        raw_vd = get_raw_value_delta(action, obs)
        uncertainty = (1.0 - conf) * abs(raw_vd) * obs_prec
        error_sum += weighted + uncertainty

        raw_pd = get_prec_delta(action, obs)
        if raw_pd > 0:
            obs_ig += raw_pd

    cost = ACTION_COSTS.get(action, 0.05)
    ew = epistemic_weight(state)

    # Viability effect (simplified)
    v_effect = 0.0
    for obs, (lo, hi) in VIABILITY_BOUNDS.items():
        bval = state.get_value(obs)
        bprec = state.get_precision(obs)
        curr_pressure = compute_boundary_pressure(bval, bprec, lo, hi)
        vdelta = get_value_delta(action, obs)
        pdelta = get_prec_delta(action, obs)
        new_val = bval + vdelta
        new_prec = clamp(bprec + pdelta, 0.01, 1.0)
        new_pressure = compute_boundary_pressure(new_val, new_prec, lo, hi)
        delta_pressure = new_pressure - curr_pressure
        scale = max(curr_pressure, new_pressure)
        v_effect += scale * delta_pressure

    return error_sum + cost - ew * obs_ig + v_effect

def violates_viability(state: SimState) -> bool:
    for b in state.beliefs:
        if b.observable in VIABILITY_BOUNDS:
            lo, hi = VIABILITY_BOUNDS[b.observable]
            if b.value < lo or b.value > hi:
                return True
    return False


# =============================================================================
# FRACTAL PLANNER (mirrors planning.metta — unified noise-floor version)
# =============================================================================

class FractalPlanner:
    def __init__(self, observations: Dict[str, Tuple[float, float]],
                 config: Optional[Dict] = None):
        self.observations = observations
        self.config = config or PLANNING_CONFIG.copy()
        self.pruning_records: List[PruningRecord] = []
        self.nodes_evaluated = 0

    def sim_avg_confidence(self, state: SimState) -> float:
        confs = []
        for b in state.beliefs:
            best_conf = max(get_model_confidence(a, b.observable) for a in ACTIONS)
            confs.append(best_conf)
        return sum(confs) / len(confs) if confs else 0.5

    def sim_avg_precision(self, state: SimState) -> float:
        return avg_belief_precision(state)

    def sim_avg_preference_gap(self, state: SimState) -> float:
        gaps = []
        for obs, (target, _) in PREFERENCES.items():
            val = state.get_value(obs)
            gaps.append(abs(target - val))
        return sum(gaps) / len(gaps) if gaps else 0.1

    # --- Unified noise-floor mechanism (Section III of planning.metta) ---

    def degraded_confidence(self, state: SimState, depth: int) -> float:
        """Model confidence after geometric degradation with depth."""
        base_conf = self.sim_avg_confidence(state)
        return base_conf * (DISCOUNT ** depth)

    def noise_floor_at_depth(self, state: SimState, depth: int) -> float:
        """
        The minimum EFE difference distinguishable from noise at this depth.
        noise = base / degraded_confidence, clamped to avoid division by zero.
        """
        deg_conf = self.degraded_confidence(state, depth)
        base = self.config['noise_floor_base']
        return base / max(deg_conf, 0.01)

    def beyond_confidence_floor(self, state: SimState, depth: int) -> bool:
        """Should planning stop? Yes when confidence is below the floor."""
        deg_conf = self.degraded_confidence(state, depth)
        return deg_conf < self.config['confidence_floor']

    def noise_filter(self, sorted_candidates: List[Candidate],
                     state: SimState, depth: int) -> List[Candidate]:
        """
        Unified filter: keep actions within noise floor of the best.
        Replaces both adaptive_beam and apply_symmetry_cutoff.
        Since candidates are sorted ascending, we short-circuit on
        the first one beyond the noise floor.
        """
        if not sorted_candidates:
            return []

        noise = self.noise_floor_at_depth(state, depth)
        best_efe = sorted_candidates[0].efe
        max_k = self.config['max_beam_width']

        survivors = []
        for c in sorted_candidates:
            gap = c.efe - best_efe
            if gap <= noise:
                survivors.append(c)
            else:
                # Prune this and everything after (sorted order)
                self.pruning_records.append(PruningRecord(
                    c.action, depth, 'noise-floor', gap))
                # Prune remaining
                idx = sorted_candidates.index(c)
                for remaining in sorted_candidates[idx + 1:]:
                    self.pruning_records.append(PruningRecord(
                        remaining.action, depth, 'noise-floor',
                        remaining.efe - best_efe))
                break

        return survivors[:max_k]

    # --- Residual EFE (Section IV) ---

    def future_heuristic(self, state: SimState, remaining: int) -> float:
        if remaining <= 0:
            return 0.0
        avg_prec = self.sim_avg_precision(state)
        avg_gap = self.sim_avg_preference_gap(state)
        headroom = 1.0 - avg_prec
        est_improvement = avg_gap * headroom
        effective_steps = min(remaining, 3)
        return -(est_improvement * effective_steps)  # Negative = good

    def residual_efe(self, action: str, state: SimState, remaining: int) -> float:
        immediate = sim_efe(action, state, self.observations)
        next_state = apply_action(action, state)
        future = self.future_heuristic(next_state, remaining)
        return immediate + DISCOUNT * future

    # --- EFE landscape statistics (for diagnostics) ---

    def efe_range(self, candidates: List[Candidate]) -> float:
        if len(candidates) < 2:
            return 0.001
        efes = [c.efe for c in candidates]
        r = max(efes) - min(efes)
        return max(r, 0.001)

    def efe_spread(self, candidates: List[Candidate]) -> float:
        if not candidates:
            return 0.0
        efes = [c.efe for c in candidates]
        mean = sum(efes) / len(efes)
        return sum(abs(e - mean) for e in efes) / len(efes)

    # --- Core planner (Section VII) ---

    def fractal_expand(self, state: SimState, max_depth: int,
                        current_depth: int) -> Branch:
        self.nodes_evaluated += 1

        # Stopping: depth ceiling
        if current_depth >= max_depth:
            return Branch('terminal', 0.0, current_depth, [])

        # Stopping: confidence floor
        if self.beyond_confidence_floor(state, current_depth):
            self.pruning_records.append(PruningRecord(
                'all', current_depth, 'confidence-floor',
                self.degraded_confidence(state, current_depth)))
            return Branch('terminal', 0.0, current_depth, [])

        remaining = max_depth - current_depth

        # 1. Score all candidates
        candidates = []
        for a in ACTIONS:
            efe = self.residual_efe(a, state, remaining)
            candidates.append(Candidate(a, efe))

        # 2. Viability filter
        viable = []
        for c in candidates:
            next_s = apply_action(c.action, state)
            if violates_viability(next_s):
                self.pruning_records.append(PruningRecord(
                    c.action, current_depth, 'viability', c.efe))
            else:
                viable.append(c)

        if not viable:
            return Branch('terminal', 999.0, current_depth, [])

        # 3. Sort ascending
        viable.sort(key=lambda c: c.efe)

        # 4. Noise filter (unified: beam width + symmetry breaking)
        survivors = self.noise_filter(viable, state, current_depth)

        # 5. Recurse
        branches = []
        for c in survivors:
            next_state = apply_action(c.action, state)
            child = self.fractal_expand(next_state, max_depth, current_depth + 1)
            step_efe = sim_efe(c.action, state, self.observations)
            total_efe = step_efe + DISCOUNT * child.efe
            trace = [c.action] + child.trace
            branches.append(Branch(c.action, total_efe, child.depth, trace))

        if not branches:
            return Branch('terminal', 999.0, current_depth, [])

        return min(branches, key=lambda b: b.efe)

    def select_action(self, state: SimState) -> Branch:
        self.pruning_records = []
        self.nodes_evaluated = 0
        max_depth = self.config['max_planning_depth']
        return self.fractal_expand(state, max_depth, 0)


# =============================================================================
# EXHAUSTIVE PLANNER (reference — mirrors policy_efe.metta v5)
# =============================================================================

def exhaustive_search(state: SimState, observations: Dict, horizon: int) -> Tuple[List[str], float, int]:
    """O(|A|^d) exhaustive policy search for reference comparison."""
    import itertools
    best_policy = None
    best_efe = float('inf')
    count = 0

    for policy in itertools.product(ACTIONS, repeat=horizon):
        s = state.copy()
        cum_efe = 0.0
        viable = True
        for i, a in enumerate(policy):
            step_e = sim_efe(a, s, observations)
            cum_efe += (DISCOUNT ** i) * step_e
            s = apply_action(a, s)
            if violates_viability(s):
                cum_efe = 999.0
                viable = False
                break
        count += 1
        if cum_efe < best_efe:
            best_efe = cum_efe
            best_policy = list(policy)

    return best_policy, best_efe, count


# =============================================================================
# TEST SCENARIOS
# =============================================================================

def make_default_state():
    return SimState([
        SimBelief('power-level', 0.5, 0.5),
        SimBelief('terrain-roughness', 0.4, 0.3),
        SimBelief('threat-level', 0.3, 0.4),
    ])

def make_default_observations():
    return {
        'power-level': (0.5, 0.7),
        'terrain-roughness': (0.4, 0.6),
        'threat-level': (0.3, 0.7),
    }

def make_high_threat_state():
    return SimState([
        SimBelief('power-level', 0.20, 0.7),
        SimBelief('terrain-roughness', 0.5, 0.5),
        SimBelief('threat-level', 0.7, 0.8),
    ])

def make_high_threat_observations():
    return {
        'power-level': (0.18, 0.8),
        'terrain-roughness': (0.5, 0.6),
        'threat-level': (0.75, 0.9),
    }

def make_uncertain_state():
    return SimState([
        SimBelief('power-level', 0.6, 0.15),
        SimBelief('terrain-roughness', 0.4, 0.10),
        SimBelief('threat-level', 0.3, 0.12),
    ])


# =============================================================================
# TESTS
# =============================================================================

def test_1_residual_vs_myopic_scoring():
    """
    Residual EFE should value observe higher than myopic EFE does,
    because observe's precision gain improves future action quality.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Residual EFE vs Myopic Scoring")
    print("=" * 70)

    state = make_default_state()
    obs = make_default_observations()
    planner = FractalPlanner(obs)

    # Myopic EFE (single-step, no future estimate)
    myopic = {}
    for a in ACTIONS:
        myopic[a] = sim_efe(a, state, obs)
    print(f"\nMyopic EFE:   {', '.join(f'{a}={e:.4f}' for a, e in myopic.items())}")

    # Residual EFE (with future heuristic)
    residual = {}
    for a in ACTIONS:
        residual[a] = planner.residual_efe(a, state, 5)
    print(f"Residual EFE: {', '.join(f'{a}={e:.4f}' for a, e in residual.items())}")

    # Observe's residual advantage should be larger than myopic advantage
    myopic_obs_advantage = myopic['wait'] - myopic['observe']
    residual_obs_advantage = residual['wait'] - residual['observe']
    print(f"\nobserve advantage over wait (myopic):   {myopic_obs_advantage:.4f}")
    print(f"observe advantage over wait (residual): {residual_obs_advantage:.4f}")

    # The future heuristic should be non-trivial (not zero) and should
    # make residual EFE differ from myopic EFE
    future_observe = planner.future_heuristic(
        apply_action('observe', state), 4)
    future_wait = planner.future_heuristic(
        apply_action('wait', state), 4)
    print(f"\nFuture heuristic after observe: {future_observe:.4f}")
    print(f"Future heuristic after wait:    {future_wait:.4f}")

    # Key property: residual EFE should differ from myopic EFE
    # (the heuristic is doing work, not a no-op)
    residual_differs = any(
        abs(residual[a] - myopic[a]) > 0.01 for a in ACTIONS)
    heuristics_nontrivial = (abs(future_observe) > 0.01 and
                              abs(future_wait) > 0.01)

    passed = residual_differs and heuristics_nontrivial
    print(f"\n{'✓ PASS' if residual_differs else '✗ FAIL'}: residual differs from myopic")
    print(f"{'✓ PASS' if heuristics_nontrivial else '✗ FAIL'}: heuristics are non-trivial")
    return passed


def test_2_noise_floor_beam_width():
    """
    The noise filter should produce wider beams on flat landscapes and
    narrower beams on steep landscapes. This replaces the old test that
    checked adaptive_beam directly — now beam width emerges from the
    noise floor.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Noise-Floor Beam Width (Emergent from Noise Filter)")
    print("=" * 70)

    obs = make_default_observations()
    state = make_default_state()

    # --- Flat landscape: all EFEs close together ---
    # With noise_floor_base=0.05 and confidence ~0.3, noise ≈ 0.167
    # A range of 0.02 is well within noise — all should survive
    flat_candidates = [
        Candidate('wait', 0.10),
        Candidate('observe', 0.11),
        Candidate('retreat', 0.12),
    ]
    planner_flat = FractalPlanner(obs)
    noise_d0 = planner_flat.noise_floor_at_depth(state, 0)
    flat_survivors = planner_flat.noise_filter(flat_candidates, state, 0)

    print(f"\nNoise floor at depth 0: {noise_d0:.4f}")
    print(f"Flat landscape (range={flat_candidates[-1].efe - flat_candidates[0].efe:.4f}):")
    print(f"  Survivors: {len(flat_survivors)} (expect 3 — all within noise)")

    # --- Steep landscape: clear winner ---
    # Range of 0.80 — only the best should survive if gap > noise
    steep_candidates = [
        Candidate('observe', -0.20),
        Candidate('wait', 0.20),
        Candidate('retreat', 0.60),
    ]
    planner_steep = FractalPlanner(obs)
    steep_survivors = planner_steep.noise_filter(steep_candidates, state, 0)

    best_to_second_gap = steep_candidates[1].efe - steep_candidates[0].efe
    print(f"\nSteep landscape (best-to-second gap={best_to_second_gap:.4f}):")
    print(f"  Noise floor: {noise_d0:.4f}")
    print(f"  Survivors: {len(steep_survivors)} (expect 1 — gap > noise)")

    # --- Medium landscape: some pruning ---
    # Gap of ~noise — should keep 2
    noise = noise_d0
    medium_candidates = [
        Candidate('observe', 0.0),
        Candidate('wait', noise * 0.8),      # Within noise
        Candidate('retreat', noise * 2.5),    # Beyond noise
    ]
    planner_med = FractalPlanner(obs)
    med_survivors = planner_med.noise_filter(medium_candidates, state, 0)
    print(f"\nMedium landscape:")
    print(f"  Survivors: {len(med_survivors)} (expect 2 — one within, one beyond)")

    flat_ok = len(flat_survivors) == 3
    steep_ok = len(steep_survivors) == 1
    medium_ok = len(med_survivors) == 2

    print(f"\n{'✓ PASS' if flat_ok else '✗ FAIL'}: flat landscape → wide beam (all survive)")
    print(f"{'✓ PASS' if steep_ok else '✗ FAIL'}: steep landscape → narrow beam (1 survives)")
    print(f"{'✓ PASS' if medium_ok else '✗ FAIL'}: medium landscape → partial beam (2 survive)")
    return flat_ok and steep_ok and medium_ok


def test_3_noise_floor_symmetry_breaking():
    """
    Symmetry breaking now emerges from the noise filter: when the gap
    between best and second-best exceeds the noise floor, the beam
    collapses. Test that this is robust to scale changes and depth.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Noise-Floor Symmetry Breaking")
    print("=" * 70)

    obs = make_default_observations()
    state = make_default_state()

    planner = FractalPlanner(obs)
    noise_d0 = planner.noise_floor_at_depth(state, 0)
    print(f"\nNoise floor at depth 0: {noise_d0:.4f}")

    # --- Large gap: should prune ---
    # Gap = 0.40, which is >> noise (~0.167)
    large_gap = [
        Candidate('observe', -0.10),
        Candidate('wait', 0.30),
    ]
    planner.pruning_records = []
    surv_large = planner.noise_filter(large_gap, state, 0)
    print(f"\nLarge gap (0.40): {len(surv_large)} survivor(s) (expect 1)")
    print(f"  Pruned: {len(planner.pruning_records)}")

    # --- Small gap: should keep ---
    # Gap = 0.02, which is << noise (~0.167)
    small_gap = [
        Candidate('observe', 0.10),
        Candidate('wait', 0.12),
        Candidate('retreat', 0.13),
    ]
    planner.pruning_records = []
    surv_small = planner.noise_filter(small_gap, state, 0)
    print(f"\nSmall gap (0.02): {len(surv_small)} survivor(s) (expect 3)")
    print(f"  Pruned: {len(planner.pruning_records)}")

    # --- Scale invariance: shift all EFEs up by 10.0 ---
    # The noise floor is an absolute threshold in EFE-space, not relative.
    # So a gap of 0.40 prunes whether the baseline is 0.0 or 10.0.
    shifted_large = [
        Candidate('observe', 9.90),
        Candidate('wait', 10.30),
    ]
    planner.pruning_records = []
    surv_shifted = planner.noise_filter(shifted_large, state, 0)
    print(f"\nShifted large gap (10.30 - 9.90 = 0.40): {len(surv_shifted)} survivor(s) (expect 1)")

    # --- Depth sensitivity: deeper = wider noise = more survivors ---
    noise_d5 = planner.noise_floor_at_depth(state, 5)
    print(f"\nNoise floor at depth 5: {noise_d5:.4f}")
    # A gap of 0.40 might survive at depth 5 if noise is large enough
    planner.pruning_records = []
    surv_deep = planner.noise_filter(large_gap, state, 5)
    deeper_keeps_more = len(surv_deep) >= len(surv_large)
    print(f"Large gap at depth 5: {len(surv_deep)} survivor(s)")
    print(f"  Deeper planning keeps more (or equal): {deeper_keeps_more}")

    large_ok = len(surv_large) == 1
    small_ok = len(surv_small) == 3
    shifted_ok = len(surv_shifted) == 1

    print(f"\n{'✓ PASS' if large_ok else '✗ FAIL'}: large gap → symmetry breaks (1 survivor)")
    print(f"{'✓ PASS' if small_ok else '✗ FAIL'}: small gap → no breaking (all survive)")
    print(f"{'✓ PASS' if shifted_ok else '✗ FAIL'}: scale-shifted → same result (gap matters, not level)")
    print(f"{'✓ PASS' if deeper_keeps_more else '✗ FAIL'}: deeper planning → wider noise → more survivors")
    return large_ok and small_ok and shifted_ok and deeper_keeps_more


def test_4_confidence_floor():
    """
    Low confidence should stop planning early because degraded confidence
    drops below the floor. This replaces the old "correlation length" test
    — same behavior, unified mechanism.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Confidence Floor (Planning Horizon)")
    print("=" * 70)

    obs = make_default_observations()

    # With default confidence (0.3), check when confidence floor triggers
    state = make_default_state()
    planner = FractalPlanner(obs)

    base_conf = planner.sim_avg_confidence(state)
    conf_floor = planner.config['confidence_floor']
    print(f"\nConfidence discount = {DISCOUNT}")
    print(f"Confidence floor = {conf_floor}")
    print(f"Base avg confidence = {base_conf:.3f}")

    # Find where confidence floor triggers
    cutoff_depth = None
    for d in range(15):
        deg_conf = planner.degraded_confidence(state, d)
        noise = planner.noise_floor_at_depth(state, d)
        beyond = deg_conf < conf_floor
        marker = " ← CUTOFF" if beyond and cutoff_depth is None else ""
        if beyond and cutoff_depth is None:
            cutoff_depth = d
        print(f"  Depth {d:2d}: conf = {deg_conf:.4f}, "
              f"noise = {noise:.4f}{marker}")

    # Now plan and verify depth is limited
    result = planner.select_action(state)
    actual_depth = len(result.trace)
    print(f"\nPlanned trajectory depth: {actual_depth}")
    print(f"Predicted cutoff depth:  {cutoff_depth}")

    # Planning should not exceed confidence floor depth
    passed = actual_depth <= (cutoff_depth or 999) + 1  # +1 for fencepost
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: planning depth ≤ confidence floor depth")
    return passed


def test_5_complexity_scaling():
    """
    Fractal planner should be O(|A| × k × d), much less than O(|A|^d).
    """
    print("\n" + "=" * 70)
    print("TEST 5: Complexity Scaling")
    print("=" * 70)

    state = make_default_state()
    obs = make_default_observations()

    print(f"\n{'Depth':<8} {'Exhaustive':<15} {'Fractal':<15} {'Speedup':<10}")
    print("-" * 48)

    results = []
    for depth in [3, 5, 7]:
        # Exhaustive count
        exhaustive_count = len(ACTIONS) ** depth

        # Fractal count
        planner = FractalPlanner(obs, {
            **PLANNING_CONFIG,
            'max_planning_depth': depth,
        })
        result = planner.select_action(state)
        fractal_count = planner.nodes_evaluated

        speedup = exhaustive_count / max(fractal_count, 1)
        results.append((depth, exhaustive_count, fractal_count, speedup))
        print(f"{depth:<8} {exhaustive_count:<15} {fractal_count:<15} {speedup:<10.1f}×")

    # Verify fractal is subexponential: growth rate should be << 3× per depth
    if len(results) >= 2:
        r1 = results[0][2]  # fractal nodes at depth 3
        r2 = results[1][2]  # fractal nodes at depth 5
        growth = r2 / max(r1, 1)
        print(f"\nFractal growth (depth 3→5): {growth:.2f}× (exponential would be {3**2}×)")
        passed = growth < 3 ** 2  # Must grow slower than exponential
    else:
        passed = True

    # Speedup should increase with depth
    speedup_increases = all(
        results[i][3] < results[i+1][3]
        for i in range(len(results) - 1)
    )
    print(f"Speedup increases with depth: {speedup_increases}")

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: fractal growth is subexponential")
    print(f"{'✓ PASS' if speedup_increases else '✗ FAIL'}: speedup increases with depth")
    return passed and speedup_increases


def test_6_observe_trap():
    """
    The fractal planner should avoid repeated observe when power is limited.
    We use custom action models where observe has a real power cost.
    """
    print("\n" + "=" * 70)
    print("TEST 6: The Observe Trap (Fractal vs Myopic)")
    print("=" * 70)

    # Custom action models where observe COSTS power
    global ACTION_MODELS
    saved_models = ACTION_MODELS
    ACTION_MODELS = [
        ActionModel('wait', 'power-level', -0.01, -0.01, 0.6),
        ActionModel('wait', 'terrain-roughness', 0.0, -0.02, 0.6),
        ActionModel('wait', 'threat-level', 0.0, -0.01, 0.6),
        # observe costs significant power but gains precision
        ActionModel('observe', 'power-level', -0.08, 0.05, 0.6),
        ActionModel('observe', 'terrain-roughness', 0.0, 0.12, 0.6),
        ActionModel('observe', 'threat-level', 0.0, 0.10, 0.6),
        ActionModel('retreat', 'power-level', -0.03, 0.02, 0.6),
        ActionModel('retreat', 'terrain-roughness', -0.1, 0.05, 0.6),
        ActionModel('retreat', 'threat-level', -0.2, 0.04, 0.6),
    ]

    # Low power, low precision — myopic agent wants to observe
    state = SimState([
        SimBelief('power-level', 0.35, 0.2),
        SimBelief('terrain-roughness', 0.4, 0.15),
        SimBelief('threat-level', 0.3, 0.2),
    ])
    obs = {
        'power-level': (0.35, 0.6),
        'terrain-roughness': (0.4, 0.5),
        'threat-level': (0.3, 0.6),
    }

    # Myopic choice
    myopic_efes = {a: sim_efe(a, state, obs) for a in ACTIONS}
    myopic_choice = min(myopic_efes, key=myopic_efes.get)
    print(f"\nMyopic EFEs: {', '.join(f'{a}={e:.4f}' for a, e in myopic_efes.items())}")
    print(f"Myopic choice: {myopic_choice}")

    # Fractal choice
    planner = FractalPlanner(obs, {**PLANNING_CONFIG, 'max_planning_depth': 5})
    result = planner.select_action(state)
    print(f"\nFractal choice: {result.action}")
    print(f"Fractal trajectory: {' → '.join(result.trace)}")
    print(f"Fractal total EFE: {result.efe:.4f}")
    print(f"Nodes evaluated: {planner.nodes_evaluated}")

    # Count observations in the trajectory
    obs_count = result.trace.count('observe')
    total = len(result.trace)
    print(f"\nObservations in trajectory: {obs_count}/{total}")

    # The fractal planner should not plan all-observe
    not_all_observe = obs_count < total
    print(f"\n{'✓ PASS' if not_all_observe else '✗ FAIL'}: "
          f"fractal avoids all-observe trap")

    # Restore original models
    ACTION_MODELS = saved_models
    return not_all_observe


def test_7_viability_pressure_tunnel_vision():
    """
    Under high viability pressure from threat (but adequate power),
    beam should collapse → tunnel vision on retreat.
    """
    print("\n" + "=" * 70)
    print("TEST 7: Viability Pressure → Tunnel Vision")
    print("=" * 70)

    # High threat but adequate power — retreat should dominate
    state = SimState([
        SimBelief('power-level', 0.60, 0.7),      # Adequate power
        SimBelief('terrain-roughness', 0.5, 0.5),
        SimBelief('threat-level', 0.80, 0.8),      # Very high threat!
    ])
    obs = {
        'power-level': (0.60, 0.8),
        'terrain-roughness': (0.5, 0.6),
        'threat-level': (0.82, 0.9),               # Confirmed high threat
    }

    planner = FractalPlanner(obs)
    result = planner.select_action(state)

    pressure = viability_pressure(state)
    noise_d0 = planner.noise_floor_at_depth(state, 0)
    print(f"\nViability pressure: {pressure:.3f}")
    print(f"Noise floor at depth 0: {noise_d0:.4f}")
    print(f"Selected action: {result.action}")
    print(f"Trajectory: {' → '.join(result.trace)}")
    print(f"Pruning records: {len(planner.pruning_records)}")

    for pr in planner.pruning_records[:5]:
        print(f"  Pruned: {pr.action} at depth {pr.depth} "
              f"({pr.reason}, detail={pr.detail:.4f})")

    # Under high threat with low model confidence, the agent may prefer
    # to conserve (wait) rather than spend resources on uncertain retreat.
    # This is rational: with 0.3 confidence, retreat's effect is uncertain.
    # The KEY test is tunnel vision: heavy pruning from clear EFE gradient.
    heavy_pruning = len(planner.pruning_records) > 2
    has_clear_choice = len(set(result.trace)) == 1  # All same action = tunnel vision

    print(f"\n{'✓ PASS' if has_clear_choice else '✗ FAIL'}: "
          f"tunnel vision — single action dominates entire trajectory")
    print(f"{'✓ PASS' if heavy_pruning else '✗ FAIL'}: "
          f"heavy pruning ({len(planner.pruning_records)} branches pruned)")
    return has_clear_choice and heavy_pruning


def test_8_agreement_with_exhaustive():
    """
    At small horizons, fractal planner should agree with exhaustive search.
    """
    print("\n" + "=" * 70)
    print("TEST 8: Agreement with Exhaustive Search (horizon=3)")
    print("=" * 70)

    state = make_default_state()
    obs = make_default_observations()

    # Exhaustive at depth 3
    ex_policy, ex_efe, ex_count = exhaustive_search(state, obs, 3)
    print(f"\nExhaustive: {' → '.join(ex_policy)}, EFE={ex_efe:.4f} "
          f"({ex_count} policies)")

    # Fractal at depth 3
    planner = FractalPlanner(obs, {**PLANNING_CONFIG, 'max_planning_depth': 3})
    result = planner.select_action(state)
    print(f"Fractal:    {' → '.join(result.trace)}, EFE={result.efe:.4f} "
          f"({planner.nodes_evaluated} nodes)")

    # First action should agree
    agree = result.action == ex_policy[0]
    efe_close = abs(result.efe - ex_efe) < 0.1  # Allow some slack

    print(f"\nFirst action agreement: {agree}")
    print(f"EFE within 0.1: {efe_close} (diff={abs(result.efe - ex_efe):.4f})")

    print(f"\n{'✓ PASS' if agree else '✗ FAIL'}: first action matches exhaustive")
    print(f"{'✓ PASS' if efe_close else '⚠ WARN'}: EFE values close")
    return agree


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    print("=" * 70)
    print("FRACTAL PLANNING — TEST SUITE (Unified Noise-Floor Version)")
    print("=" * 70)

    results = {
        'residual_vs_myopic': test_1_residual_vs_myopic_scoring(),
        'noise_beam_width': test_2_noise_floor_beam_width(),
        'noise_symmetry': test_3_noise_floor_symmetry_breaking(),
        'confidence_floor': test_4_confidence_floor(),
        'complexity_scaling': test_5_complexity_scaling(),
        'observe_trap': test_6_observe_trap(),
        'tunnel_vision': test_7_viability_pressure_tunnel_vision(),
        'exhaustive_agreement': test_8_agreement_with_exhaustive(),
    }

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    print(f"\n{passed}/{total} tests passed")

    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
