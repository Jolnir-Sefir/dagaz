"""
dagaz_planner.py — Python Fractal Planner for the Dagaz Cognitive Server.

Canonical specification: planning.metta + policy_efe.metta
EFE formula: test_efe.py (Tier 2 benchmark)

This is the execution path. The MeTTa source is the specification.
Same relationship as Perception (dagaz_server_v2.py) to
structure_learning.metta — identical formulas, Python execution.

Hot loops are for Python. Structure is for MeTTa.
"""

import math
import time as _time
from dagaz_runtime import parse

# Pre-computed query patterns (parsed once at import, not per tick)
_PAT_ACTION_MODEL = parse('(action-model $a $o $vd $pd $c)')[0]
_PAT_ACTION_COST = parse('(action-cost $a $c $conf)')[0]
_PAT_PREFERENCE = parse('(preference $o $v $i)')[0]
_PAT_VIABILITY = parse('(viability-bound $o $lo $hi)')[0]
_PAT_BELIEF = parse('(belief $o $v $p)')[0]

# Inline safe_float to avoid import dependency issues
def _sf(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SimBelief:
    __slots__ = ('obs', 'val', 'prec')
    def __init__(self, obs, val, prec):
        self.obs = obs
        self.val = val
        self.prec = prec

class SimState:
    __slots__ = ('beliefs',)
    def __init__(self, beliefs):
        self.beliefs = beliefs  # list[SimBelief]

    def get(self, obs):
        """Return (value, precision) for an observable, or (0.5, 0.5)."""
        for b in self.beliefs:
            if b.obs == obs:
                return b.val, b.prec
        return 0.5, 0.5

    def copy(self):
        return SimState([SimBelief(b.obs, b.val, b.prec) for b in self.beliefs])

    def avg_precision(self):
        if not self.beliefs:
            return 0.5
        return sum(b.prec for b in self.beliefs) / len(self.beliefs)


# =============================================================================
# ACTION MOMENTUM — metabolic switching cost (v3: cost-injection)
# =============================================================================
#
# The invariant: actions are temporal commitments, not instantaneous choices.
# Starting an action invests metabolic energy (orientation, sensor config,
# heading change). Switching before the action delivers its predicted
# information gain wastes that investment.
#
# Mechanism: when an action is selected, its predicted info-gain becomes a
# "commitment deposit." The deposit decays as the action's predicted effects
# materialize (precision changes actually occur). Switching to a different
# action incurs a penalty proportional to the unrealized deposit.
#
# v3 integration: the penalty is injected into the planner's cost dict
# before the fractal search runs. This means _sim_efe, _residual_efe,
# and _fractal_expand are COMPLETELY UNTOUCHED. The penalty flows through
# the existing cost term: self._costs.get(action, 0.05).

class ActionMomentum:
    """Track unrealized investment in the current action."""

    def __init__(self):
        self.current_action = None
        self.deposit = 0.0
        self.realized = 0.0
        self.completion = 0.0
        self._start_precisions = {}
        self._predicted_gains = {}
        self._ticks_held = 0

    def update(self, action, beliefs, models, ew):
        """Called each tick with the action the planner selected last tick.

        If the action changed, compute a new commitment deposit from
        predicted info-gain. If the same action continues, measure how
        much of the predicted precision gain has actually materialized.
        """
        if action != self.current_action:
            # New action: compute deposit
            self.current_action = action
            self._ticks_held = 0
            self._start_precisions = {}
            self._predicted_gains = {}
            self.deposit = 0.0

            for b in beliefs:
                self._start_precisions[b.obs] = b.prec
                vd, pd, conf = models.get((action, b.obs), (0.0, 0.0, 0.0))
                if pd > 0:
                    uncertainty = 1.0 - b.prec
                    effective_gain = pd * uncertainty * conf
                    self._predicted_gains[b.obs] = effective_gain
                    self.deposit += effective_gain

            self.deposit *= ew
            self.realized = 0.0
            self.completion = 0.0 if self.deposit > 0.001 else 1.0
        else:
            # Same action: measure realized precision changes
            self._ticks_held += 1
            self.realized = 0.0
            for obs, start_prec in self._start_precisions.items():
                if obs in self._predicted_gains:
                    for b in beliefs:
                        if b.obs == obs:
                            delta = max(0.0, b.prec - start_prec)
                            self.realized += delta
                            break

            if self.deposit > 0.001:
                self.completion = min(1.0, self.realized / self.deposit)
            else:
                self.completion = 1.0

    def get_switching_penalty(self):
        """The current penalty for switching away from the current action.
        Returns 0.0 if no active commitment. Capped at 0.15."""
        if self.current_action is None or self.deposit < 0.001:
            return 0.0
        raw = self.deposit * (1.0 - self.completion)
        return min(raw, 0.15)

    def diagnostics(self):
        return {
            'action': self.current_action or '',
            'deposit': round(self.deposit, 5),
            'realized': round(self.realized, 5),
            'completion': round(self.completion, 3),
            'ticks_held': self._ticks_held,
        }

    def reset(self):
        self.current_action = None
        self.deposit = 0.0
        self.realized = 0.0
        self.completion = 0.0
        self._start_precisions = {}
        self._predicted_gains = {}
        self._ticks_held = 0


# =============================================================================
# PLANNER
# =============================================================================

class Planner:
    """Fractal beam-search planner. Reads state from the runtime's &state.

    Mirrors planning.metta §I-VIII. Same formulas, same thresholds,
    same noise-floor pruning. Runs ~1000x faster than MeTTa interpretation
    because it's native Python arithmetic instead of interpreted dispatch.
    """

    # Planning configs (mirror planning.metta §I)
    NOISE_FLOOR_BASE = 0.05
    CONFIDENCE_FLOOR = 0.15
    MAX_BEAM_WIDTH = 8
    MAX_PLANNING_DEPTH = 5
    DISCOUNT = 0.9
    TIME_BUDGET_MS = 60  # Wall-clock budget for planning phase

    def __init__(self, rt):
        self.rt = rt
        # Cached per-plan data (rebuilt at plan start)
        self._actions = []
        self._models = {}       # (action, obs) → (vd, pd, conf)
        self._costs = {}        # action → cost
        self._preferences = {}  # obs → (value, importance)
        self._viability = {}    # obs → (min, max)
        self._observables = []
        # Diagnostics
        self.pruning_records = []
        self.nodes_evaluated = 0
        self.last_trace = []
        self.last_depth = 0
        self._current_zone = None
        self._deadline = 0      # Wall-clock deadline (monotonic)
        self._timed_out = False

        # Policy persistence — commit to multi-step plans
        # until the EFE landscape shifts significantly.
        self._policy = []           # Current committed policy (action trace)
        self._policy_step = 0       # Next step to execute
        self._policy_continued = False  # Was last result a policy continuation?
        self.REPLAN_MARGIN = 0.15   # Re-plan if best action beats policy
                                    # step by more than this margin

        # Minimum commitment: survival actions (return-to-hq) get a forced
        # continuation window proportional to viability pressure. Without this,
        # per-tick EFE noise causes the drone to abandon homeward journeys
        # after 1-2 cycles — never enough to actually reach HQ.
        self._policy_min_steps = 0  # Forced continuation (0 = normal persistence)

        # Action momentum: metabolic switching cost from unrealized investment.
        # Injected into self._costs before each plan() call — no changes to
        # _sim_efe, _residual_efe, or _fractal_expand.
        self.momentum = ActionMomentum()


    # -----------------------------------------------------------------
    # STATE LOADING — read from &state at plan start
    # -----------------------------------------------------------------

    def _load_state(self):
        """Read all planning-relevant state from the runtime."""
        # Actions
        ba = self.rt.run('!(base-actions)')
        if isinstance(ba, tuple):
            self._actions = [str(a) for a in ba]
        else:
            self._actions = ['wait', 'observe', 'retreat']

        # Action models: (action-model action obs vd pd conf)
        self._models = {}
        for b in self.rt.state.query(_PAT_ACTION_MODEL):
            key = (str(b['$a']), str(b['$o']))
            self._models[key] = (
                _sf(b['$vd']), _sf(b['$pd']), _sf(b['$c'], 0.3))

        # Action costs: (action-cost action cost conf)
        self._costs = {}
        for b in self.rt.state.query(_PAT_ACTION_COST):
            self._costs[str(b['$a'])] = _sf(b['$c'], 0.05)

        # Preferences: (preference obs value importance)
        self._preferences = {}
        for b in self.rt.state.query(_PAT_PREFERENCE):
            self._preferences[str(b['$o'])] = (_sf(b['$v']), _sf(b['$i']))

        # Viability bounds: (viability-bound obs min max)
        self._viability = {}
        for b in self.rt.state.query(_PAT_VIABILITY):
            self._viability[str(b['$o'])] = (_sf(b['$lo']), _sf(b['$hi']))

        # Build sim-state from current beliefs
        beliefs = []
        self._observables = []
        for b in self.rt.state.query(_PAT_BELIEF):
            obs = str(b['$o'])
            beliefs.append(SimBelief(obs, _sf(b['$v']), _sf(b['$p'])))
            self._observables.append(obs)
        return SimState(beliefs)

    # -----------------------------------------------------------------
    # SIMULATED EFE — canonical formula from test_efe.py
    # -----------------------------------------------------------------

    def _get_model(self, action, obs):
        """Return (value_delta, prec_delta, confidence).

        Zone awareness: if the action is navigate-to-{current_zone} and
        the drone is already in that zone, the precision gain is zeroed.
        You can't gain information by arriving where you already are.
        Battery drain and hull risk still apply (the drone is still moving).
        """
        vd, pd, conf = self._models.get((action, obs), (0.0, 0.0, 0.0))
        if (self._current_zone
                and action == 'navigate-to-{}'.format(self._current_zone)
                and pd > 0):
            pd = 0.0
        return vd, pd, conf

    def _sim_efe(self, action, state):
        """EFE for action given simulated beliefs.

        Aligned with MeTTa specification (core_actions.metta):
          EFE = predicted_error + cost - ew * info_gain + viability_effect

        Key spec features ported:
          - epistemic-weight suppressed by viability pressure (§VIII)
          - confidence case-split pragmatic error with precision weighting (§VI)
          - differential viability effect (§IX): measures pressure CHANGE

        only applies at depth 0 (the immediate next action). Deeper
        planning steps are simulated futures where the "current action"
        """
        total = 0.0

        # --- Epistemic weight: suppress curiosity under viability pressure ---
        # Spec: (= (epistemic-weight) (max 0.05 (- 1.0 (get-viability-pressure))))
        viability_pressure = 0.0
        for b in state.beliefs:
            if b.obs in self._viability:
                viability_pressure = max(viability_pressure,
                                         self._boundary_pressure(
                                             b.val, b.prec,
                                             *self._viability[b.obs]))
        ew = max(0.05, 1.0 - viability_pressure)

        for b in state.beliefs:
            vd, pd, conf = self._get_model(action, b.obs)
            pref_val, importance = self._preferences.get(
                b.obs, (b.val, 0.0))

            # Predicted value (confidence-weighted delta)
            pred_val = max(0.0, min(1.0, b.val + vd * conf))

            # --- Pragmatic error (spec-aligned confidence case-split) ---
            # Spec §VI: conf × |predicted - preferred| × new_prec
            #         + (1-conf) × |current - preferred| × old_prec
            # Precision weighting: certainty about deviation amplifies the signal.
            # Importance scales the entire term (game-layer weighting).
            new_prec = max(0.05, min(0.95, b.prec + pd * conf))
            error_right = abs(pred_val - pref_val) * new_prec
            error_wrong = abs(b.val - pref_val) * b.prec
            pragmatic = conf * error_right + (1.0 - conf) * error_wrong
            total += pragmatic * importance

            # --- Epistemic value (spec-aligned, viability-modulated) ---
            # obs_gain: precision gain from this action on this observable.
            # Diminishing returns at high precision (only for positive pd).
            uncertainty = 1.0 - b.prec
            obs_gain = pd * uncertainty if pd > 0 else pd
            total -= ew * obs_gain

            # --- Viability effect (spec-aligned differential) ---
            # Spec §IX: scale × delta_pressure + uncertainty_risk
            # Measures the CHANGE in boundary pressure the action causes.
            # Actions that relieve pressure score negative (good).
            # Actions that worsen pressure score positive (bad).
            if b.obs in self._viability:
                lo, hi = self._viability[b.obs]
                cur_pressure = self._boundary_pressure(
                    b.val, b.prec, lo, hi)
                new_pressure = self._boundary_pressure(
                    pred_val, new_prec, lo, hi)

                delta = new_pressure - cur_pressure
                scale = max(cur_pressure, new_pressure)
                uncertainty_risk = scale * (1.0 - conf) * 0.1
                total += scale * delta + uncertainty_risk

        # --- Model info gain (per-action, not per-observable) ---
        # How uncertain are this action's models? Lower average confidence
        # → more learning potential. Computed as a SINGLE term to avoid
        # volume bias (an action with 10 models shouldn't get 10× the
        # curiosity bonus of an action with 2 models).
        action_confs = [self._models[k][2] for k in self._models
                        if k[0] == action]
        if action_confs:
            avg_conf = sum(action_confs) / len(action_confs)
            total -= ew * 0.05 * (1.0 - avg_conf)

        total += self._costs.get(action, 0.05)

        return total

    @staticmethod
    def _boundary_pressure(val, prec, lo, hi):
        """Viability boundary pressure. Spec: compute-boundary-pressure.

        Returns 0.0 when safe, up to prec (≤1.0) when at the lower boundary.
        Precision amplifies: certainty about proximity to death is urgent.

        Only checks the LOWER bound. The upper bound (1.0 for hull, battery,
        sensor) is a physical ceiling, not a danger threshold. Hull=1.0
        (perfect health) must produce pressure=0.0, not 0.95.
        """
        rng = max(hi - lo, 0.001)
        margin = (val - lo) / rng
        danger = 0.35  # viability-danger-zone config
        if margin < 0.0:
            return 1.0
        if margin < danger:
            return prec * (1.0 - margin / danger)
        return 0.0

    def _apply_action(self, action, state):
        """Simulate applying action to state. Returns new SimState."""
        new = state.copy()
        for b in new.beliefs:
            vd, pd, conf = self._get_model(action, b.obs)
            b.val = max(0.0, min(1.0, b.val + vd * conf))
            b.prec = max(0.05, min(0.95, b.prec + pd * conf))
        return new

    def _violates_viability(self, state):
        """Check if any belief value is outside viability bounds."""
        for b in state.beliefs:
            if b.obs in self._viability:
                lo, hi = self._viability[b.obs]
                if b.val < lo or b.val > hi:
                    return True
        return False

    # -----------------------------------------------------------------
    # NOISE FLOOR — mirrors planning.metta §III
    # -----------------------------------------------------------------

    def _avg_model_confidence(self, state):
        """Average best model confidence across observables."""
        confs = []
        for b in state.beliefs:
            best = max((self._get_model(a, b.obs)[2]
                        for a in self._actions), default=0.1)
            confs.append(best)
        return sum(confs) / len(confs) if confs else 0.3

    def _degraded_confidence(self, state, depth):
        base = self._avg_model_confidence(state)
        return base * (self.DISCOUNT ** depth)

    def _noise_floor(self, state, depth):
        deg = self._degraded_confidence(state, depth)
        return self.NOISE_FLOOR_BASE / max(deg, 0.01)

    def _beyond_confidence_floor(self, state, depth):
        return self._degraded_confidence(state, depth) < self.CONFIDENCE_FLOOR

    # -----------------------------------------------------------------
    # RESIDUAL EFE — mirrors planning.metta §IV
    # -----------------------------------------------------------------

    def _future_heuristic(self, state, remaining):
        if remaining <= 0:
            return 0.0
        avg_prec = state.avg_precision()
        # Average preference gap
        gaps = []
        for b in state.beliefs:
            pv, _ = self._preferences.get(b.obs, (0.5, 0.5))
            gaps.append(abs(pv - b.val))
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.1
        headroom = 1.0 - avg_prec
        est = avg_gap * headroom
        effective = min(remaining, 3)
        return -(est * effective)

    def _residual_efe(self, action, state, remaining):
        immediate = self._sim_efe(action, state)
        next_state = self._apply_action(action, state)
        future = self._future_heuristic(next_state, remaining)
        return immediate + self.DISCOUNT * future

    # -----------------------------------------------------------------
    # CORE PLANNER — mirrors planning.metta §VII
    # -----------------------------------------------------------------

    def _fractal_expand(self, state, max_depth, depth):
        self.nodes_evaluated += 1

        # Stopping: time budget exhausted — return best at current depth
        if _time.monotonic() >= self._deadline:
            self._timed_out = True
            return ('terminal', 0.0, depth, [])

        # Stopping: depth ceiling
        if depth >= max_depth:
            return ('terminal', 0.0, depth, [])

        # Stopping: confidence floor
        if self._beyond_confidence_floor(state, depth):
            self.pruning_records.append(
                ('all', depth, 'confidence-floor',
                 self._degraded_confidence(state, depth)))
            return ('terminal', 0.0, depth, [])

        remaining = max_depth - depth

        # 1. Score all candidates via residual EFE
        candidates = []
        for a in self._actions:
            efe = self._residual_efe(a, state, remaining)
            candidates.append((a, efe))

        # 2. Viability filter
        viable = []
        for a, efe in candidates:
            next_s = self._apply_action(a, state)
            if self._violates_viability(next_s):
                self.pruning_records.append((a, depth, 'viability', efe))
            else:
                viable.append((a, efe))

        if not viable:
            return ('terminal', 999.0, depth, [])

        # 3. Sort ascending by EFE
        viable.sort(key=lambda x: x[1])

        # 4. Noise filter
        noise = self._noise_floor(state, depth)
        best_efe = viable[0][1]
        survivors = []
        for a, efe in viable:
            gap = efe - best_efe
            if gap <= noise and len(survivors) < self.MAX_BEAM_WIDTH:
                survivors.append((a, efe))
            else:
                self.pruning_records.append((a, depth, 'noise-floor', gap))

        if not survivors:
            return ('terminal', 999.0, depth, [])

        # 5. Recurse on survivors
        branches = []
        for a, _ in survivors:
            next_state = self._apply_action(a, state)
            child = self._fractal_expand(next_state, max_depth, depth + 1)
            step_efe = self._sim_efe(a, state)
            total_efe = step_efe + self.DISCOUNT * child[1]  # child efe
            trace = [a] + child[3]  # child trace
            branches.append((a, total_efe, child[2], trace))

        # 6. Return best
        best = min(branches, key=lambda b: b[1])
        return best

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------

    def plan(self, max_depth=None, current_zone=None):
        """Run the fractal planner. Returns (action, target, metadata).

        Policy persistence: if a multi-step policy was committed on a
        previous tick, the planner first checks whether the next step
        is still reasonable (its EFE is within REPLAN_MARGIN of the
        best available single-step action). If so, the policy step is
        returned without full re-planning. If the landscape shifted
        significantly, the policy is abandoned and a full re-plan runs.

        This is not hysteresis — it's the Active Inference commitment
        to policies. The agent evaluates policy continuation against
        the current state and only switches when there's genuine
        evidence that the policy is no longer optimal.
        """
        self.pruning_records = []
        self.nodes_evaluated = 0
        self._current_zone = current_zone
        self._timed_out = False
        self._policy_continued = False
        self._deadline = _time.monotonic() + self.TIME_BUDGET_MS / 1000.0

        state = self._load_state()
        depth = max_depth if max_depth is not None else self.MAX_PLANNING_DEPTH

        # --- Action momentum: inject switching penalty into costs ---
        # Determine what action was executed last tick from the policy.
        # Update the momentum tracker, then add the switching penalty
        # to self._costs for all non-current actions. The penalty flows
        # through _sim_efe via the existing self._costs.get() call —
        # no changes to _sim_efe, _residual_efe, or _fractal_expand.
        _momentum_last = None
        if self._policy and self._policy_step > 0:
            idx = self._policy_step - 1
            if idx < len(self._policy):
                _momentum_last = self._policy[idx]
        if _momentum_last:
            # Compute epistemic weight for deposit scaling
            _vp = 0.0
            for b in state.beliefs:
                if b.obs in self._viability:
                    _vp = max(_vp, self._boundary_pressure(
                        b.val, b.prec, *self._viability[b.obs]))
            _ew = max(0.05, 1.0 - _vp)
            self.momentum.update(_momentum_last, state.beliefs,
                                 self._models, _ew)

        _penalty = self.momentum.get_switching_penalty()
        if _penalty > 0.001:
            _cur = self.momentum.current_action
            for a in self._actions:
                if a != _cur:
                    self._costs[a] = self._costs.get(a, 0.05) + _penalty

        # --- Policy continuation check ---
        if (self._policy
                and self._policy_step < len(self._policy)):
            continue_action = self._policy[self._policy_step]

            # Verify the action still exists (zone might have been lost)
            if continue_action in self._actions:

                # --- Minimum commitment window (Fix 3.1) ---
                # Survival actions get forced continuation. Without this,
                # per-tick EFE noise causes the drone to abandon homeward
                # journeys after 1-2 cycles on flat EFE landscapes.
                if self._policy_step < self._policy_min_steps:
                    self._policy_step += 1
                    self._policy_continued = True
                    self.nodes_evaluated = 0

                    action = continue_action
                    efe_scores = {}
                    for a in self._actions:
                        efe_scores[a] = round(self._sim_efe(a, state), 6)

                    target = None
                    if action.startswith('navigate-to-'):
                        target = action[len('navigate-to-'):]
                        action = 'navigate-to'

                    return action, target, {
                        'efe_scores': efe_scores,
                        'planned_efe': round(
                            self._sim_efe(continue_action, state), 6),
                        'depth': len(self._policy) - self._policy_step,
                        'trace': self._policy[self._policy_step - 1:],
                        'nodes_evaluated': self.nodes_evaluated,
                        'pruning_records': 0,
                        'timed_out': False,
                        'policy_continued': True,
                        'replan_margin': -1.0,  # forced continuation
                        'momentum': self.momentum.diagnostics(),
                    }

                # --- Normal adaptive margin check ---
                continue_efe = self._sim_efe(continue_action, state)

                # Quick scan: score all actions to find best and spread
                all_efes = []
                best_efe = float('inf')
                for a in self._actions:
                    e = self._sim_efe(a, state)
                    all_efes.append(e)
                    if e < best_efe:
                        best_efe = e

                # Compute EFE spread (range of all action scores)
                spread = max(all_efes) - min(all_efes) if all_efes else 1.0

                # Adaptive margin: continue the policy unless a challenger
                # beats it by more than half the total EFE spread.
                # When actions are tightly clustered (spread=0.1), margin=0.05
                # When spread is wide (spread=2.0), margin=1.0
                # Floor of 0.02 prevents continuation when challenger is
                # genuinely better by any measurable amount.
                adaptive_margin = max(0.02, spread * 0.25)
                margin = continue_efe - best_efe

                if margin <= adaptive_margin:
                    # Policy is still good — continue without full search
                    self._policy_step += 1
                    self._policy_continued = True
                    self.nodes_evaluated = len(self._actions)  # single scan

                    action = continue_action
                    # Compute all EFEs for dashboard
                    efe_scores = {}
                    for a in self._actions:
                        efe_scores[a] = round(self._sim_efe(a, state), 6)

                    # Parse navigate-to targets
                    target = None
                    if action.startswith('navigate-to-'):
                        target = action[len('navigate-to-'):]
                        action = 'navigate-to'

                    return action, target, {
                        'efe_scores': efe_scores,
                        'planned_efe': round(continue_efe, 6),
                        'depth': len(self._policy) - self._policy_step,
                        'trace': self._policy[self._policy_step - 1:],
                        'nodes_evaluated': self.nodes_evaluated,
                        'pruning_records': 0,
                        'timed_out': False,
                        'policy_continued': True,
                        'replan_margin': round(margin, 6),
                        'momentum': self.momentum.diagnostics(),
                    }
                # else: margin exceeded threshold — fall through to re-plan

        # --- Full re-plan ---
        self._policy = []
        self._policy_step = 0

        # --- Viability depth clamp ---
        # Under survival pressure, clamp planning depth to 1 (myopic).
        # The fractal search undermines viability by looking past the crisis:
        # at depth 1, return-to-hq simulates hull recovery, which reduces
        # viability pressure, which increases epistemic weight, which makes
        # exploration look attractive at depth 2+. The multi-step total EFE
        # for return-to-hq ends up comparable to wait.
        #
        # This is architecturally correct: viability is a STRATUM, not a term.
        # Under existential threat, the agent should act on the immediate
        # signal. Multi-step planning is a luxury for when you're safe.
        viability_pressure = 0.0
        for b in state.beliefs:
            if b.obs in self._viability:
                viability_pressure = max(viability_pressure,
                                             self._boundary_pressure(
                                                 b.val, b.prec,
                                                 *self._viability[b.obs]))
        if viability_pressure > 0.5:
            depth = 1

        result = self._fractal_expand(state, depth, 0)

        action = result[0]
        efe = result[1]
        self.last_depth = result[2]
        self.last_trace = result[3]

        # Commit to the new policy
        if self.last_trace and len(self.last_trace) > 1:
            self._policy = self.last_trace
            self._policy_step = 1  # step 0 is the action we're returning now

            # Minimum commitment for survival actions (Fix 3.1).
            # return-to-hq needs ~20 consecutive cycles to reach HQ.
            # Scale commitment with viability pressure: higher pressure
            # → longer forced window → stronger guarantee of follow-through.
            if self.last_trace[0] == 'return-to-hq':
                self._policy_min_steps = max(5, int(15 * viability_pressure))
            else:
                self._policy_min_steps = 0

        elif action == 'return-to-hq' and viability_pressure > 0.5:
            # Depth was clamped to 1, so trace is only [return-to-hq].
            # Synthesize a committed policy: repeat return-to-hq for enough
            # cycles to reach HQ. The agent can't afford to reconsider
            # every cycle — that's what caused the previous failure.
            commit_cycles = max(5, int(15 * viability_pressure))
            self._policy = ['return-to-hq'] * commit_cycles
            self._policy_step = 1
            self._policy_min_steps = commit_cycles

        # Parse navigate-to targets
        target = None
        if action.startswith('navigate-to-'):
            target = action[len('navigate-to-'):]
            action = 'navigate-to'

        # Compute all action EFEs at depth 0 for the dashboard
        efe_scores = {}
        for a in self._actions:
            efe_scores[a] = round(self._sim_efe(a, state), 6)

        return action, target, {
            'efe_scores': efe_scores,
            'planned_efe': round(efe, 6),
            'depth': self.last_depth,
            'trace': self.last_trace,
            'nodes_evaluated': self.nodes_evaluated,
            'pruning_records': len(self.pruning_records),
            'timed_out': self._timed_out,
            'policy_continued': False,
            'replan_margin': None,
            'momentum': self.momentum.diagnostics(),
        }

    def configure(self, **kwargs):
        """Override planning parameters."""
        if 'noise_floor_base' in kwargs:
            self.NOISE_FLOOR_BASE = kwargs['noise_floor_base']
        if 'confidence_floor' in kwargs:
            self.CONFIDENCE_FLOOR = kwargs['confidence_floor']
        if 'max_beam_width' in kwargs:
            self.MAX_BEAM_WIDTH = kwargs['max_beam_width']
        if 'max_planning_depth' in kwargs:
            self.MAX_PLANNING_DEPTH = kwargs['max_planning_depth']
        if 'discount' in kwargs:
            self.DISCOUNT = kwargs['discount']
        if 'time_budget_ms' in kwargs:
            self.TIME_BUDGET_MS = kwargs['time_budget_ms']
        if 'replan_margin' in kwargs:
            self.REPLAN_MARGIN = kwargs['replan_margin']

    def reset_policy(self):
        """Clear the current committed policy. Called on init or body loss."""
        self._policy = []
        self._policy_step = 0
        self._policy_min_steps = 0
        self.momentum.reset()
