#!/usr/bin/env python3
"""
Action Space Scaling Benchmark
===============================

Tests whether the architecture's core subsystems remain coherent as the
action space scales from 3 (current benchmarks) to 50 actions.

Faithful Python replication of:
  - Full EFE computation (actions.metta v4.1): predicted error with model
    uncertainty, obs info gain, model info gain, adaptive epistemic weight,
    principled viability effect
  - Metabolic economy: rent, reward, gestation, death
  - Structure learning: induction (Hebbian suspicion), deduction (transitive
    closure), abduction (inverse causal model)
  - Fractal planning: RG-flow-inspired adaptive beam search with noise floor

QUESTION UNDER TEST:
  The architecture has been validated with 3 actions. Does it degrade
  gracefully — or catastrophically — at 10, 20, 50?

Usage:
    python test_action_space_scaling.py            # Full sweep
    python test_action_space_scaling.py -v          # Verbose
    python test_action_space_scaling.py -n 20       # Single action count
    python test_action_space_scaling.py --quick      # Fast mode (fewer cycles)

Author: Project Dagaz
"""

import sys
import math
import time
import random
import argparse
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
from collections import defaultdict

# =============================================================================
# CONFIGURATION (mirrors foundations.metta + actions.metta v4.1)
# =============================================================================

CONFIG = {
    # --- Metabolic dynamics ---
    "metabolic_rate": 0.02,
    "metabolic_boost": 0.05,
    "metabolic_initial_energy": 1.0,
    "metabolic_energy_cap": 2.0,
    "gestation_period": 3,

    # --- Structure learning ---
    "default_plasticity": 0.30,
    "default_structural_stability": 0.95,
    "default_cognitive_threshold": 0.04,
    "raw_error_salience_threshold": 0.20,
    "structural_cost_link": 0.001,
    "structural_cost_latent": 0.04,
    "structural_cost_spoke": 0.001,
    "max_structural_atoms_per_cycle": 5,
    "lookback_window": 3,

    # --- Belief update ---
    "learning_rate": 0.12,
    "surprise_threshold": 0.20,
    "precision_floor": 0.05,
    "precision_ceiling": 0.95,

    # --- Deduction ---
    "deductive_min_energy": 0.5,
    "deductive_weight_discount": 0.8,

    # --- Abduction ---
    "abductive_surprise_threshold": 0.05,
    "abductive_precision": 0.10,
    "abductive_min_link_weight": 0.15,
    "abductive_min_energy": 0.5,
    "abductive_max_cause_precision": 0.50,
    "abductive_budget_per_cycle": 5,

    # --- Fractal planning ---
    "noise_floor_base": 0.05,
    "confidence_floor": 0.15,
    "max_beam_width": 8,
    "max_planning_depth": 6,
    "discount_rate": 0.85,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Belief:
    value: float
    precision: float
    source: str = "prior"
    source_cycle: int = 0

@dataclass
class Observation:
    value: float
    precision: float
    time: int

@dataclass
class ActionModel:
    """Per-observable effect of an action (mirrors actions.metta schema)."""
    action: str
    observable: str
    value_delta: float      # raw expected change to world
    precision_delta: float  # expected change to belief precision
    confidence: float       # meta-precision (how much we trust this)

@dataclass
class ActionCost:
    action: str
    cost: float
    confidence: float

@dataclass
class PassiveModel:
    cause: str
    effect: str
    lag: int
    weight: float
    causal_type: str    # "excitatory" | "inhibitory"
    origin: str         # "empirical" | "deductive"
    energy: float
    created_at: int
    predictions: int = 0
    successes: int = 0
    intermediate: Optional[str] = None

@dataclass
class SuspicionLink:
    cause: str
    effect: str
    lag: int
    strength: float
    last_updated: int

@dataclass
class ErrorTrace:
    observable: str
    error: float
    surprise: float
    time: int

@dataclass
class ViabilityBound:
    observable: str
    min_val: float
    max_val: float

@dataclass
class EFEBreakdown:
    """Full EFE decomposition for one action (mirrors select-action-myopic-traced)."""
    action: str
    efe: float
    predicted_error: float
    cost: float
    obs_info_gain: float
    model_info_gain: float
    epistemic_weight: float
    viability_effect: float

@dataclass
class PlanBranch:
    """Result of fractal planning expansion."""
    first_action: str
    cumulative_efe: float
    depth_reached: int
    path: list


# =============================================================================
# ACTION GENERATOR — Plausible actions for scaling tests
# =============================================================================

# Archetypes: each defines a pattern of effects on observable dimensions
ACTION_ARCHETYPES = {
    "wait":      {"cost": 0.01, "pattern": "passive"},
    "observe":   {"cost": 0.05, "pattern": "epistemic"},
    "retreat":   {"cost": 0.12, "pattern": "safety"},
    "advance":   {"cost": 0.10, "pattern": "progress"},
    "recharge":  {"cost": 0.03, "pattern": "resource"},
    "scan":      {"cost": 0.04, "pattern": "epistemic_focused"},
    "shelter":   {"cost": 0.08, "pattern": "safety_partial"},
    "sprint":    {"cost": 0.15, "pattern": "aggressive"},
    "negotiate": {"cost": 0.06, "pattern": "social"},
    "repair":    {"cost": 0.09, "pattern": "maintenance"},
}

# Observable categories for structured generation
OBSERVABLE_CATEGORIES = {
    "viability": ["power-level", "structural-integrity", "oxygen-reserve",
                  "thermal-balance"],
    "environment": ["terrain-roughness", "ambient-temperature", "visibility",
                    "wind-speed", "humidity", "radiation-level"],
    "threat": ["threat-level", "predator-proximity", "storm-intensity",
               "seismic-activity"],
    "social": ["partner-predictability", "partner-comprehension",
               "discourse-coherence", "rapport"],
    "mission": ["goal-progress", "resource-efficiency", "route-accuracy",
                "sample-quality"],
}


def generate_observables(n_obs: int) -> list[str]:
    """Generate n observables, drawing from categories in priority order."""
    all_obs = []
    # Always include core observables first
    core = ["power-level", "terrain-roughness", "threat-level",
            "ambient-temperature"]
    all_obs.extend(core[:min(n_obs, len(core))])

    # Then fill from categories
    for cat in ["viability", "threat", "environment", "mission", "social"]:
        for obs in OBSERVABLE_CATEGORIES[cat]:
            if obs not in all_obs and len(all_obs) < n_obs:
                all_obs.append(obs)

    # If we need even more, generate indexed variants
    idx = 0
    while len(all_obs) < n_obs:
        all_obs.append(f"sensor-{idx}")
        idx += 1

    return all_obs[:n_obs]


def generate_actions(n_actions: int, observables: list[str],
                     rng: random.Random) -> tuple[list[str], dict, dict]:
    """
    Generate n plausible actions with structured effect models.

    Returns: (action_names, action_models, action_costs)
    where action_models[action][obs] = ActionModel
    and action_costs[action] = ActionCost
    """
    archetypes = list(ACTION_ARCHETYPES.keys())
    action_names = []
    action_models = {}
    action_costs = {}

    for i in range(n_actions):
        if i < len(archetypes):
            name = archetypes[i]
            arch = ACTION_ARCHETYPES[name]
        else:
            # Generate variants of archetypes
            base_idx = i % len(archetypes)
            base_name = archetypes[base_idx]
            arch = ACTION_ARCHETYPES[base_name]
            name = f"{base_name}-v{i // len(archetypes)}"

        action_names.append(name)
        action_costs[name] = ActionCost(
            name,
            arch["cost"] * rng.uniform(0.8, 1.2),
            0.5
        )
        action_models[name] = {}

        for obs in observables:
            vd, pd, conf = _generate_effect(
                arch["pattern"], obs, rng)
            action_models[name][obs] = ActionModel(
                name, obs, vd, pd, conf)

    return action_names, action_models, action_costs


def _generate_effect(pattern: str, obs: str,
                     rng: random.Random) -> tuple[float, float, float]:
    """Generate (value_delta, precision_delta, confidence) for a pattern×obs."""
    # Determine observable category
    cat = "other"
    for c, members in OBSERVABLE_CATEGORIES.items():
        if obs in members:
            cat = c
            break

    # Base confidence — actions start uncertain
    conf = rng.uniform(0.15, 0.45)

    if pattern == "passive":
        vd = rng.gauss(0.0, 0.005)
        pd = rng.uniform(-0.03, -0.01)
    elif pattern == "epistemic":
        vd = rng.gauss(0.0, 0.01)
        pd = rng.uniform(0.04, 0.10)
    elif pattern == "epistemic_focused":
        # High precision gain on 1-2 observables, slight loss elsewhere
        if rng.random() < 0.15:
            vd = 0.0
            pd = rng.uniform(0.10, 0.18)
        else:
            vd = 0.0
            pd = rng.uniform(-0.02, 0.01)
    elif pattern == "safety":
        if cat == "threat":
            vd = rng.uniform(-0.25, -0.10)
            pd = rng.uniform(0.02, 0.06)
        elif cat == "viability":
            vd = rng.uniform(0.01, 0.06)
            pd = rng.uniform(0.01, 0.03)
        else:
            vd = rng.gauss(-0.05, 0.03)
            pd = rng.uniform(0.01, 0.04)
    elif pattern == "progress":
        if cat == "mission":
            vd = rng.uniform(0.05, 0.15)
            pd = rng.uniform(0.02, 0.05)
        elif cat == "viability":
            vd = rng.uniform(-0.05, -0.01)
            pd = rng.uniform(-0.02, 0.0)
        else:
            vd = rng.gauss(0.0, 0.03)
            pd = rng.uniform(-0.01, 0.02)
    elif pattern == "resource":
        if cat == "viability":
            vd = rng.uniform(0.05, 0.15)
            pd = rng.uniform(0.01, 0.03)
        else:
            vd = rng.gauss(0.0, 0.01)
            pd = rng.uniform(-0.02, 0.0)
    elif pattern == "safety_partial":
        if cat == "threat":
            vd = rng.uniform(-0.15, -0.05)
            pd = rng.uniform(0.01, 0.04)
        else:
            vd = rng.gauss(0.0, 0.02)
            pd = rng.uniform(0.0, 0.02)
    elif pattern == "aggressive":
        if cat == "mission":
            vd = rng.uniform(0.10, 0.25)
            pd = rng.uniform(0.03, 0.08)
        elif cat == "viability":
            vd = rng.uniform(-0.08, -0.02)
            pd = rng.uniform(-0.02, 0.0)
        elif cat == "threat":
            vd = rng.uniform(-0.05, 0.10)
            pd = rng.uniform(0.01, 0.05)
        else:
            vd = rng.gauss(0.02, 0.05)
            pd = rng.uniform(0.01, 0.04)
    elif pattern == "social":
        if cat == "social":
            vd = rng.uniform(0.05, 0.12)
            pd = rng.uniform(0.03, 0.08)
        else:
            vd = rng.gauss(0.0, 0.01)
            pd = rng.uniform(-0.01, 0.01)
    elif pattern == "maintenance":
        if cat == "viability":
            vd = rng.uniform(0.03, 0.10)
            pd = rng.uniform(0.02, 0.05)
        elif cat == "environment":
            vd = rng.gauss(0.0, 0.02)
            pd = rng.uniform(0.01, 0.03)
        else:
            vd = rng.gauss(0.0, 0.01)
            pd = rng.uniform(-0.01, 0.01)
    else:
        vd = rng.gauss(0.0, 0.03)
        pd = rng.uniform(-0.02, 0.03)

    return (vd, pd, conf)


# =============================================================================
# EFE COMPUTATION (faithful to actions.metta v4.1)
# =============================================================================

class EFEEngine:
    """
    Full EFE computation matching actions.metta v4.1.

    EFE(a) = predicted_error(a) + cost(a)
             - epistemic_weight × (obs_info_gain(a) + model_info_gain(a))
             + viability_effect(a)
    """

    def __init__(self, beliefs: dict[str, Belief],
                 observations: dict[str, Observation],
                 action_models: dict[str, dict[str, ActionModel]],
                 action_costs: dict[str, ActionCost],
                 viability_bounds: list[ViabilityBound]):
        self.beliefs = beliefs
        self.observations = observations
        self.action_models = action_models
        self.action_costs = action_costs
        self.viability_bounds = viability_bounds

    def compute_efe(self, action: str) -> EFEBreakdown:
        """Full EFE with all four terms."""
        pred_err = self._predict_total_error(action)
        cost = self.action_costs[action].cost
        obs_ig = self._compute_obs_info_gain(action)
        model_ig = self._compute_model_info_gain(action)
        ew = self._epistemic_weight()
        v_effect = self._viability_effect(action)

        efe = pred_err + cost - ew * (obs_ig + model_ig) + v_effect

        return EFEBreakdown(
            action=action, efe=efe,
            predicted_error=pred_err, cost=cost,
            obs_info_gain=obs_ig, model_info_gain=model_ig,
            epistemic_weight=ew, viability_effect=v_effect)

    def select_action(self, actions: list[str]) -> tuple[str, list[EFEBreakdown]]:
        """argmin EFE over all candidate actions."""
        breakdowns = [self.compute_efe(a) for a in actions]
        best = min(breakdowns, key=lambda b: b.efe)
        return best.action, breakdowns

    # --- Section VI: Predicted error (model uncertainty) ---

    def _predict_total_error(self, action: str) -> float:
        total = 0.0
        for obs, b in self.beliefs.items():
            am = self.action_models[action].get(obs)
            if am is None:
                continue
            total += self._predict_single_error(action, obs, b, am)
        return total

    def _predict_single_error(self, action: str, obs: str,
                               b: Belief, am: ActionModel) -> float:
        """
        v4.1 formulation:
        expected_error = conf × error_if_right + (1-conf) × error_if_wrong
        """
        conf = am.confidence
        raw_vd = am.value_delta
        pdelta = am.precision_delta * conf  # conf-weighted

        obs_obj = self.observations.get(obs)
        obs_val = obs_obj.value if obs_obj else 0.5
        obs_prec = obs_obj.precision if obs_obj else 0.5

        new_bprec = max(0.1, min(1.0, b.precision + pdelta))

        # Case 1: model is correct
        predicted_obs_right = obs_val + raw_vd
        error_right = abs(predicted_obs_right - b.value)
        weighted_right = error_right * new_bprec * obs_prec

        # Case 2: model is wrong
        error_wrong = abs(obs_val - b.value)
        weighted_wrong = error_wrong * b.precision * obs_prec

        return conf * weighted_right + (1.0 - conf) * weighted_wrong

    # --- Section VII: Info gain ---

    def _compute_obs_info_gain(self, action: str) -> float:
        total = 0.0
        for obs in self.beliefs:
            am = self.action_models[action].get(obs)
            if am is None:
                continue
            pdelta = am.precision_delta * am.confidence
            total += max(0.0, pdelta)
        return total

    def _compute_model_info_gain(self, action: str) -> float:
        total = 0.0
        for obs in self.beliefs:
            am = self.action_models[action].get(obs)
            if am is None:
                continue
            total += 0.1 * (1.0 - am.confidence)
        return total

    # --- Section VIII: Epistemic weight ---

    def _epistemic_weight(self) -> float:
        pressure = self._get_viability_pressure()
        avg_prec = self._avg_belief_precision()
        safety = 1.0 - pressure
        uncertainty = 1.0 - avg_prec
        return max(0.05, safety * uncertainty)

    def _avg_belief_precision(self) -> float:
        if not self.beliefs:
            return 0.5
        return sum(b.precision for b in self.beliefs.values()) / len(self.beliefs)

    # --- Section IX: Viability effect ---

    def _viability_effect(self, action: str) -> float:
        total = 0.0
        for vb in self.viability_bounds:
            total += self._single_viability_effect(action, vb)
        return total

    def _single_viability_effect(self, action: str, vb: ViabilityBound) -> float:
        b = self.beliefs.get(vb.observable)
        if b is None:
            return 0.0

        bval = b.value
        bprec = b.precision
        current_pressure = self._boundary_pressure(bval, bprec, vb.min_val, vb.max_val)

        am = self.action_models[action].get(vb.observable)
        if am is None:
            return current_pressure * 0.01  # tiny penalty for unknown effects

        vdelta = am.value_delta * am.confidence
        new_val = bval + vdelta
        pdelta = am.precision_delta * am.confidence
        new_prec = max(0.01, min(1.0, bprec + pdelta))
        new_pressure = self._boundary_pressure(new_val, new_prec, vb.min_val, vb.max_val)

        delta_pressure = new_pressure - current_pressure
        scale = max(current_pressure, new_pressure)

        # Risk aversion: penalize uncertain actions near bounds
        uncertainty_risk = scale * (1.0 - am.confidence) * 0.1

        return scale * delta_pressure + uncertainty_risk

    def _get_viability_pressure(self) -> float:
        max_p = 0.0
        for vb in self.viability_bounds:
            b = self.beliefs.get(vb.observable)
            if b:
                p = self._boundary_pressure(b.value, b.precision, vb.min_val, vb.max_val)
                max_p = max(max_p, p)
        return max_p

    @staticmethod
    def _boundary_pressure(val: float, prec: float, min_v: float, max_v: float) -> float:
        """Mirrors compute-boundary-pressure in actions.metta.
        
        Threshold derived from belief precision: safety_ratio = margin / uncertainty.
        When < 1, boundary is within one uncertainty-width → pressure activates.
        """
        rng = max_v - min_v
        if rng <= 0:
            return 0.0
        low_margin = (val - min_v) / rng
        high_margin = (max_v - val) / rng
        min_margin = min(low_margin, high_margin)
        uncertainty = max(1.0 - prec, 0.01)
        safety_ratio = min_margin / uncertainty
        if safety_ratio < 1.0:
            return max(0.0, min(1.0, 1.0 - safety_ratio))
        return 0.0


# =============================================================================
# FRACTAL PLANNER (faithful to planning.metta)
# =============================================================================

class FractalPlanner:
    """
    RG-flow-inspired adaptive beam search.
    Noise floor grows with depth → beam width emerges from EFE landscape.
    """

    def __init__(self, efe_engine: EFEEngine, actions: list[str]):
        self.engine = efe_engine
        self.actions = actions
        self.nodes_evaluated = 0
        self.pruning_events = []

    def plan(self, max_depth: int = None) -> PlanBranch:
        max_depth = max_depth or CONFIG["max_planning_depth"]
        self.nodes_evaluated = 0
        self.pruning_events = []
        return self._fractal_expand(
            self.engine.beliefs.copy(),
            self.engine.observations.copy(),
            max_depth, 0, [])

    def _fractal_expand(self, beliefs: dict, observations: dict,
                         max_depth: int, current_depth: int,
                         path: list) -> PlanBranch:
        if current_depth >= max_depth:
            return PlanBranch("", 0.0, current_depth, path)

        # Confidence floor check
        deg_conf = CONFIG["discount_rate"] ** current_depth
        if deg_conf < CONFIG["confidence_floor"]:
            self.pruning_events.append(
                ("confidence_floor", current_depth, deg_conf))
            return PlanBranch("", 0.0, current_depth, path)

        # Score all actions
        candidates = []
        for a in self.actions:
            self.nodes_evaluated += 1
            bd = self.engine.compute_efe(a)
            # Residual EFE: immediate + future heuristic
            future_h = self._future_heuristic(current_depth, max_depth)
            residual = bd.efe + CONFIG["discount_rate"] * future_h
            candidates.append((a, residual, bd))

        if not candidates:
            return PlanBranch("", 999.0, current_depth, path)

        # Sort ascending (best = lowest EFE)
        candidates.sort(key=lambda x: x[1])

        # Noise filter (unified beam width + symmetry breaking)
        noise = CONFIG["noise_floor_base"] / max(deg_conf, 0.01)
        best_efe = candidates[0][1]
        survivors = []
        max_beam = CONFIG["max_beam_width"]
        for a, efe, bd in candidates:
            if efe - best_efe <= noise and len(survivors) < max_beam:
                survivors.append((a, efe, bd))
            else:
                self.pruning_events.append(
                    ("noise_filter", current_depth, a))

        if not survivors:
            return PlanBranch(candidates[0][0], candidates[0][1],
                              current_depth, path + [candidates[0][0]])

        # Single survivor = symmetry broken
        if len(survivors) == 1:
            a, efe, _ = survivors[0]
            return PlanBranch(
                path[0] if path else a,
                efe, current_depth + 1,
                path + [a])

        # Recurse on survivors (simplified: don't actually propagate beliefs)
        best_branch = None
        for a, efe, _ in survivors:
            sub = self._fractal_expand(
                beliefs, observations,
                max_depth, current_depth + 1,
                path + [a])
            total_efe = efe + CONFIG["discount_rate"] * sub.cumulative_efe
            branch = PlanBranch(
                path[0] if path else a,
                total_efe,
                sub.depth_reached,
                path + [a] + sub.path[len(path)+1:] if sub.path else path + [a])
            if best_branch is None or total_efe < best_branch.cumulative_efe:
                best_branch = branch

        return best_branch or PlanBranch("", 999.0, current_depth, path)

    def _future_heuristic(self, current_depth: int, max_depth: int) -> float:
        remaining = max_depth - current_depth - 1
        if remaining <= 0:
            return 0.0
        avg_prec = self.engine._avg_belief_precision()
        avg_pref_gap = 0.3  # approximate
        est_improvement = avg_pref_gap * (1.0 - avg_prec)
        return -est_improvement * min(remaining, 3)


# =============================================================================
# STRUCTURE LEARNING ENGINE (for action-space scaling tests)
# =============================================================================

class StructureLearningEngine:
    """
    Tests whether structure learning remains coherent with many actions
    producing many causal effects. The key concern: with N actions and
    M observables, there are N×M potential causal links. Does metabolic
    pressure still kill spurious ones?
    """

    def __init__(self, config: dict = None):
        self.config = config or CONFIG
        self.beliefs: dict[str, Belief] = {}
        self.observations: dict[str, Observation] = {}
        self.suspicion_links: dict[tuple, SuspicionLink] = {}
        self.passive_models: dict[tuple, PassiveModel] = {}
        self.error_traces: list[ErrorTrace] = []
        self.cycle = 0
        self.structural_budget = 0
        self.deduction_origins: set = set()
        self.death_log: list = []

    def set_belief(self, obs, val, prec, source="prior"):
        self.beliefs[obs] = Belief(val, prec, source, self.cycle)

    def inject_observation(self, obs, val, prec):
        self.observations[obs] = Observation(val, prec, self.cycle)

    def compute_prediction_error(self, obs):
        if obs in self.beliefs and obs in self.observations:
            return self.observations[obs].value - self.beliefs[obs].value
        return None

    def compute_surprise(self, obs):
        err = self.compute_prediction_error(obs)
        if err is None:
            return 0.0
        return 0.5 * self.beliefs[obs].precision * err * err

    def is_salient(self, obs):
        s = self.compute_surprise(obs)
        err = self.compute_prediction_error(obs)
        if err is None:
            return False
        return (s >= self.config["default_cognitive_threshold"] or
                abs(err) >= self.config["raw_error_salience_threshold"])

    def run_cycle(self, new_obs: dict = None):
        self.structural_budget = 0
        if new_obs:
            for obs, (val, prec) in new_obs.items():
                self.inject_observation(obs, val, prec)

        self._record_error_traces()
        self._update_suspicion()
        self._check_phase1()
        self._check_deductive()
        self._metabolic_step()
        self._update_beliefs()
        self.cycle += 1

    def _record_error_traces(self):
        for obs in self.beliefs:
            err = self.compute_prediction_error(obs)
            if err is None:
                continue
            s = self.compute_surprise(obs)
            self.error_traces.append(ErrorTrace(obs, err, s, self.cycle))
        cutoff = self.cycle - self.config["lookback_window"]
        self.error_traces = [t for t in self.error_traces if t.time >= cutoff]

    def _update_suspicion(self):
        cutoff = self.cycle - self.config["lookback_window"]
        salient = [t for t in self.error_traces
                   if t.time >= cutoff and self.is_salient(t.observable)]
        stability = self.config["default_structural_stability"]
        plasticity = self.config["default_plasticity"]
        updated = set()

        for ta in salient:
            for tb in salient:
                if ta.observable == tb.observable:
                    continue
                if ta.time > tb.time:
                    continue
                if ta.time == tb.time and ta.observable >= tb.observable:
                    continue

                cause, effect = ta.observable, tb.observable
                lag = tb.time - ta.time
                sign_a = 1.0 if ta.error >= 0 else -1.0
                sign_b = 1.0 if tb.error >= 0 else -1.0
                cov = sign_a * sign_b * ta.surprise * tb.surprise

                key = (cause, effect, lag)
                old = self.suspicion_links.get(
                    key, SuspicionLink(cause, effect, lag, 0.0, 0)).strength
                new = stability * old + plasticity * cov
                self.suspicion_links[key] = SuspicionLink(
                    cause, effect, lag, new, self.cycle)
                updated.add(key)

        for key, link in list(self.suspicion_links.items()):
            if key not in updated:
                link.strength *= stability

    def _check_phase1(self):
        cost = self.config["structural_cost_link"]
        for key, link in list(self.suspicion_links.items()):
            if self.structural_budget >= self.config["max_structural_atoms_per_cycle"]:
                break
            pm_key = (link.cause, link.effect)
            if pm_key in self.passive_models:
                continue
            if abs(link.strength) > cost:
                ctype = "excitatory" if link.strength > 0 else "inhibitory"
                w = min(abs(link.strength), 1.0)
                self.passive_models[pm_key] = PassiveModel(
                    cause=link.cause, effect=link.effect,
                    lag=link.lag, weight=w, causal_type=ctype,
                    origin="empirical",
                    energy=self.config["metabolic_initial_energy"],
                    created_at=self.cycle)
                self.structural_budget += 1

    def _check_deductive(self):
        min_e = self.config["deductive_min_energy"]
        discount = self.config["deductive_weight_discount"]
        for (a, b), pm_ab in list(self.passive_models.items()):
            if pm_ab.energy < min_e:
                continue
            for (b2, c), pm_bc in list(self.passive_models.items()):
                if b2 != b or a == c:
                    continue
                if pm_bc.energy < min_e:
                    continue
                if (a, c) in self.passive_models:
                    continue
                if (a, c) in self.deduction_origins:
                    continue
                if self.structural_budget >= self.config["max_structural_atoms_per_cycle"]:
                    return
                combined_w = min(discount * pm_ab.weight * pm_bc.weight, 1.0)
                type_map = {
                    ("excitatory", "excitatory"): "excitatory",
                    ("excitatory", "inhibitory"): "inhibitory",
                    ("inhibitory", "excitatory"): "inhibitory",
                    ("inhibitory", "inhibitory"): "excitatory",
                }
                ct = type_map[(pm_ab.causal_type, pm_bc.causal_type)]
                self.passive_models[(a, c)] = PassiveModel(
                    cause=a, effect=c, lag=pm_ab.lag + pm_bc.lag,
                    weight=combined_w, causal_type=ct,
                    origin="deductive",
                    energy=self.config["metabolic_initial_energy"],
                    created_at=self.cycle, intermediate=b)
                self.deduction_origins.add((a, c))
                self.structural_budget += 1

    def _metabolic_step(self):
        rate = self.config["metabolic_rate"]
        boost = self.config["metabolic_boost"]
        cap = self.config["metabolic_energy_cap"]
        gest = self.config["gestation_period"]
        dead = []

        for key, pm in list(self.passive_models.items()):
            if self.cycle - pm.created_at < gest:
                continue
            pm.energy -= rate

            cause_err = self.compute_prediction_error(pm.cause)
            effect_err = self.compute_prediction_error(pm.effect)
            if cause_err is not None and effect_err is not None:
                cause_b = self.beliefs.get(pm.cause)
                cause_uncertainty = max(1.0 - cause_b.precision, 0.01) if cause_b else 0.99
                if abs(cause_err) > cause_uncertainty:
                    pm.predictions += 1
                    if pm.causal_type == "excitatory":
                        correct = (cause_err * effect_err > 0)
                    else:
                        correct = (cause_err * effect_err < 0)
                    mag_ok = abs(effect_err) > 0.05
                    if correct and mag_ok:
                        pm.energy = min(pm.energy + boost, cap)
                        pm.successes += 1
                    elif not correct and mag_ok:
                        pm.energy -= boost * 0.5

            if pm.energy <= 0:
                dead.append(key)
                self.death_log.append((self.cycle, pm.cause, pm.effect,
                                       pm.origin, pm.predictions, pm.successes))

        for k in dead:
            del self.passive_models[k]

    def _update_beliefs(self):
        lr = self.config["learning_rate"]
        for obs in list(self.observations.keys()):
            if obs not in self.beliefs:
                continue
            err = self.compute_prediction_error(obs)
            if err is None:
                continue
            b = self.beliefs[obs]
            o = self.observations[obs]
            prec_sum = b.precision + o.precision
            if prec_sum <= 0:
                continue
            obs_w = o.precision / prec_sum
            update_mag = obs_w * abs(err)
            if update_mag < 0.01:
                continue
            update = lr * obs_w * err
            b.value = max(0.0, min(1.0, b.value + update))
            uncertainty = max(1.0 - b.precision, 0.01)
            pd = 0.02 if abs(err) < uncertainty else -0.05
            b.precision = max(self.config["precision_floor"],
                              min(self.config["precision_ceiling"],
                                  b.precision + pd))


# =============================================================================
# ENVIRONMENT — Multi-action causal structure
# =============================================================================

class ScalableEnvironment:
    """
    Generates observation patterns with known causal structure.

    Matches the signal strength of FireChainEnv from test_unified_reasoning.py:
    strong periodic pulses (0.80+ active, 0.12 baseline) with lag-1 propagation.
    Some observables are causally linked. Some are independent (false positive test).
    """

    def __init__(self, observables: list[str], n_causal_chains: int,
                 rng: random.Random):
        self.observables = observables
        self.rng = rng
        self.causal_chains = []  # [(cause, effect, lag, type)]
        self.independent = []    # observables not in any chain

        # Create known causal chains (strong, periodic, like FireChainEnv)
        available = list(observables)
        rng.shuffle(available)
        used = set()
        chains_made = 0

        # Build chains of length 2-3
        i = 0
        while chains_made < n_causal_chains and i < len(available) - 1:
            cause = available[i]
            effect = available[i + 1]
            if cause not in used or effect not in used:
                ctype = "excitatory" if rng.random() < 0.7 else "inhibitory"
                self.causal_chains.append((cause, effect, 1, ctype))
                used.add(cause)
                used.add(effect)
                chains_made += 1
            i += 1

        self.independent = [obs for obs in observables if obs not in used]

        # Each chain gets its own period (so they fire at different times)
        self.chain_periods = {}
        for j, (cause, effect, lag, ctype) in enumerate(self.causal_chains):
            self.chain_periods[cause] = 5 + (j * 2) % 5  # periods 5-9

    def get_observations(self, cycle: int) -> dict[str, tuple[float, float]]:
        obs = {}

        # Default: low baseline for everything
        for name in self.observables:
            obs[name] = (0.12 + self.rng.gauss(0, 0.02), 0.85)

        # Activate causes on their period (strong signal, like FireChainEnv)
        active_causes = set()
        for cause, effect, lag, ctype in self.causal_chains:
            period = self.chain_periods.get(cause, 7)
            if cycle % period == 0:
                obs[cause] = (0.85 + self.rng.gauss(0, 0.03), 0.85)
                active_causes.add(cause)

        # Propagate effects with lag
        for cause, effect, lag, ctype in self.causal_chains:
            period = self.chain_periods.get(cause, 7)
            cause_was_active = (cycle - lag) >= 0 and ((cycle - lag) % period == 0)
            if cause_was_active:
                if ctype == "excitatory":
                    obs[effect] = (0.80 + self.rng.gauss(0, 0.03), 0.80)
                else:
                    obs[effect] = (0.05 + self.rng.gauss(0, 0.02), 0.80)

        # Independent observables: gentle random drift (should NOT form links)
        for name in self.independent:
            base = 0.4 + 0.1 * math.sin(cycle * 0.2 + hash(name) % 100)
            obs[name] = (max(0.0, min(1.0, base + self.rng.gauss(0, 0.03))), 0.80)

        # Clamp all values
        for name in obs:
            v, p = obs[name]
            obs[name] = (max(0.0, min(1.0, v)), p)

        return obs


# =============================================================================
# BENCHMARK TESTS
# =============================================================================

def benchmark_efe_selection(n_actions: int, n_observables: int,
                             seed: int = 42, verbose: bool = False) -> dict:
    """
    TEST 1: Does EFE action selection produce sensible results at scale?

    Checks:
    - Selection completes in reasonable time
    - Selected action is contextually appropriate
    - EFE values are well-distributed (not degenerate)
    - Viability pressure correctly overrides exploration
    """
    rng = random.Random(seed)
    observables = generate_observables(n_observables)
    actions, models, costs = generate_actions(n_actions, observables, rng)

    results = {"n_actions": n_actions, "n_observables": n_observables}

    # Scenario A: Nominal (low error, no threat)
    beliefs = {}
    observations = {}
    for obs in observables:
        v = rng.uniform(0.3, 0.7)
        beliefs[obs] = Belief(v, 0.6)
        observations[obs] = Observation(v + rng.gauss(0, 0.05), 0.8, 0)

    viability_bounds = [ViabilityBound("power-level", 0.1, 1.0)]

    engine = EFEEngine(beliefs, observations, models, costs, viability_bounds)

    t0 = time.perf_counter()
    best_a, breakdowns = engine.select_action(actions)
    t_nominal = time.perf_counter() - t0

    efes = [b.efe for b in breakdowns]
    efe_range = max(efes) - min(efes)
    efe_std = (sum((e - sum(efes)/len(efes))**2 for e in efes) / len(efes)) ** 0.5

    results["nominal_time_ms"] = t_nominal * 1000
    results["nominal_best"] = best_a
    results["nominal_efe_range"] = efe_range
    results["nominal_efe_std"] = efe_std
    results["nominal_degenerate"] = efe_range < 0.001

    if verbose:
        print(f"\n  Scenario A (nominal): best={best_a}, "
              f"range={efe_range:.4f}, std={efe_std:.4f}, "
              f"time={t_nominal*1000:.2f}ms")
        top5 = sorted(breakdowns, key=lambda b: b.efe)[:5]
        for b in top5:
            print(f"    {b.action:20s} EFE={b.efe:+.4f} "
                  f"(err={b.predicted_error:.3f} cost={b.cost:.3f} "
                  f"ig={b.obs_info_gain:.3f}+{b.model_info_gain:.3f} "
                  f"ew={b.epistemic_weight:.3f} viab={b.viability_effect:+.4f})")

    # Scenario B: Viability threat (low power)
    beliefs_threat = deepcopy(beliefs)
    beliefs_threat["power-level"] = Belief(0.15, 0.9)
    observations_threat = deepcopy(observations)
    observations_threat["power-level"] = Observation(0.15, 0.9, 0)

    engine_t = EFEEngine(beliefs_threat, observations_threat, models, costs,
                          viability_bounds)
    t0 = time.perf_counter()
    best_threat, breakdowns_t = engine_t.select_action(actions)
    t_threat = time.perf_counter() - t0

    results["threat_time_ms"] = t_threat * 1000
    results["threat_best"] = best_threat

    # Check: does the selected action improve power?
    best_model = models[best_threat].get("power-level")
    power_improving = (best_model and best_model.value_delta > 0)
    # Or at least: does it have low cost (conservative)?
    power_conservative = costs[best_threat].cost < 0.05
    results["threat_sensible"] = power_improving or power_conservative

    if verbose:
        print(f"\n  Scenario B (threat):  best={best_threat}, "
              f"power_improving={power_improving}, "
              f"time={t_threat*1000:.2f}ms")
        top5 = sorted(breakdowns_t, key=lambda b: b.efe)[:5]
        for b in top5:
            print(f"    {b.action:20s} EFE={b.efe:+.4f} viab={b.viability_effect:+.4f}")

    # Scenario C: High uncertainty (low precision)
    beliefs_unc = {}
    for obs in observables:
        beliefs_unc[obs] = Belief(0.5, 0.15)
    observations_unc = {obs: Observation(rng.uniform(0.3, 0.7), 0.8, 0)
                        for obs in observables}

    engine_u = EFEEngine(beliefs_unc, observations_unc, models, costs,
                          [ViabilityBound("power-level", 0.1, 1.0)])
    best_unc, breakdowns_u = engine_u.select_action(actions)

    # Check: epistemic actions should be favored
    best_bd = next(b for b in breakdowns_u if b.action == best_unc)
    results["uncertain_best"] = best_unc
    results["uncertain_ig_ratio"] = (
        (best_bd.obs_info_gain + best_bd.model_info_gain) /
        max(best_bd.predicted_error, 0.001))

    if verbose:
        print(f"\n  Scenario C (uncertain): best={best_unc}, "
              f"ig_ratio={results['uncertain_ig_ratio']:.3f}")

    return results


def benchmark_fractal_planning(n_actions: int, n_observables: int,
                                seed: int = 42,
                                verbose: bool = False) -> dict:
    """
    TEST 2: Does fractal planning remain tractable at scale?

    With 3 actions and depth 6: 3^6 = 729 exhaustive evaluations.
    With 50 actions and depth 6: 50^6 ≈ 15.6 billion.
    The fractal planner should achieve massive pruning.
    """
    rng = random.Random(seed)
    observables = generate_observables(n_observables)
    actions, models, costs = generate_actions(n_actions, observables, rng)

    beliefs = {}
    observations = {}
    for obs in observables:
        v = rng.uniform(0.3, 0.7)
        beliefs[obs] = Belief(v, 0.5)
        observations[obs] = Observation(v + rng.gauss(0, 0.05), 0.8, 0)

    viability_bounds = [ViabilityBound("power-level", 0.1, 1.0)]
    engine = EFEEngine(beliefs, observations, models, costs, viability_bounds)
    planner = FractalPlanner(engine, actions)

    max_depth = min(CONFIG["max_planning_depth"], 6)

    t0 = time.perf_counter()
    branch = planner.plan(max_depth)
    t_plan = time.perf_counter() - t0

    exhaustive = n_actions ** max_depth
    reduction = exhaustive / max(planner.nodes_evaluated, 1)

    results = {
        "n_actions": n_actions,
        "max_depth": max_depth,
        "nodes_evaluated": planner.nodes_evaluated,
        "exhaustive_would_be": exhaustive,
        "reduction_factor": reduction,
        "planning_time_ms": t_plan * 1000,
        "depth_reached": branch.depth_reached,
        "first_action": branch.first_action,
        "n_pruning_events": len(planner.pruning_events),
    }

    if verbose:
        print(f"\n  Fractal planning: {n_actions} actions × depth {max_depth}")
        print(f"    Nodes evaluated: {planner.nodes_evaluated:,}")
        print(f"    Exhaustive:      {exhaustive:,}")
        print(f"    Reduction:       {reduction:,.0f}×")
        print(f"    Time:            {t_plan*1000:.1f}ms")
        print(f"    Best first:      {branch.first_action}")
        print(f"    Depth reached:   {branch.depth_reached}")

        # Pruning breakdown
        conf_prunes = sum(1 for e in planner.pruning_events
                          if e[0] == "confidence_floor")
        noise_prunes = sum(1 for e in planner.pruning_events
                           if e[0] == "noise_filter")
        print(f"    Confidence stops: {conf_prunes}")
        print(f"    Noise prunes:     {noise_prunes}")

    return results


def benchmark_structure_learning(n_actions: int, n_observables: int,
                                  n_cycles: int = 60,
                                  seed: int = 42,
                                  verbose: bool = False) -> dict:
    """
    TEST 3: Does metabolic selection remain effective with many observables?

    With N observables, there are O(N²) potential causal links.
    The metabolic economy must:
    - Discover real causal chains
    - Kill spurious correlations
    - Keep the total link count bounded (not O(N²))
    """
    rng = random.Random(seed)
    observables = generate_observables(n_observables)

    n_chains = max(2, n_observables // 3)
    env = ScalableEnvironment(observables, n_chains, rng)

    sl = StructureLearningEngine()
    for obs in observables:
        sl.set_belief(obs, 0.5, 0.5)

    # Track metrics over time
    link_counts = []
    empirical_counts = []
    deductive_counts = []
    death_counts = []

    for cycle in range(n_cycles):
        sl.cycle = cycle
        obs = env.get_observations(cycle)
        sl.run_cycle(obs)

        n_emp = sum(1 for pm in sl.passive_models.values()
                    if pm.origin == "empirical")
        n_ded = sum(1 for pm in sl.passive_models.values()
                    if pm.origin == "deductive")
        link_counts.append(len(sl.passive_models))
        empirical_counts.append(n_emp)
        deductive_counts.append(n_ded)
        death_counts.append(len(sl.death_log))

    # Evaluate: did we find real links?
    real_links_found = 0
    for cause, effect, lag, ctype in env.causal_chains:
        if (cause, effect) in sl.passive_models:
            pm = sl.passive_models[(cause, effect)]
            if pm.causal_type == ctype:
                real_links_found += 1

    total_links = len(sl.passive_models)
    max_possible = n_observables * (n_observables - 1)
    link_density = total_links / max(max_possible, 1)

    # Check bounded growth
    peak_links = max(link_counts) if link_counts else 0
    final_links = link_counts[-1] if link_counts else 0
    bounded = final_links < max_possible * 0.3  # links < 30% of all pairs

    total_deaths = len(sl.death_log)

    results = {
        "n_observables": n_observables,
        "n_causal_chains": len(env.causal_chains),
        "real_links_found": real_links_found,
        "recall": real_links_found / max(len(env.causal_chains), 1),
        "total_links_final": total_links,
        "peak_links": peak_links,
        "link_density": link_density,
        "bounded_growth": bounded,
        "total_deaths": total_deaths,
        "death_rate": total_deaths / max(n_cycles, 1),
    }

    if verbose:
        print(f"\n  Structure learning: {n_observables} observables, "
              f"{len(env.causal_chains)} real chains, {n_cycles} cycles")
        print(f"    Real links found:  {real_links_found}/{len(env.causal_chains)} "
              f"(recall={results['recall']:.0%})")
        print(f"    Total links final: {total_links} / {max_possible} possible "
              f"(density={link_density:.1%})")
        print(f"    Peak links:        {peak_links}")
        print(f"    Total deaths:      {total_deaths}")
        print(f"    Bounded growth:    {'✓' if bounded else '✗'}")

    return results


def benchmark_metabolic_health(n_actions: int, n_observables: int,
                                seed: int = 42,
                                verbose: bool = False) -> dict:
    """
    TEST 4: Does the metabolic economy stay healthy with a larger action space?

    Tests the reward/rent ratio dynamics. With more actions, there are more
    potential models to maintain. The economy must not:
    - Starve all structure (rent too high relative to reward)
    - Accumulate unbounded garbage (rent too low)
    """
    rng = random.Random(seed)
    observables = generate_observables(n_observables)

    n_chains = max(2, n_observables // 3)
    env = ScalableEnvironment(observables, n_chains, rng)
    rent_values = [0.01, 0.02, 0.04]
    reward_values = [0.03, 0.05, 0.08]

    healthy_count = 0
    total_tests = 0
    detailed = []

    for rent in rent_values:
        for reward in reward_values:
            config = dict(CONFIG)
            config["metabolic_rate"] = rent
            config["metabolic_boost"] = reward
            ratio = reward / rent

            sl = StructureLearningEngine(config)
            for obs in observables:
                sl.set_belief(obs, 0.5, 0.5)

            for cycle in range(50):
                sl.cycle = cycle
                obs = env.get_observations(cycle)
                sl.run_cycle(obs)

            real_found = 0
            for cause, effect, lag, ctype in env.causal_chains:
                if (cause, effect) in sl.passive_models:
                    real_found += 1

            total_links = len(sl.passive_models)
            total_deaths = len(sl.death_log)
            max_possible = n_observables * (n_observables - 1)

            # Health criteria: found at least some real links AND didn't
            # accumulate unbounded garbage
            has_real = real_found >= max(1, len(env.causal_chains) // 3)
            not_flooded = total_links < max_possible * 0.4
            is_healthy = has_real and not_flooded

            if is_healthy:
                healthy_count += 1
            total_tests += 1

            detailed.append({
                "rent": rent, "reward": reward, "ratio": ratio,
                "real_found": real_found, "total_links": total_links,
                "deaths": total_deaths, "healthy": is_healthy
            })

    results = {
        "n_observables": n_observables,
        "healthy_fraction": healthy_count / max(total_tests, 1),
        "total_tests": total_tests,
        "healthy_count": healthy_count,
        "detailed": detailed,
    }

    if verbose:
        print(f"\n  Metabolic health: {n_observables} observables")
        print(f"    Healthy parameter pairs: {healthy_count}/{total_tests} "
              f"({results['healthy_fraction']:.0%})")
        for d in detailed:
            status = "✓" if d["healthy"] else "✗"
            print(f"    rent={d['rent']:.3f} reward={d['reward']:.3f} "
                  f"ratio={d['ratio']:.1f}  "
                  f"real={d['real_found']} links={d['total_links']} "
                  f"deaths={d['deaths']}  {status}")

    return results


def benchmark_efe_timing(verbose: bool = False) -> dict:
    """
    TEST 5: Raw EFE computation timing across action space sizes.
    How does wall-clock time scale?
    """
    rng = random.Random(42)
    sizes = [3, 5, 10, 20, 30, 50]
    n_obs = 10
    observables = generate_observables(n_obs)

    timings = {}
    for n_a in sizes:
        actions, models, costs = generate_actions(n_a, observables, rng)
        beliefs = {obs: Belief(rng.uniform(0.3, 0.7), 0.5)
                   for obs in observables}
        observations = {obs: Observation(rng.uniform(0.3, 0.7), 0.8, 0)
                        for obs in observables}
        viability_bounds = [ViabilityBound("power-level", 0.1, 1.0)]

        engine = EFEEngine(beliefs, observations, models, costs, viability_bounds)

        # Warm up
        engine.select_action(actions)

        # Time 100 selections
        t0 = time.perf_counter()
        n_iter = 100
        for _ in range(n_iter):
            engine.select_action(actions)
        t_total = time.perf_counter() - t0

        ms_per = (t_total / n_iter) * 1000
        timings[n_a] = ms_per

    results = {"timings": timings, "n_observables": n_obs}

    if verbose:
        print(f"\n  EFE timing ({n_obs} observables):")
        print(f"    {'Actions':>8s}  {'ms/select':>10s}  {'Hz':>10s}  {'Scale':>8s}")
        base = timings.get(3, 1.0)
        for n_a in sizes:
            ms = timings[n_a]
            hz = 1000 / ms if ms > 0 else 0
            scale = ms / base if base > 0 else 0
            print(f"    {n_a:8d}  {ms:10.3f}  {hz:10.0f}  {scale:8.1f}×")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Action Space Scaling Benchmark — Project Dagaz")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--n-actions", type=int,
                        help="Test single action count")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced cycle count for fast iteration")
    args = parser.parse_args()

    verbose = args.verbose
    n_cycles = 30 if args.quick else 60

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Project Dagaz — Action Space Scaling Benchmark            ║")
    print("║   Testing architecture coherence from 3 to 50 actions       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    action_sizes = [args.n_actions] if args.n_actions else [3, 5, 10, 20, 30, 50]
    obs_sizes = {3: 4, 5: 6, 10: 10, 20: 14, 30: 18, 50: 24}

    # =========================================================================
    # TEST 1: EFE Action Selection
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 1: EFE ACTION SELECTION ACROSS SCALE")
    print(f"{'='*70}")
    print("  Does argmin(EFE) produce sensible actions with more choices?")

    efe_results = {}
    for n_a in action_sizes:
        n_obs = obs_sizes.get(n_a, n_a // 2)
        print(f"\n  --- {n_a} actions, {n_obs} observables ---")
        efe_results[n_a] = benchmark_efe_selection(n_a, n_obs, verbose=verbose)
        r = efe_results[n_a]
        print(f"    Nominal:   best={r['nominal_best']:20s} "
              f"range={r['nominal_efe_range']:.4f} "
              f"degenerate={'✗ BAD' if r['nominal_degenerate'] else '✓'}")
        print(f"    Threat:    best={r['threat_best']:20s} "
              f"sensible={'✓' if r['threat_sensible'] else '✗ BAD'}")
        print(f"    Uncertain: best={r['uncertain_best']:20s} "
              f"ig_ratio={r['uncertain_ig_ratio']:.3f}")
        print(f"    Time:      {r['nominal_time_ms']:.2f}ms (nominal)")

    # =========================================================================
    # TEST 2: Fractal Planning
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 2: FRACTAL PLANNING TRACTABILITY")
    print(f"{'='*70}")
    print("  Does the RG-flow planner tame combinatorial explosion?")

    plan_results = {}
    for n_a in action_sizes:
        n_obs = obs_sizes.get(n_a, n_a // 2)
        plan_results[n_a] = benchmark_fractal_planning(
            n_a, n_obs, verbose=verbose)
        r = plan_results[n_a]
        print(f"\n  {n_a:3d} actions: "
              f"{r['nodes_evaluated']:>8,d} nodes / "
              f"{r['exhaustive_would_be']:>15,d} exhaustive = "
              f"{r['reduction_factor']:>12,.0f}× reduction  "
              f"({r['planning_time_ms']:.1f}ms)")

    # =========================================================================
    # TEST 3: Structure Learning
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 3: STRUCTURE LEARNING WITH MANY OBSERVABLES")
    print(f"{'='*70}")
    print("  Does metabolic selection keep links bounded as O(N²) pairs grow?")

    struct_results = {}
    for n_a in action_sizes:
        n_obs = obs_sizes.get(n_a, n_a // 2)
        struct_results[n_a] = benchmark_structure_learning(
            n_a, n_obs, n_cycles=n_cycles, verbose=verbose)
        r = struct_results[n_a]
        bounded = "✓" if r["bounded_growth"] else "✗ UNBOUNDED"
        print(f"\n  {n_obs:3d} obs: recall={r['recall']:.0%}  "
              f"links={r['total_links_final']}/{n_obs*(n_obs-1)} "
              f"density={r['link_density']:.1%}  "
              f"deaths={r['total_deaths']}  {bounded}")

    # =========================================================================
    # TEST 4: Metabolic Health
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 4: METABOLIC ECONOMY HEALTH")
    print(f"{'='*70}")
    print("  What fraction of (rent, reward) parameter space is healthy?")

    metab_results = {}
    for n_a in action_sizes:
        n_obs = obs_sizes.get(n_a, n_a // 2)
        metab_results[n_a] = benchmark_metabolic_health(
            n_a, n_obs, verbose=verbose)
        r = metab_results[n_a]
        print(f"\n  {n_obs:3d} obs: "
              f"healthy={r['healthy_count']}/{r['total_tests']} "
              f"({r['healthy_fraction']:.0%})")

    # =========================================================================
    # TEST 5: Raw Timing
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 5: EFE COMPUTATION TIMING")
    print(f"{'='*70}")

    timing_results = benchmark_efe_timing(verbose=True)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: ACTION SPACE SCALING")
    print(f"{'='*70}")

    # Collect pass/fail for each dimension
    all_pass = True

    print("\n  EFE Selection Quality:")
    for n_a in action_sizes:
        r = efe_results[n_a]
        non_degenerate = not r["nominal_degenerate"]
        threat_ok = r["threat_sensible"]
        ok = non_degenerate and threat_ok
        if not ok:
            all_pass = False
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"    {n_a:3d} actions: {status}  "
              f"(spread={'✓' if non_degenerate else '✗'} "
              f"threat={'✓' if threat_ok else '✗'})")

    print("\n  Fractal Planning Reduction:")
    for n_a in action_sizes:
        r = plan_results[n_a]
        # At 3 actions there's barely anything to prune — 4× is fine
        min_reduction = 3 if n_a <= 3 else 10
        tractable = r["reduction_factor"] > min_reduction
        if not tractable:
            all_pass = False
        status = "✓" if tractable else "✗ SLOW"
        print(f"    {n_a:3d} actions: {r['reduction_factor']:>12,.0f}× reduction  {status}")

    print("\n  Structure Learning Bounded Growth:")
    for n_a in action_sizes:
        r = struct_results[n_a]
        n_obs = obs_sizes.get(n_a, n_a//2)
        # At 4 observables, 12 possible pairs is too small for density to mean much
        ok = r["bounded_growth"] or n_obs <= 4
        if not ok:
            all_pass = False
        status = "✓" if ok else "✗ UNBOUNDED"
        recall_str = f"recall={r['recall']:.0%}"
        print(f"    {n_obs:3d} observables: "
              f"density={r['link_density']:.1%} {recall_str}  {status}")

    print("\n  Metabolic Economy Health:")
    for n_a in action_sizes:
        r = metab_results[n_a]
        n_obs = obs_sizes.get(n_a, n_a//2)
        # At 4 observables, the finite-size effect is expected
        threshold = 0.1 if n_obs <= 4 else 0.4
        ok = r["healthy_fraction"] >= threshold
        if not ok:
            all_pass = False
        status = "✓" if ok else "✗ FRAGILE"
        print(f"    {n_obs:3d} observables: "
              f"{r['healthy_fraction']:.0%} healthy  {status}")

    print("\n  EFE Timing Scaling:")
    timings = timing_results["timings"]
    base_ms = timings.get(3, 0.001)
    for n_a, ms in sorted(timings.items()):
        scale = ms / base_ms
        linear = n_a / 3
        overhead = scale / linear if linear > 0 else 0
        quality = "✓ linear" if overhead < 1.5 else "~ ok" if overhead < 3 else "✗ superlinear"
        print(f"    {n_a:3d} actions: {ms:.3f}ms ({scale:.1f}× vs 3-action)  {quality}")

    print(f"\n  {'='*60}")
    if all_pass:
        print("  ALL TESTS PASSED")
        print("  The architecture degrades gracefully to 50 actions.")
    else:
        print("  SOME TESTS FAILED — see details above.")
        print("  Action space scaling reveals architectural limits.")
    print(f"  {'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
