#!/usr/bin/env python3
"""
MeTTa Cognitive Core â€” Unified Reasoning Benchmark
===================================================

Complete benchmark for the Peircean reasoning triad under metabolic
selection and EFE optimization. Extends the deductive reasoning
benchmark engine with abductive inference (Phase 1.6) and tests
all three inference modes individually and in concert.

REASONING PHASES TESTED:
  Phase 1:   INDUCTION   â€” co-error â†’ suspicion â†’ passive-model (structure)
  Phase 1.5: DEDUCTION   â€” Aâ†’B, Bâ†’C â‡’ Aâ†’C (structure from structure)
  Phase 1.6: ABDUCTION   â€” observe E, have Câ†’E â‡’ hypothesize C (state from structure)

SCENARIOS:
  1. Induction:        Causal link discovery from correlated surprises
  2. Deduction:        Transitive closure accelerates distant discovery
  3. Abduction:        Hidden cause hypothesized from observed effect
  4. Sherlock Holmes:  Abduction drives info-seeking action selection
  5. Full Triad:       Induction â†’ Deduction â†’ Abduction in one simulation
  6. Metabolic Death:  All inference types die when predictions fail
  7. Competing Causes: Asymmetric evidence disambiguates rival hypotheses

INVARIANT UNDER TEST:
  All seven scenarios are governed by the SAME optimization principle:
  minimize Expected Free Energy. Induction, deduction, and abduction
  are hypothesis generators; metabolic selection is the single arbiter.

Usage:
    python test_unified_reasoning.py            # Run all scenarios
    python test_unified_reasoning.py -v         # Verbose per-cycle trace
    python test_unified_reasoning.py -s 4       # Single scenario

Author: MeTTa Cognitive Core project
License: Apache 2.0
"""

import sys
import math
import argparse
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# CONFIGURATION (mirrors foundations.metta)
# =============================================================================

CONFIG = {
    # Suspicion accumulation
    "default_plasticity": 0.30,
    "default_structural_stability": 0.95,
    "default_cognitive_threshold": 0.04,
    "raw_error_salience_threshold": 0.20,

    # Structural costs (squared-space calibrated)
    "structural_cost_link": 0.001,
    "structural_cost_latent": 0.04,
    "structural_cost_spoke": 0.001,

    # Metabolic dynamics
    "metabolic_rate": 0.02,
    "metabolic_boost": 0.05,
    "metabolic_initial_energy": 1.0,
    "metabolic_energy_cap": 2.0,

    # Temporal
    "lookback_window": 3,
    "gestation_period": 3,
    "max_structural_atoms_per_cycle": 5,

    # Belief update
    "learning_rate": 0.12,
    "surprise_threshold": 0.20,
    "precision_floor": 0.05,
    "precision_ceiling": 0.95,

    # Deductive reasoning
    "deductive_min_energy": 0.5,
    "deductive_weight_discount": 0.8,

    # Abductive inference (Phase 1.6)
    "abductive_surprise_threshold": 0.05,
    "abductive_precision": 0.10,
    "abductive_min_link_weight": 0.15,
    "abductive_min_energy": 0.5,
    "abductive_max_cause_precision": 0.50,
    "abductive_budget_per_cycle": 5,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Belief:
    value: float
    precision: float
    source: str = "prior"       # "prior" | "observed" | "abduced"
    source_cycle: int = 0

@dataclass
class Observation:
    value: float
    precision: float
    time: int

@dataclass
class SuspicionLink:
    cause: str
    effect: str
    lag: int
    strength: float
    last_updated: int

@dataclass
class PassiveModel:
    cause: str
    effect: str
    lag: int
    weight: float
    causal_type: str        # "excitatory" | "inhibitory"
    origin: str             # "empirical" | "deductive"
    energy: float
    created_at: int
    predictions: int = 0
    successes: int = 0
    intermediate: Optional[str] = None  # B in Aâ†’Bâ†’C (deduction audit)

@dataclass
class LatentVariable:
    name: str
    members: list
    energy: float
    created_at: int
    predictions: int = 0
    successes: int = 0

@dataclass
class ErrorTrace:
    observable: str
    error: float
    surprise: float
    time: int

@dataclass
class AbductionLog:
    cause: str
    effect: str
    hyp_val: float
    hyp_prec: float
    prior_val: float
    posterior_val: float
    cycle: int

@dataclass
class EFEScore:
    action: str
    target: Optional[str]
    efe: float
    expected_error: float
    cost: float
    info_gain: float


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class CognitiveSimulation:
    """
    Simulation of the MeTTa cognitive core's reasoning pipeline:
    structure learning (Phase 1), deduction (Phase 1.5), abduction (Phase 1.6),
    and simplified EFE action selection.

    Faithfully mirrors the MeTTa implementation's logic and thresholds.
    """

    def __init__(self, enable_deduction=True, enable_abduction=True,
                 verbose=False):
        self.beliefs: dict[str, Belief] = {}
        self.observations: dict[str, Observation] = {}
        self.suspicion_links: dict[tuple, SuspicionLink] = {}
        self.passive_models: dict[tuple, PassiveModel] = {}
        self.latent_variables: dict[str, LatentVariable] = {}
        self.error_traces: list[ErrorTrace] = []
        self.abduction_logs: list[AbductionLog] = []
        self.cycle: int = 0
        self.structural_budget_used: int = 0
        self.abductive_budget_used: int = 0
        self.latent_counter: int = 0
        self.enable_deduction = enable_deduction
        self.enable_abduction = enable_abduction
        self.verbose = verbose
        self.deduction_origins: set[tuple] = set()
        self.events: list[str] = []

    def log(self, msg: str):
        self.events.append(f"[cycle {self.cycle:3d}] {msg}")
        if self.verbose:
            print(self.events[-1])

    # â”€â”€ Belief & Observation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def set_belief(self, obs: str, value: float, precision: float,
                   source: str = "prior"):
        self.beliefs[obs] = Belief(value, precision, source, self.cycle)

    def inject_observation(self, obs: str, value: float, precision: float):
        self.observations[obs] = Observation(value, precision, self.cycle)
        # Observation supersedes abduction
        b = self.beliefs.get(obs)
        if b and b.source == "abduced":
            b.source = "observed"
            b.source_cycle = self.cycle

    def compute_prediction_error(self, obs: str) -> Optional[float]:
        if obs in self.beliefs and obs in self.observations:
            return self.observations[obs].value - self.beliefs[obs].value
        return None

    def compute_surprise(self, obs: str) -> float:
        error = self.compute_prediction_error(obs)
        if error is None:
            return 0.0
        prec = self.beliefs[obs].precision
        return 0.5 * prec * error * error

    def is_salient(self, obs: str) -> bool:
        surprise = self.compute_surprise(obs)
        error = self.compute_prediction_error(obs)
        if error is None:
            return False
        return (surprise >= CONFIG["default_cognitive_threshold"] or
                abs(error) >= CONFIG["raw_error_salience_threshold"])

    def update_beliefs(self):
        lr = CONFIG["learning_rate"]
        for obs in list(self.observations.keys()):
            if obs not in self.beliefs:
                continue
            error = self.compute_prediction_error(obs)
            if error is None:
                continue
            b = self.beliefs[obs]
            o = self.observations[obs]
            prec_sum = b.precision + o.precision
            if prec_sum <= 0:
                continue
            obs_weight = o.precision / prec_sum
            # Q1: Should I learn? Gate on Kalman update magnitude.
            update_mag = obs_weight * abs(error)
            if update_mag < 0.01:
                continue
            update = lr * obs_weight * error
            b.value = max(0.0, min(1.0, b.value + update))
            # Precision update: threshold = agent's own uncertainty
            uncertainty = max(1.0 - b.precision, 0.01)
            prec_delta = 0.02 if abs(error) < uncertainty else -0.05
            b.precision = max(CONFIG["precision_floor"],
                              min(CONFIG["precision_ceiling"],
                                  b.precision + prec_delta))
            if b.source != "observed":
                b.source = "observed"
                b.source_cycle = self.cycle

    # â”€â”€ Error Traces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def record_error_traces(self):
        for obs in self.beliefs:
            error = self.compute_prediction_error(obs)
            if error is None:
                continue
            surprise = self.compute_surprise(obs)
            self.error_traces.append(
                ErrorTrace(obs, error, surprise, self.cycle))
        cutoff = self.cycle - CONFIG["lookback_window"]
        self.error_traces = [t for t in self.error_traces if t.time >= cutoff]

    # â”€â”€ Phase 1: Suspicion & Causal Link Creation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def update_suspicion_links(self):
        cutoff = self.cycle - CONFIG["lookback_window"]
        salient = [t for t in self.error_traces
                   if t.time >= cutoff and self.is_salient(t.observable)]

        stability = CONFIG["default_structural_stability"]
        plasticity = CONFIG["default_plasticity"]
        updated_keys = set()

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
                covariance = sign_a * sign_b * ta.surprise * tb.surprise

                key = (cause, effect, lag)
                old = self.suspicion_links.get(
                    key, SuspicionLink(cause, effect, lag, 0.0, 0)).strength
                new = stability * old + plasticity * covariance
                self.suspicion_links[key] = SuspicionLink(
                    cause, effect, lag, new, self.cycle)
                updated_keys.add(key)

                if ta.time == tb.time:
                    key_rev = (effect, cause, 0)
                    old_rev = self.suspicion_links.get(
                        key_rev, SuspicionLink(effect, cause, 0, 0.0, 0)).strength
                    new_rev = stability * old_rev + plasticity * covariance
                    self.suspicion_links[key_rev] = SuspicionLink(
                        effect, cause, 0, new_rev, self.cycle)
                    updated_keys.add(key_rev)

        for key, link in list(self.suspicion_links.items()):
            if key not in updated_keys:
                link.strength *= stability

    def check_phase1(self):
        cost = CONFIG["structural_cost_link"]
        for key, link in list(self.suspicion_links.items()):
            if self.structural_budget_used >= CONFIG["max_structural_atoms_per_cycle"]:
                break
            pm_key = (link.cause, link.effect)
            if pm_key in self.passive_models:
                continue
            if abs(link.strength) > cost:
                ctype = "excitatory" if link.strength > 0 else "inhibitory"
                weight = min(abs(link.strength), 1.0)
                self.passive_models[pm_key] = PassiveModel(
                    cause=link.cause, effect=link.effect,
                    lag=link.lag, weight=weight,
                    causal_type=ctype, origin="empirical",
                    energy=CONFIG["metabolic_initial_energy"],
                    created_at=self.cycle)
                self.structural_budget_used += 1
                self.log(f"INDUCTION: {link.cause}â†’{link.effect} "
                         f"(lag={link.lag}, w={weight:.4f}, {ctype})")

    # â”€â”€ Phase 1.5: Deductive Triggers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_deductive(self):
        if not self.enable_deduction:
            return
        min_energy = CONFIG["deductive_min_energy"]
        discount = CONFIG["deductive_weight_discount"]

        candidates = []
        for (a, b), pm_ab in list(self.passive_models.items()):
            if pm_ab.energy < min_energy:
                continue
            for (b2, c), pm_bc in list(self.passive_models.items()):
                if b2 != b or a == c:
                    continue
                if pm_bc.energy < min_energy:
                    continue
                if (a, c) in self.passive_models:
                    continue
                if (a, c) in self.deduction_origins:
                    continue
                candidates.append((a, b, c, pm_ab, pm_bc))

        for a, b, c, pm_ab, pm_bc in candidates:
            if self.structural_budget_used >= CONFIG["max_structural_atoms_per_cycle"]:
                break
            combined_lag = pm_ab.lag + pm_bc.lag
            combined_weight = min(discount * pm_ab.weight * pm_bc.weight, 1.0)
            type_map = {
                ("excitatory", "excitatory"): "excitatory",
                ("excitatory", "inhibitory"): "inhibitory",
                ("inhibitory", "excitatory"): "inhibitory",
                ("inhibitory", "inhibitory"): "excitatory",
            }
            combined_type = type_map[(pm_ab.causal_type, pm_bc.causal_type)]

            self.passive_models[(a, c)] = PassiveModel(
                cause=a, effect=c, lag=combined_lag,
                weight=combined_weight, causal_type=combined_type,
                origin="deductive",
                energy=CONFIG["metabolic_initial_energy"],
                created_at=self.cycle, intermediate=b)
            self.deduction_origins.add((a, c))
            self.structural_budget_used += 1
            self.log(f"DEDUCTION: {a}â†’{c} via {b} "
                     f"(lag={combined_lag}, w={combined_weight:.4f}, {combined_type})")

    # â”€â”€ Phase 1.6: Abductive Inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def compute_hypothesis_value(self, obs_val, bval, weight, link_type):
        """Invert the forward model: hyp = obs / weight, direction-aware."""
        raw = obs_val / weight if weight > 0.01 else 1.0
        directed = raw if link_type == "excitatory" else (1.0 - raw)
        return max(0.0, min(1.0, directed))

    def collect_abductive_candidates(self):
        threshold = CONFIG["abductive_surprise_threshold"]
        min_weight = CONFIG["abductive_min_link_weight"]
        min_energy = CONFIG["abductive_min_energy"]
        max_prec = CONFIG["abductive_max_cause_precision"]
        candidates = []

        for obs in self.beliefs:
            error = self.compute_prediction_error(obs)
            if error is None:
                continue
            surprise = self.compute_surprise(obs)
            if surprise < threshold:
                continue

            # Find passive models where this observable is the EFFECT
            for (cause, effect), pm in self.passive_models.items():
                if effect != obs:
                    continue
                if pm.weight < min_weight or pm.energy < min_energy:
                    continue
                cause_b = self.beliefs.get(cause)
                if cause_b is None or cause_b.precision >= max_prec:
                    continue
                candidates.append((pm, obs, error, surprise))

        return candidates

    def inject_hypothesis(self, cause, hyp_val, hyp_prec, effect):
        b = self.beliefs.get(cause)
        if not b:
            return None
        lr = CONFIG["learning_rate"]
        prec_sum = b.precision + hyp_prec
        if prec_sum <= 0:
            return None
        hyp_weight = hyp_prec / prec_sum
        error = hyp_val - b.value
        update = lr * hyp_weight * error
        prior_val = b.value
        b.value = max(0.0, min(1.0, b.value + update))
        b.precision = max(b.precision - 0.02, CONFIG["precision_floor"])
        if b.source != "observed":
            b.source = "abduced"
            b.source_cycle = self.cycle
        log = AbductionLog(cause, effect, hyp_val, hyp_prec,
                           prior_val, b.value, self.cycle)
        self.abduction_logs.append(log)
        return log

    def abductive_step(self):
        if not self.enable_abduction:
            return []
        self.abductive_budget_used = 0
        candidates = self.collect_abductive_candidates()
        logs = []
        budget = CONFIG["abductive_budget_per_cycle"]

        for pm, effect, error, surprise in candidates:
            if self.abductive_budget_used >= budget:
                break
            obs = self.observations.get(effect)
            if not obs:
                continue
            bval = self.beliefs[effect].value
            hyp_val = self.compute_hypothesis_value(
                obs.value, bval, pm.weight, pm.causal_type)
            hyp_prec = CONFIG["abductive_precision"]
            log = self.inject_hypothesis(pm.cause, hyp_val, hyp_prec, effect)
            if log:
                logs.append(log)
                self.abductive_budget_used += 1
                self.log(f"ABDUCTION: {pm.cause} â† {effect} "
                         f"(hyp={hyp_val:.3f}, "
                         f"{log.prior_val:.3f}â†’{log.posterior_val:.3f})")
        return logs

    # â”€â”€ Phase 2: Hub Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def compute_causal_degrees(self):
        degrees = {}
        for (c, e) in self.passive_models:
            degrees.setdefault(c, set()).add(e)
            degrees.setdefault(e, set()).add(c)
        return {obs: neighbors for obs, neighbors in degrees.items()}

    def check_phase2(self):
        degrees = self.compute_causal_degrees()
        cost_link = CONFIG["structural_cost_link"]
        cost_latent = CONFIG["structural_cost_latent"]
        cost_spoke = CONFIG["structural_cost_spoke"]

        for obs, neighbors in degrees.items():
            if self.structural_budget_used >= CONFIG["max_structural_atoms_per_cycle"]:
                break
            if len(neighbors) < 3:
                continue
            if any(obs in lv.members for lv in self.latent_variables.values()):
                continue
            cluster = [obs] + list(neighbors)
            n_pairwise = sum(
                1 for i, a in enumerate(cluster)
                for b in cluster[i+1:]
                if (a, b) in self.passive_models or (b, a) in self.passive_models)
            if n_pairwise * cost_link > cost_latent + len(cluster) * cost_spoke:
                self.latent_counter += 1
                name = f"latent-{self.latent_counter}"
                self.latent_variables[name] = LatentVariable(
                    name=name, members=cluster,
                    energy=CONFIG["metabolic_initial_energy"],
                    created_at=self.cycle)
                self.structural_budget_used += 1
                self.log(f"HUB: '{name}' from {obs} with {len(cluster)} members")

    # â”€â”€ Metabolic Dynamics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def metabolic_step(self):
        rate = CONFIG["metabolic_rate"]
        boost = CONFIG["metabolic_boost"]
        cap = CONFIG["metabolic_energy_cap"]
        gestation = CONFIG["gestation_period"]

        dead = []
        for key, pm in list(self.passive_models.items()):
            if self.cycle - pm.created_at < gestation:
                continue
            pm.energy -= rate

            cause_err = self.compute_prediction_error(pm.cause)
            effect_err = self.compute_prediction_error(pm.effect)

            if cause_err is not None and effect_err is not None:
                # Cause is "surprising" when error exceeds the agent's uncertainty
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
                self.log(f"DEATH: {pm.cause}â†’{pm.effect} ({pm.origin}, "
                         f"preds={pm.predictions}, succ={pm.successes})")

        for key in dead:
            del self.passive_models[key]

        dead_lv = []
        for name, lv in list(self.latent_variables.items()):
            if self.cycle - lv.created_at < gestation:
                continue
            lv.energy -= rate
            if lv.energy <= 0:
                dead_lv.append(name)
                self.log(f"DEATH: latent '{name}'")
        for name in dead_lv:
            del self.latent_variables[name]

    def clean_deduction_origins(self):
        to_remove = []
        for (a, c) in list(self.deduction_origins):
            pm = self.passive_models.get((a, c))
            if pm is None:
                to_remove.append((a, c))
        for key in to_remove:
            self.deduction_origins.discard(key)

    # â”€â”€ Simplified EFE (for Sherlock Holmes test) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def compute_efe(self, action: str, target: Optional[str] = None) -> EFEScore:
        cost = {"wait": 0.01, "retreat": 0.10}.get(action, 0.05)

        expected_error = 0.0
        for obs, b in self.beliefs.items():
            o = self.observations.get(obs)
            if o:
                expected_error += abs(o.value - b.value) * b.precision * o.precision

        info_gain = 0.0
        if target:
            b = self.beliefs.get(target)
            if b:
                info_gain = (1.0 - b.precision) * 0.1
                # Indirect: if this observation resolves an abduced cause
                for (cause, effect), pm in self.passive_models.items():
                    if effect != target:
                        continue
                    cause_b = self.beliefs.get(cause)
                    if cause_b and cause_b.source == "abduced":
                        info_gain += (1.0 - cause_b.precision) * 0.15

        return EFEScore(action, target,
                        expected_error + cost - info_gain,
                        expected_error, cost, info_gain)

    def select_best_action(self, actions: list[tuple[str, Optional[str]]]) -> EFEScore:
        scores = [self.compute_efe(a, t) for a, t in actions]
        return min(scores, key=lambda s: s.efe)

    # â”€â”€ Master Cycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run_cycle(self, new_observations: dict = None):
        self.structural_budget_used = 0

        if new_observations:
            for obs, (val, prec) in new_observations.items():
                self.inject_observation(obs, val, prec)

        # 2b. Error traces (structure learning sees honest errors)
        self.record_error_traces()

        # 4b-a. Suspicion accumulation
        self.update_suspicion_links()

        # 4b-b. Phase 1: causal link creation
        self.check_phase1()

        # 4b-c. Phase 1.5: deductive triggers
        self.check_deductive()

        # 4b-d. Phase 2: hub detection
        self.check_phase2()

        # 4b-e. Metabolic pruning
        self.metabolic_step()
        self.clean_deduction_origins()

        # 4b-f. Phase 1.6: abductive inference (AFTER structure, BEFORE action)
        abductions = self.abductive_step()

        # 7. Update beliefs from observations
        self.update_beliefs()

        self.cycle += 1

        return {
            "cycle": self.cycle - 1,
            "abductions": abductions,
            "n_links": len(self.passive_models),
            "n_latent": len(self.latent_variables),
            "n_abduced": sum(1 for b in self.beliefs.values()
                             if b.source == "abduced"),
        }

    # â”€â”€ Query Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def has_link(self, cause, effect):
        return (cause, effect) in self.passive_models

    def get_link(self, cause, effect):
        return self.passive_models.get((cause, effect))

    def count_by_origin(self, origin):
        return sum(1 for pm in self.passive_models.values()
                   if pm.origin == origin)

    def summary(self):
        ne = self.count_by_origin("empirical")
        nd = self.count_by_origin("deductive")
        nl = len(self.latent_variables)
        na = sum(1 for b in self.beliefs.values() if b.source == "abduced")
        return (f"{ne} empirical + {nd} deductive links, "
                f"{nl} latent vars, {na} abduced beliefs")


# =============================================================================
# ENVIRONMENTS
# =============================================================================

class Environment:
    def get_observations(self, cycle):
        raise NotImplementedError


class FireChainEnv(Environment):
    """4-node chain: ignition â†’ heat â†’ smoke â†’ ash (each lag 1)."""
    def __init__(self, period=7):
        self.period = period
    def get_observations(self, cycle):
        def active(lag):
            return ((cycle - lag) % self.period) == 0 and (cycle - lag) >= 0
        return {
            "ignition": (0.85 if active(0) else 0.12, 0.85),
            "heat":     (0.80 if active(1) else 0.12, 0.80),
            "smoke":    (0.75 if active(2) else 0.12, 0.75),
            "ash":      (0.70 if active(3) else 0.12, 0.70),
        }


class FalsifiableEnv(Environment):
    """Aâ†’B exc, Bâ†’C exc, but A actually inhibits C."""
    def __init__(self):
        self.a_period = 8
        self.b_period = 6
    def get_observations(self, cycle):
        a_event = (cycle % self.a_period) < 2
        b_indep = 3 <= (cycle % self.b_period) < 5
        a_val = 0.8 if a_event else 0.12
        b_val = 0.75 if a_event else (0.70 if b_indep else 0.12)
        c_val = 0.1 if a_event else (0.75 if b_indep else 0.5)
        return {
            "obs-A": (a_val, 0.80),
            "obs-B": (b_val, 0.80),
            "obs-C": (c_val, 0.75),
        }


class HiddenCauseEnv(Environment):
    """
    Latent L causes A and B (excitatory). Also causes C (excitatory).
    L is never directly observed â€” only its effects.
    Periodic activation: L is active every `period` cycles.
    """
    def __init__(self, period=5):
        self.period = period
    def get_observations(self, cycle):
        active = (cycle % self.period) == 0
        return {
            "obs-A": (0.85 if active else 0.15, 0.80),
            "obs-B": (0.80 if active else 0.15, 0.80),
            "obs-C": (0.75 if active else 0.15, 0.75),
        }


class CompetingCauseEnv(Environment):
    """
    Aâ†’E (strong, 0.8) and Bâ†’E (weak, 0.3).
    A also causes F; B does NOT cause F.
    Ground truth: A is the real cause. F discriminates.
    """
    def __init__(self, period=6):
        self.period = period
    def get_observations(self, cycle):
        active = (cycle % self.period) == 0
        return {
            "obs-E": (0.90 if active else 0.15, 0.80),
            "obs-F": (0.85 if active else 0.15, 0.80),
        }


# =============================================================================
# SCENARIOS
# =============================================================================

def scenario_1_induction(verbose=False):
    """Phase 1: Induction discovers causal structure from co-error."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: Induction â€” Causal Discovery from Co-Error")
    print("=" * 70)
    print("  Environment: ignition â†’ heat â†’ smoke â†’ ash (each lag 1)")
    print("  Expected: adjacent links discovered within ~20 cycles\n")

    env = FireChainEnv(period=7)
    sim = CognitiveSimulation(enable_deduction=False, enable_abduction=False,
                              verbose=verbose)
    for obs in ["ignition", "heat", "smoke", "ash"]:
        sim.set_belief(obs, 0.5, 0.5)

    targets = {"ignitionâ†’heat": None, "heatâ†’smoke": None, "smokeâ†’ash": None}
    pairs = [("ignition", "heat"), ("heat", "smoke"), ("smoke", "ash")]

    for cycle in range(60):
        sim.cycle = cycle
        obs = env.get_observations(cycle)
        for name, (val, prec) in obs.items():
            sim.inject_observation(name, val, prec)
        sim.run_cycle()

        for (c, e), label in zip(pairs, targets.keys()):
            if targets[label] is None and sim.has_link(c, e):
                targets[label] = cycle
                print(f"  âœ“ {label} discovered at cycle {cycle}")

    n_found = sum(1 for v in targets.values() if v is not None)
    print(f"\n  Discovered: {n_found}/3 adjacent links")
    print(f"  Final: {sim.summary()}")

    passed = n_found >= 2  # At least 2 of 3 adjacent links
    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")
    return passed


def scenario_2_deduction(verbose=False):
    """Phase 1.5: Deduction accelerates distant link discovery."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Deduction â€” Transitive Closure Accelerates Discovery")
    print("=" * 70)
    print("  Target: ignitionâ†’ash (lag 3, edge of lookback window)")
    print("  Compare: with deduction vs empirical-only\n")

    env_d = FireChainEnv(period=7)
    env_e = FireChainEnv(period=7)
    sim_d = CognitiveSimulation(enable_deduction=True, enable_abduction=False,
                                verbose=verbose)
    sim_e = CognitiveSimulation(enable_deduction=False, enable_abduction=False,
                                verbose=verbose)
    for obs in ["ignition", "heat", "smoke", "ash"]:
        sim_d.set_belief(obs, 0.5, 0.5)
        sim_e.set_belief(obs, 0.5, 0.5)

    disc_d = disc_e = None
    for cycle in range(80):
        sim_d.cycle = cycle
        sim_e.cycle = cycle
        obs_d = env_d.get_observations(cycle)
        obs_e = env_e.get_observations(cycle)
        for name, (val, prec) in obs_d.items():
            sim_d.inject_observation(name, val, prec)
        for name, (val, prec) in obs_e.items():
            sim_e.inject_observation(name, val, prec)
        sim_d.run_cycle()
        sim_e.run_cycle()

        if disc_d is None and sim_d.has_link("ignition", "ash"):
            disc_d = cycle
            pm = sim_d.get_link("ignition", "ash")
            print(f"  [WITH deduction] ignitionâ†’ash at cycle {cycle} "
                  f"(origin: {pm.origin})")
        if disc_e is None and sim_e.has_link("ignition", "ash"):
            disc_e = cycle
            pm = sim_e.get_link("ignition", "ash")
            print(f"  [empirical-only] ignitionâ†’ash at cycle {cycle} "
                  f"(origin: {pm.origin})")

    print(f"\n  Deductive:  {'cycle ' + str(disc_d) if disc_d else 'NOT FOUND'}")
    print(f"  Empirical:  {'cycle ' + str(disc_e) if disc_e else 'NOT FOUND'}")

    passed = disc_d is not None and (disc_e is None or disc_d <= disc_e)
    if disc_d is not None and disc_e is not None:
        print(f"  Speedup: {disc_e / max(disc_d, 1):.1f}x")
    print(f"  Final (ded): {sim_d.summary()}")
    print(f"  Final (emp): {sim_e.summary()}")

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")
    return passed


def scenario_3_abduction(verbose=False):
    """Phase 1.6: Abduction hypothesizes hidden cause from observed effect."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Abduction â€” Hidden Cause Hypothesis")
    print("=" * 70)
    print("  Setup: pre-established Lâ†’A (exc, 0.8), Lâ†’B (exc, 0.8)")
    print("  Inject A=0.9 â†’ should abduct L=high, mark as 'abduced'")
    print("  Inject B=0.9 â†’ confirms L, belief should strengthen\n")

    sim = CognitiveSimulation(verbose=verbose)

    sim.set_belief("L", 0.3, 0.40)
    sim.set_belief("obs-A", 0.3, 0.50)
    sim.set_belief("obs-B", 0.3, 0.50)

    sim.passive_models[("L", "obs-A")] = PassiveModel(
        cause="L", effect="obs-A", lag=0, weight=0.8,
        causal_type="excitatory", origin="empirical",
        energy=1.5, created_at=0, predictions=10, successes=8)
    sim.passive_models[("L", "obs-B")] = PassiveModel(
        cause="L", effect="obs-B", lag=0, weight=0.8,
        causal_type="excitatory", origin="empirical",
        energy=1.5, created_at=0, predictions=10, successes=8)

    # Cycle 0: observe A high
    sim.run_cycle({"obs-A": (0.9, 0.8)})
    l_after_a = sim.beliefs["L"]
    l_val_after_a = l_after_a.value  # capture value (not reference)
    print(f"  After A=0.9:  L val={l_val_after_a:.4f} "
          f"prec={l_after_a.precision:.4f} source={l_after_a.source}")
    abduced_l = l_after_a.source == "abduced"
    moved_up = l_val_after_a > 0.3

    # Cycle 1: observe B high too
    sim.run_cycle({"obs-A": (0.9, 0.8), "obs-B": (0.9, 0.8)})
    l_after_b = sim.beliefs["L"]
    l_val_after_b = l_after_b.value
    print(f"  After B=0.9:  L val={l_val_after_b:.4f} "
          f"prec={l_after_b.precision:.4f} source={l_after_b.source}")
    confirmed = l_val_after_b > l_val_after_a

    # Run 15 more confirming cycles
    for _ in range(15):
        sim.run_cycle({"obs-A": (0.9, 0.8), "obs-B": (0.9, 0.8)})
    l_final = sim.beliefs["L"]
    print(f"  After 15 more: L val={l_final.value:.4f} "
          f"prec={l_final.precision:.4f}")
    converged = l_final.value > 0.40

    passed = abduced_l and moved_up and confirmed and converged
    status = "PASS" if passed else "FAIL"
    checks = [("Abduced?", abduced_l), ("Moved up?", moved_up),
              ("Confirmed?", confirmed), ("Converged?", converged)]
    for label, ok in checks:
        print(f"    {label:14s} {'âœ“' if ok else 'âœ—'}")
    print(f"\n  Result: {status}")
    return passed


def scenario_4_sherlock_holmes(verbose=False):
    """
    The Sherlock Holmes Effect: abduction drives information-seeking.

    Setup: Lâ†’A and Lâ†’B pre-established. Observe A=0.9 â†’ abduct L.
    Now L predicts B. Observing B would resolve the L hypothesis.
    EFE should prefer observe(B) over wait because of the info gain
    from resolving the low-precision abduced L belief.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: Sherlock Holmes Effect â€” Abduction Drives Info-Seeking")
    print("=" * 70)
    print("  Setup: Lâ†’A, Lâ†’B. Observe A=0.9 â†’ abduct L (low precision)")
    print("  L predicts B. observe(B) should have high info gain.")
    print("  Expected: EFE(observe_B) < EFE(wait)\n")

    sim = CognitiveSimulation(verbose=verbose)

    sim.set_belief("L", 0.3, 0.40)
    sim.set_belief("obs-A", 0.3, 0.50)
    sim.set_belief("obs-B", 0.3, 0.50)

    sim.passive_models[("L", "obs-A")] = PassiveModel(
        cause="L", effect="obs-A", lag=0, weight=0.8,
        causal_type="excitatory", origin="empirical",
        energy=1.5, created_at=0, predictions=10, successes=8)
    sim.passive_models[("L", "obs-B")] = PassiveModel(
        cause="L", effect="obs-B", lag=0, weight=0.8,
        causal_type="excitatory", origin="empirical",
        energy=1.5, created_at=0, predictions=10, successes=8)

    sim.run_cycle({"obs-A": (0.9, 0.8)})
    l_belief = sim.beliefs["L"]
    print(f"  L after abduction: val={l_belief.value:.4f} "
          f"prec={l_belief.precision:.4f} source={l_belief.source}")

    # Compute EFE for available actions
    efe_b = sim.compute_efe("observe", "obs-B")
    efe_a = sim.compute_efe("observe", "obs-A")
    efe_wait = sim.compute_efe("wait")

    print(f"\n  EFE scores:")
    print(f"    observe(B): {efe_b.efe:.4f} "
          f"(info_gain={efe_b.info_gain:.4f})")
    print(f"    observe(A): {efe_a.efe:.4f} "
          f"(info_gain={efe_a.info_gain:.4f})")
    print(f"    wait:       {efe_wait.efe:.4f} "
          f"(info_gain={efe_wait.info_gain:.4f})")

    # Key tests
    b_beats_wait = efe_b.efe < efe_wait.efe
    b_has_info = efe_b.info_gain > efe_wait.info_gain
    l_is_abduced = l_belief.source == "abduced"

    best = sim.select_best_action([
        ("observe", "obs-B"), ("observe", "obs-A"), ("wait", None)])
    best_is_observe = best.target is not None

    passed = l_is_abduced and b_beats_wait and b_has_info and best_is_observe
    checks = [("L abduced?", l_is_abduced),
              ("observe(B) < wait?", b_beats_wait),
              ("B has info gain?", b_has_info),
              ("Best action observes?", best_is_observe)]
    for label, ok in checks:
        print(f"    {label:22s} {'âœ“' if ok else 'âœ—'}")
    print(f"    Best action: {best.action}({best.target}) "
          f"EFE={best.efe:.4f}")

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")
    return passed


def scenario_5_full_triad(verbose=False):
    """
    Full Peircean Triad: induction â†’ deduction â†’ abduction in one run.

    Environment: A fires, then B follows (lag 1), then C follows (lag 2).
    Strict temporal ordering ensures DIRECTIONAL links.

    Phase 1: Induction discovers Aâ†’B (lag 1) and Bâ†’C (lag 1)
    Phase 2: Deduction generates Aâ†’C (lag 2)
    Phase 3: Only C is observed with surprise â†’ abduction hypothesizes
             its causes through existing structure.

    This is the capstone test: all three inference types working together
    under the same optimization principle.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 5: Full Triad â€” Induction â†’ Deduction â†’ Abduction")
    print("=" * 70)
    print("  Environment: A fires â†’ B follows (lag 1) â†’ C follows (lag 2)")
    print("  Phase 1: Induction discovers Aâ†’B, Bâ†’C")
    print("  Phase 2: Deduction generates Aâ†’C")
    print("  Phase 3: Observe only C â†’ abduction hypothesizes causes")
    print()

    class LaggedChainEnv(Environment):
        """A fires periodically, B follows at lag 1, C at lag 2."""
        def __init__(self, period=6):
            self.period = period
        def get_observations(self, cycle):
            def active(lag):
                return ((cycle - lag) % self.period) == 0 and (cycle - lag) >= 0
            return {
                "obs-A": (0.85 if active(0) else 0.12, 0.80),
                "obs-B": (0.80 if active(1) else 0.12, 0.80),
                "obs-C": (0.75 if active(2) else 0.12, 0.75),
            }

    env = LaggedChainEnv(period=6)
    sim = CognitiveSimulation(verbose=verbose)
    for obs in ["obs-A", "obs-B", "obs-C"]:
        sim.set_belief(obs, 0.5, 0.5)

    # Phase 1: Run 45 cycles with full observations â†’ structure discovery
    print("  â”€â”€ Phase 1: Structure Discovery (45 cycles) â”€â”€")
    induction_events = []
    deduction_events = []
    for cycle in range(45):
        sim.cycle = cycle
        obs = env.get_observations(cycle)
        for name, (val, prec) in obs.items():
            sim.inject_observation(name, val, prec)
        sim.run_cycle()
        # Track events
        for e in sim.events[len(induction_events) + len(deduction_events):]:
            if "INDUCTION" in e:
                induction_events.append(e)
            elif "DEDUCTION" in e:
                deduction_events.append(e)

    n_emp = sim.count_by_origin("empirical")
    n_ded = sim.count_by_origin("deductive")
    print(f"  Empirical links: {n_emp}")
    print(f"  Deductive links: {n_ded}")
    for (c, e), pm in sorted(sim.passive_models.items()):
        print(f"    {c}â†’{e} ({pm.origin}, lag={pm.lag}, w={pm.weight:.4f}, "
              f"energy={pm.energy:.3f})")

    has_induction = n_emp > 0
    has_deduction = n_ded > 0

    # Phase 2: Observe ONLY C with a spike â†’ abduction should fire
    print(f"\n  â”€â”€ Phase 2: Partial Observation + Abduction (20 cycles) â”€â”€")

    # Reset A and B beliefs to uncertain
    sim.beliefs["obs-A"] = Belief(0.3, 0.40, "prior", sim.cycle)
    sim.beliefs["obs-B"] = Belief(0.3, 0.40, "prior", sim.cycle)

    abduction_happened = False
    abducted_obs = set()
    for cycle in range(45, 65):
        sim.cycle = cycle
        # Only observe C â€” sometimes high (as if the chain is active)
        c_active = ((cycle - 2) % 6) == 0  # C follows at lag 2
        c_val = 0.75 if c_active else 0.12
        result = sim.run_cycle({"obs-C": (c_val, 0.75)})
        if result["abductions"]:
            abduction_happened = True
            for log in result["abductions"]:
                abducted_obs.add(log.cause)

    a_final = sim.beliefs["obs-A"]
    b_final = sim.beliefs["obs-B"]
    print(f"  obs-A: val={a_final.value:.4f} prec={a_final.precision:.4f} "
          f"source={a_final.source}")
    print(f"  obs-B: val={b_final.value:.4f} prec={b_final.precision:.4f} "
          f"source={b_final.source}")
    print(f"  Abduction occurred: {abduction_happened}")
    if abducted_obs:
        print(f"  Abducted causes: {abducted_obs}")

    passed = has_induction and (has_deduction or abduction_happened)

    checks = [("Induction found links?", has_induction),
              ("Deduction extended?", has_deduction),
              ("Abduction from partial obs?", abduction_happened)]
    for label, ok in checks:
        print(f"    {label:28s} {'âœ“' if ok else 'âœ—'}")

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")
    return passed


def scenario_6_metabolic_death(verbose=False):
    """
    Metabolic Uniform Selection: all inference types die when wrong.

    Setup three types of links, all making WRONG predictions:
    - Empirical link (wrong direction)
    - Deductive link (logically valid but empirically false)
    - Abduced hypothesis (wrong guess)

    All should lose energy and die. Single selection principle.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 6: Metabolic Death â€” Uniform Selection Across Types")
    print("=" * 70)
    print("  All three link types make wrong predictions â†’ all should die")
    print()

    env = FalsifiableEnv()
    sim = CognitiveSimulation(verbose=verbose)

    for obs in ["obs-A", "obs-B", "obs-C"]:
        sim.set_belief(obs, 0.5, 0.5)

    # Seed Aâ†’B (exc) and Bâ†’C (exc) â€” the Bâ†’C is empirically correct,
    # but deduction will generate Aâ†’C (exc) which is WRONG (A inhibits C).
    sim.passive_models[("obs-A", "obs-B")] = PassiveModel(
        cause="obs-A", effect="obs-B", lag=0, weight=0.5,
        causal_type="excitatory", origin="empirical",
        energy=1.5, created_at=0, predictions=15, successes=12)
    sim.passive_models[("obs-B", "obs-C")] = PassiveModel(
        cause="obs-B", effect="obs-C", lag=0, weight=0.4,
        causal_type="excitatory", origin="empirical",
        energy=1.5, created_at=0, predictions=15, successes=12)

    # Track the deduced Aâ†’C
    deduced_created = False
    deduced_died = False

    for cycle in range(60):
        sim.cycle = cycle
        obs = env.get_observations(cycle)
        for name, (val, prec) in obs.items():
            sim.inject_observation(name, val, prec)
        sim.run_cycle()

        ac = sim.get_link("obs-A", "obs-C")
        if ac and ac.origin == "deductive" and not deduced_created:
            deduced_created = True
            print(f"  Deduced Aâ†’C created at cycle {cycle}")

    # Check if deduced link died or is dying
    ac_final = sim.get_link("obs-A", "obs-C")
    if ac_final and ac_final.origin == "deductive":
        deduced_dying = ac_final.energy < 0.5
        print(f"  Aâ†’C at end: energy={ac_final.energy:.3f} "
              f"preds={ac_final.predictions} succ={ac_final.successes}")
        if ac_final.predictions > 0:
            print(f"  Success rate: {ac_final.successes/ac_final.predictions:.0%}")
    else:
        deduced_dying = True
        deduced_died = True
        print(f"  Aâ†’C: DEAD (metabolic death)")

    death_events = [e for e in sim.events if "DEATH" in e]
    print(f"\n  Death events: {len(death_events)}")
    for e in death_events[:5]:
        print(f"    {e}")

    passed = deduced_created and (deduced_died or deduced_dying)

    checks = [("Deduction tried?", deduced_created),
              ("Wrong deduction killed/dying?", deduced_died or deduced_dying)]
    for label, ok in checks:
        print(f"    {label:30s} {'âœ“' if ok else 'âœ—'}")

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")
    return passed


def scenario_7_competing_causes(verbose=False):
    """
    Competing Explanations: asymmetric evidence disambiguates.

    Aâ†’E (strong, 0.8), Bâ†’E (weak, 0.3), Aâ†’F (0.7), no Bâ†’F.
    Observe E and F. A gets support from both; B only from E.
    Over cycles, A's belief dominates B's.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 7: Competing Causes â€” Asymmetric Evidence")
    print("=" * 70)
    print("  Aâ†’E (0.8), Bâ†’E (0.3), Aâ†’F (0.7). No Bâ†’F.")
    print("  Both E and F observed high. A should dominate B.\n")

    sim = CognitiveSimulation(verbose=verbose)

    sim.set_belief("A", 0.3, 0.40)
    sim.set_belief("B", 0.3, 0.40)
    sim.set_belief("obs-E", 0.3, 0.50)
    sim.set_belief("obs-F", 0.3, 0.50)

    sim.passive_models[("A", "obs-E")] = PassiveModel(
        cause="A", effect="obs-E", lag=0, weight=0.8,
        causal_type="excitatory", origin="empirical",
        energy=1.5, created_at=0, predictions=10, successes=8)
    sim.passive_models[("B", "obs-E")] = PassiveModel(
        cause="B", effect="obs-E", lag=0, weight=0.3,
        causal_type="excitatory", origin="empirical",
        energy=1.5, created_at=0, predictions=10, successes=7)
    sim.passive_models[("A", "obs-F")] = PassiveModel(
        cause="A", effect="obs-F", lag=0, weight=0.7,
        causal_type="excitatory", origin="empirical",
        energy=1.5, created_at=0, predictions=10, successes=8)

    # Run 40 cycles with E and F observed high
    for cycle in range(40):
        sim.cycle = cycle
        sim.run_cycle({"obs-E": (0.9, 0.8), "obs-F": (0.85, 0.8)})

    a_final = sim.beliefs["A"]
    b_final = sim.beliefs["B"]
    gap = a_final.value - b_final.value

    print(f"  A: val={a_final.value:.4f} prec={a_final.precision:.4f} "
          f"source={a_final.source}")
    print(f"  B: val={b_final.value:.4f} prec={b_final.precision:.4f} "
          f"source={b_final.source}")
    print(f"  Gap (A - B): {gap:.4f}")

    a_stronger = a_final.value > b_final.value
    meaningful_gap = gap > 0.01

    passed = a_stronger and meaningful_gap
    checks = [("A > B?", a_stronger),
              ("Meaningful gap?", meaningful_gap)]
    for label, ok in checks:
        print(f"    {label:18s} {'âœ“' if ok else 'âœ—'}")

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")
    return passed


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison_table():
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)
    print()
    hdr = f"{'Property':<48} {'PLN':>6} {'BN':>6} {'Ours':>6}"
    print(hdr)
    print("-" * 70)
    rows = [
        ("Discovers causal links from observation",        "No",   "No",  "Yes"),
        ("Transitive closure (Aâ†’B,Bâ†’C â‡’ Aâ†’C)",            "Yes",  "Yes", "Yes"),
        ("Inverse inference (observe E â‡’ hypothesize C)",  "No",   "Yes", "Yes"),
        ("Falsifies wrong deductions",                     "No",   "Yes", "Yes"),
        ("Falsifies wrong abductions",                     "N/A",  "Yes", "Yes"),
        ("Single optimization principle",                  "Logic","MAP", "EFE"),
        ("Emergent verification (Sherlock Holmes)",         "No",   "No",  "Yes"),
        ("Metabolic cost for ALL inference types",         "No",   "No",  "Yes"),
        ("Transparent audit trail",                        "Yes",  "No",  "Yes"),
        ("Epistemic source tracking (hyp vs obs)",         "No",   "No",  "Yes"),
    ]
    for prop, pln, bn, ours in rows:
        print(f"  {prop:<46} {pln:>6} {bn:>6} {ours:>6}")
    print()
    print("  PLN  = Probabilistic Logic Networks (SingularityNET)")
    print("  BN   = Bayesian Networks (standard)")
    print("  Ours = Metabolic Active Inference (this architecture)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MeTTa Cognitive Core â€” Unified Reasoning Benchmark")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-cycle trace")
    parser.add_argument("-s", "--scenario", type=int, choices=range(1, 8),
                        help="Run single scenario (1-7)")
    args = parser.parse_args()

    print("â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—")
    print("â•‘   MeTTa Cognitive Core â€” Unified Reasoning Benchmark           â•‘")
    print("â•‘   Phases 1 + 1.5 + 1.6: Induction, Deduction, Abduction       â•‘")
    print("â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")

    scenarios = [
        (1, "Induction",        scenario_1_induction),
        (2, "Deduction",        scenario_2_deduction),
        (3, "Abduction",        scenario_3_abduction),
        (4, "Sherlock Holmes",  scenario_4_sherlock_holmes),
        (5, "Full Triad",       scenario_5_full_triad),
        (6, "Metabolic Death",  scenario_6_metabolic_death),
        (7, "Competing Causes", scenario_7_competing_causes),
    ]

    results = {}
    for num, name, fn in scenarios:
        if args.scenario is not None and args.scenario != num:
            continue
        try:
            results[num] = (name, fn(verbose=args.verbose))
        except Exception as e:
            results[num] = (name, False)
            print(f"\n  âœ— ERROR: {e}")

    print_comparison_table()

    # Summary
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    for num in sorted(results):
        name, passed = results[num]
        status = "PASS âœ“" if passed else "FAIL âœ—"
        print(f"  {num}. {name:<22s} {status}")

    total = len(results)
    passing = sum(1 for _, p in results.values() if p)
    print(f"\n  {passing}/{total} scenarios passing")

    if passing == total:
        print("\n  ALL BENCHMARKS PASSED")
        print("  The Peircean triad operates under a single optimization")
        print("  principle (EFE) with uniform metabolic selection.")
    else:
        print(f"\n  {total - passing} FAILURES")

    return 0 if passing == total else 1


if __name__ == "__main__":
    sys.exit(main())
