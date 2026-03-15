#!/usr/bin/env python3
"""
Metabolic Parameter Sensitivity Analysis
=========================================

Sweeps c_rent and c_reward across a grid to characterize the basin of
attraction for healthy system behavior. Tests two critical scenarios:

1. INDUCTION: Can the system discover causal links? (structure must survive)
2. METABOLIC DEATH: Do wrong hypotheses die? (garbage must be collected)

A healthy parameter regime satisfies BOTH: correct structure lives AND
wrong structure dies. The sweep maps which (rent, reward) pairs achieve this.

Output: console summary + heatmap data + CSV for paper figure.

Author: Project Dagaz
"""

import sys
import math
import csv
import os
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

# =============================================================================
# Inline simulation engine (extracted from test_unified_reasoning.py)
# =============================================================================

BASE_CONFIG = {
    "default_plasticity": 0.30,
    "default_structural_stability": 0.95,
    "default_cognitive_threshold": 0.04,
    "raw_error_salience_threshold": 0.20,
    "structural_cost_link": 0.001,
    "structural_cost_latent": 0.04,
    "structural_cost_spoke": 0.001,
    "metabolic_rate": 0.02,       # Will be swept
    "metabolic_boost": 0.05,      # Will be swept
    "metabolic_initial_energy": 1.0,
    "metabolic_energy_cap": 2.0,
    "lookback_window": 3,
    "gestation_period": 3,
    "max_structural_atoms_per_cycle": 5,
    "learning_rate": 0.12,
    "surprise_threshold": 0.20,
    "precision_floor": 0.05,
    "precision_ceiling": 0.95,
    "deductive_min_energy": 0.5,
    "deductive_weight_discount": 0.8,
    "abductive_surprise_threshold": 0.05,
    "abductive_precision": 0.10,
    "abductive_min_link_weight": 0.15,
    "abductive_min_energy": 0.5,
    "abductive_max_cause_precision": 0.50,
    "abductive_budget_per_cycle": 5,
}


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
    causal_type: str
    origin: str
    energy: float
    created_at: int
    predictions: int = 0
    successes: int = 0
    intermediate: Optional[str] = None

@dataclass
class ErrorTrace:
    observable: str
    error: float
    surprise: float
    time: int


class CognitiveSimulation:
    """Minimal simulation engine for metabolic sensitivity analysis."""

    def __init__(self, config: dict):
        self.config = config
        self.beliefs: dict[str, Belief] = {}
        self.observations: dict[str, Observation] = {}
        self.suspicion_links: dict[tuple, SuspicionLink] = {}
        self.passive_models: dict[tuple, PassiveModel] = {}
        self.error_traces: list[ErrorTrace] = []
        self.cycle: int = 0
        self.structural_budget_used: int = 0
        self.deduction_origins: set[tuple] = set()
        self.death_count: int = 0

    def set_belief(self, obs, value, precision, source="prior"):
        self.beliefs[obs] = Belief(value, precision, source, self.cycle)

    def inject_observation(self, obs, value, precision):
        self.observations[obs] = Observation(value, precision, self.cycle)

    def compute_prediction_error(self, obs):
        if obs in self.beliefs and obs in self.observations:
            return self.observations[obs].value - self.beliefs[obs].value
        return None

    def compute_surprise(self, obs):
        error = self.compute_prediction_error(obs)
        if error is None:
            return 0.0
        prec = self.beliefs[obs].precision
        return 0.5 * prec * error * error

    def is_salient(self, obs):
        surprise = self.compute_surprise(obs)
        error = self.compute_prediction_error(obs)
        if error is None:
            return False
        return (surprise >= self.config["default_cognitive_threshold"] or
                abs(error) >= self.config["raw_error_salience_threshold"])

    def update_beliefs(self):
        lr = self.config["learning_rate"]
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
            b.precision = max(self.config["precision_floor"],
                              min(self.config["precision_ceiling"],
                                  b.precision + prec_delta))

    def record_error_traces(self):
        for obs in self.beliefs:
            error = self.compute_prediction_error(obs)
            if error is None:
                continue
            surprise = self.compute_surprise(obs)
            self.error_traces.append(ErrorTrace(obs, error, surprise, self.cycle))
        cutoff = self.cycle - self.config["lookback_window"]
        self.error_traces = [t for t in self.error_traces if t.time >= cutoff]

    def update_suspicion_links(self):
        cutoff = self.cycle - self.config["lookback_window"]
        salient = [t for t in self.error_traces
                   if t.time >= cutoff and self.is_salient(t.observable)]
        stability = self.config["default_structural_stability"]
        plasticity = self.config["default_plasticity"]
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
        cost = self.config["structural_cost_link"]
        for key, link in list(self.suspicion_links.items()):
            if self.structural_budget_used >= self.config["max_structural_atoms_per_cycle"]:
                break
            pm_key = (link.cause, link.effect)
            if pm_key in self.passive_models:
                continue
            if abs(link.strength) > cost:
                ctype = "excitatory" if link.strength > 0 else "inhibitory"
                weight = min(abs(link.strength), 1.0)
                self.passive_models[pm_key] = PassiveModel(
                    cause=link.cause, effect=link.effect,
                    lag=link.lag, weight=weight, causal_type=ctype,
                    origin="empirical",
                    energy=self.config["metabolic_initial_energy"],
                    created_at=self.cycle)
                self.structural_budget_used += 1

    def metabolic_step(self):
        rate = self.config["metabolic_rate"]
        boost = self.config["metabolic_boost"]
        cap = self.config["metabolic_energy_cap"]
        gestation = self.config["gestation_period"]

        dead = []
        for key, pm in list(self.passive_models.items()):
            if self.cycle - pm.created_at < gestation:
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

        for key in dead:
            del self.passive_models[key]
            self.death_count += 1

    def run_cycle(self, new_observations=None):
        self.structural_budget_used = 0
        if new_observations:
            for obs, (val, prec) in new_observations.items():
                self.inject_observation(obs, val, prec)
        self.record_error_traces()
        self.update_suspicion_links()
        self.check_phase1()
        self.metabolic_step()
        self.update_beliefs()
        self.cycle += 1


# =============================================================================
# Test environments
# =============================================================================

class FireChainEnv:
    """4-node chain: ignition → heat → smoke → ash (each lag 1)."""
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


# =============================================================================
# Sensitivity tests
# =============================================================================

def test_induction(config, n_cycles=25):
    """
    Can the system discover causal links AND keep them alive?
    Returns: (n_links_discovered, n_links_alive_at_end, max_links_ever)
    """
    sim = CognitiveSimulation(config)
    env = FireChainEnv(period=7)

    for obs in ["ignition", "heat", "smoke", "ash"]:
        sim.set_belief(obs, 0.5, 0.5)

    max_links = 0
    for cycle in range(n_cycles):
        sim.cycle = cycle
        obs = env.get_observations(cycle)
        for name, (val, prec) in obs.items():
            sim.inject_observation(name, val, prec)
        sim.run_cycle()
        max_links = max(max_links, len(sim.passive_models))

    alive = len(sim.passive_models)
    return max_links, alive


def test_wrong_hypothesis_death(config, n_cycles=60):
    """
    Does a wrong hypothesis die?
    Seed an incorrect excitatory link where A actually inhibits C.
    Returns: (died: bool, cycles_to_death: int or None, final_energy: float)
    """
    sim = CognitiveSimulation(config)

    for obs in ["obs-A", "obs-C"]:
        sim.set_belief(obs, 0.5, 0.5)

    # Seed wrong link: claims A excites C, but A actually inhibits C
    sim.passive_models[("obs-A", "obs-C")] = PassiveModel(
        cause="obs-A", effect="obs-C", lag=0, weight=0.5,
        causal_type="excitatory", origin="empirical",
        energy=config["metabolic_initial_energy"],
        created_at=0, predictions=0, successes=0)

    death_cycle = None
    for cycle in range(n_cycles):
        sim.cycle = cycle
        # A goes high → C goes LOW (inhibitory relationship)
        a_phase = (cycle % 8) < 4
        a_val = 0.85 if a_phase else 0.15
        c_val = 0.15 if a_phase else 0.85  # Opposite of A
        sim.run_cycle({"obs-A": (a_val, 0.8), "obs-C": (c_val, 0.8)})

        if ("obs-A", "obs-C") not in sim.passive_models:
            death_cycle = cycle
            break

    if death_cycle is not None:
        return True, death_cycle, 0.0
    else:
        pm = sim.passive_models.get(("obs-A", "obs-C"))
        return False, None, pm.energy if pm else 0.0


def test_correct_hypothesis_survival(config, n_cycles=60):
    """
    Does a correct hypothesis accumulate energy?
    Seed a correct excitatory link and track energy.
    Returns: (alive: bool, final_energy: float, peak_energy: float)
    """
    sim = CognitiveSimulation(config)

    for obs in ["obs-A", "obs-B"]:
        sim.set_belief(obs, 0.5, 0.5)

    sim.passive_models[("obs-A", "obs-B")] = PassiveModel(
        cause="obs-A", effect="obs-B", lag=0, weight=0.5,
        causal_type="excitatory", origin="empirical",
        energy=config["metabolic_initial_energy"],
        created_at=0, predictions=0, successes=0)

    peak_energy = config["metabolic_initial_energy"]
    for cycle in range(n_cycles):
        sim.cycle = cycle
        # A and B co-vary (correct excitatory)
        a_phase = (cycle % 8) < 4
        a_val = 0.85 if a_phase else 0.15
        b_val = 0.80 if a_phase else 0.18  # Same direction
        sim.run_cycle({"obs-A": (a_val, 0.8), "obs-B": (b_val, 0.8)})

        pm = sim.passive_models.get(("obs-A", "obs-B"))
        if pm:
            peak_energy = max(peak_energy, pm.energy)

    pm = sim.passive_models.get(("obs-A", "obs-B"))
    if pm:
        return True, pm.energy, peak_energy
    else:
        return False, 0.0, peak_energy


# =============================================================================
# Main sweep
# =============================================================================

def run_sweep():
    # Parameter ranges
    rent_values = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    reward_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.15, 0.20]

    results = []

    print("=" * 78)
    print("METABOLIC PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 78)
    print(f"\nSweeping c_rent: {rent_values}")
    print(f"Sweeping c_reward: {reward_values}")
    print(f"Grid: {len(rent_values)} × {len(reward_values)} = "
          f"{len(rent_values) * len(reward_values)} parameter pairs\n")

    print("Running...", flush=True)

    for rent in rent_values:
        for reward in reward_values:
            config = dict(BASE_CONFIG)
            config["metabolic_rate"] = rent
            config["metabolic_boost"] = reward

            # Test 1: Induction discovery
            max_links, alive_links = test_induction(config, n_cycles=25)

            # Test 2: Wrong hypothesis death
            died, death_cycle, wrong_final_e = test_wrong_hypothesis_death(config, n_cycles=60)

            # Test 3: Correct hypothesis survival
            correct_alive, correct_final_e, correct_peak_e = test_correct_hypothesis_survival(config, n_cycles=60)

            # Health classification
            induction_ok = max_links >= 3          # Discovered at least 3 links
            retention_ok = alive_links >= 2        # Kept at least 2 alive
            death_ok = died                        # Wrong hypothesis died
            survival_ok = correct_alive            # Correct hypothesis survived
            correct_thriving = correct_final_e > config["metabolic_initial_energy"]

            healthy = induction_ok and retention_ok and death_ok and survival_ok

            ratio = reward / rent if rent > 0 else float('inf')

            results.append({
                "rent": rent,
                "reward": reward,
                "ratio": ratio,
                "max_links": max_links,
                "alive_links": alive_links,
                "wrong_died": died,
                "wrong_death_cycle": death_cycle,
                "wrong_final_energy": wrong_final_e,
                "correct_alive": correct_alive,
                "correct_final_energy": correct_final_e,
                "correct_peak_energy": correct_peak_e,
                "induction_ok": induction_ok,
                "retention_ok": retention_ok,
                "death_ok": death_ok,
                "survival_ok": survival_ok,
                "correct_thriving": correct_thriving,
                "healthy": healthy,
            })

    return results, rent_values, reward_values


def print_heatmap(results, rent_values, reward_values):
    """Print ASCII heatmaps for key metrics."""

    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r["rent"], r["reward"])] = r

    # --- Heatmap 1: Health status ---
    print("\n" + "=" * 78)
    print("HEATMAP: SYSTEM HEALTH  (✓=healthy  D=death fails  R=retention fails")
    print("                         I=induction fails  S=survival fails  ·=multiple)")
    print("=" * 78)
    header = "rent\\reward"
    print(f"  {header:>10s}", end="")
    for rew in reward_values:
        print(f" {rew:5.3f}", end="")
    print()
    print("  " + "-" * (11 + 6 * len(reward_values)))

    for rent in rent_values:
        print(f"  {rent:10.3f}", end="")
        for rew in reward_values:
            r = lookup[(rent, rew)]
            if r["healthy"]:
                ch = " ✓"
            else:
                fails = []
                if not r["induction_ok"]: fails.append("I")
                if not r["retention_ok"]: fails.append("R")
                if not r["death_ok"]: fails.append("D")
                if not r["survival_ok"]: fails.append("S")
                if len(fails) == 1:
                    ch = f" {fails[0]}"
                else:
                    ch = " ·"
            print(f"   {ch} ", end="")
        print()

    # --- Heatmap 2: Wrong hypothesis death cycle ---
    print("\n" + "=" * 78)
    print("HEATMAP: CYCLES TO DEATH OF WRONG HYPOTHESIS  (-- = survived)")
    print("=" * 78)
    print(f"  {header:>10s}", end="")
    for rew in reward_values:
        print(f" {rew:5.3f}", end="")
    print()
    print("  " + "-" * (11 + 6 * len(reward_values)))

    for rent in rent_values:
        print(f"  {rent:10.3f}", end="")
        for rew in reward_values:
            r = lookup[(rent, rew)]
            if r["wrong_died"]:
                print(f"   {r['wrong_death_cycle']:2d} ", end="")
            else:
                print(f"   -- ", end="")
        print()

    # --- Heatmap 3: Correct hypothesis final energy ---
    print("\n" + "=" * 78)
    print("HEATMAP: CORRECT HYPOTHESIS FINAL ENERGY  (0.00 = dead)")
    print("=" * 78)
    print(f"  {header:>10s}", end="")
    for rew in reward_values:
        print(f" {rew:5.3f}", end="")
    print()
    print("  " + "-" * (11 + 6 * len(reward_values)))

    for rent in rent_values:
        print(f"  {rent:10.3f}", end="")
        for rew in reward_values:
            r = lookup[(rent, rew)]
            e = r["correct_final_energy"]
            print(f" {e:5.2f}", end="")
        print()

    # --- Heatmap 4: Links alive at end of induction test ---
    print("\n" + "=" * 78)
    print("HEATMAP: CAUSAL LINKS ALIVE AFTER 25 CYCLES")
    print("=" * 78)
    print(f"  {header:>10s}", end="")
    for rew in reward_values:
        print(f" {rew:5.3f}", end="")
    print()
    print("  " + "-" * (11 + 6 * len(reward_values)))

    for rent in rent_values:
        print(f"  {rent:10.3f}", end="")
        for rew in reward_values:
            r = lookup[(rent, rew)]
            print(f"   {r['alive_links']:2d} ", end="")
        print()


def print_summary(results, rent_values, reward_values):
    """Print summary statistics."""

    healthy = [r for r in results if r["healthy"]]
    unhealthy = [r for r in results if not r["healthy"]]

    print("\n" + "=" * 78)
    print("SUMMARY STATISTICS")
    print("=" * 78)
    total = len(results)
    n_healthy = len(healthy)
    print(f"\n  Total parameter pairs tested: {total}")
    print(f"  Healthy (all criteria pass):  {n_healthy} ({100*n_healthy/total:.0f}%)")
    print(f"  Unhealthy:                    {len(unhealthy)} ({100*len(unhealthy)/total:.0f}%)")

    if healthy:
        ratios = [r["ratio"] for r in healthy]
        rents = [r["rent"] for r in healthy]
        rewards = [r["reward"] for r in healthy]
        print(f"\n  Healthy region:")
        print(f"    c_rent range:   [{min(rents):.3f}, {max(rents):.3f}]")
        print(f"    c_reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
        print(f"    ratio (reward/rent) range: [{min(ratios):.1f}, {max(ratios):.1f}]")
        print(f"    median ratio: {sorted(ratios)[len(ratios)//2]:.1f}")

    # Failure mode analysis
    fail_induction = sum(1 for r in results if not r["induction_ok"])
    fail_retention = sum(1 for r in results if not r["retention_ok"])
    fail_death = sum(1 for r in results if not r["death_ok"])
    fail_survival = sum(1 for r in results if not r["survival_ok"])

    print(f"\n  Failure modes:")
    print(f"    Induction fails (no links discovered): {fail_induction}")
    print(f"    Retention fails (links die too fast):  {fail_retention}")
    print(f"    Death fails (wrong hyp survives):      {fail_death}")
    print(f"    Survival fails (correct hyp dies):     {fail_survival}")

    # Ratio analysis
    print(f"\n  Ratio analysis (reward/rent):")
    ratio_bins = {}
    for r in results:
        ratio_bucket = round(r["ratio"], 1)
        if ratio_bucket not in ratio_bins:
            ratio_bins[ratio_bucket] = {"total": 0, "healthy": 0}
        ratio_bins[ratio_bucket]["total"] += 1
        if r["healthy"]:
            ratio_bins[ratio_bucket]["healthy"] += 1

    for ratio in sorted(ratio_bins.keys()):
        b = ratio_bins[ratio]
        pct = 100 * b["healthy"] / b["total"] if b["total"] > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"    ratio {ratio:5.1f}: {b['healthy']:2d}/{b['total']:2d} healthy "
              f"({pct:3.0f}%) {bar}")

    # Default parameters assessment
    default_result = None
    for r in results:
        if abs(r["rent"] - 0.02) < 0.001 and abs(r["reward"] - 0.05) < 0.001:
            default_result = r
            break

    if default_result:
        print(f"\n  DEFAULT PARAMETERS (rent=0.02, reward=0.05):")
        print(f"    Healthy: {'YES ✓' if default_result['healthy'] else 'NO ✗'}")
        print(f"    Links discovered/alive: {default_result['max_links']}/{default_result['alive_links']}")
        print(f"    Wrong hyp died: {'yes, cycle ' + str(default_result['wrong_death_cycle']) if default_result['wrong_died'] else 'NO'}")
        ce = default_result["correct_final_energy"]
        ca = default_result["correct_alive"]
        print(f"    Correct hyp alive: {'yes, energy=' + f'{ce:.2f}' if ca else 'NO'}")


def write_csv(results, filepath):
    """Write results to CSV for external plotting."""
    fieldnames = list(results[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV written to: {filepath}")


def main():
    results, rent_values, reward_values = run_sweep()
    print_heatmap(results, rent_values, reward_values)
    print_summary(results, rent_values, reward_values)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "metabolic_sensitivity.csv")
    write_csv(results, csv_path)

    print("\n" + "=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)
    print("""
  The healthy region is where ALL of these hold simultaneously:
    1. Induction discovers ≥3 causal links (structure learning works)
    2. At least 2 links survive to cycle 25 (correct structure retained)
    3. Wrong hypotheses die within 60 cycles (garbage collected)
    4. Correct hypotheses survive 60 cycles (useful structure persists)

  Key question: How WIDE is the healthy region?
    - Wide basin → robust system, meta-parameter bootstrap is feasible
    - Narrow basin → brittle system, manual tuning required

  The ratio (reward/rent) is the key invariant:
    - Too low → everything dies (starvation)
    - Too high → nothing dies (bloat)
    - The healthy range tells you how much slack the system has.
""")

    # Return exit code based on default params health
    for r in results:
        if abs(r["rent"] - 0.02) < 0.001 and abs(r["reward"] - 0.05) < 0.001:
            return 0 if r["healthy"] else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
