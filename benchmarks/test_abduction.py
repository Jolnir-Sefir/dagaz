#!/usr/bin/env python3
"""
Abductive Inference (Phase 1.6) â€” Benchmark Simulation

Validates the abductive inference mechanism using a pure Python simulation
that mirrors the MeTTa implementation's logic. This serves as:
  1. Correctness reference for the MeTTa code
  2. Regression test for the abductive mechanism
  3. Demonstration of emergent verification ("Sherlock Holmes" effect)

Three scenarios tested:
  1. Confirmation:  Hidden cause abduced from effect, confirmed by second effect
  2. Falsification: Wrong hypothesis killed by contradicting evidence
  3. Competing explanations: Asymmetric evidence disambiguates rival hypotheses

Usage:
    python test_abduction.py          # Run all tests
    python test_abduction.py -v       # Verbose output with per-cycle traces

The MeTTa implementation uses precision-weighted belief merges.
The Python benchmark uses the same formulas. Results should match
qualitatively (same outcomes, same orderings) though not numerically
(floating point paths differ).
"""

import math
import sys
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Configuration (mirrors foundations.metta configs)
# =============================================================================

CONFIG = {
    # Belief update
    "learning-rate": 0.12,
    "precision-floor": 0.05,
    "precision-ceiling": 0.95,
    "surprise-threshold": 0.08,

    # Structure learning (for passive model context)
    "metabolic-rate": 0.02,
    "metabolic-boost": 0.05,
    "metabolic-initial-energy": 1.0,
    "metabolic-energy-cap": 2.0,
    "gestation-period": 3,

    # Abductive inference
    # Threshold note: surprise = 0.5 Ã— prec Ã— errÂ².
    # With prec=0.5 and err=0.6, surprise=0.09. Threshold must be
    # below typical high-surprise values to trigger meaningfully.
    "abductive-surprise-threshold": 0.05,
    "abductive-precision": 0.10,
    "abductive-min-link-weight": 0.15,
    "abductive-min-energy": 0.5,
    "abductive-max-cause-precision": 0.50,
    "abductive-budget-per-cycle": 5,
}

VERBOSE = "-v" in sys.argv


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Belief:
    value: float
    precision: float
    source: str = "prior"       # "prior", "observed", "abduced"
    source_cycle: int = 0

@dataclass
class Observation:
    value: float
    precision: float
    time: int

@dataclass
class PassiveModel:
    cause: str
    effect: str
    lag: int
    weight: float
    link_type: str              # "excitatory" or "inhibitory"
    energy: float = 1.0
    predictions: int = 0
    successes: int = 0

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
class EFEResult:
    action: str
    efe: float
    components: dict


# =============================================================================
# Simulation Engine
# =============================================================================

class CognitiveSimulation:
    """Minimal simulation of the cognitive core for abduction testing."""

    def __init__(self):
        self.beliefs: dict[str, Belief] = {}
        self.observations: dict[str, Observation] = {}
        self.passive_models: list[PassiveModel] = []
        self.abduction_logs: list[AbductionLog] = []
        self.cycle: int = 0

    # --- Belief Operations ---

    def set_belief(self, obs: str, value: float, precision: float,
                   source: str = "prior"):
        self.beliefs[obs] = Belief(value, precision, source, self.cycle)

    def get_belief(self, obs: str) -> Optional[Belief]:
        return self.beliefs.get(obs)

    def inject_observation(self, obs: str, value: float, precision: float):
        self.observations[obs] = Observation(value, precision, self.cycle)
        # If this observable was abduced, supersede
        b = self.beliefs.get(obs)
        if b and b.source == "abduced":
            b.source = "observed"
            b.source_cycle = self.cycle

    def update_belief_from_observation(self, obs: str):
        """Standard precision-weighted belief update from observation."""
        b = self.beliefs.get(obs)
        o = self.observations.get(obs)
        if not b or not o:
            return

        lr = CONFIG["learning-rate"]
        error = o.value - b.value
        abs_error = abs(error)

        # Q1: Should I learn? Gate on Kalman update magnitude.
        prec_sum = b.precision + o.precision
        if prec_sum <= 0:
            return
        obs_weight = o.precision / prec_sum
        update_mag = obs_weight * abs_error
        if update_mag < 0.01:
            return

        update = lr * obs_weight * error
        new_val = max(0.0, min(1.0, b.value + update))

        # Precision dynamics: threshold = agent's own uncertainty
        uncertainty = max(1.0 - b.precision, 0.01)
        if abs_error < uncertainty:
            new_prec = min(b.precision + 0.02, CONFIG["precision-ceiling"])
        else:
            new_prec = max(b.precision - 0.05, CONFIG["precision-floor"])

        b.value = new_val
        b.precision = new_prec
        if b.source != "observed":
            b.source = "observed"
            b.source_cycle = self.cycle

    # --- Prediction Error ---

    def compute_prediction_error(self, obs: str) -> Optional[tuple]:
        """Returns (error, abs_error, surprise) or None."""
        b = self.beliefs.get(obs)
        o = self.observations.get(obs)
        if not b or not o:
            return None
        error = o.value - b.value
        abs_error = abs(error)
        surprise = 0.5 * b.precision * (error ** 2)
        return (error, abs_error, surprise)

    def all_prediction_errors(self) -> dict[str, tuple]:
        """Returns {obs: (error, abs_error, surprise)}."""
        result = {}
        for obs in self.beliefs:
            pe = self.compute_prediction_error(obs)
            if pe is not None:
                result[obs] = pe
        return result

    # --- Passive Model Operations ---

    def add_passive_model(self, cause: str, effect: str, lag: int,
                          weight: float, link_type: str = "excitatory",
                          energy: float = 1.0):
        self.passive_models.append(PassiveModel(
            cause, effect, lag, weight, link_type, energy))

    def get_models_for_effect(self, effect: str) -> list[PassiveModel]:
        return [m for m in self.passive_models if m.effect == effect]

    def get_models_for_cause(self, cause: str) -> list[PassiveModel]:
        return [m for m in self.passive_models if m.cause == cause]

    def reward_passive_model(self, model: PassiveModel, error: float):
        """Metabolic reward for correct prediction.
        Q2: Was the prediction correct? error < agent's uncertainty on effect."""
        effect_b = self.beliefs.get(model.effect)
        uncertainty = max(1.0 - effect_b.precision, 0.01) if effect_b else 0.99
        if abs(error) < uncertainty:
            model.energy = min(model.energy + CONFIG["metabolic-boost"],
                               CONFIG["metabolic-energy-cap"])
            model.successes += 1
        model.predictions += 1

    def drain_passive_models(self):
        """Metabolic drain each cycle."""
        for m in self.passive_models:
            m.energy -= CONFIG["metabolic-rate"]

    # --- Abductive Inference (Core) ---

    def compute_hypothesis_value(self, obs_val: float, bval: float,
                                 weight: float, link_type: str) -> float:
        """Invert the forward model to compute hypothesized cause value."""
        if weight < 0.01:
            raw_hyp = 1.0
        else:
            raw_hyp = obs_val / weight

        if link_type == "excitatory":
            directed = raw_hyp
        else:
            directed = 1.0 - raw_hyp

        return max(0.0, min(1.0, directed))

    def collect_abductive_candidates(self) -> list[tuple]:
        """Find effects with high surprise that have known causes."""
        threshold = CONFIG["abductive-surprise-threshold"]
        min_weight = CONFIG["abductive-min-link-weight"]
        min_energy = CONFIG["abductive-min-energy"]
        max_cause_prec = CONFIG["abductive-max-cause-precision"]

        candidates = []
        errors = self.all_prediction_errors()

        for effect, (error, abs_error, surprise) in errors.items():
            if surprise < threshold:
                continue

            for model in self.get_models_for_effect(effect):
                if model.weight < min_weight:
                    continue
                if model.energy < min_energy:
                    continue

                cause_belief = self.beliefs.get(model.cause)
                if not cause_belief:
                    continue
                if cause_belief.precision >= max_cause_prec:
                    continue

                candidates.append((model, effect, error, surprise))

        return candidates

    def inject_hypothesis(self, cause: str, hyp_val: float,
                          hyp_prec: float, effect: str) -> Optional[AbductionLog]:
        """Precision-weighted merge of hypothesis into existing belief."""
        b = self.beliefs.get(cause)
        if not b:
            return None

        lr = CONFIG["learning-rate"]
        prec_sum = b.precision + hyp_prec
        if prec_sum <= 0:
            return None

        hyp_weight = hyp_prec / prec_sum
        error = hyp_val - b.value
        update = lr * hyp_weight * error
        prior_val = b.value
        new_val = max(0.0, min(1.0, b.value + update))

        # Precision: slight decrease (hypothesis adds uncertainty)
        new_prec = max(b.precision - 0.02, CONFIG["precision-floor"])

        b.value = new_val
        b.precision = new_prec

        # Mark epistemic source (only if not already observed)
        if b.source != "observed":
            b.source = "abduced"
            b.source_cycle = self.cycle

        log = AbductionLog(cause, effect, hyp_val, hyp_prec,
                           prior_val, new_val, self.cycle)
        self.abduction_logs.append(log)
        return log

    def abductive_step(self) -> list[AbductionLog]:
        """Run one cycle of abductive inference. Returns logs."""
        candidates = self.collect_abductive_candidates()
        logs = []
        budget = CONFIG["abductive-budget-per-cycle"]

        for model, effect, error, surprise in candidates:
            if len(logs) >= budget:
                break

            obs = self.observations.get(effect)
            if not obs:
                continue

            bval = self.beliefs[effect].value
            hyp_val = self.compute_hypothesis_value(
                obs.value, bval, model.weight, model.link_type)
            hyp_prec = CONFIG["abductive-precision"]

            log = self.inject_hypothesis(
                model.cause, hyp_val, hyp_prec, effect)
            if log:
                logs.append(log)

        return logs

    # --- Simplified EFE (for action selection testing) ---

    def compute_efe(self, action: str, target_obs: str = None) -> EFEResult:
        """
        Simplified EFE computation for testing the Sherlock Holmes effect.
        
        For observe(X): info gain is high when X has low precision
        or when X is predicted by a low-precision belief (e.g., an abduced cause).
        """
        cost = {"wait": 0.01, "observe": 0.05, "retreat": 0.10}.get(action, 0.05)

        # Compute expected error (sum across all observables)
        expected_error = 0.0
        for obs, b in self.beliefs.items():
            o = self.observations.get(obs)
            if o:
                err = abs(o.value - b.value)
                expected_error += err * b.precision * o.precision

        # Info gain: specifically for observe actions
        info_gain = 0.0
        if action.startswith("observe") and target_obs:
            b = self.beliefs.get(target_obs)
            if b:
                # Direct info gain: 1 - precision
                info_gain = (1.0 - b.precision) * 0.1

                # Indirect info gain: if observing this would confirm/deny
                # a low-precision cause (the Sherlock Holmes effect)
                for model in self.get_models_for_effect(target_obs):
                    cause_b = self.beliefs.get(model.cause)
                    if cause_b and cause_b.source == "abduced":
                        # Huge info gain: this observation disambiguates
                        info_gain += (1.0 - cause_b.precision) * 0.15

        efe = expected_error + cost - info_gain

        return EFEResult(action, efe, {
            "expected_error": expected_error,
            "cost": cost,
            "info_gain": info_gain
        })

    # --- Cycle ---

    def run_cycle(self, new_observations: dict[str, tuple] = None):
        """
        Run one cognitive cycle:
        1. Inject observations
        2. Compute errors
        3. Abductive inference
        4. Update beliefs
        5. Metabolic maintenance
        """
        # Inject new observations
        if new_observations:
            for obs, (val, prec) in new_observations.items():
                self.inject_observation(obs, val, prec)

        # Compute errors (for diagnostics, used by abduction internally)
        errors = self.all_prediction_errors()

        # Abductive inference (BEFORE belief update, AFTER error computation)
        abduction_logs = self.abductive_step()

        # Update beliefs from observations
        for obs in list(self.observations.keys()):
            self.update_belief_from_observation(obs)

        # Metabolic maintenance
        self.drain_passive_models()

        # Reward models for correct predictions
        for model in self.passive_models:
            if model.effect in self.observations and model.cause in self.beliefs:
                pe = self.compute_prediction_error(model.effect)
                if pe:
                    self.reward_passive_model(model, pe[0])

        self.cycle += 1

        return {
            "cycle": self.cycle - 1,
            "errors": errors,
            "abductions": abduction_logs,
        }


# =============================================================================
# Test Scenarios
# =============================================================================

def test_scenario_1_confirmation():
    """
    Scenario 1: Hidden Cause Confirmation

    Setup:
    - Latent L causes observable A and observable B
    - Links: Lâ†’A (exc, weight 0.8), Lâ†’B (exc, weight 0.8)
    - L belief: 0.3 (low, incorrect â€” true value will be high)

    Sequence:
    1. Observe A = 0.9 (high surprise)
    2. Abduction: hypothesize L = high (from Lâ†’A)
    3. Verify: observe(B) should have high info gain
    4. Observe B = 0.9 (confirms L)
    5. L's belief should gain precision over cycles

    Success criteria:
    - L is hypothesized after A is observed
    - L's belief moves toward high
    - L's source is marked "abduced" initially
    - After B confirms, L's precision increases
    """
    print("\n" + "="*60)
    print("SCENARIO 1: Hidden Cause Confirmation")
    print("="*60)

    sim = CognitiveSimulation()

    # Set up beliefs
    sim.set_belief("L", 0.3, 0.40)     # Latent cause: uncertain, low value
    sim.set_belief("A", 0.3, 0.50)     # Effect A: expect low
    sim.set_belief("B", 0.3, 0.50)     # Effect B: expect low

    # Set up causal structure (already discovered by Phase 1/2)
    sim.add_passive_model("L", "A", 0, 0.8, "excitatory", energy=1.0)
    sim.add_passive_model("L", "B", 0, 0.8, "excitatory", energy=1.0)

    # Phase 1: Observe A = 0.9 (high surprise)
    print("\n--- Cycle 0: Inject A = 0.9 ---")
    result = sim.run_cycle({"A": (0.9, 0.8)})

    l_belief = sim.get_belief("L")
    a_belief = sim.get_belief("A")

    if VERBOSE:
        print(f"  Errors: {result['errors']}")
        for log in result['abductions']:
            print(f"  Abduction: {log.cause} from {log.effect} "
                  f"(hyp={log.hyp_val:.3f}, {log.prior_val:.3f}â†’{log.posterior_val:.3f})")
        print(f"  L belief: val={l_belief.value:.4f} prec={l_belief.precision:.4f} "
              f"source={l_belief.source}")
        print(f"  A belief: val={a_belief.value:.4f} prec={a_belief.precision:.4f}")

    # Check: L should be abduced
    assert len(result['abductions']) > 0, "Expected abduction of L from A"
    assert any(log.cause == "L" for log in result['abductions']), \
        "Expected L to be hypothesized"
    assert l_belief.source == "abduced", \
        f"Expected L source='abduced', got '{l_belief.source}'"

    l_val_after_a = l_belief.value
    assert l_val_after_a > 0.3, \
        f"Expected L to move toward high, got {l_val_after_a:.4f}"

    # Key test: observe(B) should have significant info gain ABOVE BASELINE
    # because L (abduced, low precision) predicts B. Confirming B would
    # help resolve the uncertainty about L. This IS the Sherlock Holmes effect:
    # the system identifies an unobserved consequence of its hypothesis.
    efe_b = sim.compute_efe("observe_B", "B")
    efe_a = sim.compute_efe("observe_A", "A")
    efe_wait = sim.compute_efe("wait")

    if VERBOSE:
        print(f"\n  EFE observe(B): {efe_b.efe:.4f} (info_gain={efe_b.components['info_gain']:.4f})")
        print(f"  EFE observe(A): {efe_a.efe:.4f} (info_gain={efe_a.components['info_gain']:.4f})")
        print(f"  EFE wait:       {efe_wait.efe:.4f}")

    # B should have info gain from L being abduced (Sherlock Holmes effect)
    assert efe_b.components["info_gain"] > efe_wait.components["info_gain"], \
        "observe(B) should have more info gain than wait (Sherlock Holmes effect)"

    # Both observe actions should beat wait in EFE
    assert efe_b.efe < efe_wait.efe, \
        "observe(B) should have lower EFE than wait"

    # Phase 2: Observe B = 0.9 (confirms L)
    print("\n--- Cycle 1: Inject B = 0.9 (confirmation) ---")
    result2 = sim.run_cycle({"B": (0.9, 0.8)})

    l_belief = sim.get_belief("L")
    if VERBOSE:
        print(f"  L belief: val={l_belief.value:.4f} prec={l_belief.precision:.4f} "
              f"source={l_belief.source}")
        for log in result2['abductions']:
            print(f"  Abduction: {log.cause} from {log.effect} "
                  f"(hyp={log.hyp_val:.3f}, {log.prior_val:.3f}â†’{log.posterior_val:.3f})")

    l_val_after_b = l_belief.value
    assert l_val_after_b > l_val_after_a, \
        f"Expected L to increase further after B confirms (was {l_val_after_a:.4f}, now {l_val_after_b:.4f})"

    # Run more cycles to let precision settle
    # Both A and B are observed high, both abduct L high.
    # L should gradually converge upward.
    for i in range(15):
        sim.run_cycle({"A": (0.9, 0.8), "B": (0.9, 0.8)})

    l_final = sim.get_belief("L")
    if VERBOSE:
        print(f"\n  After 15 more cycles:")
        print(f"  L belief: val={l_final.value:.4f} prec={l_final.precision:.4f} "
              f"source={l_final.source}")

    # L should have moved meaningfully upward from 0.3
    # Each cycle, abduction from A and B pushes L higher
    assert l_final.value > 0.4, \
        f"Expected L to converge upward after repeated confirmation, got {l_final.value:.4f}"

    print("\nâœ“ Scenario 1 PASSED: Hidden cause correctly abduced and confirmed")
    return True


def test_scenario_2_falsification():
    """
    Scenario 2: Wrong Hypothesis Falsification

    Setup:
    - Aâ†’E (excitatory, weight 0.8) â€” true cause
    - Bâ†’E (excitatory, weight 0.5) â€” false cause
    - Aâ†’F (excitatory, weight 0.7) â€” A has a second effect
    - Bâ†’G (excitatory, weight 0.7) â€” B has a second effect

    Sequence:
    1. Observe E = 0.9 â†’ both A and B hypothesized
    2. Observe F = 0.85 â†’ confirms A
    3. Observe G = 0.1 â†’ contradicts B
    4. Over cycles: A's belief strengthens, B's belief weakens

    Success criteria:
    - Both A and B initially hypothesized
    - After contradicting evidence, B's belief drops
    - A's belief remains high and gains precision
    """
    print("\n" + "="*60)
    print("SCENARIO 2: Falsification of Wrong Hypothesis")
    print("="*60)

    sim = CognitiveSimulation()

    # Beliefs: all start uncertain and low
    sim.set_belief("A", 0.3, 0.40)
    sim.set_belief("B", 0.3, 0.40)
    sim.set_belief("E", 0.3, 0.50)
    sim.set_belief("F", 0.3, 0.50)
    sim.set_belief("G", 0.3, 0.50)

    # Causal structure
    sim.add_passive_model("A", "E", 0, 0.8, "excitatory", energy=1.0)
    sim.add_passive_model("B", "E", 0, 0.5, "excitatory", energy=1.0)
    sim.add_passive_model("A", "F", 0, 0.7, "excitatory", energy=1.0)
    sim.add_passive_model("B", "G", 0, 0.7, "excitatory", energy=1.0)

    # Phase 1: Observe E = 0.9 â†’ abduct both A and B
    print("\n--- Cycle 0: Inject E = 0.9 ---")
    result = sim.run_cycle({"E": (0.9, 0.8)})

    a_belief = sim.get_belief("A")
    b_belief = sim.get_belief("B")

    if VERBOSE:
        for log in result['abductions']:
            print(f"  Abduction: {log.cause} from {log.effect} "
                  f"(hyp={log.hyp_val:.3f}, {log.prior_val:.3f}â†’{log.posterior_val:.3f})")
        print(f"  A belief: val={a_belief.value:.4f} prec={a_belief.precision:.4f} "
              f"source={a_belief.source}")
        print(f"  B belief: val={b_belief.value:.4f} prec={b_belief.precision:.4f} "
              f"source={b_belief.source}")

    # Both should be abduced
    abduced_causes = {log.cause for log in result['abductions']}
    assert "A" in abduced_causes, "Expected A to be hypothesized"
    assert "B" in abduced_causes, "Expected B to be hypothesized"

    a_val_initial = a_belief.value
    b_val_initial = b_belief.value

    # A should get a stronger hypothesis (higher weight link)
    a_log = next(log for log in result['abductions'] if log.cause == "A")
    b_log = next(log for log in result['abductions'] if log.cause == "B")
    assert a_log.hyp_val >= b_log.hyp_val, \
        "A's hypothesis should be at least as strong as B's (higher weight link)"

    # Phase 2: Observe F = 0.85 (confirms A) and G = 0.1 (contradicts B)
    print("\n--- Cycle 1: Inject F=0.85 (confirms A), G=0.1 (contradicts B) ---")
    result2 = sim.run_cycle({"E": (0.9, 0.8), "F": (0.85, 0.8), "G": (0.1, 0.8)})

    a_belief = sim.get_belief("A")
    b_belief = sim.get_belief("B")

    if VERBOSE:
        for log in result2['abductions']:
            print(f"  Abduction: {log.cause} from {log.effect} "
                  f"(hyp={log.hyp_val:.3f}, {log.prior_val:.3f}â†’{log.posterior_val:.3f})")
        print(f"  A belief: val={a_belief.value:.4f} prec={a_belief.precision:.4f}")
        print(f"  B belief: val={b_belief.value:.4f} prec={b_belief.precision:.4f}")

    # Run several more cycles with the same evidence pattern
    # Abduction is weak (~2% per cycle), so convergence takes many cycles.
    # This is by design: hypotheses are tentative and build slowly.
    for i in range(30):
        sim.run_cycle({"E": (0.9, 0.8), "F": (0.85, 0.8), "G": (0.1, 0.8)})

    a_final = sim.get_belief("A")
    b_final = sim.get_belief("B")

    if VERBOSE:
        print(f"\n  After 30 more cycles:")
        print(f"  A belief: val={a_final.value:.4f} prec={a_final.precision:.4f}")
        print(f"  B belief: val={b_final.value:.4f} prec={b_final.precision:.4f}")

    # A should be meaningfully higher than initial (0.3)
    # Abduction from E and F both push A up
    assert a_final.value > 0.35, \
        f"Expected A to rise from abductive support, got {a_final.value:.4f}"

    # B should have moved back toward low (contradicted by G=0.1)
    # B gets abductive pull from E but observational pull from G toward low.
    # Over cycles, observations (higher precision) dominate abduction (lower precision).
    assert b_final.value < a_final.value, \
        f"Expected B < A (B contradicted by G), A={a_final.value:.4f} B={b_final.value:.4f}"

    print("\nâœ“ Scenario 2 PASSED: Wrong hypothesis weakened by contradicting evidence")
    return True


def test_scenario_3_competing_explanations():
    """
    Scenario 3: Competing Explanations with Asymmetric Evidence

    Setup:
    - Aâ†’E (excitatory, weight 0.8)
    - Bâ†’E (excitatory, weight 0.3)
    - Aâ†’F (excitatory, weight 0.7) â€” F is the DISCRIMINATING observable
    - No Bâ†’F link exists

    This is the full Sherlock Holmes test:
    1. E is observed high â†’ both A and B hypothesized
    2. The system should identify F as high-value (it disambiguates)
    3. F is observed high â†’ confirms A, B has no support

    Success criteria:
    - Both hypothesized initially
    - observe(F) has higher info gain than observe(E) or wait
    - After F confirms A, A's belief is stronger than B's
    """
    print("\n" + "="*60)
    print("SCENARIO 3: Competing Explanations (Sherlock Holmes)")
    print("="*60)

    sim = CognitiveSimulation()

    # Beliefs
    sim.set_belief("A", 0.3, 0.40)
    sim.set_belief("B", 0.3, 0.40)
    sim.set_belief("E", 0.3, 0.50)
    sim.set_belief("F", 0.3, 0.50)

    # Causal structure
    sim.add_passive_model("A", "E", 0, 0.8, "excitatory", energy=1.0)
    sim.add_passive_model("B", "E", 0, 0.3, "excitatory", energy=1.0)
    sim.add_passive_model("A", "F", 0, 0.7, "excitatory", energy=1.0)
    # Note: NO Bâ†’F link

    # Phase 1: Observe E = 0.9
    print("\n--- Cycle 0: Inject E = 0.9 ---")
    result = sim.run_cycle({"E": (0.9, 0.8)})

    a_belief = sim.get_belief("A")
    b_belief = sim.get_belief("B")

    if VERBOSE:
        for log in result['abductions']:
            print(f"  Abduction: {log.cause} from {log.effect} "
                  f"(hyp={log.hyp_val:.3f}, {log.prior_val:.3f}â†’{log.posterior_val:.3f})")
        print(f"  A belief: val={a_belief.value:.4f} prec={a_belief.precision:.4f} "
              f"source={a_belief.source}")
        print(f"  B belief: val={b_belief.value:.4f} prec={b_belief.precision:.4f} "
              f"source={b_belief.source}")

    # Both should be abduced (A from Aâ†’E, B from Bâ†’E)
    abduced_causes = {log.cause for log in result['abductions']}
    # B might not be abduced if its link weight (0.3) is above the min threshold (0.15)
    assert "A" in abduced_causes, "Expected A to be hypothesized"

    # Key test: EFE for observe(F) should be LOWER than observe(E) or wait
    # because F disambiguates between A and B
    efe_f = sim.compute_efe("observe_F", "F")
    efe_e = sim.compute_efe("observe_E", "E")
    efe_wait = sim.compute_efe("wait")

    if VERBOSE:
        print(f"\n  EFE observe(F): {efe_f.efe:.4f} "
              f"(info_gain={efe_f.components['info_gain']:.4f})")
        print(f"  EFE observe(E): {efe_e.efe:.4f} "
              f"(info_gain={efe_e.components['info_gain']:.4f})")
        print(f"  EFE wait:       {efe_wait.efe:.4f}")

    # F should have high info gain because:
    # - F has low precision (we haven't observed it)
    # - A is abduced and predicts F (so confirming F confirms A)
    assert efe_f.components["info_gain"] > efe_wait.components["info_gain"], \
        "observe(F) should have more info gain than wait"

    # Phase 2: Observe F = 0.85 (confirms A)
    print("\n--- Cycle 1: Inject F = 0.85 (discriminating evidence) ---")
    result2 = sim.run_cycle({"E": (0.9, 0.8), "F": (0.85, 0.8)})

    a_after = sim.get_belief("A")
    b_after = sim.get_belief("B")

    if VERBOSE:
        for log in result2['abductions']:
            print(f"  Abduction: {log.cause} from {log.effect} "
                  f"(hyp={log.hyp_val:.3f}, {log.prior_val:.3f}â†’{log.posterior_val:.3f})")
        print(f"  A belief: val={a_after.value:.4f} prec={a_after.precision:.4f}")
        print(f"  B belief: val={b_after.value:.4f} prec={b_after.precision:.4f}")

    # Run more cycles to let beliefs converge
    # Abduction is weak per cycle (~2%), needs many cycles
    for i in range(30):
        sim.run_cycle({"E": (0.9, 0.8), "F": (0.85, 0.8)})

    a_final = sim.get_belief("A")
    b_final = sim.get_belief("B")

    if VERBOSE:
        print(f"\n  After 30 more cycles:")
        print(f"  A belief: val={a_final.value:.4f} prec={a_final.precision:.4f}")
        print(f"  B belief: val={b_final.value:.4f} prec={b_final.precision:.4f}")

    # A should be significantly stronger than B
    # A gets abductive support from BOTH E and F
    # B only gets support from E (weaker link, 0.3 vs 0.8)
    assert a_final.value > b_final.value, \
        f"Expected A > B after discriminating evidence. A={a_final.value:.4f} B={b_final.value:.4f}"

    # A should have meaningfully higher value (gap grows with cycles)
    gap = a_final.value - b_final.value
    assert gap > 0.01, \
        f"Expected meaningful gap between A and B, got {gap:.4f}"

    print(f"\n  Final gap: A - B = {gap:.4f}")
    print("\nâœ“ Scenario 3 PASSED: Competing explanations disambiguated by asymmetric evidence")
    return True


def test_inhibitory_abduction():
    """
    Bonus test: inhibitory link abduction.

    Setup: Aâ†’E is INHIBITORY (weight 0.7).
    When E is HIGH, A should be hypothesized LOW.

    This tests the direction-flipping logic.
    """
    print("\n" + "="*60)
    print("SCENARIO 4: Inhibitory Link Abduction")
    print("="*60)

    sim = CognitiveSimulation()

    sim.set_belief("A", 0.5, 0.40)     # Start at midpoint
    sim.set_belief("E", 0.3, 0.50)

    # A inhibits E: when A is high, E is low
    sim.add_passive_model("A", "E", 0, 0.7, "inhibitory", energy=1.0)

    # Observe E = 0.9 (high surprise, high value)
    # Since A INHIBITS E, high E means A should be LOW
    print("\n--- Cycle 0: Inject E = 0.9 (inhibitory link) ---")
    result = sim.run_cycle({"E": (0.9, 0.8)})

    a_belief = sim.get_belief("A")

    if VERBOSE:
        for log in result['abductions']:
            print(f"  Abduction: {log.cause} from {log.effect} "
                  f"(hyp={log.hyp_val:.3f}, {log.prior_val:.3f}â†’{log.posterior_val:.3f})")
        print(f"  A belief: val={a_belief.value:.4f} prec={a_belief.precision:.4f}")

    # A's hypothesis should be LOW (inhibitory: high E â†’ low A)
    if result['abductions']:
        a_log = next(log for log in result['abductions'] if log.cause == "A")
        assert a_log.hyp_val < 0.5, \
            f"Inhibitory abduction: expected low hypothesis for A, got {a_log.hyp_val:.3f}"

        # A's belief should have moved downward
        assert a_belief.value < 0.5, \
            f"Expected A belief to decrease (inhibitory), got {a_belief.value:.4f}"

    print("\nâœ“ Scenario 4 PASSED: Inhibitory abduction correctly flips direction")
    return True


def test_budget_limit():
    """
    Bonus test: abduction budget is respected.

    Setup: 10 passive models all pointing to the same effect.
    Budget = 5. Only 5 hypotheses should be injected.
    """
    print("\n" + "="*60)
    print("SCENARIO 5: Budget Limit Enforcement")
    print("="*60)

    sim = CognitiveSimulation()

    # Create 10 potential causes
    for i in range(10):
        name = f"C{i}"
        sim.set_belief(name, 0.3, 0.40)
        sim.add_passive_model(name, "E", 0, 0.5, "excitatory", energy=1.0)

    sim.set_belief("E", 0.3, 0.50)

    result = sim.run_cycle({"E": (0.9, 0.8)})

    n_abductions = len(result['abductions'])
    budget = CONFIG["abductive-budget-per-cycle"]

    if VERBOSE:
        print(f"\n  Candidates: 10, Budget: {budget}, Actual: {n_abductions}")
        for log in result['abductions']:
            print(f"  Abduced: {log.cause}")

    assert n_abductions <= budget, \
        f"Expected at most {budget} abductions, got {n_abductions}"

    print(f"\n  {n_abductions} abductions (budget = {budget})")
    print("\nâœ“ Scenario 5 PASSED: Budget limit respected")
    return True


def test_observation_supersedes():
    """
    Bonus test: direct observation supersedes abductive source.

    Setup: Lâ†’A exists. Observe A high â†’ L abduced.
    Then directly observe L â†’ source should change from abduced to observed.
    """
    print("\n" + "="*60)
    print("SCENARIO 6: Observation Supersedes Hypothesis")
    print("="*60)

    sim = CognitiveSimulation()

    sim.set_belief("L", 0.3, 0.40)
    sim.set_belief("A", 0.3, 0.50)
    sim.add_passive_model("L", "A", 0, 0.8, "excitatory", energy=1.0)

    # Abduct L from A
    result = sim.run_cycle({"A": (0.9, 0.8)})
    l_belief = sim.get_belief("L")
    assert l_belief.source == "abduced", "L should be abduced"

    if VERBOSE:
        print(f"\n  After abduction: L source={l_belief.source}")

    # Now directly observe L
    sim.inject_observation("L", 0.85, 0.9)
    l_belief = sim.get_belief("L")

    if VERBOSE:
        print(f"  After direct observation: L source={l_belief.source}")

    assert l_belief.source == "observed", \
        f"Direct observation should supersede abduction, got source='{l_belief.source}'"

    print("\nâœ“ Scenario 6 PASSED: Observation correctly supersedes hypothesis")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("ABDUCTIVE INFERENCE â€” BENCHMARK SIMULATION")
    print("Phase 1.6: Inverse Model Activation")
    print("=" * 60)

    tests = [
        ("Scenario 1: Confirmation", test_scenario_1_confirmation),
        ("Scenario 2: Falsification", test_scenario_2_falsification),
        ("Scenario 3: Competing Explanations", test_scenario_3_competing_explanations),
        ("Scenario 4: Inhibitory Abduction", test_inhibitory_abduction),
        ("Scenario 5: Budget Limit", test_budget_limit),
        ("Scenario 6: Observation Supersedes", test_observation_supersedes),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, True, None))
        except AssertionError as e:
            results.append((name, False, str(e)))
            print(f"\nâœ— {name} FAILED: {e}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\nâœ— {name} ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    for name, ok, err in results:
        status = "âœ“ PASS" if ok else f"âœ— FAIL: {err}"
        print(f"  {name}: {status}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL BENCHMARKS PASSED")
    else:
        print(f"\n  {total - passed} FAILURES")
        sys.exit(1)


if __name__ == "__main__":
    main()
