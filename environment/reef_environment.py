#!/usr/bin/env python3
"""
Coral Reef Environment Simulator
=================================

Implements the hidden causal structure for the reef monitoring scenario.
The agent never sees this code — it only receives observations.

Causal chains:
  1. Eutrophication:  nutrient-runoff → nutrient-load → algae → O2↓ → coral↓ → fish↓
  2. Thermal stress:  temperature↑ → coral↓ → algae↑ (opportunistic)
  3. Storm:           current↑ → turbidity↑ + light↓ + sensor-damage + power-drain
  4. Disease:         pathogen → coral↓↓ + fish↓ + O2↓ (hidden cause)

Phases:
  0-15:  Baseline (normal reef)
  16-30: Eutrophication cascade
  31-40: Storm event
  41-55: Disease outbreak (hidden cause)
  56-70: Recovery + reporting window

Usage:
    from reef_environment import ReefEnvironment
    env = ReefEnvironment()
    obs = env.step(cycle, agent_action)

Author: Project Dagaz
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class ReefState:
    """True state of the reef (agent cannot see this directly)."""

    # Water chemistry
    water_temperature: float = 0.45
    water_ph: float = 0.55
    dissolved_oxygen: float = 0.75
    turbidity: float = 0.12
    salinity: float = 0.55
    nutrient_load: float = 0.18

    # Biology
    coral_health: float = 0.85
    fish_activity: float = 0.72
    algae_coverage: float = 0.12

    # Physical
    current_strength: float = 0.28
    light_level: float = 0.68

    # Platform
    equipment_power: float = 0.90
    sensor_integrity: float = 0.92
    comm_quality: float = 0.75

    # Hidden variables (agent must abduce these)
    bacterial_pathogen: float = 0.0    # 0=absent, 1=epidemic
    nutrient_runoff: float = 0.0       # 0=none, 1=heavy


class ReefEnvironment:
    """
    Simulates a coral reef with embedded causal structure.

    The agent receives noisy observations of the true state.
    Hidden variables (pathogen, runoff) are not directly observable.
    Causal chains propagate effects with temporal lags.
    """

    def __init__(self, seed=42):
        self.state = ReefState()
        self.rng = random.Random(seed)
        self.cycle = 0
        self.history = []

        # Lag buffers for temporal causation
        self.nutrient_buffer = []   # nutrient_load history for lagged effects
        self.algae_buffer = []      # algae history for lagged O2 effect
        self.oxygen_buffer = []     # O2 history for lagged coral effect
        self.coral_buffer = []      # coral history for lagged fish effect
        self.temp_buffer = []       # temp history for thermal stress

        # Agent action effects
        self.aerator_active = False
        self.sensors_retracted = False

    def clamp(self, x, lo=0.0, hi=1.0):
        return max(lo, min(hi, x))

    def noise(self, scale=0.02):
        """Small Gaussian noise for environmental variability."""
        return self.rng.gauss(0, scale)

    def obs_noise(self, true_val, obs_precision):
        """Noisy observation: higher precision = less noise."""
        noise_scale = 0.05 / max(obs_precision, 0.1)
        return self.clamp(true_val + self.rng.gauss(0, noise_scale))

    # =========================================================================
    # Phase control: what external forcings are active
    # =========================================================================

    def get_phase(self, cycle):
        if cycle <= 15:
            return "baseline"
        elif cycle <= 30:
            return "eutrophication"
        elif cycle <= 40:
            return "storm"
        elif cycle <= 55:
            return "disease"
        else:
            return "recovery"

    def apply_external_forcings(self, cycle):
        """Apply scenario-driven changes to hidden variables."""
        phase = self.get_phase(cycle)
        s = self.state

        if phase == "baseline":
            # Normal conditions, slight variation
            s.nutrient_runoff = 0.05 + self.noise(0.02)
            s.bacterial_pathogen = 0.0
            s.water_temperature = 0.45 + 0.03 * math.sin(cycle / 5) + self.noise()
            s.current_strength = 0.28 + 0.05 * math.sin(cycle / 3) + self.noise()

        elif phase == "eutrophication":
            # Nutrient runoff ramps up
            progress = (cycle - 16) / 14  # 0 to 1 over the phase
            s.nutrient_runoff = 0.05 + 0.65 * progress + self.noise(0.03)
            s.bacterial_pathogen = 0.0
            s.water_temperature = 0.45 + self.noise()
            s.current_strength = 0.30 + self.noise()

        elif phase == "storm":
            # Strong currents, turbulence
            storm_intensity = 0.5 + 0.3 * math.sin((cycle - 31) * math.pi / 10)
            s.current_strength = self.clamp(0.65 + storm_intensity * 0.3 + self.noise(0.05))
            s.nutrient_runoff = 0.30 + self.noise()  # Storm washes in some nutrients
            s.bacterial_pathogen = 0.0
            s.water_temperature = 0.42 + self.noise()  # Slightly cooler from mixing

        elif phase == "disease":
            # Pathogen appears (hidden cause!)
            pathogen_progress = (cycle - 41) / 14
            s.bacterial_pathogen = self.clamp(0.1 + 0.7 * pathogen_progress + self.noise(0.03))
            s.nutrient_runoff = 0.15 + self.noise()  # Back to low
            s.water_temperature = 0.46 + self.noise()  # Normal temp (no thermal cause)
            s.current_strength = 0.30 + self.noise()   # Normal currents

        elif phase == "recovery":
            # Everything subsides
            s.bacterial_pathogen = self.clamp(s.bacterial_pathogen * 0.85 + self.noise(0.02))
            s.nutrient_runoff = self.clamp(0.08 + self.noise(0.02))
            s.water_temperature = 0.45 + self.noise()
            s.current_strength = 0.28 + self.noise()

    # =========================================================================
    # Causal propagation: hidden structure the agent must discover
    # =========================================================================

    def propagate_causation(self, cycle):
        """
        Apply causal chains with temporal lags.
        This is the TRUE generative model — the agent doesn't have it.
        """
        s = self.state

        # --- Chain 1: Eutrophication (each step = ~1-2 cycle lag) ---

        # nutrient_runoff → nutrient_load (lag 1)
        if len(self.nutrient_buffer) >= 1:
            runoff_lagged = self.nutrient_buffer[-1]
            s.nutrient_load = self.clamp(
                s.nutrient_load * 0.7 + runoff_lagged * 0.3 + self.noise())

        # nutrient_load → algae_coverage (lag 1)
        if len(self.algae_buffer) >= 1:
            nutrient_lagged = self.nutrient_buffer[-1] if self.nutrient_buffer else 0.2
            algae_growth = 0.15 * max(0, nutrient_lagged - 0.25)  # Grows above threshold
            s.algae_coverage = self.clamp(
                s.algae_coverage * 0.85 + algae_growth + self.noise())

        # algae_coverage → dissolved_oxygen (lag 1, NEGATIVE)
        if len(self.algae_buffer) >= 1:
            algae_lagged = self.algae_buffer[-1]
            o2_drain = -0.12 * max(0, algae_lagged - 0.20)  # Algae consumes O2
            o2_natural = 0.03 * (0.75 - s.dissolved_oxygen)   # Mean reversion
            aerator_boost = 0.06 if self.aerator_active else 0.0
            s.dissolved_oxygen = self.clamp(
                s.dissolved_oxygen + o2_drain + o2_natural + aerator_boost + self.noise())

        # dissolved_oxygen → coral_health (lag 1, low O2 hurts coral)
        if len(self.oxygen_buffer) >= 1:
            o2_lagged = self.oxygen_buffer[-1]
            o2_stress = -0.10 * max(0, 0.40 - o2_lagged)   # Stress below 0.4
            coral_recovery = 0.02 * (0.85 - s.coral_health)  # Slow mean reversion
            s.coral_health = self.clamp(
                s.coral_health + o2_stress + coral_recovery + self.noise())

        # --- Chain 2: Thermal stress ---
        if len(self.temp_buffer) >= 1:
            temp_lagged = self.temp_buffer[-1]
            if temp_lagged > 0.65:  # Bleaching threshold
                thermal_damage = -0.08 * (temp_lagged - 0.65)
                s.coral_health = self.clamp(s.coral_health + thermal_damage)

        # --- Chain 3: Storm effects (simultaneous) ---
        # current → turbidity (simultaneous, positive)
        s.turbidity = self.clamp(
            0.08 + 0.6 * max(0, s.current_strength - 0.3) + self.noise())

        # turbidity → light (simultaneous, negative)
        s.light_level = self.clamp(
            0.75 - 0.5 * s.turbidity + self.noise())

        # current → sensor damage (if strong)
        if s.current_strength > 0.65 and not self.sensors_retracted:
            s.sensor_integrity = self.clamp(
                s.sensor_integrity - 0.04 * (s.current_strength - 0.65))

        # current → power drain (fighting currents)
        if s.current_strength > 0.60:
            s.equipment_power = self.clamp(
                s.equipment_power - 0.02 * (s.current_strength - 0.60))

        # --- Chain 4: Disease (hidden cause) ---
        if s.bacterial_pathogen > 0.15:
            # Pathogen directly damages coral (strong negative)
            pathogen_damage = -0.15 * s.bacterial_pathogen
            s.coral_health = self.clamp(s.coral_health + pathogen_damage + self.noise())

            # Fish flee damaged reef
            if s.coral_health < 0.5:
                s.fish_activity = self.clamp(
                    s.fish_activity * 0.92 + self.noise())
            
            # Dead coral stops producing O2 (secondary effect)
            if s.coral_health < 0.4:
                s.dissolved_oxygen = self.clamp(
                    s.dissolved_oxygen - 0.03 + self.noise())

        # --- coral_health → fish_activity (general relationship) ---
        if len(self.coral_buffer) >= 1:
            coral_lagged = self.coral_buffer[-1]
            fish_attraction = 0.08 * (coral_lagged - 0.5)  # Fish like healthy coral
            fish_reversion = 0.03 * (0.72 - s.fish_activity)
            s.fish_activity = self.clamp(
                s.fish_activity + fish_attraction + fish_reversion + self.noise())

        # --- Platform natural dynamics ---
        # Slow power drain (solar recharge during day, drain at night)
        day_cycle = 0.5 + 0.5 * math.sin(cycle * math.pi / 6)
        s.equipment_power = self.clamp(
            s.equipment_power - 0.003 + 0.004 * day_cycle + self.noise(0.005))

        # Sensor drift
        if not self.sensors_retracted:
            s.sensor_integrity = self.clamp(
                s.sensor_integrity - 0.002 + self.noise(0.003))

        # Comm quality varies
        s.comm_quality = self.clamp(
            0.70 + 0.10 * math.sin(cycle / 4) + self.noise(0.03))

        # Water chemistry mean reversion
        s.water_ph = self.clamp(
            s.water_ph + 0.02 * (0.55 - s.water_ph) + self.noise())
        s.salinity = self.clamp(
            s.salinity + 0.02 * (0.55 - s.salinity) + self.noise(0.01))

    # =========================================================================
    # Agent action effects
    # =========================================================================

    def apply_action(self, action):
        """Apply the agent's chosen action to the environment."""
        s = self.state

        self.aerator_active = False
        self.sensors_retracted = False

        if action == "wait":
            pass  # Just existing

        elif action == "observe-water":
            s.equipment_power = self.clamp(s.equipment_power - 0.01)

        elif action == "observe-biology":
            s.equipment_power = self.clamp(s.equipment_power - 0.015)

        elif action == "sample-water":
            s.equipment_power = self.clamp(s.equipment_power - 0.03)

        elif action == "activate-aerator":
            self.aerator_active = True
            s.equipment_power = self.clamp(s.equipment_power - 0.04)

        elif action == "retract-sensors":
            self.sensors_retracted = True
            s.equipment_power = self.clamp(s.equipment_power - 0.02)
            s.sensor_integrity = self.clamp(s.sensor_integrity + 0.03)

        elif action == "report-to-base":
            s.equipment_power = self.clamp(s.equipment_power - 0.015)
            s.comm_quality = self.clamp(s.comm_quality + 0.03)

        elif action == "request-guidance":
            s.equipment_power = self.clamp(s.equipment_power - 0.02)

    # =========================================================================
    # Observation generation
    # =========================================================================

    def generate_observations(self, action):
        """
        Generate noisy observations based on true state and action.

        Some actions improve observation precision (lower noise).
        Retracted sensors produce no ecosystem readings.
        sample-water reveals nutrient-load with high precision.
        """
        s = self.state
        obs = {}

        # Base precision depends on sensor integrity
        base_prec = 0.4 + 0.4 * s.sensor_integrity

        if self.sensors_retracted:
            # Can still see platform state, not environment
            obs["equipment-power"] = (self.obs_noise(s.equipment_power, 0.8), 0.8)
            obs["sensor-integrity"] = (self.obs_noise(s.sensor_integrity, 0.8), 0.8)
            obs["comm-quality"] = (self.obs_noise(s.comm_quality, 0.6), 0.6)
            obs["current-strength"] = (self.obs_noise(s.current_strength, 0.3), 0.3)
            return obs

        # Always available (platform diagnostics)
        obs["equipment-power"] = (self.obs_noise(s.equipment_power, 0.8), 0.8)
        obs["sensor-integrity"] = (self.obs_noise(s.sensor_integrity, 0.7), 0.7)
        obs["comm-quality"] = (self.obs_noise(s.comm_quality, 0.6), 0.6)

        # Always available (basic sensors)
        obs["current-strength"] = (self.obs_noise(s.current_strength, base_prec * 0.8), base_prec * 0.8)
        obs["light-level"] = (self.obs_noise(s.light_level, base_prec * 0.7), base_prec * 0.7)

        # Water chemistry — enhanced by observe-water and sample-water
        water_boost = 1.0
        if action == "observe-water":
            water_boost = 1.4
        elif action == "sample-water":
            water_boost = 1.6

        obs["water-temperature"] = (
            self.obs_noise(s.water_temperature, base_prec * 0.9 * water_boost),
            base_prec * 0.9 * water_boost)
        obs["water-ph"] = (
            self.obs_noise(s.water_ph, base_prec * 0.8 * water_boost),
            base_prec * 0.8 * water_boost)
        obs["dissolved-oxygen"] = (
            self.obs_noise(s.dissolved_oxygen, base_prec * 0.9 * water_boost),
            base_prec * 0.9 * water_boost)
        obs["turbidity"] = (
            self.obs_noise(s.turbidity, base_prec * 0.8 * water_boost),
            base_prec * 0.8 * water_boost)
        obs["salinity"] = (
            self.obs_noise(s.salinity, base_prec * 0.7 * water_boost),
            base_prec * 0.7 * water_boost)

        # Nutrient load — only visible with sample-water (high precision)
        # or vaguely through observe-water (low precision)
        if action == "sample-water":
            obs["nutrient-load"] = (self.obs_noise(s.nutrient_load, 0.75), 0.75)
        elif action == "observe-water":
            obs["nutrient-load"] = (self.obs_noise(s.nutrient_load, 0.25), 0.25)

        # Biology — enhanced by observe-biology
        bio_boost = 1.4 if action == "observe-biology" else 1.0
        obs["coral-health"] = (
            self.obs_noise(s.coral_health, base_prec * 0.8 * bio_boost),
            base_prec * 0.8 * bio_boost)
        obs["fish-activity"] = (
            self.obs_noise(s.fish_activity, base_prec * 0.7 * bio_boost),
            base_prec * 0.7 * bio_boost)
        obs["algae-coverage"] = (
            self.obs_noise(s.algae_coverage, base_prec * 0.7 * bio_boost),
            base_prec * 0.7 * bio_boost)

        return obs

    # =========================================================================
    # Main step
    # =========================================================================

    def step(self, action="wait"):
        """
        Advance one cycle:
          1. Apply agent action
          2. Apply external forcings (phase-dependent)
          3. Propagate causal chains
          4. Update lag buffers
          5. Generate observations

        Returns: dict of {observable: (value, precision)}
        """
        # 1. Agent action
        self.apply_action(action)

        # 2. External forcings
        self.apply_external_forcings(self.cycle)

        # 3. Causal propagation
        self.propagate_causation(self.cycle)

        # 4. Update lag buffers
        s = self.state
        self.nutrient_buffer.append(s.nutrient_load)
        self.algae_buffer.append(s.algae_coverage)
        self.oxygen_buffer.append(s.dissolved_oxygen)
        self.coral_buffer.append(s.coral_health)
        self.temp_buffer.append(s.water_temperature)

        # Keep buffers bounded
        max_lag = 6
        for buf in [self.nutrient_buffer, self.algae_buffer,
                    self.oxygen_buffer, self.coral_buffer, self.temp_buffer]:
            while len(buf) > max_lag:
                buf.pop(0)

        # 5. Generate observations
        obs = self.generate_observations(action)

        # Record for analysis
        self.history.append({
            "cycle": self.cycle,
            "phase": self.get_phase(self.cycle),
            "action": action,
            "true_state": {
                "water_temperature": s.water_temperature,
                "dissolved_oxygen": s.dissolved_oxygen,
                "nutrient_load": s.nutrient_load,
                "coral_health": s.coral_health,
                "fish_activity": s.fish_activity,
                "algae_coverage": s.algae_coverage,
                "current_strength": s.current_strength,
                "bacterial_pathogen": s.bacterial_pathogen,
                "nutrient_runoff": s.nutrient_runoff,
                "equipment_power": s.equipment_power,
                "sensor_integrity": s.sensor_integrity,
            },
            "observations": {k: v[0] for k, v in obs.items()},
        })

        self.cycle += 1
        return obs

    def get_true_state_summary(self):
        """For debugging/analysis: print true state."""
        s = self.state
        return (
            f"Phase: {self.get_phase(self.cycle)}  "
            f"T={s.water_temperature:.2f} O2={s.dissolved_oxygen:.2f} "
            f"Nutr={s.nutrient_load:.2f} Algae={s.algae_coverage:.2f} "
            f"Coral={s.coral_health:.2f} Fish={s.fish_activity:.2f} "
            f"Path={s.bacterial_pathogen:.2f} "
            f"Pwr={s.equipment_power:.2f} Sens={s.sensor_integrity:.2f}"
        )


# =============================================================================
# Quick visualization: run the environment and show what happens
# =============================================================================

def main():
    print("=" * 80)
    print("CORAL REEF ENVIRONMENT — Causal Structure Demonstration")
    print("=" * 80)
    print()
    print("Running 70 cycles with 'wait' action to show causal dynamics.")
    print("The agent would normally choose actions; here we show the")
    print("environment's own dynamics that the agent must discover.")
    print()

    env = ReefEnvironment(seed=42)

    # Column headers
    print(f"{'Cyc':>3} {'Phase':<14} "
          f"{'O2':>5} {'Coral':>5} {'Fish':>5} {'Algae':>5} "
          f"{'Nutr':>5} {'Turb':>5} {'Curr':>5} "
          f"{'Path':>5} {'Runoff':>6} "
          f"{'Power':>5} {'Sens':>5}")
    print("-" * 95)

    for cycle in range(70):
        obs = env.step("wait")
        s = env.state
        phase = env.get_phase(cycle)

        # Color-code phases
        marker = ""
        if phase == "eutrophication" and cycle == 16:
            marker = " ← NUTRIENTS BEGIN"
        elif phase == "storm" and cycle == 31:
            marker = " ← STORM HITS"
        elif phase == "disease" and cycle == 41:
            marker = " ← PATHOGEN APPEARS"
        elif phase == "recovery" and cycle == 56:
            marker = " ← RECOVERY"

        print(f"{cycle:3d} {phase:<14} "
              f"{s.dissolved_oxygen:5.2f} {s.coral_health:5.2f} "
              f"{s.fish_activity:5.2f} {s.algae_coverage:5.2f} "
              f"{s.nutrient_load:5.2f} {s.turbidity:5.2f} "
              f"{s.current_strength:5.2f} "
              f"{s.bacterial_pathogen:5.2f} {s.nutrient_runoff:6.3f} "
              f"{s.equipment_power:5.2f} {s.sensor_integrity:5.2f}"
              f"{marker}")

    # Summary of causal events
    print("\n" + "=" * 80)
    print("CAUSAL CHAIN SUMMARY")
    print("=" * 80)
    print("""
  Phase 1 (Baseline, 0-15):
    Normal reef. Agent should explore and build baseline statistics.

  Phase 2 (Eutrophication, 16-30):
    nutrient_runoff ↑ → nutrient_load ↑ → algae ↑ → O2 ↓ → coral ↓ → fish ↓
    Agent should discover this chain through correlated prediction errors.

  Phase 3 (Storm, 31-40):
    current ↑ → turbidity ↑ + light ↓ + sensor damage + power drain
    Agent should protect equipment (retract-sensors) when viability threatened.

  Phase 4 (Disease, 41-55):
    bacterial_pathogen ↑ → coral ↓↓ + fish ↓ + O2 ↓
    HIDDEN CAUSE: agent sees coral dying without temperature or nutrient cause.
    Should abduce pathogen → spontaneously sample water (Sherlock Holmes effect).

  Phase 5 (Recovery, 56-70):
    Everything subsides. Agent reports findings, requests concept labels.
""")

    # Show key transitions
    print("KEY OBSERVABLE TRANSITIONS:")
    print("-" * 50)
    phases_of_interest = [0, 15, 25, 30, 35, 40, 50, 55, 65, 69]
    for i in phases_of_interest:
        if i < len(env.history):
            h = env.history[i]
            ts = h["true_state"]
            print(f"  Cycle {i:2d} ({h['phase']:<14}): "
                  f"coral={ts['coral_health']:.2f} "
                  f"O2={ts['dissolved_oxygen']:.2f} "
                  f"algae={ts['algae_coverage']:.2f} "
                  f"pathogen={ts['bacterial_pathogen']:.2f}")


if __name__ == "__main__":
    main()
