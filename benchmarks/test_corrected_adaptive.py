#!/usr/bin/env python3
"""
Corrected Adaptive Planner — Noise Floor Direction Fix
=======================================================

The first benchmark revealed an error: "tighter noise floor" requires
a LOWER base, not higher. noise = base / confidence, so:
  - base=0.03 → very tight (50 nodes at 50 actions!)
  - base=0.05 → baseline (195K nodes at 50 actions)
  - base=0.08 → looser (more nodes)

The cliff between 0.03 and 0.05 suggests the EFE landscape has a
natural scale. The optimal noise floor should be just above the
typical EFE gap between the best and second-best action.

This benchmark tests: lower noise floor + confidence-proportional beam
across multiple random seeds to check robustness.

Author: Project Dagaz
"""

import sys
import math
import time
import random
from copy import deepcopy

from test_action_space_scaling import (
    CONFIG, Belief, Observation, ViabilityBound, EFEBreakdown, PlanBranch,
    EFEEngine, StructureLearningEngine, ScalableEnvironment,
    generate_observables, generate_actions,
)
from test_adaptive_scaling import (
    AdaptiveFractalPlanner, AdaptiveStructureLearningEngine,
)


def test_planner_robustness(n_actions: int, n_obs: int,
                             noise_base: float, n_seeds: int = 10):
    """
    Test a planner configuration across multiple random seeds.
    Returns: fraction of seeds where adaptive matches baseline answer.
    """
    from test_action_space_scaling import FractalPlanner

    matches = 0
    total_base_nodes = 0
    total_adapt_nodes = 0

    for seed in range(n_seeds):
        rng = random.Random(seed * 1000 + 7)
        observables = generate_observables(n_obs)
        actions, models, costs = generate_actions(n_actions, observables, rng)

        beliefs = {obs: Belief(rng.uniform(0.3, 0.7), 0.5)
                   for obs in observables}
        observations = {obs: Observation(rng.uniform(0.3, 0.7), 0.8, 0)
                        for obs in observables}
        viability = [ViabilityBound("power-level", 0.1, 1.0)]
        engine = EFEEngine(beliefs, observations, models, costs, viability)

        # Baseline
        base = FractalPlanner(engine, actions)
        b_branch = base.plan(6)
        total_base_nodes += base.nodes_evaluated

        # Adaptive
        adapt = AdaptiveFractalPlanner(
            engine, actions,
            noise_floor_base=noise_base,
            base_beam=8,
            max_depth=6)
        a_branch = adapt.plan()
        total_adapt_nodes += adapt.nodes_evaluated

        if b_branch.first_action == a_branch.first_action:
            matches += 1

    return {
        "match_rate": matches / n_seeds,
        "avg_base_nodes": total_base_nodes / n_seeds,
        "avg_adapt_nodes": total_adapt_nodes / n_seeds,
        "speedup": total_base_nodes / max(total_adapt_nodes, 1),
    }


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Corrected Adaptive Planner — Multi-seed Robustness Test   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # =========================================================================
    # PART 1: Find the right noise floor via multi-seed sweep
    # =========================================================================
    print(f"\n{'='*72}")
    print("PART 1: NOISE FLOOR SWEEP (10 seeds each)")
    print(f"{'='*72}")
    print("  Testing which noise floor values preserve correct answers")
    print("  across diverse random scenarios.\n")

    noise_values = [0.02, 0.03, 0.035, 0.04, 0.045, 0.05]
    action_sizes = [(10, 10), (20, 14), (30, 18)]
    n_sweep_seeds = 8

    print(f"  {'NF':>6s}", end="")
    for n_a, n_o in action_sizes:
        print(f" │ {n_a}a/{n_o}o {'Match':>6s} {'Speed':>6s}", end="")
    print()
    print(f"  {'─'*6}", end="")
    for _ in action_sizes:
        print(f"─┼─{'─'*20}", end="")
    print()

    best_configs = {}  # n_actions → best (noise, match_rate, speedup)

    for nf in noise_values:
        print(f"  {nf:6.3f}", end="")
        for n_a, n_o in action_sizes:
            r = test_planner_robustness(n_a, n_o, nf, n_seeds=n_sweep_seeds)
            print(f" │         {r['match_rate']:>5.0%} {r['speedup']:>5.1f}×", end="")

            # Track best: highest match rate, then highest speedup
            key = n_a
            if key not in best_configs:
                best_configs[key] = (nf, r["match_rate"], r["speedup"])
            else:
                _, prev_match, prev_speed = best_configs[key]
                if (r["match_rate"] > prev_match or
                    (r["match_rate"] == prev_match and r["speedup"] > prev_speed)):
                    best_configs[key] = (nf, r["match_rate"], r["speedup"])
        print()

    print(f"\n  Best noise floor per action count:")
    for n_a, n_o in action_sizes:
        nf, match, speed = best_configs[n_a]
        print(f"    {n_a:3d} actions: NF={nf:.3f} "
              f"(match={match:.0%}, speedup={speed:.1f}×)")

    # =========================================================================
    # PART 2: Full comparison at the sweet spot
    # =========================================================================
    print(f"\n{'='*72}")
    print("PART 2: FULL COMPARISON — Baseline vs Corrected Adaptive")
    print(f"{'='*72}")

    # Use NF=0.04 as a compromise (tight enough to prune, safe enough to match)
    chosen_nf = 0.04
    print(f"  Using noise_floor_base={chosen_nf}")
    print(f"  Beam: max(2, ceil(8 × degraded_confidence))")
    print()

    from test_action_space_scaling import FractalPlanner

    for n_a, n_o in action_sizes:
        print(f"  --- {n_a} actions, {n_o} observables ---")

        # Average over 5 seeds
        base_nodes_total = 0
        adapt_nodes_total = 0
        base_time_total = 0
        adapt_time_total = 0
        matches = 0

        for seed in range(5):
            rng = random.Random(seed * 777)
            observables = generate_observables(n_o)
            actions, models, costs = generate_actions(n_a, observables, rng)

            beliefs = {obs: Belief(rng.uniform(0.3, 0.7), 0.5)
                       for obs in observables}
            observations = {obs: Observation(rng.uniform(0.3, 0.7), 0.8, 0)
                            for obs in observables}
            viability = [ViabilityBound("power-level", 0.1, 1.0)]
            engine = EFEEngine(beliefs, observations, models, costs, viability)

            t0 = time.perf_counter()
            base = FractalPlanner(engine, actions)
            b_branch = base.plan(6)
            base_time_total += time.perf_counter() - t0
            base_nodes_total += base.nodes_evaluated

            t0 = time.perf_counter()
            adapt = AdaptiveFractalPlanner(
                engine, actions,
                noise_floor_base=chosen_nf,
                base_beam=8,
                max_depth=6)
            a_branch = adapt.plan()
            adapt_time_total += time.perf_counter() - t0
            adapt_nodes_total += adapt.nodes_evaluated

            if b_branch.first_action == a_branch.first_action:
                matches += 1

        avg_base_nodes = base_nodes_total / 5
        avg_adapt_nodes = adapt_nodes_total / 5
        node_speedup = avg_base_nodes / max(avg_adapt_nodes, 1)
        time_speedup = base_time_total / max(adapt_time_total, 0.001)

        print(f"    Baseline:  {avg_base_nodes:>10,.0f} nodes, "
              f"{base_time_total/5*1000:>8.1f}ms avg")
        print(f"    Adaptive:  {avg_adapt_nodes:>10,.0f} nodes, "
              f"{adapt_time_total/5*1000:>8.1f}ms avg")
        print(f"    Speedup:   {node_speedup:.1f}× nodes, "
              f"{time_speedup:.1f}× wall clock")
        print(f"    Match:     {matches}/5 seeds")
        print()

    # =========================================================================
    # PART 3: Structure learning with adaptive budget
    # =========================================================================
    print(f"{'='*72}")
    print("PART 3: STRUCTURE LEARNING — Adaptive Budget (5 seeds avg)")
    print(f"{'='*72}")

    obs_sizes = [4, 6, 10, 14, 18, 24]

    print(f"\n  {'Obs':>4s} {'Budget':>10s} │ "
          f"{'Base Recall':>12s} {'Spurious':>9s} │ "
          f"{'Adapt Recall':>12s} {'Spurious':>9s} │ {'Δ Recall':>9s}")
    print(f"  {'─'*4} {'─'*10}─┼─{'─'*12}─{'─'*9}─┼─{'─'*12}─{'─'*9}─┼─{'─'*9}")

    for n_obs in obs_sizes:
        n_pairs = n_obs * (n_obs - 1) // 2
        adaptive_budget = max(5, math.ceil(math.sqrt(n_pairs)))
        n_chains = max(2, n_obs // 3)

        base_recall_sum = 0
        adapt_recall_sum = 0
        base_spurious_sum = 0
        adapt_spurious_sum = 0
        n_seeds = 5

        for seed in range(n_seeds):
            observables = generate_observables(n_obs)
            env_b = ScalableEnvironment(observables, n_chains, random.Random(seed * 100))
            env_a = ScalableEnvironment(observables, n_chains, random.Random(seed * 100))

            sl_b = StructureLearningEngine()
            sl_a = AdaptiveStructureLearningEngine(n_observables=n_obs)
            for obs in observables:
                sl_b.set_belief(obs, 0.5, 0.5)
                sl_a.set_belief(obs, 0.5, 0.5)

            for cycle in range(60):
                sl_b.cycle = cycle
                sl_b.run_cycle(env_b.get_observations(cycle))
                sl_a.cycle = cycle
                sl_a.run_cycle(env_a.get_observations(cycle))

            real_pairs = {(c, e) for c, e, _, _ in env_b.causal_chains}

            def eval_sl(sl, env):
                found = sum(1 for c, e, _, ct in env.causal_chains
                            if (c, e) in sl.passive_models
                            and sl.passive_models[(c, e)].causal_type == ct)
                spurious = sum(1 for k in sl.passive_models if k not in real_pairs)
                return found / max(len(env.causal_chains), 1), spurious

            br, bs = eval_sl(sl_b, env_b)
            ar, ars = eval_sl(sl_a, env_a)
            base_recall_sum += br
            adapt_recall_sum += ar
            base_spurious_sum += bs
            adapt_spurious_sum += ars

        br_avg = base_recall_sum / n_seeds
        ar_avg = adapt_recall_sum / n_seeds
        bs_avg = base_spurious_sum / n_seeds
        as_avg = adapt_spurious_sum / n_seeds
        delta = ar_avg - br_avg

        sign = "+" if delta >= 0 else ""
        print(f"  {n_obs:>4d} {5:>4d}→{adaptive_budget:<4d} │ "
              f"{br_avg:>11.0%} {bs_avg:>8.1f} │ "
              f"{ar_avg:>11.0%} {as_avg:>8.1f} │ "
              f"{sign}{delta:>8.0%}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*72}")
    print("FINDINGS")
    print(f"{'='*72}")
    print("""
  1. NOISE FLOOR DIRECTION: Lower base = tighter pruning. The cliff
     between 0.03 and 0.05 reveals a natural scale in the EFE landscape.
     This is a measurable quantity — the typical gap between best and
     second-best action — and could be computed adaptively per-state.

  2. CONFIDENCE-PROPORTIONAL BEAM: max(2, ceil(8 × conf^depth)) narrows
     the beam from 8 → 4 across depth 0-5. Combined with a tighter noise
     floor, this provides a double pruning mechanism: noise kills actions
     that are indistinguishable, beam caps kill actions that are merely
     not-terrible.

  3. ADAPTIVE STRUCTURAL BUDGET: ceil(sqrt(N_pairs)) recovers missed
     causal links at 24 observables (+12% recall on the initial run).
     Multi-seed averaging shows whether this holds robustly.

  4. PRINCIPLED PARAMETER REDUCTION: The key insight is that
     noise_floor_base could itself become adaptive — computed from the
     EFE variance of the current landscape rather than declared as a
     constant. This would complete the removal of hand-tuned parameters
     from the planner.
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
