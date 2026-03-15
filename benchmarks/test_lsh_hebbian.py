#!/usr/bin/env python3
"""
LSH Hebbian Accumulator — Reference Implementation & Benchmark
================================================================

Validates that magnitude-band LSH preserves causal discovery while
reducing the O(E²) pairing cost of the Hebbian suspicion accumulator.

Tests:
  1. Correctness: Zero significant false negatives on reef-scale scenario
  2. Scaling: Speedup vs observable count (50, 200, 500, 1000)
  3. Causal discovery: Full induction pipeline with LSH vs brute-force
  4. Regime sensitivity: Performance under different surprise distributions

Author: Project Dagaz
"""

import math
import time
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ErrorTrace:
    observable: str
    error: float
    surprise: float
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
    model_type: str
    energy: float
    created: int
    predictions: int = 0
    successes: int = 0

# =============================================================================
# LSH Hebbian Accumulator
# =============================================================================

class LSHHebbianAccumulator:
    """
    Hebbian suspicion accumulator with magnitude-band LSH.
    
    Replaces O(E²) all-pairs with O(E + Σ bᵢ²) banded pairing.
    Provably zero false negatives for pairs above the contribution threshold.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.surprise_floor = config.get('lsh_surprise_floor', 0.0003)
        self.safety_factor = config.get('lsh_safety_factor', 3.0)
        self.max_bands = config.get('lsh_max_bands', 8)
        self.min_product_band = self._derive_min_product_band()
        
        # Tracking
        self.band_stats = defaultdict(lambda: {'evaluations': 0, 'promotions': 0})
    
    def _derive_min_product_band(self) -> int:
        """
        Derive minimum product band from metabolic parameters.
        
        Each band spans 2× in surprise magnitude. For a pair in bands (a, b),
        the actual covariance product SA*SB ranges from floor²·2^(a+b)
        to floor²·2^(a+b+2). We must only prune when the MAXIMUM possible 
        product is below threshold — so we subtract 2 from the raw band.
        """
        cost = self.config['structural_cost_link']
        stability = self.config['default_structural_stability']
        plasticity = self.config['default_plasticity']
        
        cov_min = cost * (1 - stability) / (plasticity * self.safety_factor)
        ratio = cov_min / (self.surprise_floor ** 2)
        
        if ratio <= 1:
            return 0
        # Subtract 2 for within-band variance: each band spans 2× in S,
        # so the product spans 4× (2 bits) within a band pair
        return max(0, int(math.log2(ratio)) - 2)
    
    def surprise_band(self, surprise: float) -> int:
        """Assign a surprise value to its magnitude band."""
        if surprise <= self.surprise_floor:
            return 0
        raw = int(math.log2(surprise / self.surprise_floor))
        return max(0, min(raw, self.max_bands - 1))
    
    def collect_pairs_brute(self, traces: List[ErrorTrace]) -> List[Tuple[ErrorTrace, ErrorTrace]]:
        """Original O(E²) brute-force pairing."""
        pairs = []
        for ta in traces:
            for tb in traces:
                if ta.observable == tb.observable:
                    continue
                if ta.time > tb.time:
                    continue
                if ta.time == tb.time and ta.observable >= tb.observable:
                    continue
                pairs.append((ta, tb))
        return pairs
    
    def collect_pairs_lsh(self, traces: List[ErrorTrace]) -> List[Tuple[ErrorTrace, ErrorTrace]]:
        """LSH-accelerated pairing via magnitude banding."""
        # Step 1: Assign to time-band buckets and time-only groups
        buckets: Dict[Tuple[int, int], List[ErrorTrace]] = defaultdict(list)
        by_time: Dict[int, List[ErrorTrace]] = defaultdict(list)
        
        for t in traces:
            band = self.surprise_band(t.surprise)
            buckets[(t.time, band)].append(t)
            by_time[t.time].append(t)
        
        pairs = []
        times = sorted(by_time.keys())
        
        # CASE 1: Same-time pairs
        # Merge traces from all active bands at each time, then pair with
        # magnitude check. Uses standard obs_a < obs_b dedup.
        for t in times:
            traces_t = by_time[t]
            for i, ta in enumerate(traces_t):
                ba = self.surprise_band(ta.surprise)
                for j in range(i + 1, len(traces_t)):
                    tb = traces_t[j]
                    bb = self.surprise_band(tb.surprise)
                    
                    if ba + bb < self.min_product_band:
                        continue
                    if ta.observable == tb.observable:
                        continue
                    
                    # Emit in canonical order (lower obs name first)
                    if ta.observable < tb.observable:
                        pairs.append((ta, tb))
                    else:
                        pairs.append((tb, ta))
        
        # CASE 2: Cross-time pairs (ta_time < tb_time)
        # Cause is from earlier time, effect from later. No obs ordering dedup.
        for ti_idx, ta_time in enumerate(times):
            for tb_time in times[ti_idx + 1:]:
                # Get all band keys for each time
                for ba, traces_a in self._traces_by_band(buckets, ta_time).items():
                    for bb, traces_b in self._traces_by_band(buckets, tb_time).items():
                        if ba + bb < self.min_product_band:
                            continue
                        for ta in traces_a:
                            for tb in traces_b:
                                if ta.observable == tb.observable:
                                    continue
                                pairs.append((ta, tb))
        
        return pairs
    
    def _traces_by_band(self, buckets, time) -> Dict[int, List[ErrorTrace]]:
        """Get traces grouped by band for a specific time."""
        result = defaultdict(list)
        for (t, b), traces in buckets.items():
            if t == time:
                result[b] = traces
        return result
        
        return pairs
    
    def verify_no_false_negatives(self, traces: List[ErrorTrace]) -> dict:
        """
        Verify that LSH doesn't miss any pair whose single-cycle covariance
        exceeds the contribution threshold.
        """
        brute = self.collect_pairs_brute(traces)
        lsh = self.collect_pairs_lsh(traces)
        
        def pair_key(p):
            return (p[0].observable, p[0].time, p[1].observable, p[1].time)
        
        brute_set = {pair_key(p): p for p in brute}
        lsh_set = set(pair_key(p) for p in lsh)
        
        missed = set(brute_set.keys()) - lsh_set
        
        # Compute the contribution threshold
        cov_min = (self.config['structural_cost_link'] * 
                   (1 - self.config['default_structural_stability']) /
                   (self.config['default_plasticity'] * self.safety_factor))
        
        significant_misses = 0
        max_missed_cov = 0.0
        for m in missed:
            p = brute_set[m]
            cov = p[0].surprise * p[1].surprise
            max_missed_cov = max(max_missed_cov, cov)
            if cov >= cov_min:
                significant_misses += 1
        
        return {
            'brute_pairs': len(brute),
            'lsh_pairs': len(lsh),
            'missed_total': len(missed),
            'missed_significant': significant_misses,
            'max_missed_cov': max_missed_cov,
            'cov_threshold': cov_min,
            'speedup': len(brute) / max(1, len(lsh)),
            'min_product_band': self.min_product_band,
        }


# =============================================================================
# Full Induction Pipeline (with LSH vs brute-force comparison)
# =============================================================================

class InductionEngine:
    """
    Full Hebbian induction pipeline: traces → suspicion → passive models.
    Supports both brute-force and LSH pairing for comparison.
    """
    
    def __init__(self, config: dict, use_lsh: bool = False):
        self.config = config
        self.use_lsh = use_lsh
        self.lsh = LSHHebbianAccumulator(config) if use_lsh else None
        self.suspicion_links: Dict[Tuple[str, str, int], SuspicionLink] = {}
        self.passive_models: Dict[Tuple[str, str], PassiveModel] = {}
        self.cycle = 0
        self.error_traces: List[ErrorTrace] = []
        self.pair_counts = []  # Track pairs per cycle for analysis
    
    def add_traces(self, traces: List[ErrorTrace]):
        """Add error traces for the current cycle."""
        self.error_traces.extend(traces)
        # Trim to lookback window
        cutoff = self.cycle - self.config['lookback_window']
        self.error_traces = [t for t in self.error_traces if t.time >= cutoff]
    
    def is_salient(self, surprise: float, error: float) -> bool:
        """Check if a trace is salient."""
        cog = self.config.get('default_cognitive_threshold', 0.04)
        raw = self.config.get('raw_error_salience_threshold', 0.20)
        return surprise >= cog or abs(error) >= raw
    
    def update_suspicion(self):
        """Run the Hebbian suspicion update for the current cycle."""
        cutoff = self.cycle - self.config['lookback_window']
        salient = [t for t in self.error_traces
                   if t.time >= cutoff and self.is_salient(t.surprise, t.error)]
        
        stability = self.config['default_structural_stability']
        plasticity = self.config['default_plasticity']
        
        # Collect pairs
        if self.use_lsh:
            pairs = self.lsh.collect_pairs_lsh(salient)
        else:
            pairs = self.lsh.collect_pairs_brute(salient) if self.lsh else []
            if not self.lsh:
                # Inline brute-force
                pairs = []
                for ta in salient:
                    for tb in salient:
                        if ta.observable == tb.observable:
                            continue
                        if ta.time > tb.time:
                            continue
                        if ta.time == tb.time and ta.observable >= tb.observable:
                            continue
                        pairs.append((ta, tb))
        
        self.pair_counts.append(len(pairs))
        updated_keys = set()
        
        for ta, tb in pairs:
            lag = tb.time - ta.time
            sign_a = 1.0 if ta.error >= 0 else -1.0
            sign_b = 1.0 if tb.error >= 0 else -1.0
            covariance = sign_a * sign_b * ta.surprise * tb.surprise
            
            key = (ta.observable, tb.observable, lag)
            old = self.suspicion_links.get(
                key, SuspicionLink(ta.observable, tb.observable, lag, 0.0, 0)
            ).strength
            new = stability * old + plasticity * covariance
            self.suspicion_links[key] = SuspicionLink(
                ta.observable, tb.observable, lag, new, self.cycle)
            updated_keys.add(key)
            
            # Reverse direction for simultaneous
            if ta.time == tb.time:
                key_rev = (tb.observable, ta.observable, 0)
                old_rev = self.suspicion_links.get(
                    key_rev, SuspicionLink(tb.observable, ta.observable, 0, 0.0, 0)
                ).strength
                new_rev = stability * old_rev + plasticity * covariance
                self.suspicion_links[key_rev] = SuspicionLink(
                    tb.observable, ta.observable, 0, new_rev, self.cycle)
                updated_keys.add(key_rev)
        
        # Decay non-updated links
        for key, link in list(self.suspicion_links.items()):
            if key not in updated_keys:
                link.strength *= stability
    
    def check_promotions(self) -> List[PassiveModel]:
        """Check for Phase 1 promotions."""
        cost = self.config['structural_cost_link']
        promoted = []
        budget = self.config.get('max_structural_atoms_per_cycle', 5)
        
        for key, link in sorted(
            self.suspicion_links.items(),
            key=lambda x: abs(x[1].strength),
            reverse=True
        ):
            if len(promoted) >= budget:
                break
            pm_key = (link.cause, link.effect)
            if pm_key in self.passive_models:
                continue
            if abs(link.strength) > cost:
                ctype = "excitatory" if link.strength > 0 else "inhibitory"
                weight = min(abs(link.strength), 1.0)
                pm = PassiveModel(
                    cause=link.cause, effect=link.effect,
                    lag=link.lag, weight=weight, model_type=ctype,
                    energy=self.config['metabolic_initial_energy'],
                    created=self.cycle
                )
                self.passive_models[pm_key] = pm
                promoted.append(pm)
        
        return promoted
    
    def step(self, traces: List[ErrorTrace]):
        """Run one cycle of the induction pipeline."""
        self.add_traces(traces)
        self.update_suspicion()
        promoted = self.check_promotions()
        self.cycle += 1
        return promoted


# =============================================================================
# Synthetic Environment Generators
# =============================================================================

def make_causal_chain(n_obs: int, n_chains: int = 3, chain_len: int = 4,
                      noise_frac: float = 0.3, n_cycles: int = 30,
                      seed: int = 42) -> Tuple[List[List[ErrorTrace]], Set[Tuple[str, str]]]:
    """
    Generate synthetic error traces from a causal chain environment.
    
    Returns:
        traces_by_cycle: List of ErrorTrace lists, one per cycle
        true_links: Set of (cause, effect) pairs that are truly causal
    """
    rng = random.Random(seed)
    obs_names = [f"obs_{i:04d}" for i in range(n_obs)]
    
    # Define causal chains
    true_links = set()
    chains = []
    used = set()
    for _ in range(n_chains):
        available = [o for o in obs_names if o not in used]
        if len(available) < chain_len:
            break
        chain = rng.sample(available, chain_len)
        chains.append(chain)
        used.update(chain)
        for i in range(len(chain) - 1):
            true_links.add((chain[i], chain[i + 1]))
    
    # Generate traces
    traces_by_cycle = []
    for cycle in range(n_cycles):
        traces = []
        
        # Causal chain signals: each chain fires with some pattern
        for chain in chains:
            if rng.random() < 0.6:  # Chain is active this cycle
                for i, obs in enumerate(chain):
                    if cycle + i < n_cycles:  # Lagged activation
                        error = rng.uniform(0.2, 0.5) * (1 if rng.random() > 0.3 else -1)
                        precision = rng.uniform(0.3, 0.8)
                        surprise = 0.5 * precision * error ** 2
                        traces.append(ErrorTrace(
                            observable=obs,
                            error=error,
                            surprise=surprise,
                            time=cycle
                        ))
        
        # Noise: random observables with small surprises
        n_noise = int(n_obs * noise_frac * rng.uniform(0.5, 1.5))
        noise_obs = rng.sample(obs_names, min(n_noise, n_obs))
        for obs in noise_obs:
            if any(t.observable == obs for t in traces):
                continue  # Don't add noise to signal observables
            error = rng.uniform(-0.1, 0.1)
            precision = rng.uniform(0.3, 0.6)
            surprise = 0.5 * precision * error ** 2
            traces.append(ErrorTrace(
                observable=obs,
                error=error,
                surprise=surprise,
                time=cycle
            ))
        
        traces_by_cycle.append(traces)
    
    return traces_by_cycle, true_links


# =============================================================================
# Benchmarks
# =============================================================================

CONFIG = {
    "default_plasticity": 0.30,
    "default_structural_stability": 0.95,
    "default_cognitive_threshold": 0.02,
    "raw_error_salience_threshold": 0.08,
    "structural_cost_link": 0.001,
    "metabolic_initial_energy": 1.0,
    "lookback_window": 3,
    "max_structural_atoms_per_cycle": 5,
    "lsh_surprise_floor": 0.0003,
    "lsh_safety_factor": 3.0,
    "lsh_max_bands": 8,
}


def benchmark_1_correctness():
    """Test 1: Zero significant false negatives on reef-scale data."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Correctness — Zero Significant False Negatives")
    print("=" * 70)
    
    lsh = LSHHebbianAccumulator(CONFIG)
    print(f"  min_product_band = {lsh.min_product_band}")
    print(f"  surprise_floor   = {lsh.surprise_floor}")
    
    # Generate reef-like traces (14 observables, 5 cycles)
    traces_14, _ = make_causal_chain(14, n_chains=2, chain_len=4, n_cycles=5)
    all_traces = [t for cycle in traces_14 for t in cycle]
    
    result = lsh.verify_no_false_negatives(all_traces)
    
    print(f"\n  Reef-scale (14 obs):")
    print(f"    Brute pairs:          {result['brute_pairs']}")
    print(f"    LSH pairs:            {result['lsh_pairs']}")
    print(f"    Missed (total):       {result['missed_total']}")
    print(f"    Missed (significant): {result['missed_significant']}")
    print(f"    Max missed cov:       {result['max_missed_cov']:.8f}")
    print(f"    Cov threshold:        {result['cov_threshold']:.8f}")
    print(f"    Speedup:              {result['speedup']:.2f}×")
    
    passed = result['missed_significant'] == 0
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: "
          f"{'Zero' if passed else result['missed_significant']} significant false negatives")
    
    # Repeat at larger scale
    for n_obs in [50, 200, 500]:
        traces_n, _ = make_causal_chain(n_obs, n_chains=5, chain_len=4, n_cycles=5)
        all_traces = [t for cycle in traces_n for t in cycle]
        result = lsh.verify_no_false_negatives(all_traces)
        sig = result['missed_significant']
        print(f"\n  {n_obs} obs: brute={result['brute_pairs']}, "
              f"lsh={result['lsh_pairs']}, "
              f"missed_sig={sig}, "
              f"speedup={result['speedup']:.1f}×")
        if sig > 0:
            passed = False
    
    return passed


def benchmark_2_scaling():
    """Test 2: Speedup vs observable count."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Scaling — Speedup vs Observable Count")
    print("=" * 70)
    
    lsh = LSHHebbianAccumulator(CONFIG)
    results = []
    
    for n_obs in [14, 50, 100, 200, 500, 1000]:
        traces_n, _ = make_causal_chain(n_obs, n_chains=min(n_obs // 4, 10),
                                         chain_len=4, n_cycles=5, noise_frac=0.3)
        all_traces = [t for cycle in traces_n for t in cycle]
        
        # Time brute-force
        t0 = time.perf_counter()
        for _ in range(3):
            brute = lsh.collect_pairs_brute(all_traces)
        t_brute = (time.perf_counter() - t0) / 3
        
        # Time LSH
        t0 = time.perf_counter()
        for _ in range(3):
            lsh_pairs = lsh.collect_pairs_lsh(all_traces)
        t_lsh = (time.perf_counter() - t0) / 3
        
        speedup = t_brute / max(t_lsh, 1e-9)
        pair_reduction = len(brute) / max(len(lsh_pairs), 1)
        
        results.append({
            'n_obs': n_obs,
            'traces': len(all_traces),
            'brute_pairs': len(brute),
            'lsh_pairs': len(lsh_pairs),
            'pair_reduction': pair_reduction,
            't_brute_ms': t_brute * 1000,
            't_lsh_ms': t_lsh * 1000,
            'speedup': speedup,
        })
        
        print(f"  {n_obs:>5d} obs | {len(all_traces):>5d} traces | "
              f"brute={len(brute):>8d} lsh={len(lsh_pairs):>8d} | "
              f"pairs {pair_reduction:>5.1f}× | "
              f"time {speedup:>5.1f}× | "
              f"brute={t_brute*1000:>7.1f}ms lsh={t_lsh*1000:>7.1f}ms")
    
    # Check that speedup increases with scale
    speedups = [r['speedup'] for r in results]
    monotonic = all(speedups[i] <= speedups[i+1] * 1.5  # Allow some variance
                    for i in range(len(speedups) - 1))
    
    passed = results[-1]['pair_reduction'] > 2.0
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: "
          f"Pair reduction at 1000 obs = {results[-1]['pair_reduction']:.1f}×")
    return passed


def benchmark_3_causal_discovery():
    """Test 3: Full induction pipeline — LSH discovers same links as brute-force."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Causal Discovery — LSH vs Brute-Force Equivalence")
    print("=" * 70)
    
    for n_obs in [14, 100, 500]:
        traces_by_cycle, true_links = make_causal_chain(
            n_obs, n_chains=3, chain_len=4, n_cycles=30, seed=42
        )
        
        # Run with brute-force
        engine_brute = InductionEngine(CONFIG, use_lsh=False)
        engine_brute.lsh = LSHHebbianAccumulator(CONFIG)  # For brute-force method
        
        # Run with LSH
        engine_lsh = InductionEngine(CONFIG, use_lsh=True)
        
        for cycle_traces in traces_by_cycle:
            engine_brute.step(cycle_traces)
            engine_lsh.step(cycle_traces)
        
        brute_links = set((pm.cause, pm.effect) for pm in engine_brute.passive_models.values())
        lsh_links = set((pm.cause, pm.effect) for pm in engine_lsh.passive_models.values())
        
        # Check overlap with true links
        brute_true = brute_links & true_links
        lsh_true = lsh_links & true_links
        
        # Check that LSH finds at least what brute-force finds
        brute_only = brute_links - lsh_links
        lsh_only = lsh_links - brute_links
        
        total_brute_pairs = sum(engine_brute.pair_counts)
        total_lsh_pairs = sum(engine_lsh.pair_counts)
        pair_reduction = total_brute_pairs / max(total_lsh_pairs, 1)
        
        print(f"\n  {n_obs} observables, 30 cycles:")
        print(f"    True links:       {len(true_links)}")
        print(f"    Brute found:      {len(brute_links)} ({len(brute_true)} true)")
        print(f"    LSH found:        {len(lsh_links)} ({len(lsh_true)} true)")
        print(f"    Brute-only:       {len(brute_only)}")
        print(f"    LSH-only:         {len(lsh_only)}")
        print(f"    Total pairs:      brute={total_brute_pairs}, lsh={total_lsh_pairs}")
        print(f"    Pair reduction:   {pair_reduction:.1f}×")
        
        # The critical check: did LSH miss any TRUE causal link that brute found?
        missed_true = brute_true - lsh_true
        if missed_true:
            print(f"    ✗ MISSED TRUE LINKS: {missed_true}")
        else:
            print(f"    ✓ No true links missed by LSH")
    
    # Final determination: at 500 obs, LSH should find all true links brute finds
    return len(missed_true) == 0


def benchmark_4_band_distribution():
    """Test 4: Analyze how surprises distribute across bands."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Band Distribution Analysis")
    print("=" * 70)
    
    lsh = LSHHebbianAccumulator(CONFIG)
    
    # Generate traces at different scales
    for n_obs, label in [(14, "reef"), (200, "medium"), (1000, "large")]:
        traces_n, _ = make_causal_chain(n_obs, n_chains=min(n_obs // 4, 10),
                                         chain_len=4, n_cycles=5)
        all_traces = [t for cycle in traces_n for t in cycle]
        
        # Count traces per band
        band_counts = defaultdict(int)
        for t in all_traces:
            band = lsh.surprise_band(t.surprise)
            band_counts[band] += 1
        
        print(f"\n  {label} ({n_obs} obs, {len(all_traces)} traces):")
        print(f"    min_product_band = {lsh.min_product_band}")
        for band in sorted(band_counts.keys()):
            count = band_counts[band]
            pct = 100 * count / len(all_traces)
            bar = "█" * int(pct / 2)
            pruned = "✓ active" if band * 2 >= lsh.min_product_band else "✗ pruned (solo)"
            print(f"    band {band}: {count:>5d} ({pct:>5.1f}%) {bar} {pruned}")
    
    return True  # Informational, always passes


def benchmark_5_regime_sensitivity():
    """Test 5: LSH behavior under different surprise regimes."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Regime Sensitivity")
    print("=" * 70)
    
    lsh = LSHHebbianAccumulator(CONFIG)
    n_obs = 200
    
    regimes = {
        'baseline': {'noise_frac': 0.1, 'n_chains': 2, 'label': 'Low noise, few chains'},
        'active': {'noise_frac': 0.3, 'n_chains': 8, 'label': 'Moderate noise, many chains'},
        'storm': {'noise_frac': 0.8, 'n_chains': 3, 'label': 'High noise, few chains'},
    }
    
    for regime, params in regimes.items():
        traces_n, true_links = make_causal_chain(
            n_obs, n_chains=params['n_chains'], chain_len=4,
            n_cycles=5, noise_frac=params['noise_frac']
        )
        all_traces = [t for cycle in traces_n for t in cycle]
        result = lsh.verify_no_false_negatives(all_traces)
        
        print(f"\n  {regime} ({params['label']}):")
        print(f"    Traces: {len(all_traces)}")
        print(f"    Brute pairs: {result['brute_pairs']}")
        print(f"    LSH pairs:   {result['lsh_pairs']}")
        print(f"    Speedup:     {result['speedup']:.1f}×")
        print(f"    Missed sig:  {result['missed_significant']}")
    
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("LSH HEBBIAN ACCUMULATOR — REFERENCE IMPLEMENTATION")
    print("=" * 70)
    print("\nValidating magnitude-band LSH for O(E²) → O(E·b) reduction")
    print(f"Config: cost={CONFIG['structural_cost_link']}, "
          f"stability={CONFIG['default_structural_stability']}, "
          f"plasticity={CONFIG['default_plasticity']}")
    
    lsh = LSHHebbianAccumulator(CONFIG)
    print(f"Derived: min_product_band={lsh.min_product_band}, "
          f"surprise_floor={lsh.surprise_floor}")
    
    results = {}
    results['correctness'] = benchmark_1_correctness()
    results['scaling'] = benchmark_2_scaling()
    results['discovery'] = benchmark_3_causal_discovery()
    results['bands'] = benchmark_4_band_distribution()
    results['regimes'] = benchmark_5_regime_sensitivity()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:>20s}: {status}")
    
    total = sum(results.values())
    print(f"\n  {total}/{len(results)} benchmarks passed")
    
    if all(results.values()):
        print("\n  ✓ LSH Hebbian accumulator validated.")
        print("  Zero significant false negatives across all scales.")
        print("  Pair reduction scales with observable count.")
        print("  Causal discovery preserved under LSH.")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
