# Grounded Deductive Reasoning â€” Phase 1.5

## Summary

Transitive closure over empirically-validated passive models, subject to the same metabolic selection pressure as all other discovered structure. Deduction is a hypothesis generator; EFE via metabolism is the single optimization principle.

**Added**: ~160 lines MeTTa (structure_learning.metta Â§IV-B), 3 config values (foundations.metta), 1 type (DeductiveOrigin), 1200-line Python benchmark (test_deductive_reasoning.py).

## Design

When two passive models form a chain Aâ†’Bâ†’C, and both have sufficient metabolic energy (meaning they've survived selection pressure), generate a candidate Aâ†’C link.

### Load-Bearing Decisions

| Decision | Rationale |
|---|---|
| **Weight = discount Ã— wâ‚ Ã— wâ‚‚** | Conservative: deduced links start weaker than their sources |
| **Lag = lagâ‚ + lagâ‚‚** | Transitive effects are slower |
| **Type = sign product** | excÃ—exc=exc, excÃ—inh=inh, inhÃ—inh=exc |
| **Same metabolic regime** | Initial energy, gestation, rent, reward â€” identical to empirical |
| **Audit trail** | `(deduction-source A C B)` links conclusion to premises |
| **Re-deduction guard** | `(deduction-origin A C)` prevents re-creating dead deductions while sources live |
| **Origin cleanup** | When both source links die, origin marker cleared, allowing re-deduction |

### What This Gives Us Over PLN

| Property | PLN | This Architecture |
|---|---|---|
| Discovers Aâ†’C from Aâ†’B, Bâ†’C | Yes | Yes |
| Falsifies wrong deductions | No (valid â‰  true) | Yes (metabolic death) |
| Grounds deduction in observation | No (logical consistency) | Yes (prediction success) |
| Single optimization principle | Logical consistency | EFE |
| Deduction feeds causal discovery | No (separate system) | Yes (feeds Phase 2) |

## Benchmark Results (3/3 passing)

### Scenario 1: Acceleration
4-node chain: ignitionâ†’heatâ†’smokeâ†’ash (each lag 1). Target: ignitionâ†’ash (lag 3).

- **With deduction**: discovered at cycle 4 (deductive origin)
- **Without deduction**: NOT FOUND in 80 cycles
- The lag-3 relationship is at the edge of the lookback window. Empirical Hebbian accumulation produces a very weak signal. Deduction bypasses this entirely.

### Scenario 2: Falsification
Seeded: Aâ†’B (exc), Bâ†’C (exc). Ground truth: A inhibits C.

- Deduction generates Aâ†’C (excitatory) at cycle 0
- **0% success rate** â€” every directional prediction is wrong
- **Metabolic death at cycle 32** (24 predictions, 0 successes)
- Gets re-deduced from still-living sources, starts dying again
- Source links Aâ†’B and Bâ†’C survive fine

This is the PLN killer. PLN keeps the valid deduction forever because valid â‰  true. We kill it because it fails to predict.

### Scenario 3: Composition
Seeded tree with max degree 2 (below hub threshold). Deduction creates transitive links, pushing degree to 4 at cycle 0. Empirical reaches same degree at cycle 1.

- 8 deductive links generated from 4 seeds
- Deduction reaches hub threshold (degree â‰¥ 3) before empirical system
- Deduced links participate in the same causal graph that Phase 2 hub detection operates on

## New Atoms

```metta
; Audit trail
(deduction-origin $cause $effect)                    ; Marks deductive provenance
(deduction-source $cause $effect $intermediate)      ; Links to premises

; Config
(config deductive-min-energy 0.5)
(config deductive-weight-discount 0.8)
(config deductive-max-chain-length 2)

; Type
(: DeductiveOrigin Type)
(: deductive DeductiveOrigin)
```

## Integration Points

- **structure-learning-step!**: Phase 1.5 fires after Phase 1, before Phase 2
- **structure-learning-step-v2!**: Same insertion point, plus origin cleanup
- **structure-summary**: Now includes `(deductive-models (deductive-model-count))`
- **metabolic-step!**: No changes â€” deduced links use identical metabolic machinery
- **safety.metta**: No changes â€” deduced links inherit stratum 3 (learned), subject to same safety constraints

## Running the Benchmark

```bash
python test_deductive_reasoning.py          # All scenarios
python test_deductive_reasoning.py -v       # Verbose (per-cycle trace)
python test_deductive_reasoning.py -s 2     # Single scenario
```
