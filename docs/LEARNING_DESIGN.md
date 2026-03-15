# Actions v4: From Hand-Tuned to Learned

## What Changed

### 1. Action Models Gain Confidence (meta-precision)

**Before (v3):**
```metta
(action-model retreat power-level -0.03 0.02)
; "retreat reduces power by 0.03 and improves precision by 0.02"
; Says who? The programmer. How do we know? We don't.
```

**After (v4):**
```metta
(action-model retreat power-level -0.03 0.02 0.3)
; Same prediction, but confidence 0.3 = "weak prior, not sure about this"
; Effective prediction: 0.3 Ã— -0.03 = -0.009 (discounted by uncertainty)
; After 20 cycles of experience, confidence might be 0.7+ and values refined
```

The confidence field does three things simultaneously:
- **Discounts predictions** (low confidence â†’ smaller effective deltas)
- **Adds uncertainty penalty** to EFE (don't trust what you don't know)
- **Creates exploration incentive** via model info gain (testing uncertain models has epistemic value)

### 2. Adaptive Epistemic Weight

**Before:** `$efe = error + cost - 0.5 Ã— info_gain + penalty`

**After:** `$efe = error + cost - epistemic_weight() Ã— info_gain + effect`

Where `epistemic_weight = max(0.05, safety Ã— uncertainty)`:
- Safe + uncertain â†’ ~0.8 (explore!)
- Threatened + uncertain â†’ ~0.05 (survive!)
- Safe + confident â†’ ~0.05 (no need to explore)

### 3. Principled Viability Effect

**Before:** Hardcoded retreat bonus
```metta
(if (== $action retreat) (* -0.25 $pressure) (* 0.15 $pressure))
```

**After:** Effect derived from action model predictions
```metta
; Does this action move us toward or away from viability bounds?
; Based on the action model's predicted value delta.
; Works for ANY action, including ones not yet invented.
```

If you added `(action-model recharge power-level 0.3 0.0 0.2)`, it would
automatically get a viability bonus when power is low. No special code needed.

### 4. Model Info Gain (Natural Exploration)

New EFE term: `model_info_gain(action) = Î£ (1 - confidence(action, obs))`

Actions with uncertain models have higher epistemic value. The system
naturally explores them through the same EFE mechanism that drives all
decisions. No Îµ-greedy, no exploration schedule, no temperature parameter.

Exploration diminishes as confidence grows. The agent settles into
exploiting well-understood actions.

### 5. Recursive List Processing

All fixed-arity pattern matches replaced with recursive traversal:
```metta
; Before: separate cases for 1, 2, 3, 4 elements
(= (predict-error-sum $action (($o1 $v1 $p1) ($o2 $v2 $p2) ($o3 $v3 $p3))) ...)

; After: recursive, handles any length
(= (predict-error-sum $action ()) 0.0)
(= (predict-error-sum $action (($obs $bval $bprec) . $rest))
   (+ (predict-single-error $action $obs $bval $bprec)
      (predict-error-sum $action $rest)))
```

## Learning Flow

```
Cycle N:
  observations arrive
  â”œâ”€â”€ Learn: compare snapshot(N-1) + model predictions to current observations
  â”‚         â†’ update action model parameters
  â”‚         â†’ update model confidence
  â”œâ”€â”€ Compute errors, affect
  â”œâ”€â”€ Select action via EFE (using updated, confidence-weighted models)
  â”œâ”€â”€ Snapshot current beliefs + record selected action
  â”œâ”€â”€ Update beliefs from observations
  â””â”€â”€ Execute action

Cycle N+1:
  new observations arrive (reflecting action from cycle N)
  â”œâ”€â”€ Learn: snapshot(N) + model predictions vs new observations
  â”‚         â†’ action model improves
  ...
```

## What's Still Hand-Tuned (and why)

| Parameter | Value | Role | Can it be derived? |
|-----------|-------|------|-------------------|
| Initial model confidence | 0.3 | Prior strength on action effects | Could be 0.5 (maximum entropy prior) |
| Model info gain scale | 0.1 | Relative value: model learning vs world learning | Could track actual learning rates |
| Epistemic weight floor | 0.05 | Minimum information value | Architectural choice (always value info) |
| Viability uncertainty risk | 0.1 | Risk aversion scaling near bounds | Could derive from viability margin |
| Model learning rate | 0.15 | Speed of action model updates | Could be adaptive (see below) |
| Good prediction threshold | 0.05 | What counts as a "good" model prediction | Could be mean model error |

**Key observation:** These are all *meta-parameters about learning dynamics*, not
*domain parameters about the world*. The system learns domain knowledge; the
meta-parameters control how fast and how cautiously it learns.

### Path to eliminating remaining hand-tuned values:

1. **Initial confidence 0.3 â†’ 0.5**: Maximum entropy prior. No reason to
   prefer any confidence level without data.

2. **Model info gain scale 0.1 â†’ tracked**: Measure how much confidence
   actually improves per action. Use that as the info gain estimate.

3. **Learning rate 0.15 â†’ adaptive**: Track model error variance. High
   variance â†’ increase learning rate (environment is changing). Low
   variance â†’ decrease (environment is stable).

4. **Good prediction threshold 0.05 â†’ population statistic**: Use the
   median model error across all action models as the threshold. "Good"
   means "better than typical."

## Files Changed

| File | Status | Changes |
|------|--------|---------|
| `actions.metta` | **Rewritten** | Confidence, adaptive EFE, principled viability, recursive |
| `action_learning.metta` | **New** | Snapshot, comparison, model update, diagnostics |
| `cycle.metta` | **Updated** | Learning step + snapshot step integrated |
| `policy_efe.metta` | **Needs update** | Must adopt 5-field schema + same observable names |
| `foundations.metta` | Unchanged | (config entries added in action_learning.metta) |
| `beliefs.metta` | Unchanged | |
| `affect.metta` | Unchanged | |

## Open: policy_efe.metta Reconciliation

The policy EFE module currently uses different observable names:
```metta
; policy_efe.metta uses:
(action-model wait (terrain) 0.0 -0.02)    ; 4-field, "(terrain)"

; actions.metta uses:
(action-model wait terrain-roughness 0.0 -0.02 0.3)  ; 5-field, "terrain-roughness"
```

These need to be unified:
1. Same observable names across all modules
2. Same 5-field schema with confidence
3. `sim-efe` should use confidence-weighted predictions
4. Policy selection should benefit from the same learning

The `SimState` machinery in policy_efe.metta also needs the confidence
propagation â€” when simulating a policy trajectory, model confidence
should affect how much we trust each step's prediction.
