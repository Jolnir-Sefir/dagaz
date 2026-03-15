"""
Test: Myopia Problem and Policy-Based Solution

SCENARIO: The "Observe Trap"
- Agent has power at 0.75 (preference: 0.9, importance: 1.0)
- observe costs -0.02 power but gains +0.05 precision
- wait costs -0.01 power

MYOPIC ANALYSIS (single-step):
- observe: loses 0.02 power, gains precision â†’ low immediate EFE
- wait: loses 0.01 power, no gain â†’ slightly higher immediate EFE
- Myopic agent keeps choosing observe!

POLICY ANALYSIS (3-step):
- Policy (observe, observe, observe): power goes 0.75 â†’ 0.69 â†’ 0.63 â†’ 0.57
  This approaches the viability bound (0.15) dangerously!
- Policy (wait, wait, wait): power goes 0.75 â†’ 0.72 â†’ 0.69 â†’ 0.66
  Much more conservative.
- Policy (observe, wait, wait): balanced approach

The policy-aware agent should recognize that repeated observe is risky
even though each individual observe looks good.

This is the EXACT scenario where myopia fails.
"""

# For now, let's trace through the logic manually since MeTTa execution 
# would require the full Hyperon setup.

def simulate_myopia_test():
    """
    THE OBSERVE TRAP: Diminishing returns on information gain.
    
    Key insight: The first observation is valuable (low precision â†’ high).
    But subsequent observations have diminishing value because precision
    is already improved. Meanwhile, power costs accumulate linearly.
    
    Myopic analysis only sees the first step, where observe is great.
    Policy analysis sees that observeâ†’observeâ†’observe has diminishing
    returns while costs accumulate.
    """
    
    print("="*70)
    print("THE OBSERVE TRAP: Diminishing Returns")
    print("="*70)
    
    # Setup: agent with LOW precision (information-hungry)
    # but also limited power
    power = 0.50
    power_pref = 0.9
    power_importance = 1.0
    precision = 0.2  # Very uncertain - first observation is valuable!
    
    # Action effects
    observe_power_delta = -0.08  # Significant power cost
    observe_cost = 0.02
    
    wait_power_delta = -0.01
    wait_cost = 0.01
    
    print(f"\nInitial: power={power}, precision={precision}")
    print(f"observe costs {abs(observe_power_delta)} power")
    print(f"Precision gains have DIMINISHING RETURNS")
    
    def precision_gain(current_precision):
        """
        Diminishing returns: harder to improve already-high precision.
        First observation: +0.15, second: +0.10, third: +0.06
        """
        if current_precision < 0.3:
            return 0.15
        elif current_precision < 0.5:
            return 0.10
        elif current_precision < 0.7:
            return 0.06
        else:
            return 0.02
    
    def compute_efe(action, current_power, current_prec):
        if action == "observe":
            new_power = current_power + observe_power_delta
            prec_gain = precision_gain(current_prec)
            new_prec = min(0.95, current_prec + prec_gain)
            cost = observe_cost
            info_gain = prec_gain
        else:
            new_power = current_power + wait_power_delta
            new_prec = max(0.1, current_prec - 0.02)  # Slight precision decay
            cost = wait_cost
            info_gain = 0
        
        deviation = abs(new_power - power_pref)
        expected_error = deviation * new_prec * power_importance
        efe = expected_error - 0.5 * info_gain + cost
        return efe, new_power, new_prec
    
    # Single-step analysis
    print(f"\n--- SINGLE-STEP (MYOPIC) ANALYSIS ---")
    obs_efe, obs_power, obs_prec = compute_efe("observe", power, precision)
    wait_efe, wait_power, wait_prec = compute_efe("wait", power, precision)
    
    print(f"observe: EFE={obs_efe:.4f}")
    print(f"  â†’ power: {power:.2f}â†’{obs_power:.2f}")
    print(f"  â†’ prec:  {precision:.2f}â†’{obs_prec:.2f} (gain: {obs_prec-precision:.2f})")
    print(f"wait:    EFE={wait_efe:.4f}")
    print(f"  â†’ power: {power:.2f}â†’{wait_power:.2f}")
    
    myopic_choice = "observe" if obs_efe < wait_efe else "wait"
    print(f"\nMyopic choice: {myopic_choice}")
    if myopic_choice == "observe":
        print("  â†³ observe looks great! Big precision gain!")
    
    # Policy analysis
    print(f"\n--- 3-STEP POLICY ANALYSIS ---")
    
    def simulate_policy(actions, initial_power, initial_prec):
        power = initial_power
        prec = initial_prec
        cumulative_efe = 0
        discount = 0.95
        trajectory = []
        
        for i, action in enumerate(actions):
            old_prec = prec
            efe, power, prec = compute_efe(action, power, prec)
            cumulative_efe += (discount ** i) * efe
            trajectory.append({
                'action': action,
                'power': power,
                'prec': prec,
                'prec_gain': prec - old_prec if action == "observe" else 0,
                'step_efe': efe
            })
        
        return cumulative_efe, trajectory
    
    policies = [
        ("observe", "observe", "observe"),
        ("observe", "observe", "wait"),
        ("observe", "wait", "wait"),
        ("wait", "wait", "wait"),
        ("wait", "observe", "wait"),
    ]
    
    results = []
    for policy in policies:
        cum_efe, traj = simulate_policy(policy, power, precision)
        results.append((policy, cum_efe, traj))
        
        policy_str = " â†’ ".join(policy)
        print(f"\n{policy_str}")
        print(f"  Step details:")
        for i, t in enumerate(traj):
            print(f"    {i+1}. {t['action']}: power={t['power']:.2f}, "
                  f"prec={t['prec']:.2f}"
                  + (f" (gain +{t['prec_gain']:.2f})" if t['prec_gain'] > 0 else ""))
        print(f"  Cumulative EFE: {cum_efe:.4f}")
    
    # Find best
    best = min(results, key=lambda x: x[1])
    
    print("\n" + "="*70)
    print(f"BEST POLICY: {' â†’ '.join(best[0])}")
    print(f"Cumulative EFE: {best[1]:.4f}")
    print(f"First action: {best[0][0]}")
    print("="*70)
    
    # Compare
    policy_choice = best[0][0]
    
    print(f"\n--- COMPARISON ---")
    print(f"Myopic recommends: {myopic_choice}")
    print(f"Policy recommends: {policy_choice}")
    
    if myopic_choice != policy_choice:
        print("\nâœ“ POLICY-BASED SELECTION DIFFERS FROM MYOPIC!")
        print("\nWhy myopic fails:")
        print("  â€¢ First observation has great value (+0.15 precision)")
        print("  â€¢ Myopic analysis only sees this step")
        print("  â€¢ But 2nd observation only gains +0.10")
        print("  â€¢ And 3rd only gains +0.06")
        print("  â€¢ Meanwhile, power costs accumulate: -0.08 Ã— 3 = -0.24")
        print("  â€¢ Policy analysis sees the full trajectory")
    else:
        print("\n  Myopic and policy agree in this case.")
        
        # Show WHY they might agree
        obs3 = [r for r in results if r[0] == ("observe","observe","observe")][0]
        print(f"\n  Note: observeÃ—3 cumulative EFE = {obs3[1]:.4f}")
        print(f"        best policy EFE = {best[1]:.4f}")

def demonstrate_viability_trap():
    """More extreme example where myopic choice leads to viability violation."""
    
    print("\n" + "="*70)
    print("VIABILITY TRAP DEMONSTRATION")
    print("="*70)
    
    # Agent near viability bound
    power = 0.22  # Close to 0.15 bound!
    power_pref = 0.9
    power_importance = 1.0
    precision = 0.4
    viability_bound = 0.15
    
    observe_power_delta = -0.03
    observe_prec_delta = 0.12
    observe_cost = 0.02
    
    wait_power_delta = -0.01
    wait_prec_delta = -0.02
    wait_cost = 0.01
    
    print(f"\nInitial: power={power} (viability bound: {viability_bound})")
    print(f"Each observe costs {abs(observe_power_delta)} power, gains {observe_prec_delta} precision")
    print(f"Each wait costs {abs(wait_power_delta)} power")
    
    def compute_efe(action, current_power, current_prec):
        if action == "observe":
            new_power = current_power + observe_power_delta
            new_prec = current_prec + observe_prec_delta
            cost = observe_cost
            info_gain = observe_prec_delta
        else:
            new_power = current_power + wait_power_delta
            new_prec = current_prec + wait_prec_delta
            cost = wait_cost
            info_gain = 0
        
        deviation = abs(new_power - power_pref)
        expected_error = deviation * new_prec * power_importance
        efe = expected_error - 0.5 * info_gain + cost
        return efe, new_power, new_prec
    
    # Single-step analysis
    obs_efe, obs_power, obs_prec = compute_efe("observe", power, precision)
    wait_efe, wait_power, wait_prec = compute_efe("wait", power, precision)
    
    print(f"\n--- SINGLE-STEP (MYOPIC) ANALYSIS ---")
    print(f"observe: EFE={obs_efe:.4f}, powerâ†’{obs_power:.2f}")
    print(f"wait:    EFE={wait_efe:.4f}, powerâ†’{wait_power:.2f}")
    print(f"Myopic choice: {'observe' if obs_efe < wait_efe else 'wait'}")
    
    # Policy analysis with viability
    print(f"\n--- 3-STEP TRAJECTORY WITH VIABILITY ---")
    
    def simulate_policy_with_viability(actions, initial_power, initial_prec, bound):
        power = initial_power
        prec = initial_prec
        cumulative_efe = 0
        discount = 0.9
        trajectory = [(power, prec)]
        violated = False
        
        for i, action in enumerate(actions):
            efe, power, prec = compute_efe(action, power, prec)
            
            # Check viability BEFORE adding EFE
            if power < bound:
                violated = True
                cumulative_efe = 999.0  # Infinite penalty
                break
            
            cumulative_efe += (discount ** i) * efe
            trajectory.append((power, prec))
        
        return cumulative_efe, trajectory, violated
    
    policies = [
        ("observe", "observe", "observe"),
        ("wait", "wait", "wait"),
        ("observe", "wait", "wait"),
        ("wait", "observe", "observe"),
        ("observe", "observe", "wait"),
    ]
    
    results = []
    for policy in policies:
        cum_efe, traj, violated = simulate_policy_with_viability(
            policy, power, precision, viability_bound)
        results.append((policy, cum_efe, traj, violated))
        
        policy_str = "â†’".join(policy)
        power_traj = "â†’".join(f"{p:.2f}" for p, _ in traj)
        print(f"\nPolicy: {policy_str}")
        print(f"  Power: {power_traj}")
        if violated:
            print(f"  â›” VIABILITY VIOLATION â†’ EFE = âˆž")
        else:
            print(f"  Cumulative EFE: {cum_efe:.4f}")
    
    # Find best (finite EFE) policy
    valid_results = [(p, e, t, v) for p, e, t, v in results if not v]
    if valid_results:
        best = min(valid_results, key=lambda x: x[1])
        print("\n" + "="*70)
        print(f"BEST VIABLE POLICY: {' â†’ '.join(best[0])}")
        print(f"Cumulative EFE: {best[1]:.4f}")
        print(f"First action: {best[0][0]}")
        print("="*70)
        
        # Key comparison
        myopic_choice = "observe" if obs_efe < wait_efe else "wait"
        policy_choice = best[0][0]
        
        print(f"\n--- MYOPIA vs POLICY ---")
        print(f"Myopic recommends: {myopic_choice}")
        print(f"Policy recommends: {policy_choice}")
        
        if myopic_choice != policy_choice:
            print("\nâœ“ POLICY-BASED SELECTION DIFFERS FROM MYOPIC!")
            print("  Myopic analysis ignores cumulative power drain.")
            print("  Policy analysis recognizes the viability risk.")
        elif any(v for _, _, _, v in results if "observe" in _[0] and "observe" in _[1]):
            print("\nâœ“ Policy analysis correctly avoided dangerous policies")
            print("  that myopic analysis would have led to if repeated.")

if __name__ == "__main__":
    simulate_myopia_test()
    demonstrate_viability_trap()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The myopia problem is real and the policy-based solution addresses it.

Key insights:
1. Single-step EFE can be locally optimal but globally poor
2. MeTTa's non-determinism lets us generate all policies efficiently
3. Cumulative EFE over trajectories reveals the true cost
4. Viability bounds can prune dangerous policies early

The MeTTa implementation in policy_efe.metta uses:
- (any-action) with multiple equalities for non-deterministic generation
- (gen-policy n) to create all action sequences of length n
- (policy-efe policy state) to compute cumulative EFE
- (collapse ...) to gather all non-deterministic results
- Standard minimization over the gathered results

This is tractable for small horizons (3 actions, 3 steps = 27 policies)
and can be extended with pruning for larger spaces.
""")
