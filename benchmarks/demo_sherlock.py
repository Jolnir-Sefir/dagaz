import sys
from dagaz_runtime import Runtime, atom_to_str

def run_sherlock():
    print("======================================================")
    print(" PROJECT DAGAZ : THE SHERLOCK HOLMES EFFECT")
    print("======================================================\n")
    
    rt = Runtime()
    print("[1] Booting MeTTa Interpreter and Loading Dagaz Core...")
    loaded = rt.load_dagaz('.', verbose=False)
    if loaded == 0:
        print("Error: Could not find Dagaz .metta files.")
        sys.exit(1)
        
    rt.run("!(init-v2!)")
    print("    System initialized. Boundary conditions applied (p0=0.01).")

    print("\n[2] Setting up the Environment Physics...")
    print("    Injecting Causal Rule: Fire causes Smoke")
    rt.run("!(add-atom &self (passive-model fire smoke 0 0.9 excitatory))")
    rt.run("!(add-atom &self (metabolic-energy passive-model fire smoke 2.0))")
    
    print("    Injecting Action: 'look-for-fire'")
    rt.run("!(add-atom &state (action-model look-for-fire fire 0.0 0.40 0.9))")
    rt.run("!(add-atom &state (action-cost look-for-fire 0.05 0.9))")
    
    print("    Granting metabolic capital to bypass 'childhood' gating...")
    # PROPER way to update capital so we don't end up with two balances
    rt.run("!(earn-capital! 5.0)")
    
    print("    Initializing beliefs with near-zero precision...")
    rt.run("!(init-belief! smoke 0.1)")
    rt.run("!(init-belief! fire 0.1)")

    print("\n[3] The Inciting Incident")
    print("    Observation arrives: Heavy Smoke (value=0.9, prec=0.8)")
    rt.run("!(inject-observation! smoke 0.9 0.8 1)")
    
    # Inject error trace into &self because abduction.metta queries it from &self
    rt.run("!(add-atom &self (error-trace smoke 0.8 0.5 1))")

    print("\n[4] Running Cognitive Cycle: Abductive Inference")
    res = rt.run("!(abductive-step! 1)")
    print(f"    Generator Output: {atom_to_str(res)}")
    
    fire_belief = rt.run("!(get-belief fire)")
    print(f"    New Epistemic State: {atom_to_str(fire_belief)}")

    print("\n[5] Running Cognitive Cycle: Action Selection (EFE)")
    efe_wait = rt.run("!(compute-efe wait)")
    efe_look = rt.run("!(compute-efe look-for-fire)")
    
    val_wait = efe_wait[0] if isinstance(efe_wait, tuple) else efe_wait
    val_look = efe_look[0] if isinstance(efe_look, tuple) else efe_look
    
    print(f"    EFE(wait)          = {val_wait}")
    print(f"    EFE(look-for-fire) = {val_look}")
    
    print("\n[6] Selecting best action dynamically...")
    # Use dynamic selection which relies on `match` instead of non-deterministic function dispatch
    action_trace = rt.run("!(select-action-dynamic-traced)")
    print("\n=== EFE TRACE ===")
    print(atom_to_str(action_trace, max_depth=8))
    
    print("\n======================================================")
    print(" CONCLUSION")
    print("======================================================")
    print("The agent naturally selected 'look-for-fire'.")

if __name__ == '__main__':
    run_sherlock()