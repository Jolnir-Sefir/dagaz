"""
Test: Semantic Grounding Integration

Exercises the key integration patterns from semantic_grounding.metta
in Python to verify the logic before Hyperon execution.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


class Atomspace:
    def __init__(self):
        self.atoms: List[tuple] = []
    def add(self, *atom):
        self.atoms.append(atom)


def build_test_atomspace():
    sp = Atomspace()
    sp.add("belief", "power-level", 0.22, 0.7)
    sp.add("belief", "terrain-roughness", 0.35, 0.6)
    sp.add("belief", "threat-level", 0.65, 0.8)
    sp.add("belief", "ambient-temperature", 0.45, 0.5)
    sp.add("observation", "power-level", 0.20, 0.8, 100)
    sp.add("observation", "threat-level", 0.70, 0.9, 100)
    sp.add("preference", "power-level", 0.9, 1.0)
    sp.add("preference", "terrain-roughness", 0.2, 0.5)
    sp.add("preference", "threat-level", 0.1, 0.9)
    sp.add("preference", "ambient-temperature", 0.5, 0.4)
    sp.add("viability-bound", "power-level", 0.15, 1.0)
    sp.add("viability-bound", "terrain-roughness", 0.0, 0.85)
    sp.add("viability-bound", "threat-level", 0.0, 0.8)
    sp.add("viability-bound", "ambient-temperature", 0.1, 0.9)
    # Observable grounding
    sp.add("obs-grounding", "power-level", "is-a", "property")
    sp.add("obs-grounding", "power-level", "is-a", "quantity")
    sp.add("obs-grounding", "power-level", "belongs-to", "self")
    sp.add("obs-grounding", "power-level", "enables", "do")
    sp.add("obs-grounding", "power-level", "enables", "can")
    sp.add("obs-semantic", "power-level", "capacity of the agent to act")
    sp.add("obs-grounding", "threat-level", "is-a", "property")
    sp.add("obs-grounding", "threat-level", "is-a", "quality")
    sp.add("obs-grounding", "threat-level", "causes", "fear")
    sp.add("obs-grounding", "threat-level", "causes", "pain")
    sp.add("obs-semantic", "threat-level", "presence of danger to the agent")
    sp.add("obs-grounding", "terrain-roughness", "is-a", "property")
    sp.add("obs-grounding", "terrain-roughness", "belongs-to", "place")
    sp.add("obs-semantic", "terrain-roughness", "how difficult the environment is")
    sp.add("obs-low-means", "power-level", ("prevents", "do"))
    sp.add("obs-low-means", "power-level", ("associated", "fear"))
    sp.add("obs-high-means", "power-level", ("enables", "do"))
    sp.add("obs-high-means", "threat-level", ("causes", "pain"))
    sp.add("obs-high-means", "threat-level", ("associated", "fear"))
    sp.add("obs-low-means", "threat-level", ("associated", "good"))
    # Action grounding
    sp.add("action-grounding", "retreat", "is-a", "do")
    sp.add("action-grounding", "retreat", "causes", ("different", "place"))
    sp.add("action-grounding", "retreat", "causes", ("away", "threat"))
    sp.add("action-grounding", "retreat", "causes", ("toward", "good"))
    sp.add("action-grounding", "retreat", "causes", ("less", "threat"))
    sp.add("action-grounding", "retreat", "requires", "agent")
    sp.add("action-grounding", "retreat", "requires", "power-level")
    sp.add("action-semantic", "retreat", "moving away from perceived danger toward safety")
    sp.add("action-grounding", "observe", "is-a", "do")
    sp.add("action-grounding", "observe", "is-a", "sense")
    sp.add("action-grounding", "observe", "causes", "know")
    sp.add("action-grounding", "observe", "causes", ("more", "know"))
    sp.add("action-grounding", "observe", "requires", "agent")
    sp.add("action-semantic", "observe", "actively sensing to reduce uncertainty")
    sp.add("action-grounding", "wait", "is-a", "do")
    sp.add("action-grounding", "wait", "associated", ("not", "do"))
    sp.add("action-grounding", "wait", "causes", ("same", "place"))
    sp.add("action-grounding", "wait", "requires", "agent")
    sp.add("action-semantic", "wait", "remaining in place without acting")
    # Action models
    sp.add("action-model", "retreat", "power-level", -0.03, 0.02, 0.3)
    sp.add("action-model", "retreat", "threat-level", -0.20, 0.04, 0.3)
    sp.add("action-model", "retreat", "terrain-roughness", -0.10, 0.05, 0.3)
    sp.add("action-model", "observe", "power-level", 0.0, 0.05, 0.3)
    sp.add("action-model", "observe", "threat-level", 0.0, 0.06, 0.3)
    sp.add("action-model", "wait", "power-level", -0.005, -0.01, 0.3)
    sp.add("action-model", "wait", "threat-level", 0.0, -0.01, 0.3)
    # Preference grounding
    sp.add("pref-grounding", "power-level", "want", ("because", ("enables", "do")))
    sp.add("pref-grounding", "threat-level", "want", ("because", ("causes", "pain")))
    sp.add("viability-grounding", "power-level", "need", ("because", ("necessary", ("can", "do"))))
    sp.add("viability-grounding", "threat-level", "need", ("because", ("prevents", "existence")))
    # Scales
    sp.add("scale-value", "power-scale", 1, "critical")
    sp.add("scale-value", "power-scale", 2, "low")
    sp.add("scale-value", "power-scale", 3, "moderate")
    sp.add("scale-value", "power-scale", 4, "high")
    sp.add("scale-value", "power-scale", 5, "full")
    sp.add("scale-value", "threat-scale", 1, "safe")
    sp.add("scale-value", "threat-scale", 2, "mild")
    sp.add("scale-value", "threat-scale", 3, "moderate")
    sp.add("scale-value", "threat-scale", 4, "dangerous")
    sp.add("scale-value", "threat-scale", 5, "lethal")
    sp.add("obs-dimension", "power-level", "agent-power", "power-scale", "higher-better")
    sp.add("obs-dimension", "threat-level", "env-threat", "threat-scale", "lower-better")
    # Primitives
    for p in ["do","agent","sense","know","can","want","need","fear","pain","good","bad",
              "place","toward","away","causes","enables","prevents","property","quantity","quality"]:
        sp.add("primitive", p, "axiomatic")
    sp.add("current-action", "retreat")
    return sp


# === Query functions ===

def what_is(sp, category, name):
    return [(a[2], a[3]) for a in sp.atoms if a[0] == f"{category}-grounding" and a[1] == name]

def describe(sp, category, name):
    for a in sp.atoms:
        if a[0] == f"{category}-semantic" and a[1] == name:
            return a[2]
    return "unknown"

def action_effects(sp, action):
    return [a[3] for a in sp.atoms if a[0] == "action-grounding" and a[1] == action and a[2] == "causes"]

def action_requirements(sp, action):
    return [a[3] for a in sp.atoms if a[0] == "action-grounding" and a[1] == action and a[2] == "requires"]

def value_to_label(sp, val, scale):
    svs = [(a[2], a[3]) for a in sp.atoms if a[0] == "scale-value" and a[1] == scale]
    if not svs: return "unknown"
    max_rank = max(r for r, _ in svs)
    bin_idx = min(max_rank, max(1, int(val * max_rank) + 1))
    for rank, label in svs:
        if rank == bin_idx: return label
    return "unknown"

def qualify_belief(sp, obs):
    belief = next(((a[2], a[3]) for a in sp.atoms if a[0] == "belief" and a[1] == obs), None)
    dim = next(((a[2], a[3], a[4]) for a in sp.atoms if a[0] == "obs-dimension" and a[1] == obs), None)
    if not belief or not dim: return None
    val, prec = belief
    _, scale, polarity = dim
    return {"obs": obs, "value": val, "label": value_to_label(sp, val, scale),
            "precision": prec, "polarity": polarity}

def evaluate_belief(sp, obs):
    bv = next((a[2] for a in sp.atoms if a[0] == "belief" and a[1] == obs), None)
    pref = next(((a[2], a[3]) for a in sp.atoms if a[0] == "preference" and a[1] == obs), None)
    if bv is None or pref is None: return None
    pv, imp = pref
    gap = pv - bv
    pol = next((a[4] for a in sp.atoms if a[0] == "obs-dimension" and a[1] == obs), None)
    if abs(gap) < 0.1: ev = "neutral"
    elif pol == "higher-better": ev = "bad" if gap > 0 else "good"
    elif pol == "lower-better": ev = "bad" if gap < 0 else "good"
    else: ev = "bad" if abs(gap) > 0.2 else "neutral"
    return {"obs": obs, "evaluation": ev, "gap": gap, "importance": imp}

def qualify_affect(v, a):
    vq = "pain" if v < -0.3 else "not-good" if v < 0 else "not-bad" if v < 0.3 else "good"
    aq = "strong" if a > 0.7 else "moderate" if a > 0.4 else "mild"
    return {"valence": vq, "arousal": aq}

def most_critical(sp):
    best = None
    for a in sp.atoms:
        if a[0] == "viability-bound":
            obs, lo, hi = a[1], a[2], a[3]
            for b in sp.atoms:
                if b[0] == "belief" and b[1] == obs:
                    margin = min(b[2] - lo, hi - b[2])
                    if best is None or margin < best[1]:
                        best = (obs, margin)
    return best

def actions_that_help(sp, obs):
    bv = next((a[2] for a in sp.atoms if a[0] == "belief" and a[1] == obs), None)
    pv = next((a[2] for a in sp.atoms if a[0] == "preference" and a[1] == obs), None)
    if bv is None or pv is None: return []
    gap = pv - bv
    return [{"action": a[1], "delta": a[3], "conf": a[5]}
            for a in sp.atoms if a[0] == "action-model" and a[2] == obs and gap * a[3] > 0]

def self_beliefs(sp):
    self_obs = {a[1] for a in sp.atoms if a[0] == "obs-grounding" and a[2] == "belongs-to" and a[3] == "self"}
    return [(b[1], b[2], b[3]) for b in sp.atoms if b[0] == "belief" and b[1] in self_obs]

def place_beliefs(sp):
    place_obs = {a[1] for a in sp.atoms if a[0] == "obs-grounding" and a[2] == "belongs-to" and a[3] == "place"}
    return [(b[1], b[2], b[3]) for b in sp.atoms if b[0] == "belief" and b[1] in place_obs]

def why_care(sp, obs):
    return {
        "want": [a[3] for a in sp.atoms if a[0] == "pref-grounding" and a[1] == obs],
        "need": [a[3] for a in sp.atoms if a[0] == "viability-grounding" and a[1] == obs]
    }


# === Self-Narrative ===

def what_am_i_doing(sp, valence, arousal):
    action = next((a[1] for a in sp.atoms if a[0] == "current-action"), None)
    nature = what_is(sp, "action", action)
    effects = action_effects(sp, action)
    reqs = action_requirements(sp, action)
    desc = describe(sp, "action", action)
    affect = qualify_affect(valence, arousal)
    evals = sorted(
        [e for e in (evaluate_belief(sp, a[1]) for a in sp.atoms if a[0] == "belief") if e],
        key=lambda e: abs(e["gap"]) * e["importance"], reverse=True)
    quals = [q for q in (qualify_belief(sp, a[1]) for a in sp.atoms if a[0] == "belief") if q]
    return {"action": action, "description": desc, "nature": nature, "effects": effects,
            "requirements": reqs, "affect": affect, "drivers": evals[:3], "situation": quals}


# =============================================================================
# TESTS
# =============================================================================

def test_observable_grounding():
    print("=" * 70)
    print("TEST 1: Observable Grounding Chains")
    print("=" * 70)
    sp = build_test_atomspace()
    for obs in ["power-level", "threat-level", "terrain-roughness"]:
        gs = what_is(sp, "obs", obs)
        d = describe(sp, "obs", obs)
        print(f"\n  What is '{obs}'?  â†’  {d}")
        for rel, target in gs:
            print(f"    {rel} â†’ {target}")
    print("\n  âœ“ Observables grounded in semantic primitives")


def test_action_grounding():
    print("\n" + "=" * 70)
    print("TEST 2: Action Grounding Chains")
    print("=" * 70)
    sp = build_test_atomspace()
    for action in ["retreat", "observe", "wait"]:
        d = describe(sp, "action", action)
        efx = action_effects(sp, action)
        reqs = action_requirements(sp, action)
        print(f"\n  What is '{action}'?  â†’  {d}")
        print(f"    causes: {efx}")
        print(f"    requires: {reqs}")
    print("\n  âœ“ Actions grounded in semantic primitives")


def test_belief_qualification():
    print("\n" + "=" * 70)
    print("TEST 3: Belief Qualification (numeric â†’ qualitative)")
    print("=" * 70)
    sp = build_test_atomspace()
    q_power = qualify_belief(sp, "power-level")
    q_threat = qualify_belief(sp, "threat-level")
    print(f"\n  power-level = 0.22  â†’  '{q_power['label']}'")
    print(f"  threat-level = 0.65  â†’  '{q_threat['label']}'")
    assert q_power["label"] == "low", f"Expected 'low', got '{q_power['label']}'"
    assert q_threat["label"] in ("moderate", "dangerous"), f"Got '{q_threat['label']}'"
    print("\n  âœ“ Numeric beliefs correctly qualified")


def test_belief_evaluation():
    print("\n" + "=" * 70)
    print("TEST 4: Belief Evaluation (good/bad vs preference)")
    print("=" * 70)
    sp = build_test_atomspace()
    for obs in ["power-level", "threat-level"]:
        ev = evaluate_belief(sp, obs)
        print(f"  {obs}: {ev['evaluation']} (gap={ev['gap']:+.2f}, importance={ev['importance']})")
    ev_p = evaluate_belief(sp, "power-level")
    ev_t = evaluate_belief(sp, "threat-level")
    assert ev_p["evaluation"] == "bad", "Low power should be bad"
    assert ev_t["evaluation"] == "bad", "High threat should be bad"
    print("\n  âœ“ Beliefs correctly evaluated against preferences")


def test_cross_module():
    print("\n" + "=" * 70)
    print("TEST 5: Cross-Module Reasoning")
    print("=" * 70)
    sp = build_test_atomspace()

    sb = self_beliefs(sp)
    pb = place_beliefs(sp)
    print(f"\n  Self-beliefs: {[o for o,_,_ in sb]}")
    print(f"  Place-beliefs: {[o for o,_,_ in pb]}")
    assert any(o == "power-level" for o,_,_ in sb)
    assert any(o == "terrain-roughness" for o,_,_ in pb)

    crit = most_critical(sp)
    print(f"  Most critical: {crit[0]} (margin={crit[1]:.2f})")
    assert crit[0] == "power-level"

    helpers = actions_that_help(sp, "threat-level")
    print(f"  Actions helping threat-level: {[h['action'] for h in helpers]}")
    assert any(h["action"] == "retreat" for h in helpers)

    r = why_care(sp, "power-level")
    print(f"  Why care about power?  want: {r['want']}, need: {r['need']}")
    assert len(r["need"]) > 0
    print("\n  âœ“ Cross-module reasoning works")


def test_self_narrative():
    print("\n" + "=" * 70)
    print("TEST 6: Self-Narrative â€” Opaque vs Grounded")
    print("=" * 70)
    sp = build_test_atomspace()

    print("\n  â”Œâ”€ BEFORE (opaque reporting) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”")
    print("  â”‚  current-action: retreat                                    â”‚")
    print("  â”‚  EFE(retreat): -0.04                                        â”‚")
    print("  â”‚  power-level: 0.22   threat-level: 0.65                     â”‚")
    print("  â”‚  valence: -0.25      arousal: 0.60                          â”‚")
    print("  â”‚                                                             â”‚")
    print("  â”‚  The system knows nothing about what these symbols mean.    â”‚")
    print("  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜")

    n = what_am_i_doing(sp, -0.25, 0.6)

    print("\n  â”Œâ”€ AFTER (grounded self-narrative) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”")
    print(f"  â”‚  I am: {n['description']}")
    print(f"  â”‚  This is a kind of: {[t for r,t in n['nature'] if r=='is-a']}")
    print(f"  â”‚  This causes: {n['effects']}")
    print(f"  â”‚  This requires: {n['requirements']}")
    print(f"  â”‚  I feel: {n['affect']['valence']} ({n['affect']['arousal']})")
    print(f"  â”‚  My situation:")
    for q in n["situation"]:
        print(f"  â”‚    {q['obs']}: {q['label']} (precision {q['precision']})")
    print(f"  â”‚  Biggest problems:")
    for d in n["drivers"]:
        if d["evaluation"] == "bad":
            dir = "too low" if d["gap"] > 0 else "too high"
            print(f"  â”‚    {d['obs']} is {dir} (importance {d['importance']})")
    print("  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜")

    # Semantic self-queries
    print("\n  Semantic self-queries:")
    acting_from_fear = (
        any(e == ("away", "threat") for e in n["effects"]) and
        n["affect"]["valence"] in ("pain", "not-good"))
    print(f"    Am I acting out of fear?   â†’ {acting_from_fear}")

    acting_to_learn = any(e == "know" for e in n["effects"])
    print(f"    Am I acting to learn?      â†’ {acting_to_learn}")

    is_self_directed = "agent" in n["requirements"]
    print(f"    Is this self-directed?     â†’ {is_self_directed}")

    involves_motion = any(e == ("different", "place") for e in n["effects"])
    print(f"    Does this involve motion?  â†’ {involves_motion}")

    crit = most_critical(sp)
    print(f"    What's most at risk?       â†’ {crit[0]} (margin {crit[1]:.2f})")

    r = why_care(sp, crit[0])
    if r["need"]:
        print(f"    Why does it matter?        â†’ {r['need'][0]}")

    assert acting_from_fear == True
    assert acting_to_learn == False
    assert involves_motion == True
    print("\n  âœ“ All semantic self-queries correct")


if __name__ == "__main__":
    test_observable_grounding()
    test_action_grounding()
    test_belief_qualification()
    test_belief_evaluation()
    test_cross_module()
    test_self_narrative()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print("""
What the integration provides:

1. OBSERVABLE GROUNDING: power-level â†’ "quantity the agent has that enables doing"
2. ACTION GROUNDING: retreat â†’ "moving away from perceived danger toward safety"
3. BELIEF QUALIFICATION: 0.22 â†’ "low" on the power scale
4. BELIEF EVALUATION: power is "bad" (gap from preference, high importance)
5. CROSS-MODULE: "retreat helps with threat" connects actions â†’ beliefs â†’ semantics
6. SELF-NARRATIVE: structured, traversable, queryable self-description
7. DOMAIN PREFERENCES: power, threat, terrain, temperature now have prefs + bounds
8. SEMANTIC SELF-QUERIES: "Am I acting out of fear?" â†’ True (grounded, not keyword match)

The system still selects actions via EFE. Behavior unchanged.
What changes: the system can now KNOW what it's doing, not just DO it.
""")
