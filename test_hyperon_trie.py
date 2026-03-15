"""
Test: Does the trie crash still occur when loading Dagaz modules via native Hyperon?

This loads the same modules in the same order as dagaz_runtime.load_dagaz(),
but using the Hyperon MeTTa API instead of the Python evaluator.
It then adds a single belief atom and queries it — the pattern that
originally triggered the trie.rs:179 crash.

Usage:
    python test_hyperon_trie.py

Requires:
    pip install hyperon
    Run from the project root (where the core_*.metta files live),
    OR from a directory containing a core/ subdirectory with the .metta files.
"""

import re
import sys
from pathlib import Path

try:
    from hyperon import MeTTa
except ImportError:
    print("ERROR: hyperon package not installed. Run: pip install hyperon")
    sys.exit(1)

# Same module order as dagaz_runtime.load_dagaz()
MODULES = [
    'foundations', 'beliefs', 'affect', 'actions', 'safety',
    'structure_learning', 'atom_lifecycle', 'cycle', 'policy_efe',
    'planning', 'action_learning', 'self_model',
    'analogy_blending', 'conversation_model', 'perception',
    'proprioception', 'abduction', 'semantic_primitives',
    'dimensional_primitives', 'semantic_grounding',
    'action_grounding', 'grounding_hypotheses'
]

def find_module(mod_name: str, base: Path) -> Path | None:
    """Look for module files in both flat (core_X.metta) and nested (core/X.metta) layouts."""
    candidates = [
        base / f"core_{mod_name}.metta",       # flat layout: core_foundations.metta
        base / f"{mod_name}.metta",             # flat layout: foundations.metta
        base / "core" / f"{mod_name}.metta",    # nested layout: core/foundations.metta
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def main():
    base = Path(".")
    metta = MeTTa()

    # Phase 1: Load modules (strip import! directives, same as original repro)
    total_loaded = 0
    for mod in MODULES:
        fpath = find_module(mod, base)
        if fpath is None:
            print(f"  SKIP: {mod} (not found)")
            continue

        code = fpath.read_text(encoding='utf-8')
        # Strip import directives (they reference modules by path, not needed here)
        code = re.sub(r'!\(import!\s+[^)]*\)', '', code)
        # Strip bind! directives
        code = re.sub(r'!\(bind!\s+[^)]*\)', '', code)

        try:
            metta.run(code)
            total_loaded += 1
            print(f"  OK: {mod} ({fpath})")
        except Exception as e:
            print(f"  FAIL: {mod} — {e}")

    print(f"\nLoaded {total_loaded}/{len(MODULES)} modules")

    # Phase 2: Count expressions (approximate)
    try:
        atoms = metta.run("!(get-atoms &self)")
        count = len(atoms[0]) if atoms and atoms[0] else "unknown"
        print(f"Approximate atom count in &self: {count}")
    except Exception as e:
        print(f"Could not count atoms: {e}")

    # Phase 3: Match BEFORE adding any new atom (should be safe)
    print("\n--- Pre-mutation test ---")
    try:
        result = metta.run("!(match &self (belief $o $v $p) $o)")
        print(f"Match before add-atom: {result}")
    except Exception as e:
        print(f"Match before add-atom FAILED: {e}")

    # Phase 4: Add a single data atom post-load
    print("\n--- Adding data atom ---")
    try:
        metta.run("(belief test-obs 0.5 0.5)")
        print("Added (belief test-obs 0.5 0.5) to &self")
    except Exception as e:
        print(f"add-atom FAILED: {e}")
        return

    # Phase 5: Query the newly-added atom (THIS is where the crash occurred)
    print("\n--- Post-mutation test (crash expected here if bug is live) ---")
    try:
        result = metta.run("!(match &self (belief $o $v $p) $o)")
        print(f"Match after add-atom: {result}")
        print("\n*** NO CRASH — Bug may be fixed or not triggered ***")
    except Exception as e:
        print(f"\n*** CRASH: {e} ***")
        print("Bug is confirmed. File the issue.")

if __name__ == "__main__":
    main()
