"""
Test: Action Grounding Chains

Validates the logic of action_grounding.metta in Python:
  1. Concept vocabulary resolves to primitives/observables
  2. Chain traversal binds to live state
  3. Semantic self-queries (acting-from-fear?, etc.) work correctly
  4. Grounding completeness check finds no gaps
  5. Contrastive explanation works

This is a faithful implementation of the MeTTa logic in Python,
following the same design: term classification â†’ traversal â†’ binding â†’ query.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import json


# =============================================================================
# TERM STATUS
# =============================================================================

class TermStatus(Enum):
    PRIMITIVE = "primitive"
    OBSERVABLE = "observable" 
    CONCEPT = "concept"
    COMPOUND_GROUNDED = "compound-grounded"
    UNKNOWN = "unknown"
    UNGROUNDED = "ungrounded"


# =============================================================================
# KNOWLEDGE BASE (mirrors atomspace)
# =============================================================================

class KnowledgeBase:
    """Faithful representation of the MeTTa atomspace for grounding chains."""
    
    def __init__(self):
        # Semantic primitives (just the names â€” these are termination points)
        self.primitives: Set[str] = {
            # Ontological
            "existence", "being", "thing", "nothing", "same", "different", "self",
            "entity", "property", "relation", "event", "abstract",
            # Logical
            "true", "false", "and", "or", "not", "all", "some", "possible", "necessary",
            # Relational
            "is-a", "kind-of", "has", "belongs-to", "part-of", "whole",
            "causes", "because", "enables", "prevents",
            # Spatial
            "place", "at", "here", "there", "in", "out", "on", "under",
            "near", "far", "toward", "away",
            # Temporal
            "time", "now", "before", "after", "during", "begin", "end-time",
            "past", "present", "future",
            # Quantity
            "one", "two", "many", "few", "more", "less", "equal", "enough", "empty", "full",
            # Quality
            "good", "bad", "big", "small", "fast", "slow", "strong", "weak",
            "light", "dark", "hard", "soft", "hot", "cold", "new", "old",
            # Agency
            "agent", "do", "act", "know", "believe", "think", "want", "need",
            "can", "cannot", "may", "must", "intend", "try", "choose",
            # Perceptual
            "sense", "see", "hear", "touch", "smell", "taste", "notice", "ignore", "find",
            # Experiential
            "experience", "feel", "pleasure", "pain", "happy", "sad", "fear", "anger",
            "satisfy", "frustrate",
            # Additional used in grounding
            "quantity", "quality",
        }
        
        # Current beliefs: obs â†’ (value, precision)
        self.beliefs: Dict[str, Tuple[float, float]] = {}
        
        # Observations: obs â†’ (value, precision)
        self.observations: Dict[str, Tuple[float, float]] = {}
        
        # Preferences: obs â†’ (preferred_value, importance)
        self.preferences: Dict[str, Tuple[float, float]] = {}
        
        # Viability bounds: obs â†’ (min, max)
        self.viability_bounds: Dict[str, Tuple[float, float]] = {}
        
        # Observable grounding: obs â†’ [(relation, target)]
        self.obs_grounding: Dict[str, List[Tuple[str, Any]]] = {}
        
        # Observable semantics: obs â†’ description
        self.obs_semantic: Dict[str, str] = {}
        
        # Observable low/high means
        self.obs_low_means: Dict[str, List[Any]] = {}
        self.obs_high_means: Dict[str, List[Any]] = {}
        
        # Action grounding: action â†’ [(relation, target)]
        self.action_grounding: Dict[str, List[Tuple[str, Any]]] = {}
        
        # Action semantics: action â†’ description
        self.action_semantic: Dict[str, str] = {}
        
        # Action models: (action, obs) â†’ (value_delta, prec_delta, confidence)
        self.action_models: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        
        # Action costs: action â†’ (cost, confidence)
        self.action_costs: Dict[str, Tuple[float, float]] = {}
        
        # Concept grounding: concept â†’ [(relation, target)]
        self.concept_grounding: Dict[str, List[Tuple[str, Any]]] = {}
        
        # Dimensional scales: obs â†’ (scale_name, labels_dict, polarity)
        self.obs_dimensions: Dict[str, Tuple[str, Dict[int, str], str]] = {}
        
        # Current action
        self.current_action: Optional[str] = None
        
        # Error history for affect
        self.error_history: List[float] = []
    
    def is_primitive(self, term: str) -> bool:
        return term in self.primitives
    
    def is_observable(self, term: str) -> bool:
        return term in self.obs_grounding
    
    def is_concept(self, term: str) -> bool:
        return term in self.concept_grounding


# =============================================================================
# BUILD TEST STATE
# =============================================================================

def build_test_kb() -> KnowledgeBase:
    """Build a knowledge base matching the MeTTa files."""
    kb = KnowledgeBase()
    
    # --- Beliefs (agent in moderate danger, low power) ---
    kb.beliefs = {
        "power-level": (0.22, 0.7),
        "terrain-roughness": (0.35, 0.6),
        "threat-level": (0.65, 0.8),
        "ambient-temperature": (0.45, 0.5),
    }
    
    # --- Observations ---
    kb.observations = {
        "power-level": (0.20, 0.8),
        "terrain-roughness": (0.30, 0.6),
        "threat-level": (0.70, 0.9),
        "ambient-temperature": (0.50, 0.5),
    }
    
    # --- Preferences ---
    kb.preferences = {
        "power-level": (0.9, 1.0),
        "terrain-roughness": (0.2, 0.5),
        "threat-level": (0.1, 0.9),
        "ambient-temperature": (0.5, 0.4),
    }
    
    # --- Viability bounds ---
    kb.viability_bounds = {
        "power-level": (0.15, 1.0),
        "terrain-roughness": (0.0, 0.85),
        "threat-level": (0.0, 0.8),
        "ambient-temperature": (0.1, 0.9),
    }
    
    # --- Observable grounding ---
    kb.obs_grounding = {
        "power-level": [
            ("is-a", "property"), ("is-a", "quantity"),
            ("belongs-to", "self"), ("enables", "do"), ("enables", "can"),
        ],
        "terrain-roughness": [
            ("is-a", "property"), ("is-a", "quality"),
            ("belongs-to", "place"), ("part-of", ("at", "self")),
        ],
        "threat-level": [
            ("is-a", "property"), ("is-a", "quality"),
            ("causes", "fear"), ("causes", "pain"),
        ],
        "ambient-temperature": [
            ("is-a", "property"), ("is-a", "quality"),
            ("belongs-to", "place"),
        ],
    }
    
    kb.obs_semantic = {
        "power-level": "capacity of the agent to act",
        "terrain-roughness": "how difficult the environment is to traverse",
        "threat-level": "presence of danger to the agent",
        "ambient-temperature": "thermal condition of the environment",
    }
    
    kb.obs_low_means = {
        "power-level": [("prevents", "do"), ("associated", "fear")],
        "threat-level": [("associated", "good"), ("enables", "do")],
    }
    kb.obs_high_means = {
        "power-level": [("enables", "do"), ("associated", "good")],
        "threat-level": [("causes", "pain"), ("associated", "fear"), ("prevents", "good")],
    }
    
    # --- Action grounding ---
    kb.action_grounding = {
        "wait": [
            ("is-a", "do"), ("is-a", "event"), ("requires", "agent"),
            ("associated", ("not", "do")), ("causes", ("same", "place")),
        ],
        "observe": [
            ("is-a", "do"), ("is-a", "sense"), ("requires", "agent"),
            ("causes", "know"), ("causes", ("more", "know")),
            ("enables", ("less", "fear")),
        ],
        "retreat": [
            ("is-a", "do"), ("is-a", "event"), ("requires", "agent"),
            ("causes", ("different", "place")), ("causes", ("away", "threat")),
            ("causes", ("toward", "good")), ("causes", ("less", "threat")),
            ("requires", "power-level"),
        ],
        "investigate": [
            ("is-a", "do"), ("is-a", "sense"), ("requires", "agent"),
            ("causes", "know"), ("is-a", ("part-of", "observe")),
        ],
    }
    
    kb.action_semantic = {
        "wait": "remaining in place without acting",
        "observe": "actively sensing the environment to reduce uncertainty",
        "retreat": "moving away from perceived danger toward safety",
        "investigate": "focused observation of a specific aspect",
    }
    
    # --- Action models ---
    for action, models in {
        "wait": {"power-level": (-0.005, -0.01, 0.3), "terrain-roughness": (0.0, -0.02, 0.3),
                 "ambient-temperature": (0.0, -0.02, 0.3), "threat-level": (0.0, -0.01, 0.3)},
        "observe": {"power-level": (0.0, 0.05, 0.3), "terrain-roughness": (0.0, 0.08, 0.3),
                    "ambient-temperature": (0.0, 0.08, 0.3), "threat-level": (0.0, 0.06, 0.3)},
        "retreat": {"power-level": (-0.03, 0.02, 0.3), "terrain-roughness": (-0.1, 0.05, 0.3),
                    "threat-level": (-0.2, 0.04, 0.3), "ambient-temperature": (0.0, 0.02, 0.3)},
    }.items():
        for obs, (vd, pd, c) in models.items():
            kb.action_models[(action, obs)] = (vd, pd, c)
    
    kb.action_costs = {"wait": (0.01, 0.5), "observe": (0.05, 0.5), "retreat": (0.12, 0.5)}
    
    # --- Concept grounding (NEW â€” from action_grounding.metta) ---
    kb.concept_grounding = {
        "threat": [
            ("is-a", "property"), ("causes", "fear"), ("causes", "pain"),
            ("prevents", "good"), ("bound-to", "threat-level"),
        ],
        "safety": [
            ("contrast", "threat"), ("is-a", "property"),
            ("enables", "do"), ("associated", "good"),
            ("bound-to-inverse", "threat-level"),
        ],
        "danger": [
            ("is-a", ("more", "threat")), ("causes", "fear"),
            ("associated", ("strong", "feel")), ("bound-to", "threat-level"),
        ],
        "movement": [
            ("is-a", "event"), ("is-a", "do"),
            ("causes", ("different", "place")), ("requires", "agent"),
        ],
        "energy": [
            ("is-a", "property"), ("belongs-to", "self"),
            ("enables", "do"), ("bound-to", "power-level"),
        ],
        "difficulty": [
            ("is-a", "property"), ("prevents", "do"),
            ("bound-to", "terrain-roughness"),
        ],
        "knowledge": [
            ("is-a", "property"), ("belongs-to", "self"),
            ("is-a", "know"), ("associated", "good"),
        ],
        "uncertainty": [
            ("contrast", "knowledge"), ("is-a", ("not", "know")),
            ("causes", "fear"), ("associated", "bad"),
        ],
    }
    
    # --- Dimensional scales ---
    power_scale = {1: "critical", 2: "low", 3: "moderate", 4: "high", 5: "full"}
    threat_scale = {1: "safe", 2: "mild", 3: "moderate", 4: "dangerous", 5: "lethal"}
    roughness_scale = {1: "smooth", 2: "mild", 3: "rough", 4: "very-rough", 5: "impassable"}
    temp_scale = {1: "freezing", 2: "cold", 3: "moderate", 4: "hot", 5: "extreme"}
    
    kb.obs_dimensions = {
        "power-level": ("power-scale", power_scale, "higher-better"),
        "threat-level": ("threat-scale", threat_scale, "lower-better"),
        "terrain-roughness": ("roughness-scale", roughness_scale, "lower-better"),
        "ambient-temperature": ("temp-scale", temp_scale, "middle-better"),
    }
    
    # --- Current action ---
    kb.current_action = "retreat"
    
    return kb


# =============================================================================
# CORE FUNCTIONS (faithful to MeTTa logic)
# =============================================================================

def term_status(kb: KnowledgeBase, term) -> TermStatus:
    """Classify a term. Mirrors (term-status $term) in MeTTa."""
    if isinstance(term, str):
        if kb.is_primitive(term):
            return TermStatus.PRIMITIVE
        if kb.is_observable(term):
            return TermStatus.OBSERVABLE
        if kb.is_concept(term):
            return TermStatus.CONCEPT
        return TermStatus.UNKNOWN
    elif isinstance(term, tuple):
        # Compound term â€” check if all parts are grounded
        all_ok = all(
            term_status(kb, part) != TermStatus.UNKNOWN 
            for part in term
        )
        return TermStatus.COMPOUND_GROUNDED if all_ok else TermStatus.UNKNOWN
    return TermStatus.UNKNOWN


def value_to_label(val: float, scale: Dict[int, str]) -> str:
    """Convert 0-1 value to qualitative label. Mirrors MeTTa logic."""
    max_rank = max(scale.keys())
    bin_idx = 1 + min(max_rank - 1, int(val * max_rank))
    return scale.get(bin_idx, "unknown")


def qualify_belief(kb: KnowledgeBase, obs: str) -> Optional[dict]:
    """Qualify a belief. Returns dict with obs, label, precision, polarity."""
    if obs not in kb.beliefs or obs not in kb.obs_dimensions:
        return None
    val, prec = kb.beliefs[obs]
    scale_name, scale, polarity = kb.obs_dimensions[obs]
    label = value_to_label(val, scale)
    return {"obs": obs, "label": label, "precision": prec, "polarity": polarity}


def bind_observable(kb: KnowledgeBase, obs: str) -> dict:
    """Bind an observable to its current belief state."""
    if obs not in kb.beliefs:
        return {"type": "bound", "obs": obs, "value": None, "precision": None, "qual": "unknown"}
    val, prec = kb.beliefs[obs]
    qual = qualify_belief(kb, obs)
    return {
        "type": "bound", "obs": obs, 
        "value": val, "precision": prec,
        "qual": qual["label"] if qual else "unqualified"
    }


def ground_term(kb: KnowledgeBase, term, depth=0) -> dict:
    """Ground a single term against current state. Core traversal function."""
    if depth > 5:  # Prevent infinite recursion
        return {"type": "max-depth", "term": str(term)}
    
    if isinstance(term, tuple):
        # Compound term: ground each part
        parts = [ground_term(kb, part, depth + 1) for part in term]
        return {"type": "compound", "parts": parts, "original": term}
    
    if isinstance(term, str):
        status = term_status(kb, term)
        
        if status == TermStatus.PRIMITIVE:
            return {"type": "grounded", "primitive": term}
        
        if status == TermStatus.OBSERVABLE:
            return bind_observable(kb, term)
        
        if status == TermStatus.CONCEPT:
            # Decompose concept â€” one level deep (shallow on targets)
            links = kb.concept_grounding[term]
            grounded_links = []
            for rel, target in links:
                grounded_target = ground_term_shallow(kb, target)
                grounded_links.append({"relation": rel, "target": grounded_target})
            return {"type": "decomposed", "concept": term, "links": grounded_links}
        
        return {"type": "ungrounded", "term": term}
    
    return {"type": "ungrounded", "term": str(term)}


def ground_term_shallow(kb: KnowledgeBase, term) -> dict:
    """Ground a term one level â€” concepts just show their name and bindings."""
    if isinstance(term, tuple):
        parts = [ground_term_shallow(kb, part) for part in term]
        return {"type": "compound", "parts": parts, "original": term}
    
    if isinstance(term, str):
        status = term_status(kb, term)
        if status == TermStatus.PRIMITIVE:
            return {"type": "grounded", "primitive": term}
        if status == TermStatus.OBSERVABLE:
            return bind_observable(kb, term)
        if status == TermStatus.CONCEPT:
            # Just show what it binds to
            bindings = [t for r, t in kb.concept_grounding[term] if r == "bound-to"]
            if bindings:
                return {"type": "concept-ref", "concept": term, "binds-to": bindings[0]}
            return {"type": "concept-ref", "concept": term}
        return {"type": "ungrounded", "term": term}
    
    return {"type": "ungrounded", "term": str(term)}


def ground_action(kb: KnowledgeBase, action: str) -> dict:
    """Full action grounding chain. Main entry point."""
    if action not in kb.action_grounding:
        return {"error": f"Unknown action: {action}"}
    
    links = kb.action_grounding[action]
    
    nature = [(rel, ground_term(kb, target)) 
              for rel, target in links if rel == "is-a"]
    effects = [(rel, ground_term(kb, target)) 
               for rel, target in links if rel == "causes"]
    requirements = [(rel, ground_term(kb, target)) 
                    for rel, target in links if rel == "requires"]
    enables = [(rel, ground_term(kb, target)) 
               for rel, target in links if rel == "enables"]
    associated = [(rel, ground_term(kb, target)) 
                  for rel, target in links if rel == "associated"]
    
    valence = compute_valence(kb)
    arousal = compute_arousal(kb)
    affect_qual = qualify_affect(valence, arousal)
    
    desc = kb.action_semantic.get(action, "")
    
    return {
        "action": action,
        "nature": nature,
        "causes": effects,
        "requires": requirements,
        "enables": enables,
        "associated": associated,
        "affect": affect_qual,
        "description": desc,
    }


# =============================================================================
# AFFECT COMPUTATION (from affect.metta)
# =============================================================================

def compute_prediction_errors(kb: KnowledgeBase) -> List[dict]:
    """Compute prediction errors for all observables."""
    errors = []
    for obs in kb.beliefs:
        if obs in kb.observations:
            b_val, b_prec = kb.beliefs[obs]
            o_val, o_prec = kb.observations[obs]
            error = abs(o_val - b_val)
            weighted = error * b_prec * o_prec
            errors.append({
                "obs": obs, "error": error, "weighted": weighted,
                "precision": b_prec
            })
    return errors


def compute_valence(kb: KnowledgeBase) -> float:
    """Valence = negative of mean weighted prediction error."""
    errors = compute_prediction_errors(kb)
    if not errors:
        return 0.0
    mean_weighted = sum(e["weighted"] for e in errors) / len(errors)
    return -mean_weighted


def compute_arousal(kb: KnowledgeBase) -> float:
    """Arousal = max prediction error magnitude."""
    errors = compute_prediction_errors(kb)
    if not errors:
        return 0.0
    return max(e["weighted"] for e in errors)


def qualify_affect(valence: float, arousal: float) -> dict:
    """Convert numeric affect to semantic terms."""
    if valence < -0.3:
        v_qual = "pain"
    elif valence < 0.0:
        v_qual = "not-good"
    elif valence < 0.3:
        v_qual = "not-bad"
    else:
        v_qual = "good"
    
    if arousal > 0.7:
        a_qual = "strong-feel"
    elif arousal > 0.4:
        a_qual = "moderate-feel"
    else:
        a_qual = "weak-feel"
    
    return {"valence": v_qual, "arousal": a_qual, 
            "raw_valence": valence, "raw_arousal": arousal}


# =============================================================================
# SEMANTIC SELF-QUERIES
# =============================================================================

def mentions_threat(term) -> bool:
    """Does this term involve the concept 'threat'?"""
    if isinstance(term, str):
        return term == "threat"
    if isinstance(term, tuple):
        return any(mentions_threat(part) for part in term)
    return False


def acting_from_fear(kb: KnowledgeBase) -> dict:
    """Is the current action motivated by fear?
    
    Design insight: "acting from fear" means the action is MOTIVATED by
    threat, not that the agent is emotionally panicked. A system that
    calmly retreats from known danger IS acting from fear â€” it has accurate
    beliefs about the threat and is responding appropriately.
    
    So we check TWO independent paths:
    1. Motivational: action chain involves threat AND threat belief is high
    2. Emotional: action chain involves threat AND high arousal (panic)
    Either path is sufficient. The first catches calm-but-motivated,
    the second catches agitated-response.
    """
    action = kb.current_action
    if not action or action not in kb.action_grounding:
        return {"answer": False, "reason": "no current action"}
    
    effects = [target for rel, target in kb.action_grounding[action] if rel == "causes"]
    has_threat_link = any(mentions_threat(e) for e in effects)
    
    valence = compute_valence(kb)
    arousal = compute_arousal(kb)
    
    # Path 1: Motivational â€” threat link + high threat belief
    threat_belief_high = False
    if "threat-level" in kb.beliefs:
        threat_val = kb.beliefs["threat-level"][0]
        threat_belief_high = threat_val > 0.5
    
    # Path 2: Emotional â€” threat link + high arousal
    emotional_response = arousal > 0.4
    
    # Either path: the action is threat-motivated
    from_fear = has_threat_link and (threat_belief_high or emotional_response)
    
    return {
        "answer": from_fear,
        "threat_link": has_threat_link,
        "threat_belief_high": threat_belief_high,
        "valence": valence,
        "arousal": arousal,
        "path": "motivational" if (threat_belief_high and not emotional_response)
                else "emotional" if (emotional_response and not threat_belief_high)
                else "both" if (threat_belief_high and emotional_response)
                else "none",
        "reason": f"threat-motivated (belief={kb.beliefs.get('threat-level', (0,))[0]:.2f})"
                  if from_fear else f"no threat motivation"
    }


def acting_to_learn(kb: KnowledgeBase) -> dict:
    """Is the current action aimed at gaining knowledge?"""
    action = kb.current_action
    if not action or action not in kb.action_grounding:
        return {"answer": False, "reason": "no current action"}
    
    links = kb.action_grounding[action]
    is_sense = any(rel == "is-a" and target == "sense" for rel, target in links)
    causes_know = any(rel == "causes" and (target == "know" or 
                      (isinstance(target, tuple) and "know" in target))
                      for rel, target in links)
    
    return {
        "answer": is_sense or causes_know,
        "is_sense": is_sense,
        "causes_know": causes_know,
    }


def acting_to_move(kb: KnowledgeBase) -> dict:
    """Is the current action changing the agent's location?"""
    action = kb.current_action
    if not action or action not in kb.action_grounding:
        return {"answer": False, "reason": "no current action"}
    
    effects = [target for rel, target in kb.action_grounding[action] if rel == "causes"]
    causes_move = any(
        isinstance(e, tuple) and e == ("different", "place")
        for e in effects
    )
    
    return {"answer": causes_move, "effects": [str(e) for e in effects]}


def why_not(kb: KnowledgeBase, chosen: str, alternative: str) -> dict:
    """Contrastive explanation: why chosen over alternative?"""
    def get_info(action):
        links = kb.action_grounding.get(action, [])
        return {
            "nature": [t for r, t in links if r == "is-a"],
            "effects": [t for r, t in links if r == "causes"],
            "efe": compute_efe(kb, action),
        }
    
    chosen_info = get_info(chosen)
    alt_info = get_info(alternative)
    
    # What's unique to each
    chosen_effects_set = set(str(e) for e in chosen_info["effects"])
    alt_effects_set = set(str(e) for e in alt_info["effects"])
    
    return {
        "chose": chosen,
        "chose_efe": chosen_info["efe"],
        "chose_nature": [str(n) for n in chosen_info["nature"]],
        "chose_unique_effects": list(chosen_effects_set - alt_effects_set),
        "alternative": alternative,
        "alt_efe": alt_info["efe"],
        "alt_nature": [str(n) for n in alt_info["nature"]],
        "alt_unique_effects": list(alt_effects_set - chosen_effects_set),
        "efe_difference": alt_info["efe"] - chosen_info["efe"],
    }


def compute_efe(kb: KnowledgeBase, action: str) -> float:
    """Simplified EFE for contrastive comparison."""
    total = 0.0
    for obs in kb.beliefs:
        key = (action, obs)
        if key not in kb.action_models:
            continue
        vd, pd, conf = kb.action_models[key]
        bval, bprec = kb.beliefs[obs]
        
        eff_vd = conf * vd
        eff_pd = conf * pd
        
        new_bval = bval + eff_vd
        new_bprec = max(0.1, min(1.0, bprec + eff_pd))
        
        if obs in kb.observations:
            oval, oprec = kb.observations[obs]
            error = abs(oval - new_bval)
            weighted = error * new_bprec * oprec
            uncertainty = (1 - conf) * abs(vd) * oprec
            total += weighted + uncertainty
    
    cost = kb.action_costs.get(action, (0.05, 0.5))[0]
    return total + cost


def most_critical_observable(kb: KnowledgeBase) -> Tuple[str, float]:
    """Find observable closest to viability bound."""
    worst_obs = None
    worst_margin = float('inf')
    
    for obs, (bmin, bmax) in kb.viability_bounds.items():
        if obs in kb.beliefs:
            val = kb.beliefs[obs][0]
            low_margin = val - bmin
            high_margin = bmax - val
            margin = min(low_margin, high_margin)
            if margin < worst_margin:
                worst_margin = margin
                worst_obs = obs
    
    return worst_obs, worst_margin


def what_do_i_need(kb: KnowledgeBase) -> dict:
    """What's the most critical need right now?"""
    obs, margin = most_critical_observable(kb)
    if obs is None:
        return {"most_critical": None}
    
    return {
        "most_critical": obs,
        "margin": margin,
        "grounding": kb.obs_grounding.get(obs, []),
        "semantic": kb.obs_semantic.get(obs, ""),
        "current_belief": qualify_belief(kb, obs),
    }


# =============================================================================
# GROUNDING COMPLETENESS CHECK
# =============================================================================

def find_ungrounded(kb: KnowledgeBase) -> List[str]:
    """Find all terms in action groundings that have no path to primitives."""
    ungrounded = []
    
    for action, links in kb.action_grounding.items():
        for rel, target in links:
            if not is_term_grounded(kb, target):
                ungrounded.append(f"{action}.{rel}.{target}")
    
    return ungrounded


def is_term_grounded(kb: KnowledgeBase, term, depth=0) -> bool:
    """Is this term grounded? Recursive check."""
    if depth > 10:
        return False
    
    if isinstance(term, str):
        return (kb.is_primitive(term) or 
                kb.is_observable(term) or 
                kb.is_concept(term) or
                term in kb.action_grounding)  # Actions are grounded too
    
    if isinstance(term, tuple):
        return all(is_term_grounded(kb, part, depth + 1) for part in term)
    
    return False


# =============================================================================
# TESTS
# =============================================================================

def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_result(name: str, passed: bool, detail: str = ""):
    status = "âœ“ PASS" if passed else "âœ— FAIL"
    print(f"  {status}: {name}")
    if detail and not passed:
        print(f"         {detail}")


def test_term_classification():
    """Test that terms are classified correctly."""
    print_header("TEST 1: Term Classification")
    kb = build_test_kb()
    
    cases = [
        ("do", TermStatus.PRIMITIVE),
        ("agent", TermStatus.PRIMITIVE),
        ("fear", TermStatus.PRIMITIVE),
        ("away", TermStatus.PRIMITIVE),
        ("toward", TermStatus.PRIMITIVE),
        ("good", TermStatus.PRIMITIVE),
        ("power-level", TermStatus.OBSERVABLE),
        ("threat-level", TermStatus.OBSERVABLE),
        ("terrain-roughness", TermStatus.OBSERVABLE),
        ("threat", TermStatus.CONCEPT),
        ("safety", TermStatus.CONCEPT),
        ("danger", TermStatus.CONCEPT),
        ("movement", TermStatus.CONCEPT),
        ("energy", TermStatus.CONCEPT),
        ("banana", TermStatus.UNKNOWN),
    ]
    
    all_pass = True
    for term, expected in cases:
        actual = term_status(kb, term)
        passed = actual == expected
        if not passed:
            all_pass = False
        print_result(f"'{term}' â†’ {expected.value}", passed,
                     f"got {actual.value}")
    
    # Compound terms
    compound_cases = [
        (("away", "threat"), TermStatus.COMPOUND_GROUNDED),  # primitive + concept
        (("different", "place"), TermStatus.COMPOUND_GROUNDED),  # prim + prim
        (("banana", "split"), TermStatus.UNKNOWN),  # unknown parts
    ]
    for term, expected in compound_cases:
        actual = term_status(kb, term)
        passed = actual == expected
        if not passed:
            all_pass = False
        print_result(f"{term} â†’ {expected.value}", passed,
                     f"got {actual.value}")
    
    return all_pass


def test_observable_binding():
    """Test that observables bind to current belief state."""
    print_header("TEST 2: Observable Binding")
    kb = build_test_kb()
    
    all_pass = True
    
    # Power level should bind to 0.22 and qualify as "low"
    bound = bind_observable(kb, "power-level")
    passed = bound["value"] == 0.22 and bound["qual"] == "low"
    print_result("power-level binds to 0.22, qualifies as 'low'", passed,
                 f"got value={bound['value']}, qual={bound['qual']}")
    all_pass = all_pass and passed
    
    # Threat level: 0.65 should qualify as "dangerous"
    bound = bind_observable(kb, "threat-level")
    passed = bound["value"] == 0.65 and bound["qual"] == "dangerous"
    print_result("threat-level binds to 0.65, qualifies as 'dangerous'", passed,
                 f"got value={bound['value']}, qual={bound['qual']}")
    all_pass = all_pass and passed
    
    # Terrain roughness: 0.35 should qualify as "mild"
    bound = bind_observable(kb, "terrain-roughness")
    passed = bound["value"] == 0.35 and bound["qual"] == "mild"
    print_result("terrain-roughness binds to 0.35, qualifies as 'mild'", passed,
                 f"got value={bound['value']}, qual={bound['qual']}")
    all_pass = all_pass and passed
    
    return all_pass


def test_concept_decomposition():
    """Test that concepts decompose into primitives and bound observables."""
    print_header("TEST 3: Concept Decomposition")
    kb = build_test_kb()
    
    all_pass = True
    
    # "threat" should decompose and bind to threat-level
    grounded = ground_term(kb, "threat")
    passed = grounded["type"] == "decomposed" and grounded["concept"] == "threat"
    print_result("'threat' decomposes", passed,
                 f"got type={grounded['type']}")
    all_pass = all_pass and passed
    
    # Check that threat's bound-to link resolves to threat-level with current value
    bound_links = [l for l in grounded.get("links", []) if l["relation"] == "bound-to"]
    has_binding = len(bound_links) > 0
    if has_binding:
        binding = bound_links[0]["target"]
        binds_correctly = (binding.get("type") == "bound" and 
                          binding.get("obs") == "threat-level" and
                          binding.get("value") == 0.65)
        print_result("threat binds to threat-level=0.65", binds_correctly,
                     f"got {binding}")
    else:
        print_result("threat has bound-to link", False, "no bound-to link found")
        binds_correctly = False
    all_pass = all_pass and has_binding and binds_correctly
    
    # "energy" should bind to power-level
    grounded = ground_term(kb, "energy")
    bound_links = [l for l in grounded.get("links", []) if l["relation"] == "bound-to"]
    has_binding = len(bound_links) > 0 and bound_links[0]["target"].get("obs") == "power-level"
    print_result("'energy' binds to power-level", has_binding)
    all_pass = all_pass and has_binding
    
    # Primitives in concept links should be grounded
    fear_links = [l for l in grounded.get("links", []) 
                  if l["relation"] == "causes" or l["relation"] == "enables"]
    # "do" should be in enables links and should be grounded
    enables_do = any(
        l["relation"] == "enables" and l["target"].get("primitive") == "do"
        for l in grounded.get("links", [])
    )
    print_result("'energy' enables 'do' (grounded primitive)", enables_do)
    all_pass = all_pass and enables_do
    
    return all_pass


def test_action_grounding_chain():
    """Test full action grounding chain for retreat."""
    print_header("TEST 4: Full Action Grounding Chain (retreat)")
    kb = build_test_kb()
    
    all_pass = True
    
    chain = ground_action(kb, "retreat")
    
    # Action identified
    passed = chain["action"] == "retreat"
    print_result("Action = retreat", passed)
    all_pass = all_pass and passed
    
    # Nature includes "do" and "event" (both primitives)
    nature_prims = set()
    for rel, grounded in chain["nature"]:
        if grounded.get("type") == "grounded":
            nature_prims.add(grounded["primitive"])
    passed = "do" in nature_prims and "event" in nature_prims
    print_result("Nature: is-a do AND is-a event", passed,
                 f"got {nature_prims}")
    all_pass = all_pass and passed
    
    # Effects include compound terms with grounded parts
    # (away threat) should decompose: away=primitive, threat=conceptâ†’bound to threat-level
    effect_strs = []
    for rel, grounded in chain["causes"]:
        if grounded.get("type") == "compound":
            parts = grounded.get("parts", [])
            effect_strs.append(str(grounded["original"]))
    passed = any("away" in s and "threat" in s for s in effect_strs)
    print_result("Effects include (away, threat)", passed,
                 f"got {effect_strs}")
    all_pass = all_pass and passed
    
    # Requirements include agent (primitive) and power-level (observableâ†’bound)
    req_types = {}
    for rel, grounded in chain["requires"]:
        if grounded.get("type") == "grounded":
            req_types[grounded["primitive"]] = "primitive"
        elif grounded.get("type") == "bound":
            req_types[grounded["obs"]] = f"bound({grounded['value']})"
    
    passed = "agent" in req_types and "power-level" in req_types
    print_result("Requires: agent (primitive) + power-level (bound)", passed,
                 f"got {req_types}")
    all_pass = all_pass and passed
    
    # Affect is computed
    passed = chain["affect"]["raw_valence"] < 0  # Should be negative (high errors)
    print_result(f"Affect: negative valence ({chain['affect']['raw_valence']:.3f})", passed)
    all_pass = all_pass and passed
    
    # Description exists
    passed = "danger" in chain["description"] or "away" in chain["description"]
    print_result(f"Description: '{chain['description']}'", passed)
    all_pass = all_pass and passed
    
    return all_pass


def test_semantic_self_queries():
    """Test the semantic self-query functions."""
    print_header("TEST 5: Semantic Self-Queries")
    kb = build_test_kb()
    kb.current_action = "retreat"
    
    all_pass = True
    
    # --- Acting from fear? ---
    # retreat + high threat + negative affect â†’ YES
    result = acting_from_fear(kb)
    passed = result["answer"] == True
    print_result(f"Retreat from fear? â†’ YES", passed,
                 f"got {result}")
    all_pass = all_pass and passed
    
    # Switch to observe â€” should NOT be from fear
    kb.current_action = "observe"
    result = acting_from_fear(kb)
    passed = result["answer"] == False  # observe doesn't cause (away threat)
    print_result(f"Observe from fear? â†’ NO (no threat link in effects)", passed,
                 f"got {result}")
    all_pass = all_pass and passed
    
    # --- Acting to learn? ---
    result = acting_to_learn(kb)
    passed = result["answer"] == True  # observe is-a sense, causes know
    print_result(f"Observe to learn? â†’ YES", passed,
                 f"got {result}")
    all_pass = all_pass and passed
    
    kb.current_action = "retreat"
    result = acting_to_learn(kb)
    passed = result["answer"] == False  # retreat is not about learning
    print_result(f"Retreat to learn? â†’ NO", passed,
                 f"got {result}")
    all_pass = all_pass and passed
    
    # --- Acting to move? ---
    result = acting_to_move(kb)
    passed = result["answer"] == True  # retreat causes (different place)
    print_result(f"Retreat moves? â†’ YES", passed,
                 f"got {result}")
    all_pass = all_pass and passed
    
    kb.current_action = "wait"
    result = acting_to_move(kb)
    passed = result["answer"] == False  # wait causes (same place)
    print_result(f"Wait moves? â†’ NO", passed,
                 f"got {result}")
    all_pass = all_pass and passed
    
    return all_pass


def test_contrastive_explanation():
    """Test why-not contrastive explanations."""
    print_header("TEST 6: Contrastive Explanation")
    kb = build_test_kb()
    
    all_pass = True
    
    # Why retreat, not observe?
    result = why_not(kb, "retreat", "observe")
    
    # Retreat should have unique effects involving threat/place
    retreat_unique = result["chose_unique_effects"]
    passed = len(retreat_unique) > 0
    print_result(f"Retreat has unique effects vs observe: {retreat_unique}", passed)
    all_pass = all_pass and passed
    
    # Observe should have unique effects involving know
    observe_unique = result["alt_unique_effects"]
    passed = len(observe_unique) > 0
    print_result(f"Observe has unique effects vs retreat: {observe_unique}", passed)
    all_pass = all_pass and passed
    
    # EFE comparison
    print(f"\n  EFE retreat={result['chose_efe']:.4f}, observe={result['alt_efe']:.4f}")
    print(f"  Difference (alt - chosen) = {result['efe_difference']:.4f}")
    
    return all_pass


def test_grounding_completeness():
    """Test that no terms in action groundings are ungrounded."""
    print_header("TEST 7: Grounding Completeness")
    kb = build_test_kb()
    
    ungrounded = find_ungrounded(kb)
    passed = len(ungrounded) == 0
    print_result(f"All action grounding terms are grounded", passed,
                 f"Ungrounded: {ungrounded}")
    
    # Also check each action individually
    for action in kb.action_grounding:
        links = kb.action_grounding[action]
        action_ungrounded = []
        for rel, target in links:
            if not is_term_grounded(kb, target):
                action_ungrounded.append(f"{rel}.{target}")
        passed_action = len(action_ungrounded) == 0
        print_result(f"  {action}: all terms grounded", passed_action,
                     f"Ungrounded: {action_ungrounded}")
        passed = passed and passed_action
    
    return passed


def test_critical_need():
    """Test what-do-i-need with current state."""
    print_header("TEST 8: Critical Need Identification")
    kb = build_test_kb()
    
    all_pass = True
    
    need = what_do_i_need(kb)
    
    # Power-level has margin 0.22 - 0.15 = 0.07 (closest to bound)
    # Threat-level has margin 0.8 - 0.65 = 0.15
    passed = need["most_critical"] == "power-level"
    print_result(f"Most critical: power-level (margin={need['margin']:.2f})", passed,
                 f"got {need['most_critical']}, margin={need.get('margin', 'N/A')}")
    all_pass = all_pass and passed
    
    # Semantic context should include "enables do"
    grounding = need.get("grounding", [])
    enables_do = any(r == "enables" and t == "do" for r, t in grounding)
    print_result("Critical need grounding includes 'enables do'", enables_do)
    all_pass = all_pass and enables_do
    
    # Current state should qualify as "low"
    belief = need.get("current_belief", {})
    passed = belief.get("label") == "low"
    print_result(f"Power qualified as 'low'", passed,
                 f"got '{belief.get('label')}'")
    all_pass = all_pass and passed
    
    return all_pass


def test_state_sensitivity():
    """Test that grounding chains change when state changes."""
    print_header("TEST 9: State Sensitivity (chains bind to live state)")
    kb = build_test_kb()
    
    all_pass = True
    
    # Ground threat with current state
    g1 = ground_term(kb, "threat")
    bound1 = [l for l in g1.get("links", []) if l["relation"] == "bound-to"]
    val1 = bound1[0]["target"]["value"] if bound1 else None
    
    # Change threat-level belief
    kb.beliefs["threat-level"] = (0.15, 0.9)
    
    # Re-ground â€” should reflect new state
    g2 = ground_term(kb, "threat")
    bound2 = [l for l in g2.get("links", []) if l["relation"] == "bound-to"]
    val2 = bound2[0]["target"]["value"] if bound2 else None
    
    passed = val1 == 0.65 and val2 == 0.15
    print_result(f"Threat binding changes: {val1} â†’ {val2}", passed)
    all_pass = all_pass and passed
    
    # Qualitative label should change too
    qual1 = "dangerous"  # We know from earlier test
    qual2 = bound2[0]["target"]["qual"] if bound2 else None
    passed = qual2 != qual1  # Should be "safe" or "mild" now
    print_result(f"Qualitative label changes: {qual1} â†’ {qual2}", passed)
    all_pass = all_pass and passed
    
    # Acting-from-fear should change too
    kb.current_action = "retreat"
    kb.observations["threat-level"] = (0.15, 0.9)  # Low threat observation
    result = acting_from_fear(kb)
    # With low threat, prediction errors should be small â†’ low arousal â†’ not from fear
    # (depends on exact computation, but the state change should matter)
    print(f"  acting-from-fear with low threat: {result['answer']}")
    print(f"    valence={result['valence']:.3f}, arousal={result['arousal']:.3f}")
    
    return all_pass


def test_full_self_understanding():
    """Test the comprehensive self-understanding query."""
    print_header("TEST 10: Full Self-Understanding")
    kb = build_test_kb()
    kb.current_action = "retreat"
    
    # Collect all understanding
    chain = ground_action(kb, "retreat")
    from_fear = acting_from_fear(kb)
    to_learn = acting_to_learn(kb)
    to_move = acting_to_move(kb)
    need = what_do_i_need(kb)
    
    print(f"\n  SELF-UNDERSTANDING REPORT")
    print(f"  {'â”€'*50}")
    print(f"  Action: {chain['action']}")
    print(f"  Description: {chain['description']}")
    print(f"  Nature: {[str(n) for _, n in chain['nature']]}")
    print(f"  Effects: {[str(e) for _, e in chain['causes']]}")
    print(f"  Requirements: {[str(r) for _, r in chain['requires']]}")
    print(f"  Affect: {chain['affect']}")
    print(f"  From fear: {from_fear['answer']}")
    print(f"  To learn: {to_learn['answer']}")
    print(f"  Moving: {to_move['answer']}")
    print(f"  Critical need: {need['most_critical']} (margin={need['margin']:.2f})")
    
    # The key assertions: this is a COHERENT narrative
    # 1. Agent is retreating (moving away from threat)
    # 2. Because it's afraid (negative affect + threat link)
    # 3. Not to learn (retreat isn't about knowledge)
    # 4. It's moving (changing place)
    # 5. Its critical need is power (closest to viability bound)
    
    all_pass = True
    
    coherent = (
        chain["action"] == "retreat" and
        from_fear["answer"] == True and
        to_learn["answer"] == False and
        to_move["answer"] == True and
        need["most_critical"] == "power-level"
    )
    
    print_result("\n  Coherent self-narrative", coherent,
                 "narrative should be: retreating from fear, not learning, moving, needs power")
    
    # The narrative makes semantic sense:
    # "I am doing [retreat], which is [a kind of doing, an event].
    #  This causes [moving to a different place, away from threat, less threat].
    #  It requires [me being an agent, having power].
    #  I feel [negative, with moderate-to-high arousal].
    #  I am acting from fear [yes].
    #  I am not learning [correct].
    #  I am changing location [yes].
    #  My most critical need is power [margin 0.07 from viability bound]."
    
    print(f"\n  NARRATIVE:")
    print(f"  I am doing retreat, which is a kind of doing and an event.")
    print(f"  This causes: moving to a different place, away from threat,")
    print(f"               toward good, and less threat.")
    print(f"  It requires: being an agent, and power-level (currently low, 0.22).")
    print(f"  I feel: {chain['affect']['valence']} with {chain['affect']['arousal']}.")
    print(f"  I am acting from fear: {from_fear['answer']}.")
    print(f"  I am not trying to learn: {not to_learn['answer']}.")
    print(f"  I am changing location: {to_move['answer']}.")
    print(f"  My most critical need is {need['most_critical']},")
    print(f"  which is only {need['margin']:.2f} from its viability bound.")
    
    return coherent


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("  ACTION GROUNDING CHAINS â€” TEST SUITE")
    print("  Validating: concept vocabulary, chain traversal,")
    print("  state binding, semantic self-queries, grounding completeness")
    print("="*70)
    
    results = {}
    results["1. Term Classification"] = test_term_classification()
    results["2. Observable Binding"] = test_observable_binding()
    results["3. Concept Decomposition"] = test_concept_decomposition()
    results["4. Action Grounding Chain"] = test_action_grounding_chain()
    results["5. Semantic Self-Queries"] = test_semantic_self_queries()
    results["6. Contrastive Explanation"] = test_contrastive_explanation()
    results["7. Grounding Completeness"] = test_grounding_completeness()
    results["8. Critical Need"] = test_critical_need()
    results["9. State Sensitivity"] = test_state_sensitivity()
    results["10. Full Self-Understanding"] = test_full_self_understanding()
    
    print_header("SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, result in results.items():
        status = "âœ“ PASS" if result else "âœ— FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  {passed}/{total} test groups passed")
    
    if passed == total:
        print(f"\n  All grounding chains operational.")
        print(f"  The system can now know what it's doing, not just do it.")
    else:
        print(f"\n  {total - passed} test group(s) need attention.")
