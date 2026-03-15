"""
Orchestrator — The Sensory-Motor Boundary for Project Dagaz.

ARCHITECTURE:
    User text ──→ [LLM parse] ──→ structured observations ──→ MeTTa cycle
                                                                    ↓
    Response  ←── [LLM verbalize] ←── structured intent ←──────────┘

DESIGN PRINCIPLES COMPLIANCE:
    - Stateless: The Python orchestrator holds NO cognitive state.
    - Honest: Failed LLM parses produce low-precision observations, not errors.
    - Unified Perception: Text and hardware sensors use the exact same pathway.
    - Bottom-Up (No Enums): Utterance types are open-vocabulary, allowing emergent structure learning.
    - Transparent: The LLM translates the agent's actual predicted causal 
                   effects into words, not a scripted lookup dictionary.

USAGE:
    python orchestrator.py --trace-pipeline
"""

import json
import os
import re
import sys
import time
import logging
import argparse
from dataclasses import dataclass, field
import urllib.request
import urllib.error

logger = logging.getLogger("orchestrator")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Config:
    llm_backend: str = "ollama"
    llm_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_timeout: int = 30
    parse_max_tokens: int = 200
    parse_temperature: float = 0.1
    verbalize_max_tokens: int = 300
    verbalize_temperature: float = 0.7
    max_turns: int = 100

@dataclass
class ParseResult:
    utterance_type: str = "unknown"  # Open vocabulary! (No Enum)
    topics: list = field(default_factory=list)
    comprehension: float = 0.5
    coherence: float = 0.5
    rapport: float = 0.5
    goal_progress: float = 0.5
    urgency: float = 0.0
    threat: float = 0.0
    summary: str = ""
    reliable_fields: set = field(default_factory=set)
    dynamic_fields: dict = field(default_factory=dict)

# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """Minimal HTTP client for local LLMs."""
    def __init__(self, config: Config):
        self.config = config

    def complete(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if self.config.llm_backend == "ollama":
            url = f"{self.config.llm_url.rstrip('/')}/api/generate"
            payload = {
                "model": self.config.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature, "stop":["\n\n\n", "---", "User:"]}
            }
            resp_key = "response"
        else:
            url = f"{self.config.llm_url.rstrip('/')}/completion"
            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stop":["\n\n\n", "---", "User:"],
                "cache_prompt": True
            }
            resp_key = "content"

        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.config.llm_timeout) as resp:
                return json.loads(resp.read().decode()).get(resp_key, "").strip()
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return ""

# =============================================================================
# PERCEPTUAL BOUNDARY (Parser & Verbalizer)
# =============================================================================

class Parser:
    """Translates unstructured text into structured state observations."""
    
    TEMPLATE = (
        "You analyze user messages in a conversation. Extract structured information. Respond ONLY in the exact format shown, one field per line. No other text.\n"
        "Format:\n"
        "TYPE: speech act category (e.g., query, assertion, complaint, greeting, etc.)\n"
        "TOPICS: comma,separated,keywords\n"
        "COMPREHENSION: 0.0-1.0 (does the user understand?)\n"
        "COHERENCE: 0.0-1.0 (does this follow logically?)\n"
        "RAPPORT: 0.0-1.0 (emotional quality?)\n"
        "PROGRESS: 0.0-1.0 (advancing toward goals?)\n"
        "URGENCY: 0.0-1.0 (time-sensitive?)\n"
        "THREAT: 0.0-1.0 (physical danger or mission failure)\n"
        "SUMMARY: one sentence paraphrase\n\n"
        "Context: {context}\n"
        "User message: {message}\n---\n"
    )

    REGEXES = {
        "type": (re.compile(r"TYPE:\s*([a-z-]+)", re.I), lambda x, res: setattr(res, 'utterance_type', x.lower().strip())),
        "topics": (re.compile(r"TOPICS:\s*(.+)", re.I), lambda x, res: setattr(res, 'topics',[t.strip().lower() for t in x.split(",") if t.strip().lower() not in ("none", "n/a", "")])),
        "comprehension": (re.compile(r"COMPREHENSION:\s*([\d.]+)", re.I), lambda x, res: setattr(res, 'comprehension', float(x))),
        "coherence": (re.compile(r"COHERENCE:\s*([\d.]+)", re.I), lambda x, res: setattr(res, 'coherence', float(x))),
        "rapport": (re.compile(r"RAPPORT:\s*([\d.]+)", re.I), lambda x, res: setattr(res, 'rapport', float(x))),
        "goal_progress": (re.compile(r"PROGRESS:\s*([\d.]+)", re.I), lambda x, res: setattr(res, 'goal_progress', float(x))),
        "urgency": (re.compile(r"URGENCY:\s*([\d.]+)", re.I), lambda x, res: setattr(res, 'urgency', float(x))),
        "threat": (re.compile(r"THREAT:\s*([\d.]+)", re.I), lambda x, res: setattr(res, 'threat', float(x))),
        "summary": (re.compile(r"SUMMARY:\s*(.+)", re.I), lambda x, res: setattr(res, 'summary', x.strip())),
    }

    def extract(self, llm_output: str) -> ParseResult:
        res = ParseResult()
        for field, (pattern, setter) in self.REGEXES.items():
            m = pattern.search(llm_output)
            if m:
                try: 
                    setter(m.group(1), res)
                    res.reliable_fields.add(field)
                except ValueError: pass
        return res

class Verbalizer:
    """Translates the structured MeTTa intent back into natural language."""
    def build_prompt(self, intent: dict, context: str, user_text: str, knowledge: dict) -> str:
        action = intent.get("action", "respond")
        tone = "warm" if intent.get("valence", 0) >= 0 else "urgent/concerned"
        
        # Format what the core actually knows
        epistemic_state = "\n".join([f"- {k}: {v}" for k, v in knowledge.items()])
        if not epistemic_state:
            epistemic_state = "- No specific topics identified."

        return (
            f"SYSTEM ROLE: You are the vocal tract for a symbolic cognitive core. "
            f"Your ONLY job is to translate the core's logical intent into natural English syntax. "
            f"You are strictly forbidden from answering factual questions using your pre-trained memory.\n\n"
            f"=== USER INPUT ===\n"
            f"\"{user_text}\"\n\n"
            f"=== COGNITIVE CORE STATE ===\n"
            f"Selected Action: {action}\n"
            f"Core's Knowledge on Topics:\n{epistemic_state}\n"
            f"Emotional Tone: {tone}\n\n"
            f"=== VERBALIZATION RULES ===\n"
            f"1. If the user asks a question and the topic is 'UNKNOWN (Not in Metagraph)', "
            f"you MUST state that you do not know, or ask the user to explain it to you.\n"
            f"2. You may only claim knowledge if it is marked as 'KNOWN'.\n"
            f"3. Do not explain these rules. Just output the spoken dialogue.\n\n"
            f"Response:\n"
        )

# =============================================================================
# METTA ENGINE INTERFACE
# =============================================================================

class MeTTaInterface:
    """Native Python bridge directly to dagaz_runtime.py"""
    
    def __init__(self, script_dir: str):
        from dagaz_runtime import Runtime
        self.rt = Runtime()
        
        logger.info("Initializing Dagaz Micro-Runtime...")
        loaded = self.rt.load_dagaz(script_dir, verbose=False)
        logger.info(f"Loaded {loaded} MeTTa modules natively.")

        # --- ARCHITECTURAL BOUNDARY CONDITIONS ---
        # 1. Use pure single-step EFE (bypassing the missing policy chunker)
        #    NOTE: config atoms live in &self (code space), not &state.
        #    get-config does (match &self (config $key $val) $val).
        self.rt.run("!(remove-atom &self (config action-selection-mode policy))")
        self.rt.run("!(add-atom &self (config action-selection-mode single))")

        # 2. Establish biological embodiment / hardware bounds
        self.rt.run("!(add-atom &state (preference threat-level 0.0 1.0))")
        self.rt.run("!(add-atom &state (preference power-level 1.0 1.0))")
        
        self.rt.run("!(add-atom &state (viability-bound power-level 0.0 1.0))")
        self.rt.run("!(add-atom &state (viability-bound threat-level 0.0 1.0))")

        # Hardware sensors start fully calibrated
        self.rt.run("!(init-belief-with-precision! threat-level 0.0 0.9)")
        self.rt.run("!(init-belief-with-precision! power-level 1.0 0.9)")

        self.rt.run("!(init-belief! threat-level 0.0)")
        self.rt.run("!(init-belief! power-level 1.0)")
        
        # 3. Boot cognitive loop
        self.rt.run("!(init!)")
        self.rt.run("!(begin-conversation!)")

    def check_knowledge(self, topics: list) -> dict:
        """Queries the metagraph to see if the core actually knows about these topics."""
        knowledge = {}
        for topic in topics:
            # Check state (beliefs)
            state_query = self.rt.run(f"!(match &state (belief {topic} $v $p) True)")
            # Check ontology (semantic primitives)
            ont_query = self.rt.run(f"!(match &ontology (semantic-primitive {topic}) True)")
            
            if state_query or ont_query:
                knowledge[topic] = "KNOWN (Exists in Metagraph)"
            else:
                knowledge[topic] = "UNKNOWN (Not in Metagraph)"
        return knowledge        

    def has_pending_grounding_requests(self) -> bool:
        """Check if the grounding system has requests needing conversational input."""
        result = self.rt.run("!(has-pending-grounding-requests?)")
        return bool(result) and self._safe_val(result) == True

    def inject_observations(self, parse: ParseResult, timestamp: int):
        """Map text observations into the canonical sensor pathways."""
        topics = "(" + " ".join(f'"{t}"' for t in parse.topics) + ")"
        
        # Standard conversational injection (Notice utterance_type is now an open string)
        self.rt.run(f"!(perceive-utterance! {parse.utterance_type} {topics} "
                    f"{parse.comprehension} {parse.coherence} {parse.rapport} "
                    f"{parse.goal_progress} {parse.urgency} {timestamp})")
        
        # Hardware-level injection (extracted from text)
        if "threat" in parse.reliable_fields:
            self.rt.run(f"!(inject-observation! threat-level {parse.threat} 0.45 {timestamp})")

    def run_cycle(self) -> dict:
        """Execute the pure math of the cycle and extract the intent."""
        self.rt.run("!(cognitive-cycle!)")
        
        # Pull the structured intent tuple from the Atomspace
        cycle_count = self._safe_val(self.rt.run("!(get-cycle-count)"))
        action = self._safe_val(self.rt.run("!(get-current-action)"))
        intent_tuple = self.rt.run(f"!(package-verbalization-intent {action} {cycle_count})")
        
        logger.debug(f"RAW INTENT TUPLE: {intent_tuple}")

        return self._parse_intent(intent_tuple, str(action))

    def _safe_val(self, v):
        """Recursively unwrap dagaz_runtime tuples."""
        while isinstance(v, tuple):
            if not v: return ""
            v = v[0]
        return v

    def _extract_drivers(self, data, res_list):
        """Robust recursive extraction for 'driver' atoms from nested MeTTa outputs."""
        if not isinstance(data, tuple): return
        if len(data) >= 2 and data[0] == "driver":
            res_list.append(str(self._safe_val(data[1])))
        else:
            for child in data:
                self._extract_drivers(child, res_list)

    def _extract_effects(self, data, res_list):
        """Robust recursive extraction for 'effect' causal models from nested MeTTa outputs."""
        if not isinstance(data, tuple): return
        if len(data) >= 3 and data[0] == "effect":
            obs = str(self._safe_val(data[1]))
            try:
                val = float(self._safe_val(data[2]))
                direction = "decrease" if val < 0 else "increase"
                res_list.append(f"{direction} {obs}")
            except:
                res_list.append(f"affect {obs}")
        else:
            for child in data:
                self._extract_effects(child, res_list)

    def _parse_intent(self, intent_tuple, fallback_action) -> dict:
        res = {"action": fallback_action, "drivers":[], "valence": 0.0, "arousal": 0.3, "dominance": 0.5, "topics": [], "effects":[]}
        
        if isinstance(intent_tuple, tuple) and len(intent_tuple) == 1 and isinstance(intent_tuple[0], tuple):
            intent_tuple = intent_tuple[0]

        if not isinstance(intent_tuple, tuple) or not intent_tuple or intent_tuple[0] != "verbalization-intent":
            return res

        for item in intent_tuple[1:]:
            if not isinstance(item, tuple) or not item: continue
            
            if item[0] == "action" and len(item) >= 2: 
                res["action"] = str(self._safe_val(item[1]))
            elif item[0] == "affect" and len(item) >= 3:
                for sub in item[1:]:
                    if isinstance(sub, tuple) and len(sub) == 2:
                        val = self._safe_val(sub[1])
                        try:
                            if sub[0] == "valence": res["valence"] = float(val)
                            elif sub[0] == "arousal": res["arousal"] = float(val)
                            elif sub[0] == "dominance": res["dominance"] = float(val)
                        except (ValueError, TypeError): pass
            elif item[0] == "topics" and len(item) >= 2:
                topic_list = item[1] if isinstance(item[1], tuple) else item[1:]
                res["topics"] =[str(self._safe_val(x)).strip('"') for x in topic_list if self._safe_val(x) != "empty"]
            elif item[0] == "drivers":
                self._extract_drivers(item, res["drivers"])
            elif item[0] == "self-knowledge":
                self._extract_effects(item, res["effects"])
                
        return res

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        self.metta = MeTTaInterface(os.path.dirname(os.path.abspath(__file__)))
        self.parser = Parser()
        self.verbalizer = Verbalizer()
        self.turn_count = 0

    def process_turn(self, user_text: str, trace: bool) -> str:
        t0 = time.time()
        self.turn_count += 1

        # 1. Perception
        ctx = self.metta._safe_val(self.metta.rt.run("!(conversation-state-description)"))
        parse = self.parser.extract(self.llm.complete(self.parser.TEMPLATE.format(context=ctx, message=user_text), self.config.parse_max_tokens, self.config.parse_temperature))
        
        # 2. Epistemic Injection
        self.metta.inject_observations(parse, self.turn_count)
        
        # 3. Active Inference Cycle
        t_meta_start = time.time()
        intent = self.metta.run_cycle()
        t_meta_end = time.time()

        # --- THE EPISTEMIC FIREWALL (Short-Circuit) ---
        # Internal actions that don't require LLM verbalization:
        #   wait:           No state change, no output
        #   reflect:        Internal introspection, no partner interaction needed
        #   ground-concept: Internal unless conversational grounding is required
        action = intent.get("action", "")
        
        if action == "wait":
            if trace:
                self._print_trace(parse, intent, "[Agent maintains state. No LLM generation triggered.]", t_meta_end - t_meta_start)
            return "*Agent remains silent / maintains current state.*"
        
        if action == "reflect":
            if trace:
                self._print_trace(parse, intent, "[Agent reflects on own state. No LLM generation triggered.]", t_meta_end - t_meta_start)
            return "*Agent reflects on its own capabilities and understanding.*"
        
        if action.startswith("ground-concept"):
            # Check if conversational grounding is needed (Generator 4)
            has_requests = self.metta.has_pending_grounding_requests()
            if not has_requests:
                if trace:
                    self._print_trace(parse, intent, "[Agent works on grounding internally. No LLM generation triggered.]", t_meta_end - t_meta_start)
                return "*Agent works to understand its learned concepts.*"
            # If requests exist, fall through to verbalization —
            # the agent may need to ask the partner what a concept means
            
        # 4. Epistemic Grounding (Check what the core knows)
        knowledge = self.metta.check_knowledge(parse.topics)

        # 5. Verbalization
        response = self.llm.complete(
            self.verbalizer.build_prompt(intent, ctx, user_text, knowledge), 
            self.config.verbalize_max_tokens, 
            self.config.verbalize_temperature
        )
        
        if trace:
            self._print_trace(parse, intent, response, t_meta_end - t_meta_start)

        return response if response else "I see."

    def _print_trace(self, p, i, r, m_time):
        print("\n  ┌──[ Cognitive Pipeline ]───────────────────────────────────────────────────")
        print(f"  │ 1. Perc  : [LLM] comp={p.comprehension:.2f}, rap={p.rapport:.2f}, urg={p.urgency:.2f}, threat={p.threat:.2f}")
        print(f"  │ 2. Cycle : [MeTTa] Action ➔ '{i['action']}' (Val={i['valence']:.2f}, Aro={i['arousal']:.2f}, Dom={i['dominance']:.2f}), in {m_time*1000:.0f}ms")
        print(f"  │ 3. Verb  : [LLM] \"{r}\"")
        print("  └─────────────────────────────────────────────────────────────────────────\n")

    def _print_state(self):
        print("\n  ┌──[ Brain X-Ray ]──────────────────────────────────────────────────────────")
        beliefs = {}
        for b in self.metta.rt.state.query(('belief', '$obs', '$val', '$prec')):
            beliefs[str(b.get('$obs', ''))] = b
            
        for obs in sorted(beliefs.keys()):
            val, prec = beliefs[obs].get('$val', 0.0), beliefs[obs].get('$prec', 0.0)
            if isinstance(val, (int, float)):
                bar = "█" * int(max(0, min(1.0, val)) * 20) + "░" * (20 - int(max(0, min(1.0, val)) * 20))
                print(f"  │  {obs:25s} {bar} {val:.2f} (±{prec:.2f})")
        print("  └─────────────────────────────────────────────────────────────────────────\n")

    def run_chat(self, trace_pipeline: bool = False):
        print("=" * 70)
        print("  Project Dagaz — Pure Cognitive Core")
        print("  Commands: 'state' (view beliefs), 'sensor <obs> <val>', 'quit'")
        print("=" * 70)
        
        while self.turn_count < self.config.max_turns:
            try: user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt): break
            if not user_input: continue
            
            if user_input.lower() in ["quit", "exit"]: break
            if user_input.lower() == "state":
                self._print_state()
                continue
                
            # HARDWARE SENSOR (Cognitive Reflex)
            if user_input.lower().startswith("sensor "):
                parts = user_input.split()
                if len(parts) == 3:
                    try:
                        self.turn_count += 1
                        obs_name = parts[1]
                        obs_val = float(parts[2])
                        
                        # 1. Inject the hardware observation directly into state
                        self.metta.rt.run(f"!(inject-observation! {obs_name} {obs_val} 0.95 {self.turn_count})")
                        print(f"  [Hardware] Injected: {obs_name} = {obs_val}")
                        
                        # 2. FORCE IMMEDIATE COGNITIVE REFLEX
                        # This prevents the LLM from executing a conversational turn and wiping the threat.
                        t0 = time.time()
                        intent = self.metta.run_cycle()
                        t1 = time.time()
                        
                        # 3. Print the pipeline trace
                        mock_p = ParseResult()
                        if obs_name == "threat-level":
                            mock_p.threat = obs_val
                            
                        self._print_trace(mock_p, intent, "[Non-verbal hardware reflex execution]", t1 - t0)
                        
                    except Exception as e: 
                        print(f"  [Error] Failed to process sensor reflex: {e}")
                continue
            
            resp = self.process_turn(user_input, trace_pipeline)
            print(f"Agent: {resp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-pipeline", action="store_true", default=False, help="Show cognitive flow")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama URL")
    args = parser.parse_args()

    # Mute standard warnings
    logging.basicConfig(level=logging.WARNING)

    cfg = Config(llm_url=args.url)
    orch = Orchestrator(cfg)
    orch.run_chat(trace_pipeline=args.trace_pipeline)