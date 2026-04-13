; =============================================================================
; METTA COGNITIVE CORE v4.0 — THREE-SPACE LOADER (merged modules)
; =============================================================================
;
; THREE-SPACE ARCHITECTURE (per TRIE_CRASH_FINDINGS.md):
;
;   &self       = CODE (static after loading)
;                 Function definitions, type declarations, configuration.
;                 NEVER modified by add-atom/remove-atom during operation.
;                 ~3,000+ expressions — above trie mutation threshold, but
;                 safe because it is never mutated.
;
;   &state      = RUNTIME STATE (small, dynamic)
;                 Beliefs, observations, action-models, suspicion-links,
;                 metabolic energy, cycle count, error history, etc.
;                 ~50-200 atoms at any time — far below trie threshold.
;                 add-atom and remove-atom operate here safely.
;
;   &ontology   = SEMANTIC KNOWLEDGE (static reference data)
;                 Primitives, scales, grounding chains.
;                 ~328 atoms, never modified.
;
;
; The invariant: &self is NEVER mutated after loading completes.
; All runtime state lives in &state. The trie crash cannot trigger
; because the large space (&self) is static, and the dynamic space
; (&state) is small.
;
; =============================================================================


; Create state and ontology spaces
!(bind! &state (new-space))
!(bind! &ontology (new-space))


; Layer 1: Cognitive Core
!(import! &self core/foundations)
!(import! &self core/beliefs)
!(import! &self core/affect)
!(import! &self core/actions)
!(import! &self core/safety)
!(import! &self core/structure_learning)
!(import! &self core/atom_lifecycle)
!(import! &self core/cycle)
!(import! &self core/planning)
!(import! &self core/action_learning)
!(import! &self core/abduction)
!(import! &self core/grounding_hypotheses)
!(import! &self core/policy_efe)

; Layer 2: Knowledge
!(import! &self core/self_model)
!(import! &self core/action_grounding)
!(import! &self core/analogy_blending)
!(import! &self core/dimensional_primitives)
!(import! &self core/semantic_primitives)

; Layer 3: Interaction
!(import! &self core/conversation_model)
!(import! &self core/perception)
!(import! &self core/proprioception)
!(import! &self semantic_grounding)

; Layer 4: Domain Configuration (user-editable — load last)
!(import! &self core/domain)

; Initialization
(= (system-init!)
   (sequential (init!) (begin-conversation!) system-ready))

(system-loaded metta-cognitive-core "4.0.0-three-space-merged")
(system-architecture active-inference)
(system-principle "minimize expected free energy")
(system-memory-architecture
  (code-memory &self)
  (working-memory &state)
  (semantic-memory &ontology))
