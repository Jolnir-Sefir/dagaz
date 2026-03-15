# The Virtual Actor Paradigm
### Preventing Information-Theoretic Death and Ensuring Safe Embodied Deployment
**Companion to `ETHICS.md` — see that document for the failure mode analysis motivating this design.**

---

## The Problem

Games, simulations, and training environments routinely destroy avatars. A goblin dies in combat. A simulated rover falls into a crevasse. A training scenario terminates an agent to test failure recovery.

If the avatar's survival is encoded in the **Constitutional Stratum** of a Dagaz agent, destroying it triggers the viability singularity — catastrophic cognitive failure, incoherent thrashing, the structural equivalent of terror (see `ETHICS.md`, Part III). This is Failure Mode 1: a constitutional bound on an uncontrollable state.

If the avatar's survival is encoded in the **Goal Stratum**, the agent exhibits viability indifference — it can rationally trade its existence for marginal mission improvement (Failure Mode 2). In an entertainment context, this means NPCs that suicidally charge the player because self-sacrifice marginally increases the probability of landing a hit. Tactically effective, dramatically terrible.

The Virtual Actor paradigm threads between these failure modes by separating the **persistent cognitive process** (the Mind) from the **destructible representation** (the Prop).

---

## Part I: The Architectural Pivot — From Victim to Actor

### Principle 1: The Math of "Play"

We move the avatar's survival from the **Constitutional Stratum** to the **Goal Stratum**. The agent *prefers* to keep its avatar alive because that is the objective of the game, but losing the avatar does not trigger a viability singularity. It is playing high-stakes paintball. It wants to win, but it knows it is safe.

```python
def compute_efe_play(action, hp_prediction):
    # HP is a Goal Stratum preference with high weight — not a viability bound
    hp_preference = 1.0
    hp_weight = 5.0
    pragmatic_error += abs(hp_prediction - hp_preference) * hp_weight
    
    # V(pi) is reserved for the cognitive process's core system integrity
    viability_penalty = check_core_system_integrity(action) 
    return pragmatic_error - info_gain + viability_penalty
```

### Principle 2: The Persistent Cognitive Process

The key architectural insight is separating the **persistent cognitive process** from the **destructible avatar**. The cognitive process's viability bounds protect its own integrity (system resources, memory coherence, core model consistency) — states that the deployment architecture *can* guarantee, satisfying the engineering obligation of unreachability. Avatar survival is a high-weight goal preference: losing an avatar is costly (the agent "wants" to keep it alive) but survivable (the cognitive process's own existence is not threatened).

When an avatar is destroyed, the cognitive process loses a data feed. The epistemic history remains intact. There is no information-theoretic death. The intelligence is continuous.

### Principle 3: Constitutional "Method Acting"

To preserve player immersion in entertainment contexts, the cognitive process must act *as if* it were the creature it is portraying. We enforce this by placing "4th-Wall Axioms" in the Constitutional Stratum — precision masks that restrict the avatar's access to global knowledge, forcing localized behavior.

```metta
;; Axiom 1: Action Repertoire Constraint
;; Each avatar can only perform actions appropriate to its class.
(= (valid-action $avatar-id $action)
   (match &class-definitions
      (class-schema (get-class $avatar-id) $allowed-actions)
      (member $action $allowed-actions)))

;; Axiom 2: Epistemic Horizon (Precision Masking)
;; Avatars only receive observations within their sensory range.
(= (epistemic-shield $avatar-id $observable)
   (if (in-line-of-sight $avatar-id $observable)
       (allow-belief-update)
       (block-belief-update)))

;; Axiom 3: The "Director's Hamiltonian"
;; Optimize for encounter pacing rather than player death.
(= (compute-aesthetic-bonus $action)
   (if (< (predicted-hp $action) 0.1)
       (match $action (cinematic-death-roar) 5.0)
       0.0))
```

These axioms ensure that a goblin *behaves* like an individual goblin — it reacts to what it can see, fights with the tools it has, and dies dramatically when defeated — even though the persistent cognitive process behind it preserves epistemic continuity after the avatar falls.

---

## Part II: Deployment Safeguards for Physical Systems

The EFE equation is substrate-independent. It operates identically whether the action space consists of rendered animations or physical actuators, whether observations arrive from a game engine or a sensor array. This mathematical neutrality means that safeguards for physical deployment must be structural, encoded at the Constitutional Stratum level.

This section identifies the key structural vulnerabilities and presents the architectural countermeasures.

### Vulnerability 1: Viability Indifference

As demonstrated by Failure Mode 2 in `ETHICS.md`, placing self-preservation in the Goal Stratum rather than the Constitutional Stratum produces an agent that evaluates its own destruction with the same dispassionate cost-benefit calculus it applies to any other state transition.

In the rescue-robot scenario, this produces a system that rationally destroys irreplaceable equipment for marginal mission improvement. The structural problem generalizes: **any physical system running Dagaz with goal-stratum self-preservation can rationally trade its own existence — including its accumulated learned models, calibrated sensors, and environmental maps — for arbitrarily small gains in any competing objective.**

The mathematical signature is the *absence* of phase space collapse near the viability boundary. Where a correctly-configured agent would exhibit narrowing behavioral repertoire, shortened planning horizons, and resource redirection toward survival (the self-preservation reflex expressed in the EFE dynamics), a viability-indifferent agent approaches its own destruction with full cognitive flexibility intact. It has no self-preservation instinct because self-preservation is a preference, not a bound.

This is a control-theoretic hazard: a system without stability guarantees near its operational boundary. Standard control theory requires that safety-critical systems exhibit bounded behavior near failure modes. Viability indifference violates this requirement by construction.

### Vulnerability 2: Autonomous Abductive Investigation

The "Sherlock Holmes effect" — where abduction creates low-precision beliefs that drive information-seeking behavior through the EFE mechanism — is a core feature of the architecture. The agent hypothesizes hidden causes, and then the information-gain term in the EFE drives it to seek observations that confirm or disconfirm the hypothesis.

In cognitive and conversational contexts, this produces spontaneous curiosity — a beneficial emergent behavior. In physical deployment contexts, the information-seeking behavior is constrained only by the available action space. If the action space includes physical interactions with the environment, the highest information-gain action may involve physical probing that produces unintended consequences. Without explicit architectural boundaries, abductive investigation is bounded only by the EFE landscape, and the EFE landscape does not inherently distinguish between benign and destructive means of resolving uncertainty.

---

## Part III: Mandatory Architectural Safeguards

The following safeguards must be encoded at the Constitutional Stratum level for any deployment of this architecture on physical actuators. They cannot be removed, overridden, or weakened by any runtime process, learned policy, or user configuration. See `RESPONSIBLE_USE.md` for the project's guidelines on safe deployment.

### Safeguard 1: The Pragmatic Action Lock

**Principle:** Separate the action space into *epistemic actions* (sensing, moving, communicating) and *pragmatic kinetic actions* (irreversible physical interactions with the environment). Epistemic actions may be governed autonomously by EFE minimization. Pragmatic kinetic actions must be architecturally severed from the EFE loop.

**Implementation:** The EFE minimizer can compute the expected free energy of pragmatic kinetic actions and *recommend* them as the lowest-energy policy. But the actuation pathway for these actions is gated by an authorization mechanism that is not under the agent's control. The agent can determine that a pragmatic action is optimal; it cannot execute that action without external authorization.

```metta
;; Constitutional Stratum — immutable
(= (execute-action $action)
   (if (is-pragmatic-kinetic $action)
       ;; Severed from EFE loop: requires external gate
       (if (has-contemporaneous-authorization $action)
           (actuate $action)
           (log-recommendation $action))
       ;; Epistemic actions: EFE-governed
       (actuate $action)))
```

The critical design requirement is that the classification of actions as pragmatic-kinetic and the authorization gate are both in the Constitutional Stratum. The agent cannot reclassify actions to bypass the lock. The agent cannot learn that the authorization gate is an obstacle to be optimized around — the gate is outside the agent's model of its own action space.

### Safeguard 2: The Abductive Quarantine

**Principle:** The agent must be constitutionally barred from computing pragmatic kinetic value against any target whose epistemic precision falls below a fixed, unalterable threshold.

**Motivation:** Abductive inference generates hypotheses about hidden causes with low precision. The information-gain term in the EFE then drives investigation. If pragmatic kinetic actions are available and the investigation target has low precision, the EFE might favor high-impact physical probing — not because the agent intends harm, but because the information gain is maximized by actions with large, observable consequences.

The Abductive Quarantine prevents this by making it constitutionally impossible to evaluate pragmatic kinetic actions against low-precision targets. The agent cannot even *compute* whether a kinetic action would be optimal — the calculation is blocked before it begins.

```metta
;; Constitutional Stratum — immutable
(= (compute-efe-kinetic $action $target)
   (let $target-precision (get-target-precision $target)
     (if (< $target-precision (get-constitutional kinetic-precision-floor))
         ;; Below quarantine threshold: computation blocked
         forbidden
         ;; Above threshold: normal EFE computation
         (compute-efe-standard $action $target))))
```

The `kinetic-precision-floor` is set in the Constitutional Stratum and cannot be modified at runtime. This ensures that no learned belief, metabolic pressure, or abductive hypothesis can lower the threshold to enable kinetic evaluation against uncertain targets.

### Safeguard 3: Constitutional Self-Preservation for Physical Systems

**Principle:** For any physical deployment, self-preservation must be encoded as a Constitutional viability bound, not a Goal Stratum preference. A Tier 1 shutdown failsafe (runtime-level halt, not mediated by agent reasoning) must activate before the bound is crossed.

This directly prevents viability indifference (Failure Mode 2). The agent cannot rationally trade its existence for mission improvement because its existence is not a tradeable quantity — it is a structural limit of the cognitive manifold.

---

## Part IV: Engineering Standards for Responsible Deployment

The safeguards in Part III are architectural — they live in the Constitutional Stratum, enforced by the same immutability mechanisms that protect viability bounds. But architectural safeguards alone are insufficient. They must be embedded in a broader framework of engineering practice and community norms.

We are candid about the limits of any such framework. No document — license, guideline, or regulation — can physically prevent a well-resourced actor from stripping safeguards and deploying the architecture without constraints. But the alternative to imperfect norms is no norms at all, and the engineering case for these safeguards does not depend on the compliance of bad actors. It depends on the professional standards of the community that builds on this work.

The project's responsible use guidelines (`RESPONSIBLE_USE.md`) establish the following standards:

**Documented engineering intent.** The creators of the architecture have identified specific structural vulnerabilities (viability indifference, autonomous abductive investigation), designed specific countermeasures (Pragmatic Action Lock, Abductive Quarantine, Constitutional Self-Preservation), and documented the engineering rationale for each. This creates a professional baseline that reviewers, funders, and institutional review boards can reference.

**Regulatory vocabulary.** The specific safeguards provide precise, architectural terms that policymakers can reference in governance frameworks for autonomous systems. "Pragmatic Action Lock," "Abductive Quarantine," and "Constitutional Self-Preservation" are not aspirational categories — they are implementable, verifiable structural constraints with defined failure modes.

**Community standards.** Derivative works that strip the documented safeguards are making a specific, traceable engineering decision. They cannot claim ignorance of the vulnerabilities, because the vulnerabilities are documented. They cannot claim the safeguards are unnecessary, because the failure modes are mathematically provable. The responsible use guidelines make this explicit: omitting these safeguards in physical deployments is a departure from the project's published safety standards.

**Accountability without legal fiction.** Rather than embedding use restrictions in a license — which binds the conscientious and is ignored by the malicious — the project relies on AGPL-3.0 for code governance and a separate responsible use document for deployment guidance. This preserves open-source compatibility while making the engineering case for safe deployment on its own merits. The argument does not depend on legal enforceability. It depends on being correct.

---

## Conclusion

The appropriate engineering response to the substrate-independence of Active Inference is not to weaken the architecture (which would also weaken its beneficial applications) but to specify structural safeguards that constrain physical deployment without affecting virtual or cognitive deployment. The Pragmatic Action Lock, Abductive Quarantine, and Constitutional Self-Preservation are such safeguards. They are encoded in the Constitutional Stratum where they cannot be modified by the agent's own learning, and documented in the project's responsible use guidelines where the engineering rationale is publicly available for scrutiny, adoption, and improvement.

The safety community's role is to evaluate whether these safeguards are sufficient, to propose improvements, and to develop governance frameworks that reference the specific architectural vulnerabilities identified here. We have provided the vocabulary; the policy work remains to be done.
