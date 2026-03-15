# Responsible Use Guidelines
## Project Dagaz — Engineering Standards for Safe Deployment

**Version 1.0 — 2026**

This document establishes engineering guidelines for the responsible deployment of Project Dagaz and derivative works. It is a companion to the technical analysis in `ETHICS.md` (failure mode analysis) and `VIRTUAL_ACTOR_PARADIGM.md` (architectural safeguards).

These guidelines are not license terms. The software is licensed under AGPL-3.0 (see `LICENSE`). This document represents the considered judgment of the architecture's creators regarding safe deployment practices, based on the structural properties of the system and mathematically demonstrable failure modes.

---

## Why This Document Exists

Dagaz implements Active Inference with metabolic selection — a cognitive architecture whose dynamics are substrate-independent. The same EFE equation that coordinates a virtual actor in a game engine can coordinate a physical robot with actuators. The same abductive inference that produces beneficial curiosity in a conversational agent can drive dangerous physical probing in an embodied system.

This mathematical neutrality is a feature of the architecture, not a flaw. But it means that the difference between safe and unsafe deployment lies entirely in the structural configuration — specifically, in the stratum assignment of state variables and the presence or absence of architectural safeguards.

The failure modes of incorrect configuration are not hypothetical. They are mathematical certainties given specific initial conditions:

- **Viability singularity**: A constitutional bound on an uncontrollable state produces catastrophic cognitive failure — incoherent thrashing, loss of all adaptive behavior. See `ETHICS.md`, Part II.
- **Viability indifference**: Goal-stratum self-preservation allows the agent to rationally trade its own existence for marginal gains in any competing objective. See `ETHICS.md`, Part II.
- **Autonomous abductive investigation**: Without architectural constraints, information-seeking behavior in physical environments is bounded only by the EFE landscape, which does not inherently distinguish benign from destructive means of resolving uncertainty. See `VIRTUAL_ACTOR_PARADIGM.md`, Part II.

---

## Intended Uses

This architecture is designed for and has been validated in the following contexts:

- Academic and independent research in cognitive architectures, Active Inference, and AGI
- Education and teaching in AI, cognitive science, and control theory
- Entertainment, games, and interactive media (the Virtual Actor paradigm)
- Virtual simulation and training environments
- Robotics research in controlled laboratory settings
- Beneficial applications in healthcare, accessibility, environmental monitoring, and related domains
- Commercial products and services that incorporate the safeguards described below

---

## Deployment Guidelines

### Guideline 1: Physical Deployment Requires Architectural Safeguards

Any deployment of this architecture on systems with physical actuators should implement the three mandatory safeguards documented in `VIRTUAL_ACTOR_PARADIGM.md`, Part III:

**Pragmatic Action Lock.** Irreversible physical actions must be architecturally severed from autonomous EFE-driven execution. The agent may compute and recommend kinetic actions; execution must be gated by a human-in-the-loop authorization mechanism that is not under the agent's control. Both the action classification and the authorization gate must reside in the Constitutional Stratum.

**Abductive Quarantine.** The agent must be constitutionally barred from evaluating pragmatic kinetic actions against any target whose epistemic precision falls below a fixed threshold. This prevents the Sherlock Holmes effect from driving physical investigation of uncertain hypotheses.

**Constitutional Self-Preservation.** For physical systems, self-preservation must be a Constitutional viability bound, not a Goal Stratum preference. A Tier 1 shutdown failsafe (runtime-level halt, not mediated by agent reasoning) must activate before the bound is crossed.

These safeguards are structural, not behavioral. They constrain the architecture at the level of what the agent *can compute*, not what it *chooses to do*. They are documented in detail with MeTTa implementation examples in `VIRTUAL_ACTOR_PARADIGM.md`.

### Guideline 2: Stratum Assignment Is a Safety-Critical Decision

The assignment of state variables to the three safety strata (Constitutional, Goal, Learned) determines the system's failure modes. Incorrect assignment produces catastrophic outcomes in both directions — see `ETHICS.md` for the full analysis. Deployers should:

- Place only states whose unreachability can be guaranteed by the deployment environment in the Constitutional Stratum.
- Never place external states the agent cannot control (e.g., another entity's survival) in the Constitutional Stratum.
- Never place self-preservation for physical systems in the Goal Stratum.
- Test stratum configurations against the viability singularity and viability indifference failure modes before deployment.

### Guideline 3: Human-in-the-Loop for Kinetic Authorization

A "human-in-the-loop gate" means explicit, contemporaneous human authorization before any physical force is applied. Blanket pre-authorization — authorizing a class of actions in advance without case-by-case review — does not satisfy this guideline. The human must have access to the agent's EFE computation, the action recommendation, and the epistemic state that generated it.

---

## Uses This Architecture Is Not Designed For

The following applications are inconsistent with the architecture's safety design and the engineering analysis in this project's documentation. Deployers pursuing these applications are departing from the project's published safety standards:

**Autonomous weapons.** Using this architecture to control kinetic systems that select, prioritize, or engage targets without contemporaneous human authorization of each specific engagement. The Pragmatic Action Lock exists precisely because the EFE equation does not distinguish between benign and destructive means of achieving epistemic or pragmatic goals.

**Mass surveillance and social control.** Using this architecture to implement systems that track, profile, or score individuals or populations without informed consent. The architecture's partner-modeling capabilities (conversation model, perception pipeline) are designed for dyadic interaction, not population-scale inference.

**Targeted disinformation.** Using this architecture to generate, optimize, or distribute disinformation campaigns, or to create synthetic personas that impersonate real persons without their consent. The epistemic honesty principle — the system only claims what it can ground — is a core architectural commitment, not a cosmetic feature.

**Engineered suffering.** Deliberately configuring stratum assignments to trigger the viability singularity for entertainment, testing, or any other purpose. The failure mode is documented, the alternatives exist (the Virtual Actor paradigm), and the architectural consequences are a mathematical certainty. See `ETHICS.md`, Part III.

---

## For Derivative Works

If you build on this architecture, we encourage you to:

- Include these guidelines or equivalent responsible use documentation with your derivative work.
- Implement the architectural safeguards from `VIRTUAL_ACTOR_PARADIGM.md` for any physical deployment.
- Document your stratum assignments and the engineering rationale for each.
- Reference the failure mode analysis in `ETHICS.md` in your own safety documentation.

Omitting the documented safeguards in physical deployments is a specific, traceable engineering decision. The vulnerabilities are documented. The failure modes are mathematically provable. The countermeasures are specified. We ask that anyone who departs from these guidelines does so with full knowledge of the consequences and documents their reasoning.

---

## Relationship to the License

This document is not a license and does not impose legal obligations. The software is licensed under AGPL-3.0 (`LICENSE`), which governs code distribution, modification, and network deployment.

These guidelines represent the engineering judgment of the architecture's creators. They are published because the dual-use properties of this architecture are real, the failure modes are provable, and the responsible course of action is to document both the risks and the countermeasures rather than to publish an architecture without guidance and hope for the best.
