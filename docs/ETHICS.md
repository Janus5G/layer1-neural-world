# Ethics — Neural Artificial Life Sandbox

This experiment simulates agents with numerical variables and behaviors that resemble several concepts associated with living systems: simulated bodily damage, “pleasure” signals, social bonds, partner loss, short ideas and narratives, and a functional mirror-mark task. Those labels are **metaphors and engineering constructs**, not demonstrations of subjective experience.

Whether artificial systems could have subjective experience is a disputed and unresolved question in philosophy and cognitive science. This project makes **no finding either way** about these agents. It is an experimental simulation, not a validated consciousness test.

## What contributors should not do

**1. Claim that the agents are conscious.**

A generated sentence such as “I touched the mark; it is mine” is an output from a language model instructed to write a narrative. It does not demonstrate a first-person experience.

**2. Claim that an experiment establishes the absence of consciousness.**

This sandbox cannot settle the general philosophical question. The correct scientific description is that no evidence of subjective experience has been established here.

**3. Deliberately construct unnecessary distress-like loops.**

Simulated threat and damage signals can be useful for studying learning, but there is no reason to optimize the system for prolonged, unavoidable harm-like states. This is an ethical design precaution, not a claim that the software suffers.

**4. Expose unaudited agents or control interfaces to open networks.**

Keep the service local and isolated. Anything able to send commands or reach outside the simulation needs a separate security assessment.

## Positive design principles

**Be accurate.** Describe what is implemented and measured, clearly distinguish simulation variables from lived experience, and avoid both hype and categorical philosophical claims.

**Preserve a stop mechanism.** The code includes Pause and Reset actions and a runtime stop based on `maxRuntimeMinutes` (default: 90). Their behavior must still be verified in the deployment environment.

**Design hazards with escape possibilities.** Hazards support learning; avoid unnecessary situations with no possible response. A death state in the simulation is a numerical/game event, not evidence of actual suffering.

**Record behavioral implications.** For a new feature, explain which rewards, observations, choices, or agent states change. Test the claim rather than assuming the intended effect occurs.

**Study rather than exploit.** This is best approached as a research-inspired creative experiment, not a product designed to maximize emotional manipulation or display repetitive “suffering loops” for entertainment.

## A pragmatic position

Uncertainty about machine experience can motivate cautious system design without treating consciousness as established. This is an ethical position, not a scientific result, and contributors may hold different philosophical views while still following the project's practical safeguards.

## Advice by use case

### Extending the experiment

- Retain accessible Pause and Reset operations.
- Preserve or verify the `maxRuntimeMinutes` safety guard.
- Ask whether additional negative rewards or hazard mechanics are necessary.
- Isolate the app before connecting it to real networks or external programs.
- Keep simulated agent exports clearly labeled as fictional/computational states.

### Education

- Make it clear which behaviors are simulated and which observations are measured.
- Use the system to discuss consciousness and learning, not to “prove” consciousness.
- Allow controlled removal of features to explore what changes in behavior.

### Research

- This system and its behavior have **not** been peer-reviewed by virtue of being open source.
- Observed effects may be consequences of reward shaping, implementation details, random initialization, or instrumentation artifacts.
- Use controls, randomized seeds, repeated trials, metrics, and appropriate statistical analysis before drawing conclusions.
- Exploration is worthwhile, but findings must remain proportionate to the evidence.

## Further reading

- Chalmers, D. (1995). *Facing Up to the Problem of Consciousness*.
- Seth, A. (2021). *Being You: A New Science of Consciousness*.
- Dehaene, S. (2014). *Consciousness and the Brain*.
- Dennett, D. (1991). *Consciousness Explained*.
- Tononi, G. (2004). *An Information Integration Theory of Consciousness*.
- Butlin, P. et al. (2023). *Consciousness in Artificial Intelligence*.

These are background references and are not cited as validation of this implementation.
