# Agentic Closed-Loop Planning References

Date accessed: 2026-04-30.

Purpose:

This page records papers and official repositories relevant to turning QTRM
from a donor-backed memory/evidence adapter into a self-acting closed-loop
planner. The key lesson is conservative: current literature does not support
"just let the model think longer" as the main mechanism. Strong agent systems
separate policy, environment execution, feedback, trace memory, verifier
signals, and training/credit assignment.

Local papers:
`references/papers/agentic_closed_loop_planner/`

Local official repos:

| Repo | Commit | What To Extract |
| --- | --- | --- |
| `references/official/agent-lightning` | `0b40cb724a0a` | Disaggregate agent execution from training; convert traces into trainable transitions. |
| `references/official/ragen` | `20daedc47558` | Multi-turn agent RL environments, StarPO, collapse diagnostics. |
| `references/official/agentgym` | `c3b300f0381a` | Unified agent environments, trajectories, and self-evolution benchmark design. |
| `references/official/language-agent-tree-search` | `853d81614607` | MCTS-style search over language-agent actions with value/reflection. |
| `references/official/ace` | `4f679bef3b78` | Evolving context/playbook memory through generation, reflection, and curation. |
| `references/official/dynamic-cheatsheet` | `5cfe3c37e8e5` | Adaptive test-time memory and curated strategy sheets. |
| `references/official/skillrl` | `299909b2f5e2` | Recursive skill-augmented RL and skill memory promotion. |
| `references/official/aworld-rl` | `2082e70bcd54` | Agentic RL algorithms over multi-turn dynamic environments. |

## Source Map

| Source | Local PDF | Mechanism | QTRM Reading |
| --- | --- | --- | --- |
| [ReAct](https://arxiv.org/abs/2210.03629) | `react_2210.03629.pdf` | Interleave reasoning, actions, and observations. | Base trace grammar: `state -> action -> observation`. |
| [Reflexion](https://arxiv.org/abs/2303.11366) | `reflexion_2303.11366.pdf` | Verbal feedback memory after failed episodes. | Reflection memory must be typed separately from factual evidence. |
| [Voyager](https://arxiv.org/abs/2305.16291) | `voyager_2305.16291.pdf` | Skill library, execution feedback, continual exploration. | Promote only repeatedly verified MemoryOS/tool procedures into skills. |
| [Toolformer](https://arxiv.org/abs/2302.04761) | `toolformer_2302.04761.pdf` | Self-supervised API/tool-call use. | Tool-use labels can be generated, filtered, then distilled. |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | `tree_of_thoughts_2305.10601.pdf` | Search over multiple reasoning branches. | Useful for planner search, but expensive without value pruning. |
| [RAP](https://arxiv.org/abs/2305.14992) | `rap_2305.14992.pdf` | Reasoning as planning with world-model style state transitions. | Aligns with QTRM core-world-model planning, but needs executable feedback. |
| [LATS](https://arxiv.org/abs/2310.04406) | `lats_2310.04406.pdf` | MCTS + LM value + reflection + environment feedback. | Best direct prior for search over QTRM actions. |
| [AgentGym](https://arxiv.org/abs/2406.04151) | `agentgym_2406.04151.pdf` | Diverse agent environments and AgentEvol. | Need multiple environments, not one synthetic MemoryOS task type. |
| [RAGEN](https://arxiv.org/abs/2504.20073) | `ragen_2504.20073.pdf` | StarPO trajectory-level agent RL. | Multi-turn RL needs stable rollouts, critic/filters, and reasoning-aware rewards. |
| [Agent Lightning](https://arxiv.org/abs/2508.03680) | `agent_lightning_2508.03680.pdf` | Markov trace interface and credit assignment over existing agents. | Most compatible path: keep MemoryOS harness external, train from logged transitions. |
| [ACE](https://arxiv.org/abs/2510.04618) | `ace_2510.04618.pdf` | Evolving contexts as curated playbooks. | Store learned strategies outside factual memory; feed compact playbooks back in. |
| [SAGE](https://arxiv.org/abs/2512.17102) | `sage_skill_library_2512.17102.pdf` | Skill-integrated GRPO over chained tasks. | Skills should be reward-aware, not just text snippets. |
| [SkillRL](https://arxiv.org/abs/2602.08234) | `skillrl_2602.08234.pdf` | Recursive skill-augmented agent RL. | Candidate for MemoryOS skill promotion and reuse loop. |
| [ASTER](https://arxiv.org/abs/2602.01204) | `aster_tool_integrated_reasoning_2602.01204.pdf` | Avoids interaction collapse with interaction-dense cold starts. | QTRM must learn to keep using tools/evidence, not retreat into text-only thought. |
| [RAGEN-2](https://arxiv.org/abs/2604.06268) | `ragen2_reasoning_collapse_2604.06268.pdf` | Template collapse diagnosis via MI proxies and SNR-aware filtering. | Add input-dependence diagnostics; entropy alone is insufficient. |
| [Credit Assignment Survey](https://arxiv.org/abs/2604.09459) | `agentic_rl_credit_assignment_2604.09459.pdf` | Token/segment/turn/multi-agent credit assignment taxonomy. | Use turn-level rewards first; avoid episode-only reward for long traces. |
| [Agent-World](https://arxiv.org/abs/2604.18292) | `agent_world_2604.18292.pdf` | Environment/task discovery plus continuous self-evolving RL arena. | Long-term target: dynamic tasks that expose QTRM weakness, then retrain. |
| [DEVS World Models](https://arxiv.org/abs/2603.03784) | `devs_world_models_2603.03784.pdf` | Executable discrete-event world models from natural-language specs. | For tool workflows, explicit simulators are safer than purely neural imagination. |
| [Agent2World](https://arxiv.org/abs/2512.22336) | `agent2world_2512.22336.pdf` | Generate symbolic world models with research/developer/tester agents. | Use verifier-tested simulators as training environments for planner rollouts. |

## Design Lessons

1. Closed-loop agency is a system property.
   The loop needs an environment/runtime that executes actions and returns
   observations. QTRM alone, in one forward pass, is not a closed-loop agent.

2. Trace format is the first architecture.
   Agent Lightning, RAGEN, AgentGym, and Agent-World all depend on structured
   traces. QTRM needs a durable trace schema before serious RL.

3. Reflection is not evidence.
   Reflexion, Voyager, ACE, and Dynamic Cheatsheet are useful, but their memory
   products are strategies or critiques. They must not be mixed with source
   evidence in MemoryOS.

4. Agent RL has collapse modes.
   RAGEN/RAGEN-2/ASTER warn about echo traps, template collapse, and interaction
   collapse. QTRM should track reward variance, input-dependence, repeated
   action templates, and tool/evidence interaction density.

5. World models should be verifiable.
   LeWorldModel-style latent prediction can help, but closed-loop tool planning
   also benefits from explicit executable simulators or environment mocks.
   Agent2World and DEVS-style generation are relevant when real tools are too
   expensive or risky for exploration.

## Architecture Candidates For QTRM

### Candidate A: Trace-First External Closed Loop

Limitation solved:
Current QTRM has no learned or executable action loop.

Architecture change:
Add an `AgentHarness` around the existing donor/QTRM model:

```text
task
-> QTRM proposes action
-> MemoryOS/tool runtime executes action
-> verifier scores observation/result
-> trace store records transition
-> loop continues or final answer is emitted
```

Training/eval change:
Start with SFT/preference over logged good/bad traces, not full RL.

Why this is first:
It gives real data for future controller heads and RL without changing model
internals prematurely.

### Candidate B: Learned Action Controller

Limitation solved:
`ControllerHeads` currently exist but are not a trained action policy.

Architecture change:
Train discrete action logits such as:

```text
OBSERVE, RETRIEVE_MEMORY, SEARCH_WEB, VERIFY_EVIDENCE,
WRITE_MEMORY, WRITE_SKILL, SIMULATE, ANSWER, STOP
```

Training/eval change:
Supervise from successful traces, then add turn-level preference/RL.

Main risk:
With weak trace data, the controller learns shallow templates.

### Candidate C: Model-Based Latent Planner

Limitation solved:
Pure external loops can be slow; QTRM should eventually predict action
consequences before executing.

Architecture change:
Use the existing core-world-model head as an action-conditioned rollout model:

```text
latent state z_t + candidate action a_t
-> predicted z_{t+1}
-> evidence/value/verifier heads
-> pick action or request real execution
```

Training/eval change:
Compare imagined next states against real post-action latent states. Accept
only if imagined rollouts improve action choice under ablation.

Main risk:
Neural imagination can hallucinate. High-risk actions still need real verifier
or executable simulator feedback.

## Recommended Order

1. Build Candidate A first: trace schema, closed-loop harness, deterministic
   MemoryOS/tool actions, verifier rewards, and replayable logs.
2. Train Candidate B from successful/failed traces with action-density and
   input-dependence diagnostics.
3. Add Candidate C only after real traces exist, so the LeWM-style core world
   model predicts actual transitions rather than synthetic labels.

Acceptance gates:

- trace replay reproduces actions, observations, and verifier outcomes;
- action controller beats fixed `RETRIEVE -> VERIFY -> ANSWER` scripts;
- disabling MemoryOS/tools or verifier reward reduces task success;
- MI/input-dependence proxy does not collapse while entropy looks stable;
- tool/evidence interaction density stays above a minimum threshold;
- final answers improve over donor-only and non-agentic QTRM baselines.
