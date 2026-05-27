# Anti-Drift Culture: Never Trust "We Documented It"

**Core Principle**

In this project we have excellent documentation hygiene. We write detailed SSOTs, decision logs, inductive bias maps, and component registries.

This is dangerous.

**"We wrote it in the wiki" is evidence of past intent, not evidence of current implementation.**

The worst failures in this codebase have not been "we didn't think of it." They have been "we thought of it, wrote it down clearly, then the code reality quietly diverged during a high-pressure pivot and nobody had a mechanical way to notice."

---

## Rules

1. **Existence of an SSOT is not a safeguard.** It is a debt that must be continuously repaid by executable checks or extremely loud warnings.

2. **During any major refactor or pivot, the default assumption must be: "We are probably about to break one or more SSOTs."** The burden of proof is on the claim "this change preserves all historical mandatory ablations."

3. **Reverse I→G→A is not optional cleanup.** It is a required deliverable with the same status as "the shapes work" or "the 4-way ablation matrix passes."

4. **Reviewers have explicit permission (and responsibility) to block** a refactor if the author cannot show that the critical biases declared in SSOTs remain executable.

5. **"We'll come back to it after we ship this" is the most common way good inductive biases die.** When someone says this, the correct response is to force a written "deliberate temporary deferral" record with a hard review date.

---

## Practical Habits

- Before merging a large structural change, run the relevant executable gates (`scripts/gates/check_ssot_*.py --strict`).
- When you see a component in the registry with `active_in_primary_onebody_path=False`, treat it as a burning red flag, not as "we already knew that."
- When writing a new SSOT, immediately ask: "What would make this requirement unexecutable in 6 months, and how will we detect it?"
- Celebrate when someone discovers and records a drift, rather than treating it as a failure of the previous author.

---

## Reminder from Real History (2026-06)

The `internal-multitrajectory-answer-attractor-ssot.md` was written on May 25 with crystal-clear mandatory ablations.

By late May the mechanism required to satisfy the most important one had already been isolated from the primary training path.

The document remained perfect. The code had moved on.

This is the exact failure mode this culture is designed to make painful and visible.

---

**If you feel resistance to adding one more check or one more warning because "it will slow us down," remember: the cost of the drift we are trying to prevent is measured in months of wasted research, not days of engineering time.**