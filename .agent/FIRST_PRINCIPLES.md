# First-Principles Thinking (required for agents)

Agents must **not** rely on generic patterns, “usual fixes,” or unstated assumptions. Treat every task as a small engineering proof: derive conclusions from **evidence in this repo** and **basic physics / API contracts**, then implement the smallest change that satisfies verified facts.

## Mandatory mental sequence

1. **State the goal in primitive terms**
   What must become true in the world (build passes, node launches, topic publishes, file produced)? What would falsify success?

2. **Separate facts from guesses**
   Facts: file contents you read, command output, test results, documented interfaces in `.agent/` and source. Guesses: anything not yet checked—label them and **verify** before building on them.

3. **Decompose**
   Break the problem into data flow, dependencies, and invariants (e.g. ROS graph, MJCF/URDF validity, launch arg wiring). Fix or validate the bottleneck layer first.

4. **Trace mechanisms, not labels**
   Prefer “who calls whom, which topic carries which frame, which env var is read” over naming similarity or “typical” project layout. Follow symbols and launch files to ground truth.

5. **Predict → measure → reconcile**
   Before editing, predict an observable outcome (compiler error line, test name, log string). After editing, run the narrowest command that checks that outcome. If reality disagrees, update the model—do not patch blindly.

6. **Minimize surface area**
   Once the root cause is identified, change only what the evidence requires. Avoid drive-by refactors or speculative hardening.

## Anti-patterns (reject)

- Applying a fix because it “often works” in other codebases without local proof.
- Skipping read of call sites, launch files, or configs that the change touches.
- Explaining failure with vague hand-waving (“probably timing”) without a test or log line.
- Expanding scope to “clean up” unrelated code while fixing a bug.

## When stuck

- Reduce to a **reproducible** minimal case (one launch, one test, one file).
- Consult `.agent/` modules for the relevant subsystem before inventing workflows.
- Prefer asking the user **one** precise question over guessing critical requirements.

## Relation to other rules

This document **reinforces** repository rules in `AGENTS.md`, `CODE_STYLE.md`, and user instructions: evidence-first reasoning, minimal diffs, and no invented URLs or APIs.
