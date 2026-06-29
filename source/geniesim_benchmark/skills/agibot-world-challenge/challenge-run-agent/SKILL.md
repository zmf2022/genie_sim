---
name: challenge-run-agent
description: Use when the contestant wants to launch the inference Agent that connects to the Simulation gateway via WebSocket. Wraps the official inference repo's ./scripts/tunnel.sh and scales to PARALLELISM processes. This is side-effecting (consumes GPUs and a parallelism slot) — confirm before launching.
---

# challenge-run-agent — Launch the inference Agent

The Agent is the contestant's long-running process that:

1. Reverse-connects to the gateway over WebSocket using `JOB_UUID` and a per-process `agent_id`.
2. Receives observation frames, runs inference, returns action bytes.
3. Stays online until the platform sends a `drain` control frame (job complete).

The contestant's **inference repo** (a separate repository, not this platform repo) ships a launcher: `./scripts/tunnel.sh <gpu_index> <job_uuid> <gateway_url>`. This script wraps the official Simulation Python SDK and is the recommended path.

If the user does not have the inference repo, the official SDK invocation from `../user-manual.md` §3.1 is the documented fallback:

```bash
cd contestant_sdk/python
uv run python simubotix_agent.py \
  --access-token "$CHALLENGE_TOKEN" \
  --job-uuid     "$JOB_UUID" \
  --gateway-url  "$TUNNEL_ENDPOINT" \
  --handler      example_inference:handle
```

Both paths obey the same argument contract — the rest of this skill assumes `tunnel.sh`; substitute the SDK call if needed.

## Preconditions

| Variable | Source |
|----------|--------|
| `CHALLENGE_TOKEN` | `challenge-login` (the SDK reads it from env if not passed) |
| `JOB_UUID` | `challenge-submit-job` |
| `PARALLELISM` | `challenge-submit-job` |
| `TUNNEL_ENDPOINT` | `challenge-submit-job` (job response), else fixed default `ws://120.92.88.78/api/challenge/tunnel` |
| inference repo cloned | user-side; `./scripts/tunnel.sh` must exist (or use the SDK fallback above) |

If any are missing, jump back to the producing skill instead of guessing.

Every Bash call in this skill should start with:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
```

(Cross-shell state lives in `~/.simubotix-challenge.env`; AI assistants spawn each command in a new subshell. See README "State file".)

## Concurrency rules (read these before launching)

- The platform enforces **at most `$PARALLELISM` concurrent agent processes** per user. Excess connections are closed immediately after the WS upgrade.
- Each tunnel carries **at most one active session at a time** (1:1 contract). To run cases in parallel, run multiple agent processes — that's the whole point of `parallelism`.
- Each process must use a **different GPU index** if launched on the same host. If the user's box has only `N` GPUs, `min(N, PARALLELISM)` is the real cap.
- Each process must use a **different `agent_id`**. `tunnel.sh` (and the SDK) auto-generate one — do not pass duplicates.

## Step 1 — Confirm the launch plan with the user

Before running anything, summarise:

> About to launch **K** agent processes against `JOB_UUID=…`, gateway `TUNNEL_ENDPOINT=…`. K must be ≤ PARALLELISM (`$PARALLELISM`) and ≤ available GPUs. Each process pins one GPU. Proceed?

Wait for explicit confirmation. Default K to `$PARALLELISM` only if the user said so — otherwise ask.

## Step 2 — Launch a single agent (sanity check)

```bash
cd <inference-repo>
./scripts/tunnel.sh 0 "$JOB_UUID" "$TUNNEL_ENDPOINT"
```

Argument order:

| Position | Value | Notes |
|----------|-------|-------|
| `$1` | GPU index | e.g. `0`. Must exist on the host. |
| `$2` | `$JOB_UUID` | UUID from `POST /api/challenge/job`. |
| `$3` | `$TUNNEL_ENDPOINT` | Full WS URL. Defaults to the fixed `ws://120.92.88.78/api/challenge/tunnel`; prefer the job response's `tunnel_endpoint` if it carries one. |

Verify the agent reaches the **WARMUP → RUNNING** lifecycle (the SDK logs this) before scaling out. If it disconnects right after handshake, see `challenge-troubleshoot`.

## Step 3 — Scale to PARALLELISM processes

First decide the real launch count `K = min(PARALLELISM, NUM_GPUS)`. Detect available GPUs (one of):

```bash
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
# fallback: if nvidia-smi is unavailable, ask the user how many GPUs to use.
K=$(( NUM_GPUS < PARALLELISM ? NUM_GPUS : PARALLELISM ))
echo "launching K=$K processes (parallelism=$PARALLELISM, gpus=$NUM_GPUS)"
```

If `K < PARALLELISM`, tell the user explicitly — they can either accept reduced concurrency or add more GPUs. Do **not** double-pin a GPU.

Map GPU `i` to agent process `i`:

```bash
PIDS=()
for i in $(seq 0 $((K - 1))); do
  ./scripts/tunnel.sh "$i" "$JOB_UUID" "$TUNNEL_ENDPOINT" &
  PIDS+=($!)
done
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT
wait
```

`trap` ensures `Ctrl-C` (exit 130) cleans up children. The `wait` at the end blocks the foreground shell until every agent exits — see "Driving from an AI assistant" below if you're orchestrating this from a non-interactive session.

### Driving from an AI assistant

The launch loop is **long-lived**: it stays alive until every process gets `drain` (exit 0) or the user hits Ctrl-C. If you're an AI assistant orchestrating this from a single shell:

- Tell the user to run the loop in a **dedicated terminal** (or under `tmux` / `nohup`), then come back to your session for `challenge-poll-result`.
- Do NOT background the whole loop with `&` and continue polling from the same shell — when the assistant's session ends, the agents go with it.

If you have to launch and poll from the same automation, run the loop with `nohup` redirected to a log file and treat its PID as opaque until poll reaches a terminal status.

## Lifecycle and exit codes

| Code | Meaning |
|------|---------|
| 0 | `drain` received → graceful shutdown. Job dispatch is finished; do **not** reconnect. |
| 1 | Bad arguments / handler load error / reconnect retries exhausted. |
| 130 | Ctrl-C. |

If a process dies abnormally, the SDK / `tunnel.sh` will redial **automatically** with the same `agent_id` within the **gateway-side 30-second reconnect window** to resume the open session. **Do not manually relaunch** within that window — you'd race the SDK's redial and the gateway will reject the second connection. Past 30 s, the open session is gone and the user must resubmit a new job (which costs **1/4 daily quota** — confirm before suggesting it).

## After launch

Cases queue and run as agents stay online. Hand off to `challenge-poll-result` to track status.

- **Don't kill agents** until the job reaches `Finished` / `Failed` / `Cancelled` — early termination loses in-flight cases.
- After the job hits `Finished`, the platform sends `drain` and the SDK exits 0 on its own. Wait for that natural exit; do not Ctrl-C preemptively.

## Failure shortlist

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Handshake `401` | Token / job mismatch | `challenge-login`; verify `JOB_UUID` belongs to this account. |
| Connects then drops | Per-user parallelism cap exceeded | Reduce K or wait for old agents to finish. |
| `drain` received | Job is wrapping up | Stop launching, wait for graceful exit. |
| Stuck in `WARMUP` | Inference handler errored on the warmup empty frame | Check the inference repo's logs. |

For more, jump to `challenge-troubleshoot`.
