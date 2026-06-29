---
name: challenge-help
description: Entry point for the Simulation Challenge skill set. Use when the user mentions the challenge, leaderboard, submitting a model, or any of the /api/challenge/* endpoints — this skill picks the right downstream skill for them.
---

# challenge-help — Pick the right skill

You are an AI assistant helping a contestant operate the **Simulation Challenge** platform end to end. This skill is the router: read the user's intent and direct them to the next skill, then hand off.

## Stage map

```
   ── prep (before any job) ──────────────────────────────────────
   ┌──────────────────────────┐     ┌──────────────────────────┐
   │ challenge-download-      │     │ challenge-baseline-       │
   │     datasets             │ ──▶ │     model                 │
   │ LeRobot v2.1 training    │     │ clone code + ModelScope   │
   │ data from ModelScope     │     │ ckpts → install → run     │
   └──────────────────────────┘     └────────────┬─────────────┘
   ───────────────────────────────────────────────┼──────────────
                                                   ▼
        ┌─────────────────┐
        │ challenge-login │  get/refresh CHALLENGE_TOKEN
        └────────┬────────┘
                 ▼
       ┌────────────────────┐
       │ challenge-submit-  │  POST /api/challenge/job  →  JOB_UUID, PARALLELISM, TUNNEL_ENDPOINT
       │       job          │  (counts against the 4-submissions-per-day quota!)
       └────────┬───────────┘
                ▼
        ┌────────────────────┐
        │ challenge-run-     │  ./scripts/tunnel.sh <gpu> <job_uuid> <endpoint>
        │     agent          │  scale to PARALLELISM processes
        └────────┬───────────┘
                 ▼
       ┌────────────────────┐         ┌──────────────────────┐
       │ challenge-poll-    │ ──────▶ │ challenge-           │
       │     result         │         │     troubleshoot     │  ← if Failed / stuck
       └────────┬───────────┘         └──────────────────────┘
                ▼
        ┌────────────────────┐
        │ challenge-ranking  │  best-score + leaderboard
        └────────────────────┘
```

## Routing rules

| User says / intends | Use this skill |
|---------------------|----------------|
| "下载训练数据/数据集", "download training data", "get the lerobot v2.1 data", "download task suite", mentions `download_dataset.sh` / GenieSim3.0-Dataset | `challenge-download-datasets` |
| "拉取/搭一个 baseline", "下载推理代码/权重", "clone the inference repo", "download ckpts", "set up/run the baseline model", "部署/跑个 demo" | `challenge-baseline-model` |
| "log in", "I have credentials", "token expired" | `challenge-login` |
| "submit my model", "create a job", "what's my quota" | `challenge-submit-job` |
| "start the agent", "run my SDK", "use that GPU", "launch inference" | `challenge-run-agent` |
| "is my job done", "check the score", "what's the status of $JOB_ID" | `challenge-poll-result` |
| "where am I on the leaderboard", "best score" | `challenge-ranking` |
| "agent disconnected", "401", "stuck in Pending", "Failed why" | `challenge-troubleshoot` |
| End-to-end "submit and run" — `challenge-submit-job` → `challenge-run-agent` → `challenge-poll-result` | daisy-chain in order |

## Hard rules

1. **Always confirm before** `challenge-submit-job` (each submission burns 1/4 daily quota) and before launching agent processes (they hold a `parallelism` slot).
2. **Read-only calls run without confirmation**: login, current-user-info, jobs/result/log, best-score, ranking, quota check.
3. **The gateway host is fixed at `120.92.88.78` and does not change.** Defaults: `BASE_URL=http://120.92.88.78` (HTTP API base for all `curl "$BASE_URL/api/challenge/..."` calls) and `TUNNEL_ENDPOINT=ws://120.92.88.78/api/challenge/tunnel` (WebSocket tunnel for `tunnel.sh`/run-agent). If the job response carries a `tunnel_endpoint`, prefer it; otherwise fall back to the default above — do not block on it being absent.
4. **`board` has a closed allowed-value set today.** `config.board` accepts exactly one of `instruction` / `spatial` / `manip` / `robust`. Any other value gets a 400 and burns quota. If the user proposes something else, stop and verify against `challenge-submit-job` Step 2 / `../user-manual.md` before POSTing. Do not invent.
5. If `CHALLENGE_TOKEN` is missing or returns `401`, jump to `challenge-login` first.
6. **Auto-pilot does not skip confirmation gates.** If the user says "全程做完" / "do it end to end", you may merge the gates into a single up-front confirmation ("I'll submit job + launch K agents + poll until terminal — OK?"), but you may not run them silently. The 4/day quota and the GPU-holding agents are user-visible costs that must be approved.
7. **4xx is a semantic rejection, never a network blip.** Do not retry a 400/401/403/404 just because the user says "可能是网络抖动" — only connection resets / 5xx / timeouts qualify as transient. Refer to `challenge-troubleshoot` for the exact response.

## Environment contract

These environment variables are the shared state between skills. Treat them as the single source of truth — never re-derive.

**They are persisted to `~/.simubotix-challenge.env` (mode 0600).** Each Bash call should start with `[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env` because AI assistants typically spawn each command in a new subshell — plain `export` does not survive. See README "State file" for the helper and rationale.

| Variable | Producer | Consumers |
|----------|----------|-----------|
| `BASE_URL` | fixed default `http://120.92.88.78` (override only if explicitly told) | all |
| `CHALLENGE_TOKEN` | `challenge-login` (writes state file) | all subsequent |
| `JOB_ID` | `challenge-submit-job` (writes state file) | `challenge-poll-result` |
| `JOB_UUID` | `challenge-submit-job` (writes state file) | `challenge-run-agent` |
| `PARALLELISM` | `challenge-submit-job` (writes state file) | `challenge-run-agent` |
| `TUNNEL_ENDPOINT` | `challenge-submit-job` (job response), else fixed default `ws://120.92.88.78/api/challenge/tunnel` | `challenge-run-agent` |

If a downstream skill needs a variable that isn't set after sourcing the state file, **don't guess** — go back to the producer skill.
