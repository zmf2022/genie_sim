---
name: challenge-submit-job
description: Use when the contestant wants to submit a model evaluation job to the Simulation Challenge — POST /api/challenge/job. This is a quota-consuming, side-effecting action; always confirm with the user first. Captures JOB_UUID, PARALLELISM, TUNNEL_ENDPOINT for the rest of the pipeline.
---

# challenge-submit-job — Create a model evaluation job

Submitting a job (a) **counts against the daily quota of 4 submissions per user** (test accounts are exempt) and (b) immediately starts incurring scheduling work on the platform. Treat this as side-effecting: **always confirm before running the POST**.

## Preconditions

- `CHALLENGE_TOKEN` must be set (else → `challenge-login`).
- `BASE_URL` defaults to the fixed `http://120.92.88.78` (override only if explicitly told).

Every Bash call in this skill should start with:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
```

(Cross-shell state lives in `~/.simubotix-challenge.env`; AI assistants spawn each command in a new subshell. See README "State file".)

## Step 1 — Probe the daily quota (mandatory)

Always run this **immediately before** every POST, even if you submitted a job earlier in the same session — there's no in-session counter you can trust, and a shared account may have been used elsewhere.

Two endpoints work:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env

# Preferred: structured remaining count
curl -fsS "$BASE_URL/api/challenge/submission/quota" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
# { "limit": 4, "used": 1, "remaining": 3 }

# Legacy boolean check (still works):
curl -fsS "$BASE_URL/api/challenge/model/upload/check" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN"
# { "status": "ok" }  → quota remains
```

If `remaining == 0` (or `upload/check` errors with an "upload limit" message), **stop**: the user has used today's four slots. The daily window resets at Beijing midnight (UTC+8). Suggest they retry tomorrow or wait. Do not proceed to Step 2.

## Step 2 — Build the request body

The request body has a **top-level `name`** plus a `config` object. `model_path` is **no longer required** — the platform resolves the model from `model_name`.

| Field | Where | Required | Notes |
|-------|-------|----------|-------|
| `name` | top-level | yes | Job name shown in the contestant's job list. |
| `config.board` | `config` | yes | **Board short-id (single value). Allowed values: `instruction`, `spatial`, `manip`, `robust`.** |
| `config.model_name` | `config` | yes | Model identifier. |
| `config.description` | `config` | no | Free-form text. |
| `config.paper_link` | `config` | no | Optional URL to a paper/arxiv page describing the model. Must be a valid URL, ≤ 512 chars. Used for audit/leaderboard display. |

> **Allowed-value enforcement.** If the user supplies a board name other than `instruction` / `spatial` / `manip` / `robust`, **stop and ask** — don't POST. Submitting with an unknown board will return `400 invalid board` / `400 no task templates for board` and burn 1/4 daily quota. If they say "use the new one X" / "I heard the platform added Y", verify with the organizers or with `../user-manual.md` / `../quickstart.md` before proceeding — this skill is the single source of truth for what's currently accepted.

> **One submission = one board = one job.** To evaluate multiple boards, submit once per board; each call consumes one slot of the **4 daily submissions per user**.

Read each missing required field back to the user. If `board` is omitted, default to `"instruction"` (the other accepted values are `spatial`, `manip`, `robust`). Say the defaults explicitly so the user can object.

## Step 3 — Confirm, then POST

Show the assembled body, the daily-quota cost (1/4), and ask for explicit "yes" before running:

> **Overwrite check (single-file state).** If `~/.simubotix-challenge.env` already has `JOB_ID` set from a previous submission, this POST will overwrite the `JOB_*` vars and the previous job's IDs will be lost from the state file. Before POSTing, surface this and check whether the previous job is terminal:
>
> ```bash
> [ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
> if [ -n "$JOB_ID" ]; then
>   PREV_STATUS=$(curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/result" \
>     -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq -r '.status // "?"')
>   echo "previous JOB_ID=$JOB_ID status=$PREV_STATUS — submitting will overwrite the state file"
> fi
> ```
>
> - If `PREV_STATUS` is `Finished` / `Failed` / `Cancelled`, just confirm with the user that they're OK losing the IDs (job history is still queryable via `GET /api/challenge/jobs`) and proceed.
> - If `PREV_STATUS` is `Pending` / `Running`, **stop and warn**: overwriting means losing the handle to a still-running job. Ask the user explicitly whether they want to (a) wait for it via `challenge-poll-result`, (b) record `JOB_ID=$JOB_ID` / `JOB_UUID=$JOB_UUID` themselves before continuing, or (c) proceed knowing they'll need to find the job by name in `GET /api/challenge/jobs` later.

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
JOB_RESP=$(curl -fsS -X POST "$BASE_URL/api/challenge/job" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" \
  -d '{
    "name": "challenge_test",
    "config": {
      "board":        "instruction",
      "model_name":   "test",
      "description":  "challenge_test"
    }
  }')

echo "$JOB_RESP" | jq
```

A 400 with `invalid board` or `no task templates for board` means the board is wrong — go back to Step 2, do not retry.

## Step 4 — Capture the response

The response is a single job descriptor. These variables are the contract handed to every later skill. **Persist all four** to `~/.simubotix-challenge.env` so they survive across new Bash subshells:

```bash
JOB_ID=$(echo          "$JOB_RESP" | jq -r '.id')
JOB_UUID=$(echo        "$JOB_RESP" | jq -r '.uuid')
PARALLELISM=$(echo     "$JOB_RESP" | jq -r '.parallelism')
TUNNEL_ENDPOINT=$(echo "$JOB_RESP" | jq -r '.tunnel_endpoint')

challenge_save_var JOB_ID           "$JOB_ID"
challenge_save_var JOB_UUID         "$JOB_UUID"
challenge_save_var PARALLELISM      "$PARALLELISM"
challenge_save_var TUNNEL_ENDPOINT  "$TUNNEL_ENDPOINT"

echo "persisted: job_id=$JOB_ID uuid=$JOB_UUID parallelism=$PARALLELISM endpoint=$TUNNEL_ENDPOINT"
```

(`challenge_save_var` is the helper defined in `challenge-login` Step 1 / README. If it isn't loaded, re-paste it once per shell.)

> **Multiple boards**: a single submission accepts exactly one `board`. To evaluate `instruction` + `manip`, submit twice — each call gets its own job (and uses 1 of the 4 daily quota slots). Each job's `uuid` / `tunnel_endpoint` is independent; launch a separate agent process (or process group, up to that job's `parallelism`) per job.

Sanity-check before handing off:

- `JOB_UUID` matches a UUIDv4-shape string.
- `PARALLELISM` is an integer ≥ 1. **`0` means the user's concurrency cap is misconfigured** (not "exhausted" — exhaustion is a runtime live-connection check at the gateway, not a response field). Stop and surface to the organizers; do not launch agents.
- `TUNNEL_ENDPOINT` starts with `ws://` or `wss://`. **If empty, do NOT hard-code a URL.** Retry `GET /api/challenge/tunnel/endpoint` after a few seconds; an empty string means the gateway is not yet ready.

## Step 5 — Hand off

Tell the user:

> Job `$JOB_ID` (`$JOB_UUID`) is **READY**. The platform allows up to `$PARALLELISM` concurrent agent processes for this job. Cases will sit in queue until at least one agent is connected. Run `challenge-run-agent` next to launch the inference SDK.

Also remind them: the next step needs `./scripts/tunnel.sh` from the **inference repo** (separate from this platform repo). If they haven't cloned it yet, this is the moment to do so — otherwise their cases will sit in the queue with no agent attached.

Do NOT auto-launch the agent. The user controls when GPUs start spinning.

## Common errors

| Error from POST | Likely cause | Fix |
|-----------------|--------------|-----|
| 400 `invalid board` / `no task templates for board` | `board` not in `instruction`/`spatial`/`manip`/`robust` | Verify board with organizers; do not retry blindly (each retry burns 1/4 quota). |
| 400 `paper_link is not a valid URL` / `paper_link too long` | Optional `paper_link` field malformed | Fix the URL (≤ 512 chars, must parse as a URL) or omit it. |
| `upload limit` error / quota `remaining == 0` | Daily 4-submissions cap hit | Wait until Beijing midnight (UTC+8). |
| 401 | Token expired/invalid | `challenge-login` → refresh. |
