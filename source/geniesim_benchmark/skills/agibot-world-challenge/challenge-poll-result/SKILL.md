---
name: challenge-poll-result
description: Use to track a Simulation Challenge job's progress — list jobs, fetch a job's per-task scores, or pull execution logs when a job ended in Failed. Read-only; safe to run without confirmation.
---

# challenge-poll-result — Track job status, scores, and logs

All endpoints here are **read-only** — run them directly and report findings to the user.

## Preconditions

- `CHALLENGE_TOKEN` set (else → `challenge-login`).
- `JOB_ID` set (else → `challenge-submit-job`, or list jobs first to find one).

Every Bash call should start with:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
```

(See README "State file" — AI assistants spawn each command in a new subshell, so file-backed state is the only reliable handoff.)

## List my jobs

```bash
curl -fsS "$BASE_URL/api/challenge/jobs?page=1&per_page=20" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
```

Returns paginated `items` with `id`, `name`, `status`, `score`, `created_at`. Use it to recover a `JOB_ID` the user lost.

## Fetch a single job's result

```bash
curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/result" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
```

Response shape:

```json
{
  "status": "Finished",
  "tasks": {
    "pick_task":  { "score": [95.0, 92.0, 88.0], "total": 275.0 },
    "place_task": { "score": [100.0, 100.0, 98.0], "total": 298.0 }
  },
  "total": 573.0
}
```

`status` values:

| Status | Meaning | Next action |
|--------|---------|-------------|
| `Pending` / `Queued` | Submitted, awaiting scheduling | Make sure agents are online (`challenge-run-agent`). |
| `Running` | At least one Case is executing | Continue polling. |
| `Finished` | All Cases done | Stop polling. Suggest `challenge-ranking`. |
| `Failed` | Unrecoverable error | Pull `/log` (next section) and route the user to `challenge-troubleshoot`. |
| `Cancelled` | Cancelled | Stop polling. |

## Polling loop

Poll every **2–5 seconds** until terminal. Don't poll faster — the platform doesn't change state that quickly and you'll burn rate budget.

```bash
while true; do
  [ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
  RESULT=$(curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/result" \
    -H "Authorization: Bearer $CHALLENGE_TOKEN")
  STATUS=$(echo "$RESULT" | jq -r '.status // "?"')
  SCORE=$(echo "$RESULT"  | jq -r '.total  // "?"')
  printf "status=%-10s score=%s\n" "$STATUS" "$SCORE"
  case "$STATUS" in
    Finished|Failed|Cancelled) break ;;
  esac
  sleep 3
done
```

When you call this from an AI assistant, do not loop indefinitely without checking in — every ~30 seconds report progress (the latest `status` and `score`) so the user can interrupt if something looks wrong (e.g. stuck in `Pending` because no agent is connected).

## Fetch failed-job log

Only meaningful when `status == "Failed"`:

```bash
curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/log" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
```

Returns an `EmuCaseExecLog` (container stdout / stderr / exit code) for the latest failed case.

When the log is large, **don't dump everything**. Surface the load-bearing tail:

1. The final `exit_code` (always include).
2. The last `Traceback (most recent call last):` block, if present, in full — that's the actual failure.
3. The last ~20 lines of stderr around it for context.

A common failure pattern: a `Traceback` in `decode_observation` / `handle()` during a case that started with `frame_bytes=b''` is the **warmup empty-frame bug** — see `challenge-troubleshoot` row #11. It can surface as `Failed` mid-job, not only as "stuck in WARMUP".

## Hand-off

- `Finished` → suggest `challenge-ranking` to see where the score lands.
- `Failed` → pull `/log`, then `challenge-troubleshoot`.
- Stuck `Pending`/`Running` for more than a few minutes with no progress → `challenge-troubleshoot` (check agents are still connected; the **gateway** holds a 30 s reconnect window for any agent that drops, and `tunnel.sh`/the SDK redials automatically inside that window).
