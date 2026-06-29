---
name: challenge-troubleshoot
description: Use when something in the Simulation Challenge pipeline is misbehaving — auth errors, agent disconnects, jobs stuck in Pending, jobs ending in Failed, drain frames. Maps a symptom to its likely cause and the next command to run.
---

# challenge-troubleshoot — Symptom → cause → next command

When the user reports a problem, do this in order:

1. **Reproduce / verify the symptom** with a read-only call (`/result`, `/log`, `/current-user-info`) before doing anything destructive.
2. Match the symptom to the table below.
3. Hand the user the next command from the **Action** column. Do NOT auto-resubmit jobs (quota cost) or auto-kill agents (loses in-flight cases) without explicit confirmation.

## Diagnostic heuristic: 4xx is never a network blip

A structured `{"status":"error", ...}` body with HTTP **4xx** (400/401/403/404) is a **semantic rejection** by the platform — the request reached the server and was refused on its merits. Retrying it without changing the request gets the same answer and, for `POST /api/challenge/job`, **burns 1/4 daily quota each time**.

Only these qualify as transient and may be retried with backoff:

- TCP connection reset / timeout / DNS failure (curl exits non-zero before getting a response)
- HTTP `5xx` (server-side, signals a transient backend failure)

If the user says "可能是网络抖动 / maybe a network blip" but you have a 4xx body in hand, push back: name the actual error and point at the table row.

## Diagnostic heuristic: 5xx is often a missing / wrong token, not a real platform outage

`curl -fsS` collapses any HTTP 5xx to a terse `(22) The requested URL returned error: 5XX` and hides the body. Before concluding the platform is down, check this in order:

1. **State file sourced?** Each Bash call should start with `[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env` — AI assistants spawn every command in a new subshell, so plain `export` from a previous call is gone.
2. **Token presence after sourcing.** `echo "len=${#CHALLENGE_TOKEN}"` — if it's 0, either the file doesn't exist (re-run `challenge-login` Step 1) or the file exists but doesn't contain the key (the previous login silently failed — see `challenge-login` Step 1's status check).
3. **Token shape.** Should be a 3-segment JWT (two dots). `echo "$CHALLENGE_TOKEN" | head -c 20; echo` — gibberish or `null` means a previous step persisted a bad value.
4. **Body of the actual response.** Drop `-f` so curl prints the body even on non-2xx:
   ```bash
   [ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
   curl -sS -i "$BASE_URL/api/challenge/current-user-info" \
     -H "Authorization: Bearer $CHALLENGE_TOKEN" | tail -15
   ```
   The body usually says exactly what's wrong (often a 401-style message returned with a 500 status code).

Only after the token is confirmed valid (e.g. the same value just succeeded against `/login`) should you treat the 5xx as a real platform issue and retry with backoff.

## Symptom table

| # | Symptom | Likely cause | Action |
|---|---------|--------------|--------|
| 1 | `POST /api/challenge/job` returns **HTTP 400** `invalid board` / `no task templates for board` | `config.board` is missing or not in the currently-supported set | Today the supported values are `instruction` (default), `spatial`, `manip`, `robust`. Use one of those and resubmit. **Do not retry the same wrong board** — 400 is a semantic rejection, retrying burns 1/4 daily quota each time. This holds even when the user says "maybe a network blip" — see the heuristic above. If a contestant insists a new board exists, verify against `../user-manual.md` / organizers first. |
| 2 | `POST /api/challenge/job` returns an "upload limit" error / `/api/challenge/submission/quota` returns `remaining: 0` | Today's 4-submissions quota is used up | Wait until Beijing midnight (UTC+8), or work with an existing job. |
| 3 | Any `/api/challenge/*` call (login, result, log, job, tunnel-endpoint, **or** the WS handshake) returns `401` | Invalid / expired `access_token`; for the WS handshake also: `job_uuid` not owned by this account | `challenge-login` Step 3 to refresh; if refresh also 401s, fall back to Step 1 (email + password). For the WS handshake specifically, also verify `JOB_UUID` came from this account's `POST /api/challenge/job`. |
| 4 | Agent connects, then immediately disconnects | Per-user `parallelism` cap exceeded — too many agents online | Reduce K in `challenge-run-agent`, or wait for older agents to finish. |
| 5 | Job stuck in `Pending` / `Running` with no score change for minutes | No agent online, or all agents dropped past the 30 s window | `ps`/check the agent processes; relaunch via `challenge-run-agent` if needed. |
| 6 | Agent disconnected unexpectedly mid-run | Network blip | Within the gateway-side **30 s reconnect window**, the SDK / `tunnel.sh` redials **automatically** with the same `agent_id` to resume the open session. **Do not manually relaunch** in that window — you'd race the SDK and the gateway will reject the duplicate. Past 30 s the open session is gone; resubmitting a new job costs 1/4 daily quota — confirm before suggesting it. |
| 7 | Received `drain` control frame | Platform finished dispatching cases and is asking for graceful shutdown | Let in-flight sessions finish, close the socket, and **do NOT reconnect**. |
| 8 | Job ended in `Failed` | An evaluation case crashed | `GET /api/challenge/job/$JOB_ID/log` (see `challenge-poll-result`); read the latest stderr / exit code. Common causes: handler exception during warmup (see row #11 — surfaces here, not just as "stuck WARMUP"), OOM, action-encoding mismatch. **Fix the bug locally before resubmitting** — resubmission costs 1/4 daily quota. |
| 9 | `/api/challenge/tunnel/endpoint` returns empty string | Gateway not yet ready | Fall back to the fixed default `ws://120.92.88.78/api/challenge/tunnel`; the endpoint may simply be omitted because the host never changes. |
| 10 | `parallelism` in the job response is `0` | User's `MaxConcurrentCases` is misconfigured (not "exhausted" — exhaustion is a runtime gateway check, not a response field) | Stop and surface to organizers; do not launch agents. |
| 11 | Agent stuck in `WARMUP`, never reaches `RUNNING` — **or** the job ends `Failed` with a traceback referencing `frame_bytes=b''` / zero-length decode in the `/log` output | Inference handler errored on the warmup empty frame | The warmup call passes an **empty frame** — handlers must tolerate that (e.g. short-circuit when `len(frame_bytes) == 0` and return a no-op action). Fix in the inference repo, then resubmit (mind quota). |
| 12 | `tunnel.sh` exits with code 1 immediately | Bad arguments / handler import failure / reconnect retries exhausted | Re-check `<gpu_index> <job_uuid> <gateway_url>` argument order; check inference-repo handler import path. |
| 13 | `tunnel.sh` exits with code 130 | Ctrl-C | Expected; user-driven. |
| 14 | `POST /api/challenge/job` returns **HTTP 400** `paper_link is not a valid URL` / `paper_link too long` | Optional `paper_link` field is malformed or > 512 chars | Fix the URL or omit the field entirely; resubmit. Counts as 1/4 quota only if the call gets past validation. |

## Diagnostics shortlist

Run these in order when triaging an unclear failure:

```bash
# 1. Token still valid?
curl -fsS "$BASE_URL/api/challenge/current-user-info" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq

# 2. Job exists, what status?
curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/result" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq

# 3. If Failed, get the log
curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/log" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq

# 4. Gateway endpoint sane?
curl -fsS "$BASE_URL/api/challenge/tunnel/endpoint" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN"
```

## Things NOT to do

- The gateway host is **fixed at `120.92.88.78`**: `BASE_URL=http://120.92.88.78`, `TUNNEL_ENDPOINT=ws://120.92.88.78/api/challenge/tunnel`. Prefer a `tunnel_endpoint` from the job response if present, but the fixed default is a safe fallback.
- **Do not** reconnect after a `drain` — the platform is finishing up, and the connection will be rejected.
- **Do not** spam `POST /api/challenge/job` to "retry" — each call is one of the four daily slots. **Remaining quota is not authorization to guess** (e.g. guessing a `board` because "quota 还有"). Skill rules forbid the guess itself, independent of remaining slots.
- **Do not** kill running agents to "free a slot" without checking `/result` first; you may be killing the agent that's about to finish your last case.
- **Do not** retry a 4xx as if it were a network blip — see the diagnostic heuristic at the top of this file.

## Reference

For the WS wire protocol (binary frame layout, control frames, state machine), see `../tunnel-protocol.md`. Most contestants don't need it — `./scripts/tunnel.sh` from the inference repo wraps the official SDK.
