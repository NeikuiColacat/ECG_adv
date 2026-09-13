# Compact Project Handoff

Use when a handoff is requested or needed to preserve authorized ongoing work.
Fill only relevant fields; do not manufacture five steps or reopen old chats.
Use the actual checkout/branch, not a fixed LLM worktree path.

```markdown
# Handoff: <task>
Date: <timestamp/timezone>
Checkout / branch / HEAD: <verified identity>

## Objective and current state
<Requested outcome; completed, running or blocked; evidence timestamp.>

## Decisions and protected work
- Direction, recipe/config and metric/data contract:
- Dirty files and their owner:
- Authorization/resource limit; do-not-touch paths/processes:

## Evidence and running jobs
- Run root, resolved config, source/checkpoint hashes, result paths:
- Coordinator/control/status paths and verified PID identity, if running:
- Temporary RAM root and required durable artifacts, if applicable:
- Passed checks and remaining gaps; no unsupported performance claim:

## Next action
<Smallest safe next step and its success criterion.>
<Exact command when needed; no implicit delete, relaunch or overwrite.>

## Resume
Read AGENTS first 100 lines and verify the actual checkout/branch and live state.
Do not edit frozen source, assume a stale PID is alive, or restart healthy jobs.
If the task is already running normally, leave it in the background unless
monitoring or another action was requested.
```

Keep only evidence needed for continuation. Never paste secrets, raw ECG data,
large logs, checkpoints or unrelated global sessions. Describe recovery before
any separately approved retirement; a handoff is not archive/delete permission.
