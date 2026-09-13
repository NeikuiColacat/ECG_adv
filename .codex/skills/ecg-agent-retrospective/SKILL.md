---
name: ecg-agent-retrospective
description: Review or update ECG_manual_refactor project instructions, skills and handoffs from concrete repo evidence, without collecting unrelated private sessions.
---

# ECG Project Instruction Maintenance

Keep project guidance useful at startup; move experiment facts out of instructions.

## Scope and Authority

- Review/plan requests are report-only. An explicit request to edit or optimize
  named project instructions authorizes those edits; do not ask for the same
  approval again. Prefer focused changes over a new policy framework.
- It does not authorize deleting files/sessions/worktrees, changing global Codex
  state, launching experiments, committing or publishing.
- Follow `AGENTS.md` and the keep manifest. Read only task-relevant project
  docs, registry entries and selected skill references; use `rg` to locate them.
- Use user-provided transcripts only when needed. Do not rummage through global
  sessions, credentials or memories, or copy them into the repository.

## Choose the Owner

- `AGENTS.md`: durable startup/action rules and direction routing.
- Existing skill: a reusable procedure with a clear trigger.
- Evidence registry: run identities, measured results and claim status.
- Manifest/direction docs: retained paths, ownership and historical decisions.
- Handoff: live task state, exact next action, authorization and recovery.
- Ignore: isolated trivia or a rule that would not change future decisions.

Read [update rules](references/update-rules.md) before editing instructions.
Prefer updating an existing owner; add a skill only for a genuinely separate
reusable workflow. Preserve scientific/safety contracts and historical evidence.

## Apply and Verify

1. Inspect the relevant diff and references; identify the concrete friction.
2. Make the smallest authorized edit with `apply_patch`. For multi-worktree
   updates, preserve local deltas and verify the intended common text.
3. Check frontmatter, local references, discovery descriptions and consistency
   with AGENTS; use the available skill-creator validator for changed skills.
4. Exercise realistic decision cases when routing/authority rules change.
   Documentation-only work needs no GPU job or scientific rerun.
5. Report changed files, practical improvements, validation and remaining limits.
   Do not paste entire skills or force empty report sections into the answer.

Use [handoff template](references/handoff-template.md) for a requested handoff,
and [session hygiene](references/session-hygiene.md) only for session/archive work.
Archiving still requires its separate review and explicit apply authorization.
