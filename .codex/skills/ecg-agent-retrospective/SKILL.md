---
name: ecg-agent-retrospective
description: Use when the user asks Codex to review recent ECG_adv_Gen work, conversation handoffs, session summaries, or repeated friction and propose minimal AGENTS.md updates or tiny reusable skills for long-running agent productivity.
---

# ECG Agent Retrospective

This skill keeps ECG_adv_Gen's agent memory useful without turning
`AGENTS.md` into a dumping ground.

Use it when the user says things like:

- "扫描对话 session 看看哪些需要整理成 skill"
- "把这次经验更新到 AGENTS.md"
- "做一次 retrospective"
- "以后 Codex 接手别再犯这个错"

## Core Rule

Default to **report-only**. Propose exact minimal changes first; apply them only
after the user accepts.

## Quality Bar

A good retrospective output must look like a compact engineering handoff:

- It names repeated patterns from concrete sessions, docs, or runs.
- It proposes exact AGENTS.md locations and text, not vague advice.
- It creates at most 0-2 tiny skills, and only when a procedure is reusable.
- It ends with an application plan and validation commands.
- If evidence is weak, it says "no durable update recommended".

## Inputs To Inspect

Prefer project-local, durable inputs:

1. `AGENTS.md` first 100 lines, then relevant sections.
2. `.codex/skills/*/SKILL.md` and only the specific reference files needed.
3. `docs/codex-handoffs/*` if present.
4. Recent `docs/pipelines/*`, especially handoff/repro/refactor docs.
5. Recent `docs/reports/archive/**/*summary*` and
   `docs/reports/archive/**/*report*` files when they are the only source of a
   decision.
6. `configs/active_scripts.yaml` and active experiment/evidence registry files.
7. User-provided session transcript paths. Do not rummage through unrelated
   global Codex sessions by default.

Run the inventory helper when useful:

```bash
micromamba run -n ECGTwin python scripts/agent/audit_retrospective_inputs.py --format markdown
```

## Triage

Classify each useful observation:

- **AGENTS.md**: short, durable rule needed at startup or before risky actions.
- **Tiny skill**: reusable procedure with a clear trigger and enough repeated
  value to justify a new file.
- **Evidence registry**: experiment fact, run lineage, metric comparison, or
  claim status.
- **Handoff/doc**: one-off context that may matter later but should not load on
  every agent startup.
- **Ignore**: single low-impact incident, stale branch detail, or vague advice.

## Output Format

Always respond in this order:

1. **Retrospective Summary**: 3-6 bullets, evidence-based.
2. **Proposed AGENTS.md Updates**: location, exact small diff, rationale.
3. **Proposed Tiny Skills**: 0-2 max, with filename and complete `SKILL.md`.
4. **Evidence & Rationale**: point to sessions, handoffs, docs, or runs.
5. **Application Plan**: exact files to edit and validation commands.

For long-session handoffs, use `references/handoff-template.md`. The handoff
must let a fresh agent continue without reopening the old giant chat.

For session/archive hygiene requests, read `references/session-hygiene.md`.
This ECG skill can diagnose and propose handoffs, but it does not archive or
delete sessions by itself.

## Hard Constraints

- Do not propose broad AGENTS.md rewrites.
- Do not add vague rules such as "be careful" or "think harder".
- Do not create large skills. Keep new skills narrow and initially under about
  100 lines unless the user explicitly asks for more.
- Do not base a new durable rule on one minor incident.
- Do not copy secrets, private keys, raw large logs, checkpoints, datasets, or
  generated samples into skills or AGENTS.md.
- Do not auto-archive sessions, delete files, or edit global Codex state.

## References

- `references/update-rules.md`: what may and may not go into `AGENTS.md`.
- `references/handoff-template.md`: compact handoff format for long sessions.
- `references/session-hygiene.md`: report-only session hygiene and archive guardrails.
