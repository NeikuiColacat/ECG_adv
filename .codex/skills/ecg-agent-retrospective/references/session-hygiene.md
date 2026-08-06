# Session Hygiene Guardrails

Use this only when the user asks about Codex session bloat, stale chats,
archiving, or keeping long-running agent work manageable.

## ECG Manual Refactor Policy

- First pass is always report-only.
- Handoff before archive: every active thread worth continuing needs a compact
  handoff with a reactivation prompt.
- Backup before mutate: any future archive/cleanup flow must create a
  timestamped backup first.
- Archive, never delete.
- Do not touch credentials, private keys, global runtime skills still in use,
  model checkpoints, datasets, or generated experiment outputs.
- Do not edit global Codex state or archive sessions unless the user explicitly
  asks for an apply step after reviewing a report.

## Report Should Include

- Candidate handoffs that are missing.
- Oversized or stale project docs/session inputs, if explicitly provided.
- Which material is active, archived/superseded, or safe to ignore.
- Exact follow-up action, usually "write handoff first" rather than "clean up".

## Red Flags

- Pressure to archive without a handoff.
- A proposed cleanup would touch files outside `/home/linbinhao`.
- A proposed cleanup would remove experiment evidence before it is represented
  in an evidence registry or handoff.
