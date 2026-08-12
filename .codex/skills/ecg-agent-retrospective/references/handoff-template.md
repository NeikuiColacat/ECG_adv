# ECG Manual Refactor Handoff Template

Use this for long sessions before context compaction, archiving, or a new agent
handoff. A good handoff is short enough to read, but complete enough that a
fresh agent can continue without reopening the old chat.

```markdown
# Handoff: <short topic>

Date: YYYY-MM-DD
Author: Codex
Repo: /home/linbinhao/ECG_manual_refactor_paper_kernel_v2

## Executive Summary
2-4 sentences with the goal and current state.

## Key Decisions & Rationale
- Decision:
- Rationale:
- Tradeoff considered:

## Current Truth
- Mapping/version/hash:
- Main config(s):
- Run roots:
- Metrics/table paths:
- Git state:

## Current Codebase State
- `path/to/file`: why it matters / recent change
- `path/to/file`: why it matters / current risk

## Completed
- [x] ...

## Next 5 Steps
1. High: task + success criterion
2. High: task + success criterion
3. Medium: task + success criterion
4. Medium: task + success criterion
5. Low: task + success criterion

## Constraints & Style
- Shared-server constraints:
- Do not touch:
- Files likely dirty from user/agent:
- User preference or project style:

## Archive / Context Notes
- Old chat/session still needed? yes/no, why:
- Any large logs or artifacts should be searched, not pasted:
- Evidence registry or docs that supersede old context:

## Reactivation Prompt
Read `AGENTS.md` first 100 lines, then this handoff. Verify the current repo
state before running GPU, writing files, or modifying environment. Continue
from "Next 5 Steps"; do not assume old chat history is available.
```

## Handoff Quality Checklist

- It names the active objective in the first paragraph.
- It lists only files and paths that matter for continuation.
- It preserves decisions and tradeoffs, not every message.
- It gives prioritized next steps with measurable success criteria.
- It includes a copy-paste reactivation prompt.
- It does not paste secrets, huge logs, checkpoints, datasets, or raw transcripts.
