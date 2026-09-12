# Repo-Tracked Codex Skills

This directory stores public project procedures that should travel with the
clean-room repository. It intentionally excludes global Codex memories, session
transcripts, credentials, local model links, and machine-specific agent config.

Active skills:

```text
.codex/skills/ecg-adv-gen/SKILL.md
.codex/skills/ecg-vae-online-at/SKILL.md
.codex/skills/ecg-code-simplifier/SKILL.md
.codex/skills/shared-gpu-server-discipline/SKILL.md
.codex/skills/data-prep-validator/SKILL.md
.codex/skills/reproducibility-check/SKILL.md
.codex/skills/artifact-git-guard/SKILL.md
.codex/skills/model-eval/SKILL.md
.codex/skills/ecg-agent-retrospective/SKILL.md
```

This project-local directory is authoritative for this repository. Do not
automatically copy these Skills into `/home/linbinhao/.codex/skills`: that scope
is shared by other repositories, and identically named Skills are discovered as
separate candidates rather than merged into one definition.

Keep cross-repository Skills globally named and maintained separately. If a
project-local Skill change is not visible in an existing Codex session, restart
the session instead of overwriting a user-level Skill.
