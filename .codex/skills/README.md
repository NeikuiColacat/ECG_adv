# Repo-Tracked Codex Skills

This directory stores public project procedures that should travel with the
clean-room repository. It intentionally excludes global Codex memories, session
transcripts, credentials, local model links, and machine-specific agent config.

Active skills:

```text
.codex/skills/ecg-adv-gen/SKILL.md
.codex/skills/ecg-vae-online-at/SKILL.md
.codex/skills/shared-gpu-server-discipline/SKILL.md
.codex/skills/data-prep-validator/SKILL.md
.codex/skills/reproducibility-check/SKILL.md
.codex/skills/artifact-git-guard/SKILL.md
.codex/skills/model-eval/SKILL.md
.codex/skills/ecg-agent-retrospective/SKILL.md
```

Codex can discover these skills from the repository. To refresh the current
user-level runtime copies explicitly:

```bash
REPO_ROOT=/home/linbinhao/ECG_manual_refactor_clean
RUNTIME_SKILLS=/home/linbinhao/.codex/skills
mkdir -p "${RUNTIME_SKILLS}"
for skill in \
  ecg-adv-gen \
  ecg-vae-online-at \
  shared-gpu-server-discipline \
  data-prep-validator \
  reproducibility-check \
  artifact-git-guard \
  model-eval \
  ecg-agent-retrospective
do
  mkdir -p "${RUNTIME_SKILLS}/${skill}"
  cp -r "${REPO_ROOT}/.codex/skills/${skill}/." \
    "${RUNTIME_SKILLS}/${skill}/"
done
```

The runtime Codex skill directory is outside git, so keep this repo copy updated
whenever durable project memory changes.
