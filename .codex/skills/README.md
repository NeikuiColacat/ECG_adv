# Repo-Tracked Codex Skills

This directory stores project skill copies that should travel with the repo.

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

To install/update them on the current migrated user host:

```bash
mkdir -p /home/linbinhao/.codex/skills
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
  mkdir -p "/home/linbinhao/.codex/skills/${skill}"
  cp -r "/home/linbinhao/ECG_adv_Gen/.codex/skills/${skill}/." \
    "/home/linbinhao/.codex/skills/${skill}/"
done
```

To install/update them on the original root AutoDL host:

```bash
mkdir -p /root/.codex/skills/ecg-adv-gen
cp /root/ECG_adv_Gen/.codex/skills/ecg-adv-gen/SKILL.md \
  /root/.codex/skills/ecg-adv-gen/SKILL.md

mkdir -p /root/.codex/skills/ecg-vae-online-at
cp -r /root/ECG_adv_Gen/.codex/skills/ecg-vae-online-at/. \
  /root/.codex/skills/ecg-vae-online-at/

for skill in \
  shared-gpu-server-discipline \
  data-prep-validator \
  reproducibility-check \
  artifact-git-guard \
  model-eval \
  ecg-agent-retrospective
do
  mkdir -p "/root/.codex/skills/${skill}"
  cp -r "/root/ECG_adv_Gen/.codex/skills/${skill}/." \
    "/root/.codex/skills/${skill}/"
done
```

The runtime Codex skill directory is outside git, so keep this repo copy updated
whenever durable project memory changes.
