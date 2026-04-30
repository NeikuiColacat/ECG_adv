# Repo-Tracked Codex Skills

This directory stores project skill copies that should travel with the repo.

Active skill:

```text
.codex/skills/ecg-adv-gen/SKILL.md
```

To install/update it on an AutoDL host:

```bash
mkdir -p /root/.codex/skills/ecg-adv-gen
cp /root/ECG_adv_Gen/.codex/skills/ecg-adv-gen/SKILL.md \
  /root/.codex/skills/ecg-adv-gen/SKILL.md
```

The runtime Codex skill directory is outside git, so keep this repo copy updated
whenever durable project memory changes.
