# Repo-Tracked Codex Skills

This directory stores project skill copies that should travel with the repo.

Active skills:

```text
.codex/skills/ecg-adv-gen/SKILL.md
.codex/skills/ecg-vae-online-at/SKILL.md
```

To install/update them on an AutoDL host:

```bash
mkdir -p /root/.codex/skills/ecg-adv-gen
cp /root/ECG_adv_Gen/.codex/skills/ecg-adv-gen/SKILL.md \
  /root/.codex/skills/ecg-adv-gen/SKILL.md

mkdir -p /root/.codex/skills/ecg-vae-online-at
cp -r /root/ECG_adv_Gen/.codex/skills/ecg-vae-online-at/. \
  /root/.codex/skills/ecg-vae-online-at/
```

The runtime Codex skill directory is outside git, so keep this repo copy updated
whenever durable project memory changes.
