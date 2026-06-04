# AGENTS.md Organization Suggestions 2026-05-29

This scan is advisory. It does not change the experiment protocol by itself.

## Current Findings

- `AGENTS.md` is useful but long: about 700 lines, with startup rules,
  historical experiment facts, old AutoDL commands, and current migrated-host
  overrides in one file.
- The first 100 lines are in good shape and should stay focused on shared-server
  safety, current host paths, agent operating commands, and repository
  navigation.
- Later sections still contain many `/root/...` examples from the original
  AutoDL host. They are valuable for historical replay but risky as default
  commands on the migrated `/home/linbinhao` host.
- Mapping notes are mixed across eras: older v3/v5/v6 text remains in the
  durable memory, while current trusted evidence lives in
  `configs/active_evidence_registry.yaml` and the v7 pipeline docs.
- Some detailed training commands and old metric memories are better kept in
  `docs/pipelines/` or archived reports, with AGENTS linking to them.

## Recommended Target Shape

Keep `AGENTS.md` as a startup and safety index:

- shared-server constraints and current host override;
- active evidence registry and agent audit commands;
- repository navigation and do-not-move/do-not-commit boundaries;
- short current-method summary and links to canonical docs;
- stable ECGTwin/Super5 contracts that every agent must remember.

Move or de-emphasize from `AGENTS.md`:

- long historical training commands with `/root/...` absolute paths;
- old disk-size notes from previous hosts;
- obsolete mapping version details that conflict with active registry entries;
- old pilot metrics that are already captured in archived reports.

## Safe Cleanup Phases

1. Keep the first 100 lines stable. Do not move the shared-server safety rules
   below the startup boundary.
2. Add short warnings around historical `/root/...` command blocks instead of
   rewriting them immediately.
3. Replace detailed historical results with links to archived reports under
   `docs/reports/archive/YYYYMMDD/`.
4. For mapping facts, make `configs/active_evidence_registry.yaml` and
   `docs/pipelines/v7_sjr_rgq_effnet_mainline_repro_20260528.md` the current
   truth, and mark older v3/v5/v6 notes as historical.
5. After each reduction, run the CPU-only agent audit and the retrospective
   inventory test before committing.

## Validation Commands

```bash
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
micromamba run -n ECGTwin pytest util/tests/test_agent_retrospective_audit.py -q
micromamba run -n cli-tools git grep -n "docs/tmp_md\\|docs/tmp_html\\|docs/tmp_jsonl" || true
```
