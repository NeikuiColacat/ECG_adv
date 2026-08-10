# Miniforge environments

This directory defines the user-level Miniforge runtime for the clean-room ECG
mainline. It intentionally excludes the unrelated legacy packages from the old
Micromamba environment.

## Prefixes

- Miniforge root: `/home/linbinhao/miniforge3`
- ECG runtime prefix: `/home/linbinhao/miniforge3/envs/ECGTwin`
- CLI tools prefix: `/home/linbinhao/miniforge3/envs/cli-tools`
- Retired prefix (deleted on 2026-08-08 after validation):
  `/home/linbinhao/micromamba/envs/ECGTwin`
- Retired legacy CLI prefix (deleted on 2026-08-08 after the clean switch):
  `/home/linbinhao/micromamba/envs/cli-tools`

Future Bash shells initialize Miniforge without auto-activating `base` and put
the Miniforge CLI prefix ahead of system tools. The default tmux server now runs
Miniforge tmux 3.7b; unrelated named report servers remain separate.

## Build contract

1. Install the pinned Miniforge 26.3.2-2 Linux x86_64 installer after verifying
   SHA-256 `42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94`.
2. Create the environment from `ecgtwin-miniforge.yml` with conda-forge as the
   only Conda channel.
3. Install `ecgtwin-pip-lock.txt`.
4. Install the three local CUDA 11.8 PyTorch wheels with `--no-deps` after
   checking `pytorch-cu118-wheel-manifest.sha256`.
5. Require a clean `python -m pip check`, the retained CPU contract suite,
   managed launcher dry-run, and a one-GPU smoke test before retiring the old
   prefix.

The wheel files and environment directories are external artifacts and must not
enter Git.

## CLI tools contract

1. Create `cli-tools` from `cli-tools-miniforge.yml` with strict conda-forge
   priority.
2. Require `tmux 3.7b` and `git 2.51.0`; 3.6a is retained only until the old
   server's maintenance-window restart.
3. Before the final upgrade, verify that the staged 3.6a client can query the
   existing tmux 3.6a server; do not mix the final 3.7b client with it.
4. Load `~/.tmux.conf` on an isolated socket and verify `mode-keys vi`,
   `status-keys vi`, and the `v`/`V`/`y` copy-mode bindings.
5. Point future Bash shells and `~/.bash_aliases` to the Miniforge CLI prefix.
6. Retire the old CLI prefix only after an explicit server restart window, a
   new 3.7b default server, restored long-running services, and a zero-process
   check.
