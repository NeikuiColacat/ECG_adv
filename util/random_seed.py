"""Project-wide deterministic random-seed infrastructure.

One tracked YAML owns the base seed.  Every stochastic subsystem derives an
isolated stream from an explicit namespace and identity, so adding a random
draw in one subsystem cannot silently perturb another subsystem.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from util.pn2021_artifact_contract import sha256_file as _sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANDOM_SEED_CONFIG_PATH = PROJECT_ROOT / "configs" / "random_seed.yaml"
SEED_DERIVATION = "sha256_first_uint32_little_endian"


def _validated_seed(value: Any, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer")
    if not 0 <= value < 2**32:
        raise ValueError(f"{description} must be in [0, 2**32)")
    return int(value)


@dataclass(frozen=True)
class RandomSeedConfig:
    path: Path
    sha256: str
    base_seed: int

    def describe(self) -> dict[str, Any]:
        return {
            "config_path": str(self.path),
            "config_sha256": self.sha256,
            "base_seed": self.base_seed,
        }


@dataclass(frozen=True)
class SeedIdentity:
    base_seed: int
    effective_seed: int
    namespace: str
    identity: tuple[str, ...]
    config_path: Path
    config_sha256: str

    def describe(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "effective_seed": self.effective_seed,
            "namespace": self.namespace,
            "identity": list(self.identity),
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "derivation": SEED_DERIVATION,
        }


def load_random_seed_config(
    path: str | Path = DEFAULT_RANDOM_SEED_CONFIG_PATH,
) -> RandomSeedConfig:
    """Load and identify one seed YAML without mutating any RNG state."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"random seed config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("random seed config must be a YAML mapping")
    if set(payload) != {"schema_version", "random_seed"}:
        raise ValueError(
            "random seed config must contain exactly schema_version and random_seed"
        )
    if payload["schema_version"] != 1:
        raise ValueError("random seed config schema_version must be 1")
    return RandomSeedConfig(
        path=config_path,
        sha256=_sha256_file(config_path),
        base_seed=_validated_seed(
            payload["random_seed"], description="random_seed"
        ),
    )


def _seed_identity(
    namespace: str,
    *identity: Any,
    base_seed: int | None = None,
    config_path: str | Path = DEFAULT_RANDOM_SEED_CONFIG_PATH,
) -> SeedIdentity:
    namespace_value = str(namespace)
    if not namespace_value:
        raise ValueError("seed namespace must be non-empty")
    config = load_random_seed_config(config_path)
    resolved_base = (
        config.base_seed
        if base_seed is None
        else _validated_seed(base_seed, description="base_seed")
    )
    identity_values = tuple(str(value) for value in identity)
    payload = "|".join(
        str(value) for value in (resolved_base, namespace_value, *identity_values)
    ).encode("utf-8")
    effective = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return SeedIdentity(
        base_seed=resolved_base,
        effective_seed=effective,
        namespace=namespace_value,
        identity=identity_values,
        config_path=config.path,
        config_sha256=config.sha256,
    )


def derive_seed(
    namespace: str,
    *identity: Any,
    base_seed: int | None = None,
    config_path: str | Path = DEFAULT_RANDOM_SEED_CONFIG_PATH,
) -> int:
    """Derive a stable uint32 seed for one isolated stochastic stream."""

    return _seed_identity(
        namespace,
        *identity,
        base_seed=base_seed,
        config_path=config_path,
    ).effective_seed


def make_python_rng(
    namespace: str,
    *identity: Any,
    base_seed: int | None = None,
    config_path: str | Path = DEFAULT_RANDOM_SEED_CONFIG_PATH,
) -> random.Random:
    return random.Random(
        derive_seed(
            namespace,
            *identity,
            base_seed=base_seed,
            config_path=config_path,
        )
    )


def make_numpy_rng(
    namespace: str,
    *identity: Any,
    base_seed: int | None = None,
    config_path: str | Path = DEFAULT_RANDOM_SEED_CONFIG_PATH,
) -> np.random.RandomState:
    """Return an isolated legacy-compatible NumPy RandomState."""

    return np.random.RandomState(
        derive_seed(
            namespace,
            *identity,
            base_seed=base_seed,
            config_path=config_path,
        )
    )


def make_torch_generator(
    device: Any,
    namespace: str,
    *identity: Any,
    base_seed: int | None = None,
    config_path: str | Path = DEFAULT_RANDOM_SEED_CONFIG_PATH,
):
    """Return an isolated Torch generator on the requested device."""

    import torch

    return torch.Generator(device=device).manual_seed(
        derive_seed(
            namespace,
            *identity,
            base_seed=base_seed,
            config_path=config_path,
        )
    )


def seed_process(
    namespace: str,
    *identity: Any,
    base_seed: int | None = None,
    config_path: str | Path = DEFAULT_RANDOM_SEED_CONFIG_PATH,
    deterministic_algorithms: bool = False,
    warn_only: bool = True,
) -> SeedIdentity:
    """Seed process-global Python, NumPy, Torch and visible CUDA backends."""

    import torch

    resolved = _seed_identity(
        namespace,
        *identity,
        base_seed=base_seed,
        config_path=config_path,
    )
    random.seed(resolved.effective_seed)
    np.random.seed(resolved.effective_seed)
    torch.manual_seed(resolved.effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved.effective_seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return resolved


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed worker-local global RNGs from PyTorch's loader-assigned seed."""

    del worker_id
    import torch

    worker_seed = int(torch.initial_seed() % 2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


__all__ = [
    "DEFAULT_RANDOM_SEED_CONFIG_PATH",
    "SEED_DERIVATION",
    "RandomSeedConfig",
    "SeedIdentity",
    "derive_seed",
    "load_random_seed_config",
    "make_numpy_rng",
    "make_python_rng",
    "make_torch_generator",
    "seed_dataloader_worker",
    "seed_process",
]
