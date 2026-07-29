"""Runtime-only adapter for the isolated Diverse Latent AugMax sandbox.

This file never mutates whitelist source.  While :func:`patch_online_trainer_runtime`
is active it replaces only ``core.online_trainer.build_method_runtime`` for
the sandbox-owned D0--D6 Diverse runtime ids and scales only those objectives.
Data loading, normalization, optimizer/scheduler steps, validation,
checkpointing and run recording remain owned by the whitelist trainer.

The shared endpoint protocol is deliberately strict:

* each base batch creates E=8 depth-1/2/3 corrupted endpoints exactly once;
* clean and endpoints are encoded with deterministic VAE posterior means;
* the train400 ``latent_pool.standardizer`` defines the geometry;
* K=3 directions are selected by prediction shift plus greedy diversity;
* both generated views reuse that exact selected endpoint pool;
* a registry-owned train400 P95 standardized-RMS cap bounds every view;
* only an explicit profile smoke flag permits a missing cap;
* the per-generate cache is cleared in ``finally`` and never survives a batch.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from core.augmix import AugMixConfig
from core.corruption import (
    CANONICAL_OPERATORS,
    COMPOSITIONS,
    CORRUPTION_DOMAIN_POINTS,
    CORRUPTION_DOMAIN_SAMPLING_RATE_HZ,
    INTERPOLATION_ALIGN_CORNERS,
    INTERPOLATION_MODE,
    OUTPUT_POINTS,
    generate_canonical_corruption,
)
from core.methods.contracts import NodeContext, Provenance, ViewValue, WaveformView
from core.methods.registry import CompiledMethod
from core.methods.runtime import MethodViewRuntime
from models import get_model_spec
from models.contracts import validate_model_output
from models.input_adapter import prepare_canonical_model_input
from models.vae import decode_to_ptbxl_waveform, prepare_ecgtwin_encoder_input
from util.augmentations.torch_operators import apply_operator_batch_prevalidated
from util.config_bundle import resolve_config_reference
from util.random_seed import derive_seed

try:  # Package import and direct delegate subprocess import are both supported.
    from .canonical_pool import (
        CanonicalPoolSourceIdentity,
        POOL_SOURCE_IDENTITY_KEYS,
    )
    from .geometry import (
        DiversitySelection,
        LatentProjection,
        StandardizedEndpointGeometry,
        build_standardized_endpoint_geometry,
        greedy_select_diverse_directions,
        mix_endpoint_residuals,
        project_clean_anchored_simplex,
        sample_simplex_state,
    )
except ImportError:  # pragma: no cover - direct sibling import in delegate.py
    from canonical_pool import (  # type: ignore[no-redef]
        CanonicalPoolSourceIdentity,
        POOL_SOURCE_IDENTITY_KEYS,
    )
    from geometry import (  # type: ignore[no-redef]
        DiversitySelection,
        LatentProjection,
        StandardizedEndpointGeometry,
        build_standardized_endpoint_geometry,
        greedy_select_diverse_directions,
        mix_endpoint_residuals,
        project_clean_anchored_simplex,
        sample_simplex_state,
    )


SANDBOX_ROOT = Path(__file__).resolve().parent
D3_METHOD_ID = "diverse_augmax_d3_endpoint_hard_v1"
D3_DEPTH23_METHOD_ID = "diverse_augmax_d3_endpoint_hard_depth23_v1"
D6_METHOD_ID = "diverse_augmax_d6_endpointset_boundary"
DIVERSE_METHOD_VIEW_MODES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "diverse_augmax_d0_random_v1": MappingProxyType(
            {"view_1": "random", "view_2": "random"}
        ),
        "diverse_augmax_d1_hard_v1": MappingProxyType(
            {"view_1": "hard", "view_2": "hard"}
        ),
        "diverse_augmax_d2_diverse_hard_v1": MappingProxyType(
            {"view_1": "random", "view_2": "hard"}
        ),
        D3_METHOD_ID: MappingProxyType(
            {"view_1": "raw_endpoint", "view_2": "hard"}
        ),
        D3_DEPTH23_METHOD_ID: MappingProxyType(
            {"view_1": "raw_endpoint", "view_2": "hard"}
        ),
        D6_METHOD_ID: MappingProxyType(
            {
                "endpoint_1": "raw_endpoint",
                "endpoint_2": "raw_endpoint",
                "endpoint_3": "raw_endpoint",
                "hard_view": "boundary_hard",
            }
        ),
    }
)
DIVERSE_METHOD_IDS = frozenset(DIVERSE_METHOD_VIEW_MODES)
KNOWN_CENTERS = frozenset(
    {"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"}
)
RADIUS_REGISTRY_SCHEMA_VERSION = 3
RADIUS_METRIC = "standardized_clean_to_corruption_latent_l2_over_sqrt_D"
ENDPOINT_CHUNK_SIZE = 2
CLASSIFIER_CHUNK_SIZE = 64
OBJECTIVE_GLOBAL_SCALE = 0.5
RADIUS_CALIBRATION_NAMESPACE = "diverse_latent_augmax_radius_calibration_v1"
DIAGNOSTIC_CONTRACT_VERSION = 1
MULTILABEL_DECISION_THRESHOLD = 0.5
M_LOW_BOUNDARY_INCLUSIVE = 0.05
M_HIGH_BOUNDARY_INCLUSIVE = 0.95


def _diagnostic_contract_description() -> dict[str, Any]:
    """Describe the exact denominators used by the sandbox P1 diagnostics."""

    return {
        "version": DIAGNOSTIC_CONTRACT_VERSION,
        "multilabel_decision": {
            "target_positive": (
                f"target >= {MULTILABEL_DECISION_THRESHOLD:g}"
            ),
            "prediction_positive": (
                f"sigmoid(logit) >= {MULTILABEL_DECISION_THRESHOLD:g}"
            ),
            "clean_correct_multilabel_asr": (
                "denominator = quality-accepted, finite hard-search records for "
                "which all five clean thresholded decisions equal the targets; "
                "numerator = denominator records for which at least one final "
                "hard-view decision is wrong"
            ),
            "positive_hide": (
                "denominator = target-positive class decisions within the "
                "clean-correct multilabel ASR record denominator; numerator = "
                "those decisions predicted negative by the final hard view"
            ),
            "negative_add": (
                "denominator = target-negative class decisions within the "
                "clean-correct multilabel ASR record denominator; numerator = "
                "those decisions predicted positive by the final hard view"
            ),
        },
        "hard_vs_random": {
            "random": (
                "the exact per-record random simplex/m initialization from which "
                "the hard coefficient search starts"
            ),
            "hard": "the best finite per-record state found by the hard search",
            "denominator": (
                "quality-accepted records whose complete hard search and both "
                "comparison losses are finite"
            ),
            "gain": "hard BCE minus its matched random-initialization BCE",
            "quantiles": [0.5, 0.9],
        },
        "mixing_radius": {
            "denominator": "quality-accepted records with finite raw/effective m",
            "quantiles": [0.05, 0.5, 0.95],
            "low_boundary": f"m <= {M_LOW_BOUNDARY_INCLUSIVE:g}",
            "high_boundary": f"m >= {M_HIGH_BOUNDARY_INCLUSIVE:g}",
        },
        "aggregation": (
            "rate diagnostics carry their true denominator as diagnostic weight; "
            "quantiles are exact per generated base batch and are named batch_*"
        ),
        "extra_classifier_forward": False,
    }


def _endpoint_protocol_sha256() -> str:
    """Bind a calibrated radius to the exact endpoint-generation protocol.

    ``augmix.yaml`` alone is not sufficient identity: it references the
    operator profile, and the generated endpoints also depend on the sandbox
    depth-1/2/3 wrapper plus whitelist interpolation/operator code.  Hash the
    small implementation closure so stale radius caps fail closed after any of
    those semantics change.
    """

    project_root = SANDBOX_ROOT.parents[1]
    paths = (
        Path(__file__).resolve(),
        (SANDBOX_ROOT / "calibrate_radius.py").resolve(),
        (SANDBOX_ROOT / "canonical_pool.py").resolve(),
        (SANDBOX_ROOT / "geometry.py").resolve(),
        (project_root / "core" / "augmix.py").resolve(),
        (project_root / "core" / "corruption.py").resolve(),
        (project_root / "util" / "augmentations" / "profile.py").resolve(),
        (project_root / "util" / "augmentations" / "torch_operators.py").resolve(),
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"endpoint protocol source is missing: {path}")
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _calibration_algorithm_sha256() -> str:
    """Identity of the executable radius-calibration algorithm itself."""

    path = (SANDBOX_ROOT / "calibrate_radius.py").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"radius calibration source is missing: {path}")
    return _sha256(path)


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return dict(value)


def _strict_int(value: Any, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{description} must be an integer >= {minimum}")
    return int(value)


def _finite_float(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{description} must be finite")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, description: str) -> str:
    resolved = str(value).strip().lower()
    if len(resolved) != 64 or any(
        character not in "0123456789abcdef" for character in resolved
    ):
        raise ValueError(f"{description} must be a 64-character SHA256")
    return resolved


def _radius_entry_payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("entry_id", None)
    payload.pop("entry_payload_sha256", None)
    return _canonical_json_sha256(payload)


def _radius_entry_id(entry_payload_sha256: str) -> str:
    digest = _require_sha256(
        entry_payload_sha256, "radius entry payload SHA256"
    )
    return f"radius_p95_{digest}"


def _radius_entries_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_json_sha256([dict(entry) for entry in entries])


def _derive_seed(
    base_seed: int,
    rng_identity: Sequence[str],
    stream: str,
) -> int:
    """Derive replayable sandbox seeds without touching process-global RNG."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if not 0 <= base_seed < 2**32:
        raise ValueError("base_seed must be uint32")
    identity = tuple(str(value) for value in rng_identity)
    if any(not value for value in identity):
        raise ValueError("rng_identity must contain non-empty values")
    payload = "|".join(
        (
            str(base_seed),
            "diverse_latent_augmax_shared_protocol_v1",
            stream,
            *identity,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _generator(device: torch.device, seed: int) -> torch.Generator:
    value = torch.Generator(device=device)
    value.manual_seed(int(seed))
    return value


def _positions(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in torch.nonzero(mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )


def _hash_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DiverseAugMaxContract:
    """Strict parser for ``contracts.diverse_augmax``."""

    candidate_endpoints: int
    selected_directions: int
    allowed_depths: tuple[int, ...]
    depth_sampling: str
    view_modes: Mapping[str, str]
    search_steps: int
    coefficient_learning_rate: float
    hard_search_mode: str
    registry_path: str
    radius_required: bool
    radius_quantile: float
    selector_first: str
    latent_cosine_weight: float
    prediction_rms_weight: float
    smoke_allow_missing_radius_cap: bool

    @classmethod
    def from_method(cls, method: CompiledMethod) -> "DiverseAugMaxContract":
        if method.profile_name not in DIVERSE_METHOD_IDS:
            raise ValueError(f"unsupported Diverse AugMax method {method.profile_name!r}")
        raw = _mapping(
            method.contracts.get("diverse_augmax"),
            "contracts.diverse_augmax",
        )
        expected = {
            "candidate_endpoints",
            "selected_directions",
            "allowed_depths",
            "view_modes",
            "hard_search",
            "radius_cap",
            "selector",
            "smoke_allow_missing_radius_cap",
        }
        permitted = expected | {"depth_sampling"}
        if not expected.issubset(raw) or not set(raw).issubset(permitted):
            raise ValueError(
                "contracts.diverse_augmax keys are incomplete or unexpected; "
                f"required={sorted(expected)}, got={sorted(raw)}"
            )
        candidate_count = _strict_int(
            raw["candidate_endpoints"],
            "contracts.diverse_augmax.candidate_endpoints",
            minimum=1,
        )
        selected_count = _strict_int(
            raw["selected_directions"],
            "contracts.diverse_augmax.selected_directions",
            minimum=1,
        )
        if candidate_count != 8 or selected_count != 3:
            raise ValueError("Diverse AugMax sandbox fixes E=8 and K=3")
        depths = tuple(raw["allowed_depths"])
        depth_sampling = str(raw.get("depth_sampling", "uniform_per_candidate"))
        if method.profile_name in {D3_DEPTH23_METHOD_ID, D6_METHOD_ID}:
            if depths != (2, 3) or depth_sampling != "balanced_half_each":
                raise ValueError(
                    f"{method.profile_name} fixes depths [2,3] with "
                    "balanced_half_each sampling"
                )
        elif depths != (1, 2, 3) or depth_sampling != "uniform_per_candidate":
            raise ValueError(
                "D0-D3 require depths [1,2,3] with uniform_per_candidate sampling"
            )

        view_modes = _mapping(
            raw["view_modes"], "contracts.diverse_augmax.view_modes"
        )
        resolved_modes = {str(name): str(role) for name, role in view_modes.items()}
        expected_modes = dict(DIVERSE_METHOD_VIEW_MODES[method.profile_name])
        if resolved_modes != expected_modes:
            raise ValueError(
                f"{method.profile_name} requires view modes "
                f"{expected_modes}"
            )

        search = _mapping(
            raw["hard_search"], "contracts.diverse_augmax.hard_search"
        )
        if not {"steps", "coefficient_learning_rate"}.issubset(search) or not set(
            search
        ).issubset({"steps", "coefficient_learning_rate", "mode"}):
            raise ValueError(
                "hard_search keys are incomplete or unexpected"
            )
        steps = _strict_int(
            search["steps"], "contracts.diverse_augmax.hard_search.steps", minimum=1
        )
        learning_rate = _finite_float(
            search["coefficient_learning_rate"],
            "contracts.diverse_augmax.hard_search.coefficient_learning_rate",
        )
        if steps != 5 or learning_rate <= 0.0:
            raise ValueError("hard search fixes 5 steps and requires positive coefficient LR")
        hard_search_mode = str(
            search.get("mode", "clean_anchored_simplex_and_m")
        )
        expected_search_mode = (
            "boundary_simplex_m1"
            if method.profile_name == D6_METHOD_ID
            else "clean_anchored_simplex_and_m"
        )
        if hard_search_mode != expected_search_mode:
            raise ValueError(
                f"{method.profile_name} requires hard-search mode "
                f"{expected_search_mode!r}"
            )

        radius = _mapping(raw["radius_cap"], "contracts.diverse_augmax.radius_cap")
        if set(radius) != {"registry_path", "required", "quantile"}:
            raise ValueError(
                "radius_cap must contain registry_path, required and quantile"
            )
        registry_path = str(radius["registry_path"])
        if not registry_path:
            raise ValueError("radius_cap.registry_path must be non-empty")
        if radius["required"] is not True:
            raise ValueError("radius_cap.required must be true")
        quantile = _finite_float(
            radius["quantile"], "contracts.diverse_augmax.radius_cap.quantile"
        )
        if not math.isclose(quantile, 0.95, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("the first sandbox protocol fixes radius quantile at 0.95")

        selector = _mapping(raw["selector"], "contracts.diverse_augmax.selector")
        required_selector = {"latent_cosine_weight", "prediction_rms_weight"}
        if not required_selector.issubset(selector) or not set(selector).issubset(
            required_selector | {"first"}
        ):
            raise ValueError(
                "selector keys are incomplete or unexpected"
            )
        selector_first = str(
            selector.get("first", "maximum_clean_endpoint_prediction_rms")
        )
        expected_selector_first = (
            "maximum_supervised_endpoint_bce"
            if method.profile_name == D6_METHOD_ID
            else "maximum_clean_endpoint_prediction_rms"
        )
        if selector_first != expected_selector_first:
            raise ValueError(
                f"{method.profile_name} requires selector.first "
                f"{expected_selector_first!r}"
            )
        latent_weight = _finite_float(
            selector["latent_cosine_weight"],
            "contracts.diverse_augmax.selector.latent_cosine_weight",
        )
        prediction_weight = _finite_float(
            selector["prediction_rms_weight"],
            "contracts.diverse_augmax.selector.prediction_rms_weight",
        )
        if latent_weight < 0.0 or prediction_weight < 0.0 or not math.isclose(
            latent_weight + prediction_weight, 1.0, abs_tol=1.0e-12
        ):
            raise ValueError("selector weights must be non-negative and sum to one")
        smoke = raw["smoke_allow_missing_radius_cap"]
        if not isinstance(smoke, bool):
            raise TypeError("smoke_allow_missing_radius_cap must be boolean")
        return cls(
            candidate_endpoints=candidate_count,
            selected_directions=selected_count,
            allowed_depths=tuple(int(value) for value in depths),
            depth_sampling=depth_sampling,
            view_modes=MappingProxyType(dict(resolved_modes)),
            search_steps=steps,
            coefficient_learning_rate=learning_rate,
            hard_search_mode=hard_search_mode,
            registry_path=registry_path,
            radius_required=True,
            radius_quantile=quantile,
            selector_first=selector_first,
            latent_cosine_weight=latent_weight,
            prediction_rms_weight=prediction_weight,
            smoke_allow_missing_radius_cap=smoke,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "candidate_endpoints": self.candidate_endpoints,
            "selected_directions": self.selected_directions,
            "allowed_depths": list(self.allowed_depths),
            "depth_sampling": self.depth_sampling,
            "view_modes": dict(self.view_modes),
            "hard_search": {
                "steps": self.search_steps,
                "coefficient_learning_rate": self.coefficient_learning_rate,
                "mode": self.hard_search_mode,
                "optimized_variables": (
                    ["mix_logits"]
                    if self.hard_search_mode == "boundary_simplex_m1"
                    else ["mix_logits", "m_logit"]
                ),
            },
            "radius_cap": {
                "registry_path": self.registry_path,
                "required": self.radius_required,
                "quantile": self.radius_quantile,
                "dynamic_batch_quantile_allowed": False,
            },
            "selector": {
                "first": self.selector_first,
                "greedy": "maximum_minimum_weighted_pair_distance",
                "latent_cosine_weight": self.latent_cosine_weight,
                "prediction_rms_weight": self.prediction_rms_weight,
            },
            "smoke_allow_missing_radius_cap": self.smoke_allow_missing_radius_cap,
        }


@dataclass(frozen=True)
class RadiusCapEntry:
    entry_id: str
    entry_payload_sha256: str
    center: str
    pool_identity_sha256: str
    standardizer_sha256: str
    ordered_train_hash_ids_sha256: str
    pool_selection_identity: CanonicalPoolSourceIdentity
    vae_sha256: str
    calibration_seed: int
    seed_config_sha256: str
    augmix_config_sha256: str
    operator_profile_config_sha256: str
    endpoint_protocol_sha256: str
    calibration_algorithm_sha256: str
    sample_count: int
    standardizer_fit_count: int
    candidate_endpoints: int
    radius_observation_count: int
    p90_standardized_rms: float
    p95_standardized_rms: float

    def cap_for_quantile(self, quantile: float) -> float:
        if math.isclose(float(quantile), 0.90, abs_tol=1.0e-12):
            return self.p90_standardized_rms
        if math.isclose(float(quantile), 0.95, abs_tol=1.0e-12):
            return self.p95_standardized_rms
        raise ValueError("radius registry contains only P90 and P95")

    def describe(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entry_payload_sha256": self.entry_payload_sha256,
            "center": self.center,
            "pool_identity_sha256": self.pool_identity_sha256,
            "standardizer_sha256": self.standardizer_sha256,
            "ordered_train_hash_ids_sha256": self.ordered_train_hash_ids_sha256,
            "pool_selection_identity": self.pool_selection_identity.describe(),
            "vae_sha256": self.vae_sha256,
            "calibration_seed": self.calibration_seed,
            "seed_config_sha256": self.seed_config_sha256,
            "augmix_config_sha256": self.augmix_config_sha256,
            "operator_profile_config_sha256": self.operator_profile_config_sha256,
            "endpoint_protocol_sha256": self.endpoint_protocol_sha256,
            "calibration_algorithm_sha256": self.calibration_algorithm_sha256,
            "sample_count": self.sample_count,
            "standardizer_fit_count": self.standardizer_fit_count,
            "candidate_endpoints": self.candidate_endpoints,
            "radius_observation_count": self.radius_observation_count,
            "p90_standardized_rms": self.p90_standardized_rms,
            "p95_standardized_rms": self.p95_standardized_rms,
        }


@dataclass(frozen=True)
class RadiusCapRegistry:
    path: Path
    sha256: str
    allowed_depths: tuple[int, ...]
    depth_sampling: str
    entries: tuple[RadiusCapEntry, ...]

    def resolve(
        self,
        *,
        center: str,
        latent_pool: Any,
        candidate_endpoints: int,
        seed_config_sha256: str,
        augmix_config_sha256: str,
        operator_profile_config_sha256: str,
        endpoint_protocol_sha256: str,
        calibration_algorithm_sha256: str,
        calibration_seed: int,
    ) -> RadiusCapEntry:
        identity = getattr(latent_pool, "identity", None)
        standardizer = getattr(latent_pool, "standardizer", None)
        if identity is None or standardizer is None:
            raise TypeError("latent_pool lacks identity or standardizer")
        source_raw = getattr(latent_pool, "canonical_source_identity", None)
        if isinstance(source_raw, CanonicalPoolSourceIdentity):
            source_identity = source_raw
        elif isinstance(source_raw, Mapping):
            source_identity = CanonicalPoolSourceIdentity.from_mapping(source_raw)
        else:
            raise TypeError(
                "latent_pool lacks the canonical train400 loader selection identity"
            )
        exact = [
            entry
            for entry in self.entries
            if entry.center == center
            and entry.pool_identity_sha256 == str(identity.identity_sha256)
            and entry.standardizer_sha256 == str(standardizer.identity_sha256)
            and entry.ordered_train_hash_ids_sha256
            == str(identity.ordered_hash_ids_sha256)
            and entry.pool_selection_identity == source_identity
            and entry.vae_sha256 == str(identity.encoder_identity)
            and entry.sample_count == int(identity.record_count)
            and entry.standardizer_fit_count == int(standardizer.count)
            and entry.candidate_endpoints == int(candidate_endpoints)
            and entry.seed_config_sha256 == str(seed_config_sha256)
            and entry.augmix_config_sha256 == str(augmix_config_sha256)
            and entry.operator_profile_config_sha256
            == str(operator_profile_config_sha256)
            and entry.endpoint_protocol_sha256 == str(endpoint_protocol_sha256)
            and entry.calibration_algorithm_sha256
            == str(calibration_algorithm_sha256)
            and entry.calibration_seed == int(calibration_seed)
        ]
        if len(exact) != 1:
            raise KeyError(
                "radius cap registry requires exactly one matching train400 entry; "
                f"center={center!r}, matches={len(exact)}"
            )
        return exact[0]


def load_radius_cap_registry(path: str | Path) -> RadiusCapRegistry:
    """Load the strict train400-only radius-cap registry schema."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"radius cap registry not found: {resolved}")
    root = _mapping(
        yaml.safe_load(resolved.read_text(encoding="utf-8")),
        "radius cap registry",
    )
    expected_root = {
        "schema_version",
        "metric",
        "calibration_split",
        "candidate_endpoints",
        "allowed_depths",
        "entries_sha256",
        "entries",
    }
    permitted_root = expected_root | {"depth_sampling"}
    if not expected_root.issubset(root) or not set(root).issubset(permitted_root):
        raise ValueError(
            "radius registry keys are incomplete or unexpected"
        )
    if root["schema_version"] != RADIUS_REGISTRY_SCHEMA_VERSION:
        raise ValueError("unsupported radius registry schema version")
    if root["metric"] != RADIUS_METRIC:
        raise ValueError("radius registry metric mismatch")
    if root["calibration_split"] != "per_center_train400_only":
        raise ValueError("radius registry must be calibrated on per-center train400 only")
    allowed_depths = tuple(int(value) for value in root["allowed_depths"])
    depth_sampling = str(root.get("depth_sampling", "uniform_per_candidate"))
    if root["candidate_endpoints"] != 8 or (
        (allowed_depths, depth_sampling)
        not in {
            ((1, 2, 3), "uniform_per_candidate"),
            ((2, 3), "balanced_half_each"),
        }
    ):
        raise ValueError(
            "radius registry must use E=8 with a supported depth protocol"
        )
    raw_entries = root["entries"]
    if not isinstance(raw_entries, list):
        raise TypeError("radius registry entries must be a list")
    entries_sha256 = _require_sha256(
        root["entries_sha256"], "radius registry entries SHA256"
    )
    actual_entries_sha256 = _radius_entries_sha256(raw_entries)
    if entries_sha256 != actual_entries_sha256:
        raise ValueError(
            "radius registry entries SHA256 mismatch; entries were edited outside "
            "the calibration writer"
        )
    expected_entry = {
        "entry_id",
        "entry_payload_sha256",
        "center",
        "pool_identity_sha256",
        "standardizer_sha256",
        "ordered_train_hash_ids_sha256",
        "pool_selection_identity",
        "vae_sha256",
        "calibration_seed",
        "seed_config_sha256",
        "augmix_config_sha256",
        "operator_profile_config_sha256",
        "endpoint_protocol_sha256",
        "calibration_algorithm_sha256",
        "sample_count",
        "standardizer_fit_count",
        "candidate_endpoints",
        "radius_observation_count",
        "p90_standardized_rms",
        "p95_standardized_rms",
    }
    entries: list[RadiusCapEntry] = []
    for index, value in enumerate(raw_entries):
        item = _mapping(value, f"radius registry entries[{index}]")
        if set(item) != expected_entry:
            raise ValueError(
                f"radius registry entry keys must be exactly {sorted(expected_entry)}"
            )
        center = str(item["center"])
        if center not in KNOWN_CENTERS:
            raise ValueError(f"unknown radius registry center: {center!r}")
        digests = {
            name: str(item[name])
            for name in (
                "pool_identity_sha256",
                "standardizer_sha256",
                "ordered_train_hash_ids_sha256",
                "vae_sha256",
                "seed_config_sha256",
                "augmix_config_sha256",
                "operator_profile_config_sha256",
                "endpoint_protocol_sha256",
                "calibration_algorithm_sha256",
            )
        }
        digests = {
            name: _require_sha256(value, f"entries[{index}].{name}")
            for name, value in digests.items()
        }
        entry_payload_sha256 = _require_sha256(
            item["entry_payload_sha256"],
            f"entries[{index}].entry_payload_sha256",
        )
        actual_entry_payload_sha256 = _radius_entry_payload_sha256(item)
        if entry_payload_sha256 != actual_entry_payload_sha256:
            raise ValueError(
                f"radius registry entry payload SHA256 mismatch at index {index}; "
                "P90/P95 or provenance was edited outside calibration"
            )
        entry_id = str(item["entry_id"])
        expected_entry_id = _radius_entry_id(entry_payload_sha256)
        if entry_id != expected_entry_id:
            raise ValueError(
                f"radius registry entry_id mismatch at index {index}: "
                f"expected={expected_entry_id!r}, actual={entry_id!r}"
            )
        source_identity = CanonicalPoolSourceIdentity.from_mapping(
            _mapping(
                item["pool_selection_identity"],
                f"entries[{index}].pool_selection_identity",
            )
        )
        if set(source_identity.describe()) != POOL_SOURCE_IDENTITY_KEYS:
            raise RuntimeError("canonical pool source identity schema drift")
        if (
            source_identity.partition != "k500_tune_train"
            or source_identity.center != center
            or source_identity.record_count != 400
        ):
            raise ValueError(
                "radius entry pool selection must be the same-center train400"
            )
        sample_count = _strict_int(
            item["sample_count"], f"entries[{index}].sample_count", minimum=1
        )
        candidate_count = _strict_int(
            item["candidate_endpoints"],
            f"entries[{index}].candidate_endpoints",
            minimum=1,
        )
        observation_count = _strict_int(
            item["radius_observation_count"],
            f"entries[{index}].radius_observation_count",
            minimum=1,
        )
        if sample_count != 400 or candidate_count != 8 or observation_count != 3200:
            raise ValueError("each radius entry must describe 400 x 8 observations")
        standardizer_fit_count = _strict_int(
            item["standardizer_fit_count"],
            f"entries[{index}].standardizer_fit_count",
            minimum=2,
        )
        if standardizer_fit_count > sample_count:
            raise ValueError("standardizer_fit_count cannot exceed sample_count")
        p90 = _finite_float(
            item["p90_standardized_rms"], f"entries[{index}].p90_standardized_rms"
        )
        p95 = _finite_float(
            item["p95_standardized_rms"], f"entries[{index}].p95_standardized_rms"
        )
        if not 0.0 < p90 <= p95:
            raise ValueError("radius registry requires 0 < P90 <= P95")
        entries.append(
            RadiusCapEntry(
                entry_id=entry_id,
                entry_payload_sha256=entry_payload_sha256,
                center=center,
                pool_identity_sha256=digests["pool_identity_sha256"],
                standardizer_sha256=digests["standardizer_sha256"],
                ordered_train_hash_ids_sha256=digests[
                    "ordered_train_hash_ids_sha256"
                ],
                pool_selection_identity=source_identity,
                vae_sha256=digests["vae_sha256"],
                calibration_seed=_strict_int(
                    item["calibration_seed"],
                    f"entries[{index}].calibration_seed",
                    minimum=0,
                ),
                seed_config_sha256=digests["seed_config_sha256"],
                augmix_config_sha256=digests["augmix_config_sha256"],
                operator_profile_config_sha256=digests[
                    "operator_profile_config_sha256"
                ],
                endpoint_protocol_sha256=digests["endpoint_protocol_sha256"],
                calibration_algorithm_sha256=digests[
                    "calibration_algorithm_sha256"
                ],
                sample_count=sample_count,
                standardizer_fit_count=standardizer_fit_count,
                candidate_endpoints=candidate_count,
                radius_observation_count=observation_count,
                p90_standardized_rms=p90,
                p95_standardized_rms=p95,
            )
        )
    identities = [entry.entry_id for entry in entries]
    if len(set(identities)) != len(identities):
        raise ValueError("radius registry entry_id values must be unique")
    centers = [entry.center for entry in entries]
    if centers != sorted(centers):
        raise ValueError("radius registry entries must be sorted by center")
    if len(set(centers)) != len(centers):
        raise ValueError("radius registry may contain at most one entry per center")
    return RadiusCapRegistry(
        path=resolved,
        sha256=_sha256(resolved),
        allowed_depths=allowed_depths,
        depth_sampling=depth_sampling,
        entries=tuple(entries),
    )


@dataclass(frozen=True)
class SharedEndpointPool:
    clean_standardized: torch.Tensor
    selected_standardized: torch.Tensor
    selected_raw: torch.Tensor
    endpoint_residuals: torch.Tensor
    selection: DiversitySelection
    eligible_mask: torch.Tensor
    selected_finite: torch.Tensor
    candidate_depths: torch.Tensor
    candidate_operator_mask: torch.Tensor
    candidate_nonfinite_count: torch.Tensor
    candidate_radii_rms: torch.Tensor
    clean_probabilities: torch.Tensor
    candidate_probabilities: torch.Tensor
    candidate_bce: torch.Tensor
    radius_cap_rms: float | None
    radius_entry: RadiusCapEntry | None
    center: str | None
    endpoint_seed: int


@dataclass(frozen=True)
class ViewCandidate:
    waveform: torch.Tensor
    projection: LatentProjection
    initial_bce: torch.Tensor
    final_bce: torch.Tensor
    search_finite: torch.Tensor
    final_probabilities: torch.Tensor | None


def _hard_vs_random_diagnostics(
    random_initial_bce: torch.Tensor,
    hard_final_bce: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[dict[str, float], dict[str, int]]:
    """Compare each hard result with its exact random search initialization."""

    if (
        random_initial_bce.shape != hard_final_bce.shape
        or random_initial_bce.ndim != 1
        or hard_final_bce.device != random_initial_bce.device
    ):
        raise ValueError("hard/random BCE tensors must be aligned rank-1 tensors")
    finite = (
        valid_mask
        & torch.isfinite(random_initial_bce)
        & torch.isfinite(hard_final_bce)
    )
    random_loss = random_initial_bce[finite].float()
    hard_loss = hard_final_bce[finite].float()
    gain = hard_loss - random_loss
    count = int(gain.numel())
    values = {"hard_vs_random_batch_record_count": float(count)}
    weights = {"hard_vs_random_batch_record_count": 1}
    if count == 0:
        return values, weights
    statistics = {
        "random_initial_bce_mean": random_loss.mean(),
        "random_initial_bce_batch_median": torch.quantile(random_loss, 0.50),
        "random_initial_bce_batch_p90": torch.quantile(random_loss, 0.90),
        "hard_final_bce_mean": hard_loss.mean(),
        "hard_final_bce_batch_median": torch.quantile(hard_loss, 0.50),
        "hard_final_bce_batch_p90": torch.quantile(hard_loss, 0.90),
        "hard_minus_random_bce_gain_mean": gain.mean(),
        "hard_minus_random_bce_gain_batch_median": torch.quantile(gain, 0.50),
        "hard_minus_random_bce_gain_batch_p90": torch.quantile(gain, 0.90),
        "hard_strictly_greater_than_random_fraction": (gain > 0.0).float().mean(),
    }
    for key, scalar in statistics.items():
        values[key] = float(scalar.detach().cpu())
        weights[key] = count
    return values, weights


def _clean_correct_multilabel_attack_diagnostics(
    clean_probabilities: torch.Tensor,
    hard_probabilities: torch.Tensor,
    targets: torch.Tensor,
    accepted: torch.Tensor,
    *,
    threshold: float = MULTILABEL_DECISION_THRESHOLD,
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute thresholded multilabel attack rates with explicit denominators."""

    if (
        clean_probabilities.ndim != 2
        or clean_probabilities.shape != hard_probabilities.shape
        or clean_probabilities.shape != targets.shape
        or clean_probabilities.shape[1] != 5
        or clean_probabilities.device != hard_probabilities.device
        or clean_probabilities.device != targets.device
    ):
        raise ValueError("clean/hard probabilities and targets must align as (B,5)")
    if (
        accepted.dtype != torch.bool
        or accepted.shape != (clean_probabilities.shape[0],)
        or accepted.device != clean_probabilities.device
    ):
        raise ValueError("accepted must be an aligned bool tensor")
    resolved_threshold = _finite_float(threshold, "multilabel decision threshold")
    if not 0.0 < resolved_threshold < 1.0:
        raise ValueError("multilabel decision threshold must lie in (0,1)")

    finite = (
        torch.isfinite(clean_probabilities).all(dim=1)
        & torch.isfinite(hard_probabilities).all(dim=1)
        & torch.isfinite(targets).all(dim=1)
    )
    target_positive = targets >= resolved_threshold
    clean_positive = clean_probabilities >= resolved_threshold
    hard_positive = hard_probabilities >= resolved_threshold
    clean_correct = (clean_positive == target_positive).all(dim=1)
    denominator_records = accepted & finite & clean_correct
    hard_record_wrong = (hard_positive != target_positive).any(dim=1)

    asr_denominator = int(denominator_records.sum().detach().cpu())
    asr_numerator = int(
        (denominator_records & hard_record_wrong).sum().detach().cpu()
    )
    positive_denominator_mask = denominator_records.unsqueeze(1) & target_positive
    negative_denominator_mask = denominator_records.unsqueeze(1) & ~target_positive
    positive_hide_mask = positive_denominator_mask & ~hard_positive
    negative_add_mask = negative_denominator_mask & hard_positive
    positive_denominator = int(positive_denominator_mask.sum().detach().cpu())
    positive_numerator = int(positive_hide_mask.sum().detach().cpu())
    negative_denominator = int(negative_denominator_mask.sum().detach().cpu())
    negative_numerator = int(negative_add_mask.sum().detach().cpu())

    values = {
        "multilabel_decision_threshold": resolved_threshold,
        "clean_correct_multilabel_asr_denominator_batch_count": float(
            asr_denominator
        ),
        "clean_correct_multilabel_asr_numerator_batch_count": float(asr_numerator),
        "positive_hide_denominator_batch_count": float(positive_denominator),
        "positive_hide_numerator_batch_count": float(positive_numerator),
        "negative_add_denominator_batch_count": float(negative_denominator),
        "negative_add_numerator_batch_count": float(negative_numerator),
    }
    weights = {key: 1 for key in values}
    if asr_denominator > 0:
        values["clean_correct_multilabel_asr"] = (
            asr_numerator / asr_denominator
        )
        weights["clean_correct_multilabel_asr"] = asr_denominator
    if positive_denominator > 0:
        values["positive_hide_rate"] = positive_numerator / positive_denominator
        weights["positive_hide_rate"] = positive_denominator
    if negative_denominator > 0:
        values["negative_add_rate"] = negative_numerator / negative_denominator
        weights["negative_add_rate"] = negative_denominator
    return values, weights


def _mixing_radius_diagnostics(
    projection: LatentProjection,
    accepted: torch.Tensor,
) -> tuple[dict[str, float], dict[str, int]]:
    """Summarize raw/effective mixing radius and inclusive boundary bands."""

    finite = (
        accepted
        & torch.isfinite(projection.raw_m)
        & torch.isfinite(projection.effective_m)
    )
    raw = projection.raw_m[finite].float()
    effective = projection.effective_m[finite].float()
    count = int(raw.numel())
    values = {
        "mixing_m_batch_record_count": float(count),
        "mixing_m_low_boundary_inclusive": M_LOW_BOUNDARY_INCLUSIVE,
        "mixing_m_high_boundary_inclusive": M_HIGH_BOUNDARY_INCLUSIVE,
    }
    weights = {key: 1 for key in values}
    if count == 0:
        return values, weights
    statistics = {
        "raw_m_mean": raw.mean(),
        "raw_m_batch_q05": torch.quantile(raw, 0.05),
        "raw_m_batch_median": torch.quantile(raw, 0.50),
        "raw_m_batch_q95": torch.quantile(raw, 0.95),
        "raw_m_low_boundary_hit_fraction": (
            raw <= M_LOW_BOUNDARY_INCLUSIVE
        ).float().mean(),
        "raw_m_high_boundary_hit_fraction": (
            raw >= M_HIGH_BOUNDARY_INCLUSIVE
        ).float().mean(),
        "effective_m_mean": effective.mean(),
        "effective_m_batch_q05": torch.quantile(effective, 0.05),
        "effective_m_batch_median": torch.quantile(effective, 0.50),
        "effective_m_batch_q95": torch.quantile(effective, 0.95),
        "effective_m_low_boundary_hit_fraction": (
            effective <= M_LOW_BOUNDARY_INCLUSIVE
        ).float().mean(),
        "effective_m_high_boundary_hit_fraction": (
            effective >= M_HIGH_BOUNDARY_INCLUSIVE
        ).float().mean(),
    }
    for key, scalar in statistics.items():
        values[key] = float(scalar.detach().cpu())
        weights[key] = count
    return values, weights


def _frozen_classifier_for_attack(module: nn.Module) -> Iterator[None]:
    """Temporarily freeze classifier parameters while preserving input gradients."""

    @contextmanager
    def manager() -> Iterator[None]:
        module_states = tuple((child, bool(child.training)) for child in module.modules())
        parameter_states = tuple(
            (parameter, bool(parameter.requires_grad)) for parameter in module.parameters()
        )
        module.eval()
        for parameter, _ in parameter_states:
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, requires_grad in parameter_states:
                parameter.requires_grad_(requires_grad)
            for child, training in module_states:
                child.training = training

    return manager()


@contextmanager
def _frozen_vae_components(*modules: nn.Module) -> Iterator[None]:
    """Use deterministic frozen VAE components and restore every input state."""

    module_states: list[tuple[nn.Module, bool]] = []
    parameter_states: list[tuple[torch.nn.Parameter, bool]] = []
    for module in modules:
        module_states.extend(
            (child, bool(child.training)) for child in module.modules()
        )
        parameter_states.extend(
            (parameter, bool(parameter.requires_grad))
            for parameter in module.parameters()
        )
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in parameter_states:
            parameter.requires_grad_(requires_grad)
        for child, training in module_states:
            child.training = training


def _quality_mask(
    waveform: torch.Tensor,
    *,
    minimum_std_mV: float,
    maximum_abs_mV: float,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    flat = waveform.flatten(1)
    finite = torch.isfinite(flat).all(dim=1)
    safe = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    standard_deviation = safe.std(dim=1, correction=0)
    maximum_absolute = safe.abs().amax(dim=1)
    accepted = finite & (standard_deviation >= float(minimum_std_mV)) & (
        maximum_absolute <= float(maximum_abs_mV)
    )
    rows = torch.stack(
        (finite.float(), standard_deviation, maximum_absolute, accepted.float()), dim=1
    ).detach().cpu().tolist()
    reasons: list[str] = []
    for finite_value, std_value, maximum_value, accepted_value in rows:
        if bool(accepted_value):
            reasons.append("accepted")
            continue
        failures: list[str] = []
        if not bool(finite_value):
            failures.append("nonfinite")
        if float(std_value) < float(minimum_std_mV):
            failures.append("flatline")
        if float(maximum_value) > float(maximum_abs_mV):
            failures.append("severe_amplitude")
        reasons.append("+".join(failures))
    return accepted, tuple(reasons)


def _classifier_logits_and_loss(
    waveform: torch.Tensor,
    targets: torch.Tensor,
    classifier: nn.Module,
    model_spec: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    waveform_finite = torch.isfinite(waveform).flatten(1).all(dim=1)
    safe_waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    model_input = prepare_canonical_model_input(safe_waveform, model_spec)
    logits = validate_model_output(
        classifier(model_input),
        model_spec,
        batch_size=int(targets.shape[0]),
        check_finite=False,
    )
    logits_finite = torch.isfinite(logits).all(dim=1)
    safe_logits = torch.nan_to_num(
        logits.float(), nan=0.0, posinf=30.0, neginf=-30.0
    )
    loss = F.binary_cross_entropy_with_logits(
        safe_logits, targets.float(), reduction="none"
    ).mean(dim=1)
    finite = waveform_finite & logits_finite & torch.isfinite(loss)
    return logits, torch.where(finite, loss, torch.zeros_like(loss)), finite


def _classifier_probabilities(
    waveform: torch.Tensor,
    classifier: nn.Module,
    model_spec: Any,
    *,
    chunk_size: int = CLASSIFIER_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities: list[torch.Tensor] = []
    logits_values: list[torch.Tensor] = []
    finite_masks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(waveform.shape[0]), int(chunk_size)):
            chunk = waveform[start : start + int(chunk_size)]
            waveform_finite = torch.isfinite(chunk).flatten(1).all(dim=1)
            safe = torch.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            logits = validate_model_output(
                classifier(prepare_canonical_model_input(safe, model_spec)),
                model_spec,
                batch_size=int(chunk.shape[0]),
                check_finite=False,
            )
            finite = waveform_finite & torch.isfinite(logits).all(dim=1)
            safe_logits = torch.nan_to_num(
                logits.float(), nan=0.0, posinf=30.0, neginf=-30.0
            ).detach()
            logits_values.append(safe_logits)
            probabilities.append(torch.sigmoid(safe_logits))
            finite_masks.append(finite.detach())
    return (
        torch.cat(probabilities, dim=0),
        torch.cat(logits_values, dim=0),
        torch.cat(finite_masks, dim=0),
    )


@dataclass(frozen=True)
class GeneralizedCorruptionBatch:
    """Sandbox-owned depth-1/2/3 extension of the whitelist corruption kernel."""

    waveform_raw_100hz: torch.Tensor
    depth: torch.Tensor
    operator_mask: torch.Tensor
    output_nonfinite_count: torch.Tensor


def generate_generalized_corruption(
    clean_raw_100hz: torch.Tensor,
    *,
    operator_params: Mapping[str, Mapping[str, Any]],
    generator: torch.Generator,
    operator_mask: torch.Tensor,
) -> GeneralizedCorruptionBatch:
    """Apply explicit depth-1/2/3 masks with whitelist domain conventions.

    This is intentionally sandbox-owned because the whitelist public kernel
    accepts only the official depth-2/3 composition table.  It uses the exact
    whitelist operator order, 100->500->100 linear interpolation settings and
    Torch operator implementation.  It does not patch module globals.
    """

    if (
        not isinstance(clean_raw_100hz, torch.Tensor)
        or clean_raw_100hz.ndim != 3
        or tuple(clean_raw_100hz.shape[1:]) != (OUTPUT_POINTS, 12)
    ):
        raise ValueError("clean_raw_100hz must have shape (B,1000,12)")
    if not clean_raw_100hz.is_floating_point() or not bool(
        torch.isfinite(clean_raw_100hz).all()
    ):
        raise ValueError("clean_raw_100hz must be finite floating point")
    if not isinstance(generator, torch.Generator) or torch.device(
        generator.device
    ).type != clean_raw_100hz.device.type:
        raise ValueError("generator device must match waveform device")
    if (
        not isinstance(operator_mask, torch.Tensor)
        or operator_mask.dtype != torch.bool
        or operator_mask.shape
        != (int(clean_raw_100hz.shape[0]), len(CANONICAL_OPERATORS))
        or operator_mask.device != clean_raw_100hz.device
    ):
        raise ValueError("operator_mask must be bool (B,5) on waveform device")
    depths = operator_mask.sum(dim=1, dtype=torch.int64)
    if bool(((depths < 1) | (depths > 3)).any()):
        raise ValueError("generalized corruption depths must lie in [1,3]")
    if set(operator_params) != set(CANONICAL_OPERATORS):
        raise ValueError("operator_params must contain the canonical five operators")
    waveform = F.interpolate(
        clean_raw_100hz.to(dtype=torch.float32).transpose(1, 2),
        size=CORRUPTION_DOMAIN_POINTS,
        mode=INTERPOLATION_MODE,
        align_corners=INTERPOLATION_ALIGN_CORNERS,
    ).transpose(1, 2).contiguous()
    for operator_index, operator in enumerate(CANONICAL_OPERATORS):
        candidate = apply_operator_batch_prevalidated(
            operator,
            waveform,
            params=dict(operator_params[operator]),
            sampling_rate_hz=CORRUPTION_DOMAIN_SAMPLING_RATE_HZ,
            rng=generator,
        )
        waveform = torch.where(
            operator_mask[:, operator_index].view(-1, 1, 1),
            candidate,
            waveform,
        )
    output = F.interpolate(
        waveform.transpose(1, 2),
        size=OUTPUT_POINTS,
        mode=INTERPOLATION_MODE,
        align_corners=INTERPOLATION_ALIGN_CORNERS,
    ).transpose(1, 2).contiguous()
    nonfinite = (~torch.isfinite(output)).sum(dim=(1, 2), dtype=torch.int64)
    return GeneralizedCorruptionBatch(
        waveform_raw_100hz=torch.nan_to_num(
            output, nan=0.0, posinf=0.0, neginf=0.0
        ).contiguous(),
        depth=depths.contiguous(),
        operator_mask=operator_mask.contiguous(),
        output_nonfinite_count=nonfinite.contiguous(),
    )


def probe_generalized_corruption_parity(
    clean_raw_100hz: torch.Tensor,
    *,
    operator_params: Mapping[str, Mapping[str, Any]],
    composition_index: int,
    seed: int,
) -> dict[str, Any]:
    """Compare one fixed depth-2/3 mask against the whitelist public kernel."""

    index = _strict_int(composition_index, "composition_index", minimum=0)
    if index >= len(COMPOSITIONS):
        raise ValueError("composition_index is outside the whitelist table")
    first = _generator(clean_raw_100hz.device, int(seed))
    second = _generator(clean_raw_100hz.device, int(seed))
    indices = torch.full(
        (int(clean_raw_100hz.shape[0]),),
        index,
        device=clean_raw_100hz.device,
        dtype=torch.int64,
    )
    canonical = generate_canonical_corruption(
        clean_raw_100hz,
        operator_params=operator_params,
        generator=first,
        composition_indices=indices,
    )
    composition = COMPOSITIONS[index]
    mask = torch.tensor(
        [operator in composition for operator in CANONICAL_OPERATORS],
        device=clean_raw_100hz.device,
        dtype=torch.bool,
    ).unsqueeze(0).expand(int(clean_raw_100hz.shape[0]), -1)
    generalized = generate_generalized_corruption(
        clean_raw_100hz,
        operator_params=operator_params,
        generator=second,
        operator_mask=mask,
    )
    difference = (
        canonical.waveform_raw_100hz - generalized.waveform_raw_100hz
    ).abs()
    return {
        "allclose": bool(torch.equal(
            canonical.waveform_raw_100hz, generalized.waveform_raw_100hz
        )),
        "maximum_absolute_difference": float(difference.max().detach().cpu()),
        "operator_mask_equal": bool(
            torch.equal(canonical.diagnostics.operator_mask, generalized.operator_mask)
        ),
        "depth_equal": bool(
            torch.equal(canonical.diagnostics.depth, generalized.depth)
        ),
        "nonfinite_count_equal": bool(
            torch.equal(
                canonical.diagnostics.output_nonfinite_count,
                generalized.output_nonfinite_count,
            )
        ),
    }


def _generate_candidate_waveforms(
    clean: torch.Tensor,
    *,
    candidate_count: int,
    allowed_depths: tuple[int, ...],
    depth_sampling: str = "uniform_per_candidate",
    augmix_config: AugMixConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate E candidates with bounded 500-Hz working-set chunks."""

    batch_size = int(clean.shape[0])
    if depth_sampling == "uniform_per_candidate":
        depth_indices = torch.randint(
            0,
            len(allowed_depths),
            (batch_size, candidate_count),
            device=clean.device,
            dtype=torch.int64,
            generator=generator,
        )
        depth_values = torch.tensor(
            allowed_depths, device=clean.device, dtype=torch.int64
        )
        depths = depth_values[depth_indices]
    elif depth_sampling == "balanced_half_each":
        if tuple(allowed_depths) != (2, 3) or candidate_count % 2:
            raise ValueError(
                "balanced_half_each requires depths (2,3) and an even candidate count"
            )
        base_depths = torch.tensor(
            [2] * (candidate_count // 2) + [3] * (candidate_count // 2),
            device=clean.device,
            dtype=torch.int64,
        ).unsqueeze(0).expand(batch_size, -1)
        depth_permutation = torch.rand(
            batch_size,
            candidate_count,
            device=clean.device,
            dtype=torch.float32,
            generator=generator,
        ).argsort(dim=1)
        depths = torch.gather(base_depths, 1, depth_permutation)
    else:
        raise ValueError(f"unsupported depth_sampling policy: {depth_sampling!r}")
    random_order = torch.rand(
        batch_size,
        candidate_count,
        len(CANONICAL_OPERATORS),
        device=clean.device,
        dtype=torch.float32,
        generator=generator,
    ).argsort(dim=2)
    operator_mask = random_order.argsort(dim=2) < depths.unsqueeze(2)
    parameters = {
        operator: augmix_config.operator_profile_config.parameters_for(operator)
        for operator in CANONICAL_OPERATORS
    }
    chunks: list[torch.Tensor] = []
    nonfinite_chunks: list[torch.Tensor] = []
    for start in range(0, candidate_count, ENDPOINT_CHUNK_SIZE):
        stop = min(candidate_count, start + ENDPOINT_CHUNK_SIZE)
        width = stop - start
        flat = clean.unsqueeze(1).expand(-1, width, -1, -1).reshape(
            batch_size * width, OUTPUT_POINTS, 12
        )
        mask_chunk = operator_mask[:, start:stop].reshape(
            batch_size * width, len(CANONICAL_OPERATORS)
        )
        generated = generate_generalized_corruption(
            flat,
            operator_params=parameters,
            generator=generator,
            operator_mask=mask_chunk,
        )
        chunks.append(
            generated.waveform_raw_100hz.reshape(
                batch_size, width, OUTPUT_POINTS, 12
            )
        )
        nonfinite_chunks.append(
            generated.output_nonfinite_count.reshape(batch_size, width)
        )
    return (
        torch.cat(chunks, dim=1).contiguous(),
        depths.contiguous(),
        operator_mask.contiguous(),
        torch.cat(nonfinite_chunks, dim=1).contiguous(),
    )


def _encode_decode_waveforms(
    waveform: torch.Tensor,
    *,
    encoder: nn.Module,
    decoder: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode posterior means and decode in chunks, preserving input order."""

    latents: list[torch.Tensor] = []
    decoded: list[torch.Tensor] = []
    finite_masks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(waveform.shape[0]), CLASSIFIER_CHUNK_SIZE):
            chunk = waveform[start : start + CLASSIFIER_CHUNK_SIZE]
            encoded = encoder(prepare_ecgtwin_encoder_input(chunk), sample=False)
            if not isinstance(encoded, tuple) or len(encoded) != 3:
                raise TypeError(
                    "VAE encoder must return (scaled_latent, mean, log_variance)"
                )
            latent = encoded[0]
            if latent.ndim != 3 or tuple(latent.shape[1:]) != (4, 128):
                raise ValueError("VAE encoder returned an unexpected latent shape")
            reconstruction = decode_to_ptbxl_waveform(
                decoder, latent, target_points=OUTPUT_POINTS
            )
            finite = (
                torch.isfinite(latent).flatten(1).all(dim=1)
                & torch.isfinite(reconstruction).flatten(1).all(dim=1)
                & torch.isfinite(chunk).flatten(1).all(dim=1)
            )
            latents.append(torch.nan_to_num(latent).detach())
            decoded.append(torch.nan_to_num(reconstruction).detach())
            finite_masks.append(finite.detach())
    return (
        torch.cat(latents, dim=0).contiguous(),
        torch.cat(decoded, dim=0).contiguous(),
        torch.cat(finite_masks, dim=0).contiguous(),
    )


def _subset_geometry(
    value: StandardizedEndpointGeometry, mask: torch.Tensor
) -> StandardizedEndpointGeometry:
    return StandardizedEndpointGeometry(
        clean=value.clean[mask],
        endpoints=value.endpoints[mask],
        deltas=value.deltas[mask],
        unit_directions=value.unit_directions[mask],
        radii_rms=value.radii_rms[mask],
        finite_mask=value.finite_mask[mask],
    )


def _gather_candidate(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    shape = (indices.shape[0], indices.shape[1]) + (1,) * (value.ndim - 2)
    expanded = indices.view(shape).expand(indices.shape[0], indices.shape[1], *value.shape[2:])
    return torch.gather(value, 1, expanded).contiguous()


class DiverseLatentAugMaxRuntime(MethodViewRuntime):
    """One-batch shared endpoint runtime for the isolated D0-D6 arms."""

    def __init__(
        self,
        method: CompiledMethod,
        *,
        model_name: str,
        config_root: Path,
        latent_pool: Any | None,
        encoder: nn.Module | None,
        decoder: nn.Module | None,
        minimum_std_mV: float,
        maximum_abs_mV: float,
    ) -> None:
        super().__init__(
            method,
            model_name=model_name,
            config_root=config_root,
            latent_pool=latent_pool,
            encoder=encoder,
            decoder=decoder,
            minimum_std_mV=minimum_std_mV,
            maximum_abs_mV=maximum_abs_mV,
        )
        self.diverse = DiverseAugMaxContract.from_method(method)
        if latent_pool is None or encoder is None or decoder is None:
            raise ValueError(
                "Diverse AugMax requires train400 latent_pool, encoder and decoder"
            )
        identity = getattr(latent_pool, "identity", None)
        standardizer = getattr(latent_pool, "standardizer", None)
        if identity is None or standardizer is None:
            raise TypeError("latent_pool lacks managed identity/standardizer")
        source_identity = getattr(latent_pool, "canonical_source_identity", None)
        if not isinstance(source_identity, CanonicalPoolSourceIdentity):
            raise TypeError("Diverse AugMax pool lacks canonical source identity")
        if (
            source_identity.partition != "k500_tune_train"
            or source_identity.record_count != 400
        ):
            raise ValueError("Diverse AugMax geometry source must be frozen train400")
        if int(identity.record_count) != 400:
            raise ValueError("Diverse AugMax requires the per-center train400 pool")
        if not 2 <= int(standardizer.count) <= 400:
            raise ValueError("train400 latent standardizer fit count is invalid")
        if str(identity.standardizer_sha256) != str(standardizer.identity_sha256):
            raise ValueError("latent pool standardizer identity is inconsistent")
        if self.augmix_config is None:
            raise ValueError("Diverse AugMax requires the managed AugMix config")
        if self.augmix_config.latent_threechain_posterior_sample:
            raise ValueError("Diverse endpoints must use deterministic posterior means")
        if not self.augmix_config.latent_threechain_reconstruction_residual_bypass:
            raise ValueError("Diverse AugMax requires reconstruction residual bypass")
        if self.augmix_config.latent_threechain_post_decode_clean_beta_mix:
            raise ValueError("Diverse AugMax forbids post-decode clean Beta mixing")
        if method.profile_name == D6_METHOD_ID:
            expected_objective = {
                ("bce", ("clean_view",)): 1.0,
                ("bce", ("endpoint_1",)): 4.0 / 15.0,
                ("bce", ("endpoint_2",)): 4.0 / 15.0,
                ("bce", ("endpoint_3",)): 4.0 / 15.0,
                ("bce", ("hard_view",)): 0.2,
            }
            expected_bn_weights = {
                "clean_view": 0.5,
                "endpoint_1": 2.0 / 15.0,
                "endpoint_2": 2.0 / 15.0,
                "endpoint_3": 2.0 / 15.0,
                "hard_view": 0.1,
            }
        else:
            expected_objective = {
                ("bce", ("clean_view",)): 1.0,
                ("bce", ("view_1",)): 0.5,
                ("bce", ("view_2",)): 0.5,
                ("bernoulli_jsd", ("clean_view", "view_1", "view_2")): 2.0,
            }
            expected_bn_weights = {
                "clean_view": 0.5,
                "view_1": 0.25,
                "view_2": 0.25,
            }
        actual_objective = {
            (term.kind, tuple(term.views)): float(term.weight)
            for term in method.objective.terms
        }
        if actual_objective != expected_objective:
            raise ValueError(
                "Diverse AugMax objective does not match its frozen arm contract"
            )
        if method.contracts.get("objective_global_scale") != OBJECTIVE_GLOBAL_SCALE:
            raise ValueError("contracts.objective_global_scale must be 0.5")
        actual_bn_weights = _mapping(
            method.contracts.get("batch_norm_objective_view_weights"),
            "batch_norm_objective_view_weights",
        )
        if set(actual_bn_weights) != set(expected_bn_weights) or any(
            not math.isclose(
                float(actual_bn_weights[name]),
                expected_bn_weights[name],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for name in expected_bn_weights
        ):
            raise ValueError(
                "Diverse AugMax BN view weights do not match the arm objective"
            )
        self._radius_registry_path = resolve_config_reference(
            self.diverse.registry_path,
            owner_config_path=method.source_path,
            config_root=self.config_root,
            description="contracts.diverse_augmax.radius_cap.registry_path",
            must_exist=False,
        )
        self._radius_registry: RadiusCapRegistry | None = None
        if self._radius_registry_path.is_file():
            self._radius_registry = load_radius_cap_registry(self._radius_registry_path)
            if (
                self._radius_registry.allowed_depths != self.diverse.allowed_depths
                or self._radius_registry.depth_sampling
                != self.diverse.depth_sampling
            ):
                raise ValueError(
                    "radius registry depth protocol does not match the method"
                )
        elif not self.diverse.smoke_allow_missing_radius_cap:
            raise FileNotFoundError(
                f"required radius cap registry is missing: {self._radius_registry_path}"
            )
        augmix_path = self._resource_paths.get("augmix_config")
        if augmix_path is None:
            raise ValueError("Diverse AugMax lacks a resolved augmix_config path")
        self._augmix_config_sha256 = _sha256(augmix_path)
        self._operator_profile_config_sha256 = str(
            self.augmix_config.operator_profile_config.config_sha256
        )
        self._endpoint_protocol_sha256 = _endpoint_protocol_sha256()
        self._calibration_algorithm_sha256 = _calibration_algorithm_sha256()
        seed_paths = set(self._rng_seed_paths.values())
        if len(seed_paths) != 1:
            raise ValueError("Diverse AugMax requires exactly one RNG seed config")
        self._seed_config_sha256 = _sha256(next(iter(seed_paths)))
        self._search_chunk_size = 8 if "ecgfounder" in self.model_name.lower() else 16
        self._shared_cache: SharedEndpointPool | None = None
        self._generation_state: tuple[int, tuple[str, ...]] | None = None
        self._role_ordinals: dict[str, int] = {}

    @property
    def shared_cache_active(self) -> bool:
        return self._shared_cache is not None

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["sandbox_diverse_latent_augmax"] = {
            "contract": self.diverse.describe(),
            "runtime_adapter_path": str(Path(__file__).resolve()),
            "runtime_adapter_sha256": _sha256(Path(__file__).resolve()),
            "geometry_path": str((SANDBOX_ROOT / "geometry.py").resolve()),
            "geometry_sha256": _sha256((SANDBOX_ROOT / "geometry.py").resolve()),
            "train400_standardizer": self.latent_pool.standardizer.describe(),
            "latent_pool_identity": self.latent_pool.identity.describe(),
            "canonical_pool_source_identity": (
                self.latent_pool.canonical_source_identity.describe()
            ),
            "radius_registry": {
                "path": str(self._radius_registry_path),
                "present": self._radius_registry is not None,
                "sha256": (
                    None if self._radius_registry is None else self._radius_registry.sha256
                ),
                "calibration_algorithm_sha256": (
                    self._calibration_algorithm_sha256
                ),
            },
            "endpoint_cache_scope": "one_generate_call_two_views_then_clear",
            "endpoint_chunk_size": ENDPOINT_CHUNK_SIZE,
            "hard_search_chunk_size": self._search_chunk_size,
            "objective_global_scale": OBJECTIVE_GLOBAL_SCALE,
            "effective_objective_weights": {
                term.name: float(term.weight) * OBJECTIVE_GLOBAL_SCALE
                for term in self.method.objective.terms
            },
            "diagnostic_contract": _diagnostic_contract_description(),
        }
        return base

    def _center_from_identity(self, rng_identity: Sequence[str]) -> str | None:
        matches = [str(value) for value in rng_identity if str(value) in KNOWN_CENTERS]
        if len(matches) == 1:
            return matches[0]
        if not matches and self.diverse.smoke_allow_missing_radius_cap:
            return None
        raise ValueError(
            "rng_identity must contain exactly one canonical PN2021 center"
        )

    def _resolve_radius_cap(
        self, center: str | None
    ) -> tuple[float | None, RadiusCapEntry | None]:
        if self._radius_registry is None or center is None:
            if self.diverse.smoke_allow_missing_radius_cap:
                return None, None
            raise RuntimeError("radius cap is unavailable outside smoke mode")
        try:
            seed_config_path = next(iter(self._rng_seed_paths.values()))
            expected_calibration_seed = derive_seed(
                RADIUS_CALIBRATION_NAMESPACE,
                center,
                "train400",
                "E=8",
                (
                    "depths=2,3;sampling=balanced_half_each"
                    if self.diverse.depth_sampling == "balanced_half_each"
                    else "depths=1,2,3;sampling=uniform_per_candidate"
                ),
                config_path=seed_config_path,
            )
            entry = self._radius_registry.resolve(
                center=center,
                latent_pool=self.latent_pool,
                candidate_endpoints=self.diverse.candidate_endpoints,
                seed_config_sha256=self._seed_config_sha256,
                augmix_config_sha256=self._augmix_config_sha256,
                operator_profile_config_sha256=(
                    self._operator_profile_config_sha256
                ),
                endpoint_protocol_sha256=self._endpoint_protocol_sha256,
                calibration_algorithm_sha256=(
                    self._calibration_algorithm_sha256
                ),
                calibration_seed=expected_calibration_seed,
            )
        except KeyError:
            if self.diverse.smoke_allow_missing_radius_cap:
                return None, None
            raise
        return entry.cap_for_quantile(self.diverse.radius_quantile), entry

    def _build_shared_pool(
        self,
        clean_raw: torch.Tensor,
        targets: torch.Tensor,
        classifier: nn.Module,
        *,
        endpoint_seed: int,
        center: str | None,
    ) -> SharedEndpointPool:
        assert self.encoder is not None and self.decoder is not None
        assert self.augmix_config is not None
        clean = torch.nan_to_num(
            clean_raw.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        ).contiguous()
        if targets.shape != (int(clean.shape[0]), 5) or not bool(
            torch.isfinite(targets).all()
        ):
            raise ValueError("targets must be finite (B,5) for endpoint selection")
        generator = _generator(clean.device, endpoint_seed)
        candidate_raw, depths, operator_mask, chain_nonfinite = (
            _generate_candidate_waveforms(
                clean,
                candidate_count=self.diverse.candidate_endpoints,
                allowed_depths=self.diverse.allowed_depths,
                depth_sampling=self.diverse.depth_sampling,
                augmix_config=self.augmix_config,
                generator=generator,
            )
        )
        batch_size, candidate_count = candidate_raw.shape[:2]
        clean_latent, clean_decoded, clean_finite = _encode_decode_waveforms(
            clean, encoder=self.encoder, decoder=self.decoder
        )
        flat_candidates = candidate_raw.reshape(
            batch_size * candidate_count, OUTPUT_POINTS, 12
        )
        candidate_latent_flat, candidate_decoded_flat, candidate_finite_flat = (
            _encode_decode_waveforms(
                flat_candidates, encoder=self.encoder, decoder=self.decoder
            )
        )
        candidate_latent = candidate_latent_flat.reshape(
            batch_size, candidate_count, 4, 128
        )
        candidate_decoded = candidate_decoded_flat.reshape(
            batch_size, candidate_count, OUTPUT_POINTS, 12
        )
        candidate_finite = candidate_finite_flat.reshape(batch_size, candidate_count)
        geometry = build_standardized_endpoint_geometry(
            clean_latent, candidate_latent, self.latent_pool.standardizer
        )
        radius_cap, radius_entry = self._resolve_radius_cap(center)

        model_spec = get_model_spec(self.model_name)
        parameters = tuple(classifier.parameters())
        if any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError(
                "classifier gradients must be None before endpoint selection"
            )
        with _frozen_classifier_for_attack(classifier):
            (
                clean_probabilities,
                _,
                clean_prediction_finite,
            ) = _classifier_probabilities(clean, classifier, model_spec)
            (
                candidate_probabilities_flat,
                candidate_logits_flat,
                candidate_prediction_finite_flat,
            ) = (
                _classifier_probabilities(flat_candidates, classifier, model_spec)
            )
        if any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError("endpoint selector leaked classifier gradients")
        candidate_probabilities = candidate_probabilities_flat.reshape(
            batch_size, candidate_count, 5
        )
        candidate_bce = F.binary_cross_entropy_with_logits(
            candidate_logits_flat.reshape(batch_size, candidate_count, 5),
            targets.float().unsqueeze(1).expand(-1, candidate_count, -1),
            reduction="none",
        ).mean(dim=2)
        candidate_prediction_finite = candidate_prediction_finite_flat.reshape(
            batch_size, candidate_count
        )
        valid_candidates = (
            geometry.finite_mask
            & candidate_finite
            & candidate_prediction_finite
            & (chain_nonfinite == 0)
            & clean_finite.unsqueeze(1)
            & clean_prediction_finite.unsqueeze(1)
        )
        # D3 exposes a selected raw endpoint directly.  Keep that endpoint
        # inside the same train-only P95 geometry contract used by latent
        # projections so the raw waveform and its one-hot/m=1 diagnostic
        # projection describe the same point.  Earlier D0-D2 arms retain
        # their already-run selection semantics.
        if self.method.profile_name in {
            D3_METHOD_ID,
            D3_DEPTH23_METHOD_ID,
            D6_METHOD_ID,
        }:
            if radius_cap is None:
                raise RuntimeError("raw-endpoint exposure requires a radius cap")
            valid_candidates = valid_candidates & (
                geometry.radii_rms <= float(radius_cap)
            )
        geometry = replace(geometry, finite_mask=valid_candidates)
        eligible = valid_candidates.sum(dim=1) >= self.diverse.selected_directions

        placeholder = torch.arange(
            self.diverse.selected_directions,
            device=clean.device,
            dtype=torch.int64,
        ).unsqueeze(0).expand(batch_size, -1).clone()
        indices = placeholder
        if bool(eligible.any()):
            subset_selection = greedy_select_diverse_directions(
                _subset_geometry(geometry, eligible),
                clean_probabilities[eligible],
                candidate_probabilities[eligible],
                first_scores=(
                    candidate_bce[eligible]
                    if self.method.profile_name == D6_METHOD_ID
                    else None
                ),
                selected_count=self.diverse.selected_directions,
                latent_cosine_weight=self.diverse.latent_cosine_weight,
                prediction_rms_weight=self.diverse.prediction_rms_weight,
            )
            indices[eligible] = subset_selection.indices
        # Recompute a full-batch selection record from the chosen indices.  Rows
        # marked ineligible remain placeholders and are always clean-fallback.
        selected_endpoints = _gather_candidate(geometry.endpoints, indices)
        selected_raw = _gather_candidate(candidate_raw, indices)
        selected_deltas = _gather_candidate(geometry.deltas, indices)
        selected_unit = _gather_candidate(geometry.unit_directions, indices)
        selected_radii = torch.gather(geometry.radii_rms, 1, indices)
        selected_predictions = _gather_candidate(candidate_probabilities, indices)
        pairwise_cosine = torch.einsum(
            "bkd,bld->bkl", selected_unit, selected_unit
        ).clamp(-1.0, 1.0)
        pairwise_prediction = (
            selected_predictions.unsqueeze(2) - selected_predictions.unsqueeze(1)
        ).square().mean(dim=3).sqrt()
        pairwise_composite = (
            self.diverse.latent_cosine_weight * (1.0 - pairwise_cosine) * 0.5
            + self.diverse.prediction_rms_weight * pairwise_prediction
        )
        clean_shift = (
            selected_predictions - clean_probabilities.unsqueeze(1)
        ).square().mean(dim=2).sqrt()
        selection = DiversitySelection(
            indices=indices.contiguous(),
            endpoints=selected_endpoints,
            deltas=selected_deltas,
            unit_directions=selected_unit,
            radii_rms=selected_radii.contiguous(),
            prediction_signatures=selected_predictions.contiguous(),
            clean_prediction_shift_rms=clean_shift.contiguous(),
            selected_pairwise_cosine=pairwise_cosine.contiguous(),
            selected_pairwise_prediction_rms=pairwise_prediction.contiguous(),
            selected_pairwise_composite_distance=pairwise_composite.contiguous(),
            candidate_prediction_variance=candidate_probabilities.var(
                dim=1, correction=0
            ).mean(dim=1),
        )

        clean_residual = clean - clean_decoded
        candidate_residual = candidate_raw - candidate_decoded
        selected_residual = _gather_candidate(candidate_residual, indices)
        endpoint_residuals = torch.cat(
            (clean_residual.unsqueeze(1), selected_residual), dim=1
        ).contiguous()
        selected_finite = torch.gather(valid_candidates, 1, indices).all(dim=1)
        return SharedEndpointPool(
            clean_standardized=geometry.clean,
            selected_standardized=selected_endpoints,
            selected_raw=selected_raw,
            endpoint_residuals=endpoint_residuals,
            selection=selection,
            eligible_mask=eligible.contiguous(),
            selected_finite=selected_finite.contiguous(),
            candidate_depths=depths,
            candidate_operator_mask=operator_mask,
            candidate_nonfinite_count=chain_nonfinite,
            candidate_radii_rms=geometry.radii_rms,
            clean_probabilities=clean_probabilities,
            candidate_probabilities=candidate_probabilities,
            candidate_bce=candidate_bce.contiguous(),
            radius_cap_rms=radius_cap,
            radius_entry=radius_entry,
            center=center,
            endpoint_seed=endpoint_seed,
        )

    def _decode_projection(
        self,
        projection: LatentProjection,
        endpoint_residuals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.decoder is not None and self._shared_cache is not None
        raw_latent = self.latent_pool.standardizer.inverse_transform(
            projection.standardized_latent
        )
        decoded = decode_to_ptbxl_waveform(
            self.decoder, raw_latent, target_points=OUTPUT_POINTS
        )
        residual = mix_endpoint_residuals(
            (
                self._shared_cache.endpoint_residuals
                if endpoint_residuals is None
                else endpoint_residuals
            ),
            projection.effective_endpoint_weights,
        )
        return (decoded.to(dtype=torch.float32) + residual).contiguous()

    def _projection(
        self,
        weights: torch.Tensor,
        m: torch.Tensor,
        *,
        clean_standardized: torch.Tensor | None = None,
        selected_standardized: torch.Tensor | None = None,
    ) -> LatentProjection:
        if self._shared_cache is None:
            raise RuntimeError("shared endpoint cache is inactive")
        return project_clean_anchored_simplex(
            (
                self._shared_cache.clean_standardized
                if clean_standardized is None
                else clean_standardized
            ),
            (
                self._shared_cache.selected_standardized
                if selected_standardized is None
                else selected_standardized
            ),
            weights,
            m,
            radius_cap_rms=self._shared_cache.radius_cap_rms,
            smoke_allow_missing_radius_cap=self.diverse.smoke_allow_missing_radius_cap,
        )

    def _random_view(self, generator: torch.Generator) -> ViewCandidate:
        assert self._shared_cache is not None
        batch_size = int(self._shared_cache.clean_standardized.shape[0])
        weights, m = sample_simplex_state(
            batch_size,
            self.diverse.selected_directions,
            device=self._shared_cache.clean_standardized.device,
            generator=generator,
            dtype=self._shared_cache.clean_standardized.dtype,
        )
        projection = self._projection(weights, m)
        with torch.no_grad():
            waveform = self._decode_projection(projection)
        zeros = torch.zeros(batch_size, device=waveform.device, dtype=torch.float32)
        return ViewCandidate(
            waveform=waveform.detach(),
            projection=projection,
            initial_bce=zeros,
            final_bce=zeros,
            search_finite=torch.ones_like(zeros, dtype=torch.bool),
            final_probabilities=None,
        )

    def _raw_endpoint_view(
        self,
        generator: torch.Generator,
        *,
        selected_slot: int | None = None,
    ) -> ViewCandidate:
        """Return one exact selected raw endpoint per record.

        The selected directions are already diversity-filtered and, for D3,
        train-P95 bounded.  A one-hot simplex state with ``m=1`` records the
        endpoint geometry, while the waveform bypasses another VAE decode so
        the classifier sees the exact online corruption endpoint.  D3 samples
        a slot; D6 supplies slots 0, 1 and 2 exactly once each.
        """

        assert self._shared_cache is not None
        cache = self._shared_cache
        batch_size = int(cache.clean_standardized.shape[0])
        selected_count = self.diverse.selected_directions
        if selected_slot is None:
            slot = torch.randint(
                0,
                selected_count,
                (batch_size,),
                device=cache.clean_standardized.device,
                dtype=torch.int64,
                generator=generator,
            )
        else:
            if not 0 <= int(selected_slot) < selected_count:
                raise ValueError("selected_slot is outside the selected endpoint set")
            slot = torch.full(
                (batch_size,),
                int(selected_slot),
                device=cache.clean_standardized.device,
                dtype=torch.int64,
            )
        weights = torch.zeros(
            batch_size,
            selected_count,
            device=cache.clean_standardized.device,
            dtype=cache.clean_standardized.dtype,
        )
        weights.scatter_(1, slot.unsqueeze(1), 1.0)
        m = torch.ones(
            batch_size,
            device=cache.clean_standardized.device,
            dtype=cache.clean_standardized.dtype,
        )
        projection = self._projection(weights, m)
        valid = cache.eligible_mask & cache.selected_finite
        if bool(valid.any() and (projection.projection_scale[valid] != 1.0).any()):
            raise RuntimeError(
                "a selected raw endpoint exceeded the train-only radius cap"
            )
        waveform = _gather_candidate(
            cache.selected_raw, slot.unsqueeze(1)
        ).squeeze(1)
        zeros = torch.zeros(batch_size, device=waveform.device, dtype=torch.float32)
        return ViewCandidate(
            waveform=waveform.detach().contiguous(),
            projection=projection,
            initial_bce=zeros,
            final_bce=zeros,
            search_finite=torch.ones_like(zeros, dtype=torch.bool),
            final_probabilities=None,
        )

    def _hard_view(
        self,
        source: WaveformView,
        classifier: nn.Module,
        generator: torch.Generator,
        *,
        boundary_only: bool = False,
    ) -> ViewCandidate:
        assert self._shared_cache is not None
        batch_size = source.batch_size
        if boundary_only:
            initial_weights = torch.zeros(
                batch_size,
                self.diverse.selected_directions,
                device=source.waveform.device,
                dtype=self._shared_cache.clean_standardized.dtype,
            )
            # Selection slot zero is the supervised-BCE-hardest endpoint in D6.
            initial_weights[:, 0] = 1.0
            initial_m = torch.ones(
                batch_size,
                device=source.waveform.device,
                dtype=self._shared_cache.clean_standardized.dtype,
            )
        else:
            initial_weights, initial_m = sample_simplex_state(
                batch_size,
                self.diverse.selected_directions,
                device=source.waveform.device,
                generator=generator,
                dtype=self._shared_cache.clean_standardized.dtype,
            )
        parameters = tuple(classifier.parameters())
        if any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError("classifier gradients must be None before hard search")
        model_spec = get_model_spec(self.model_name)
        projection_chunks: list[LatentProjection] = []
        waveform_chunks: list[torch.Tensor] = []
        initial_loss_chunks: list[torch.Tensor] = []
        final_loss_chunks: list[torch.Tensor] = []
        finite_chunks: list[torch.Tensor] = []
        final_probability_chunks: list[torch.Tensor] = []
        with _frozen_classifier_for_attack(classifier), torch.enable_grad():
            for start in range(0, batch_size, self._search_chunk_size):
                stop = min(batch_size, start + self._search_chunk_size)
                clean_chunk = self._shared_cache.clean_standardized[start:stop]
                selected_chunk = self._shared_cache.selected_standardized[start:stop]
                residual_chunk = self._shared_cache.endpoint_residuals[start:stop]
                target_chunk = source.labels[start:stop]
                weight_logits = (
                    initial_weights[start:stop]
                    .float()
                    .clamp_min(1.0e-8)
                    .log()
                    .detach()
                    .requires_grad_(True)
                )
                m_logits = (
                    None
                    if boundary_only
                    else torch.logit(
                        initial_m[start:stop]
                        .float()
                        .clamp(1.0e-4, 1.0 - 1.0e-4)
                    )
                    .detach()
                    .requires_grad_(True)
                )
                best_loss = torch.full(
                    (stop - start,),
                    -torch.inf,
                    device=source.waveform.device,
                    dtype=torch.float32,
                )
                best_weights = initial_weights[start:stop].detach().clone()
                best_m = initial_m[start:stop].detach().clone()
                initial_loss: torch.Tensor | None = None
                all_finite = torch.ones_like(best_loss, dtype=torch.bool)

                for _ in range(self.diverse.search_steps):
                    weights = torch.softmax(weight_logits, dim=1).to(
                        dtype=clean_chunk.dtype
                    )
                    m = (
                        torch.ones(
                            stop - start,
                            device=weights.device,
                            dtype=weights.dtype,
                        )
                        if m_logits is None
                        else torch.sigmoid(m_logits).to(dtype=weights.dtype)
                    )
                    projection = self._projection(
                        weights,
                        m,
                        clean_standardized=clean_chunk,
                        selected_standardized=selected_chunk,
                    )
                    waveform = self._decode_projection(projection, residual_chunk)
                    _, loss, finite = _classifier_logits_and_loss(
                        waveform, target_chunk, classifier, model_spec
                    )
                    if initial_loss is None:
                        initial_loss = loss.detach()
                    improved = finite & (loss.detach() > best_loss)
                    best_loss = torch.where(improved, loss.detach(), best_loss)
                    best_weights = torch.where(
                        improved.unsqueeze(1), weights.detach(), best_weights
                    )
                    best_m = torch.where(improved, m.detach(), best_m)
                    if m_logits is None:
                        (gradient_weights,) = torch.autograd.grad(
                            loss.sum(), (weight_logits,), create_graph=False
                        )
                        gradient_m = None
                        gradient_finite = torch.isfinite(gradient_weights).all(dim=1)
                    else:
                        gradient_weights, gradient_m = torch.autograd.grad(
                            loss.sum(), (weight_logits, m_logits), create_graph=False
                        )
                        gradient_finite = torch.isfinite(gradient_weights).all(
                            dim=1
                        ) & torch.isfinite(gradient_m)
                    all_finite &= finite.detach() & gradient_finite.detach()
                    learning_rate = self.diverse.coefficient_learning_rate
                    weight_logits = (
                        weight_logits
                        + learning_rate * torch.nan_to_num(gradient_weights).sign()
                    ).detach().requires_grad_(True)
                    if m_logits is not None:
                        assert gradient_m is not None
                        m_logits = (
                            m_logits
                            + learning_rate * torch.nan_to_num(gradient_m).sign()
                        ).detach().requires_grad_(True)

                final_weights = torch.softmax(weight_logits, dim=1).to(
                    dtype=clean_chunk.dtype
                )
                final_m = (
                    torch.ones(
                        stop - start,
                        device=final_weights.device,
                        dtype=final_weights.dtype,
                    )
                    if m_logits is None
                    else torch.sigmoid(m_logits).to(dtype=final_weights.dtype)
                )
                final_projection = self._projection(
                    final_weights,
                    final_m,
                    clean_standardized=clean_chunk,
                    selected_standardized=selected_chunk,
                )
                final_waveform = self._decode_projection(
                    final_projection, residual_chunk
                )
                _, trial_loss, finite = _classifier_logits_and_loss(
                    final_waveform, target_chunk, classifier, model_spec
                )
                improved = finite & (trial_loss.detach() > best_loss)
                best_loss = torch.where(improved, trial_loss.detach(), best_loss)
                best_weights = torch.where(
                    improved.unsqueeze(1), final_weights.detach(), best_weights
                )
                best_m = torch.where(improved, final_m.detach(), best_m)
                all_finite &= finite.detach() & torch.isfinite(best_loss)
                if initial_loss is None:
                    raise RuntimeError("hard search did not evaluate its initial state")

                selected_projection = self._projection(
                    best_weights.detach(),
                    best_m.detach(),
                    clean_standardized=clean_chunk,
                    selected_standardized=selected_chunk,
                )
                with torch.no_grad():
                    selected_waveform = self._decode_projection(
                        selected_projection, residual_chunk
                    )
                    (
                        selected_logits,
                        selected_loss,
                        selected_finite,
                    ) = _classifier_logits_and_loss(
                        selected_waveform, target_chunk, classifier, model_spec
                    )
                    selected_probabilities = torch.sigmoid(
                        torch.nan_to_num(
                            selected_logits.float(),
                            nan=0.0,
                            posinf=30.0,
                            neginf=-30.0,
                        )
                    )
                all_finite &= selected_finite.detach()
                projection_chunks.append(selected_projection)
                waveform_chunks.append(selected_waveform.detach())
                initial_loss_chunks.append(initial_loss.detach())
                final_loss_chunks.append(selected_loss.detach())
                finite_chunks.append(all_finite.detach())
                final_probability_chunks.append(selected_probabilities.detach())

        if any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError("hard search leaked gradients into classifier parameters")
        projection = LatentProjection(
            standardized_latent=torch.cat(
                [value.standardized_latent for value in projection_chunks], dim=0
            ),
            raw_simplex_weights=torch.cat(
                [value.raw_simplex_weights for value in projection_chunks], dim=0
            ),
            raw_m=torch.cat([value.raw_m for value in projection_chunks], dim=0),
            effective_endpoint_weights=torch.cat(
                [value.effective_endpoint_weights for value in projection_chunks], dim=0
            ),
            effective_m=torch.cat(
                [value.effective_m for value in projection_chunks], dim=0
            ),
            pre_projection_radius_rms=torch.cat(
                [value.pre_projection_radius_rms for value in projection_chunks], dim=0
            ),
            post_projection_radius_rms=torch.cat(
                [value.post_projection_radius_rms for value in projection_chunks], dim=0
            ),
            projection_scale=torch.cat(
                [value.projection_scale for value in projection_chunks], dim=0
            ),
            radius_cap_rms=self._shared_cache.radius_cap_rms,
        )
        return ViewCandidate(
            waveform=torch.cat(waveform_chunks, dim=0).contiguous(),
            projection=projection,
            initial_bce=torch.cat(initial_loss_chunks, dim=0).contiguous(),
            final_bce=torch.cat(final_loss_chunks, dim=0).contiguous(),
            search_finite=torch.cat(finite_chunks, dim=0).contiguous(),
            final_probabilities=torch.cat(
                final_probability_chunks, dim=0
            ).contiguous(),
        )

    def _diagnostics(
        self,
        candidate: ViewCandidate,
        *,
        role: str,
        accepted: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[dict[str, float], dict[str, int]]:
        assert self._shared_cache is not None
        cache = self._shared_cache
        projection = candidate.projection
        selection = cache.selection
        selected_count = self.diverse.selected_directions
        eye = torch.eye(selected_count, device=accepted.device, dtype=torch.bool).unsqueeze(0)
        off_diagonal = ~eye.expand(accepted.shape[0], -1, -1)
        pairwise_cosine = selection.selected_pairwise_cosine[off_diagonal]
        pairwise_prediction = selection.selected_pairwise_prediction_rms[off_diagonal]
        pairwise_composite = selection.selected_pairwise_composite_distance[off_diagonal]
        simplex = projection.raw_simplex_weights.clamp_min(1.0e-8)
        entropy = -(simplex * simplex.log()).sum(dim=1)
        residual = mix_endpoint_residuals(
            cache.endpoint_residuals, projection.effective_endpoint_weights
        )
        residual_rms = residual.square().mean(dim=(1, 2)).sqrt()
        waveform_rms = torch.nan_to_num(candidate.waveform).square().mean(
            dim=(1, 2)
        ).sqrt()
        values = {
            "role_random": 1.0 if role == "random" else 0.0,
            "role_hard": 1.0 if role == "hard" else 0.0,
            "role_boundary_hard": 1.0 if role == "boundary_hard" else 0.0,
            "role_raw_endpoint": 1.0 if role == "raw_endpoint" else 0.0,
            "endpoint_pool_shared": 1.0,
            "candidate_endpoint_count": float(self.diverse.candidate_endpoints),
            "selected_direction_count": float(selected_count),
            "eligible_fraction": float(cache.eligible_mask.float().mean().detach().cpu()),
            "candidate_radius_mean": float(cache.candidate_radii_rms.mean().detach().cpu()),
            "candidate_radius_p95_batch_diagnostic_only": float(
                torch.quantile(cache.candidate_radii_rms.flatten(), 0.95).detach().cpu()
            ),
            "selected_radius_mean": float(selection.radii_rms.mean().detach().cpu()),
            "selected_clean_prediction_shift_rms": float(
                selection.clean_prediction_shift_rms.mean().detach().cpu()
            ),
            "candidate_prediction_variance": float(
                selection.candidate_prediction_variance.mean().detach().cpu()
            ),
            "selected_pairwise_cosine_mean": float(pairwise_cosine.mean().detach().cpu()),
            "selected_pairwise_prediction_rms_mean": float(
                pairwise_prediction.mean().detach().cpu()
            ),
            "selected_pairwise_composite_min": float(
                pairwise_composite.min().detach().cpu()
            ),
            "radius_cap_present": 1.0 if cache.radius_cap_rms is not None else 0.0,
            "radius_cap_standardized_rms": float(cache.radius_cap_rms or 0.0),
            "pre_projection_radius_rms": float(
                projection.pre_projection_radius_rms.mean().detach().cpu()
            ),
            "post_projection_radius_rms": float(
                projection.post_projection_radius_rms.mean().detach().cpu()
            ),
            "projection_scale": float(projection.projection_scale.mean().detach().cpu()),
            "projection_active_fraction": float(
                (projection.projection_scale < 1.0 - 1.0e-7)
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "effective_clean_weight": float(
                projection.effective_endpoint_weights[:, 0].mean().detach().cpu()
            ),
            "simplex_entropy": float(entropy.mean().detach().cpu()),
            "simplex_top1": float(simplex.max(dim=1).values.mean().detach().cpu()),
            "endpoint_residual_rms_ratio": float(
                (residual_rms / waveform_rms.clamp_min(1.0e-8)).mean().detach().cpu()
            ),
            "quality_accepted_count": float(accepted.sum().detach().cpu()),
            "quality_rejected_count": float(
                accepted.numel() - accepted.sum().detach().cpu()
            ),
            "mean_candidate_depth": float(cache.candidate_depths.float().mean().detach().cpu()),
        }
        weights: dict[str, int] = {}
        mixing_values, mixing_weights = _mixing_radius_diagnostics(
            projection, accepted
        )
        values.update(mixing_values)
        weights.update(mixing_weights)
        # Keep the two original mean aliases while making their accepted-record
        # denominator explicit through the new contract and weights.
        if "raw_m_mean" in values:
            values["raw_m"] = values["raw_m_mean"]
            weights["raw_m"] = mixing_weights["raw_m_mean"]
        if "effective_m_mean" in values:
            values["effective_m"] = values["effective_m_mean"]
            weights["effective_m"] = mixing_weights["effective_m_mean"]

        if role in {"hard", "boundary_hard"}:
            if candidate.final_probabilities is None:
                raise RuntimeError("hard-view final probabilities are unavailable")
            hard_values, hard_weights = _hard_vs_random_diagnostics(
                candidate.initial_bce,
                candidate.final_bce,
                accepted & candidate.search_finite,
            )
            attack_values, attack_weights = (
                _clean_correct_multilabel_attack_diagnostics(
                    cache.clean_probabilities,
                    candidate.final_probabilities,
                    targets,
                    accepted & candidate.search_finite,
                )
            )
            values.update(hard_values)
            values.update(attack_values)
            weights.update(hard_weights)
            weights.update(attack_weights)
            if role == "boundary_hard":
                for source_name, target_name in (
                    (
                        "random_initial_bce_mean",
                        "seeded_hardest_endpoint_bce_mean",
                    ),
                    (
                        "hard_minus_random_bce_gain_mean",
                        "hard_minus_seeded_endpoint_bce_gain_mean",
                    ),
                    (
                        "hard_strictly_greater_than_random_fraction",
                        "hard_strictly_greater_than_seeded_endpoint_fraction",
                    ),
                ):
                    if source_name in values:
                        values[target_name] = values[source_name]
                        weights[target_name] = hard_weights[source_name]
            values["search_finite_fraction"] = float(
                candidate.search_finite.float().mean().detach().cpu()
            )
            # Backward-readable aliases use the exact accepted comparison set.
            if "random_initial_bce_mean" in values:
                values["search_initial_bce"] = values["random_initial_bce_mean"]
                weights["search_initial_bce"] = hard_weights[
                    "random_initial_bce_mean"
                ]
            if "hard_final_bce_mean" in values:
                values["search_final_bce"] = values["hard_final_bce_mean"]
                weights["search_final_bce"] = hard_weights["hard_final_bce_mean"]
            if "hard_minus_random_bce_gain_mean" in values:
                values["search_loss_gain"] = values[
                    "hard_minus_random_bce_gain_mean"
                ]
                weights["search_loss_gain"] = hard_weights[
                    "hard_minus_random_bce_gain_mean"
                ]
        if role == "raw_endpoint":
            slot = projection.raw_simplex_weights.argmax(dim=1)
            source_index = torch.gather(
                selection.indices, 1, slot.unsqueeze(1)
            ).squeeze(1)
            chosen_depth = torch.gather(
                cache.candidate_depths, 1, source_index.unsqueeze(1)
            ).squeeze(1)
            chosen_operator_mask = torch.gather(
                cache.candidate_operator_mask,
                1,
                source_index.view(-1, 1, 1).expand(
                    -1, 1, cache.candidate_operator_mask.shape[2]
                ),
            ).squeeze(1)
            values["raw_endpoint_attempted_mean_depth"] = float(
                chosen_depth.float().mean().detach().cpu()
            )
            for index in range(selected_count):
                values[
                    f"raw_endpoint_attempted_selected_slot_fraction/{index}"
                ] = float(
                    (slot == index).float().mean().detach().cpu()
                )
            for depth in self.diverse.allowed_depths:
                values[f"raw_endpoint_attempted_depth{depth}_fraction"] = float(
                    (chosen_depth == depth).float().mean().detach().cpu()
                )
            for index, operator in enumerate(CANONICAL_OPERATORS):
                values[
                    f"raw_endpoint_attempted_operator_fraction/{operator}"
                ] = float(chosen_operator_mask[:, index].float().mean().detach().cpu())
            accepted_count = int(accepted.sum().detach().cpu())
            values["raw_endpoint_accepted_record_count"] = float(accepted_count)
            if accepted_count:
                accepted_depth = chosen_depth[accepted]
                accepted_slot = slot[accepted]
                accepted_operator_mask = chosen_operator_mask[accepted]
                values["raw_endpoint_accepted_mean_depth"] = float(
                    accepted_depth.float().mean().detach().cpu()
                )
                for index in range(selected_count):
                    values[
                        f"raw_endpoint_accepted_selected_slot_fraction/{index}"
                    ] = float(
                        (accepted_slot == index).float().mean().detach().cpu()
                    )
                for depth in self.diverse.allowed_depths:
                    values[f"raw_endpoint_accepted_depth{depth}_fraction"] = float(
                        (accepted_depth == depth).float().mean().detach().cpu()
                    )
                for index, operator in enumerate(CANONICAL_OPERATORS):
                    values[
                        f"raw_endpoint_accepted_operator_fraction/{operator}"
                    ] = float(
                        accepted_operator_mask[:, index]
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    )
        for depth in self.diverse.allowed_depths:
            values[f"depth{depth}_fraction"] = float(
                (cache.candidate_depths == depth).float().mean().detach().cpu()
            )
        for index, operator in enumerate(CANONICAL_OPERATORS):
            values[f"operator_fraction/{operator}"] = float(
                cache.candidate_operator_mask[:, :, index]
                .float()
                .mean()
                .detach()
                .cpu()
            )
        for index in range(self.diverse.candidate_endpoints):
            values[f"selected_index_fraction/{index}"] = float(
                (selection.indices == index).float().mean().detach().cpu()
            )
        return values, weights

    def _diverse_view(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("Diverse AugMax view requires one clean WaveformView")
        if context.node_id not in self.diverse.view_modes:
            raise ValueError(
                "Diverse AugMax node id is outside the method view contract"
            )
        if self._shared_cache is None or self._generation_state is None:
            raise RuntimeError("Diverse AugMax shared cache is inactive")
        source = inputs[0]
        role = self.diverse.view_modes[context.node_id]
        ordinal = self._role_ordinals.get(role, 0)
        self._role_ordinals[role] = ordinal + 1
        base_seed, rng_identity = self._generation_state
        view_seed = _derive_seed(
            base_seed,
            rng_identity,
            f"{role}_view_ordinal={ordinal}",
        )
        generator = _generator(source.waveform.device, view_seed)
        if role == "random":
            candidate = self._random_view(generator)
        elif role == "raw_endpoint":
            candidate = (
                self._raw_endpoint_view(generator, selected_slot=ordinal)
                if self.method.profile_name == D6_METHOD_ID
                else self._raw_endpoint_view(generator)
            )
        elif role == "hard":
            candidate = self._hard_view(
                source, context.resource("classifier"), generator
            )
        elif role == "boundary_hard":
            candidate = self._hard_view(
                source,
                context.resource("classifier"),
                generator,
                boundary_only=True,
            )
        else:
            raise RuntimeError(f"unsupported Diverse AugMax view role {role!r}")
        quality_accepted, quality_reasons = _quality_mask(
            candidate.waveform,
            minimum_std_mV=self.minimum_std_mV,
            maximum_abs_mV=self.maximum_abs_mV,
        )
        source_valid = source.valid_mask
        accepted = (
            source_valid
            & self._shared_cache.eligible_mask
            & self._shared_cache.selected_finite
            & candidate.search_finite
            & quality_accepted
        )
        accepted_positions = _positions(accepted)
        safe_clean = torch.nan_to_num(
            source.waveform.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        full_waveform = safe_clean.clone()
        if accepted_positions:
            positions = torch.as_tensor(
                accepted_positions, device=source.waveform.device, dtype=torch.long
            )
            full_waveform.index_copy_(
                0,
                positions,
                candidate.waveform.index_select(0, positions),
            )

        eligible_cpu = self._shared_cache.eligible_mask.detach().cpu().tolist()
        selected_finite_cpu = self._shared_cache.selected_finite.detach().cpu().tolist()
        search_finite_cpu = candidate.search_finite.detach().cpu().tolist()
        source_valid_cpu = source_valid.detach().cpu().tolist()
        rejected: list[dict[str, str]] = []
        ineligible: list[str] = []
        for index, reason in enumerate(quality_reasons):
            failures: list[str] = []
            if not bool(source_valid_cpu[index]):
                failures.append("source_invalid")
            if not bool(eligible_cpu[index]):
                failures.append("fewer_than_k_valid_directions")
                ineligible.append(source.sample_ids[index])
            if not bool(selected_finite_cpu[index]):
                failures.append("selected_endpoint_nonfinite")
            if not bool(search_finite_cpu[index]):
                failures.append("search_nonfinite")
            if reason != "accepted":
                failures.append(reason)
            if failures:
                rejected.append(
                    {
                        "node_id": context.node_id,
                        "hash_id": source.sample_ids[index],
                        "reason": "+".join(failures),
                    }
                )
        diagnostics, diagnostic_weights = self._diagnostics(
            candidate,
            role=role,
            accepted=accepted,
            targets=source.labels,
        )
        for name, value in diagnostics.items():
            context.record_diagnostic(name, value)

        selection = self._shared_cache.selection
        selected_indices_cpu = tuple(
            tuple(int(item) for item in row)
            for row in selection.indices.detach().cpu().tolist()
        )
        selected_signatures_cpu = tuple(
            tuple(tuple(float(item) for item in signature) for signature in row)
            for row in selection.prediction_signatures.detach().cpu().tolist()
        )
        raw_endpoint_slots: tuple[int, ...] = ()
        raw_endpoint_source_indices: tuple[int, ...] = ()
        raw_endpoint_slot_sha256: str | None = None
        raw_endpoint_source_index_sha256: str | None = None
        if role == "raw_endpoint":
            slot_tensor = candidate.projection.raw_simplex_weights.argmax(dim=1)
            source_index_tensor = torch.gather(
                selection.indices, 1, slot_tensor.unsqueeze(1)
            ).squeeze(1)
            raw_endpoint_slots = tuple(
                int(value) for value in slot_tensor.detach().cpu().tolist()
            )
            raw_endpoint_source_indices = tuple(
                int(value) for value in source_index_tensor.detach().cpu().tolist()
            )
            raw_endpoint_slot_sha256 = _hash_tensor(slot_tensor)
            raw_endpoint_source_index_sha256 = _hash_tensor(source_index_tensor)
        radius_entry = self._shared_cache.radius_entry
        return WaveformView(
            name=context.node_id,
            waveform=full_waveform.contiguous(),
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=accepted,
            provenance=Provenance(
                node_id=context.node_id,
                operation="sandbox_diverse_latent_augmax_view",
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={
                    "sandbox_adapter": "diverse_latent_augmax_v1",
                    "method_id": self.method.profile_name,
                    "view_role": role,
                    "view_seed": view_seed,
                    "shared_endpoint_seed": self._shared_cache.endpoint_seed,
                    "candidate_endpoints": self.diverse.candidate_endpoints,
                    "selected_directions": self.diverse.selected_directions,
                    "allowed_depths": list(self.diverse.allowed_depths),
                    "selector": self.diverse.describe()["selector"],
                    "standardizer_sha256": self.latent_pool.standardizer.identity_sha256,
                    "radius_cap_standardized_rms": self._shared_cache.radius_cap_rms,
                    "radius_registry_sha256": (
                        None
                        if self._radius_registry is None
                        else self._radius_registry.sha256
                    ),
                    "radius_entry_id": None if radius_entry is None else radius_entry.entry_id,
                    "radius_dynamic_batch_quantile_used": False,
                    "smoke_missing_radius_cap": radius_entry is None,
                    "hard_search_steps": (
                        self.diverse.search_steps
                        if role in {"hard", "boundary_hard"}
                        else 0
                    ),
                    "hard_search_chunk_size": (
                        self._search_chunk_size
                        if role in {"hard", "boundary_hard"}
                        else 0
                    ),
                    "coefficient_learning_rate": (
                        self.diverse.coefficient_learning_rate
                        if role in {"hard", "boundary_hard"}
                        else 0.0
                    ),
                    "hard_optimized_variables": (
                        (
                            ["mix_logits"]
                            if role == "boundary_hard"
                            else ["mix_logits", "m_logit"]
                        )
                        if role in {"hard", "boundary_hard"}
                        else []
                    ),
                    "raw_endpoint_direct_waveform_bypass": role == "raw_endpoint",
                    "waveform_construction": (
                        "direct_selected_raw_endpoint"
                        if role == "raw_endpoint"
                        else "vae_decode_plus_endpoint_residual"
                    ),
                    "reconstruction_residual_bypass": role != "raw_endpoint",
                    "raw_endpoint_slot_sha256": raw_endpoint_slot_sha256,
                    "raw_endpoint_source_index_sha256": (
                        raw_endpoint_source_index_sha256
                    ),
                    "quality_failure_policy": "clean_fallback_and_invalid_view_mask",
                    "objective_global_scale": OBJECTIVE_GLOBAL_SCALE,
                    "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
                    "multilabel_decision_threshold": (
                        MULTILABEL_DECISION_THRESHOLD
                    ),
                    "m_low_boundary_inclusive": M_LOW_BOUNDARY_INCLUSIVE,
                    "m_high_boundary_inclusive": M_HIGH_BOUNDARY_INCLUSIVE,
                },
            ),
            metadata={
                "candidate_eligible_positions": _positions(
                    self._shared_cache.eligible_mask
                ),
                "accepted_positions": accepted_positions,
                "ineligible_hash_ids": tuple(dict.fromkeys(ineligible)),
                "quality_rejected": tuple(rejected),
                "selected_endpoint_indices": selected_indices_cpu,
                "selected_prediction_signatures": selected_signatures_cpu,
                "raw_endpoint_selected_slots": raw_endpoint_slots,
                "raw_endpoint_source_candidate_indices": (
                    raw_endpoint_source_indices
                ),
                "selected_indices_sha256": _hash_tensor(selection.indices),
                "selected_signatures_sha256": _hash_tensor(
                    selection.prediction_signatures
                ),
                "shared_pool_identity": {
                    "endpoint_seed": self._shared_cache.endpoint_seed,
                    "selected_indices_sha256": _hash_tensor(selection.indices),
                    "standardizer_sha256": self.latent_pool.standardizer.identity_sha256,
                },
                "diagnostic_means": MappingProxyType(dict(diagnostics)),
                "diagnostic_weights": MappingProxyType(
                    dict(diagnostic_weights)
                ),
            },
        )

    def _dispatch_augmix(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        if context.node_id == "resource_bridge":
            if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
                raise TypeError("resource_bridge requires one clean WaveformView")
            source = inputs[0]
            context.record_diagnostic("identity_only", 1.0)
            return WaveformView(
                name=context.node_id,
                waveform=source.waveform,
                labels=source.labels,
                sample_ids=source.sample_ids,
                valid_mask=source.valid_mask,
                provenance=Provenance(
                    node_id=context.node_id,
                    operation="sandbox_resource_requirement_identity_bridge",
                    parent_names=(source.name,),
                    rng_namespace=context.rng_namespace,
                    parameters={
                        "waveform_mutated": False,
                        "purpose": "compile_encoder_decoder_requirements_only",
                    },
                ),
                metadata={"resource_bridge_identity": True},
            )
        if context.node_id in self.diverse.view_modes:
            return self._diverse_view(context, inputs)
        return super()._dispatch_augmix(context, inputs)

    def _lhat_attack(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        if context.node_id in self.diverse.view_modes:
            return self._diverse_view(context, inputs)
        return super()._lhat_attack(context, inputs)

    def generate(
        self,
        *,
        clean_raw: torch.Tensor,
        targets: torch.Tensor,
        hash_ids: Sequence[str],
        classifier: nn.Module,
        base_seed: int,
        rng_identity: Sequence[str],
        composition_indices: torch.Tensor | None = None,
    ):
        if composition_indices is not None:
            raise ValueError("pure Diverse AugMax does not accept fixed20 exposures")
        if self._shared_cache is not None or self._generation_state is not None:
            raise RuntimeError("Diverse AugMax runtime generation is not re-entrant")
        resolved_identity = tuple(str(value) for value in rng_identity)
        endpoint_seed = _derive_seed(
            int(base_seed), resolved_identity, "shared_e8_endpoint_pool"
        )
        center = self._center_from_identity(resolved_identity)
        self._generation_state = (int(base_seed), resolved_identity)
        self._role_ordinals = {}
        try:
            assert self.encoder is not None and self.decoder is not None
            with _frozen_vae_components(self.encoder, self.decoder):
                self._shared_cache = self._build_shared_pool(
                    clean_raw,
                    targets,
                    classifier,
                    endpoint_seed=endpoint_seed,
                    center=center,
                )
                generated = super().generate(
                    clean_raw=clean_raw,
                    targets=targets,
                    hash_ids=hash_ids,
                    classifier=classifier,
                    base_seed=base_seed,
                    rng_identity=resolved_identity,
                    composition_indices=None,
                )
            # The whitelist runtime defaults every node diagnostic to the
            # accepted-view count.  Rate diagnostics such as ASR and
            # positive-hide have narrower, explicitly declared denominators;
            # replace only those sandbox-owned weights so epoch means remain
            # denominator-correct without changing the trainer interface.
            diagnostic_weights = dict(generated.diagnostic_weights)
            for value in generated.bundle.values.values():
                if not isinstance(value, WaveformView):
                    continue
                local_weights = value.metadata.get("diagnostic_weights")
                if not isinstance(local_weights, Mapping):
                    continue
                prefix = f"{value.provenance.node_id}/"
                for local_name, weight in local_weights.items():
                    full_name = f"{prefix}{local_name}"
                    if full_name not in generated.bundle.diagnostics:
                        raise RuntimeError(
                            "custom diagnostic weight lacks a recorded scalar: "
                            f"{full_name}"
                        )
                    resolved_weight = _strict_int(
                        weight,
                        f"diagnostic weight {full_name}",
                        minimum=1,
                    )
                    diagnostic_weights[full_name] = resolved_weight
            generated = replace(
                generated,
                diagnostic_weights=diagnostic_weights,
            )
            if sum(self._role_ordinals.values()) != len(self.diverse.view_modes):
                raise RuntimeError(
                    "Diverse AugMax graph generated the wrong number of views"
                )
            return generated
        finally:
            # This explicit lifecycle is part of the sandbox contract: no
            # endpoint tensor or stochastic state may leak into the next batch.
            self._shared_cache = None
            self._generation_state = None
            self._role_ordinals = {}


def _is_diverse_method(method: CompiledMethod) -> bool:
    return method.profile_name in DIVERSE_METHOD_IDS and isinstance(
        method.contracts.get("diverse_augmax"), Mapping
    )


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Patch only the D0--D6 Diverse runtimes and their objective scale."""

    import core.online_trainer as online_trainer

    original_factory = online_trainer.build_method_runtime
    original_compute_objective = online_trainer._compute_objective

    def patched_factory(
        method: CompiledMethod,
        *,
        model_name: str,
        config_root: str | Path,
        latent_pool: Any | None = None,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        minimum_std_mV: float = 1.0e-4,
        maximum_abs_mV: float = 20.0,
    ) -> MethodViewRuntime:
        if not _is_diverse_method(method):
            return original_factory(
                method,
                model_name=model_name,
                config_root=config_root,
                latent_pool=latent_pool,
                encoder=encoder,
                decoder=decoder,
                minimum_std_mV=minimum_std_mV,
                maximum_abs_mV=maximum_abs_mV,
            )
        return DiverseLatentAugMaxRuntime(
            method,
            model_name=model_name,
            config_root=Path(config_root).expanduser().resolve(),
            latent_pool=latent_pool,
            encoder=encoder,
            decoder=decoder,
            minimum_std_mV=minimum_std_mV,
            maximum_abs_mV=maximum_abs_mV,
        )

    def patched_compute_objective(**kwargs: Any):
        method = kwargs.get("method")
        scale_objective = isinstance(method, CompiledMethod) and _is_diverse_method(
            method
        )
        result = original_compute_objective(**kwargs)
        if not scale_objective:
            return result
        batch_type = type(result)
        return batch_type(
            total=result.total * OBJECTIVE_GLOBAL_SCALE,
            raw_terms=dict(result.raw_terms),
            weighted_terms={
                name: value * OBJECTIVE_GLOBAL_SCALE
                for name, value in result.weighted_terms.items()
            },
            valid_counts=dict(result.valid_counts),
        )

    online_trainer.build_method_runtime = patched_factory
    online_trainer._compute_objective = patched_compute_objective
    try:
        yield
    finally:
        online_trainer._compute_objective = original_compute_objective
        online_trainer.build_method_runtime = original_factory


install_runtime_adapter = patch_online_trainer_runtime


__all__ = [
    "D3_DEPTH23_METHOD_ID",
    "DIVERSE_METHOD_IDS",
    "DIVERSE_METHOD_VIEW_MODES",
    "DiverseAugMaxContract",
    "DiverseLatentAugMaxRuntime",
    "RADIUS_METRIC",
    "RADIUS_REGISTRY_SCHEMA_VERSION",
    "RadiusCapEntry",
    "RadiusCapRegistry",
    "SharedEndpointPool",
    "GeneralizedCorruptionBatch",
    "generate_generalized_corruption",
    "install_runtime_adapter",
    "load_radius_cap_registry",
    "patch_online_trainer_runtime",
    "probe_generalized_corruption_parity",
]
