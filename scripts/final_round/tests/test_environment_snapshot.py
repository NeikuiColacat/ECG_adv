import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.final_round import environment_snapshot


def test_collect_environment_snapshot_contains_core_runtime_keys():
    snapshot = environment_snapshot.collect_environment_snapshot()

    assert snapshot["python"]["executable"]
    assert snapshot["python"]["version"]
    assert "packages" in snapshot
    assert "torch" in snapshot["packages"]
    assert "cuda" in snapshot
    assert "tensorrt" in snapshot["packages"]
