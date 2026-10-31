import pickle
from pathlib import Path
from threading import Lock

import pytest

from langgraph.checkpoint.memory import PersistentDict


@pytest.mark.parametrize("existing", [False, True])
def test_failed_sync_preserves_error_and_removes_temporary_file(
    tmp_path: Path, existing: bool
) -> None:
    path = tmp_path / "checkpoint.pkl"
    store = PersistentDict(filename=str(path))
    if existing:
        store["value"] = "saved"
        store.sync()
    original = path.read_bytes() if existing else None

    store["lock"] = Lock()
    with pytest.raises(TypeError, match="pickle"):
        store.sync()

    assert not path.with_suffix(".pkl.tmp").exists()
    assert (path.read_bytes() if path.exists() else None) == original

    del store["lock"]
    store["value"] = "recovered"
    store.sync()
    assert pickle.loads(path.read_bytes()) == {"value": "recovered"}


def test_sync_replaces_existing_contents(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pkl"
    store = PersistentDict(filename=str(path))
    store["value"] = "initial"
    store.sync()
    store["value"] = "updated"
    store.sync()

    restored = PersistentDict(filename=str(path))
    restored.load()
    assert dict(restored) == {"value": "updated"}
    assert not path.with_suffix(".pkl.tmp").exists()
