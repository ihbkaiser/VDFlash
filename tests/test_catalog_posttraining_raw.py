"""Check that raw cataloging preserves references and rejects unsafe archives."""

import importlib
import io
import json
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def catalog_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("catalog_posttraining_raw")


def prepare(root, member="source/ytb_abcdefghijk.mp4"):
    group = root / "group"
    group.mkdir()
    archive = group / "videos.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo(member)
        info.size = 4
        output.addfile(info, io.BytesIO(b"data"))
    annotation_path = group / "source_cap_processed.json"
    annotations = [{"id": "caption-1", "video": "youtube_source/ytb_abcdefghijk.mp4",
                    "conversations": [{"from": "human", "value": "<image> Describe."},
                                      {"from": "gpt", "value": "Original answer."}]}]
    annotation_path.write_text(json.dumps(annotations))
    files = [{"path": "group/videos.tar.gz", "kind": "archive", "group": "group"},
             {"path": "group/source_cap_processed.json", "kind": "annotation", "group": "group"}]
    (root / "download_plan.json").write_text(json.dumps({"repo_id": "source", "revision": "pinned", "total_bytes": 100, "files": files}))
    (root / "download_status.json").write_text(json.dumps({"status": "complete", "errors": [], "files": [{**row, "status": "verified"} for row in files]}))
    return annotation_path.read_bytes()


def test_catalog_links_original_rows_without_mutating_source(tmp_path, catalog_module):
    original = prepare(tmp_path)
    summary = catalog_module.catalog(tmp_path)
    assert summary["unique_video_keys"] == 1
    assert summary["matched_original_annotation_rows"] == {"caption": 1}
    assert summary["features_cached"] is False
    assert summary["train_validation_test_split_created"] is False
    inventory = json.loads((tmp_path / "raw_video_inventory.jsonl").read_text())
    assert inventory["annotation_refs"][0]["row_index"] == 0
    assert (tmp_path / "group/source_cap_processed.json").read_bytes() == original
    assert not (tmp_path / "source").exists()


def test_catalog_rejects_traversal_without_extracting(tmp_path, catalog_module):
    prepare(tmp_path, "../escape.mp4")
    with pytest.raises(ValueError, match="Unsafe archive"):
        catalog_module.catalog(tmp_path)
    assert not (tmp_path / "raw_video_inventory.jsonl").exists()


def test_catalog_requires_all_downloads_verified(tmp_path, catalog_module):
    prepare(tmp_path)
    (tmp_path / "download_status.json").write_text(json.dumps({"status": "running", "errors": []}))
    with pytest.raises(RuntimeError, match="Complete and verify"):
        catalog_module.catalog(tmp_path)
