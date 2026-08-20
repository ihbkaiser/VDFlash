import json
import sys
import stat
import zipfile
from pathlib import Path

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


def test_normalize_caption_record_builds_qwen_messages():
    from specforge.data.qwen25vl_manifest import normalize_caption_record

    result = normalize_caption_record(
        {
            "id": "x",
            "image": "a/b.jpg",
            "prompt": "<image> Describe",
            "response": "A cat.",
        },
        source_line=4,
    )

    assert result["messages"][0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "Describe"},
    ]
    assert result["messages"][1]["content"] == [
        {"type": "text", "text": "A cat."}
    ]
    assert result["source_line"] == 4


def test_manifest_rejects_image_traversal(tmp_path):
    from specforge.data.qwen25vl_manifest import safe_relative_image_path

    with pytest.raises(ValueError, match="outside image root"):
        safe_relative_image_path(tmp_path, "../secret.jpg")
    with pytest.raises(ValueError, match="outside image root"):
        safe_relative_image_path(tmp_path, r"..\secret.jpg")


def test_manifest_rejects_duplicate_ids(tmp_path):
    from specforge.data.qwen25vl_manifest import normalize_caption_jsonl

    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"id":"x","image":"a.jpg","prompt":"p","response":"r"}\n'
        '{"id":"x","image":"b.jpg","prompt":"p","response":"r"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        normalize_caption_jsonl(source, tmp_path / "out.jsonl")


def test_manifest_rejects_malformed_json_without_partial_output(tmp_path):
    from specforge.data.qwen25vl_manifest import normalize_caption_jsonl

    source = tmp_path / "source.jsonl"
    output = tmp_path / "out.jsonl"
    source.write_text(
        '{"id":"x","image":"a.jpg","prompt":"p","response":"r"}\n'
        "not-json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        normalize_caption_jsonl(source, output)
    assert not output.exists()


def test_manifest_writes_records_and_metadata_atomically(tmp_path):
    from specforge.data.qwen25vl_manifest import normalize_caption_jsonl

    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "a.jpg").write_bytes(b"image")
    source = tmp_path / "source.jsonl"
    output = tmp_path / "out.jsonl"
    source.write_text(
        '{"id":"x","image":"a.jpg","prompt":"p","response":"r"}\n',
        encoding="utf-8",
    )

    metadata = normalize_caption_jsonl(
        source,
        output,
        expected_records=1,
        image_root=image_root,
    )

    assert metadata["record_count"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "x"
    assert json.loads(
        Path(f"{output}.meta.json").read_text(encoding="utf-8")
    )["schema"] == "eagle3_qwen25vl_caption_manifest_v1"


def test_archive_materialization_rejects_traversal(tmp_path):
    from specforge.data.qwen25vl_manifest import materialize_image_archive

    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.jpg", b"bad")

    with pytest.raises(ValueError, match="outside image root"):
        materialize_image_archive(archive, tmp_path / "images")


def test_archive_materialization_rejects_zip_symlink(tmp_path):
    from specforge.data.qwen25vl_manifest import materialize_image_archive

    archive = tmp_path / "images.zip"
    info = zipfile.ZipInfo("link.jpg")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, "outside.jpg")

    with pytest.raises(ValueError, match="link"):
        materialize_image_archive(archive, tmp_path / "images")


def test_archive_materialization_does_not_remove_nonempty_existing_root(tmp_path):
    from specforge.data.qwen25vl_manifest import materialize_image_archive

    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("new.jpg", b"new")
    image_root = tmp_path / "images"
    image_root.mkdir()
    sentinel = image_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        materialize_image_archive(archive, image_root)

    assert sentinel.read_text(encoding="utf-8") == "keep"
