"""Regression checks for raw-download integrity and resume handling."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/download_posttraining_raw.py"
SPEC = importlib.util.spec_from_file_location("posttraining_raw", MODULE_PATH)
raw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(raw)


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data = data
        self.status_code = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data


def row(data):
    return {"path": "group/raw.bin", "size": len(data), "kind": "archive",
            "source_sha256": hashlib.sha256(data).hexdigest()}


def test_resume_checks_range_and_complete_hash(tmp_path, monkeypatch):
    content = b"original video bytes"
    target = tmp_path / "group/raw.bin"
    target.parent.mkdir()
    target.with_name("raw.bin.part").write_bytes(content[:5])

    def get(url, headers, **kwargs):
        assert headers == {"Range": "bytes=5-"}
        return Response(content[5:], 206, {"Content-Range": f"bytes 5-{len(content)-1}/{len(content)}"})

    monkeypatch.setattr(raw.requests, "get", get)
    result = raw.download_file(tmp_path, row(content))
    assert result["status"] == "verified"
    assert target.read_bytes() == content
    assert not target.with_name("raw.bin.part").exists()


def test_server_ignoring_range_replaces_partial_instead_of_appending(tmp_path, monkeypatch):
    content = b"complete original video"
    partial = tmp_path / "group/raw.bin.part"
    partial.parent.mkdir()
    partial.write_bytes(content[:7])
    monkeypatch.setattr(raw.requests, "get", lambda *a, **k: Response(content))
    raw.download_file(tmp_path, row(content))
    assert (tmp_path / "group/raw.bin").read_bytes() == content


def test_rejects_corruption_and_retains_partial(tmp_path, monkeypatch):
    content = b"good"
    monkeypatch.setattr(raw.requests, "get", lambda *a, **k: Response(b"evil"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        raw.download_file(tmp_path, row(content))
    assert not (tmp_path / "group/raw.bin").exists()
    assert (tmp_path / "group/raw.bin.part").read_bytes() == b"evil"


def test_rejects_wrong_resume_offset(tmp_path, monkeypatch):
    content = b"complete"
    monkeypatch.setattr(raw.requests, "get", lambda *a, **k: Response(content, 206, {"Content-Range": "bytes 2-7/8"}))
    with pytest.raises(ValueError, match="Content-Range"):
        raw.download_file(tmp_path, row(content))
    assert not (tmp_path / "group/raw.bin").exists()


@pytest.mark.parametrize("relative", ["../escape.mp4", "/tmp/escape.mp4"])
def test_rejects_paths_outside_collection(tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe"):
        raw.local_path(tmp_path, relative)


def test_resume_revalidates_existing_file(tmp_path, monkeypatch):
    content = b"data"
    target = tmp_path / "group/raw.bin"
    target.parent.mkdir()
    target.write_bytes(content)
    monkeypatch.setattr(raw.requests, "get", lambda *a, **k: pytest.fail("Unexpected network request"))
    assert raw.download_file(tmp_path, row(content))["sha256"] == row(content)["source_sha256"]
    target.write_bytes(b"fake")
    with pytest.raises(ValueError, match="Existing file failed"):
        raw.download_file(tmp_path, row(content))


def test_parallel_download_consolidates_after_existing_prefix(tmp_path, monkeypatch):
    content = b"original source video bytes"
    partial = tmp_path / "group/raw.bin.part"
    partial.parent.mkdir()
    partial.write_bytes(content[:4])

    def get(url, headers, **kwargs):
        assert headers["Range"] == f"bytes=4-{len(content)-1}"
        return Response(content[4:], 206, {"Content-Range": f"bytes 4-{len(content)-1}/{len(content)}"})

    monkeypatch.setattr(raw.requests, "get", get)
    result = raw.download_parallel(tmp_path, row(content), workers=2)
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert (tmp_path / "group/raw.bin").read_bytes() == content
    assert not (tmp_path / "group/raw.bin.segments").exists()


def test_segment_resumes_and_rejects_unbounded_response(tmp_path, monkeypatch):
    dest = tmp_path / "4-7"
    dest.with_name("4-7.part").write_bytes(b"ef")

    def get(url, headers, **kwargs):
        assert headers["Range"] == "bytes=6-7"
        return Response(b"gh", 206, {"Content-Range": "bytes 6-7/10"})

    monkeypatch.setattr(raw.requests, "get", get)
    assert raw.download_segment("http://unused/?download=true", 4, 7, 10, dest).read_bytes() == b"efgh"
    monkeypatch.setattr(raw.requests, "get", lambda *a, **k: Response(b"abcdefghij"))
    with pytest.raises(ValueError, match="bounded Content-Range"):
        raw.download_segment("http://unused/?download=true", 0, 3, 10, tmp_path / "0-3")
