"""Bounded, resumable storage for raw DFlash attention records."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import torch


class AttentionShardWriter:
    """Write attention tensors in bounded compressed shards and track cases."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        max_buffer_bytes: int = 128 * 1024 * 1024,
        compression_level: int = 1,
        compression_threads: int = 2,
    ) -> None:
        if max_buffer_bytes <= 0 or compression_threads <= 0:
            raise ValueError("buffer size and compression thread count must be positive")
        if not 1 <= compression_level <= 19:
            raise ValueError("zstd compression level must be in [1, 19]")
        executable = shutil.which("zstd")
        if executable is None:
            raise RuntimeError("the zstd executable is required to write attention shards")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executable = executable
        self.max_buffer_bytes = int(max_buffer_bytes)
        self.compression_level = int(compression_level)
        self.compression_threads = int(compression_threads)
        self.records: list[dict[str, Any]] = []
        self.pending_bytes = 0
        self.shard_index = self._last_shard_index()
        self.active_case_id: str | None = None
        self.case_shards: list[dict[str, Any]] = []
        self.case_record_count = 0

    def _last_shard_index(self) -> int:
        indices = []
        for path in self.output_dir.glob("attention_*.pt.zst"):
            index_text = path.name.removeprefix("attention_").split(".", 1)[0]
            if index_text.isdigit():
                indices.append(int(index_text))
        return max(indices, default=0)

    def begin_case(self, case_id: str) -> None:
        if self.active_case_id is not None:
            raise RuntimeError(f"case already active: {self.active_case_id}")
        self.active_case_id = str(case_id)
        self.case_shards = []
        self.case_record_count = 0

    def append(self, record: Mapping[str, Any]) -> None:
        if self.active_case_id is None:
            raise RuntimeError("begin_case must be called before writing attention")
        weights = record.get("weights")
        if not torch.is_tensor(weights) or weights.ndim != 4:
            raise ValueError("attention weights must have shape [batch, head, query, key]")
        if int(weights.shape[-1]) != int(record.get("key_length", -1)):
            raise ValueError("attention key length metadata does not match the tensor")
        key_map = record.get("key_map")
        if not isinstance(key_map, Mapping) or len(key_map.get("token_ids", ())) != weights.shape[-1]:
            raise ValueError("attention record is missing its aligned token key map")
        saved = dict(record)
        saved["case_id"] = self.active_case_id
        self.records.append(saved)
        self.pending_bytes += int(weights.numel() * weights.element_size())
        self.case_record_count += 1
        if self.pending_bytes >= self.max_buffer_bytes:
            self.flush()

    def flush(self) -> dict[str, Any] | None:
        if not self.records:
            return None
        buffer = io.BytesIO()
        torch.save(self.records, buffer)
        payload = buffer.getbuffer()
        self.shard_index += 1
        final_path = self.output_dir / f"attention_{self.shard_index:06d}.pt.zst"
        partial_path = final_path.with_name(final_path.name + ".partial")
        stderr_path = partial_path.with_name(partial_path.name + ".stderr")
        try:
            with partial_path.open("wb") as output, stderr_path.open("wb") as errors:
                process = subprocess.Popen(
                    [
                        self.executable,
                        "-q",
                        f"-{self.compression_level}",
                        f"-T{self.compression_threads}",
                        "-c",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=errors,
                )
                assert process.stdin is not None
                process.stdin.write(payload)
                process.stdin.close()
                exit_code = process.wait()
                output.flush()
                os.fsync(output.fileno())
            if exit_code != 0:
                raise RuntimeError(
                    f"zstd exited {exit_code}: {stderr_path.read_text(errors='replace')}"
                )
            os.replace(partial_path, final_path)
            stderr_path.unlink(missing_ok=True)
        except BaseException:
            partial_path.unlink(missing_ok=True)
            stderr_path.unlink(missing_ok=True)
            raise
        sha256 = hashlib.sha256()
        with final_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha256.update(chunk)
        result = {
            "path": str(final_path),
            "sha256": sha256.hexdigest(),
            "compressed_bytes": final_path.stat().st_size,
            "uncompressed_bytes": len(payload),
            "record_count": len(self.records),
        }
        self.case_shards.append(result)
        self.records.clear()
        self.pending_bytes = 0
        return result

    def finish_case(self) -> dict[str, Any]:
        if self.active_case_id is None:
            raise RuntimeError("no active attention case")
        self.flush()
        result = {
            "case_id": self.active_case_id,
            "record_count": self.case_record_count,
            "shards": list(self.case_shards),
        }
        self.active_case_id = None
        self.case_shards = []
        self.case_record_count = 0
        return result

    def abort_case(self) -> dict[str, Any]:
        """Discard buffered records for a failed case and release the writer."""

        if self.active_case_id is None:
            raise RuntimeError("no active attention case")
        result = {
            "case_id": self.active_case_id,
            "record_count": sum(int(shard["record_count"]) for shard in self.case_shards),
            "discarded_record_count": len(self.records),
            "shards": list(self.case_shards),
            "aborted": True,
        }
        self.records.clear()
        self.pending_bytes = 0
        self.active_case_id = None
        self.case_shards = []
        self.case_record_count = 0
        return result


def load_attention_shard(path: str | Path, *, zstd: str | None = None) -> list[dict[str, Any]]:
    """Read one ``.pt.zst`` shard produced by :class:`AttentionShardWriter`."""

    executable = zstd or shutil.which("zstd")
    if executable is None:
        raise RuntimeError("the zstd executable is required to read attention shards")
    completed = subprocess.run(
        [executable, "-q", "-d", "-c", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    value = torch.load(io.BytesIO(completed.stdout), map_location="cpu", weights_only=False)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"invalid attention shard payload: {path}")
    return value


__all__ = ["AttentionShardWriter", "load_attention_shard"]
