"""Source file fingerprints for reproducible visual evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFingerprint:
    path: str
    exists: bool
    size_bytes: int | None
    mtime_ns: int | None
    sha256: str | None


def fingerprint_file(path: str | Path) -> SourceFingerprint:
    path = Path(path)
    if not path.exists():
        return SourceFingerprint(
            path=str(path),
            exists=False,
            size_bytes=None,
            mtime_ns=None,
            sha256=None,
        )
    stat = path.stat()
    return SourceFingerprint(
        path=str(path),
        exists=True,
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        sha256=sha256_file(path),
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_records(records) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        fp = asdict(fingerprint_file(record.stp_path))
        fp.update(
            {
                "assembly_id": record.assembly_id,
                "part_id_raw": record.part_id_raw,
                "canonical_part_id": record.canonical_part_id,
            }
        )
        rows.append(fp)
    return rows


def compare_fingerprint_rows(
    expected_rows: list[dict[str, object]] | None,
    current_rows: list[dict[str, object]],
) -> dict[str, object]:
    if expected_rows is None:
        return {
            "status": "not_in_checkpoint",
            "message": "Checkpoint has no source fingerprints; target point clouds are reconstructed from current STEP files.",
            "mismatches": [],
        }
    expected_by_id = {str(row.get("canonical_part_id")): row for row in expected_rows}
    current_by_id = {str(row.get("canonical_part_id")): row for row in current_rows}
    mismatches: list[dict[str, object]] = []
    for part_id, expected in expected_by_id.items():
        current = current_by_id.get(part_id)
        if current is None:
            mismatches.append({"canonical_part_id": part_id, "reason": "missing_current_record"})
            continue
        for field in ("exists", "size_bytes", "sha256"):
            if expected.get(field) != current.get(field):
                mismatches.append(
                    {
                        "canonical_part_id": part_id,
                        "field": field,
                        "expected": expected.get(field),
                        "current": current.get(field),
                    }
                )
    status = "match" if not mismatches else "mismatch"
    return {
        "status": status,
        "message": "Current STEP fingerprints match checkpoint." if status == "match" else "Current STEP fingerprints differ from checkpoint.",
        "mismatches": mismatches,
    }
