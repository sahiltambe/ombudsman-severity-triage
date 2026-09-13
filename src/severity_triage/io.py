"""Load the assessment workbook and check its checksum."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from severity_triage.config import Settings

__all__ = ["RawData", "load_datasets", "load_sheet", "sha256_of", "verify_checksum"]

log = logging.getLogger(__name__)

# Excel serial dates (days since 1899-12-30). CaseCreatedDate is stored as a number.
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


@dataclass(frozen=True)
class RawData:
    """The three sheets the pipeline consumes, exactly as read."""

    training: pd.DataFrame
    holdback: pd.DataFrame
    dictionary: pd.DataFrame
    workbook_sha256: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(workbook: Path, checksums: Path) -> str:
    """Compare the workbook's SHA-256 with the committed manifest.

    Returns the digest on success and raises ``ValueError`` on mismatch so a
    silently modified input can never produce a "reproducible" run.
    """
    actual = sha256_of(workbook)
    expected: str | None = None
    for line in checksums.read_text(encoding="ascii").splitlines():
        # sha256sum format: "<hex>  <filename>". Split once so filenames
        # containing spaces survive intact.
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[1].strip() == workbook.name:
            expected = parts[0].lower()
            break
    if expected is None:
        raise ValueError(f"No checksum entry for {workbook.name} in {checksums}")
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {workbook.name}: expected {expected}, got {actual}"
        )
    log.info("Workbook checksum verified (%s)", actual[:12])
    return actual


def excel_serial_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a numeric Excel serial column to ``datetime64``.

    Values that are already datetimes pass through unchanged, so the function
    is safe to call regardless of how the reader interpreted the cells.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    return _EXCEL_EPOCH + pd.to_timedelta(numeric, unit="D")


# Keep literal "None" as a category; do not treat it as NA.
_NA_TOKENS = ["", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "#N/A"]


def load_sheet(workbook: Path, sheet: str) -> pd.DataFrame:
    """Read one sheet, preserving the literal category ``"None"``."""
    frame = pd.read_excel(
        workbook,
        sheet_name=sheet,
        engine="openpyxl",
        keep_default_na=False,
        na_values=_NA_TOKENS,
    )
    log.info("Loaded sheet %r: %d rows x %d columns", sheet, *frame.shape)
    return frame


def load_datasets(settings: Settings, *, verify: bool = True) -> RawData:
    """Load training, holdback and dictionary sheets from the configured workbook."""
    workbook = settings.paths.raw_workbook
    digest = verify_checksum(workbook, settings.paths.checksums) if verify else sha256_of(workbook)
    sheets = settings.data.sheets
    return RawData(
        training=load_sheet(workbook, sheets.training),
        holdback=load_sheet(workbook, sheets.holdback),
        dictionary=load_sheet(workbook, sheets.dictionary),
        workbook_sha256=digest,
    )
