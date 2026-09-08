"""
sage_document_db/zip_sanitizer.py

Sanitizes Office ZIP packages (.docx, .pptx, .xlsx) that have invalid, missing,
or 0x0 CRC-32 checksums written by third-party exporters.

Behavioral Constraints:
1. Restrict sanitation strictly to: .docx, .pptx, .xlsx.
2. NEVER modify the original uploaded Office file in-place.
   Produces a temporary sanitized copy and preserves the original file.
3. NEVER silently skip ZIP members. If a member cannot be recovered, raises
   UnrecoverableZipMemberError (no silent deletion of XML/media).
4. The repair flow bypasses an incorrect CRC check only to recover the exact
   stored member bytes, then rewrites a valid ZIP with recalculated CRCs.
5. Verifies the repaired package with ZipFile.testzip() before returning.
6. Preserves source provenance by tracking the original uploaded-file identity
   and hash distinctly from the sanitized working-copy identity.
"""

from __future__ import annotations
import os
from pathlib import Path
import tempfile
import zipfile
from typing import NamedTuple, Optional

from .utils import sha256_file, safe_mkdir


OFFICE_ZIP_EXTENSIONS = {".docx", ".pptx", ".xlsx"}

# Decompression bomb defense limits
MAX_ZIP_MEMBERS = 10_000
MAX_ZIP_MEMBER_BYTES = 100 * 1024 * 1024          # 100 MB per member
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 250 * 1024 * 1024  # 250 MB total for entire package
MAX_ZIP_COMPRESSION_RATIO = 100  # uncompressed / compressed ratio check


class ZipSanitizerError(Exception):
    """Base exception for ZIP container sanitation issues."""
    pass


class UnrecoverableZipMemberError(ZipSanitizerError):
    """Raised when one or more ZIP members cannot be recovered.
    
    Guarantees that sanitizer NEVER silently drops or skips members.
    """
    def __init__(self, filename: str, member_name: str, cause: Exception):
        super().__init__(
            f"Failed to recover ZIP member '{member_name}' from '{filename}': {cause}. "
            f"Sanitizer will not silently drop members."
        )
        self.filename = filename
        self.member_name = member_name
        self.cause = cause


class SanitationResult(NamedTuple):
    was_sanitized: bool
    working_path: Path
    original_path: Path
    original_sha256: str
    original_size: int
    repaired_sha256: Optional[str] = None
    repaired_size: Optional[int] = None
    sanitation_warning: Optional[str] = None


def _validate_zip_limits(infolist: list[zipfile.ZipInfo], source_name: str) -> None:
    """Enforce member count, member size, total size, and compression ratio bounds."""
    if not infolist:
        raise ZipSanitizerError(
            f"ZIP package '{source_name}' contains no members."
        )

    if len(infolist) > MAX_ZIP_MEMBERS:
        raise ZipSanitizerError(
            f"ZIP package '{source_name}' exceeds maximum member limit ({len(infolist)} > {MAX_ZIP_MEMBERS})."
        )

    total_uncompressed = 0
    for item in infolist:
        if item.file_size > MAX_ZIP_MEMBER_BYTES:
            raise ZipSanitizerError(
                f"ZIP member '{item.filename}' in '{source_name}' exceeds individual size limit ({item.file_size} > {MAX_ZIP_MEMBER_BYTES} bytes)."
            )
        total_uncompressed += item.file_size
        if total_uncompressed > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
            raise ZipSanitizerError(
                f"ZIP package '{source_name}' exceeds maximum total uncompressed size ({total_uncompressed} > {MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES} bytes)."
            )
        if item.compress_size > 0:
            ratio = item.file_size / item.compress_size
            if ratio > MAX_ZIP_COMPRESSION_RATIO and item.file_size > (1024 * 1024):
                raise ZipSanitizerError(
                    f"Suspicious compression ratio ({ratio:.1f}x) for member '{item.filename}' in '{source_name}'."
                )


def sanitize_office_zip(source_path: Path | str) -> SanitationResult:
    """Check and sanitize an Office ZIP package if CRC checksums are invalid.

    Args:
        source_path: Path to the original document file.

    Returns:
        SanitationResult with working_path (either original or temporary repaired copy).
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: '{source_path}'")

    orig_size = source_path.stat().st_size
    orig_sha = sha256_file(source_path)

    # 1. Restrict to .docx, .pptx, .xlsx
    if source_path.suffix.lower() not in OFFICE_ZIP_EXTENSIONS:
        return SanitationResult(
            was_sanitized=False,
            working_path=source_path,
            original_path=source_path,
            original_sha256=orig_sha,
            original_size=orig_size,
        )

    # 2. Check if valid zip container
    if not zipfile.is_zipfile(source_path):
        return SanitationResult(
            was_sanitized=False,
            working_path=source_path,
            original_path=source_path,
            original_sha256=orig_sha,
            original_size=orig_size,
        )

    # 3. Test integrity via testzip() with decompression bomb defense
    bad_member = None
    try:
        with zipfile.ZipFile(source_path, "r") as z:
            _validate_zip_limits(z.infolist(), source_path.name)
            bad_member = z.testzip()
            if bad_member is None:
                # Completely healthy archive; no sanitation needed
                return SanitationResult(
                    was_sanitized=False,
                    working_path=source_path,
                    original_path=source_path,
                    original_sha256=orig_sha,
                    original_size=orig_size,
                )
    except ZipSanitizerError:
        raise
    except Exception as exc:
        bad_member = str(exc)

    # 4. Sanitation required: produce a temporary sanitized copy (NEVER edit in-place)
    temp_dir = Path(tempfile.gettempdir()) / "sage_sanitized"
    safe_mkdir(temp_dir)
    sanitized_path = temp_dir / f"sanitized_{orig_sha[:10]}_{source_path.name}"

    try:
        with zipfile.ZipFile(source_path, "r") as zin:
            infolist = zin.infolist()
            _validate_zip_limits(infolist, source_path.name)

            with zipfile.ZipFile(sanitized_path, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in infolist:
                    try:
                        fp = zin.open(item)
                        # Bypass the incorrect CRC-32 check only to recover the exact stored bytes
                        fp._update_crc = lambda d: None
                        member_bytes = fp.read(MAX_ZIP_MEMBER_BYTES + 1)
                        if len(member_bytes) > MAX_ZIP_MEMBER_BYTES:
                            fp.close()
                            raise ZipSanitizerError(
                                f"ZIP member '{item.filename}' exceeded uncompressed size limit during extraction."
                            )
                        fp.close()
                    except Exception as read_err:
                        if isinstance(read_err, ZipSanitizerError):
                            raise
                        # NEVER silently drop/skip members! Fail explicitly.
                        raise UnrecoverableZipMemberError(
                            filename=source_path.name,
                            member_name=item.filename,
                            cause=read_err,
                        )

                    # Write exact bytes with correctly recalculated CRC-32
                    zout.writestr(item.filename, member_bytes)

        # 5. Verify repaired package with testzip() before returning
        with zipfile.ZipFile(sanitized_path, "r") as ztest:
            test_res = ztest.testzip()
            if test_res is not None:
                raise ZipSanitizerError(
                    f"Sanitized ZIP package for '{source_path.name}' failed testzip() on member '{test_res}'."
                )

        repaired_size = sanitized_path.stat().st_size
        repaired_sha = sha256_file(sanitized_path)
        warning_msg = (
            f"Sanitized Office ZIP container with invalid CRC-32 (detected issue on '{bad_member}'). "
            f"Original file preserved (SHA: {orig_sha[:10]}...). Sanitized copy used for parsing."
        )

        return SanitationResult(
            was_sanitized=True,
            working_path=sanitized_path,
            original_path=source_path,
            original_sha256=orig_sha,
            original_size=orig_size,
            repaired_sha256=repaired_sha,
            repaired_size=repaired_size,
            sanitation_warning=warning_msg,
        )

    except Exception:
        # Clean up temporary file on failure
        if sanitized_path.exists():
            try:
                sanitized_path.unlink()
            except Exception:
                pass
        raise
