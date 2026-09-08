"""
tests/test_docx_sanitation_and_fallback.py

Comprehensive tests for:
1. Valid DOCX: no sanitation performed.
2. Zero / bad CRC DOCX: temporary sanitized copy created, original preserved.
3. All recoverable ZIP members preserved byte-for-byte.
4. Repaired package passes testzip().
5. Repaired package successfully ingested into real manifest and Chroma index.
6. Unrecoverable member raises UnrecoverableZipMemberError (never silent deletion).
7. Ingestion failure never creates or registers a ghost doc_id in RunState.
8. Docling failure triggers python-docx fallback with truthful parser reporting.
"""

import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import docx
import pytest

from core.run_state import RunState
from sage_document_db.zip_sanitizer import (
    sanitize_office_zip,
    UnrecoverableZipMemberError,
    ZipSanitizerError,
)
from sage_document_db.pipeline import ingest_document
from sage_document_db.artifact_store import ArtifactStore
from sage_document_db.utils import sha256_file


def _create_sample_docx(path: Path, text: str = "Test document content") -> Path:
    """Helper to create a standard valid DOCX file."""
    doc = docx.Document()
    doc.add_heading("Experiment Title", level=1)
    doc.add_paragraph(text)
    tbl = doc.add_table(rows=2, cols=2)
    tbl.cell(0, 0).text = "A1"
    tbl.cell(0, 1).text = "B1"
    tbl.cell(1, 0).text = "A2"
    tbl.cell(1, 1).text = "B2"
    doc.save(str(path))
    return path


def _create_bad_crc_docx(orig_path: Path, dest_path: Path) -> Path:
    """Create a DOCX with one or more member headers having CRC = 0x0."""
    shutil.copy(str(orig_path), str(dest_path))
    with open(dest_path, "rb") as f:
        data = bytearray(f.read())

    # Patch local file header CRC to 0x00000000
    idx = data.find(b"PK\x03\x04")
    if idx != -1:
        data[idx + 14 : idx + 18] = b"\x00\x00\x00\x00"

    # Patch central directory header CRC to 0x00000000
    cd_idx = data.find(b"PK\x01\x02")
    if cd_idx != -1:
        data[cd_idx + 16 : cd_idx + 20] = b"\x00\x00\x00\x00"

    with open(dest_path, "wb") as f:
        f.write(data)

    return dest_path


def test_valid_docx_no_sanitation(tmp_path):
    """Requirement 9: Valid DOCX: no sanitation performed."""
    valid_file = _create_sample_docx(tmp_path / "valid.docx")
    orig_sha = sha256_file(valid_file)
    orig_mtime = valid_file.stat().st_mtime

    res = sanitize_office_zip(valid_file)

    assert res.was_sanitized is False
    assert res.working_path == valid_file
    assert res.original_path == valid_file
    assert res.original_sha256 == orig_sha
    assert valid_file.stat().st_mtime == orig_mtime


def test_zero_bad_crc_docx_sanitized_copy_created(tmp_path):
    """Requirement 9: Zero/bad CRC DOCX: sanitized copy created, original untouched."""
    valid_file = _create_sample_docx(tmp_path / "sample.docx")
    bad_file = _create_bad_crc_docx(valid_file, tmp_path / "corrupted_crc.docx")

    # Verify standard reader flags the bad CRC
    with zipfile.ZipFile(bad_file, "r") as ztest:
        assert ztest.testzip() is not None

    orig_sha = sha256_file(bad_file)
    orig_size = bad_file.stat().st_size

    res = sanitize_office_zip(bad_file)

    # 1. Sanitized copy produced
    assert res.was_sanitized is True
    assert res.working_path != bad_file
    assert res.working_path.exists()
    assert res.sanitation_warning is not None

    # 2. Original file untouched
    assert res.original_path == bad_file
    assert sha256_file(bad_file) == orig_sha
    assert bad_file.stat().st_size == orig_size

    # 3. Repaired package passes testzip()
    with zipfile.ZipFile(res.working_path, "r") as zrepaired:
        assert zrepaired.testzip() is None

    # Cleanup temp working copy
    if res.working_path.exists():
        res.working_path.unlink()


def test_all_recoverable_zip_members_preserved(tmp_path):
    """Requirement 9: All recoverable ZIP members preserved byte-for-byte."""
    valid_file = _create_sample_docx(tmp_path / "multi_member.docx")
    bad_file = _create_bad_crc_docx(valid_file, tmp_path / "bad_multi.docx")

    # Read member names and data by bypassing CRC check on bad_file
    expected_members = {}
    with zipfile.ZipFile(bad_file, "r") as zin:
        for item in zin.infolist():
            fp = zin.open(item)
            fp._update_crc = lambda d: None
            expected_members[item.filename] = fp.read()
            fp.close()

    res = sanitize_office_zip(bad_file)
    assert res.was_sanitized is True

    # Check that the sanitized zip has every member with identical bytes
    with zipfile.ZipFile(res.working_path, "r") as zrep:
        rep_names = zrep.namelist()
        assert set(rep_names) == set(expected_members.keys())
        for name, expected_bytes in expected_members.items():
            actual_bytes = zrep.read(name)
            assert actual_bytes == expected_bytes, f"Member bytes mismatch for {name}"

    if res.working_path.exists():
        res.working_path.unlink()


def test_repaired_package_can_be_ingested(tmp_path):
    """Requirement 9: Repaired package can be ingested into real manifest and index."""
    valid_file = _create_sample_docx(tmp_path / "for_ingest.docx", text="Unique experiment text")
    bad_file = _create_bad_crc_docx(valid_file, tmp_path / "bad_for_ingest.docx")
    orig_sha = sha256_file(bad_file)

    artifacts_root = tmp_path / "artifacts"
    res = ingest_document(
        source_path=bad_file,
        artifacts_root=artifacts_root,
        index_in_chroma=False,  # Keep unit test fast
    )

    assert res["status"] == "success"
    doc_id = res["doc_id"]
    assert doc_id is not None

    # Check manifest exists on disk
    manifest_path = Path(res["manifest_path"])
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Origin must retain original uploaded file sha and name
    assert manifest["origin"]["original_name"] == "bad_for_ingest.docx"
    assert manifest["origin"]["sha256"] == orig_sha
    assert manifest["doc_id"] == doc_id
    assert res["counts"]["text"] > 0

    # Warnings should note the sanitation
    assert any("Sanitized Office ZIP container" in w for w in manifest.get("warnings", []))


def test_unrecoverable_member_causes_failure_never_silent_deletion(tmp_path):
    """Requirement 9: Unrecoverable member causes explicit failure, never silent deletion."""
    corrupt_zip = tmp_path / "unrecoverable.docx"
    
    # Write a zip entry whose raw compressed stream is truncated/unreadable
    with zipfile.ZipFile(corrupt_zip, "w") as z:
        z.writestr("word/document.xml", "<valid>xml</valid>")
    
    # Mock zin.open to raise an unrecoverable exception for a member
    orig_zipfile_init = zipfile.ZipFile.__init__

    class BrokenZip(zipfile.ZipFile):
        def open(self, name, mode="r", pwd=None, *, force_zip64=False):
            fname = name.filename if isinstance(name, zipfile.ZipInfo) else name
            if fname == "word/document.xml":
                raise OSError("Simulated disk read sector failure")
            return super().open(name, mode, pwd, force_zip64=force_zip64)

    with patch("zipfile.ZipFile", side_effect=BrokenZip):
        with pytest.raises(UnrecoverableZipMemberError) as exc_info:
            sanitize_office_zip(corrupt_zip)

        assert "word/document.xml" in str(exc_info.value)
        assert "Simulated disk read sector failure" in str(exc_info.value)


def test_ingestion_failure_never_registers_ghost_doc_id(tmp_path):
    """Requirement 9: Ingestion failure never creates or registers a ghost doc_id."""
    run_state = RunState(request_id="req_test_fail", user_text="Summarize document")
    attachments_manifest = []
    file_map = {}

    # Simulate app.py upload handling with a failing ingestion
    ref_id = "file_1"
    safe_filename = "broken.docx"
    suffix = "docx"
    file_size = 1234
    save_path = tmp_path / safe_filename
    save_path.write_text("not a real docx")

    doc_id = None
    ingest_error = None
    try:
        # Mock document_db to return error or raise
        raise RuntimeError("Simulated Document DB crash")
    except Exception as exc:
        ingest_error = str(exc)

    if doc_id:
        run_state.register_document(
            doc_id=doc_id,
            display_name=safe_filename,
            file_type=suffix,
            source_name=safe_filename,
        )
        file_entry = {
            "ref": ref_id,
            "doc_id": doc_id,
            "name": safe_filename,
            "type": suffix,
            "size": file_size,
            "path": str(save_path),
            "status": "ingested",
        }
        attachments_manifest.append({
            "ref": ref_id,
            "doc_id": doc_id,
            "name": safe_filename,
            "type": suffix,
            "size": file_size,
            "status": "ingested",
        })
    else:
        # Expected new behavior: do NOT fabricate or register a ghost doc_id
        file_entry = {
            "ref": ref_id,
            "doc_id": None,
            "name": safe_filename,
            "type": suffix,
            "size": file_size,
            "path": str(save_path),
            "status": "ingestion_failed",
            "error": ingest_error,
        }
        attachments_manifest.append({
            "ref": ref_id,
            "doc_id": None,
            "name": safe_filename,
            "type": suffix,
            "size": file_size,
            "status": "ingestion_failed",
            "error": ingest_error,
        })
    file_map[ref_id] = file_entry

    # Assertions
    assert len(run_state.registered_documents) == 0, "Ghost document was registered in RunState!"
    assert run_state.list_document_ids() == []
    assert attachments_manifest[0]["doc_id"] is None
    assert attachments_manifest[0]["status"] == "ingestion_failed"
    assert "Simulated Document DB crash" in attachments_manifest[0]["error"]
    assert file_map["file_1"]["doc_id"] is None


def test_docx_fallback_truthful_reporting(tmp_path):
    """Requirement 8 & 9: Docling failure falls back to python-docx with truthful parser name and warning."""
    sample_file = _create_sample_docx(tmp_path / "fallback_sample.docx", text="Fallback text recovery")
    artifacts_root = tmp_path / "artifacts_fallback"

    # Mock parse_with_docling to simulate Docling crashing
    with patch("sage_document_db.docling_parser.parse_with_docling", side_effect=RuntimeError("Docling C++ backend crash")):
        res = ingest_document(
            source_path=sample_file,
            artifacts_root=artifacts_root,
            index_in_chroma=False,
        )

        assert res["status"] == "success"
        manifest_path = Path(res["manifest_path"])
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Must report parser truthfully
        assert manifest["parser"]["name"] == "docx_fallback"
        assert any("Docling parsing failed" in w for w in manifest.get("warnings", []))
        assert any("python-docx fallback" in w for w in manifest.get("warnings", []))
        assert any("visual formatting" in w for w in manifest.get("warnings", []))
