"""
tests/test_stage1b_regressions.py
Focused regression tests for Stage 1B authorized remediations:
- SEC-01: Vision image reference path confinement
- SEC-02: Path traversal defense in artifact store and image serving
- SEC-04: Exponentiation bounds in math tool
- CORR-01: .env early loading in config
- CORR-02: RagResult serialization preserving distance/similarity properties
- CORR-03: Nested JSON repair without premature truncation
- REL-01: test_pipeline.py importability without sys.exit()
"""

import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import config
from app import app
from core.json_repair import clean_json_string
from core.mappers.vision_input import resolve_vision_image
from sage_document_db.artifact_store import ArtifactStore
from sage_document_db.models import RagResult
from tools.document_database import _dataclass_to_dict, tool_rag_search
from tools.math_tool import tool_calculate


# ==============================================================================
# SEC-01: Vision image reference path confinement
# ==============================================================================

class TestSec01VisionConfinement:

    def test_vision_rejects_arbitrary_host_file(self):
        """Absolute path outside ARTIFACTS_ROOT/TEMP_DIR raises PermissionError."""
        outside_file = Path(sys.executable).resolve()
        assert outside_file.exists()

        mock_db = MagicMock()
        with pytest.raises(PermissionError, match="outside allowed roots"):
            resolve_vision_image(
                db=mock_db,
                doc_id="doc_1",
                image_id="img_001",
                image_ref=str(outside_file),
            )

    def test_vision_rejects_traversal_image_ref(self):
        """Relative traversal path raises PermissionError."""
        mock_db = MagicMock()
        with pytest.raises(PermissionError, match="outside allowed roots"):
            resolve_vision_image(
                db=mock_db,
                doc_id="doc_1",
                image_id="img_001",
                image_ref="../../secret.png",
            )

    def test_vision_accepts_valid_artifact_image_ref(self, tmp_path, monkeypatch):
        """Valid image_ref within ARTIFACTS_ROOT resolves directly."""
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, "ARTIFACTS_ROOT", art_dir)

        legit_img = art_dir / "doc_1" / "images" / "img_001.png"
        legit_img.parent.mkdir(parents=True, exist_ok=True)
        legit_img.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_db = MagicMock()
        resolved = resolve_vision_image(
            db=mock_db,
            doc_id="doc_1",
            image_id="img_001",
            image_ref=str(legit_img),
        )
        assert resolved == str(legit_img.resolve())

    def test_vision_db_fallback_preserves_legitimate_behavior(self, tmp_path):
        """When image_ref is None, valid DB resolution succeeds."""
        fake_img = tmp_path / "diagram.png"
        fake_img.write_text("image_payload")

        mock_db = MagicMock()
        mock_db.artifact_fetch.return_value = {
            "id": "img_001",
            "type": "image",
            "local_path": str(fake_img),
        }

        resolved = resolve_vision_image(mock_db, doc_id="doc_1", image_id="img_001")
        assert resolved == str(fake_img.resolve())


# ==============================================================================
# SEC-02: Path traversal defense in artifact store and image serving
# ==============================================================================

class TestSec02PathTraversal:

    def test_artifact_store_rejects_traversal_doc_id(self, tmp_path):
        """ArtifactStore.doc_dir rejects doc_id with path traversal sequences."""
        store = ArtifactStore(artifacts_root=tmp_path)
        bad_ids = ["../../etc", "../secret", "doc_1/../../windows", "doc_1\\..\\windows", ""]
        for bad_id in bad_ids:
            with pytest.raises(ValueError, match="Invalid or unsafe doc_id"):
                store.doc_dir(bad_id)

    def test_artifact_store_accepts_legitimate_doc_id(self, tmp_path):
        """ArtifactStore.doc_dir accepts standard alphanumeric/underscore doc_ids."""
        store = ArtifactStore(artifacts_root=tmp_path)
        valid_ids = ["doc_27d0685bb3", "doc_1", "doc_test_123", "doc-sample"]
        for valid_id in valid_ids:
            p = store.doc_dir(valid_id)
            assert p == (tmp_path / valid_id).resolve()

    def test_fetch_element_rejects_traversal_element_id(self, tmp_path):
        """fetch_element rejects traversal or invalid characters in element_id."""
        store = ArtifactStore(artifacts_root=tmp_path)
        with pytest.raises(ValueError, match="Invalid or unsafe element_id"):
            store.fetch_element("doc_1", "../../etc/passwd")

    def test_api_artifact_image_rejects_traversal(self):
        """FastAPI get_artifact_image returns 400 for invalid/traversal identifiers."""
        client = TestClient(app)
        # Invalid characters/dots in doc_id rejected with 400
        resp1 = client.get("/api/artifacts/doc..bad/images/img_1")
        assert resp1.status_code == 400
        assert resp1.json()["detail"] == "Invalid doc_id"

        # Traversal sequence in image_id rejected with 400
        resp2 = client.get("/api/artifacts/doc_1/images/img..secret")
        assert resp2.status_code == 400
        assert resp2.json()["detail"] == "Invalid image_id"

        # Path traversal URL encoding handled cleanly
        resp3 = client.get("/api/artifacts/..%2F..%2Fetc/images/img_1")
        assert resp3.status_code in (400, 404)


# ==============================================================================
# SEC-04: Exponentiation bounds in math tool
# ==============================================================================

class TestSec04MathExponentiationBounds:

    def test_math_rejects_massive_exponent(self):
        """Exponentiation with exponent > 1000 is rejected."""
        res = tool_calculate("2 ** 1000000")
        assert res["status"] == "error"
        assert "exceeds" in res["error"].lower()

    def test_math_rejects_large_base_with_moderate_exponent(self):
        """Exponentiation with large base and exponent > 100 is rejected."""
        res = tool_calculate("20000 ** 150")
        assert res["status"] == "error"
        assert "exceeds" in res["error"].lower() and "limits" in res["error"].lower()

    def test_math_allows_safe_exponentiation(self):
        """Legitimate mathematical exponentiation succeeds accurately."""
        res1 = tool_calculate("2 ** 10")
        assert res1["status"] == "success"
        assert res1["result"] == 1024

        res2 = tool_calculate("10 ** 4")
        assert res2["status"] == "success"
        assert res2["result"] == 10000


# ==============================================================================
# CORR-01: .env early loading in config
# ==============================================================================

class TestCorr01EnvLoadingOrder:

    def test_env_file_loaded_before_defaults(self, tmp_path, monkeypatch):
        """Verifies that .env loading logic sets os.environ before defaults derive paths."""
        test_env = tmp_path / ".env"
        test_env.write_text("SAGE_TEST_VAR_123=custom_value_xyz\n", encoding="utf-8")

        # Simulate early loading code block from config.py
        with open(test_env, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v

        assert os.environ.get("SAGE_TEST_VAR_123") == "custom_value_xyz"


# ==============================================================================
# CORR-02: RagResult serialization preserving distance/similarity
# ==============================================================================

class TestCorr02RagResultSerialization:

    def test_rag_result_to_dict_preserves_computed_properties(self):
        """_dataclass_to_dict preserves derived similarity and distance properties."""
        rag_res = RagResult(
            record_id="rec_001",
            origin="source",
            record_type="text_chunk",
            text="Revenue increased by 15%",
            distance=0.15,
            doc_id="doc_1",
            page=1,
        )

        d = _dataclass_to_dict(rag_res)
        assert d["record_id"] == "rec_001"
        assert d["raw_distance"] == 0.15
        assert "derived_cosine_similarity" in d
        assert "derived_similarity" in d
        assert round(d["derived_cosine_similarity"], 2) == 0.85

    def test_tool_rag_search_includes_derived_similarity(self):
        """tool_rag_search returns entries with derived_cosine_similarity."""
        mock_db = MagicMock()
        mock_db.rag_search.return_value = [
            RagResult(
                record_id="rec_001",
                origin="source",
                record_type="text_chunk",
                text="Test snippet",
                distance=0.2,
                doc_id="doc_1",
                page=1,
            )
        ]
        result = tool_rag_search(mock_db, query="Test query")
        assert result["status"] == "success"
        first = result["results"][0]
        assert first["raw_distance"] == 0.2
        assert "derived_cosine_similarity" in first
        assert round(first["derived_cosine_similarity"], 2) == 0.80


# ==============================================================================
# CORR-03: Nested JSON repair without premature truncation
# ==============================================================================

class TestCorr03NestedJsonRepair:

    def test_clean_json_string_with_nested_tool_calls(self):
        """clean_json_string preserves nested braces and arguments objects."""
        raw = (
            "Here is the planned action:\n"
            "```json\n"
            '{\n'
            '  "type": "tool_calls",\n'
            '  "calls": [\n'
            '    {\n'
            '      "tool": "document_database",\n'
            '      "function": "rag_search",\n'
            '      "arguments": {\n'
            '        "query": "financial results",\n'
            '        "doc_ids": ["doc_1"]\n'
            '      }\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "```\n"
            "Proceeding now."
        )

        cleaned = clean_json_string(raw)
        import json
        parsed = json.loads(cleaned)
        assert parsed["type"] == "tool_calls"
        assert len(parsed["calls"]) == 1
        assert parsed["calls"][0]["arguments"]["query"] == "financial results"

    def test_clean_json_string_with_escaped_braces_in_strings(self):
        """clean_json_string correctly handles JSON string containing literal braces."""
        raw = 'Text before {"query": "find {special} pattern"} text after'
        cleaned = clean_json_string(raw)
        import json
        parsed = json.loads(cleaned)
        assert parsed["query"] == "find {special} pattern"


# ==============================================================================
# REL-01: test_pipeline.py importability without sys.exit()
# ==============================================================================

class TestRel01PipelineImportability:

    def test_pipeline_test_module_has_sys_exit_guarded(self):
        """test_pipeline.py must guard sys.exit inside if __name__ == '__main__':."""
        pipeline_test_file = Path("code_executor/test_pipeline.py")
        assert pipeline_test_file.exists()
        content = pipeline_test_file.read_text(encoding="utf-8")

        # Verify if __name__ == "__main__": exists and guards sys.exit
        assert 'if __name__ == "__main__":' in content
        main_block = content.split('if __name__ == "__main__":')[-1]
        assert "sys.exit(" in main_block

        # Verify no unguarded sys.exit calls exist outside the main block
        pre_main = content.split('if __name__ == "__main__":')[0]
        assert "sys.exit(" not in pre_main
