# SAGE utilities

These are optional development and verification commands; production starts
from the repository root with `./run_sage.ps1` or `python app.py`.

- `python -m scripts.manual_db_test ...` — document ingestion/RAG CLI.
- `python -m scripts.generate_test_docx` then `python -m scripts.run_e2e_test` — legacy document E2E helper.
- `python -m scripts.verify_remote_transport` — live remote endpoint check.
- `python -m scripts.demo_e2e_flow` — mocked orchestration demo.
- `python -m scripts.test_system_integration` — legacy broad integration harness.
