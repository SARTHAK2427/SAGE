import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from orchestrator import orchestrator
from model_manager import model_manager

def main():
    docx_path = os.path.join(config.TEMP_DIR, "test_sample_report.docx")
    if not os.path.exists(docx_path):
        print(f"Error: {docx_path} does not exist. Run generate_test_docx.py first.")
        sys.exit(1)

    file_size = os.path.getsize(docx_path)
    attachments_manifest = [
        {
            "ref": "file_1",
            "name": "test_sample_report.docx",
            "type": "docx",
            "size": file_size
        }
    ]

    file_map = {
        "file_1": {
            "ref": "file_1",
            "name": "test_sample_report.docx",
            "type": "docx",
            "size": file_size,
            "path": docx_path
        }
    }

    user_objective = (
        "Read the attached report. Extract the revenue and customer count shown in the embedded screenshot. "
        "Then have the coding specialist write a Python function that safely converts a None API response into an empty list. "
        "Explain the complete result."
    )

    print("======================================================================")
    print("STARTING END-TO-END 3-MODEL SEQUENTIAL INTEGRATION TEST")
    print("======================================================================")

    try:
        result = orchestrator.run(
            user_objective=user_objective,
            attachments_manifest=attachments_manifest,
            file_map=file_map
        )
        print("\n======================================================================")
        print("FINAL SYNTHESIS FROM GEMMA 4B:")
        print("======================================================================")
        print(result["answer"])
        print("\n======================================================================")
        print("TELEMETRY:")
        print(result["telemetry"])
        print("======================================================================")
    finally:
        print("\nStopping model servers...")
        model_manager.stop_current()

if __name__ == "__main__":
    main()
