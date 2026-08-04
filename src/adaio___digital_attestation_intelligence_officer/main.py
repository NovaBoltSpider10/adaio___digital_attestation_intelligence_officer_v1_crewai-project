#!/usr/bin/env python
import sys
import os
import json
import uvicorn

from adaio___digital_attestation_intelligence_officer.orchestrator import CaseOrchestrator
from adaio___digital_attestation_intelligence_officer.utils.tools import case_store

def run():
    """
    Run the ADAIO pipeline via local CLI mode.
    """
    print("Initializing ADAIO Case Orchestrator (CLI mode)...")
    orchestrator = CaseOrchestrator()

    project_root = os.getcwd()
    
    mock_intake_payload = {
        "applicant": {
            "name": "Jane Elizabeth Doe",
            "nationality": "United States",
            "id_number": "ID-99201"
        },
        "service_type": "attestation",
        "channel": "portal",
        "document_refs": {
            "transcript": os.path.join(project_root, "attestation.txt")
        }
    }

    case_id = orchestrator.create_case(mock_intake_payload)
    print(f"\n--- NEW CASE CREATED: {case_id} ---\n")

    orchestrator.run_pipeline(case_id)
    
    print("\n--- PIPELINE COMPLETE ---")
    
    final_case_state = orchestrator.case_store.get(case_id)

    if final_case_state:
        if final_case_state.get("state") == "CLOSED":
            print(
                "Final Recommendation:",
                final_case_state.get("decision_result", {}).get("recommendation", "Unknown")
            )
            print("\nFull Audit Log:")
            for log in final_case_state.get("audit_log", []):
                print(f"   - {log['ts']} | {log['agent'].upper()} | {log['event']}")
        else:
            print(f"Case ended in state: {final_case_state.get('state')}")
    else:
        print(f"Error: Case ID '{case_id}' was not found in case_store.")


def serve():
    """
    Run the FastAPI server for HTTP triggers.
    """
    print("Starting ADAIO FastAPI Web Server on http://0.0.0.0:8000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


def train():
    print("Training is disabled for this PoC architecture. Run 'run' or 'serve' instead.")

def replay():
    print("Replay is disabled for this PoC architecture. Run 'run' or 'serve' instead.")

def test():
    print("Testing is disabled for this PoC architecture. Run 'run' or 'serve' instead.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        print("Commands: run | serve | train | replay | test")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "serve":
        serve()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)