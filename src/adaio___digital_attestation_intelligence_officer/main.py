#!/usr/bin/env python
import sys
import os
import json

# Import the new Orchestrator we built in crew.py
from adaio___digital_attestation_intelligence_officer.crew import CaseOrchestrator, case_store

def run():
    """
    Run the ADAIO pipeline via the Orchestrator state machine.
    """
    print("Initializing ADAIO Case Orchestrator...")
    orchestrator = CaseOrchestrator()

# Define the project root and files
    project_root = os.getcwd()
    
    mock_intake_payload = {
    "applicant": {
        "name": "Jane Doe",
        "nationality": "American",
        "id_number": "2021778000"
    },
    "service_type": "attestation",

    "channel": "portal",
    "document_refs": {
        "transcript": os.path.join(project_root, "transcript.txt")
    }
    }

    # 1. Trigger the case creation
    case_id = orchestrator.create_case(mock_intake_payload)
    print(f"\n--- NEW CASE CREATED: {case_id} ---\n")

    # 2. Run the state machine pipeline
    orchestrator.run_pipeline(case_id)
    
    # 3. Print the final result
    print("\n--- PIPELINE COMPLETE ---")
    final_case_state = case_store[case_id]
    
    if final_case_state["state"] == "CLOSED":
        print(
    "Final Recommendation:",
    final_case_state.get("decision", {}).get("recommendation", "Unknown")
)
        print("\nFull Audit Log:")
        for log in final_case_state["audit_log"]:
            print(f"  - {log['ts']} | {log['agent'].upper()} | {log['event']}")
    else:
        print(f"Case ended in state: {final_case_state['state']}")


def train():
    """Disabled for PoC: Standard CrewAI training bypasses the Orchestrator."""
    print("Training is disabled for this PoC architecture. Run the orchestrator via 'run' instead.")

def replay():
    """Disabled for PoC: Standard CrewAI replay bypasses the Orchestrator."""
    print("Replay is disabled for this PoC architecture. Run the orchestrator via 'run' instead.")

def test():
    """Disabled for PoC: Standard CrewAI testing bypasses the Orchestrator."""
    print("Testing is disabled for this PoC architecture. Run the orchestrator via 'run' instead.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)