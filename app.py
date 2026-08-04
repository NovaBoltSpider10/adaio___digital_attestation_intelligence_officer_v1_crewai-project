import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

from adaio___digital_attestation_intelligence_officer.orchestrator import CaseOrchestrator

app = FastAPI(
    title="ADAIO Digital Attestation Intelligence API",
    version="1.0.0"
)

# Shared orchestrator instance
orchestrator = CaseOrchestrator()


class ApplicantInfo(BaseModel):
    name: str
    nationality: str
    id_number: str


class NewApplicationRequest(BaseModel):
    applicant: ApplicantInfo
    service_type: str = Field(default="attestation")
    channel: str = Field(default="portal")
    document_refs: Dict[str, str]  # e.g. {"transcript": "path/to/file.pdf"}


class NewApplicationResponse(BaseModel):
    case_id: str
    message: str
    status: str


# ------------------------------------------------------------------
# 1. TRIGGER ENDPOINT (Starts the orchestration pipeline)
# ------------------------------------------------------------------
@app.post("/mock/attestation/new-application", response_model=NewApplicationResponse)
async def create_new_application(payload: NewApplicationRequest, background_tasks: BackgroundTasks):
    """
    Trigger endpoint for submitting a new attestation request.
    Creates a case and runs the orchestration pipeline asynchronously.
    """
    try:
        data = payload.model_dump()
        
        # Create the case in Orchestrator
        case_id = orchestrator.create_case(data)
        
        # Run pipeline as a background task to prevent request timeouts
        background_tasks.add_task(orchestrator.run_pipeline, case_id)

        return NewApplicationResponse(
            case_id=case_id,
            message="Application submitted successfully. Pipeline processing initiated.",
            status="ACCEPTED"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 2. TOOL VERIFICATION ENDPOINT (Used by verify_document_existence)
# ------------------------------------------------------------------
@app.get("/mock/documents/{doc_id:path}")
async def verify_document(doc_id: str, request: Request):
    """
    Called by CrewAI agent tools to verify document existence.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n==================================================")
    print(f"[{timestamp}] 🎯 TOOL CALLED BY AGENT")
    print(f"   --> Endpoint: GET {request.url.path}")
    print(f"   --> Received Doc ID: '{doc_id}'")
    print(f"==================================================\n")
    
    return {
        "status": "success",
        "doc_id": doc_id,
        "verified": True,
        "message": "API call received successfully."
    }


# ------------------------------------------------------------------
# 3. CASE STATUS ENDPOINT
# ------------------------------------------------------------------
@app.get("/mock/attestation/case/{case_id}")
async def get_case_status(case_id: str):
    """Utility endpoint to poll the status and state of a case."""
    case = orchestrator.case_store.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return case