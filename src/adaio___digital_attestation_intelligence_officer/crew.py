import os
import time
import datetime
import uuid
import logging
import json
import base64
import requests
import crewai.llms.cache as _crewai_cache
from typing import List, Optional, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- OCR & DOCUMENT PROCESSING IMPORTS ---
from PIL import Image

try:
    import pypdf
except ImportError:
    import PyPDF2 as pypdf

# --- CREWAI IMPORTS ---
from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, task
from crewai_tools import FileReadTool, SerperDevTool
from crewai.tools import tool

_crewai_cache.mark_cache_breakpoint = lambda msg: msg

# =====================================================================
# SECTION 0: INITIALIZATION & ENVIRONMENT
# =====================================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = "sk-dummy-key" # Placeholder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ADAIO")
logger.info("ADAIO initialized")

# --- GLOBAL TOOLS ---
file_reader = FileReadTool()
search_tool = SerperDevTool()

@tool("Case Log Search")
def search_case_logs(query: str, query_type: str = "case_id"):
    """
    Look up cases. 
    query_type: 'case_id' or 'id_number'.
    """
    if query_type == "case_id":
        return case_store.get(query, "Case ID not found.")
    
    if query_type == "id_number":
        # Search all cases for matching applicant id_number
        matches = [
            cid for cid, data in case_store.items() 
            if data.get("request", {}).get("applicant", {}).get("id_number") == query
        ]
        return matches if matches else "No duplicate cases found."

# =====================================================================
# OCR & EXTRACTION UTILITY FUNCTION (OLLAMA / GEMMA DRIVEN)
# =====================================================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "gemma3")

def ocr_with_ollama(image_path_or_pil: Union[str, Image.Image]) -> str:
    """
    Sends an image to Ollama using a multimodal model to extract text.
    """
    try:
        if isinstance(image_path_or_pil, str):
            with open(image_path_or_pil, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        else:
            import io
            buffered = io.BytesIO()
            image_path_or_pil.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        payload = {
            "model": OLLAMA_VISION_MODEL,
            "prompt": "Extract and transcribe all legible text from this image exactly as shown. Do not add conversational commentary.",
            "images": [base64_image],
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "").strip()

    except Exception as e:
        logger.error(f"Ollama OCR failed: {e}")
        return f"[OCR ERROR: {str(e)}]"

def extract_file_content(file_path: str) -> str:
    """
    Utility function to extract text content from files deterministically.
    Handles standard text files, images via Ollama OCR, and PDFs (digital + Ollama OCR fallback).
    """
    if not os.path.exists(file_path):
        return "ERROR: File not found."

    ext = os.path.splitext(file_path)[1].lower()

    # 1. Image Files -> OCR via Ollama
    if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
        try:
            extracted = ocr_with_ollama(file_path)
            return extracted if extracted else "[OCR WARNING: No legible text detected in image]"
        except Exception as e:
            return f"ERROR performing OCR on image via Ollama: {str(e)}"

    # 2. PDF Files -> Digital extraction with Ollama OCR fallback for scanned pages
    elif ext == '.pdf':
        text = ""
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            logger.warning(f"Digital PDF extraction error on {file_path}: {e}")

        if text.strip():
            return text.strip()

        # Fallback to Ollama OCR if digital text extraction yielded no text (scanned PDF)
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path)
            ocr_text = ""
            for img in images:
                ocr_text += ocr_with_ollama(img) + "\n"
            return ocr_text.strip() if ocr_text.strip() else "[OCR WARNING: No legible text detected in PDF pages]"
        except Exception as e:
            if text.strip():
                return text.strip()
            return f"ERROR extracting PDF (Digital text empty & Ollama OCR fallback failed): {str(e)}"

    # 3. Plain text / CSV / Markdown / JSON files
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            return f"ERROR reading file: {str(e)}"

# =====================================================================
# SECTION 1: DATA SHAPES (PYDANTIC SCHEMAS)
# =====================================================================

class IntakeResult(BaseModel):
    status: str = Field(description="Must be 'complete' or 'incomplete'")
    missing_fields: List[str] = Field(description="List of missing document fields if incomplete")
    normalized_request: dict = Field(description="Cleaned/standardized applicant + doc list")

class ExtractedField(BaseModel):
    value: str
    confidence: float

class FieldValue(BaseModel):
    value: Union[str, float, int]
    confidence: float

class DocumentDetail(BaseModel):
    doc_id: str
    doc_type: str
    extracted_fields: dict[str, FieldValue]
    quality_assessment: dict # Structure: {"legibility": str, "flags": List[str]}

class DocumentAnalysisResult(BaseModel):
    documents: List[DocumentDetail]

class RiskIndicator(BaseModel):
    type: str
    detected: bool

class RiskResult(BaseModel):
    risk_indicators: List[RiskIndicator]
    risk_score: int = Field(description="Composite risk score from 0 to 100")
    risk_level: str = Field(description="low, medium, or high")

class DecisionResult(BaseModel):
    confidence_score: float
    recommendation: str = Field(description="approve, reject, or manual_review")
    rationale: str = Field(description="Natural language reason explaining the decision evidence")

class VerificationCheck(BaseModel):
    field: str
    source: str
    extracted_value: str
    source_value: str
    match: bool

class VerificationResult(BaseModel):
    checks: List[VerificationCheck]
    unverifiable_fields: List[str]
    overall_match_rate: float

class CommunicationResult(BaseModel):
    message_sent: bool
    channel: str
    message_id: str

# =====================================================================
# SECTION 2: AI WORKERS (CREWAI CONFIGURATION)
# =====================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_fast_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_base="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    temperature=0.0,
)

groq_smart_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_base="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    temperature=0.2,
)

@CrewBase
class AdaioDigitalAttestationIntelligenceOfficerCrew:
    """AdaioDigitalAttestationIntelligenceOfficer crew definition."""

    # --- AGENT DEFINITIONS ---
    @agent
    def intake_validation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["intake_validation_agent"],
            tools=[search_case_logs],
            allow_delegation=False,
            verbose=True,
            llm=groq_fast_llm,
        )

    @agent
    def document_analysis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["document_analysis_agent"],
            tools=[],
            allow_delegation=False,
            verbose=True,
            llm=groq_smart_llm,
        )

    @agent
    def verification_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["verification_agent"],
            tools=[],
            allow_delegation=False,
            verbose=True,
            llm=groq_fast_llm,
        )

    @agent
    def risk_assessment_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_assessment_agent"],
            tools=[search_case_logs],
            allow_delegation=False,
            verbose=True,
            llm=groq_fast_llm,
        )

    @agent
    def decision_support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["decision_support_agent"],
            tools=[search_case_logs],
            allow_delegation=False,
            verbose=True,
            llm=groq_smart_llm,
        )

    @agent
    def communication_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["communication_agent"],
            tools=[],
            allow_delegation=False,
            verbose=True,
            llm=groq_fast_llm,
        )

    # --- TASK DEFINITIONS ---
    @task
    def intake_payload_validation(self) -> Task:
        return Task(
            config=self.tasks_config["intake_payload_validation"],
            output_pydantic=IntakeResult 
        )
    
    @task
    def document_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["document_analysis_task"],
            output_pydantic=DocumentAnalysisResult
        )
    
    @task
    def verification_task(self) -> Task:
        return Task(
            config=self.tasks_config["verification_task"],
            output_pydantic=VerificationResult
        )
    
    @task
    def risk_assessment_task(self) -> Task:
        return Task(
            config=self.tasks_config["risk_assessment_task"],
            output_pydantic=RiskResult
        )
    
    @task
    def decision_recommendation_task(self) -> Task:
        return Task(
            config=self.tasks_config["decision_recommendation_task"],
            output_pydantic=DecisionResult
        )

    @task
    def notification_task(self) -> Task:
        return Task(
            config=self.tasks_config["notification_task"],
            output_pydantic=CommunicationResult
        )

# =====================================================================
# SECTION 3: THE TRAFFIC COP (STATE MACHINE ORCHESTRATOR)
# =====================================================================
case_store = {}

class CaseOrchestrator:
    def create_case(self, intake_payload: dict) -> str:
        """Initializes a new case and persists it to the store."""
        case_id = f"ATT-{datetime.datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        case_store[case_id] = {
            "case_id": case_id,
            "state": "NEW",
            "created_at": ts,
            "updated_at": ts,
            "operating_mode": self.operating_mode,
            "request": intake_payload,
            "audit_log": [
                { "ts": ts, "agent": "orchestrator", "event": "case_created" }
            ]
        }
        return case_id
    
    def __init__(self):
        logger.info("Initializing ADAIO Case Orchestrator...")
        self.operating_mode = "shadow" # Requirement: Hardcoded for PoC
        self.crew_factory = AdaioDigitalAttestationIntelligenceOfficerCrew()
        
        # Define the state machine structure
        self.transitions = {
            "NEW": "INTAKE_CHECK",
            "INTAKE_CHECK": "DOCUMENT_ANALYSIS",
            "AWAITING_INFO": "INTAKE_CHECK",
            "DOCUMENT_ANALYSIS": "VERIFICATION",
            "VERIFICATION": "RISK_ASSESSMENT",
            "RISK_ASSESSMENT": "DECISION",
            "DECISION": "DOSSIER_GENERATED",
            "DOSSIER_GENERATED": "NOTIFIED",
            "NOTIFIED": "CLOSED"
        }

    def _log_event(self, case_id, agent_name, event_name):
        """Atomic log entry and timestamp update."""
        case_obj = case_store[case_id]
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        entry = {"ts": ts, "agent": agent_name, "event": event_name}
        case_obj["audit_log"].append(entry)
        case_obj["updated_at"] = ts

    def transition_to(self, case_id, next_state, agent="orchestrator", event="state_change"):
        """Validates and applies state transitions."""
        case_obj = case_store[case_id]
        case_obj["state"] = next_state
        self._log_event(case_id, agent, event)
        logger.info(f"CASE {case_id} transitioned to {next_state}")

    def run_pipeline(self, case_id: str):
        """Drives the case through the state machine."""
        try:
            # 1. Intake
            self.transition_to(case_id, "INTAKE_CHECK", "orchestrator", "starting_intake")
            res = self._execute_agent_step("intake", case_store[case_id])
            if res.get("status") == "incomplete":
                self.transition_to(case_id, "AWAITING_INFO", "intake", "request_missing_info")
                return 
            
            self.transition_to(case_id, self.transitions["INTAKE_CHECK"], "intake", "intake_complete")

            # === DETERMINISTIC EXTRACTION STEP WITH OCR SUPPORT ===
            logger.info(f"Extracting raw text (with OCR support) for case {case_id}...")
            case_obj = case_store[case_id]
            doc_refs = case_obj.get("request", {}).get("document_refs", {})
            
            raw_docs_payload = {}
            for doc_key, file_path in doc_refs.items():
                raw_docs_payload[doc_key] = extract_file_content(file_path)
            
            # Inject the raw text directly into the case store context
            case_store[case_id]["extracted_raw_text"] = raw_docs_payload
            # =======================================================

            # 2. Document Analysis
            res = self._execute_agent_step("document_analysis", case_store[case_id])
            case_store[case_id]["document_analysis_result"] = res
            self.transition_to(case_id, self.transitions["DOCUMENT_ANALYSIS"], "document_analysis", "analysis_complete")

            # 3. Verification
            res = self._execute_agent_step("verification", case_store[case_id])
            case_store[case_id]["verification_result"] = res
            self.transition_to(case_id, self.transitions["VERIFICATION"], "verification", "verification_complete")

            # 4. Risk
            res = self._execute_agent_step("risk", case_store[case_id])
            case_store[case_id]["risk_result"] = res
            self.transition_to(case_id, self.transitions["RISK_ASSESSMENT"], "risk", "risk_complete")

            # 5. Decision
            res = self._execute_agent_step("decision", case_store[case_id])
            case_store[case_id]["decision_result"] = res
            self.transition_to(case_id, self.transitions["DECISION"], "decision", "decision_complete")
            
            # Final steps
            self.transition_to(case_id, "DOSSIER_GENERATED", "orchestrator", "dossier_created")
            self.transition_to(case_id, "NOTIFIED", "orchestrator", "notified_applicant")
            self.transition_to(case_id, "CLOSED", "orchestrator", "case_closed")

        except Exception as e:
            logger.error(f"UNRECOVERABLE ERROR: {e}")
            self.transition_to(case_id, "FAILED", "orchestrator", "error_occurred")

    def _execute_agent_step(self, step_type: str, case_data: dict) -> dict:
        step_map = {
            "intake": (self.crew_factory.intake_validation_agent, self.crew_factory.intake_payload_validation),
            "document_analysis": (self.crew_factory.document_analysis_agent, self.crew_factory.document_analysis_task),
            "verification": (self.crew_factory.verification_agent, self.crew_factory.verification_task),
            "risk": (self.crew_factory.risk_assessment_agent, self.crew_factory.risk_assessment_task),
            "decision": (self.crew_factory.decision_support_agent, self.crew_factory.decision_recommendation_task),
            "communication": (self.crew_factory.communication_agent, self.crew_factory.notification_task),
        }

        if step_type not in step_map:
            return {"status": "skipped"}

        agent_func, task_func = step_map[step_type]
        agent, task = agent_func(), task_func()
        
        logger.info(f"Applying rate-limit break before {step_type} step...")
        time.sleep(60)

        # =====================================================================
        # CONTEXT ISOLATION: Build scoped context tailored strictly to sub-agent SOW boundaries
        # =====================================================================
        payload_context = {}
        if step_type == "intake":
            payload_context = {"request": case_data.get("request")}
        elif step_type == "document_analysis":
            payload_context = {"extracted_raw_text": case_data.get("extracted_raw_text")}
        elif step_type == "verification":
            payload_context = {
                "request": case_data.get("request"),
                "document_analysis_result": case_data.get("document_analysis_result")
            }
        elif step_type == "risk":
            payload_context = {
                "case_id": case_data.get("case_id"),
                "request": case_data.get("request"),
                "document_analysis_result": case_data.get("document_analysis_result"),
                "verification_result": case_data.get("verification_result")
            }
        elif step_type == "decision":
            payload_context = {
                "case_id": case_data.get("case_id"),
                "request": case_data.get("request"),
                "document_analysis_result": case_data.get("document_analysis_result"),
                "verification_result": case_data.get("verification_result"),
                "risk_result": case_data.get("risk_result")
            }
        elif step_type == "communication":
            payload_context = {
                "case_id": case_data.get("case_id"),
                "decision_result": case_data.get("decision_result"),
                "request": case_data.get("request")
            }
        else:
            payload_context = case_data
        # =====================================================================

        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                single_step_crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )
                
                result = single_step_crew.kickoff(inputs={"case_data": json.dumps(payload_context)})
                
                if result.pydantic:
                    return result.pydantic.model_dump()
                return {"status": "complete", "raw": result.raw}

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {step_type}: {e}")
                if attempt == max_attempts - 1:
                    raise e
                
                logger.warning("Rate limit or error encountered. Backing off for 60 seconds before retry...")
                time.sleep(60)
        
        result = single_step_crew.kickoff(inputs={"case_data": json.dumps(payload_context)})
        
        if result.pydantic:
            return result.pydantic.model_dump() 
        return {"raw_output": result.raw}