import os
import time
import datetime
import uuid
import logging
import json
import crewai.llms.cache as _crewai_cache
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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
# SECTION 1: DATA SHAPES (PYDANTIC SCHEMAS)
# =====================================================================

class IntakeResult(BaseModel):
    status: str = Field(description="Must be 'complete' or 'incomplete'")
    missing_fields: List[str] = Field(description="List of missing document fields if incomplete")
    normalized_request: dict = Field(description="Cleaned/standardized applicant + doc list")

class ExtractedField(BaseModel):
    value: str
    confidence: float

# In SECTION 1: DATA SHAPES
class FieldValue(BaseModel):
    value: str
    confidence: float

class DocumentDetail(BaseModel):
    doc_id: str
    doc_type: str
    extracted_fields: dict[str, FieldValue]  # Maps to the structure in your requirement
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
            tools=[FileReadTool()],
            allow_delegation=False,
            verbose=True,
            llm=groq_smart_llm,
        )

    @agent
# In crew.py
    @agent
    def verification_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["verification_agent"],
            tools=[], # Keep empty: No external internet/mock tools needed for internal comparison
            allow_delegation=False,
            verbose=True,
            llm=groq_fast_llm, # Llama 3.3 is very efficient at this JSON-to-JSON comparison
        )

    @agent
    def risk_assessment_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_assessment_agent"],
            tools=[search_case_logs], # Keep this tool to query internal history
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
        
        # Here you would typically write to a database (mocked as file write)
        # with open('./data/case_logs.json', 'w') as f: json.dump(case_store, f)

    def transition_to(self, case_id, next_state, agent="orchestrator", event="state_change"):
        """Validates and applies state transitions."""
        case_obj = case_store[case_id]
        case_obj["state"] = next_state
        self._log_event(case_id, agent, event)
        logger.info(f"CASE {case_id} transitioned to {next_state}")

    def run_pipeline(self, case_id: str):
        """Drives the case through the state machine."""
        try:
            # Start the flow
            self.transition_to(case_id, "INTAKE_CHECK", "orchestrator", "starting_intake")
            
            # 1. Intake
            res = self._execute_agent_step("intake", case_store[case_id])
            if res.get("status") == "incomplete":
                self.transition_to(case_id, "AWAITING_INFO", "intake", "request_missing_info")
                return # Stop pipeline until resubmission
            
            self.transition_to(case_id, self.transitions["INTAKE_CHECK"], "intake", "intake_complete")

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
        # 1. Define the mapping of step_type to factory methods
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
        time.sleep(10)
        # 2. Implement retry logic (Retry once on failure)
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                single_step_crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )
                
                result = single_step_crew.kickoff(inputs={"case_data": json.dumps(case_data)})
                
                if result.pydantic:
                    return result.pydantic.model_dump()
                return {"status": "complete", "raw": result.raw}

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {step_type}: {e}")
                if attempt == max_attempts - 1:
                    # After the final attempt, re-raise to be handled by the Orchestrator
                    raise e
        
        result = single_step_crew.kickoff(inputs={"case_data": json.dumps(case_data)})
        
        if result.pydantic:
            return result.pydantic.model_dump() 
        return {"raw_output": result.raw}