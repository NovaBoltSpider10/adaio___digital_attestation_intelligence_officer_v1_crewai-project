import time
import datetime
import uuid
import logging
import json
from crewai import Crew, Process

from adaio___digital_attestation_intelligence_officer.crew import AdaioDigitalAttestationIntelligenceOfficerCrew
from adaio___digital_attestation_intelligence_officer.utils.tools import case_store
from adaio___digital_attestation_intelligence_officer.utils.ocr import extract_file_content

logger = logging.getLogger("ADAIO")

def calculate_auditable_score(verification_res: dict, risk_res: dict) -> float:
    """Calculates a deterministic confidence score (0.0 to 100.0) in Python."""
    match_rate = verification_res.get("overall_match_rate", 0.0) if verification_res else 0.0
    risk_score = risk_res.get("risk_score", 100) if risk_res else 100
    
    # 60% weight on match rate, 40% weight on inverted risk score
    match_component = match_rate * 60.0
    risk_component = max(0, 100 - risk_score) * 0.40
    
    return round(match_component + risk_component, 2)

class CaseOrchestrator:
    def __init__(self, case_store_ref=None):
        logger.info("Initializing ADAIO Case Orchestrator...")
        self.operating_mode = "shadow"
        self.crew_factory = AdaioDigitalAttestationIntelligenceOfficerCrew()
        self.case_store = case_store_ref if case_store_ref is not None else case_store
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

    def create_case(self, intake_payload: dict) -> str:
        case_id = f"ATT-{datetime.datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        self.case_store[case_id] = {
            "case_id": case_id,
            "state": "NEW",
            "created_at": ts,
            "updated_at": ts,
            "operating_mode": self.operating_mode,
            "request": intake_payload,
            "audit_log": [
                {"ts": ts, "agent": "orchestrator", "event": "case_created"}
            ]
        }
        return case_id

    def _log_event(self, case_id: str, agent_name: str, event_name: str):
        case_obj = self.case_store[case_id]
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        entry = {"ts": ts, "agent": agent_name, "event": event_name}
        case_obj["audit_log"].append(entry)
        case_obj["updated_at"] = ts

    def transition_to(self, case_id: str, next_state: str, agent="orchestrator", event="state_change"):
        case_obj = self.case_store[case_id]
        case_obj["state"] = next_state
        self._log_event(case_id, agent, event)
        logger.info(f"CASE {case_id} transitioned to {next_state}")

    def run_pipeline(self, case_id: str):
        try:
            # 1. Intake
            self.transition_to(case_id, "INTAKE_CHECK", "orchestrator", "starting_intake")
            res = self._execute_agent_step("intake", self.case_store[case_id])
            
            # --- FIX FOR ITEM 2: Notify applicant when intake is incomplete ---
            if res.get("status") == "incomplete":
                self.case_store[case_id]["intake_result"] = res
                self.transition_to(case_id, "AWAITING_INFO", "intake", "request_missing_info")
                
                # Execute communication step so applicant isn't "ghosted"
                comm_res = self._execute_agent_step("communication", self.case_store[case_id])
                self.case_store[case_id]["communication_result"] = comm_res
                self._log_event(case_id, "communication", "applicant_notified_missing_info")
                return

            self.case_store[case_id]["intake_result"] = res
            self.transition_to(case_id, self.transitions["INTAKE_CHECK"], "intake", "intake_complete")

            # OCR Step
            logger.info(f"Extracting raw text (with OCR & quality assessment) for case {case_id}...")
            case_obj = self.case_store[case_id]
            doc_refs = case_obj.get("request", {}).get("document_refs", {})

            # Store text and pre-evaluated quality metrics
            raw_docs_payload = {
                doc_key: extract_file_content(file_path)
                for doc_key, file_path in doc_refs.items()
            }
            self.case_store[case_id]["extracted_raw_text"] = raw_docs_payload

            # 2. Document Analysis
            res = self._execute_agent_step("document_analysis", self.case_store[case_id])
            self.case_store[case_id]["document_analysis_result"] = res
            self.transition_to(case_id, self.transitions["DOCUMENT_ANALYSIS"], "document_analysis", "analysis_complete")

            # 3. Verification
            res = self._execute_agent_step("verification", self.case_store[case_id])
            self.case_store[case_id]["verification_result"] = res
            self.transition_to(case_id, self.transitions["VERIFICATION"], "verification", "verification_complete")

            # 4. Risk
            res = self._execute_agent_step("risk", self.case_store[case_id])
            self.case_store[case_id]["risk_result"] = res
            self.transition_to(case_id, self.transitions["RISK_ASSESSMENT"], "risk", "risk_complete")

            # 5. Decision
            res = self._execute_agent_step("decision", self.case_store[case_id])
            self.case_store[case_id]["decision_result"] = res
            self.transition_to(case_id, self.transitions["DECISION"], "decision", "decision_complete")

            # --- FIX FOR ITEM 6: Aggregate Complete Verification Dossier ---
            dossier = {
                "case_id": case_id,
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "applicant_request": self.case_store[case_id].get("request"),
                "intake_result": self.case_store[case_id].get("intake_result"),
                "document_analysis_result": self.case_store[case_id].get("document_analysis_result"),
                "verification_result": self.case_store[case_id].get("verification_result"),
                "risk_result": self.case_store[case_id].get("risk_result"),
                "decision_result": self.case_store[case_id].get("decision_result"),
            }
            self.case_store[case_id]["dossier"] = dossier
            self.transition_to(case_id, "DOSSIER_GENERATED", "orchestrator", "dossier_created")

            # --- FIX FOR ITEM 6: Trigger Communication Agent before closing ---
            comm_res = self._execute_agent_step("communication", self.case_store[case_id])
            self.case_store[case_id]["communication_result"] = comm_res
            self.transition_to(case_id, "NOTIFIED", "communication", "notified_applicant")

            # Close Case
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

        # Context isolation payload builder
        context_builders = {
            "intake": lambda d: {"request": d.get("request")},
            "document_analysis": lambda d: {"extracted_raw_text": d.get("extracted_raw_text")},
            "verification": lambda d: {
                "request": d.get("request"),
                "document_analysis_result": d.get("document_analysis_result")
            },
            "risk": lambda d: {
                "case_id": d.get("case_id"),
                "request": d.get("request"),
                "document_analysis_result": d.get("document_analysis_result"),
                "verification_result": d.get("verification_result")
            },
            "decision": lambda d: {
                "case_id": d.get("case_id"),
                "request": d.get("request"),
                "document_analysis_result": d.get("document_analysis_result"),
                "verification_result": d.get("verification_result"),
                "risk_result": d.get("risk_result"),
                "calculated_score": calculate_auditable_score(
                    d.get("verification_result", {}), 
                    d.get("risk_result", {})
                )
            },
            "communication": lambda d: {
                "case_id": d.get("case_id"),
                "state": d.get("state"),
                "decision_result": d.get("decision_result"),
                "intake_result": d.get("intake_result"),
                "request": d.get("request")
            }
        }

        payload_context = context_builders.get(step_type, lambda d: d)(case_data)

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