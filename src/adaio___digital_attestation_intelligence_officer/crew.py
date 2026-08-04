import os
import crewai.llms.cache as _crewai_cache
from crewai import LLM, Agent, Task
from crewai.project import CrewBase, agent, task

from adaio___digital_attestation_intelligence_officer.utils.tools import search_case_logs, verify_document_existence
from adaio___digital_attestation_intelligence_officer.models.schemas import (
    IntakeResult,
    DocumentAnalysisResult,
    VerificationResult,
    RiskResult,
    DecisionResult,
    CommunicationResult
)

_crewai_cache.mark_cache_breakpoint = lambda msg: msg

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
            tools=[search_case_logs, verify_document_existence],
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