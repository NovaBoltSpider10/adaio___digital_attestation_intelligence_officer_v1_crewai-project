from typing import List, Union, Dict
from pydantic import BaseModel, Field


class IntakeResult(BaseModel):
    status: str = Field(description="Must be 'complete' or 'incomplete'")
    missing_fields: List[str] = Field(description="List of missing document fields if incomplete")
    normalized_request: dict = Field(description="Cleaned/standardized applicant + doc list")


class FieldValue(BaseModel):
    value: Union[str, float, int]
    confidence: float


class DocumentDetail(BaseModel):
    doc_id: str
    doc_type: str
    extracted_fields: Dict[str, FieldValue]
    quality_assessment: dict  # Structure: {"legibility": str, "flags": List[str]}


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