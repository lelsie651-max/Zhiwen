from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.fact import FactEvidenceRole
from app.schemas.fact import FactIdentityInput, FactValueInput


class FactEvidenceInput(BaseModel):
    evidence_id: uuid.UUID
    role: FactEvidenceRole = FactEvidenceRole.SUPPORTING
    is_primary: bool = False

    model_config = ConfigDict(extra="forbid")


class AIProposalInput(BaseModel):
    identity: FactIdentityInput
    value: FactValueInput
    evidences: list[FactEvidenceInput] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_evidences(self) -> "AIProposalInput":
        if not self.evidences:
            raise ValueError("At least one evidence is required.")

        evidence_ids = [evidence.evidence_id for evidence in self.evidences]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Evidence identifiers must be unique.")

        supporting_evidences = [
            evidence for evidence in self.evidences if evidence.role == FactEvidenceRole.SUPPORTING
        ]
        if not supporting_evidences:
            raise ValueError("At least one supporting evidence is required.")

        primary_evidences = [evidence for evidence in self.evidences if evidence.is_primary]
        if len(primary_evidences) != 1:
            raise ValueError("Exactly one primary evidence is required.")

        if primary_evidences[0].role != FactEvidenceRole.SUPPORTING:
            raise ValueError("Primary evidence must use the supporting role.")

        return self


class HumanFactValueInput(BaseModel):
    identity: FactIdentityInput
    value: FactValueInput
    evidences: list[FactEvidenceInput] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_evidences(self) -> "HumanFactValueInput":
        if not self.evidences:
            return self

        evidence_ids = [evidence.evidence_id for evidence in self.evidences]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Evidence identifiers must be unique.")

        primary_evidences = [evidence for evidence in self.evidences if evidence.is_primary]
        if len(primary_evidences) != 1:
            raise ValueError("Exactly one primary evidence is required when evidences are provided.")

        if primary_evidences[0].role != FactEvidenceRole.SUPPORTING:
            raise ValueError("Primary evidence must use the supporting role.")

        return self


class FactValueDecisionInput(BaseModel):
    replace_current: bool = False

    model_config = ConfigDict(extra="forbid")
