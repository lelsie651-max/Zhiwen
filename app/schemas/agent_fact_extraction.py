"""Agent 1 fact-extraction structured output contract.

Strict Pydantic models describing exactly what the fact-extraction agent may
return. Every model forbids extra keys, so the AI can never smuggle in internal
fields (fact id, identity/value hashes, version, status, actor, current value id,
evidence id). Value-type rules are reused from the existing Fact contract rather
than maintained twice.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.fact import FactValueType
from app.schemas.fact import _normalize_optional_text, _normalize_required_text
from app.services.fact import InvalidFactProposalError, _normalize_value_by_type


EvidenceRole = Literal["supporting", "context"]


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


class EvidenceProposal(BaseModel):
    """A single AI-proposed evidence span, using 0-based half-open offsets."""

    block_ref: str = Field(pattern=r"^B[0-9]{4,}$")
    start_offset: int = Field(ge=0)
    end_offset: int
    role: EvidenceRole

    model_config = ConfigDict(extra="forbid")

    @field_validator("start_offset", "end_offset", mode="before")
    @classmethod
    def _offsets_must_be_int(cls, value: Any) -> Any:
        # Offsets are exact character indices: reject bool, float, and numeric
        # strings rather than let pydantic's lax mode coerce them.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("offset must be an integer")
        return value

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidenceProposal":
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self

    @property
    def interval_key(self) -> tuple[str, int, int]:
        return (self.block_ref, self.start_offset, self.end_offset)


class FactProposal(BaseModel):
    """A single AI-proposed fact with its supporting evidence."""

    subject_kind: str
    subject_key: str
    predicate_key: str
    scope_key: str | None = None
    value_type: FactValueType
    value_json: Any | None = None
    language_code: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceProposal] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("subject_kind")
    @classmethod
    def _validate_subject_kind(cls, value: str) -> str:
        return _normalize_required_text(value, field_name="subject_kind", max_length=64)

    @field_validator("subject_key", "predicate_key")
    @classmethod
    def _validate_keys(cls, value: str, info) -> str:
        return _normalize_required_text(value, field_name=info.field_name, max_length=255)

    @field_validator("scope_key")
    @classmethod
    def _validate_scope_key(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, field_name="scope_key", max_length=255)

    @field_validator("language_code")
    @classmethod
    def _validate_language_code(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, field_name="language_code", max_length=32)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_must_be_number(cls, value: Any) -> Any:
        # A real, finite JSON number: reject bool and numeric strings, and store
        # a float so the type is uniform downstream.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("confidence must be a number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("confidence must be finite")
        return numeric

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence_must_be_list(cls, value: Any) -> Any:
        # Do not let a tuple (or other sequence) be silently coerced into a list.
        if not isinstance(value, list):
            raise ValueError("evidence must be a list")
        return value

    @model_validator(mode="after")
    def validate_value_and_evidence(self) -> "FactProposal":
        # Reuse the canonical Fact value-type rules (no implicit string->number
        # coercion, strict null handling) instead of duplicating them here.
        try:
            normalized_value = _normalize_value_by_type(self.value_type.value, self.value_json)
        except InvalidFactProposalError as error:
            raise ValueError(str(error)) from None

        # Persist the Fact-layer normalized representation so the dedupe signature
        # and any downstream Fact service observe one identical value.
        self.value_json = normalized_value

        supporting = [e for e in self.evidence if e.role == "supporting"]
        if not supporting:
            raise ValueError("each fact requires at least one supporting evidence")

        intervals = [e.interval_key for e in self.evidence]
        if len(set(intervals)) != len(intervals):
            raise ValueError("duplicate evidence intervals are not allowed within a fact")
        return self

    @property
    def dedupe_signature(self) -> str:
        return _canonical(
            {
                "subject_kind": self.subject_kind,
                "subject_key": self.subject_key,
                "predicate_key": self.predicate_key,
                "scope_key": self.scope_key,
                "value_type": self.value_type.value,
                "value_json": self.value_json,
                "evidence": sorted(
                    [list(e.interval_key) + [e.role] for e in self.evidence]
                ),
            }
        )


class FactExtractionResponse(BaseModel):
    """The whole structured reply from the fact-extraction agent."""

    facts: list[FactProposal] = Field(default_factory=list, max_length=100)
    batch_summary: str | None = Field(default=None, max_length=2000)
    uncertainties: list[str] = Field(default_factory=list, max_length=50)

    model_config = ConfigDict(extra="forbid")

    @field_validator("facts", mode="before")
    @classmethod
    def _facts_must_be_list(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("facts must be a list")
        return value

    @field_validator("uncertainties", mode="before")
    @classmethod
    def _uncertainties_must_be_list(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("uncertainties must be a list")
        return value

    @field_validator("uncertainties")
    @classmethod
    def _validate_uncertainties(cls, value: list[str]) -> list[str]:
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("each uncertainty must be a non-empty string")
            if len(item) > 500:
                raise ValueError("each uncertainty must be at most 500 characters")
        return value

    @model_validator(mode="after")
    def validate_unique_facts(self) -> "FactExtractionResponse":
        signatures = [fact.dedupe_signature for fact in self.facts]
        if len(set(signatures)) != len(signatures):
            raise ValueError("duplicate facts (identity + value + evidence) are not allowed")
        return self
