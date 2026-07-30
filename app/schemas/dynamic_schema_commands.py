from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.schemas.dynamic_schema import DynamicSchemaIdentityInput, DynamicSchemaVersionInput


class HumanSchemaDraftInput(BaseModel):
    identity: DynamicSchemaIdentityInput
    version: DynamicSchemaVersionInput

    model_config = ConfigDict(extra="forbid")


class AISchemaProposalInput(BaseModel):
    identity: DynamicSchemaIdentityInput
    version: DynamicSchemaVersionInput

    model_config = ConfigDict(extra="forbid")
