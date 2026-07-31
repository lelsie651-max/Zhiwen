from __future__ import annotations

import dataclasses
import math
import re

import pytest
from pydantic import BaseModel

from app.agents import prompt_registry as pr
from app.agents.prompt_registry import (
    InvalidPromptDefinitionError,
    PromptAlreadyRegisteredError,
    PromptDefinition,
    PromptNotFoundError,
    PromptRegistry,
    build_default_registry,
    get_prompt,
    list_prompts,
)
from app.schemas.agent_fact_extraction import FactExtractionResponse


class _AltModel(BaseModel):
    x: int


def _definition(**overrides) -> PromptDefinition:
    base = dict(
        task_type="fact_extraction",
        agent_name="agent1_fact_extractor",
        agent_version="1.0.0",
        prompt_name="agent1_fact_extraction",
        prompt_version="1.0.0",
        system_template="system rules produce json",
        instruction_template="return one json object",
        response_model=FactExtractionResponse,
        temperature=0.1,
        max_output_tokens=8192,
    )
    base.update(overrides)
    return PromptDefinition(**base)


def test_prompt_definition_is_immutable():
    definition = _definition()
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.temperature = 0.5  # type: ignore[misc]


def test_contract_hash_is_deterministic_and_hex():
    a = _definition()
    b = _definition()
    assert a.contract_hash == b.contract_hash
    assert re.fullmatch(r"[0-9a-f]{64}", a.contract_hash)


@pytest.mark.parametrize(
    "overrides",
    [
        {"system_template": "different system"},
        {"instruction_template": "different instruction"},
        {"response_model": _AltModel},
        {"prompt_version": "2.0.0"},
        {"agent_version": "1.1.0"},
        {"temperature": 0.2},
        {"max_output_tokens": 4096},
        {"task_type": "schema_inference"},
    ],
)
def test_changing_any_contract_input_changes_hash(overrides):
    assert _definition().contract_hash != _definition(**overrides).contract_hash


def test_contract_hash_does_not_embed_secrets():
    # The definition intentionally has no api-key/env fields to hash.
    field_names = {f.name for f in dataclasses.fields(PromptDefinition)}
    assert "api_key" not in field_names
    assert not any("secret" in name or "env" in name for name in field_names)


def test_response_json_schema_is_the_model_schema():
    definition = _definition()
    assert definition.response_json_schema == FactExtractionResponse.model_json_schema()


def test_registry_rejects_duplicate_and_unknown():
    registry = PromptRegistry()
    registry.register(_definition())
    with pytest.raises(PromptAlreadyRegisteredError):
        registry.register(_definition())
    with pytest.raises(PromptNotFoundError):
        registry.get("missing", "1.0.0")


def test_registry_isolated_instances_do_not_share_state():
    r1 = PromptRegistry()
    r2 = PromptRegistry()
    r1.register(_definition())
    assert r1.get("agent1_fact_extraction", "1.0.0") is not None
    with pytest.raises(PromptNotFoundError):
        r2.get("agent1_fact_extraction", "1.0.0")


def test_registry_list_is_stably_sorted():
    registry = PromptRegistry()
    registry.register(_definition(prompt_name="b_prompt", prompt_version="1.0.0"))
    registry.register(_definition(prompt_name="a_prompt", prompt_version="2.0.0"))
    registry.register(_definition(prompt_name="a_prompt", prompt_version="1.0.0"))
    keys = [d.key for d in registry.list()]
    assert keys == [
        ("a_prompt", "1.0.0"),
        ("a_prompt", "2.0.0"),
        ("b_prompt", "1.0.0"),
    ]


def test_default_registry_has_official_fact_extraction_prompt():
    definition = get_prompt("agent1_fact_extraction", "1.0.0")
    assert definition.task_type == "fact_extraction"
    assert definition.agent_name == "agent1_fact_extractor"
    assert definition.agent_version == "1.0.0"
    assert definition.temperature == 0.1
    assert definition.response_model is FactExtractionResponse
    assert definition in list_prompts("fact_extraction")


def test_build_default_registry_is_independent_each_call():
    r1 = build_default_registry()
    # Registering into a fresh default copy must not raise or affect the module one.
    r2 = build_default_registry()
    assert r1 is not r2
    assert get_prompt("agent1_fact_extraction", "1.0.0") is not None


# --------------------------------------------------------------------------- #
# Renderer version participates in the contract hash
# --------------------------------------------------------------------------- #


def test_renderer_version_feeds_contract_hash(monkeypatch):
    definition = _definition()
    baseline = definition.contract_hash
    monkeypatch.setattr(pr, "RENDERER_VERSION", "9.9.9-test")
    assert definition.contract_hash != baseline


# --------------------------------------------------------------------------- #
# PromptDefinition validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    [
        {"task_type": ""},
        {"task_type": "   "},
        {"agent_name": ""},
        {"agent_name": "a" * 101},
        {"agent_version": "v" * 33},
        {"prompt_name": "p" * 101},
        {"prompt_version": "v" * 33},
        {"system_template": ""},
        {"instruction_template": "   "},
        {"response_model": dict},
        {"response_model": "not-a-model"},
    ],
)
def test_invalid_prompt_definition_fields_are_rejected(overrides):
    with pytest.raises(InvalidPromptDefinitionError):
        _definition(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"temperature": True},
        {"temperature": "0.1"},
        {"max_output_tokens": 0},
        {"max_output_tokens": -1},
        {"max_output_tokens": True},
        {"max_output_tokens": 1.5},
    ],
)
def test_invalid_sampling_params_are_rejected(overrides):
    with pytest.raises(InvalidPromptDefinitionError):
        _definition(**overrides)


def test_temperature_negative_zero_is_normalized():
    definition = _definition(temperature=-0.0)
    assert definition.temperature == 0.0
    assert math.copysign(1.0, definition.temperature) == 1.0  # collapsed to +0.0
    # -0.0 and +0.0 must produce an identical, representation-stable hash.
    assert definition.contract_hash == _definition(temperature=0.0).contract_hash


def test_invalid_definition_never_hashable_because_unconstructable():
    with pytest.raises(InvalidPromptDefinitionError):
        _definition(temperature=5.0)


def test_register_rejects_non_prompt_definition():
    registry = PromptRegistry()
    with pytest.raises(TypeError):
        registry.register({"prompt_name": "x"})  # type: ignore[arg-type]
