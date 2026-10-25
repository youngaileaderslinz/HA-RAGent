import json

import pytest

from custom_components.ha_ragent.src.models.embedding.memory import Memory


def test_memory_prompt_preserves_identity_and_requested_facts():
    memory = Memory("1234", "The desk is in the office.", "2026-10-17")

    assert json.loads(memory.to_prompt_dict({"id": "memory_id", "content": "content", "created_at": "created_at"})) == {
        "memory_id": "1234",
        "content": "The desk is in the office.",
        "created_at": "2026-10-17",
    }


def test_memory_prompt_includes_only_requested_fields():
    memory = Memory("1234", "The desk is in the office.", "2026-10-17")

    assert json.loads(memory.to_prompt_dict({"content": "content"})) == {"content": "The desk is in the office."}
    assert memory.to_prompt_dict({}) == "{}"


@pytest.mark.parametrize("created_at", [None, ""])
def test_memory_prompt_omits_empty_fields_without_changing_storage(created_at):
    data = {
        "id": "1234", "content": "The desk is in the office.",
        "created_at": created_at, "retrieval_count": None,
    }
    memory = Memory.from_dict(data)

    assert json.loads(memory.to_prompt_dict({
        "id": "memory_id", "content": "content", "created_at": "created_at",
        "retrieval_count": "retrieval_count",
    })) == {
        "memory_id": "1234", "content": "The desk is in the office.",
    }
    assert memory.to_dict() == data


def test_memory_prompt_preserves_zero_retrieval_count_when_requested():
    memory = Memory("1234", "The desk is in the office.", "2026-10-17")

    assert json.loads(memory.to_prompt_dict({"id": "memory_id", "retrieval_count": "retrieval_count"})) == {
        "memory_id": "1234", "retrieval_count": 0,
    }


def test_memory_prompt_maps_class_fields_to_custom_names():
    memory = Memory("1234", "The desk is in the office.", "2026-10-17")

    assert json.loads(memory.to_prompt_dict({
        "id": "key", "content": "fact", "created_at": "date", "retrieval_count": "recalls",
    })) == {
        "key": "1234", "fact": "The desk is in the office.", "date": "2026-10-17", "recalls": 0,
    }
    assert memory.to_dict()["id"] == "1234"


def test_memory_prompt_rejects_list_selection():
    memory = Memory("1234", "The desk is in the office.", "2026-10-17")

    with pytest.raises(TypeError, match="dictionary mapping class field names"):
        memory.to_prompt_dict(["memory_id", "content"])
