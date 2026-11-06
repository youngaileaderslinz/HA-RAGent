from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import probatio
import pytest
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import ServiceValidationError

from custom_components.ha_ragent.src.const import CONF_MAX_MEMORY_ENTRIES, DOMAIN
from custom_components.ha_ragent.src.homeassistant.helpers.memory_manager import MemoryManager
from custom_components.ha_ragent.src.homeassistant.helpers import memory_service_helper
from custom_components.ha_ragent.src.homeassistant.services.export_memories_service import (
    _handle_export_memories,
    register_export_memories_service,
)
from custom_components.ha_ragent.src.homeassistant.services.import_memories_service import (
    _handle_import_memories,
    register_import_memories_service,
)
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.translation import RAGentTranslations


def record(number: int = 1, **overrides) -> dict:
    return {
        "id": f"{number:016x}",
        "content": f"The kitchen light is called Küche {number}.",
        "created_at": "2026-09-01T12:00:00+00:00",
        "retrieval_count": 3,
        **overrides,
    }


@pytest.fixture
def memory_hass(monkeypatch):
    RAGentTranslations._load("en")
    objects = {}

    async def list_objects(object_type, config_subentry, collection_name):
        return list(objects.get(collection_name, {}).values())

    async def upsert(config, collection, id_field, embeddings):
        objects.setdefault(collection, {}).update(
            (item.embedded_object.id, item.embedded_object) for item in embeddings
        )

    async def delete(config, collection, id_field, ids):
        for memory_id in ids:
            objects[collection].pop(memory_id, None)
        return len(ids)

    db = SimpleNamespace(
        async_list_objects=AsyncMock(side_effect=list_objects),
        async_ensure_collection_exists=AsyncMock(),
        async_upsert_objects=AsyncMock(side_effect=upsert),
        async_delete_objects=AsyncMock(side_effect=delete),
    )
    entry = SimpleNamespace(
        subentries={
            "source": SimpleNamespace(data={CONF_MAX_MEMORY_ENTRIES: 0}),
            "destination": SimpleNamespace(data={CONF_MAX_MEMORY_ENTRIES: 0}),
        },
        translations=RAGentTranslations("en"),
        embedder_backend=SimpleNamespace(async_embed_text=AsyncMock(return_value=[1.0, 0.5])),
        vector_db_backend=db,
    )
    registry_entries = {
        f"conversation.{name}": SimpleNamespace(
            platform=DOMAIN, config_entry_id="entry", config_subentry_id=subentry_id,
        )
        for name, subentry_id in (
            ("source", "source"), ("source_alias", "source"),
            ("destination", "destination"), ("no_subentry", None),
        )
    }
    registry_entries["conversation.other"] = SimpleNamespace(platform="other")
    registry_entries["sensor.not_agent"] = SimpleNamespace(platform=DOMAIN)
    registry = SimpleNamespace(async_get=registry_entries.get)
    monkeypatch.setattr(memory_service_helper.entity_registry, "async_get", lambda _: registry)
    monkeypatch.setattr(
        memory_service_helper.target, "async_extract_referenced_entity_ids",
        lambda _, selection: SimpleNamespace(
            referenced=selection.entity_ids, indirectly_referenced=set(),
        ),
    )
    hass = SimpleNamespace(data={DOMAIN: {"entry": entry}})
    return hass, entry, objects


@pytest.mark.parametrize("native_json", [False, True])
def test_services_round_trip_preserves_metadata_and_subentry_isolation(memory_hass, native_json):
    async def run():
        hass, entry, objects = memory_hass
        original = Memory.from_dict(record())
        existing = Memory.from_dict(record(2))
        objects["memories_source"] = {original.id: original}
        objects["memories_destination"] = {existing.id: existing}
        response = await _handle_export_memories(hass, SimpleNamespace(data={"entity_id": "conversation.source"}))

        assert json.loads(response["json"]) == {"memories": [record()]}
        assert "Küche" in response["json"]
        services = SimpleNamespace(has_service=Mock(return_value=False), async_register=Mock())
        register_import_memories_service(SimpleNamespace(services=services))
        schema = services.async_register.call_args.kwargs["schema"]
        payload = json.loads(response["json"]) if native_json else response["json"]
        writes = Mock()
        writes.attach_mock(entry.vector_db_backend.async_delete_objects, "delete")
        writes.attach_mock(entry.vector_db_backend.async_upsert_objects, "upsert")
        await _handle_import_memories(hass, SimpleNamespace(data=schema({
            "entity_id": "conversation.destination", "json": payload,
        })))
        assert objects["memories_source"] == {original.id: original}
        assert objects["memories_destination"] == {original.id: original}
        entry.vector_db_backend.async_delete_objects.assert_awaited_once_with(
            entry.subentries["destination"].data, "memories_destination", "id", [existing.id],
        )
        assert [call[0] for call in writes.mock_calls] == ["delete", "upsert"]
        entry.embedder_backend.async_embed_text.assert_awaited_once_with(
            entry.subentries["destination"].data,
            original.to_embedding_text(entry.translations), input_type="document",
        )
        embedding = entry.vector_db_backend.async_upsert_objects.call_args.args[3][0]
        assert isinstance(embedding, MemoryEmbedding)
        assert embedding.vector_embedding == [1.0, 0.5]

        # Repeated imports replace the previous imported memories as well.
        await _handle_import_memories(hass, SimpleNamespace(data={
            "entity_id": "conversation.destination",
            "json": json.dumps({"memories": [record(content="Updated fact.")]}),
        }))
        assert len(objects["memories_destination"]) == 1
        assert objects["memories_destination"][original.id].content == "Updated fact."

    asyncio.run(run())


@pytest.mark.parametrize("payload", [
    "invalid JSON", "null", "[]", "{}", '{"memories": {}}',
    json.dumps({"memories": [None]}),
    json.dumps({"memories": [record(id="bad")]}),
    json.dumps({"memories": [record(content=" ")]}),
    json.dumps({"memories": [record(content="x" * 1001)]}),
    json.dumps({"memories": [record(created_at="not a timestamp")]}),
    json.dumps({"memories": [record(retrieval_count=-1)]}),
    json.dumps({"memories": [record(retrieval_count=True)]}),
    json.dumps({"memories": [record(retrieval_count=1.5)]}),
    json.dumps({"memories": [record(), record()]}),
    json.dumps({"memories": [record(), record(2, content=None)]}),
    {}, {"memories": {}}, {"memories": [record(content=None)]},
])
def test_invalid_import_does_not_embed_or_write(memory_hass, payload):
    hass, entry, objects = memory_hass
    existing = Memory.from_dict(record(3))
    objects["memories_source"] = {existing.id: existing}
    with pytest.raises(ServiceValidationError):
        asyncio.run(_handle_import_memories(hass, SimpleNamespace(data={
            "entity_id": "conversation.source", "json": payload,
        })))
    entry.embedder_backend.async_embed_text.assert_not_awaited()
    entry.vector_db_backend.async_upsert_objects.assert_not_awaited()
    entry.vector_db_backend.async_delete_objects.assert_not_awaited()
    assert objects["memories_source"] == {existing.id: existing}


@pytest.mark.parametrize("handler", [_handle_export_memories, _handle_import_memories])
@pytest.mark.parametrize("entity_ids", [
    None, [], "conversation.unknown", "conversation.other", "sensor.not_agent",
    "conversation.no_subentry", ["conversation.source", "conversation.destination"],
])
def test_invalid_or_multiple_subentries_are_rejected(memory_hass, handler, entity_ids):
    hass, entry, _ = memory_hass
    with pytest.raises(ServiceValidationError):
        asyncio.run(handler(hass, SimpleNamespace(data={"entity_id": entity_ids, "json": "{}"})))
    entry.vector_db_backend.async_list_objects.assert_not_awaited()
    entry.vector_db_backend.async_upsert_objects.assert_not_awaited()


@pytest.mark.parametrize("unavailable", ["entry", "subentry"])
def test_unavailable_target_is_rejected(memory_hass, unavailable):
    hass, entry, _ = memory_hass
    if unavailable == "entry":
        hass.data[DOMAIN].pop("entry")
    else:
        entry.subentries.pop("source")
    with pytest.raises(ServiceValidationError, match="not available"):
        asyncio.run(_handle_export_memories(hass, SimpleNamespace(data={"entity_id": "conversation.source"})))
    entry.vector_db_backend.async_list_objects.assert_not_awaited()


def test_multiple_entities_for_one_subentry_are_deduplicated(memory_hass):
    hass, entry, _ = memory_hass
    response = asyncio.run(_handle_export_memories(hass, SimpleNamespace(data={
        "entity_id": ["conversation.source", "conversation.source_alias"],
    })))
    assert json.loads(response["json"]) == {"memories": []}
    entry.vector_db_backend.async_list_objects.assert_awaited_once()


def test_indirect_target_resolves_to_one_subentry(memory_hass, monkeypatch):
    hass, entry, _ = memory_hass
    monkeypatch.setattr(
        memory_service_helper.target, "async_extract_referenced_entity_ids",
        lambda *_: SimpleNamespace(
            referenced=set(), indirectly_referenced={"conversation.source", "sensor.not_agent"},
        ),
    )
    asyncio.run(_handle_export_memories(hass, SimpleNamespace(data={"device_id": "agent-device"})))
    entry.vector_db_backend.async_list_objects.assert_awaited_once_with(
        MemoryEmbedding, entry.subentries["source"].data, "memories_source",
    )


@pytest.mark.parametrize("has_existing_memories", [False, True])
def test_empty_memory_import_clears_memories_without_creating_collection(memory_hass, has_existing_memories):
    async def run():
        hass, entry, objects = memory_hass
        manager = MemoryManager(hass, "entry", "source")
        payload = await manager.async_export_memories()
        assert json.loads(payload) == {"memories": []}
        existing = Memory.from_dict(record())
        if has_existing_memories:
            objects["memories_source"] = {existing.id: existing}
        assert await manager.async_import_memories(payload) == 0
        assert objects.get("memories_source", {}) == {}
        if has_existing_memories:
            entry.vector_db_backend.async_delete_objects.assert_awaited_once_with(
                entry.subentries["source"].data, "memories_source", "id", [existing.id],
            )
        else:
            entry.vector_db_backend.async_delete_objects.assert_not_awaited()
        entry.vector_db_backend.async_ensure_collection_exists.assert_not_awaited()
        entry.vector_db_backend.async_upsert_objects.assert_not_awaited()
        entry.embedder_backend.async_embed_text.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("failure", [[], RuntimeError("Embedding failed")])
def test_embedding_failure_does_not_write_partial_import(memory_hass, failure):
    hass, entry, objects = memory_hass
    existing = Memory.from_dict(record(3))
    objects["memories_source"] = {existing.id: existing}
    entry.embedder_backend.async_embed_text.side_effect = [[1.0, 0.5], failure]
    with pytest.raises((ServiceValidationError, RuntimeError)):
        asyncio.run(_handle_import_memories(hass, SimpleNamespace(data={
            "entity_id": "conversation.source", "json": json.dumps({"memories": [record(), record(2)]}),
        })))
    entry.vector_db_backend.async_ensure_collection_exists.assert_not_awaited()
    entry.vector_db_backend.async_upsert_objects.assert_not_awaited()
    entry.vector_db_backend.async_delete_objects.assert_not_awaited()
    assert objects["memories_source"] == {existing.id: existing}


def test_import_enforces_subentry_storage_limit(memory_hass):
    hass, entry, objects = memory_hass
    entry.subentries["source"].data[CONF_MAX_MEMORY_ENTRIES] = 1
    asyncio.run(MemoryManager(hass, "entry", "source").async_import_memories(json.dumps({
        "memories": [record(retrieval_count=0), record(2, retrieval_count=10)],
    })))
    assert list(objects["memories_source"]) == [record(2)["id"]]


def test_service_schemas_accept_targets_and_require_json_text_or_object():
    hass = SimpleNamespace(services=SimpleNamespace(has_service=Mock(return_value=False), async_register=Mock()))
    register_export_memories_service(hass)
    register_import_memories_service(hass)
    export, import_ = hass.services.async_register.call_args_list
    assert export.kwargs["supports_response"] is SupportsResponse.ONLY
    for registration in (export, import_):
        data = {"entity_id": "conversation.source"}
        if registration is import_:
            data["json"] = '{"memories": []}'
        schema = registration.kwargs["schema"]
        assert schema(data) == {**data, "entity_id": ["conversation.source"]}
        with pytest.raises(probatio.Invalid):
            schema({**data, "entity_id": "not-an-entity"})
    with pytest.raises(probatio.Invalid):
        import_.kwargs["schema"]({"entity_id": "conversation.source"})
    data = {"entity_id": "conversation.source", "json": {"memories": []}}
    assert import_.kwargs["schema"](data) == {**data, "entity_id": ["conversation.source"]}
    for invalid in (None, [], 42, True):
        with pytest.raises(probatio.Invalid):
            import_.kwargs["schema"]({"entity_id": "conversation.source", "json": invalid})
