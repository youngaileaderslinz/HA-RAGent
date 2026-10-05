import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from custom_components.ha_ragent.migration import async_migrate_entry
from custom_components.ha_ragent.src.const import (
    CONF_PROMPT_LAYOUT,
    CONF_RULE_PROMPT,
    CONF_SELECTED_LANGUAGE,
    CONF_STATE_PROMPT,
    CONFIG_FLOW_VERSION,
    PROMPT_LAYOUT_COMBINED,
)


def test_migrate_legacy_language_option() -> None:
    update_entry = Mock()
    hass = SimpleNamespace(config_entries=SimpleNamespace(async_update_entry=update_entry))
    entry = SimpleNamespace(version=1, subentries={}, data={"backend": "ollama"}, options={
        CONF_SELECTED_LANGUAGE: "de", "host": "localhost"
    })

    assert asyncio.run(async_migrate_entry(hass, entry))
    update_entry.assert_called_once_with(
        entry,
        data={"backend": "ollama", CONF_SELECTED_LANGUAGE: "de"},
        options={"host": "localhost"},
        version=CONFIG_FLOW_VERSION,
    )


def test_migrate_keeps_data_language_when_both_locations_exist() -> None:
    update_entry = Mock()
    hass = SimpleNamespace(config_entries=SimpleNamespace(async_update_entry=update_entry))
    entry = SimpleNamespace(version=1, subentries={}, data={CONF_SELECTED_LANGUAGE: "en"}, options={
        CONF_SELECTED_LANGUAGE: "de"
    })

    assert asyncio.run(async_migrate_entry(hass, entry))
    assert update_entry.call_args.kwargs["data"][CONF_SELECTED_LANGUAGE] == "en"
    assert CONF_SELECTED_LANGUAGE not in update_entry.call_args.kwargs["options"]


def test_migration_is_not_repeated_for_current_entry() -> None:
    update_entry = Mock()
    update_subentry = Mock()
    hass = SimpleNamespace(config_entries=SimpleNamespace(
        async_update_entry=update_entry, async_update_subentry=update_subentry,
    ))
    subentry = SimpleNamespace(data={"rag_prompt": "legacy"})
    entry = SimpleNamespace(
        version=CONFIG_FLOW_VERSION, subentries={"agent": subentry},
        data={}, options={CONF_SELECTED_LANGUAGE: "de"},
    )

    assert asyncio.run(async_migrate_entry(hass, entry))
    update_entry.assert_not_called()
    update_subentry.assert_not_called()


def test_migration_rejects_unknown_version() -> None:
    update_entry = Mock()
    hass = SimpleNamespace(config_entries=SimpleNamespace(async_update_entry=update_entry))
    entry = SimpleNamespace(version=CONFIG_FLOW_VERSION + 1, subentries={}, data={}, options={})

    assert not asyncio.run(async_migrate_entry(hass, entry))
    update_entry.assert_not_called()


def test_migrate_legacy_agent_prompt_to_rule_prompt() -> None:
    update_entry = Mock()
    update_subentry = Mock()
    hass = SimpleNamespace(config_entries=SimpleNamespace(
        async_update_entry=update_entry, async_update_subentry=update_subentry,
    ))
    subentry = SimpleNamespace(data={"rag_prompt": "custom legacy prompt", "model": "llm"})
    entry = SimpleNamespace(version=1, data={}, options={}, subentries={"agent": subentry})

    assert asyncio.run(async_migrate_entry(hass, entry))
    update_subentry.assert_called_once_with(
        entry, subentry,
        data={
            CONF_RULE_PROMPT: "custom legacy prompt",
            CONF_STATE_PROMPT: "",
            CONF_PROMPT_LAYOUT: PROMPT_LAYOUT_COMBINED,
            "model": "llm",
        },
    )
    assert update_entry.call_args.kwargs["version"] == CONFIG_FLOW_VERSION


def test_migrate_legacy_prompt_preserves_existing_rule_prompt() -> None:
    update_subentry = Mock()
    hass = SimpleNamespace(config_entries=SimpleNamespace(
        async_update_entry=Mock(), async_update_subentry=update_subentry,
    ))
    subentry = SimpleNamespace(data={
        "rag_prompt": "old", CONF_RULE_PROMPT: "new", "rag_state_prompt": "state"
    })
    entry = SimpleNamespace(version=1, data={}, options={}, subentries={"agent": subentry})

    assert asyncio.run(async_migrate_entry(hass, entry))
    update_subentry.assert_called_once_with(
        entry, subentry, data={
            CONF_RULE_PROMPT: "new",
            CONF_STATE_PROMPT: "",
            CONF_PROMPT_LAYOUT: PROMPT_LAYOUT_COMBINED,
        }
    )
