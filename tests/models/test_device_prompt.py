import json

import pytest

from custom_components.ha_ragent.src.models.embedding.device import Device


def test_device_prompt_omits_empty_fields_and_retains_live_values():
    device = Device("sensor.desk", "Desk", "", "", domain=["sensor"])
    device.state = "0"
    device.attributes = {
        "unit_of_measurement": "°C", "device_class": "temperature",
        "friendly_name": "Desk", "precision": 0, "enabled": False,
    }
    original = dict(device.attributes)

    candidate = json.loads(device.to_prompt_dict({
        "id": "name", "friendly_name": "friendly_name", "aliases": "aliases",
        "domain": "domain", "device_class": "device_class", "floor_name": "floor",
        "area_name": "area", "area_aliases": "area_aliases", "floor_aliases": "floor_aliases",
        "state": "state", "unit_of_measurement": "unit_of_measurement", "attributes": "attributes",
    }))

    assert candidate == {
        "name": "sensor.desk", "friendly_name": "Desk", "domain": ["sensor"],
        "state": "0", "unit_of_measurement": "°C", "device_class": "temperature",
        "attributes": {"precision": 0, "enabled": False},
    }
    assert device.attributes == original


def test_device_prompt_preserves_identity_aliases_and_locations():
    device = Device(
        "light.desk", "Desk light", "Office", "Upstairs", aliases=["Work light"],
        area_aliases=["Study"], floor_aliases=["First floor"], unit_of_measurement="%",
    )

    candidate = json.loads(device.to_prompt_dict({
        "id": "name", "aliases": "aliases", "area_name": "area", "area_aliases": "area_aliases",
        "floor_name": "floor", "floor_aliases": "floor_aliases",
        "unit_of_measurement": "unit_of_measurement", "state": "state",
    }))

    assert candidate["name"] == "light.desk"
    assert candidate["aliases"] == ["Work light"]
    assert candidate["area"] == "Office"
    assert candidate["area_aliases"] == ["Study"]
    assert candidate["floor"] == "Upstairs"
    assert candidate["floor_aliases"] == ["First floor"]
    assert candidate["unit_of_measurement"] == "%"
    assert "state" not in candidate


def test_device_prompt_includes_only_requested_fields():
    device = Device("light.desk", "Desk light", "Office", "Upstairs")
    device.state = "on"

    assert json.loads(device.to_prompt_dict({"friendly_name": "friendly_name", "state": "state"})) == {
        "friendly_name": "Desk light", "state": "on",
    }
    assert device.to_prompt_dict({}) == "{}"


def test_device_prompt_omits_requested_fields_with_null_values():
    device = Device("light.desk", "Desk light", "Office", "Upstairs")

    assert json.loads(device.to_prompt_dict({
        "id": "name", "state": "state", "aliases": "aliases", "device_class": "device_class",
    })) == {
        "name": "light.desk",
    }


def test_device_prompt_omits_nested_nulls_and_preserves_false_and_zero():
    device = Device("light.desk", "Desk light", "Office", "Upstairs")
    device.attributes = {
        "missing": None,
        "precision": 0,
        "enabled": False,
        "details": {"missing": None, "enabled": False},
        "values": [None, 0, False, {"missing": None, "value": 1}],
    }

    assert json.loads(device.to_prompt_dict({"attributes": "attributes"})) == {
        "attributes": {
            "precision": 0,
            "enabled": False,
            "details": {"enabled": False},
            "values": [0, False, {"value": 1}],
        },
    }
    assert device.attributes["missing"] is None
    assert device.attributes["details"]["missing"] is None
    assert device.attributes["values"][0] is None


def test_device_prompt_maps_class_fields_to_custom_names():
    device = Device(
        "light.desk", "Desk light", "Office", "Upstairs",
        device_labels=["Work"], services=["turn_on"],
    )
    device.state = "on"
    device.attributes = {"unit_of_measurement": "%", "precision": 0, "missing": None}

    assert json.loads(device.to_prompt_dict({
        "id": "entity", "friendly_name": "label", "area_name": "room",
        "floor_name": "level", "state": "status", "device_labels": "tags",
        "services": "actions", "unit_of_measurement": "unit", "attributes": "details",
        "unknown": "ignored",
    })) == {
        "entity": "light.desk", "label": "Desk light", "room": "Office",
        "level": "Upstairs", "status": "on", "tags": ["Work"],
        "actions": ["turn_on"], "unit": "%", "details": {"precision": 0},
    }
    assert device.attributes == {"unit_of_measurement": "%", "precision": 0, "missing": None}


@pytest.mark.parametrize("fields", [[], ["name"], None, "id"])
def test_device_prompt_rejects_non_dictionary_selection(fields):
    device = Device("light.desk", "Desk light", "Office", "Upstairs")

    with pytest.raises(TypeError, match="dictionary mapping class field names"):
        device.to_prompt_dict(fields)


def test_device_prompt_returns_compact_unicode_json_with_escaped_text():
    device = Device("light.desk", 'Licht "Büro"\n', "Office", "")

    assert device.to_prompt_dict({"id": "name", "friendly_name": "label"}) == (
        r'{"label":"Licht \"Büro\"\n","name":"light.desk"}'
    )
