import json

from jinja2.sandbox import ImmutableSandboxedEnvironment

from custom_components.ha_ragent.src.const import DEFAULT_STATE_PROMPT
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.memory import Memory


def render_state(devices, memories, template=DEFAULT_STATE_PROMPT):
    environment = ImmutableSandboxedEnvironment()
    return environment.from_string(template).render(
        device_list=devices, memory_list=memories,
    )


def test_default_state_prompt_renders_compact_device_and_memory_facts():
    device = Device("sensor.desk", "Desk", "", "", domain=["sensor"])
    device.state = "0"
    device.attributes = {"unit_of_measurement": "°C", "precision": 0}
    memory = Memory("1234", "The desk is in the office.", "2026-10-17")

    rendered = render_state([device], [memory])
    records = [json.loads(line[2:]) for line in rendered.splitlines() if line.startswith("- ")]

    assert records == [
        {"memory_id": "1234", "content": memory.content},
        {
            "name": "sensor.desk", "friendly_name": "Desk", "domain": ["sensor"],
            "state": "0", "unit_of_measurement": "°C",
        },
    ]
    assert "null" not in rendered
    assert "[]" not in rendered
    assert ": " not in rendered


def test_prompt_template_supports_custom_field_names():
    device = Device("light.desk", "Desk light", "Office", "")
    memory = Memory("1234", "The desk is in the office.", "2026-10-17")
    template = """{{ device_list[0].to_prompt_dict({'id': 'entity', 'area_name': 'room'}) }}
{{ memory_list[0].to_prompt_dict({'id': 'key', 'content': 'fact'}) }}"""

    records = [json.loads(line) for line in render_state([device], [memory], template).splitlines()]

    assert records == [
        {"entity": "light.desk", "room": "Office"},
        {"key": "1234", "fact": memory.content},
    ]


def test_default_state_prompt_omits_empty_context_sections():
    rendered = render_state([], [])

    assert "<memories_context_prompt>" not in rendered
    assert "<devices_prompt>" not in rendered


def test_default_state_prompt_omits_null_memory_fields_and_unselected_metadata():
    memory = Memory.from_dict({
        "id": "1234", "content": "The desk is in the office.",
        "created_at": None, "retrieval_count": 5,
    })

    rendered = render_state([], [memory])
    records = [json.loads(line[2:]) for line in rendered.splitlines() if line.startswith("- ")]

    assert records == [{"memory_id": "1234", "content": "The desk is in the office."}]
    assert "null" not in rendered
