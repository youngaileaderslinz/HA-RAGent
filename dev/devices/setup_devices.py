import asyncio
import json

import websockets

HA_URL = "ws://localhost:8123/api/websocket"
TOKEN = input("Enter your Home Assistant Long-Lived Access Token: ")
AREAS = ["Living Room", "Kitchen", "Dining Room", "Office", "Hallway", "Bedroom 1", "Bedroom 2", "Bedroom 3", "Bathroom", "Garage"]
FLOORS = {"1st Floor": ["Living Room", "Kitchen", "Dining Room", "Office", "Hallway", "Bathroom", "Garage"], "2nd Floor": ["Bedroom 1", "Bedroom 2", "Bedroom 3"]}
# Keep this explicit so repeated dev setups produce the same Assist coverage.
ASSIST_DISABLED_DEVICES = {
    "Garage Freezer",
    "Bedroom 3 Speaker",
    "Kitchen Oven",
}
message_counter = 0

async def send_message(ws, msg_type, **extra):
    global message_counter
    message_counter += 1
    await ws.send(json.dumps({"id": message_counter, "type": msg_type, **extra}))
    while True:
        response = json.loads(await ws.recv())
        if response.get("id") == message_counter:
            return response

async def authenticate(ws):
    assert json.loads(await ws.recv())["type"] == "auth_required"
    await ws.send(json.dumps({"type": "auth", "access_token": TOKEN}))
    assert json.loads(await ws.recv())["type"] == "auth_ok"

async def create_areas(ws):
    current = (await send_message(ws, "config/area_registry/list")).get("result", [])
    for area in current:
        await send_message(ws, "config/area_registry/delete", area_id=area["area_id"])
    for name in AREAS:
        await send_message(ws, "config/area_registry/create", name=name)

async def create_floors(ws):
    current = (await send_message(ws, "config/floor_registry/list")).get("result", [])
    existing = {floor["name"] for floor in current}
    for name in FLOORS:
        if name not in existing:
            await send_message(ws, "config/floor_registry/create", name=name)

async def configure_devices(ws):
    areas = (await send_message(ws, "config/area_registry/list")).get("result", [])
    area_ids = {area["name"]: area["area_id"] for area in areas}
    floors = (await send_message(ws, "config/floor_registry/list")).get("result", [])
    floor_ids = {floor["name"]: floor["floor_id"] for floor in floors}
    area_floors = {area: floor for floor, floor_areas in FLOORS.items() for area in floor_areas}
    for area_name, floor_name in area_floors.items():
        await send_message(
            ws, "config/area_registry/update",
            area_id=area_ids[area_name], floor_id=floor_ids[floor_name],
        )
    devices = (await send_message(ws, "config/device_registry/list")).get("result", [])
    for device in devices:
        name = device.get("name_by_user") or device.get("name") or ""
        area = next((area for area in AREAS if name.startswith(area)), None)
        if area:
            await send_message(ws, "config/device_registry/update", device_id=device["id"], area_id=area_ids[area])
    device_names = {device["id"]: device.get("name_by_user") or device.get("name") or "" for device in devices}
    entities = (await send_message(ws, "config/entity_registry/list")).get("result", [])
    states = (await send_message(ws, "get_states")).get("result", [])
    friendly_names = {
        state["entity_id"]: state.get("attributes", {}).get("friendly_name")
        for state in states
    }
    for entity in entities:
        entity_id = entity["entity_id"]
        device_name = device_names.get(entity.get("device_id"), "")
        is_virtual = entity.get("platform") == "virtual"
        if is_virtual:
            name = friendly_names.get(entity_id) or entity.get("name") or entity.get("original_name") or entity_id
            room = next((room for room in AREAS if name.startswith(f"{room} ")), "")
            aliases = [name.removeprefix(f"{room} ") if room else name]
            await send_message(ws, "config/entity_registry/update", entity_id=entity_id, aliases=aliases)
        await send_message(
            ws, "homeassistant/expose_entity",
            assistants=["conversation"], entity_ids=[entity_id],
            should_expose=is_virtual and device_name not in ASSIST_DISABLED_DEVICES,
        )

    await send_message(
        ws, "homeassistant/expose_new_entities/set",
        assistant="conversation", expose_new=False,
    )

async def main():
    async with websockets.connect(HA_URL) as ws:
        await authenticate(ws)
        await create_areas(ws)
        await create_floors(ws)
        await configure_devices(ws)

asyncio.run(main())
