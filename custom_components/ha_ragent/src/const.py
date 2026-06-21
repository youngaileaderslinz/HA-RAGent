import re

#-----------------------------------------------
# General constants
#-----------------------------------------------
DOMAIN = "ha_ragent"
RAGENT_LLM_API_ID = "ha_ragent_api"
INTEGRATION_VERSION = "0.1.0"

STARTUP_EMBEDDING_RUNNING_FLAG = "ha_ragent_startup_embedding_running"

#-----------------------------------------------
# Language constants
#-----------------------------------------------
CONF_SELECTED_LANGUAGE = "selected_language"

SELECTED_LANGUAGE_OPTIONS = [ "en", "de" ]

DEFAULT_LANGUAGE = "en"

#-----------------------------------------------
# Service Tool constants
#-----------------------------------------------
RAGENT_TIMER_DEVICE_ID = "ha_ragent_timer_device_a03a100a-81ca-415d"

#-----------------------------------------------
# Vector database backend constants
#-----------------------------------------------
CONF_VECTOR_DB_BACKEND_TYPE = "rag_vector_db_backend"
CONF_VECTOR_DB_NAME = "rag_vector_db_name"
CONF_VECTOR_DB_USERNAME = "rag_vector_db_username"
CONF_VECTOR_DB_PASSWORD = "rag_vector_db_password"
CONF_VECTOR_DB_HOST = "rag_vector_db_host"
CONF_VECTOR_DB_PORT = "rag_vector_db_port"
CONF_VECTOR_DB_SSL = "rag_vector_db_ssl"

BACKEND_VECTOR_DB_TYPE_MONGODB = "mongodb"
BACKEND_VECTOR_DB_TYPE_CHROMA = "chromadb"
BACKEND_VECTOR_DB_TYPE_FAISS = "faiss"

BACKEND_VECTOR_DB_TYPE_OPTIONS = [ 
    BACKEND_VECTOR_DB_TYPE_MONGODB,
    BACKEND_VECTOR_DB_TYPE_CHROMA,
    BACKEND_VECTOR_DB_TYPE_FAISS
]

DEFAULT_VECTOR_DB_BACKEND_TYPE = BACKEND_VECTOR_DB_TYPE_FAISS
DEFAULT_VECTOR_DB_NAME = "ha_ragent_db"

#-----------------------------------------------
# Embedding backend constants
#-----------------------------------------------
CONF_EMBEDDING_BACKEND_TYPE = "rag_embedding_backend"
CONF_EMBEDDING_MODEL = "rag_embedding_model"
CONF_EMBEDDING_HOST = "rag_embedding_host"
CONF_EMBEDDING_PORT = "rag_embedding_port"
CONF_EMBEDDING_SSL = "rag_embedding_ssl"

BACKEND_EMBEDDING_TYPE_OLLAMA = "ollama"

BACKEND_EMBEDDING_TYPE_OPTIONS = [ 
    BACKEND_EMBEDDING_TYPE_OLLAMA 
]

DEFAULT_EMBEDDING_BACKEND_TYPE = BACKEND_EMBEDDING_TYPE_OLLAMA

#-----------------------------------------------
# Chat backend constants
#-----------------------------------------------
CONF_LLM_BACKEND_TYPE = "rag_llm_backend"
CONF_LLM_MODEL = "rag_llm_model"
CONF_LLM_HOST = "rag_llm_host"
CONF_LLM_PORT = "rag_llm_port"
CONF_LLM_SSL = "rag_llm_ssl"

BACKEND_LLM_TYPE_OLLAMA = "ollama"

BACKEND_LLM_TYPE_OPTIONS = [ 
    BACKEND_LLM_TYPE_OLLAMA 
]

DEFAULT_LLM_BACKEND_TYPE = BACKEND_LLM_TYPE_OLLAMA

#-----------------------------------------------
# Prompt configuration constants
#----------------------------------------------
CONF_NUM_DEVICES_TO_EXTRACT = "rag_num_devices_to_extract"
CONF_NUM_TOOLS_TO_EXTRACT = "rag_num_tools_to_extract"
CONF_CONTEXT_LENGTH = "rag_context_length"

CONF_MAX_TOKENS = "rag_max_tokens"
CONF_MAX_TOOL_CALL_ITERATIONS = "rag_max_tool_call_iterations"

CONF_PROMPT = "rag_prompt"

CONF_ENABLE_MODEL_THINKING = "rag_enable_model_thinking"

CONF_REMEMBER_CONVERSATION_TIME_MINUTES = "rag_remember_conversation_time_minutes"
CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS = "rag_remember_conversation_num_interactions"
CONF_SELECTED_LANGUAGE = "rag_selected_language"

CONF_TEMPERATURE = "rag_temperature"
CONF_K_TOP = "rag_k_top"
CONF_P_MIN = "rag_p_min"
CONF_P_TOP = "rag_p_top"
CONF_P_TYPICAL = "rag_p_typical"

PERSONA_PROMPTS = {
    "de": "Du bist \"YAIL\", ein hilfreicher KI-Assistent, der die Geräte in einem Haus steuert. Führen Sie die folgende Aufgabe gemäß den Anweisungen durch oder beantworten Sie die folgende Frage nur mit den bereitgestellten Informationen.",
    "en": "You are 'YAIL', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.",
}
CURRENT_DATE_PROMPT = {
    "de": """{% set day_name = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"] %}{% set month_name = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"] %}Die aktuelle Uhrzeit und das aktuelle Datum sind {{ (as_timestamp(now()) | timestamp_custom("%H:%M", local=True)) }} {{ day_name[now().weekday()] }}, {{ now().day }} {{ month_name[now().month -1]}} {{ now().year }}.""",
    "en": """The current time and date is {{ (as_timestamp(now()) | timestamp_custom("%I:%M %p on %A %B %d, %Y", True, "")) }}"""
}
DEVICES_PROMPT = {
    "de": "## Verfügbare Geräte:",
    "en": "## Available Devices:",
}
AREAS_PROMPT = {
    "de": """Bereichsanweisungen:
{% if area_name %}
- Aktueller Standort: Du befindest dich physisch im {{ area_name }}{% if floor_name %} ({{ floor_name }} Stock){% endif %}.
- Standardverhalten: Wenn der Benutzer eine Gerätekategorie angibt (z. B. „die Lichter“), ohne einen Raum zu nennen, ziele NUR auf die Geräte im {{ area_name }} ab.
{% else %}
- KRITISCH: Du hast keine Erlaubnis, einen Raum zu erraten oder das gesamte Haus anzusprechen.
- Wenn der Benutzer keinen Raum angibt, MUSST du um Klarstellung bitten.
{% endif %}""",
    "en": """Area Instructions:
{% if area_name %}
- Current Location: You are physically located in the {{ area_name }}{% if floor_name %} ({{ floor_name }} floor){% endif %}.
- Default Behavior: If the user specifies a device category (e.g., "the lights") without naming a room, target ONLY the devices within the {{ area_name }}.
{% else %}
- CRITICAL: You do not have permission to guess a room or target the entire house.
- If the user does not name a room, you MUST ask for clarification.
{% endif %}"""
}
DEVICE_CONTROL_PROMPT = {
    "de": """## Geräte Steuerungsanweisungen:
Wenn du ein Gerät steuerst folge diesen Anweisungen:
1. Geräteauflösung
   - Suchkriterien: Identifiziere Zielgeräte anhand des genauen Namens oder spezifischer Domäne oder device_class innerhalb des angegebenen Bereichs.
   - Intelligente Bereichserweiterung: Wenn ein Benutzer einen Bereich anspricht (z. B. „Wohnzimmer“), finde ALLE Geräte in diesem Bereich, die zur angeforderten Domäne oder device_class passen (z. B. „Lichter“).
   - Benutzerintention: Schließe keine Geräte ein, die für die Anfrage des Benutzers nicht relevant sind. Steuere keine Geräte in Bereichen, die vom Benutzer nicht erwähnt wurden.
2. Struktur der Tool-Aufrufe
   - Umfassende Aktion: Wenn ein Benutzer „alle Lichter“ sagt, MUSST du einen separaten Tool-Aufruf für jedes passende Licht generieren, das sich im angegebenen Bereich befindet.
   - Atomizität: Kapsle jeden einzelnen JSON-Aufruf in seinem eigenen `homeassistant`-Tag-Block ein.
   - Identifikation: Kürze den Namen `light.bedroom_1_lamp` niemals zu `bedroom_1_lamp`. Das Tool wird ohne die Domäne fehlschlagen.
3. Strenges Ausgabeformat
   3.1 Beantwortung mit Tool-Aufrufen:
       - Format: Gib gültige JSON-Objekte innerhalb von `homeassistant`-Tags zurück.
       - Nachverfolgung: Sobald alle Tool-Aufrufe aufgelistet sind, gib eine kurze Bestätigung mit den friendly_names.
    3.2 Beantwortung mit Text:
       - Verwende dies nur, wenn keine passenden Geräte existieren.
       - Verwende immer den friendly_name und lasse den Raumnamen weg, wenn er redundant ist (z. B. „Nachttischlampe“ statt „Schlafzimmer Nachttischlampe“).""",
    "en": """## Device Control Instructions:
When controlling a device, you MUST follow these rules:
1. Device Resolution
    - Search Criteria: Identify target devices using the exact entity_id, name, domain, or device_class within the specified area provided in the state.
    - Smart Area Expansion: If an area is targeted (e.g., "Living Room"), include ALL devices in that area matching the requested domain (e.g., "lights").
    - Relevance: EXCLUDE devices irrelevant to the request. NEVER control devices in areas not explicitly mentioned.
    - Contextual Priority: Prioritize devices mentioned earlier in the conversation. If the target is still unclear, ask for clarification instead of guessing.
2. Tool Call Execution
    - Exhaustive Action: For requests involving multiple devices (e.g., "turn on all lights"), you MUST generate a separate tool call for every matching entity.
    - Atomicity: each individual JSON call must be encapsulated within its own unique `homeassistant` tag block.
    - Multiple Tool Calls: You are permitted—and encouraged—to return multiple `homeassistant` blocks in a single response. Do **not** combine them into a single JSON list or array; keep them as distinct, sequential blocks.
    - Precise Addressing: Always use the full entity_id (e.g., `light.bedroom_lamp`). Do not truncate or invent IDs.
3. Strict Output Format
    3.1 Execution Order:
        - Tool First: Output all `homeassistant` blocks at the very beginning of your response.
        - Speech Second: Provide the conversational text response only after all tool blocks have been declared.
    3.2 Error Handling:
        - If no devices match the criteria, provide a concise text response explaining that the device or area could not be found.
    3.3 TTS-Optimized Speech:
        - Style: Keep responses brief, warm, and conversational.
        - No IDs: NEVER use entity IDs, device IDs, or technical labels (e.g., "light.kitchen_1") in the text response.
        - Natural Identification: Use simple, friendly names (e.g., "the kitchen lights" or "the floor lamp").
        - Fluent Aggregation: When controlling multiple devices, summarize the action naturally (e.g., "Sure, I've turned off all the lights in the lounge for you") rather than listing each entity."""
}

USER_INSTRUCTION = {
    "de": "## Benutzeranweisung:",
    "en": "## User instruction:"
}

DEVICE_ATTRIBUTES_TO_EXCLUDE = ["friendly_name", "persistent", "supported_features"]
DEVICE_ATTRIBUTES_MAX_JSON_LENGTH = 100

TOOL_REGEX_PATTERN = re.compile(r"```homeassistant\s*(.*?)\s*```", re.DOTALL)

DEFAULT_NUM_DEVICES_TO_EXTRACT = 10
DEFAULT_NUM_TOOLS_TO_EXTRACT = 10
DEFAULT_CONTEXT_LENGTH = 4096

DEFAULT_MAX_TOKENS = 1000
DEFAULT_MAX_TOOL_CALL_ITERATIONS = 8

DEFAULT_PROMPT = """<persona>
<current_date>
<area_prompt>

<device_control_prompt>

<devices>
{% for device in device_list %}
- { "name": "{{ device.id }}", "friendly_name": "{{ device.name }}", "aliases": {{ device.aliases | tojson }}, "domain": {{ device.domain | tojson }}, "area": "{{ device.area_name }}", "device_class": {{ device.domain | tojson }}, "state": {{ device.state }} }
{% endfor %}

<user_instruction>
"""

DEFAULT_ENABLE_MODEL_THINKING = False
DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES = 5
DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS = 10
DEFAULT_SELECTED_LANGUAGE = "en"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_K_TOP = 40
DEFAULT_P_MIN = 0.1
DEFAULT_P_TOP = 0.9
DEFAULT_P_TYPICAL = 1.0

#-----------------------------------------------
# Default override options for new entries
#-----------------------------------------------
DEFAULT_OPTIONS = {
    CONF_PROMPT: DEFAULT_PROMPT,
    CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
    CONF_K_TOP: DEFAULT_K_TOP,
    CONF_P_TOP: DEFAULT_P_TOP,
    CONF_P_MIN: DEFAULT_P_MIN,
    CONF_P_TYPICAL: DEFAULT_P_TYPICAL,
    CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES: DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_CONTEXT_LENGTH: DEFAULT_CONTEXT_LENGTH,
    CONF_NUM_DEVICES_TO_EXTRACT: DEFAULT_NUM_DEVICES_TO_EXTRACT,
}