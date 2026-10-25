<!-- markdownlint-disable first-line-heading -->
<!-- markdownlint-disable no-inline-html -->

<img src="https://raw.githubusercontent.com/youngaileaderslinz/HA-RAGent/main/custom_components/ha_ragent/brand/logo.png"
     alt="HA-RAGent logo"
     height="140px"
     align="right"
     style="float: right; margin: 10px 0 20px 20px;" />

[![GitHub Release](https://img.shields.io/github/release/youngaileaderslinz/HA-RAGent.svg?style=flat-square)](https://github.com/youngaileaderslinz/HA-RAGent/releases)
[![Build Status](https://img.shields.io/github/actions/workflow/status/youngaileaderslinz/HA-RAGent/validation.yaml?branch=main&style=flat-square)](https://github.com/youngaileaderslinz/HA-RAGent/actions/workflows/validation.yaml)
[![License](https://img.shields.io/github/license/youngaileaderslinz/HA-RAGent.svg?style=flat-square)](https://github.com/youngaileaderslinz/HA-RAGent/blob/main/LICENSE)
[![HACS](https://img.shields.io/badge/HACS-default-orange.svg?style=flat-square)](https://hacs.xyz)

# HA-RAGent (Home Assistant Retrieval-Augmented Generation Agent)

HA-RAGent is a Home Assistant conversation agent for Ollama and OpenAI-compatible APIs. It retrieves relevant devices, tools and saved facts for each request instead of sending your entire setup to the model.

Use it in Assist to ask about device states, control devices or schedule one-time actions. Limits for conversation history and retrieved context help keep prompts small for local models.

## Disclaimers

### Default prompts

New AI RAGent entries use the current translated **Rule Prompt** and **State Prompt** defaults. Existing entries keep their saved templates. To adopt updated defaults, edit both fields in the agent settings or recreate the AI RAGent entry.

### OpenAI-compatible backends

OpenAI-compatible backends have been tested only with **llama.cpp**. Compatibility with other providers is not guaranteed. Test your chosen backend before using it in production.

## Installation

### HACS (recommended)

Install HA-RAGent through HACS and restart Home Assistant.

[![Open HA-RAGent in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=youngaileaderslinz&repository=HA-RAGent)

### Manual

Download [HA-RAGent.zip](https://github.com/youngaileaderslinz/HA-RAGent/archive/refs/heads/main.zip). Copy the archive's `custom_components/ha_ragent` folder into your Home Assistant `config/custom_components` directory and restart Home Assistant.

## Configuration

Add **Home Assistant RAG Agent** from [Settings → Devices & services → Add integration](https://www.home-assistant.io/getting-started/integration/) or use the button below.

[![Set up HA-RAGent](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ha_ragent)

### Service setup

Choose the backends and enter their connection details. Embedding and chat models can use different backends.

| Setting | Description |
| --- | --- |
| Vector database backend | **FAISS** stores data locally and is the default. **MongoDB** and **ChromaDB** use a separate server. |
| Embedding backend | **Ollama** or an **OpenAI Compatible** embeddings API. |
| LLM backend | **Ollama** or an **OpenAI Compatible** chat completions API. |
| Language | English or German for the agent's default prompts and tool descriptions. |

Ollama needs an installed [embedding model](https://ollama.com/search?c=embedding) and a [chat model with tool support](https://ollama.com/search?c=tools) for device control.

#### Database connection

FAISS only needs a database name. Server connection fields appear for MongoDB and ChromaDB.

| Setting | Description |
| --- | --- |
| Database Username / Password | Optional MongoDB credentials. |
| Vector DB Hostname | Database server hostname or IP address. |
| Vector DB Port | Server port. Defaults: MongoDB `27017` or ChromaDB `8000`. |
| Use HTTPS | Enables TLS for MongoDB or HTTPS for ChromaDB. |
| Database Name | Use a unique name for each service entry. The generated default is unique. |

#### Model API connections

Configure the embedding and LLM connections separately.

| Setting | Description |
| --- | --- |
| Embedding Hostname / LLM Hostname | Server hostname or IP address without a URL scheme or API path. |
| Embedding Port / LLM Port | Server port. Defaults: Ollama `11434` or OpenAI Compatible `443`. |
| Use HTTPS | Enables HTTPS. Defaults to off for Ollama and on for OpenAI Compatible. |
| Embedding API Key / LLM API Key | Bearer token for OpenAI-compatible APIs that require authentication. |

The OpenAI-compatible backends use `/v1` endpoints and require model listing. The chat backend also sends tool parsing and thinking options that the server must accept. Compatibility depends on the server.

### Add an AI RAGent

Add an **AI RAGent** to the configured service. Each agent has its own models, prompts, history and saved facts.

| Setting | Description |
| --- | --- |
| Embedding Model | Choose a model for semantic search. |
| LLM Model | Choose a chat model with tool support for device control. |

Ollama lists models with the required capability. OpenAI-compatible APIs list all models reported by the provider so you need to choose a suitable one.

#### Agent settings

| Setting | Description |
| --- | --- |
| LLM Home Assistant API | **Assist** enables Home Assistant tools. **No Control** disables all tool calls but keeps retrieved context available. |
| Prompt layout | **Split** (default) sends rules before history and current state before the user message. **Combined** sends both before history. |
| Rule Prompt | Stable instructions for the assistant. |
| State Prompt | Current device and memory context from a Jinja template. |
| Allow Auto Embedding | Rebuilds device and tool indexes at startup or integration reload. |
| Allow Follow-up Questions | Keeps the conversation open when a reply ends with `?` or `？`. |
| Enable Model Thinking | Requests the model's thinking mode. Leave off for faster replies when supported. |
| Retrieval Method | **Automatic** (default) combines semantic search with names, aliases and metadata. **Vector search** uses embedding similarity. **Lexical search** uses text matching. |
| Tools excluded from embedding | Excludes exact tool names from indexing. Names are case-sensitive. |
| Minimum / Maximum Devices to Expose | Device candidates per retrieval. Defaults: `2` / `4`. |
| Minimum / Maximum Searched Tools to Expose | Optional tool candidates per search. Defaults: `2` / `4`. |
| Minimum / Maximum Long-Term Memories | Recalled facts per request. Defaults: `0` / `4`. Maximum `0` disables recall without deleting facts. |
| Maximum Memory Entries | Stored facts per agent. Defaults to `100`. Removes least-recalled facts first, then oldest on ties. |
| Context Length | Ollama's context window in tokens. Includes rules, history, state, tools and generated output. |
| Maximum Tokens | Maximum tokens generated per model response. |
| Temperature | Lower values make replies more predictable. |
| Maximum Tool Call Iterations | Maximum model rounds per request. One round can contain multiple tool calls. |
| Conversation Memory Interactions | Maximum previous user interactions kept. `0` disables this limit. |
| Conversation Memory Duration (minutes) | Maximum history age. `0` disables this limit. |

Device and tool retrieval use the selected method. Memory recall always uses vector search.

Minimum counts apply when enough candidates are available. Maximums cap each retrieval. Explicit tool searches can add tools during a request. The required semantic-search tool does not count toward tool limits.

Both history limits apply when set. With `10` interactions and `60` minutes, history includes at most 10 interactions from the past hour. Setting both limits to `0` disables conversation history.

Memory storage limits take effect on the next write. At capacity, a new fact can be removed immediately even when the remember tool reports success.

Choose the agent in your [Assist configuration](https://www.home-assistant.io/voice_control/voice_remote_local_assistant/) and expose the devices it should access.

### Prompts

Older entries migrated from a single prompt keep the **Combined** layout.

The default rules favor exact entity IDs for individual devices and area targeting for whole-area requests. They allow multiple tool calls and ask for essential clarification. Replies stay brief and naturally varied.

Both prompts use Home Assistant Jinja templates. Keep stable instructions in the Rule Prompt and changing context in the State Prompt. Tool definitions are sent separately.

The default State Prompt uses compact JSON records. Empty fields are omitted. Extra device attributes and memory timestamps are not included by default.

Within the device or memory loop, pass a dictionary to `to_prompt_dict()`. Keys are class field names and values are the names used in the prompt. It returns compact JSON text directly:

```jinja
{{ device.to_prompt_dict({'id': 'name', 'friendly_name': 'friendly_name', 'state': 'state', 'attributes': 'attributes'}) }}
{{ memory.to_prompt_dict({'id': 'memory_id', 'content': 'content', 'created_at': 'created_at'}) }}
```

Only mapped fields are included. Empty fields and nested null values are omitted. The default mapping uses `name` for device IDs and `memory_id` for memory IDs.

#### Template variables

| Variable | Description |
| --- | --- |
| `device_list` | Retrieved devices whose entities still exist. Fields: `id`, `friendly_name`, `aliases`, `domain`, `device_class`, `device_labels`, `services`, `area_name`, `floor_name`, `area_aliases`, `floor_aliases`, `state`, `attributes` and `unit_of_measurement`. |
| `memory_list` | Retrieved saved facts. Fields: `id`, `content`, `created_at` and `retrieval_count`. |
| `area_list` | Distinct area names from retrieved devices. Does not include every Home Assistant area. |
| `area_name` | Area of the device that started the conversation or `None`. |
| `floor_name` | Floor of that device or `None`. |
| `max_retries` | Configured maximum tool-call iterations. |

### Response time

Replies stream to Home Assistant as text arrives. Clients with streaming support can display or speak them before generation finishes.

For smaller local models, keep prompts short and limit retrieved context. **Split** layout keeps stable rules before history to help prompt-cache reuse. Changing the tool set can still require prompt reprocessing. Total response time also includes tool calls and the final model reply.

## Custom tools

Selecting **Assist** adds these tools to the Home Assistant API. Apart from semantic search, tools are exposed when retrieved for a request.

| Tool | Description |
| --- | --- |
| `HassSemanticSearch` | Finds device candidates or tools without changing device state. |
| `HassScheduleAction` | Schedules a one-time action `1`–`1440` minutes ahead. |
| `HassListScheduledActions` | Lists this agent's pending one-time actions. |
| `HassCancelScheduledActions` | Cancels all this agent's pending one-time actions. |
| `HassRememberFact` | Saves a fact when the user explicitly asks. Maximum `1000` characters. |
| `HassForgetFact` | Deletes one recalled fact using its exact memory ID. |

Semantic search accepts up to four self-contained `search_queries`. Optional `capabilities` must match the queries by position. Set `scope` to `devices`, `tools` or `devices_and_tools`. Explicit tool searches can add callable tools to the current request; omitting `scope` searches both without adding tools.

Scheduled actions are held in memory. A Home Assistant restart or integration reload cancels them.

Repeating a saved fact with different casing or whitespace replaces the existing entry.

## Services

Target one or more conversation entities created by HA-RAGent.

| Service | Description |
| --- | --- |
| `ha_ragent.embed_subentry` | Rebuilds device and tool indexes for the selected agents. |
| `ha_ragent.preload_models` | Preloads the selected agents' models. Ollama only. |
| `ha_ragent.unload_models` | Unloads the selected agents' models to free resources. Ollama only. |

## Scripts

Scripts exposed to Assist are available as tools rather than device candidates. Scripts that act on entities can return affected entity IDs in a `success` list:

```yaml
- variables:
    response:
      success:
        - '{{ target_media_player }}'
- stop: Playback command sent
  response_variable: response
```

## Help and development

Report bugs or suggest features in [GitHub issues](https://github.com/youngaileaderslinz/HA-RAGent/issues). Contributions are welcome through pull requests.

See the [development setup](dev/DEV_SETUP.md) to run the project locally.
