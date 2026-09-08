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
HA-RAGent adds a Home Assistant conversation agent that can use your own LLM. It finds the devices and tools relevant to each request, so the model does not need to receive your entire Home Assistant setup every time.

If you enable device control, the model can call Home Assistant tools and use their results to complete a request. You can also limit how much conversation history, device information, tools and memory are sent to the model. This is helpful for smaller or self-hosted models.

## Disclaimers
### Default System Prompt
Changes to the default system prompt apply only to newly created RAGent entries. Existing entries retain the prompt saved in their configuration. To use the latest default prompt with an existing entry, copy it into the entry’s **System Prompt** field and save the configuration or recreate the entry.

### Exposed Script Entities:
Scripts will be passed as tools and are excluded from device embeddings. Only scripts that are exposed to conversation/assist will be visible.

### OpenAI-Compatible Backends
OpenAI-compatible backends have currently been tested only with llamaccp. Compatibility with other providers is not guaranteed, so test the selected backend thoroughly before using it in production.

## Installation
### HACS (recommended)
If HACS is installed on your system use this link to directly go to the install page:

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=youngaileaderslinz&repository=HA-RAGent)

### Manual
To install this integration manually you have to download the repository [HA-RAGent.zip](https://github.com/youngaileaderslinz/HA-RAGent/archive/refs/heads/main.zip) and extract its contents to `config/custom_components/ha_ragent` directory.

## Configuration
### Using UI
[![Open your Home Assistant instance and start setting up a new integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ha_ragent)

From the Home Assistant front page go to `Configuration` and then select `Devices & Services` from the list.
Use the `Add Integration` button in the bottom right to add a new integration called `Home Assistant RAG Agent`.

### Add Service Entry:
**Select Backends:**
- `Vector database backend`
    - **FAISS (Local DB)** stores embeddings locally and is the default, simplest setup
    - **MongoDB** stores embeddings in an external MongoDB instance
    - **ChromaDB** stores embeddings in an external ChromaDB server
- `Embedding backend`
    - **Ollama** requires an external Ollama instance and an installed embedding model [[find embedding models]](https://ollama.com/search?c=embedding)
    - **OpenAI Compatible** works with APIs that expose an OpenAI-style embeddings endpoint
- `LLM backend`
    - **Ollama** requires an external Ollama instance and an installed chat model with tool support [[find tool-capable models]](https://ollama.com/search?c=tools)
    - **OpenAI Compatible** works with APIs that expose an OpenAI-style chat completions endpoint
- `Language`
    - **English** or **German** determines the language of the default system prompt for new AI RAGent entries

**Setup Connections:**

- `Vector Database Options`
    - **Database Username** optional database username, currently relevant for MongoDB
    - **Database Password** optional database password, currently relevant for MongoDB
    - **Vector DB Hostname** hostname or IP of the vector database server, used for MongoDB and ChromaDB
    - **Vector DB Port** port of the vector database server, used for MongoDB and ChromaDB
    - **Use HTTPS** enables SSL/TLS for the vector database connection when supported by the selected backend
    - **Database Name** can be left as is or changed (Must be unique for each instance when multiple instances of HA-RAGent are configured. The default name is already unique.)

- `Embedding Backend Options`
    - **Embedding Hostname** hostname or IP of the embedding API server
    - **Embedding Port** port of the embedding API server
    - **Use HTTPS** enables SSL/TLS for the embedding API connection
    - **Embedding API Key** optional bearer token for OpenAI-compatible embedding APIs

- `LLM Backend Options`
    - **LLM Hostname** hostname or IP of the LLM API server
    - **LLM Port** port of the LLM API server
    - **Use HTTPS** enables SSL/TLS for the LLM API connection
    - **LLM API Key** optional bearer token for OpenAI-compatible LLM APIs

### Add AI RAGent Entry:
**Pick one of the configured services**
- The name contains database, embedding and llm backend

**Pick Models**
- `Embedding Model`
    - **Only shows downloaded models** that can be used for embedding generation
    - Prefer a dedicated semantic-search embedding model.
- `LLM Model`
    - **Only shows downloaded models** that can be used as LLM model

**Fine Tuning**
- `LLM Home Assistant API`
    - **No Control** means the model is not allowed to control devices
    - **Assist** allows the model to control devices and exposes Home Assistant tools
- `System Prompt`
    - The Jinja template rendered and sent to the model as its system prompt
- `Retrieval Method`
    - **Automatic** (default) uses cached text and metadata matching, adding vector search when there is no unique, complete name or alias match
    - **Vector search** uses embedding similarity only
    - **Lexical search** uses names, aliases and metadata without semantic similarity
- `Allow Auto Embedding`
    - Automatically rebuilds embeddings for exposed entities and tools during startup and after configuration changes
- `Allow Follow-up Questions`
    - Lets the assistant ask a clarification question and keep the conversation open for the user's reply
- `Enable Model Thinking`
    - Controls whether the model may use its thinking mode. Leave disabled for faster responses when supported.
- `Number of Devices`
    - Controls how many relevant entity candidates are retrieved and added to the prompt
- `Number of Tools`
    - Controls how many relevant tools are retrieved and offered to the model (required HA-RAGent tools do not count to this limit)
- `Number of Long-Term Memories`
    - Controls how many semantically relevant, explicitly stored memories are added to each prompt. Set it to `0` to disable recall without deleting memories.
- `Maximum Memory Entries`
    - Controls how many long-term memories are retained for this RAGent.
- `Tools excluded from embedding`
    - Excludes selected tool names from the vector index. Names are matched exactly and are case-sensitive
- `Context Length` (Ollama only)
    - Sets Ollama's model context-window size
- `Maximum Tokens`
    - Sets the maximum number of tokens the model may generate in one response
- `Temperature`
    - Controls sampling randomness; lower values are more deterministic
- `Maximum Tool Call Iterations`
    - Limits the number of model/tool rounds per request. A single round may contain multiple tool calls
- `Conversation Memory Interactions`
    - Maximum number of previous user interactions retained for context and retrieval. Set to `0` to disable this limit.
- `Conversation Memory Duration`
    - Maximum age of conversation history in minutes. Set to `0` to disable this limit. If both history settings are `0`, no conversation history is used.

Both history limits apply when they are greater than `0`. For example, with `10` interactions and `60` minutes, only interactions from the last hour and within the last 10 turns are kept.

### Available Prompt Variables
The **System Prompt** is rendered as a Home Assistant Jinja template for every request. The following variables are passed to it:

- `device_list`
    - The retrieved device candidates whose entities currently exist in Home Assistant. Each device provides `id`, `friendly_name`, `area_name`, `floor_name`, `domain`, `device_class`, `device_labels`, `services`, `aliases`, `state`, `attributes` and `unit_of_measurement`.
- `memory_list`
    - The retrieved memory context candidates. Each memory provides `id`, `content` and `created_at`.
- `area_list`
    - A list of the distinct, non-empty area names found in `device_list`. It contains only areas associated with the retrieved candidates, not every area in Home Assistant.
- `area_name`
    - The area of the device through which the conversation was started or `None` when no area is available.
- `floor_name`
    - The floor of the device through which the conversation was started or `None` when no floor is available.
- `max_retries`
    - The configured maximum number of tool-call iterations.

## Custom Tools
When **Assist** is selected, HA-RAGent resolves it to its custom LLM API, which provides the following additional tools:

**HassSemanticSearch**
- Searches for devices and Home Assistant tools without changing device state.

**HassPlannedAction**
- Schedules a one-time Home Assistant action for execution after a specified delay.

**HassListPlannedActions**
- Lists all currently scheduled one-time Home Assistant actions.

**HassCancelAllPlannedActions**
- Cancels all currently scheduled one-time Home Assistant actions.

**HassRememberFact**
- Stores a fact as per-agent long-term memory when the user explicitly asks for it to be remembered.

**HassForgetFact**
- Deletes one recalled long-term memory by its exact memory ID.

## Services
HA-RAGent registers the following Home Assistant services for each conversation entity created by the integration:

- `ha_ragent.embed_subentry`
    - Rebuilds device and tool embeddings for the selected AI RAGent subentry.
- `ha_ragent.preload_models` (only works with Ollama as of now)
    - Preloads the embedding model and LLM for the selected AI RAGent subentry.
- `ha_ragent.unload_models` (only works with Ollama as of now)
    - Unloads the embedding model and LLM for the selected AI RAGent subentry to free resources.

## New Features, Help and Contribution
**Have an idea what is missing?** <br>
Open an issue [[open issue]](https://github.com/youngaileaderslinz/HA-RAGent/issues) or implement it yourself and create a pull request.

**Found a bug?** <br>
Open an issue [[open issue]](https://github.com/youngaileaderslinz/HA-RAGent/issues) and I’ll take a look or implement it yourself and create a pull request.

**How to start development?** <br>
Example of how to setup the development environment [[see more]](https://github.com/youngaileaderslinz/HA-RAGent/blob/main/dev/DEV_SETUP.md)
