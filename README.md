<img src="https://raw.github.com/Young-ai-leaders/HA-RAGent/main/custom_components/ha_ragent/brand/logo.png" alt="HA-RAGent logo" title="HA-RAGent" align="right" height="80" />

# HA-RAGent (Home Assistant Retrieval‑Augmented‑Generation Agent)
HA‑RAGent is a custom component that wraps an LLM and a vector database to let you talk to your smart home. Instead of hard‑coding every possible command, the agent embeds your question, looks up the most relevant devices, and then either replies in natural language or emits “tool calls” that turn into real service calls inside Home Assistant.

This is particularly useful on self‑hosted installs, where you deliberately keep the model’s prompt window small to keep responses snappy. As soon as you move past a dozen or so entities, a plain conversation agent has to dump the entire device list into every prompt. A large or growing device set quickly blows out the context window and drags performance. Additionally, smaller models struggle even more, getting confused by the noise and sometimes emitting seemingly random tool calls.

## Installation
### HACS (recommended)

If HACS is installed on your system use this link to directly go to the install page:

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=youngaileaderslinz&repository=HA-RAGent)

### Manual

To install this integration manually you have to download the repository [HA-RAGent.zip](https://github.com/Young-ai-leaders/HA-RAGent/archive/refs/heads/main.zip) and extract its contents to `config/custom_components/ha_ragent` directory.

## Configuration

### Using UI

[![Open your Home Assistant instance and start setting up a new integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ha_ragent)

From the Home Assistant front page go to `Configuration` and then select `Devices & Services` from the list.
Use the `Add Integration` button in the bottom right to add a new integration called `Home Assistant RAG Agent`.

### Add Service Entry:
**Select Backends:**
- `Vector database backend`
    - **FAISS (Local DB)** is the default and simplest to use, there is no additional configuration required
    - **MongoDB** requires the setup of an external MongoDB Atlas instance
    - **ChromaDB** requires the setup of an external ChromaDB instace
- `Embedding backend`
    - **Ollama** requires the setup of external ollama instance and download of an embedding model ([find embedding model](https://ollama.com/search?c=embedding))
- `LLM backend`
    - **Ollama** requires the setup of external ollama instance and download of an embedding model ([find LLM model](https://ollama.com/search?c=tools))
- `Language`
    - **English** used in order to setup the default prompt
    - **German** used in order to setup the default prompt

**Setup Connections:**

`Vector Database Options`
- **Database Username** the username which is used for the database connection (only visible for certain backends)
- **Database Password** the password which is used for the database connection (only visible for certain backends)
- **Vector DB Hostname** the hostname which is used for the database connection (only visible for certain backends)
- **Vector DB Port** the port which is used for the database connection (only visible for certain backends)
- **Use HTTPS** whether to enable SSL for the database connection (only visible for certain backends)
- **Database Name** can be left as is or changed (Must be unique for each instance when multiple instances of HA-RAGent are configured. The default name is already unique.)

`Embedding Backend Options`
- **Ebedding Hostname** the hostname which is used for the embedding backend connection
- **Ebedding Port** the port which is used for the embedding backend connection
- **Use HTTPS** whether to enable SSL for the embedding backend connection

`LLM Backend Options`
- **LLM Hostname** the hostname which is used for the LLM backend connection
- **LLM Port** the port which is used for the LLM backend connection
- **Use HTTPS** whether to enable SSL for the LLM backend connection

### Add AI RAGent Entry:
**Pick one of the configured services**
- The name contains database, embedding and llm backend

**Pick Models**
- `Embedding Model`
    - **Only shows downloaded models** that can be used for ebedding generation
- `LLM Model`
    - **Only shows downloaded models** that can be used as LLM model

**Fine Tuning**
- `LLM Home Assistant API`
    - **No Control** means the model is not allowed to control devices
    - **Assist** allows the model to control devices
- `System Prompt`
    - The prompt that is sent to the model.
- `Enable Model Thinking`
    - Controls wheter model is allowed to think (when speed is of the essence keep the default)
- `Number of Devices`
    - Controls how many devices are retrieved and sent to the LLM
- `Number of Tools`
    - Controls how many tools are retrieved and sent to the LLM
- `Context Lenght`
    - Controls the context lenght of the LLM
- `Maximum Tokens`
    - Controls the maximum number of tokens the LLM is allowed to generate
- `Temperature`
    - Controls how much the LLM halucinates.
- `Maximum Tool Call Iterations`
    - Controls how often the LLM is allowed to perform tool calls per request (**Important note:** one response can call multiple tools the LLM can respond up to 8 times per default)
- `Conversation Memory Interactions`
    - Controls how many past user interactions are kept in memory for context.
- `Conversation Memory Duration`
    - Controls how long the assistant will retain conversation history in minutes.

## Help and Contribution
**Found a bug?** <br>
Open an issue and I’ll take a look.

**Want to add a feature or otherwise improve the code?** <br>
Send a pull request (or drop a quick issue first so we can chat about it).

**How to start?** <br>
Setup development environment ([see more](https://github.com/Young-ai-leaders/HA-RAGent/blob/main/dev/DEV_SETUP.md))