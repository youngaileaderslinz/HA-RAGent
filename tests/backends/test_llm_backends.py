from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Awaitable

import pytest
import aiohttp
from homeassistant.core import HomeAssistant

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import tests

from custom_components.ha_ragent.src.const import (
    CONF_LLM_HOST,
    CONF_LLM_API_KEY,
    CONF_LLM_MODEL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    RAGENT_PREFIXED_REQUIRED_TOOL_NAMES,
    CONF_CONTEXT_LENGTH,
    CONF_ENABLE_MODEL_THINKING,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
)
from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.chat.chat_message import (
    ChatFunction,
    ChatMessage,
    ChatToolCall,
)
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend


from custom_components.ha_ragent.src.backends.llm.openai_backend import OpenAiLlmBackend
from custom_components.ha_ragent.src.backends.llm import (
    ollama_backend as ollama_backend_module,
)
from custom_components.ha_ragent.src.backends.llm.ollama_backend import OllamaLlmBackend


MOCK_LLM_DEFAULT_OPTIONS = {
    CONF_LLM_HOST: "llamacpp",
    CONF_LLM_SSL: False,
    CONF_LLM_API_KEY: None,
}
MOCK_OPENAI_CONNECTION_USER_INPUT = {**MOCK_LLM_DEFAULT_OPTIONS, CONF_LLM_PORT: 8080}
MOCK_OPENAI_CONNECTION_USER_INPUT_INVALID = {
    **MOCK_OPENAI_CONNECTION_USER_INPUT,
    CONF_LLM_HOST: "invalid_host",
}
MOCK_OLLAMA_CONNECTION_USER_INPUT = {**MOCK_LLM_DEFAULT_OPTIONS, CONF_LLM_PORT: 11434}
MOCK_OLLAMA_CONNECTION_USER_INPUT[CONF_LLM_HOST] = "ollama"
MOCK_OLLAMA_CONNECTION_USER_INPUT_INVALID = {
    **MOCK_OLLAMA_CONNECTION_USER_INPUT,
    CONF_LLM_HOST: "invalid_host",
}
MOCK_OPENAI_CHAT_CONFIG = {
    **MOCK_OPENAI_CONNECTION_USER_INPUT,
    CONF_LLM_MODEL: "Qwen/Qwen3-1.7B-GGUF:Q8_0",
    CONF_TEMPERATURE: 0.2,
    CONF_MAX_TOKENS: 128,
    CONF_ENABLE_MODEL_THINKING: False,
}
MOCK_OPENAI_CHAT_CONFIG_INVALID = {**MOCK_OPENAI_CHAT_CONFIG, CONF_LLM_MODEL: "invalid_model"}
MOCK_OLLAMA_CHAT_CONFIG = {
    **MOCK_OLLAMA_CONNECTION_USER_INPUT,
    CONF_LLM_MODEL: "qwen3:1.7b",
    CONF_TEMPERATURE: 0.2,
    CONF_MAX_TOKENS: 128,
    CONF_ENABLE_MODEL_THINKING: False,
    CONF_CONTEXT_LENGTH: 4096,
}
MOCK_OLLAMA_CHAT_CONFIG_INVALID = {**MOCK_OLLAMA_CHAT_CONFIG, CONF_LLM_MODEL: "invalid_model"}
MOCK_LLM_TOOLS = [
    LlmTool(name="test_tool", description="A tool used by backend tests.", parameters={}),
    LlmTool(name="another_tool", description="Another tool used by backend tests.", parameters={}),
]
MOCK_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
]
MOCK_TOOL_HISTORY: list[ChatMessage] = [
    ChatMessage(role="system", content="Follow instructions."),
    ChatMessage(role="user", content="Turn on the desk light."),
    ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            ChatToolCall(
                id="call_1",
                type="function",
                function=ChatFunction(name="HassTurnOn", arguments={"name": "Desk light"}),
            )
        ],
    ),
    ChatMessage(
        role="tool",
        content='{"success": ["light.desk"]}',
        tool_call_id="call_1",
        tool_name="HassTurnOn",
    ),
]
MOCK_MESSAGE_CONTEXT_OVERFLOW = [
    ChatMessage(
        role="user",
        content="".join(f"Message Overflow INDEX: {index}" for index in range(10000)),
    )
]

@dataclass(frozen=True)
class BackendCase:
    """Configuration for one LLM backend test case."""
    backend_class: type[ALlmBaseBackend]
    user_input: dict[str, Any]
    user_input_invalid: dict[str, Any]
    chat_config: dict[str, Any]
    chat_config_invalid: dict[str, Any]

LLM_BACKENDS = [
    BackendCase(
        OpenAiLlmBackend,
        MOCK_OPENAI_CONNECTION_USER_INPUT,
        MOCK_OPENAI_CONNECTION_USER_INPUT_INVALID,
        MOCK_OPENAI_CHAT_CONFIG,
        MOCK_OPENAI_CHAT_CONFIG_INVALID,
    ),
    BackendCase(
        OllamaLlmBackend,
        MOCK_OLLAMA_CONNECTION_USER_INPUT,
        MOCK_OLLAMA_CONNECTION_USER_INPUT_INVALID,
        MOCK_OLLAMA_CHAT_CONFIG,
        MOCK_OLLAMA_CHAT_CONFIG_INVALID,
    ),
]

@pytest.fixture(params=LLM_BACKENDS, ids=lambda case: case.backend_class.__name__)
def backend_case(request: pytest.FixtureRequest) -> BackendCase:
    """Provide every supported backend from the central backend list."""
    return request.param

@pytest.fixture
def hass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide native Home Assistant with an isolated Ollama HTTP session."""
    loop = asyncio.new_event_loop()

    async def create() -> tuple[HomeAssistant, aiohttp.ClientSession]:
        instance = HomeAssistant(str(tmp_path))
        await instance.async_start()
        return instance, aiohttp.ClientSession()

    instance, session = loop.run_until_complete(create())
    monkeypatch.setattr(
        ollama_backend_module,
        "async_get_clientsession",
        lambda _hass: session,
    )
    try:
        yield instance
    finally:
        loop.run_until_complete(session.close())
        loop.run_until_complete(instance.async_stop(force=True))
        loop.close()

def _run_async_test(hass: HomeAssistant, test: Awaitable[None]) -> None:
    """Run an async test on the Home Assistant instance's event loop."""
    hass.loop.run_until_complete(test)

@pytest.fixture(autouse=True)
def suppress_logging() -> Any:
    """Suppress backend log output for this test module."""
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)

async def _async_test_connection_success(backend: ALlmBaseBackend, connection_input: dict[str, Any]) -> None:
    """Test that the backend can validate a connection with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(backend._hass, connection_input)
    assert validation_error is None, validation_error

async def _async_test_connection_failure(backend: ALlmBaseBackend, connection_input_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle connection failures with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(backend._hass, connection_input_invalid)
    assert validation_error is not None, "Expected a validation error for invalid input"

async def _async_test_validate_connection(
        backend_valid: ALlmBaseBackend, 
        backend_invalid: ALlmBaseBackend,
        user_input_valid: dict[str, Any],
        user_input_invalid: dict[str, Any]
        ) -> None:
    """Test that the backend can validate a connection with a mocked Home Assistant."""
    await _async_test_connection_success(backend_valid, user_input_valid)
    await _async_test_connection_failure(backend_invalid, user_input_invalid)

async def _async_test_get_available_models_success(backend: ALlmBaseBackend) -> None:
    """Test that the backend can retrieve available models with a mocked Home Assistant."""
    models = await backend.async_get_available_models()
    assert isinstance(models, list), "Available models should be a list"
    assert len(models) > 0, "Expected at least one available model"

async def _async_test_get_available_models_failure(backend: ALlmBaseBackend) -> None:
    """Test that the backend can handle available models retrieval failures with a mocked Home Assistant."""
    with pytest.raises(Exception):
        await backend.async_get_available_models()

async def _async_test_get_available_models(backend_valid: ALlmBaseBackend, backend_invalid: ALlmBaseBackend) -> None:
    """Test that the backend can retrieve available models with a mocked Home Assistant."""
    await _async_test_get_available_models_success(backend_valid)
    await _async_test_get_available_models_failure(backend_invalid)

async def _async_test_get_model_info_success(backend: ALlmBaseBackend, chat_config: dict[str, Any]) -> None:
    """Test that the backend can retrieve model info with a mocked Home Assistant."""
    model_info = await backend.async_get_model_info(chat_config[CONF_LLM_MODEL])
    assert isinstance(model_info, ModelInfo), "Model info should be a ModelInfo instance"
    assert model_info.name == chat_config[CONF_LLM_MODEL], "Model name should match the requested model"
    assert model_info.context_size is None or model_info.context_size > 0, "Context size should be None or greater than 0"
    assert model_info.is_embedding_model is None or model_info.is_embedding_model in [True, False], "is_embedding_model should be None or a boolean"
    assert model_info.is_tool_model is None or model_info.is_tool_model in [True, False], "is_tool_model should be None or a boolean"

async def _async_test_get_model_info_failure(backend: ALlmBaseBackend, chat_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle model info retrieval failures with a mocked Home Assistant."""
    with pytest.raises(Exception):
        await backend.async_get_model_info(chat_config_invalid[CONF_LLM_MODEL])

async def _async_test_get_model_info(
        backend_valid: ALlmBaseBackend, 
        backend_invalid: ALlmBaseBackend, 
        chat_config: dict[str, Any], 
        chat_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can retrieve model info with a mocked Home Assistant."""
    await _async_test_get_model_info_success(backend_valid, chat_config)
    await _async_test_get_model_info_failure(backend_invalid, chat_config_invalid)

async def _async_test_preload_and_unload_model(backend: ALlmBaseBackend, chat_config: dict[str, Any]) -> None:
    """Test that the backend can preload and unload a model with a mocked Home Assistant."""
    await backend.async_preload_model(chat_config)
    await backend.async_unload_model(chat_config)

async def _async_test_send_chat_request_success(backend: ALlmBaseBackend, chat_config: dict[str, Any]) -> None:
    """Test that the backend can send a chat request with a mocked Home Assistant."""
    response = [
        item
        async for item in backend.async_send_chat_request(
            chat_config,
            MOCK_MESSAGES,
            MOCK_LLM_TOOLS,
        )
    ]

    assert isinstance(response, list), "Response should be a list"
    assert len(response) > 0, "Expected at least one response item"

async def _async_test_send_chat_request_overflow_failure(backend: ALlmBaseBackend, chat_config: dict[str, Any]) -> None:
    """Test handling chat request failures caused by message overflow."""
    response = [
        item
        async for item in backend.async_send_chat_request(
            chat_config,
            MOCK_MESSAGE_CONTEXT_OVERFLOW,
            MOCK_LLM_TOOLS,
        )
    ]

    assert isinstance(response, list), "Response should be a list"
    assert len(response) > 0, "Expected at least one response item"

async def _async_test_send_chat_request_connection_failure(backend: ALlmBaseBackend, chat_config_invalid: dict[str, Any]) -> None:
    """Test handling chat request failures caused by empty messages."""
    with pytest.raises(Exception):
        async for _ in backend.async_send_chat_request(
            chat_config_invalid,
            MOCK_MESSAGES,
            MOCK_LLM_TOOLS,
        ):
            pass

async def _async_test_send_chat_request(
        backend_valid: ALlmBaseBackend, 
        backend_invalid: ALlmBaseBackend,
        chat_config: dict[str, Any],
        chat_config_invalid: dict[str, Any]
        ) -> None:
    """Test that the backend can send a chat request with a mocked Home Assistant."""
    await _async_test_send_chat_request_success(backend_valid, chat_config)
    await _async_test_send_chat_request_overflow_failure(backend_valid, chat_config)
    await _async_test_send_chat_request_connection_failure(backend_invalid, chat_config_invalid)

def test_init(backend_case: BackendCase, hass: HomeAssistant) -> None:
    """Test that the backend can be initialized with a mocked Home Assistant."""
    backend = backend_case.backend_class(hass, backend_case.user_input)
    assert backend._url_base == {
        "hostname": backend_case.user_input[CONF_LLM_HOST],
        "port": backend_case.user_input[CONF_LLM_PORT],
        "ssl": backend_case.user_input[CONF_LLM_SSL],
    }

def test_url_format(backend_case: BackendCase, hass: HomeAssistant) -> None:
    """Test that the backend can format URLs correctly."""
    backend = backend_case.backend_class(hass, backend_case.user_input)
    connection_input = backend_case.user_input
    url = backend.format_url(
        hostname=connection_input[CONF_LLM_HOST],
        port=connection_input[CONF_LLM_PORT],
        ssl=connection_input[CONF_LLM_SSL],
        path="/v1",
    )
    expected_url = f"{'https' if connection_input[CONF_LLM_SSL] else 'http'}://{connection_input[CONF_LLM_HOST]}{':' + str(connection_input[CONF_LLM_PORT]) if connection_input[CONF_LLM_PORT] else ''}/v1"
    assert url == expected_url


def test_tool_names_are_split_for_request_logging() -> None:
    """Required tools are listed separately from RAG-selected tools."""
    tools = [
        LlmTool(
            name="HassTurnOn",
            description="Turn on",
            parameters={},
            metadata={},
        ),
        LlmTool(
            name=RAGENT_PREFIXED_REQUIRED_TOOL_NAMES[0],
            description="Search",
            parameters={},
            metadata={},
        ),
    ]

    required_tool_names, searched_tool_names = ALlmBaseBackend.split_tool_names(tools)

    assert required_tool_names == [RAGENT_PREFIXED_REQUIRED_TOOL_NAMES[0]]
    assert searched_tool_names == ["HassTurnOn"]

def test_validate_connection(backend_case: BackendCase, hass: HomeAssistant) -> None:
    """Test connection validation for every backend."""
    backend_valid = backend_case.backend_class(hass, backend_case.user_input)
    backend_invalid = backend_case.backend_class(hass, backend_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_validate_connection(
            backend_valid,
            backend_invalid,
            backend_case.user_input,
            backend_case.user_input_invalid,
        )
    )

def test_get_available_models(backend_case: BackendCase, hass: HomeAssistant) -> None:
    """Test model discovery for every backend."""
    backend_valid = backend_case.backend_class(hass, backend_case.user_input)
    backend_invalid = backend_case.backend_class(hass, backend_case.user_input_invalid)
    _run_async_test(hass, _async_test_get_available_models(backend_valid, backend_invalid))

def test_get_model_info(backend_case: BackendCase, hass: HomeAssistant) -> None:
    """Test model information retrieval for every backend."""
    backend_valid = backend_case.backend_class(hass, backend_case.user_input)
    backend_invalid = backend_case.backend_class(hass, backend_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_get_model_info(
            backend_valid,
            backend_invalid,
            backend_case.chat_config,
            backend_case.chat_config_invalid,
        )
    )

def test_preload_and_unload_model(backend_case: BackendCase, hass: HomeAssistant) -> None:
    """Test model lifecycle operations for every backend."""
    backend = backend_case.backend_class(hass, backend_case.user_input)
    _run_async_test(hass, _async_test_preload_and_unload_model(backend, backend_case.chat_config))

def test_prepares_linked_tool_history(backend_case: BackendCase, hass: HomeAssistant) -> None:
    """Test that a backend preserves the link between tool calls and results."""
    backend = backend_case.backend_class(hass, backend_case.user_input)
    messages = backend.format_messages_for_backend(MOCK_TOOL_HISTORY)
    tool_call = messages[2]["tool_calls"][0]
    tool_result = messages[3]

    arguments = tool_call["function"]["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    assert arguments == {"name": "Desk light"}
    assert (
        tool_result.get("tool_call_id") == tool_call.get("id")
        or tool_result.get("tool_name") == tool_call["function"]["name"]
    )

def test_openai_truncation_keeps_complete_turns() -> None:
    """Test that OpenAI truncation does not leave orphaned tool results."""
    messages = [
        *MOCK_TOOL_HISTORY,
        ChatMessage(role="user", content="What is its state now?"),
    ]
    max_chars = sum(
        len(json.dumps(message, default=str))
        for message in (messages[0], messages[-1])
    )
    truncated = ALlmBaseBackend.truncate_messages(messages, max_chars)

    assert [message["role"] for message in truncated] == ["system", "user"]
    assert truncated[-1]["content"] == "What is its state now?"

def test_send_chat_request(backend_case: BackendCase, hass: HomeAssistant) -> None:
    """Test chat requests for every backend."""
    backend_valid = backend_case.backend_class(hass, backend_case.user_input)
    backend_invalid = backend_case.backend_class(hass, backend_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_send_chat_request(
            backend_valid,
            backend_invalid,
            backend_case.chat_config,
            backend_case.chat_config_invalid,
        )
    )
