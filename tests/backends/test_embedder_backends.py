from __future__ import annotations

import asyncio
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
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_API_KEY,
    CONF_EMBEDDING_MODEL,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
)
from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder


from custom_components.ha_ragent.src.backends.embedder.openai_backend import OpenAiEmbedder
from custom_components.ha_ragent.src.backends.embedder import (
    ollama_backend as ollama_backend_module,
)
from custom_components.ha_ragent.src.backends.embedder.ollama_backend import OllamaEmbedder


MOCK_EMBEDDING_DEFAULT_OPTIONS = {
    CONF_EMBEDDING_HOST: "llamacpp_embed",
    CONF_EMBEDDING_SSL: False,
    CONF_EMBEDDING_API_KEY: None,
}
MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT = {
    **MOCK_EMBEDDING_DEFAULT_OPTIONS,
    CONF_EMBEDDING_PORT: 8080,
}
MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT_INVALID = {
    **MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT,
    CONF_EMBEDDING_HOST: "invalid_host",
}
MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT = {
    **MOCK_EMBEDDING_DEFAULT_OPTIONS,
    CONF_EMBEDDING_HOST: "ollama",
    CONF_EMBEDDING_PORT: 11434,
}
MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT_INVALID = {
    **MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT,
    CONF_EMBEDDING_HOST: "invalid_host",
}
MOCK_OPENAI_EMBEDDING_CONFIG = {
    **MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT,
    CONF_EMBEDDING_MODEL: "nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M",
}
MOCK_OPENAI_EMBEDDING_CONFIG_INVALID = {
    **MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT,
    CONF_EMBEDDING_MODEL: "invalid_model",
}
MOCK_OLLAMA_EMBEDDING_CONFIG = {
    **MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT,
    CONF_EMBEDDING_MODEL: "all-minilm:33m",
}
MOCK_OLLAMA_EMBEDDING_CONFIG_INVALID = {
    **MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT,
    CONF_EMBEDDING_MODEL: "invalid_model",
}
MOCK_LLM_TOOLS = [
    LlmTool(name="test_tool", description="A tool used by backend tests.", parameters={}),
    LlmTool(name="another_tool", description="Another tool used by backend tests.", parameters={}),
]
MOCK_LLM_TOOLS_EMBEDDING_OVERFLOW = [
    *MOCK_LLM_TOOLS,
    LlmTool(
        name="overflow_tool",
        description="A tool with a very long description to test embedding context overflow.",
        parameters={"text": "".join(f"Overflow text INDEX: {index}" for index in range(10000))},
    ),
]

@dataclass(frozen=True)
class EmbedderCase:
    """Configuration for one embedder backend test case."""

    backend_class: type[ABaseEmbedder]
    user_input: dict[str, Any]
    user_input_invalid: dict[str, Any]
    embedding_config: dict[str, Any]
    embedding_config_invalid: dict[str, Any]

EMBEDDER_BACKENDS = [
    EmbedderCase(
        OpenAiEmbedder,
        MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT,
        MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT_INVALID,
        MOCK_OPENAI_EMBEDDING_CONFIG,
        MOCK_OPENAI_EMBEDDING_CONFIG_INVALID,
    ),
    EmbedderCase(
        OllamaEmbedder,
        MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT,
        MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT_INVALID,
        MOCK_OLLAMA_EMBEDDING_CONFIG,
        MOCK_OLLAMA_EMBEDDING_CONFIG_INVALID,
    ),
]

@pytest.fixture(params=EMBEDDER_BACKENDS, ids=lambda case: case.backend_class.__name__)
def embedder_case(request: pytest.FixtureRequest) -> EmbedderCase:
    """Provide every supported embedder from the central backend list."""
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

async def _async_test_connection_success(backend: ABaseEmbedder, connection_input: dict[str, Any]) -> None:
    """Test that the backend can validate a connection with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(backend._hass, connection_input)
    assert validation_error is None, validation_error

async def _async_test_connection_failure(backend: ABaseEmbedder, connection_input_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle connection failures with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(backend._hass, connection_input_invalid)
    assert validation_error is not None, "Expected a validation error for invalid input"

async def _async_test_validate_connection(
        backend_valid: ABaseEmbedder, 
        backend_invalid: ABaseEmbedder,
        user_input_valid: dict[str, Any],
        user_input_invalid: dict[str, Any]
        ) -> None:
    """Test that the backend can validate a connection with a mocked Home Assistant."""
    await _async_test_connection_success(backend_valid, user_input_valid)
    await _async_test_connection_failure(backend_invalid, user_input_invalid)

async def _async_test_get_available_models_success(backend: ABaseEmbedder) -> None:
    """Test that the backend can retrieve available models with a mocked Home Assistant."""
    models = await backend.async_get_available_models()
    assert isinstance(models, list), "Available models should be a list"
    assert len(models) > 0, "Expected at least one available model"

async def _async_test_get_available_models_failure(backend: ABaseEmbedder) -> None:
    """Test that the backend can handle available models retrieval failures with a mocked Home Assistant."""
    with pytest.raises(Exception):
        await backend.async_get_available_models()

async def _async_test_get_available_models(backend_valid: ABaseEmbedder, backend_invalid: ABaseEmbedder) -> None:
    """Test that the backend can retrieve available models with a mocked Home Assistant."""
    await _async_test_get_available_models_success(backend_valid)
    await _async_test_get_available_models_failure(backend_invalid)

async def _async_test_get_model_info_success(backend: ABaseEmbedder, embedding_config: dict[str, Any]) -> None:
    """Test that the backend can retrieve model info with a mocked Home Assistant."""
    model_info = await backend.async_get_model_info(embedding_config[CONF_EMBEDDING_MODEL])
    assert isinstance(model_info, ModelInfo), "Model info should be a ModelInfo instance"
    assert model_info.name == embedding_config[CONF_EMBEDDING_MODEL], "Model name should match the requested model"
    assert model_info.context_size is None or model_info.context_size > 0, "Context size should be None or greater than 0"
    assert model_info.is_embedding_model is None or model_info.is_embedding_model in [True, False], "is_embedding_model should be None or a boolean"
    assert model_info.is_tool_model is None or model_info.is_tool_model in [True, False], "is_tool_model should be None or a boolean"

async def _async_test_get_model_info_failure(backend: ABaseEmbedder, embedding_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle model info retrieval failures with a mocked Home Assistant."""
    with pytest.raises(Exception):
        await backend.async_get_model_info(embedding_config_invalid[CONF_EMBEDDING_MODEL])

async def _async_test_get_model_info(
        backend_valid: ABaseEmbedder, 
        backend_invalid: ABaseEmbedder, 
        embedding_config: dict[str, Any], 
        embedding_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can retrieve model info with a mocked Home Assistant."""
    await _async_test_get_model_info_success(backend_valid, embedding_config)
    await _async_test_get_model_info_failure(backend_invalid, embedding_config_invalid)

async def _async_test_preload_and_unload_model(backend: ABaseEmbedder, embedding_config: dict[str, Any]) -> None:
    """Test that the backend can preload and unload a model with a mocked Home Assistant."""
    await backend.async_preload_model(embedding_config)
    await backend.async_unload_model(embedding_config)

async def _async_test_embed_tools(backend: ABaseEmbedder, embedding_config: dict[str, Any]) -> None:
    """Test that the backend can embed tools with a mocked Home Assistant."""
    embeddings = await backend.async_embed_object(embedding_config, MOCK_LLM_TOOLS)
    assert isinstance(embeddings, list), "Response should be a list"
    assert len(embeddings) > 0, "Expected at least one response item"
    assert all(isinstance(embedding, LlmToolEmbedding) for embedding in embeddings), "Each embedding should be a ToolEmbedding instance"
    assert len(embeddings) == len(MOCK_LLM_TOOLS)

async def _async_test_embed_tools_overflow(backend: ABaseEmbedder, embedding_config: dict[str, Any]) -> None:
    """Test that the backend can handle embedding tools with a mocked Home Assistant."""
    embeddings = await backend.async_embed_object(embedding_config, MOCK_LLM_TOOLS_EMBEDDING_OVERFLOW)
    assert isinstance(embeddings, list), "Response should be a list"
    assert len(embeddings) > 0, "Expected at least one response item"
    assert all(isinstance(embedding, LlmToolEmbedding) for embedding in embeddings), "Each embedding should be a ToolEmbedding instance"
    assert len(embeddings) == len(MOCK_LLM_TOOLS_EMBEDDING_OVERFLOW)

async def _async_test_embed_tools_connection_failure(backend: ABaseEmbedder, embedding_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle embedding tools failures with a mocked Home Assistant."""
    with pytest.raises(Exception):
        await backend.async_embed_object(embedding_config_invalid, MOCK_LLM_TOOLS)

async def _async_test_embed_tool_scenarios(
        backend_valid: ABaseEmbedder, 
        backend_invalid: ABaseEmbedder,
        embedding_config: dict[str, Any],
        embedding_config_invalid: dict[str, Any]
        ) -> None:
    """Test that the backend can send a chat request with a mocked Home Assistant."""
    await _async_test_embed_tools(backend_valid, embedding_config)
    await _async_test_embed_tools_overflow(backend_valid, embedding_config)
    await _async_test_embed_tools_connection_failure(backend_invalid, embedding_config_invalid)

def test_init(embedder_case: EmbedderCase, hass: HomeAssistant) -> None:
    """Test that the backend can be initialized with a mocked Home Assistant."""
    backend = embedder_case.backend_class(hass, embedder_case.user_input)
    assert backend._url_base == {
        "hostname": embedder_case.user_input[CONF_EMBEDDING_HOST],
        "port": embedder_case.user_input[CONF_EMBEDDING_PORT],
        "ssl": embedder_case.user_input[CONF_EMBEDDING_SSL],
    }

def test_url_format(embedder_case: EmbedderCase, hass: HomeAssistant) -> None:
    """Test that the backend can format URLs correctly."""
    backend = embedder_case.backend_class(hass, embedder_case.user_input)
    connection_input = embedder_case.user_input
    url = backend.format_url(
        hostname=connection_input[CONF_EMBEDDING_HOST],
        port=connection_input[CONF_EMBEDDING_PORT],
        ssl=connection_input[CONF_EMBEDDING_SSL],
        path="/v1",
    )
    expected_url = f"{'https' if connection_input[CONF_EMBEDDING_SSL] else 'http'}://{connection_input[CONF_EMBEDDING_HOST]}{':' + str(connection_input[CONF_EMBEDDING_PORT]) if connection_input[CONF_EMBEDDING_PORT] else ''}/v1"
    assert url == expected_url

def test_validate_connection(embedder_case: EmbedderCase, hass: HomeAssistant) -> None:
    """Test connection validation for every embedder backend."""
    backend_valid = embedder_case.backend_class(hass, embedder_case.user_input)
    backend_invalid = embedder_case.backend_class(hass, embedder_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_validate_connection(
            backend_valid,
            backend_invalid,
            embedder_case.user_input,
            embedder_case.user_input_invalid,
        )
    )

def test_get_available_models(embedder_case: EmbedderCase, hass: HomeAssistant) -> None:
    """Test model discovery for every embedder backend."""
    backend_valid = embedder_case.backend_class(hass, embedder_case.user_input)
    backend_invalid = embedder_case.backend_class(hass, embedder_case.user_input_invalid)
    _run_async_test(hass, _async_test_get_available_models(backend_valid, backend_invalid))

def test_get_model_info(embedder_case: EmbedderCase, hass: HomeAssistant) -> None:
    """Test model information retrieval for every embedder backend."""
    backend_valid = embedder_case.backend_class(hass, embedder_case.user_input)
    backend_invalid = embedder_case.backend_class(hass, embedder_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_get_model_info(
            backend_valid,
            backend_invalid,
            embedder_case.embedding_config,
            embedder_case.embedding_config_invalid,
        )
    )

def test_preload_and_unload_model(embedder_case: EmbedderCase, hass: HomeAssistant) -> None:
    """Test model lifecycle operations for every embedder backend."""
    backend = embedder_case.backend_class(hass, embedder_case.user_input)
    _run_async_test(hass,
        _async_test_preload_and_unload_model(
            backend,
            embedder_case.embedding_config,
        )
    )

def test_embed_tools(embedder_case: EmbedderCase, hass: HomeAssistant) -> None:
    """Test tool embedding for every embedder backend."""
    backend_valid = embedder_case.backend_class(hass, embedder_case.user_input)
    backend_invalid = embedder_case.backend_class(hass, embedder_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_embed_tool_scenarios(
            backend_valid,
            backend_invalid,
            embedder_case.embedding_config,
            embedder_case.embedding_config_invalid,
        )
    )
