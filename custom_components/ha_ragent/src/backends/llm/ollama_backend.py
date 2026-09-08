import asyncio
from contextlib import aclosing
import json
import logging
from functools import wraps
import aiohttp
from typing import Any, Dict, List, AsyncGenerator

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.const import (
    CONF_CONTEXT_LENGTH,
    CONF_ENABLE_MODEL_THINKING,
    CONF_LLM_HOST,
    CONF_LLM_MODEL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONNECTION_RETRIES,
    RETRY_BACKOFF_BASE_SECONDS,
    RETRY_BACKOFF_MULTIPLIER,
)
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.chat.chat_message import ChatMessage
from custom_components.ha_ragent.src.const import RAGENT_CHAT_TRUNCATE_MAX_CHARS, RAGENT_CHAT_TRUNCATE_RETRIES

_logger = logging.getLogger(__name__)

def _is_retryable_error(error: Exception) -> bool:
    if isinstance(error, (aiohttp.ClientSSLError, aiohttp.ServerFingerprintMismatch)):
        return False
    if isinstance(error, aiohttp.ClientResponseError):
        return error.status in {408, 429, 500, 502, 503, 504}
    return isinstance(error, (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, ConnectionError, TimeoutError))

async def async_request_json(session: aiohttp.ClientSession, method: str, url: str, **kwargs: Any) -> Any:
    for attempt in range(CONNECTION_RETRIES + 1):
        try:
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as error:
            if attempt == CONNECTION_RETRIES or not _is_retryable_error(error):
                raise
            await asyncio.sleep(RETRY_BACKOFF_BASE_SECONDS * (RETRY_BACKOFF_MULTIPLIER ** attempt))

def retry_stream_connection_errors(operation):
    @wraps(operation)
    async def wrapped(*args, **kwargs):
        emitted = False
        for attempt in range(CONNECTION_RETRIES + 1):
            try:
                async with aclosing(operation(*args, **kwargs)) as stream:
                    async for chunk in stream:
                        emitted = True
                        yield chunk
                return
            except Exception as error:
                if emitted or attempt == CONNECTION_RETRIES or not _is_retryable_error(error):
                    raise
                await asyncio.sleep(RETRY_BACKOFF_BASE_SECONDS * (RETRY_BACKOFF_MULTIPLIER ** attempt))
    return wrapped

class OllamaLlmBackend(ALlmBaseBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._tags_url = ALlmBaseBackend.format_url(**self._url_base, path="/api/tags")
        self._info_url = ALlmBaseBackend.format_url(**self._url_base, path="/api/show")
        self._chat_url = ALlmBaseBackend.format_url(**self._url_base, path="/api/chat")

    @staticmethod
    def get_name() -> str:
        return f"{ALlmBaseBackend.get_name()}: Ollama"
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        try:
            session = async_get_clientsession(hass)
            
            await async_request_json(
                session, "GET",
                ALlmBaseBackend.format_url(
                    hostname=user_input.get(CONF_LLM_HOST),
                    port=user_input.get(CONF_LLM_PORT),
                    ssl=user_input.get(CONF_LLM_SSL),
                    path="/api/tags"
                ),
                timeout=ALlmBaseBackend._default_timeout
            )
            return None
        except Exception as ex:
            return str(ex)
        
    async def async_get_model_info(self, model_name: str) -> ModelInfo:
        session = async_get_clientsession(self._hass)
        model_result = await async_request_json(
            session, "POST", self._info_url,
            json={"model": model_name},
            timeout=ALlmBaseBackend._default_timeout
        )

        capabilities = model_result.get("capabilities", [])
        is_tool_model = "tools" in capabilities
        is_embedding_model = "embedding" in capabilities

        return ModelInfo(
            name=model_name,
            context_size=None,
            is_tool_model=is_tool_model,
            is_embedding_model=is_embedding_model
        )
    
    async def async_preload_model(self, config_subentry: dict) -> None:
        async for _ in self.async_send_chat_request(config_subentry, [], [], keep_alive=-1):
            pass
    
    async def async_unload_model(self, config_subentry: dict) -> None:
        async for _ in self.async_send_chat_request(config_subentry, [], [], keep_alive=0):
            pass
    
    async def async_get_available_models(self) -> List[str]:
        session = async_get_clientsession(self._hass)
        models_result = await async_request_json(
            session, "GET", self._tags_url, timeout=ALlmBaseBackend._default_timeout,
        )

        names = [x["name"] for x in models_result.get("models", [])]
        infos = await asyncio.gather(*(self.async_get_model_info(name) for name in names), return_exceptions=True)
        available = []
        for info in infos:
            if isinstance(info, Exception):
                continue
            if info.is_tool_model:
                available.append(info.name)

        return available

    def format_messages_for_backend(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Convert canonical history messages to Ollama chat format."""
        prepared: List[ChatMessage] = []
        for message in messages:
            item = dict(message)
            if item.get("role") == "assistant" and item.get("tool_calls"):
                item["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        },
                    }
                    for tool_call in item["tool_calls"]
                ]
            if item.get("role") == "tool":
                item.pop("tool_call_id", None)
                if not isinstance(item.get("content"), str):
                    item["content"] = json.dumps(item.get("content"), ensure_ascii=False, default=str)
            prepared.append(item)
        return prepared
    
    @retry_stream_connection_errors
    async def _async_send_chat_request_once(self, config_subentry: dict, messages: List[ChatMessage], tools: List[LlmTool], **kwargs) -> AsyncGenerator[str, None]:
        """Send one Ollama request while preserving streaming output."""
        session = async_get_clientsession(self._hass)
        emitted = kwargs.pop("_emitted", None)
        if emitted is None:
            emitted = {"value": False}
        unexpected_reasoning_logged = False
        thinking_enabled = bool(config_subentry[CONF_ENABLE_MODEL_THINKING])

        payload = {
            "model": config_subentry[CONF_LLM_MODEL],
            "stream": "keep_alive" not in kwargs,
            "think": thinking_enabled,
            "options": {
                "temperature": config_subentry[CONF_TEMPERATURE],
                "num_ctx": config_subentry[CONF_CONTEXT_LENGTH],
                "num_predict": config_subentry[CONF_MAX_TOKENS],
            },
        }
        
        if "keep_alive" in kwargs:
            payload["keep_alive"] = kwargs["keep_alive"]
        else:
            payload["messages"] = self.format_messages_for_backend(messages)

        if tools:
            payload["tools"] = [tool.to_tool_dict() for tool in tools]
            required_tool_names, searched_tool_names = self.split_tool_names(tools)
            _logger.debug(f"Added {len(tools)} tools to Ollama request: required_tools={required_tool_names}, searched_tools={searched_tool_names}")
        
        try:
            async with session.post(self._chat_url, json=payload, timeout=ALlmBaseBackend._chat_timeout) as response:
                response.raise_for_status()
                async for line in response.content:
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        reasoning_content = data.get("message", {}).get("thinking")
                        if reasoning_content and not thinking_enabled and not unexpected_reasoning_logged:
                            _logger.warning("Model returned reasoning although model thinking is disabled in the UI.")
                            unexpected_reasoning_logged = True
                        
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            if content:
                                emitted["value"] = True
                                yield content
                        
                        if "message" in data and "tool_calls" in data["message"]:
                            tool_calls = data["message"]["tool_calls"]
                            if tool_calls:
                                emitted["value"] = True
                                _logger.debug(f"LLM tool calls received from Ollama: {tool_calls}")
                                for tc in tool_calls:
                                    if "function" in tc:
                                        func = tc["function"]
                                        tool_json = {
                                            "tool": func.get("name", "unknown"),
                                            "arguments": func.get("arguments", {})
                                        }
                                        yield f"\n```homeassistant\n{json.dumps(tool_json)}\n```\n"

                    except json.JSONDecodeError:
                        _logger.debug(f"Failed to parse Ollama response: {line}")
                        continue
        except Exception as err:
            _logger.debug("Ollama request attempt failed: %s", type(err).__name__, exc_info=True)
            raise
        return

    async def async_send_chat_request(self, config_subentry: dict, messages: List[ChatMessage], tools: List[LlmTool], **kwargs) -> AsyncGenerator[str, None]:
        """Send a chat request to Ollama and retry with truncation when empty."""
        current_messages = messages
        max_chars = RAGENT_CHAT_TRUNCATE_MAX_CHARS

        for attempt in range(RAGENT_CHAT_TRUNCATE_RETRIES + 1):
            emitted = {"value": False}
            async with aclosing(self._async_send_chat_request_once(
                config_subentry, current_messages, tools, _emitted=emitted, **kwargs,
            )) as stream:
                async for chunk in stream:
                    yield chunk

            if emitted["value"] or not messages:
                return

            if attempt == RAGENT_CHAT_TRUNCATE_RETRIES:
                return

            max_chars //= 2
            trimmed_messages = self.truncate_messages(messages, max_chars)
            if trimmed_messages == current_messages:
                _logger.debug("Ollama response was empty, but the prompt is already short enough to avoid truncation.")
                return

            current_messages = trimmed_messages
            _logger.warning(f"Ollama returned an empty response. Retrying with messages limited to {max_chars} characters.")
