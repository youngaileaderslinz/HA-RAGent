from functools import partial
import json
import logging
from openai import AsyncOpenAI, BadRequestError
from typing import Any, AsyncGenerator, Dict, List

from homeassistant.core import HomeAssistant

from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.const import (
    CONF_ENABLE_MODEL_THINKING,
    CONF_LLM_API_KEY,
    CONF_LLM_HOST,
    CONF_LLM_MODEL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONNECTION_RETRIES,
    RAGENT_CHAT_TRUNCATE_MAX_CHARS,
    RAGENT_CHAT_TRUNCATE_RETRIES
)
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.chat.chat_message import ChatMessage

_logger = logging.getLogger(__name__)

class OpenAiLlmBackend(ALlmBaseBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._openai_url = ALlmBaseBackend.format_url(**self._url_base, path="/v1")
        self._client: AsyncOpenAI | None = None

    async def _async_get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = await self._hass.async_add_executor_job(
                partial(
                    AsyncOpenAI,
                    base_url=self._openai_url,
                    api_key=self._api_key or "not-needed",
                    max_retries=CONNECTION_RETRIES,
                )
            )
        return self._client

    @staticmethod
    def get_name() -> str:
        return f"{ALlmBaseBackend.get_name()}: OpenAI API"

    @staticmethod
    def _is_context_length_error(error: Exception) -> bool:
        return isinstance(error, BadRequestError) and error.status_code == 400 and error.type == "exceed_context_size_error"

    def format_messages_for_backend(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Convert canonical history messages to OpenAI Chat Completions format."""
        prepared: List[ChatMessage] = []
        for message in messages:
            item = dict(message)
            if item.get("role") == "assistant" and item.get("tool_calls"):
                item["tool_calls"] = [
                    {
                        **tool_call,
                        "function": {
                            **tool_call["function"],
                            "arguments": (
                                tool_call["function"]["arguments"]
                                if isinstance(tool_call["function"]["arguments"], str)
                                else json.dumps(tool_call["function"]["arguments"])
                            ),
                        },
                    }
                    for tool_call in item["tool_calls"]
                ]
            if item.get("role") == "tool":
                item.pop("tool_name", None)
                if not isinstance(item.get("content"), str):
                    item["content"] = json.dumps(item.get("content"), ensure_ascii=False, default=str)
            prepared.append(item)
        return prepared

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        client = None
        try:
            base_url = ALlmBaseBackend.format_url(
                hostname=user_input.get(CONF_LLM_HOST),
                port=user_input.get(CONF_LLM_PORT),
                ssl=user_input.get(CONF_LLM_SSL),
                path="/v1",
            )
            api_key = ALlmBaseBackend.normalize_api_key(user_input.get(CONF_LLM_API_KEY))
            client = await hass.async_add_executor_job(
                partial(
                    AsyncOpenAI,
                    base_url=base_url,
                    api_key=api_key or "not-needed",
                    max_retries=CONNECTION_RETRIES,
                )
            )
            await client.models.list()
            return None
        except Exception as ex:
            return str(ex)
        finally:
            if client:
                await client.close()

    async def async_get_model_info(self, model_name: str) -> ModelInfo:
        try:
            client = await self._async_get_client()
            models = await client.models.list()
            model = next((m for m in models.data if m.id == model_name), None)

            if not model:
                raise ValueError(f"Model not found: {model_name}")

            data = model.model_dump() 
            meta = data.get("meta") or {}
            context_size = int(meta.get("n_ctx") or 0) 

            return ModelInfo( 
                name=model.id, 
                context_size=context_size, 
                is_embedding_model=None, 
                is_tool_model=None
            )
        except Exception as ex:
            _logger.error(f"Error retrieving model info for {model_name}: {ex}", exc_info=True)
            raise

    async def async_preload_model(self, config_subentry: dict) -> None:
        _logger.info("Preloading not supported for OpenAI Compatible LLM backend.")

    async def async_unload_model(self, config_subentry: dict) -> None:
        _logger.info("Unloading not supported for OpenAI Compatible LLM backend.")

    async def async_get_available_models(self) -> List[str]:
        client = await self._async_get_client()
        result = await client.models.list()
        return [model.id for model in result.data if model.id]

    async def async_send_chat_request(self, config_subentry: dict, messages: List[ChatMessage], tools: List[LlmTool], **kwargs) -> AsyncGenerator[str, None]:
        prepared_messages = self.format_messages_for_backend(messages)
        request: Dict[str, Any] = {
            "model": config_subentry[CONF_LLM_MODEL],
            "messages": prepared_messages,
            "stream": True,
            "temperature": config_subentry[CONF_TEMPERATURE],
            "max_tokens": config_subentry[CONF_MAX_TOKENS],
        }
        thinking_enabled = bool(config_subentry[CONF_ENABLE_MODEL_THINKING])

        request["extra_body"] = {
            "parse_tool_calls": True,
            "chat_template_kwargs": {
                "enable_thinking": thinking_enabled,
            },
        }

        if tools:
            request["tools"] = self.convert_tools_to_model_format(tools)
            request["tool_choice"] = "auto"
            required_tool_names, searched_tool_names = self.split_tool_names(tools)
            _logger.debug(f"Added {len(tools)} tools to OpenAI-compatible request: required_tools={required_tool_names}, searched_tools={searched_tool_names}")

        try:
            client = await self._async_get_client()
            max_chars = RAGENT_CHAT_TRUNCATE_MAX_CHARS
            unexpected_reasoning_logged = False
            for attempt in range(RAGENT_CHAT_TRUNCATE_RETRIES + 1):
                request["messages"] = (
                    prepared_messages
                    if attempt == 0
                    else self.truncate_messages(prepared_messages, max_chars)
                )
                pending_tool_calls: Dict[int, Dict[str, str]] = {}
                try:
                    stream = await client.chat.completions.create(**request)

                    async with stream:
                        async for chunk in stream:
                            for choice in chunk.choices:
                                delta = choice.delta

                                if delta.content:
                                    yield delta.content

                                reasoning_content = getattr(delta, "reasoning_content", None)
                                if reasoning_content and not thinking_enabled and not unexpected_reasoning_logged:
                                    _logger.warning("Model returned reasoning although model thinking is disabled in the UI.")
                                    unexpected_reasoning_logged = True

                                for tool_call in delta.tool_calls or []:
                                    index = tool_call.index or 0
                                    function = tool_call.function
                                    pending = pending_tool_calls.setdefault(index, {"name": "", "arguments": ""})

                                    if function and function.name:
                                        pending["name"] = function.name
                                    if function and function.arguments:
                                        pending["arguments"] += function.arguments
                    break
                except Exception as err:
                    if not self._is_context_length_error(err) or attempt == RAGENT_CHAT_TRUNCATE_RETRIES:
                        raise

                    max_chars //= 2
                    _logger.warning(f"LLM input is too large. Retrying with messages limited to {max_chars} characters.")

            # Tool-call arguments are often split across many SSE chunks.
            # Emit only after the complete JSON document has been assembled.
            if pending_tool_calls:
                _logger.debug(f"LLM tool calls received from OpenAI-compatible backend: {list(pending_tool_calls.values())}")

            for pending in pending_tool_calls.values():
                if not pending["name"]:
                    continue

                tool_dict = {
                    "tool": pending["name"],
                    "arguments": pending["arguments"],
                }

                yield (
                    "\n```homeassistant\n"
                    f"{json.dumps(tool_dict)}"
                    "\n```\n"
                )

        except Exception as err:
            _logger.error(f"Error calling llama.cpp API through OpenAI client: {err}", exc_info=True)
            raise
