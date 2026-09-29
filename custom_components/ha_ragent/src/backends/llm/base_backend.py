from abc import ABC, abstractmethod
import json
import aiohttp
from typing import Any, Dict, List, AsyncGenerator

from homeassistant.core import HomeAssistant

from custom_components.ha_ragent.src.utils import get_setting_value
from custom_components.ha_ragent.src.const import (
    CONF_LLM_API_KEY,
    CONF_LLM_HOST,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    RAGENT_PREFIXED_REQUIRED_TOOL_NAMES,
    STREAM_READ_TIMEOUT,
    HTTP_REQUEST_TIMEOUT,
    STREAM_CONNECT_TIMEOUT,
)
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.chat.chat_message import ChatMessage

class ALlmBaseBackend(ABC):
    _default_timeout = aiohttp.ClientTimeout(total=HTTP_REQUEST_TIMEOUT)
    _chat_timeout = aiohttp.ClientTimeout(total=None, sock_connect=STREAM_CONNECT_TIMEOUT, sock_read=STREAM_READ_TIMEOUT)

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self._hass = hass
        self._client_options = client_options
        self._api_key = ALlmBaseBackend.normalize_api_key(get_setting_value(CONF_LLM_API_KEY, client_options))
        self._url_base = {
            "hostname": get_setting_value(CONF_LLM_HOST, client_options),
            "port": get_setting_value(CONF_LLM_PORT, client_options),
            "ssl": get_setting_value(CONF_LLM_SSL, client_options),
        }

    @staticmethod
    def normalize_api_key(api_key: str) -> str:
        return str(api_key or "").strip()
        
    @staticmethod
    def format_url(hostname: str, port: str, ssl: bool, path: str) -> str:
        return f"{'https' if ssl else 'http'}://{hostname}{ ':' + str(port) if port else ''}{path}"

    @staticmethod
    def get_name() -> str:
        return "LLM"
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        raise NotImplementedError()

    @staticmethod
    def convert_tools_to_model_format(tools: List[LlmTool]) -> List[Dict[str, Any]]:
        """Build complete OpenAI function tool definitions from the tool models."""
        return [tool.to_tool_dict() for tool in tools]

    @staticmethod
    def split_tool_names(tools: List[LlmTool]) -> tuple[list[str], list[str]]:
        """Separate always-required and searched tool names for logging."""
        required_names = set(RAGENT_PREFIXED_REQUIRED_TOOL_NAMES)
        required_tools_names: list[str] = []
        searched_tool_names: list[str] = []
        for tool in tools:
            target = (
                required_tools_names
                if tool.name in required_names
                else searched_tool_names
            )
            target.append(tool.name)
        return required_tools_names, searched_tool_names

    @staticmethod
    def truncate_messages(messages: List[ChatMessage], max_chars: int) -> List[ChatMessage]:
        """Drop oldest turns while retaining rules, current state and latest user."""
        if not messages:
            return []
        prefix = 1 if messages[0].get("role") == "system" else 0
        state_index = next(
            (index for index in range(len(messages) - 2, prefix - 1, -1)
             if messages[index].get("role") == "system"),
            None,
        )
        suffix_start = state_index if state_index is not None else len(messages) - 1
        result = [dict(message) for message in messages[:prefix]]
        suffix = [dict(message) for message in messages[suffix_start:]]
        latest_user_index = next(
            (index for index in range(len(suffix) - 1, -1, -1)
             if suffix[index].get("role") == "user"),
            None,
        )
        if latest_user_index is not None:
            latest = suffix[latest_user_index]
            original_content = str(latest.get("content") or "")
            fixed_size = sum(
                len(json.dumps(message, default=str))
                for message in result + suffix[:latest_user_index] + suffix[latest_user_index + 1:]
            )
            available = max_chars - fixed_size
            if len(json.dumps(latest, default=str)) > available:
                marker = "[Earlier user message content omitted]\n"
                marker_size = len(json.dumps({**latest, "content": marker}, default=str))
                if available < marker_size:
                    latest["content"] = ""
                else:
                    low, high = 0, len(original_content)
                    while low < high:
                        middle = (low + high + 1) // 2
                        candidate = marker + original_content[-middle:]
                        if len(json.dumps({**latest, "content": candidate}, default=str)) <= available:
                            low = middle
                        else:
                            high = middle - 1
                    latest["content"] = marker + original_content[-low:] if low else marker
        remaining = max_chars - sum(
            len(json.dumps(message, default=str)) for message in result + suffix
        )
        turns: List[List[ChatMessage]] = []
        for message in messages[prefix:suffix_start]:
            if message.get("role") == "user":
                turns.append([message])
            elif turns:
                turns[-1].append(message)
        selected: List[List[ChatMessage]] = []
        for turn in reversed(turns):
            size = sum(len(json.dumps(message, default=str)) for message in turn)
            if size > remaining:
                break
            selected.insert(0, [dict(message) for message in turn])
            remaining -= size
        for turn in selected:
            result.extend(turn)
        result.extend(suffix)
        return result

    @abstractmethod
    def format_messages_for_backend(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        raise NotImplementedError()

    @abstractmethod
    async def async_get_model_info(self, model_name: str) -> ModelInfo:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_preload_model(self, config_subentry: dict) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_unload_model(self, config_subentry: dict) -> None:
        raise NotImplementedError()

    async def async_close(self) -> None:
        """Release resources owned by the LLM backend."""
    
    @abstractmethod
    async def async_get_available_models(self) -> List[str]:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_send_chat_request(self, config_subentry: dict, messages: List[ChatMessage], tools: List[LlmTool], **kwargs) -> AsyncGenerator[str, None]:
        raise NotImplementedError()
