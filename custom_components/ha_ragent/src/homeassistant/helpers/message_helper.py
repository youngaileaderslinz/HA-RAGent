"""Helpers for converting and cleaning conversation messages."""

from __future__ import annotations

import json
from homeassistant.components import conversation

from custom_components.ha_ragent.src.const import (
    RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
    TOOL_REGEX_PATTERN,
)
from custom_components.ha_ragent.src.models.chat.chat_message import (
    ChatFunction,
    ChatMessage,
    ChatToolCall,
    ChatToolFailure,
)


class MessageHelper:
    _MAX_RESULT_ITEMS = 12
    _MAX_RESULT_TEXT = 2000
    _MAX_RESULT_DEPTH = 4

    @staticmethod
    def _is_semantic_search(tool_name: str) -> bool:
        return str(tool_name or "").rsplit("__", 1)[-1] == RAGENT_SEMANTIC_SEARCH_TOOL_NAME

    @staticmethod
    def _compact_candidate_devices(candidates: object) -> list[object]:
        """Preserve bounded state and location data for candidate devices."""
        if not isinstance(candidates, list):
            return []
        retained_keys = (
            "name",
            "friendly_name",
            "aliases",
            "state",
            "unit_of_measurement",
            "area",
            "floor",
            "area_aliases",
            "floor_aliases",
            "domain",
            "device_class",
            "attributes",
        )
        compact: list[object] = []
        for candidate in candidates[:MessageHelper._MAX_RESULT_ITEMS]:
            if not isinstance(candidate, dict):
                compact.append(candidate)
                continue
            compact.append(
                {
                    key: MessageHelper._compact_value(candidate[key])
                    for key in retained_keys
                    if candidate.get(key) is not None
                }
            )
        return compact

    @staticmethod
    def _compact_candidate_tools(candidates: object) -> list[object]:
        """Preserve compact capability and confidence data for candidate tools."""
        if not isinstance(candidates, list):
            return []
        retained_keys = (
            "name",
            "description",
            "canonical_action",
            "supported_domains",
            "retrieval_score",
        )
        compact: list[object] = []
        for candidate in candidates[:MessageHelper._MAX_RESULT_ITEMS]:
            if not isinstance(candidate, dict):
                compact.append(candidate)
                continue
            compact.append(
                {
                    key: MessageHelper._compact_value(candidate[key])
                    for key in retained_keys
                    if candidate.get(key) is not None
                }
            )
        return compact

    @staticmethod
    def _compact_value(value: object, depth: int = 0) -> object:
        if isinstance(value, (dict, list, tuple)) and depth >= MessageHelper._MAX_RESULT_DEPTH:
            return "[truncated]"
        if isinstance(value, dict):
            return {
                key: MessageHelper._compact_value(item, depth + 1)
                for key, item in list(value.items())[:MessageHelper._MAX_RESULT_ITEMS]
            }
        if isinstance(value, (list, tuple)):
            return [
                MessageHelper._compact_value(item, depth + 1)
                for item in value[:MessageHelper._MAX_RESULT_ITEMS]
            ]
        if isinstance(value, str):
            return value[:MessageHelper._MAX_RESULT_TEXT]
        return value

    @staticmethod
    def compact_tool_result_value(tool_name: str, result: object) -> object:
        """Keep actionable result details while bounding prompt size."""
        if not isinstance(result, dict):
            return MessageHelper._compact_value(result)
        if MessageHelper._is_semantic_search(tool_name):
            devices = result.get("candidate_devices", result.get("devices", []))
            tools = result.get("candidate_tools", result.get("tools", []))
            return {
                "result_type": "candidate_search",
                "candidate_notice": "Candidates only; no action has been performed.",
                "candidate_data_notice": (
                    "Use the included state and location data directly when it answers the request."
                ),
                "candidate_devices": MessageHelper._compact_candidate_devices(devices),
                "candidate_tools": MessageHelper._compact_candidate_tools(tools),
                "candidate_device_count": len(devices) if isinstance(devices, list) else 0,
                "candidate_tool_count": len(tools) if isinstance(tools, list) else 0,
                "tool_search_status": result.get("tool_search_status", ""),
                "tool_search_confidence": result.get("tool_search_confidence", ""),
                "fallback_required": bool(result.get("fallback_required", False)),
                "tool_search_message": MessageHelper._compact_value(
                    result.get("tool_search_message", "")
                ),
                "error": MessageHelper._compact_value(result.get("error", [])),
                **({"reused": True} if result.get("reused") else {}),
            }

        retained_keys = ("success", "failed", "error", "errors", "already_executed")
        compact = {
            key: MessageHelper._compact_value(result[key])
            for key in retained_keys
            if key in result
        }
        # Status fields must not hide custom payloads, scheduled times, memory
        # IDs or details needed to correct a failed call.
        for key, value in result.items():
            if key not in compact:
                if len(compact) >= MessageHelper._MAX_RESULT_ITEMS:
                    break
                compact[key] = MessageHelper._compact_value(value)
        return compact

    @staticmethod
    def tool_result_succeeded(result: object) -> bool:
        """Return whether a tool result represents a success."""
        if isinstance(result, dict):
            if "success" in result:
                return bool(result["success"])
            if any(result.get(key) for key in ("error", "errors", "failed")):
                return False
            return True

        success = getattr(result, "success", None)
        if success is not None:
            return bool(success)
        return not any(getattr(result, key, None) for key in ("error", "errors", "failed"))

    @staticmethod
    def create_repeated_tool_result_message(
        agent_id: str | None,
        tool_call_id: str | None,
        tool_name: str,
        previous_result: object,
    ) -> conversation.ToolResultContent:
        """Return the original result for a repeated tool call."""
        if MessageHelper._is_semantic_search(tool_name):
            reused_result = dict(previous_result) if isinstance(previous_result, dict) else {"result": previous_result}
            reused_result["reused"] = True
            return conversation.ToolResultContent(
                agent_id=agent_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_result=MessageHelper.compact_tool_result_value(tool_name, reused_result),
            )

        success_value = previous_result.get("success")
        if success_value is None:
            success = not any(previous_result.get(key) for key in ("error", "errors", "failed"))
        else:
            success = bool(success_value)
        result = {
            "success": success,
            "already_executed": True,
        }
        for error_key in ("error", "errors", "failed"):
            if result["success"] is False and error_key in previous_result:
                result[error_key] = previous_result[error_key]

        return conversation.ToolResultContent(
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_result=result
        )

    @staticmethod
    def create_tool_failure_message(
        agent_id: str | None,
        tool_call_id: str | None,
        tool_name: str,
        error: Exception,
    ) -> conversation.ToolResultContent:
        """Create a tool-result message for a failed tool call."""
        error_value = (
            "Unknown error ensure you follow the tool call format."
            if isinstance(error, KeyError) and error.args
            else str(error)
        )
        failure = ChatToolFailure(
            success=False,
            tool=tool_name,
            error=error_value,
        )
        
        return conversation.ToolResultContent(
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_result=failure,
        )

    @staticmethod
    def compact_tool_result(tool_message: conversation.ToolResultContent) -> conversation.ToolResultContent:
        """Compact tool results stored in prompt history."""
        return conversation.ToolResultContent(
            agent_id=tool_message.agent_id,
            tool_call_id=tool_message.tool_call_id,
            tool_name=tool_message.tool_name,
            tool_result=MessageHelper.compact_tool_result_value(
                tool_message.tool_name,
                tool_message.tool_result,
            ),
        )

    @staticmethod
    def message_to_retrieval_text(message: conversation.Content) -> str:
        """Convert Home Assistant content into retrieval text."""
        if isinstance(
            message,
            (conversation.UserContent, conversation.AssistantContent),
        ):
            return (message.content or "").strip()

        if isinstance(message, conversation.ToolResultContent):
            return f"{message.tool_name} {message.tool_result}".strip()

        return ""

    @staticmethod
    def message_to_chat_messages(messages: list[conversation.Content]) -> list[ChatMessage]:
        """Convert Home Assistant content into canonical backend messages."""
        formatted_messages: list[ChatMessage] = []

        for message in messages:
            if isinstance(message, conversation.SystemContent):
                formatted_messages.append(
                    ChatMessage(role="system", content=message.content)
                )
            elif isinstance(message, conversation.UserContent):
                formatted_messages.append(
                    ChatMessage(role="user", content=message.content)
                )
            elif isinstance(message, conversation.AssistantContent):
                assistant_message = ChatMessage(
                    role="assistant",
                    content=message.content or "",
                )
                if message.tool_calls:
                    assistant_message["tool_calls"] = [
                        ChatToolCall(
                            id=tool_call.id,
                            type="function",
                            function=ChatFunction(
                                name=tool_call.tool_name,
                                arguments=tool_call.tool_args,
                            ),
                        )
                        for tool_call in message.tool_calls
                    ]
                formatted_messages.append(assistant_message)
            elif isinstance(message, conversation.ToolResultContent):
                tool_message = ChatMessage(
                    role="tool",
                    content=MessageHelper.compact_tool_result_value(
                        message.tool_name,
                        message.tool_result,
                    ),
                    tool_name=message.tool_name
                )
                if message.tool_call_id:
                    tool_message["tool_call_id"] = message.tool_call_id
                formatted_messages.append(tool_message)

        return formatted_messages

    @staticmethod
    def clean_assistant_content(assistant_content: str, has_tool_calls: bool) -> str:
        """Remove the internal fenced tool representation from stored content."""
        return TOOL_REGEX_PATTERN.sub("", assistant_content).strip() if has_tool_calls else assistant_content
