from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationInput
from homeassistant.util import dt as dt_util

from custom_components.ha_ragent.src.const import (
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
)
from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import MessageHelper
from custom_components.ha_ragent.src.utils import get_setting_value
from custom_components.ha_ragent.src.models.retrieval.target_group import TargetGroup
from custom_components.ha_ragent.src.models.retrieval.turn_context import TurnContext

class HistoryManager:
    def __init__(self, runtime_options: dict[str, Any]) -> None:
        self._runtime_options = runtime_options
        self._message_history: list[conversation.Content] = []

    @property
    def message_history(self) -> list[conversation.Content]:
        """Return the active Home Assistant message history."""
        return self._message_history

    def _select_retained_turns(self, chat_log: conversation.ChatLog) -> list[list[conversation.Content]]:
        """Select complete conversation turns within the configured retention limits."""
        remember_time_minutes = get_setting_value(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, self._runtime_options)
        remember_num_interactions = get_setting_value(CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS, self._runtime_options)

        if not remember_time_minutes and not remember_num_interactions:
            return []

        # Home Assistant has already appended the current user input to the log.
        # It is added explicitly when the prompt history is built below.
        raw_history = list(chat_log.content)[:-1]
        turns: list[list[conversation.Content]] = []

        for message in raw_history:
            if isinstance(message, conversation.SystemContent):
                continue
            elif isinstance(message, conversation.UserContent):
                turns.append([message])
            elif turns:
                turns[-1].append(message)

        if remember_time_minutes:
            now = dt_util.utcnow()
            cutoff = now - timedelta(minutes=remember_time_minutes)
            turns = [turn for turn in turns if getattr(turn[0], "created_at", now) >= cutoff]

        if remember_num_interactions and len(turns) > remember_num_interactions:
            turns = turns[-remember_num_interactions:]

        return turns

    def select_retained_history(self, chat_log: conversation.ChatLog) -> list[conversation.Content]:
        """Return retained history as a flat message list."""
        return [message for turn in self._select_retained_turns(chat_log) for message in turn]

    @staticmethod
    def _add_values(target: set[str], value: object) -> None:
        if isinstance(value, str) and value:
            target.add(value)
        elif isinstance(value, (list, tuple, set)):
            target.update(str(item) for item in value if isinstance(item, (str, int, float)))

    @classmethod
    def _collect_tool_result(
        cls,
        result: object,
        entities: set[str],
        tools: set[str],
        areas: set[str],
        domains: set[str],
        device_classes: set[str],
        ambiguous_entities: set[str],
    ) -> None:
        if not isinstance(result, dict):
            return
        cls._add_values(entities, result.get("success"))
        devices = result.get("devices")
        if isinstance(devices, list):
            candidate_ids: list[str] = []
            for device in devices:
                if not isinstance(device, dict):
                    continue
                name = device.get("name") or device.get("entity_id")
                cls._add_values(entities, name)
                if isinstance(name, str):
                    candidate_ids.append(name)
                cls._add_values(areas, device.get("area"))
                cls._add_values(domains, device.get("domain"))
                cls._add_values(device_classes, device.get("device_class"))
            if len(candidate_ids) > 1:
                ambiguous_entities.update(candidate_ids)
        result_tools = result.get("tools")
        if isinstance(result_tools, list):
            for tool in result_tools:
                if isinstance(tool, dict):
                    cls._add_values(tools, tool.get("name"))

    @staticmethod
    def _is_semantic_search(tool_name: str) -> bool:
        return tool_name.rsplit("__", 1)[-1] == RAGENT_SEMANTIC_SEARCH_TOOL_NAME

    @classmethod
    def _collect_search_candidates(cls, result: object, ambiguous_entities: set[str]) -> None:
        """Collect search matches as weak evidence, not successful targets."""
        if not isinstance(result, dict):
            return
        candidates = result.get("candidate_devices", result.get("devices", []))
        if not isinstance(candidates, list):
            return
        cls._add_values(
            ambiguous_entities,
            [
                device.get("name") or device.get("entity_id")
                if isinstance(device, dict)
                else device
                for device in candidates
            ],
        )

    @classmethod
    def _successful_target_entities(cls, arguments: dict[str, object], result: object) -> set[str]:
        """Return only entities confirmed by a successful call result."""
        argument_entities: set[str] = set()
        cls._add_values(argument_entities, arguments.get("name"))
        if not isinstance(result, dict):
            return argument_entities

        reported_entities: set[str] = set()
        success_value = result.get("success")
        cls._add_values(reported_entities, success_value)
        if isinstance(success_value, (str, list, tuple, set)):
            return reported_entities
        return argument_entities | reported_entities

    @staticmethod
    def _call_has_successful_result(call: object, successful_ids: set[str], successful_names: set[str]) -> bool:
        """Match an assistant tool call to a retained successful result."""
        call_id = str(getattr(call, "id", "") or "")
        if call_id:
            return call_id in successful_ids
        return str(getattr(call, "tool_name", "") or "") in successful_names

    @staticmethod
    def _take_matching_tool_call(result_message: object, calls_by_id: dict[str, object], unmatched_calls: list[object]) -> object | None:
        """Find and consume the assistant call matching a tool result."""
        call_id = str(getattr(result_message, "tool_call_id", "") or "")
        if call_id and call_id in calls_by_id:
            return calls_by_id[call_id]

        tool_name = str(getattr(result_message, "tool_name", "") or "")
        for index, call in enumerate(unmatched_calls):
            if str(getattr(call, "tool_name", "") or "") == tool_name:
                return unmatched_calls.pop(index)
        return None

    def structured_turn_contexts(self, chat_log: conversation.ChatLog) -> list[TurnContext]:
        """Extract language-independent context from retained completed turns."""
        contexts: list[TurnContext] = []
        for turn in self._select_retained_turns(chat_log):
            user_message = next(
                (message for message in turn if isinstance(message, conversation.UserContent)),
                None,
            )
            if user_message is None:
                continue
            entities: set[str] = set()
            tools: set[str] = set()
            areas: set[str] = set()
            domains: set[str] = set()
            device_classes: set[str] = set()
            actions: set[str] = set()
            ambiguous_entities: set[str] = set()
            target_groups: list[TargetGroup] = []

            calls_by_id: dict[str, object] = {}
            unmatched_calls: list[object] = []
            for message in turn:
                if not isinstance(message, conversation.AssistantContent):
                    continue
                for tool_call in getattr(message, "tool_calls", None) or []:
                    call_id = str(getattr(tool_call, "id", "") or "")
                    if call_id:
                        calls_by_id[call_id] = tool_call
                    else:
                        unmatched_calls.append(tool_call)

            for message in turn:
                if isinstance(message, conversation.ToolResultContent):
                    result = getattr(message, "tool_result", None)
                    if not MessageHelper.tool_result_succeeded(result):
                        continue
                    tool_name = str(getattr(message, "tool_name", "") or "")
                    if self._is_semantic_search(tool_name):
                        self._collect_search_candidates(result, ambiguous_entities)
                        continue
                    if tool_name:
                        tools.add(tool_name)
                        actions.add(tool_name)
                    self._collect_tool_result(
                        result,
                        entities,
                        tools,
                        areas,
                        domains,
                        device_classes,
                        ambiguous_entities,
                    )

                    tool_call = self._take_matching_tool_call(
                        message,
                        calls_by_id,
                        unmatched_calls,
                    )
                    if tool_call is None:
                        continue

                    call_tool_name = str(getattr(tool_call, "tool_name", "") or "")
                    arguments = getattr(tool_call, "tool_args", None) or {}
                    if call_tool_name:
                        tools.add(call_tool_name)
                        actions.add(call_tool_name)
                    if not isinstance(arguments, dict):
                        continue

                    group_entities = self._successful_target_entities(arguments, result)
                    group_areas: set[str] = set()
                    group_floors: set[str] = set()
                    group_domains: set[str] = set()
                    group_classes: set[str] = set()
                    self._add_values(group_areas, arguments.get("area"))
                    self._add_values(group_floors, arguments.get("floor"))
                    self._add_values(group_domains, arguments.get("domain"))
                    self._add_values(group_classes, arguments.get("device_class"))
                    entities.update(group_entities)
                    areas.update(group_areas)
                    areas.update(group_floors)
                    domains.update(group_domains)
                    device_classes.update(group_classes)
                    self._add_values(actions, arguments.get("action"))
                    if group_entities or group_areas or group_floors or group_domains or group_classes:
                        target_groups.append(TargetGroup(
                            entities=tuple(sorted(group_entities)),
                            areas=tuple(sorted(group_areas)),
                            floors=tuple(sorted(group_floors)),
                            domains=tuple(sorted(group_domains)),
                            device_classes=tuple(sorted(group_classes)),
                            tool=call_tool_name,
                            action=str(arguments.get("action", "") or call_tool_name),
                        ))

            created_at = getattr(user_message, "created_at", None)
            timestamp = created_at.timestamp() if hasattr(created_at, "timestamp") else None
            text = str(getattr(user_message, "content", "") or "").strip()
            key = "\x1f".join((
                text,
                *sorted(entities),
                *sorted(tools),
                *sorted(areas),
                *sorted(domains),
                *sorted(device_classes),
            ))
            contexts.append(TurnContext(
                key=key,
                text=text,
                entities=tuple(sorted(entities)),
                tools=tuple(sorted(tools)),
                areas=tuple(sorted(areas)),
                domains=tuple(sorted(domains)),
                device_classes=tuple(sorted(device_classes)),
                actions=tuple(sorted(actions)),
                ambiguous_entities=tuple(sorted(ambiguous_entities)),
                target_groups=tuple(target_groups),
                created_at=timestamp,
            ))
        return contexts

    def filter_prompt_history(self, chat_log: conversation.ChatLog, relevant_turn_keys: set[str] | None = None) -> list[conversation.Content]:
        """Return relevant prompt history without failed tool-call pairs."""
        prompt_history: list[conversation.Content] = []
        turns = self._select_retained_turns(chat_log)
        if relevant_turn_keys is not None:
            contexts = self.structured_turn_contexts(chat_log)
            turns = [
                turn
                for turn, context in zip(turns, contexts)
                if context.key in relevant_turn_keys
            ]

        for turn in turns:
            successful_results = [
                message
                for message in turn
                if isinstance(message, conversation.ToolResultContent)
                and MessageHelper.tool_result_succeeded(getattr(message, "tool_result", None))
            ]
            successful_ids = {
                str(getattr(message, "tool_call_id", "") or "")
                for message in successful_results
                if getattr(message, "tool_call_id", None)
            }
            successful_names = {
                str(getattr(message, "tool_name", "") or "")
                for message in successful_results
            }

            for message in turn:
                if isinstance(message, conversation.UserContent):
                    prompt_history.append(message)
                elif isinstance(message, conversation.AssistantContent):
                    original_calls = list(getattr(message, "tool_calls", None) or [])
                    retained_calls = [
                        call for call in original_calls
                        if self._call_has_successful_result(
                            call,
                            successful_ids,
                            successful_names,
                        )
                    ]
                    content = str(getattr(message, "content", "") or "")
                    if not original_calls or retained_calls or content:
                        prompt_history.append(conversation.AssistantContent(
                            agent_id=getattr(message, "agent_id", None),
                            content=content,
                            tool_calls=retained_calls,
                        ))
                elif message in successful_results:
                    prompt_history.append(MessageHelper.compact_tool_result(message))

        return prompt_history

    def build_prompt_history(
        self,
        chat_log: conversation.ChatLog,
        user_input: ConversationInput,
        system_prompt_content: str,
        relevant_turn_keys: set[str] | None = None,
    ) -> list[conversation.Content]:
        """Build model history with system prompt first and current user last."""
        self._message_history = [
            conversation.SystemContent(content=system_prompt_content)
        ]
        self._message_history.extend(
            self.filter_prompt_history(chat_log, relevant_turn_keys)
        )
        self._message_history.append(conversation.UserContent(content=user_input.text))
        return self._message_history

    def append_message(self, message: conversation.Content) -> None:
        """Append content to the active history."""
        self._message_history.append(message)

    def replace_system_prompt(self, content: str) -> None:
        """Replace the active system prompt without retaining stale candidates."""
        if self._message_history and isinstance(self._message_history[0], conversation.SystemContent):
            self._message_history[0] = conversation.SystemContent(content=content)

    def persist_chat_history(self, chat_log: conversation.ChatLog) -> None:
        """Persist the active normalized history to Home Assistant."""
        chat_log.content = self._message_history
