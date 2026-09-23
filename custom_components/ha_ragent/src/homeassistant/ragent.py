from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, List, Tuple

from custom_components.ha_ragent.src.homeassistant.helpers.history_retriever import HistoryRetriever
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from custom_components.ha_ragent.src.logging.timing_logger import TimingLogger
from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import chat_session, intent, llm
from homeassistant.helpers.template import Template
from homeassistant.helpers.llm import LLMContext
from homeassistant.helpers import area_registry as ar, device_registry as dr, floor_registry as fr

from custom_components.ha_ragent.src.homeassistant.helpers.history_manager import HistoryManager
from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import MessageHelper
from custom_components.ha_ragent.src.homeassistant.helpers.tool_helper import ToolHelper
from custom_components.ha_ragent.src.homeassistant.helpers.conversation_retriever import ConversationRetriever
from custom_components.ha_ragent.src.homeassistant.helpers.source_retriever import SourceRetriever
from custom_components.ha_ragent.src.models.retrieval.scheduled_context import ScheduledContext
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.chat.chat_message import ChatMessage
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding

from custom_components.ha_ragent.src.homeassistant.ragent_entity import RAGentEntity
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry
from custom_components.ha_ragent.src.homeassistant.ragent_api import (
    RAGentAugmentedAPIInstance,
    resolve_llm_api_id,
)
from custom_components.ha_ragent.src.models.embedding.device import Device

from custom_components.ha_ragent.src.translation import RAGentTranslations

from custom_components.ha_ragent.src.const import (
    CONF_MIN_DEVICES_TO_EXTRACT,
    CONF_MAX_DEVICES_TO_EXTRACT,
    CONF_MIN_TOOLS_TO_EXTRACT,
    CONF_MAX_TOOLS_TO_EXTRACT,
    CONF_MIN_MEMORIES_TO_EXTRACT,
    CONF_MAX_MEMORIES_TO_EXTRACT,
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_PROMPT,
    CONF_RETRIEVAL_METHOD,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    DOMAIN,
    CONF_ALLOW_QUESTIONS,
    RAGENT_SCHEDULED_REQUEST_PREFIX,
    RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES,
    RETRIEVAL_METHOD_LEXICAL,
    TRANSLATION_PROMPT_SCHEDULED_ACTION,
    TRANSLATION_PROMPT_PERSONA,
    TRANSLATION_PROMPT_AREAS,
    TRANSLATION_PROMPT_DEVICES,
    TRANSLATION_PROMPT_MEMORIES,
    TRANSLATION_PROMPT_RETRIES,
    TRANSLATION_PROMPT_INSTRUCTIONS,
    TRANSLATION_PROMPT_SEARCH_FALLBACK,
    TRANSLATION_ERROR_BACKEND,
    TRANSLATION_ERROR_MAX_RETRIES,
    TRANSLATION_ERROR_LLM_API,
    TRANSLATION_ERROR_NO_SPEECH,
    TRANSLATION_ERROR_TEMPLATE,
    TRANSLATION_ERROR_UNEXPECTED,
    TRANSLATION_ERROR_TOOL_NOT_EXPOSED,
    TRANSLATION_ERROR_TOOL_CALL_PREVIOUSLY_FAILED,
    TRANSLATION_ERROR_TOOL_CALL_ALREADY_EXECUTED,
    TRANSLATION_ERROR_TOOL_CALLING_INACTIVE,
)

from custom_components.ha_ragent.src.utils import get_setting_value

_logger = BaseLogger(__name__)

class RAGent(ConversationEntity, AbstractConversationAgent, RAGentEntity):
    """RAG-based conversation agent for Home Assistant."""
    def __init__(self, hass: HomeAssistant, entry: RAGentConfigEntry, subentry: ConfigSubentry) -> None:
        super().__init__(hass, entry, subentry)

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return RAGentTranslations.supported_languages()

    async def _async_embed_retrieval_text(self, retrieval_text: str) -> list[float] | None:
        """Embed retrieval text and handle backend failures consistently."""
        try:
            embedding = await self.entry.embedder_backend.async_embed_text(dict(self.subentry.data), retrieval_text)
        except Exception as err:
            _logger.log_string(logging.ERROR, f"Error embedding retrieval query: {err}")
            return None

        return embedding or None

    async def _async_build_continuity_context(
        self,
        history_manager: HistoryManager,
        chat_log: conversation.ChatLog,
        query_embedding: QueryEmbedding | None = None,
    ) -> ContinuityContext:
        """Retain bounded recent structured history with semantic signals."""
        contexts = history_manager.structured_turn_contexts(chat_log)
        remember_num = get_setting_value(CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS, self.runtime_options)
        remember_time = get_setting_value(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, self.runtime_options)
        _logger.log_payload("continuity.raw_history", contexts=contexts,
            remember_interactions=remember_num, remember_minutes=remember_time,
        )
        if not remember_num and not remember_time:
            continuity = ContinuityContext()
            _logger.log_payload("continuity.disabled", continuity=continuity,
            )
            return continuity
        current_vector: list[float] = []
        vectors: dict[str, list[float]] = {}
        if contexts and SourceRetriever.retrieval_method(self.runtime_options) != RETRIEVAL_METHOD_LEXICAL:
            current_vector = await query_embedding.get() if query_embedding else []
            embedded = await asyncio.gather(*(
                self._async_embed_retrieval_text(
                    HistoryRetriever.build_retrieval_text(
                        context.to_embedding_text(self.entry.translations),
                    )
                )
                for context in contexts
            ))
            vectors = {
                context.key: vector
                for context, vector in zip(contexts, embedded)
                if vector
            }
        selected = HistoryRetriever.select_history_contexts(
            contexts, vectors, current_vector,
            max_age_seconds=remember_time * 60 if remember_time > 0 else float("inf"),
            limit=remember_num if remember_num > 0 else len(contexts),
        )
        return HistoryRetriever.build_continuity_context(selected)

    async def _async_render_system_prompt(
        self,
        devices: List[Device],
        memories: List[Memory],
        area_name: str | None,
        floor_name: str | None,
        is_scheduled_request: bool = False,
        scheduled_context: ScheduledContext | None = None,
    ) -> str | None:
        """Render the system prompt with retrieved device context."""
        raw_prompt = get_setting_value(CONF_PROMPT, self.runtime_options)

        try:
            prompt_template = Template(self.build_base_prompt_template(self.entry.translations, raw_prompt), self.hass)
            rendered_prompt = prompt_template.async_render({
                "device_list": devices,
                "memory_list": memories,
                "area_list": sorted({device.area_name for device in devices if device.area_name}),
                "area_name": area_name,
                "floor_name": floor_name,
                "max_retries": get_setting_value(CONF_MAX_TOOL_CALL_ITERATIONS, self.runtime_options),
            })

            if is_scheduled_request:
                rendered_prompt += "\n\n" + self.entry.translations.prompt(TRANSLATION_PROMPT_SCHEDULED_ACTION)

            if scheduled_context:
                rendered_prompt += scheduled_context.prompt_context()

            return rendered_prompt
        except Exception as err:
            _logger.log_string(logging.ERROR, f"Error rendering prompt: {err}")
            return None

    @staticmethod
    def _exclude_prohibited_scheduled_request_tools(tool_list: List[LlmTool], scheduled_request: bool) -> List[LlmTool]:
        """Exclude scheduling-management tools from due scheduled actions."""
        if not scheduled_request:
            return tool_list
        
        prohibited = set(RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES)
        return [tool for tool in tool_list if tool.name not in prohibited]

    def _get_current_device_location(self, llm_context: LLMContext, scheduled_context: ScheduledContext | None) -> tuple[ar.AreaEntry | None, fr.FloorEntry | None]:
        area: ar.AreaEntry | None = None
        floor: fr.FloorEntry | None = None

        if llm_context.device_id and not scheduled_context:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(llm_context.device_id)

            if device:
                area_reg = ar.async_get(self.hass)
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    floor_reg = fr.async_get(self.hass)
                    if area.floor_id:
                        floor = floor_reg.async_get_floor(area.floor_id)
        elif scheduled_context:
            area_reg = ar.async_get(self.hass)
            floor_reg = fr.async_get(self.hass)
            if scheduled_context.area:
                area = area_reg.async_get_area_by_name(scheduled_context.area)
            if scheduled_context.floor:
                floor = floor_reg.async_get_floor_by_name(scheduled_context.floor)

        return area, floor

    @staticmethod
    def _candidate_context_from_devices(devices: list[Device]) -> list[dict[str, object]]:
        """Build trusted candidate context for this turn."""
        return [
            {
                "name": device.id,
                "friendly_name": device.friendly_name,
                "aliases": device.aliases,
                "area": device.area_name,
                "floor": device.floor_name,
                "area_aliases": device.area_aliases,
                "floor_aliases": device.floor_aliases,
                "domain": device.domain,
                "device_class": device.device_class,
                "state": device.state,
                "unit_of_measurement": (device.attributes or {}).get(
                    "unit_of_measurement",
                    device.unit_of_measurement,
                ),
            }
            for device in devices
        ]

    async def _async_prompt_model(
        self,
        llm_api: llm.APIInstance,
        user_input: ConversationInput,
        tool_list: List[LlmTool],
        chat_log: conversation.ChatLog,
        history_manager: HistoryManager,
        candidate_context: list[dict[str, object]],
        request_query: str
    ) -> ConversationResult:
        """Process a prompt through the RAGent."""
        timing_logger = TimingLogger(__name__ + ".prompt_model")
        max_tool_call_iterations = get_setting_value(CONF_MAX_TOOL_CALL_ITERATIONS, self.runtime_options)
        tool_helper = ToolHelper(self.hass, tool_list)

        formatted_index = 0
        formatted_messages: list[ChatMessage] = []
                
        failed_signatures: set[str] = set()
        tool_calls_overall: List[Tuple[llm.ToolInput, Any]] = []

        final_model_speech = ""
        active_candidate_context = list(candidate_context)
        
        for idx in range(max_tool_call_iterations):
            formatted_messages.extend(
                MessageHelper.message_to_chat_messages(
                    history_manager.message_history[formatted_index:]
                )
            )
            formatted_index = len(history_manager.message_history)

            tool_calls_in_iteration = []
            executed_signatures_in_iteration: set[str] = set()
            try:
                _logger.log_payload("conversation.llm_request",
                    level=logging.DEBUG,
                    iteration=idx + 1,
                    messages=formatted_messages,
                    tools=[tool.to_tool_dict() for tool in tool_list],
                )

                content_chunks = []
                async for chunk in self.entry.llm_backend.async_send_chat_request(
                    dict(self.subentry.data),
                    formatted_messages,
                    tool_list,
                ):
                    content_chunks.append(chunk)

                assistant_content = "".join(content_chunks)
                timing_logger.log_timed_string(logging.DEBUG, f"LLM iteration {idx + 1} completed")

                tool_calls_in_iteration = tool_helper.parse_tool_calls(assistant_content)
                exposed_tool_names = {tool.name for tool in tool_list}
                tool_calls_in_iteration = [
                    tool_helper.normalize_exposed_tool_call(call, exposed_tool_names) or call
                    for call in tool_calls_in_iteration
                ]
                timing_logger.log_timed_string(logging.DEBUG, "Parsed tool calls from LLM response")
                _logger.log_payload("conversation.llm_response",
                    iteration=idx + 1, raw_response=assistant_content,
                    parsed_tool_calls=tool_calls_in_iteration,
                    exposed_tool_names=sorted(exposed_tool_names),
                )

                message_content = MessageHelper.clean_assistant_content(assistant_content, bool(tool_calls_in_iteration))
                timing_logger.log_timed_string(logging.DEBUG, "Cleaned assistant content from LLM response")

                history_tool_calls = [tool_helper.to_history_tool_call(call) for call in tool_calls_in_iteration]
                timing_logger.log_timed_string(logging.DEBUG, "Converted tool calls to history format")

                message = conversation.AssistantContent(agent_id=user_input.agent_id, content=message_content, tool_calls=history_tool_calls)
                history_manager.append_message(message)
                
                for tool_call in tool_calls_in_iteration:
                    tool_name = tool_call.tool_name
                    call_signature = tool_helper.tool_call_signature(tool_call)
                    timing_logger.log_timed_string(logging.DEBUG, f"Processing tool call: {call_signature}")

                    if tool_name not in exposed_tool_names:
                        history_manager.append_message(
                            MessageHelper.create_tool_failure_message(
                                agent_id=user_input.agent_id,
                                tool_call_id=tool_call.id,
                                tool_name=tool_name,
                                error=ValueError(self.entry.translations.error(TRANSLATION_ERROR_TOOL_NOT_EXPOSED, tool_name=tool_name))
                            )
                        )
                        failed_signatures.add(call_signature)
                        continue

                    if call_signature in failed_signatures:
                        history_manager.append_message(
                            MessageHelper.create_tool_failure_message(
                                agent_id=user_input.agent_id,
                                tool_call_id=tool_call.id,
                                tool_name=tool_name,
                                error=ValueError(self.entry.translations.error(TRANSLATION_ERROR_TOOL_CALL_PREVIOUSLY_FAILED))
                            )
                        )
                        continue

                    if call_signature in executed_signatures_in_iteration:
                        history_manager.append_message(
                            MessageHelper.create_tool_failure_message(
                                agent_id=user_input.agent_id,
                                tool_call_id=tool_call.id,
                                tool_name=tool_name,
                                error=ValueError(self.entry.translations.error(TRANSLATION_ERROR_TOOL_CALL_ALREADY_EXECUTED))
                            )
                        )
                        continue

                    executed_signatures_in_iteration.add(call_signature)

                    if not llm_api:
                        _logger.log_string(logging.INFO, f"LLM API not available, skipping tool execution for tool: {tool_name}")
                        history_manager.append_message(
                            MessageHelper.create_tool_failure_message(
                                agent_id=user_input.agent_id,
                                tool_call_id=tool_call.id,
                                tool_name=tool_name,
                                error=ValueError(self.entry.translations.error(TRANSLATION_ERROR_TOOL_CALLING_INACTIVE))
                            )
                        )
                    else:
                        try:
                            is_custom_api = isinstance(llm_api, RAGentAugmentedAPIInstance)
                            is_search_tool = tool_helper.is_semantic_search_tool(tool_name)
                            is_custom_search_tool = tool_helper.is_scheduled_action_tool(tool_name)
                            sanitized_tool_call = tool_helper.sanitize_tool_call(tool_call, active_candidate_context)
                            
                            if is_custom_api and is_custom_search_tool:
                                llm_api.set_scheduling_context(request_query, formatted_messages, active_candidate_context)

                            tool_result = await llm_api.async_call_tool(sanitized_tool_call)
                            parsed_tool_result = tool_helper.parse_tool_results(tool_result)
                            tool_succeeded = MessageHelper.tool_result_succeeded(parsed_tool_result)

                            if tool_succeeded and is_search_tool:
                                discovered_candidates = tool_helper.candidate_devices(parsed_tool_result)
                                if discovered_candidates:
                                    active_candidate_context = tool_helper.merge_candidates(active_candidate_context, discovered_candidates)
                                    if is_custom_api:
                                        llm_api.refresh_search_candidates(active_candidate_context)
                            elif tool_succeeded:
                                if is_custom_api:
                                    llm_api.refresh_search_candidates(active_candidate_context)
                                tool_calls_overall.append((tool_call, parsed_tool_result))

                            history_manager.append_message(
                                MessageHelper.create_tool_result_message(
                                    agent_id=user_input.agent_id,
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_name,
                                    result=parsed_tool_result
                                )
                            )
                        except Exception as tool_err:
                            _logger.log_string(logging.ERROR, f"Error executing tool {tool_name}: {tool_err}")
                            tool_result_msg = MessageHelper.create_tool_failure_message(
                                agent_id=user_input.agent_id,
                                tool_call_id=tool_call.id,
                                tool_name=tool_name,
                                error=tool_err,
                            )
                            history_manager.append_message(tool_result_msg)
                            failed_signatures.add(call_signature)

            except Exception as err:
                _logger.log_string(logging.ERROR, f"There was a problem talking to the backend: {err}")
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, self.entry.translations.error(TRANSLATION_ERROR_BACKEND))
                return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

            history_manager.persist_chat_history(chat_log)

            if not tool_calls_in_iteration:
                final_model_speech = message_content.strip()
                break

            if idx + 1 == max_tool_call_iterations:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, self.entry.translations.error(TRANSLATION_ERROR_MAX_RETRIES))
                return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
            
        intent_response = intent.IntentResponse(language=user_input.language)

        continue_conversation = False
        if final_model_speech:
            intent_response.async_set_speech(final_model_speech)
            has_question = final_model_speech.endswith(("?", "\uff1f"))
            continue_conversation = get_setting_value(CONF_ALLOW_QUESTIONS, self.runtime_options) and has_question
        else:
            intent_response.async_set_speech(self.entry.translations.error(TRANSLATION_ERROR_NO_SPEECH))

        _logger.log_payload("conversation.final_result",
            conversation_id=user_input.conversation_id,
            final_model_speech=final_model_speech,
            tool_calls=tool_calls_overall,
            continue_conversation=continue_conversation,
        )
        return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id, continue_conversation=continue_conversation)
        

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process the user request"""
        timing_logger = TimingLogger(__name__)

        is_scheduled_request = user_input.text.startswith(RAGENT_SCHEDULED_REQUEST_PREFIX)
        if is_scheduled_request:
            user_input.text = user_input.text.removeprefix(RAGENT_SCHEDULED_REQUEST_PREFIX)

        try:
            scheduled_context = None
            if is_scheduled_request:
                user_input.text, scheduled_context = ScheduledContext.restore(self.hass, self.subentry_id, user_input.agent_id, user_input.text)
            llm_context = user_input.as_llm_context(DOMAIN)
            timing_logger.log_timed_string(level=logging.DEBUG, message="Context prepared")

            with (
                chat_session.async_get_chat_session(self.hass, user_input.conversation_id) as session,
                conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
            ):
                llm_api: llm.APIInstance | None = None

                try:
                    llm_api = await llm.async_get_api(self.hass, resolve_llm_api_id(self.runtime_options[CONF_LLM_HASS_API]), llm_context=llm_context,)
                    if isinstance(llm_api, RAGentAugmentedAPIInstance):
                        llm_api.set_conversation_agent_id(user_input.agent_id)
                        llm_api.set_search_scope(self.entry_id,self.subentry_id)
                except HomeAssistantError as err:
                    _logger.log_string(logging.ERROR, f"Error getting LLM API: {err}")
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, self.entry.translations.error(TRANSLATION_ERROR_LLM_API))
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                timing_logger.log_timed_string(level=logging.DEBUG, message="LLM API setup")
                    
                chat_log.llm_api = llm_api
                history_manager = HistoryManager(runtime_options=self.runtime_options)

                retrieval_method = get_setting_value(CONF_RETRIEVAL_METHOD, self.runtime_options)
                min_memories = get_setting_value(CONF_MIN_MEMORIES_TO_EXTRACT, self.runtime_options)
                max_memories = get_setting_value(CONF_MAX_MEMORIES_TO_EXTRACT, self.runtime_options)
                min_devices = get_setting_value(CONF_MIN_DEVICES_TO_EXTRACT, self.runtime_options)
                max_devices = get_setting_value(CONF_MAX_DEVICES_TO_EXTRACT, self.runtime_options)
                min_tools = get_setting_value(CONF_MIN_TOOLS_TO_EXTRACT, self.runtime_options)
                max_tools = get_setting_value(CONF_MAX_TOOLS_TO_EXTRACT, self.runtime_options)

                requires_embedding = retrieval_method != RETRIEVAL_METHOD_LEXICAL or max_memories > 0
                current_area, current_floor = self._get_current_device_location(llm_context, scheduled_context)
                current_area_name = current_area.name if current_area else ""
                current_floor_name = current_floor.name if current_floor else ""

                retriever = ConversationRetriever(self.hass, self.entry, self.entry_id, self.subentry_id, self.subentry)
                retrieval_query = scheduled_context.retrieval_query(user_input.text) if scheduled_context else user_input.text

                query_embedding: QueryEmbedding | None = None
                if requires_embedding:
                    query_embedding = QueryEmbedding(
                        lambda: self._async_embed_retrieval_text(
                            HistoryRetriever.build_retrieval_text(retrieval_query)
                        )
                    )

                continuity: ContinuityContext | None = None
                if not is_scheduled_request:
                    continuity = await self._async_build_continuity_context(history_manager, chat_log, query_embedding)

                _logger.log_payload(
                    event="conversation.retrieval_request",
                    conversation_id=user_input.conversation_id, 
                    user_text=user_input.text, 
                    retrieval_query=retrieval_query, 
                    retrieval_method=retrieval_method, 
                    configured_device_range=(
                        get_setting_value(CONF_MIN_DEVICES_TO_EXTRACT, self.runtime_options),
                        get_setting_value(CONF_MAX_DEVICES_TO_EXTRACT, self.runtime_options)
                    ),
                    configured_tool_range=(
                        get_setting_value(CONF_MIN_TOOLS_TO_EXTRACT, self.runtime_options),
                        get_setting_value(CONF_MAX_TOOLS_TO_EXTRACT, self.runtime_options)
                    ),
                    configured_memory_range=(min_memories, max_memories),
                    current_area_name=current_area_name, 
                    current_floor_name=current_floor_name,
                    is_scheduled_request=is_scheduled_request, 
                    continuity=continuity
                )

                async with asyncio.TaskGroup() as retrieval_tasks:
                    memory_task = retrieval_tasks.create_task(
                        retriever.async_retrieve_memories(
                            query_embedding,
                            minimum=min_memories,
                            maximum=max_memories,
                        )
                    ) if max_memories > 0 else None

                    device_task = retrieval_tasks.create_task(
                            retriever.async_retrieve_devices(
                                query_embedding,
                                retrieval_query,
                                minimum=min_devices,
                                maximum=max_devices,
                                continuity=continuity,
                                retrieval_method=retrieval_method,
                                current_area=current_area_name,
                                current_floor=current_floor_name,
                            )
                    ) if max_devices > 0 else None

                    tool_task = retrieval_tasks.create_task(
                        retriever.async_retrieve_tools(
                            query_embedding,
                            retrieval_query,
                            minimum=min_tools,
                            maximum=max_tools,
                            retrieval_method=retrieval_method,
                            llm_api=llm_api,
                        )
                    ) if llm_api and max_tools > 0 else None
                
                retrieved_memories = memory_task.result() if memory_task else []
                retrieved_devices = device_task.result() if device_task else []
                retrieved_tools = tool_task.result() if tool_task else []
                retrieved_tools = self._exclude_prohibited_scheduled_request_tools(retrieved_tools, is_scheduled_request)

                timing_logger.log_timed_string(level=logging.DEBUG, message="Retrieval completed")
                _logger.log_payload(
                    event="conversation.retrieval_result",
                    conversation_id=user_input.conversation_id,
                    retrieval_query=retrieval_query,
                    devices=retrieved_devices, tools=retrieved_tools,
                    memories=retrieved_memories, continuity=continuity,
                )

                device_list = []
                for device in retrieved_devices:
                    st = self.hass.states.get(device.id)
                    if st is None:
                        continue

                    device.state = st.state
                    attributes = Device.clean_attributes(st.attributes)
                    if "light" in (device.domain or []):
                        brightness = attributes.pop("brightness", None)
                        if isinstance(brightness, (int, float)) and not isinstance(brightness, bool):
                            attributes["brightness_percent"] = round(
                                max(0.0, min(255.0, float(brightness))) / 255.0 * 100,
                            )
                    device.attributes = attributes
                    device_list.append(device)

                candidate_context = self._candidate_context_from_devices(device_list)
                _logger.log_payload("conversation.prompt_context",
                    conversation_id=user_input.conversation_id,
                    devices=device_list, memories=retrieved_memories,
                    continuity=continuity, candidate_context=candidate_context,
                )
                if isinstance(llm_api, RAGentAugmentedAPIInstance):
                    llm_api.set_search_context(
                        latest_request=retrieval_query,
                        area=current_area_name,
                        floor=current_floor_name,
                        candidates=candidate_context,
                    )

                system_prompt_content = await self._async_render_system_prompt(
                    device_list,
                    retrieved_memories,
                    area_name=current_area_name,
                    floor_name=current_floor_name,
                    is_scheduled_request=is_scheduled_request,
                    scheduled_context=scheduled_context,
                )
                timing_logger.log_timed_string(level=logging.DEBUG, message="System prompt rendering")

                if not system_prompt_content:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, self.entry.translations.error(TRANSLATION_ERROR_TEMPLATE))
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                history_manager.build_prompt_history(
                    chat_log,
                    user_input,
                    system_prompt_content,
                    relevant_turn_keys=continuity.selected_turn_keys,
                )

                _logger.log_payload(
                    event="conversation.size_breakdown",
                    data={
                        "system_prompt": len(system_prompt_content),
                        "devices": len(json.dumps(
                            [device.to_dict() for device in device_list],
                            ensure_ascii=False,
                            default=str,
                        )),
                        "memories": len(json.dumps(
                            [memory.to_dict() for memory in retrieved_memories],
                            ensure_ascii=False,
                            default=str,
                        )),
                        "continuity_turns": len(continuity.selected_turn_keys),
                        "history": sum(
                            len(str(getattr(message, "content", "") or ""))
                            for message in history_manager.message_history[1:-1]
                        ),
                    },
                )

                result = await self._async_prompt_model(
                    llm_api,
                    user_input,
                    retrieved_tools,
                    chat_log,
                    history_manager,
                    candidate_context,
                    retrieval_query
                )
                timing_logger.log_timed_string(level=logging.DEBUG, message="Model and tool processing")
                return result
        except Exception as err:
            _logger.log_string(logging.ERROR, f"Unexpected error in async_process: {err}")
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, self.entry.translations.error(TRANSLATION_ERROR_UNEXPECTED))
            return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

    @staticmethod
    def build_base_prompt_template(translations: RAGentTranslations, prompt_template: str) -> str:
        """Build a prompt template from the selected translation file."""
        prompt_template = prompt_template.replace("<persona_prompt>", translations.prompt(TRANSLATION_PROMPT_PERSONA))
        prompt_template = prompt_template.replace("<area_prompt>", translations.prompt(TRANSLATION_PROMPT_AREAS))
        prompt_template = prompt_template.replace("<devices_prompt>", translations.prompt(TRANSLATION_PROMPT_DEVICES))
        prompt_template = prompt_template.replace("<memories_context_prompt>", translations.prompt(TRANSLATION_PROMPT_MEMORIES))
        prompt_template = prompt_template.replace("<max_retries_prompt>", translations.prompt(TRANSLATION_PROMPT_RETRIES))
        prompt_template = prompt_template.replace("<instruction_prompt>", translations.prompt(TRANSLATION_PROMPT_INSTRUCTIONS))
        prompt_template = prompt_template.replace("<search_fallback_prompt>", translations.prompt(TRANSLATION_PROMPT_SEARCH_FALLBACK))
        return prompt_template
