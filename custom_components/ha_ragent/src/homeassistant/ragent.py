from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, List, Tuple

from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import chat_session, intent, llm
from homeassistant.helpers.template import Template
from homeassistant.helpers.llm import LLMContext
from homeassistant.helpers import area_registry as ar, device_registry as dr, floor_registry as fr
from probatio import to_openapi

from custom_components.ha_ragent.src.homeassistant.helpers.history_manager import HistoryManager
from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import MessageHelper
from custom_components.ha_ragent.src.homeassistant.helpers.tool_helper import ToolHelper
from custom_components.ha_ragent.src.homeassistant.helpers.memory_manager import MemoryManager
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor import ToolExtractor
from custom_components.ha_ragent.src.models.retrieval.scheduled_context import ScheduledContext
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
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
    CONF_NUM_MEMORIES_TO_EXTRACT,
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_PROMPT,
    DEFAULT_PROMPT,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    DOMAIN,
    CONF_ALLOW_QUESTIONS,
    RAGENT_PREFIXED_REQUIRED_TOOL_NAMES,
    RAGENT_SCHEDULED_REQUEST_PREFIX,
    RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES,
    STARTUP_EMBEDDING_RUNNING_FLAG,
    RETRIEVAL_METHOD_LEXICAL,
    RAGENT_PLANNED_ACTION_TOOL_NAME,
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
)

from custom_components.ha_ragent.src.debug import log_debug_payload
from custom_components.ha_ragent.src.utils import get_entry_language, get_setting_value

_logger = logging.getLogger(__name__)

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
        return MATCH_ALL

    async def _async_embed_retrieval_text(self, retrieval_text: str) -> list[float] | None:
        """Embed retrieval text and handle backend failures consistently."""
        try:
            embedding = await self.entry.embedder_backend.async_embed_text(dict(self.subentry.data), retrieval_text)
        except Exception as err:
            _logger.error(f"Error embedding retrieval query: {err}", exc_info=True)
            return None

        return embedding or None

    async def _async_build_continuity_context(
        self,
        history_manager: HistoryManager,
        chat_log: conversation.ChatLog,
    ) -> ContinuityContext:
        """Retain bounded recent structured history without embedding turns."""
        contexts = history_manager.structured_turn_contexts(chat_log)
        remember_num = get_setting_value(CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS, self.runtime_options)
        remember_time = get_setting_value(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, self.runtime_options)
        log_debug_payload(
            _logger, "continuity.raw_history", contexts=contexts,
            remember_interactions=remember_num, remember_minutes=remember_time,
        )
        if not remember_num and not remember_time:
            continuity = ContinuityContext()
            log_debug_payload(
                _logger, "continuity.disabled", continuity=continuity,
            )
            return continuity
        selected = RetrievalHelper.select_history_contexts(
            contexts,
            {},
            [],
            max_age_seconds=remember_time * 60 if remember_time > 0 else float("inf"),
            limit=remember_num if remember_num > 0 else len(contexts),
        )
        return RetrievalHelper.build_continuity_context(selected)

    @staticmethod
    def _configured_retrieval_ranges(
        runtime_options: dict[str, object],
    ) -> tuple[int, int, int, int]:
        """Return normalized device/tool minimum and maximum exposure limits."""
        requested_min_devices = max(0, int(get_setting_value(
            CONF_MIN_DEVICES_TO_EXTRACT, runtime_options,
        )))
        requested_min_tools = max(0, int(get_setting_value(
            CONF_MIN_TOOLS_TO_EXTRACT, runtime_options,
        )))
        max_devices = max(0, int(get_setting_value(
            CONF_MAX_DEVICES_TO_EXTRACT, runtime_options,
        )))
        max_tools = max(0, int(get_setting_value(
            CONF_MAX_TOOLS_TO_EXTRACT, runtime_options,
        )))
        return (
            requested_min_devices,
            max_devices,
            requested_min_tools,
            max_tools,
        )

    async def _async_retrieve_devices(
        self,
        query_embedding: List[float] | QueryEmbedding,
        query: str,
        n_devices: int,
        continuity: ContinuityContext,
        current_area: str = "",
        current_floor: str = "",
        max_devices: int | None = None,
    ) -> List[Device]:
        """Retrieve relevant devices from vector database based on query embedding."""
        exposure_limit = max(n_devices, max_devices or n_devices)
        if exposure_limit <= 0:
            return []
        collection_name = f"devices_{self.subentry_id}"
        try:
            candidate_limit = RetrievalHelper.adaptive_candidate_limit(exposure_limit)
            options = {**self.entry.options, **self.subentry.data}
            scored_devices, all_devices = await RetrievalHelper.async_retrieve_sources(
                self.entry.vector_db_backend, DeviceEmbedding, options,
                collection_name, query_embedding, candidate_limit, query=query,
            )
        except Exception as e:
            _logger.error(f"Error retrieving devices from vector DB: {e}", exc_info=True)
            return []

        def device_text_parts(device: Device) -> tuple[object, ...]:
            return (
                device.id,
                device.friendly_name,
                *(device.aliases or []),
                device.area_name,
                device.floor_name,
                *(device.area_aliases or []),
                *(device.floor_aliases or []),
                *(device.domain or []),
                device.device_class,
                *(device.device_labels or []),
            )

        def device_identity_score(device: Device) -> float:
            return 2.0 * RetrievalHelper.device_target_score(query, device)

        def device_location_score(device: Device) -> float:
            return RetrievalHelper.trusted_location_score(
                device, current_area, current_floor,
            )

        def device_taxonomy_score(device: Device) -> float:
            return RetrievalHelper.field_match_score(
                query, (*(device.domain or ()), device.device_class),
            )

        def device_metadata_score(device: Device) -> float:
            return device_identity_score(device) + 0.5 * device_location_score(device)

        ranked_devices = await asyncio.to_thread(
            RetrievalHelper.rank_scored_candidates,
            scored_devices,
            all_devices,
            query,
            lambda device: device.id,
            device_text_parts,
            candidate_limit,
            metadata_score=device_metadata_score,
            continuity_score=continuity.device_score,
            preserve_score=continuity.successful_target_score,
            trim_confident=False,
        )
        confidence = RetrievalHelper.device_search_confidence(
            ranked_devices,
            scored_devices,
            query=query,
            key=lambda device: device.id,
            text_parts=device_text_parts,
            structured_scores={
                "identity_metadata": device_identity_score,
                "area_floor": device_location_score,
                "domain_device_class": device_taxonomy_score,
                "entity_continuity": continuity.entity_score,
                "area_continuity": continuity.area_score,
                "taxonomy_continuity": continuity.taxonomy_score,
            },
            confirmed_score=continuity.successful_target_score,
        )
        return RetrievalHelper.select_device_candidates(
            query, ranked_devices, n_devices, max_devices, confidence,
        )

    async def _async_retrieve_tools(
        self,
        query_embedding: List[float] | QueryEmbedding,
        query: str,
        n_tools: int,
        continuity: ContinuityContext,
        devices: list[Device] | None = None,
        max_tools: int | None = None,
    ) -> List[LlmTool]:
        """Retrieve relevant tools from vector database based on query embedding."""
        exposure_limit = max(n_tools, max_tools or n_tools)
        if exposure_limit <= 0:
            return []
        collection_name = f"tools_{self.subentry_id}"
        try:
            candidate_limit = RetrievalHelper.adaptive_candidate_limit(exposure_limit)
            options = {**self.entry.options, **self.subentry.data}
            scored_tools, all_tools = await RetrievalHelper.async_retrieve_sources(
                self.entry.vector_db_backend, LlmToolEmbedding, options,
                collection_name, query_embedding, candidate_limit, query=query,
            )
        except Exception as e:
            _logger.error(f"Error retrieving tools from vector DB: {e}", exc_info=True)
            return []

        ranked_tools = await asyncio.to_thread(
            RetrievalHelper.rank_tool_candidates,
            scored_tools,
            all_tools,
            query,
            [],
            max(exposure_limit, RetrievalHelper.expanded_tool_limit(exposure_limit)),
            continuity_score=continuity.tool_score,
        )
        ranked_tools = RetrievalHelper.rerank_tools_for_devices(
            ranked_tools, devices or [], len(ranked_tools),
        )
        if max_tools is None:
            return ranked_tools[:n_tools]
        confidence = RetrievalHelper.tool_search_confidence_details(
            ranked_tools, query, devices or [],
            vector_results=scored_tools,
            continuity_score=continuity.tool_score,
        )
        limit = RetrievalHelper.confidence_limit(confidence.level, n_tools, max_tools)
        log_debug_payload(
            _logger, "retrieval.tool_exposure",
            confidence=confidence.level,
            confidence_reason=confidence.reason,
            top_score=confidence.top_score,
            second_score=confidence.second_score,
            margin=confidence.margin,
            ratio=confidence.ratio,
            selected_candidate_count=min(limit, len(ranked_tools)),
        )
        return ranked_tools[:limit]

    async def _async_retrieve_memories(self, query_embedding: List[float] | QueryEmbedding, n_memories: int) -> List[Memory]:
        """Retrieve relevant persistent memories for this agent."""
        if n_memories <= 0 or not query_embedding:
            return []
        try:
            return await MemoryManager(
                self.hass,
                self.entry_id,
                self.subentry_id,
            ).async_recall(query_embedding, n_memories)
        except Exception as err:
            _logger.error(f"Error retrieving memories from vector DB: {err}", exc_info=True)
            return []

    async def _async_render_system_prompt(
        self,
        devices: List[Device],
        memories: List[Memory],
        area: ar.AreaEntry,
        floor: fr.FloorEntry,
        continuity: ContinuityContext | None = None,
        scheduled_request: bool = False,
        scheduled_context: ScheduledContext | None = None,
    ) -> str | None:
        """Render the system prompt with retrieved device context."""
        raw_prompt = get_setting_value(CONF_PROMPT, self.runtime_options) or DEFAULT_PROMPT
        language = get_entry_language(self.entry)

        try:
            template_key = (raw_prompt, language)
            if getattr(self, "_prompt_template_key", None) != template_key:
                self._prompt_template = Template(
                    self.build_base_prompt_template(self.entry.translations, raw_prompt), self.hass,
                )
                self._prompt_template_key = template_key
            rendered_prompt = self._prompt_template.async_render({
                "device_list": devices,
                "memory_list": memories,
                "area_list": sorted({device.area_name for device in devices if device.area_name}),
                "area_name": scheduled_context.area if scheduled_context else (area.name if area else None),
                "floor_name": scheduled_context.floor if scheduled_context else (floor.name if floor else None),
                "max_retries": get_setting_value(CONF_MAX_TOOL_CALL_ITERATIONS, self.runtime_options),
            })
            if scheduled_request:
                rendered_prompt += "\n\n" + self.entry.translations.prompt(
                    TRANSLATION_PROMPT_SCHEDULED_ACTION,
                )
            if scheduled_context:
                rendered_prompt += scheduled_context.prompt_context()
            return rendered_prompt
        except Exception as err:
            _logger.error(f"Error rendering prompt: {err}", exc_info=True)
            return None

    def _convert_api_tool(self, api_tool: Any, llm_api: llm.APIInstance | None) -> LlmTool | None:
        """Convert a Home Assistant LLM tool into the local tool schema."""
        tool_name = getattr(api_tool, "name", None)
        if not tool_name:
            return None

        parameters = {}
        if hasattr(api_tool, "parameters") and api_tool.parameters:
            try:
                parameters = to_openapi(api_tool.parameters, custom_serializer=llm_api.custom_serializer if llm_api else None)
                if not isinstance(parameters, dict):
                    _logger.warning(f"Could not convert parameters for tool {tool_name}: converter returned {type(parameters).__name__}")
                    parameters = {}
            except Exception as err:
                _logger.warning(f"Could not convert parameters for tool {tool_name}: {err}")
                parameters = {}

        return LlmTool(
            name=tool_name,
            description=getattr(api_tool, "description", ""),
            parameters=parameters,
            metadata=ToolExtractor.extract_tool_metadata(api_tool, parameters),
        )

    def _ensure_required_tools_exposed(self, tool_list: List[LlmTool], llm_api: llm.APIInstance | None) -> List[LlmTool]:
        """Expose required tools before tools selected by semantic retrieval."""
        if not llm_api or not hasattr(llm_api, "tools"):
            return tool_list

        required_names = set(RAGENT_PREFIXED_REQUIRED_TOOL_NAMES)
        required_tools = [tool for tool in tool_list if tool.name in required_names]
        searched_tools = [tool for tool in tool_list if tool.name not in required_names]
        seen_tool_names = {tool.name for tool in tool_list}

        for api_tool in llm_api.tools:
            tool_name = getattr(api_tool, "name", None)
            if tool_name not in required_names or tool_name in seen_tool_names:
                continue

            converted_tool = self._convert_api_tool(api_tool, llm_api)
            if converted_tool:
                required_tools.append(converted_tool)
                seen_tool_names.add(tool_name)

        return [*required_tools, *searched_tools]

    @staticmethod
    def _exclude_prohibited_scheduled_request_tools(tool_list: List[LlmTool], scheduled_request: bool) -> List[LlmTool]:
        """Exclude scheduling-management tools from due scheduled actions."""
        if not scheduled_request:
            return tool_list
        
        prohibited = set(RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES)
        return [tool for tool in tool_list if tool.name not in prohibited]

    def _get_current_device_location(self, llm_context: LLMContext) -> tuple[ar.AreaEntry | None, fr.FloorEntry | None]:
        area: ar.AreaEntry | None = None
        floor: fr.FloorEntry | None = None
        if llm_context.device_id:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(llm_context.device_id)

            if device:
                area_reg = ar.async_get(self.hass)
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    floor_reg = fr.async_get(self.hass)
                    if area.floor_id:
                        floor = floor_reg.async_get_floor(area.floor_id)

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
        request_query: str,
        scheduled_request: bool = False
    ) -> ConversationResult:
        """Process a prompt through the RAGent."""
        tool_helper = ToolHelper(self.hass)
        max_tool_call_iterations = max(1, get_setting_value(CONF_MAX_TOOL_CALL_ITERATIONS, self.runtime_options))

        tool_calls_overall: List[Tuple[llm.ToolInput, Any]] = []
        final_model_speech = ""
        tool_metadata_dict = {
            tool.name: tool.metadata
            for tool in tool_list
            if tool.metadata
        }
        formatted_messages: list[ChatMessage] = []
        formatted_index = 0
        active_candidate_context = list(candidate_context)
        failed_signatures: set[str] = set()
        tools_by_name = {tool.name: tool for tool in tool_list}

        for idx in range(max_tool_call_iterations):
            iteration_start = time.perf_counter()
            formatted_messages.extend(
                MessageHelper.message_to_chat_messages(
                    history_manager.message_history[formatted_index:]
                )
            )
            formatted_index = len(history_manager.message_history)

            if _logger.isEnabledFor(logging.DEBUG):
                message_chars = sum(
                    len(json.dumps(message, ensure_ascii=False, default=str))
                    for message in formatted_messages
                )
                tool_schema_chars = sum(
                    len(json.dumps(tool.parameters, ensure_ascii=False, default=str))
                    + len(tool.name) + len(tool.description or "")
                    for tool in tool_list
                )
                _logger.debug(f"RAGent prompt size (iteration {idx + 1}): messages={message_chars} chars, tools={tool_schema_chars} chars, tool_count={len(tool_list)}")

            tool_calls_in_iteration = []
            try:
                _logger.debug(f"Sending prompt to LLM (Iteration {idx + 1}/{max_tool_call_iterations}).")
                if _logger.isEnabledFor(logging.DEBUG):
                    _logger.debug(f"Full messages sent to the LLM:\n{json.dumps(formatted_messages, ensure_ascii=False, indent=2, default=str)}")
                
                content_chunks = []
                async for chunk in self.entry.llm_backend.async_send_chat_request(
                    dict(self.subentry.data),
                    formatted_messages,
                    tool_list,
                ):
                    content_chunks.append(chunk)
                assistant_content = "".join(content_chunks)
                _logger.debug(f"RAGent timing: LLM iteration {idx + 1}: {time.perf_counter() - iteration_start:.3f}s")

                _logger.debug(f"RAW LLM response: {assistant_content}")
                
                helper_start = time.perf_counter()
                tool_calls_in_iteration = tool_helper.parse_tool_calls(assistant_content, tool_metadata_dict)
                exposed_tool_names = {tool.name for tool in tool_list}
                tool_calls_in_iteration = [
                    tool_helper.normalize_exposed_tool_call(
                        call,
                        exposed_tool_names,
                        tool_metadata_dict,
                    ) or call
                    for call in tool_calls_in_iteration
                ]
                log_debug_payload(
                    _logger, "conversation.llm_response",
                    iteration=idx + 1, raw_response=assistant_content,
                    parsed_tool_calls=tool_calls_in_iteration,
                    exposed_tool_names=sorted(exposed_tool_names),
                    tool_metadata=tool_metadata_dict,
                )
                _logger.debug(f"RAGent timing: parse_tool_calls: {time.perf_counter() - helper_start:.3f}s")

                helper_start = time.perf_counter()
                message_content = MessageHelper.clean_assistant_content(
                    assistant_content,
                    bool(tool_calls_in_iteration),
                )
                _logger.debug(f"RAGent timing: clean_assistant_content: {time.perf_counter() - helper_start:.3f}s")

                helper_start = time.perf_counter()
                history_tool_calls = [tool_helper.to_history_tool_call(call) for call in tool_calls_in_iteration]
                _logger.debug(f"RAGent timing: to_history_tool_call: {time.perf_counter() - helper_start:.3f}s")

                message = conversation.AssistantContent(
                    agent_id=user_input.agent_id,
                    content=message_content,
                    tool_calls=history_tool_calls
                )
                history_manager.append_message(message)
                
                if tool_calls_in_iteration:
                    for tool_call in tool_calls_in_iteration:
                        tool_name = tool_call.tool_name
                        call_signature = tool_helper.tool_call_signature(tool_call)

                        if tool_name not in exposed_tool_names:
                            error = ValueError(
                                f"Tool {tool_name} was not exposed. Use only a tool from the native tool list."
                            )
                            history_manager.append_message(
                                MessageHelper.create_tool_failure_message(
                                    agent_id=user_input.agent_id,
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_name,
                                    error=error,
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
                                    error=ValueError(
                                        "This exact call already failed or contradicted the requested capability. "
                                        "Rediscover a compatible capability before retrying."
                                    ),
                                )
                            )
                            continue

                        requested_capabilities = (
                            llm_api.requested_capabilities()
                            if isinstance(llm_api, RAGentAugmentedAPIInstance)
                            else []
                        )
                        selected_tool = tools_by_name.get(tool_name)
                        capability_scores = [
                            RetrievalHelper.tool_capability_compatibility(selected_tool, capability)
                            for capability in requested_capabilities
                        ] if selected_tool else []
                        if capability_scores and max(capability_scores) < 0:
                            rediscovery_result = await llm_api.async_rediscover_capabilities(
                                requested_capabilities,
                            )
                            existing_names = {tool.name for tool in tool_list}
                            discovered_tools = tool_helper.discovered_tools(
                                rediscovery_result, existing_names,
                            )
                            tool_list.extend(discovered_tools)
                            tools_by_name.update({tool.name: tool for tool in discovered_tools})
                            exposed_tool_names.update(tool.name for tool in discovered_tools)
                            tool_metadata_dict.update({
                                tool.name: tool.metadata
                                for tool in discovered_tools
                                if tool.metadata
                            })
                            history_manager.append_message(conversation.ToolResultContent(
                                agent_id=user_input.agent_id,
                                tool_call_id=tool_call.id,
                                tool_name=tool_name,
                                tool_result={
                                    "success": False,
                                    "execution_status": {
                                        "requested_capabilities": requested_capabilities,
                                        "executed_capability": getattr(selected_tool, "canonical_action", ""),
                                        "target": tool_helper.successful_target_names(tool_call, {}),
                                        "tool_succeeded": False,
                                        "fulfillment_status": "capability_mismatch",
                                        "rediscovery_required": True,
                                        "rediscovered_tools": [tool.name for tool in discovered_tools],
                                    },
                                },
                            ))
                            failed_signatures.add(call_signature)
                            continue

                        try:
                            if llm_api:
                                execution_call = tool_helper.to_home_assistant_tool_call(
                                    tool_call, tool_metadata_dict.get(tool_name),
                                )
                                if (isinstance(llm_api, RAGentAugmentedAPIInstance)
                                    and tool_name.rsplit("__", 1)[-1] == RAGENT_PLANNED_ACTION_TOOL_NAME):
                                    llm_api.set_scheduling_context(
                                        request_query, formatted_messages, active_candidate_context,
                                    )
                                tool_start = time.perf_counter()
                                tool_result = await llm_api.async_call_tool(execution_call)
                                parsed_tool_result = tool_helper.parse_tool_results(tool_result)
                                tool_succeeded = MessageHelper.tool_result_succeeded(parsed_tool_result)
                                if tool_succeeded and tool_helper.is_semantic_search_tool(tool_name):
                                    existing_names = {tool.name for tool in tool_list}
                                    discovered_tools = tool_helper.discovered_tools(parsed_tool_result, existing_names)
                                    tool_list.extend(discovered_tools)
                                    tools_by_name.update({tool.name: tool for tool in discovered_tools})
                                    exposed_tool_names.update(tool.name for tool in discovered_tools)
                                    tool_metadata_dict.update(
                                        {
                                            tool.name: tool.metadata
                                            for tool in discovered_tools
                                            if tool.metadata
                                        }
                                    )
                                    discovered_candidates = tool_helper.candidate_devices(parsed_tool_result)
                                    if discovered_candidates:
                                        active_candidate_context = tool_helper.merge_candidates(active_candidate_context, discovered_candidates)
                                        if isinstance(llm_api, RAGentAugmentedAPIInstance):
                                            llm_api.refresh_search_candidates(active_candidate_context)
                                elif tool_succeeded:
                                    if isinstance(llm_api, RAGentAugmentedAPIInstance):
                                        llm_api.refresh_search_candidates(active_candidate_context)
                                    tool_calls_overall.append((tool_call, parsed_tool_result))
                                stored_tool_result = MessageHelper.compact_tool_result_value(
                                    tool_name,
                                    parsed_tool_result,
                                )
                                targets = tool_helper.successful_target_names(tool_call, parsed_tool_result)
                                fulfillment_status = "failed"
                                rediscovery_required = not tool_succeeded
                                observed_states: dict[str, str | None] = {}
                                if tool_succeeded:
                                    has_declared_action = bool(
                                        getattr(selected_tool, "canonical_action", "")
                                    )
                                    fulfillment_status = (
                                        "unverified"
                                        if requested_capabilities or has_declared_action
                                        else "not_evaluated"
                                    )
                                    expected_states = set(
                                        getattr(getattr(selected_tool, "metadata", None), "expected_states", ()) or ()
                                    )
                                    if targets:
                                        current_states = [self.hass.states.get(target) for target in targets]
                                        observed_states = {
                                            target: state.state if state is not None else None
                                            for target, state in zip(targets, current_states)
                                        }
                                    if expected_states and targets:
                                        if all(
                                            state is not None and str(state.state).casefold() in expected_states
                                            for state in current_states
                                        ):
                                            fulfillment_status = "verified"
                                        else:
                                            fulfillment_status = "state_mismatch"
                                            rediscovery_required = True
                                execution_status = {
                                    "requested_capabilities": requested_capabilities,
                                    "executed_capability": getattr(selected_tool, "canonical_action", ""),
                                    "target": targets,
                                    "tool_succeeded": tool_succeeded,
                                    "fulfillment_status": fulfillment_status,
                                    "observed_states": observed_states,
                                    "rediscovery_required": rediscovery_required,
                                }
                                if rediscovery_required:
                                    failed_signatures.add(call_signature)
                                    if isinstance(llm_api, RAGentAugmentedAPIInstance):
                                        rediscovery_result = await llm_api.async_rediscover_capabilities(
                                            requested_capabilities,
                                        )
                                        existing_names = {tool.name for tool in tool_list}
                                        discovered_tools = tool_helper.discovered_tools(
                                            rediscovery_result, existing_names,
                                        )
                                        tool_list.extend(discovered_tools)
                                        tools_by_name.update({tool.name: tool for tool in discovered_tools})
                                        exposed_tool_names.update(tool.name for tool in discovered_tools)
                                        tool_metadata_dict.update({
                                            tool.name: tool.metadata
                                            for tool in discovered_tools
                                            if tool.metadata
                                        })
                                        execution_status["rediscovered_tools"] = [
                                            tool.name for tool in discovered_tools
                                        ]
                                if isinstance(stored_tool_result, dict):
                                    stored_tool_result["execution_status"] = execution_status
                                else:
                                    stored_tool_result = {
                                        "result": stored_tool_result,
                                        "execution_status": execution_status,
                                    }
                                tool_result_msg = conversation.ToolResultContent(
                                    agent_id=user_input.agent_id,
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_name,
                                    tool_result=stored_tool_result,
                                )
                                history_manager.append_message(tool_result_msg)
                                log_debug_payload(
                                    _logger, "conversation.tool_result",
                                    iteration=idx + 1, requested_call=tool_call,
                                    execution_call=execution_call,
                                    raw_result=tool_result,
                                    parsed_result=parsed_tool_result,
                                    stored_result=stored_tool_result,
                                    execution_status=execution_status,
                                    active_candidate_context=active_candidate_context,
                                    exposed_tool_names=sorted(exposed_tool_names),
                                )
                                _logger.debug(f"RAGent timing: tool {tool_name}: {time.perf_counter() - tool_start:.3f}s")
                            else:
                                _logger.warning(f"LLM API not available, skipping tool execution for tool: {tool_name}")
                                tool_result_msg = conversation.ToolResultContent(
                                    agent_id=user_input.agent_id,
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_name,
                                    tool_result="Tool calling is not active on this instance instruct the user to activate it manually."
                                )
                                history_manager.append_message(tool_result_msg)

                        except Exception as tool_err:
                            _logger.debug(f"Tool {tool_name} failed; passing the failure back to the model: {tool_err}")
                            tool_result_msg = MessageHelper.create_tool_failure_message(
                                agent_id=user_input.agent_id,
                                tool_call_id=tool_call.id,
                                tool_name=tool_name,
                                error=tool_err,
                            )
                            history_manager.append_message(tool_result_msg)
                            failed_signatures.add(call_signature)
                            log_debug_payload(
                                _logger, "conversation.tool_failure",
                                iteration=idx + 1, requested_call=tool_call,
                                error=repr(tool_err), failure_message=tool_result_msg,
                                failed_signatures=failed_signatures,
                            )


            except Exception as err:
                _logger.error(f"There was a problem talking to the backend: {err}")
                if tool_calls_overall:
                    break
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, self.entry.translations.error(TRANSLATION_ERROR_BACKEND))
                return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

            history_manager.persist_chat_history(chat_log)

            if not tool_calls_in_iteration:
                final_model_speech = message_content.strip()
                break

            if idx + 1 == max_tool_call_iterations:
                if tool_calls_overall:
                    break
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, self.entry.translations.error(TRANSLATION_ERROR_MAX_RETRIES))
                return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
            
        intent_response = intent.IntentResponse(language=user_input.language)
        if len(tool_calls_overall) > 0:
            str_tools = [f"{input.tool_name}({', '.join(str(x) for x in input.tool_args.values())})" for input, response in tool_calls_overall]
            tools_str = '\n'.join(str_tools)
            intent_response.async_set_card(title="Changes", content=f"Ran the following tools:\n{tools_str}")

        continue_conversation = False
        if final_model_speech:
            intent_response.async_set_speech(final_model_speech)
            has_question = final_model_speech.endswith(("?", ";", "\uff1f"))
            continue_conversation = get_setting_value(CONF_ALLOW_QUESTIONS, self.runtime_options) and has_question
        else:
            intent_response.async_set_speech(self.entry.translations.error(TRANSLATION_ERROR_NO_SPEECH))

        log_debug_payload(
            _logger, "conversation.final_result",
            conversation_id=user_input.conversation_id,
            final_model_speech=final_model_speech,
            tool_calls=tool_calls_overall,
            continue_conversation=continue_conversation,
        )
        return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id, continue_conversation=continue_conversation)
        

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process the user request"""
        timing_start = time.perf_counter()
        timing_mark = timing_start

        def log_timing(stage: str, details: str | None = None) -> None:
            nonlocal timing_mark
            now = time.perf_counter()
            detail_text = f"{details} in " if details else ""
            _logger.debug(f"RAGent {stage}: {detail_text}{now - timing_mark:.3f}s (total {now - timing_start:.3f}s)")
            timing_mark = now

        scheduled_request = user_input.text.startswith(RAGENT_SCHEDULED_REQUEST_PREFIX)
        if scheduled_request:
            user_input.text = user_input.text.removeprefix(RAGENT_SCHEDULED_REQUEST_PREFIX)
        try:
            scheduled_context = None
            if scheduled_request:
                user_input.text, scheduled_context = ScheduledContext.restore(
                    self.hass, self.subentry_id, user_input.agent_id, user_input.text,
                )
            llm_context = user_input.as_llm_context(DOMAIN)
            with (
                chat_session.async_get_chat_session(self.hass, user_input.conversation_id) as session,
                conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
            ):
                llm_api: llm.APIInstance | None = None

                if self.runtime_options.get(CONF_LLM_HASS_API) != "none":
                    try:
                        llm_api = await llm.async_get_api(
                            self.hass,
                            resolve_llm_api_id(self.runtime_options[CONF_LLM_HASS_API]),
                            llm_context=llm_context,
                        )
                        if isinstance(llm_api, RAGentAugmentedAPIInstance):
                            llm_api.set_conversation_agent_id(user_input.agent_id)
                            llm_api.set_search_scope(self.entry_id,self.subentry_id)

                    except HomeAssistantError as err:
                        _logger.error(f"Error getting LLM API: {err}")
                        intent_response = intent.IntentResponse(language=user_input.language)
                        intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, self.entry.translations.error(TRANSLATION_ERROR_LLM_API))
                        return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                    log_timing("timing: LLM API setup")
                    
                # ensure this chat log has the LLM API instance
                chat_log.llm_api = llm_api
                area, floor = self._get_current_device_location(llm_context)
                current_area = scheduled_context.area if scheduled_context else (area.name if area else "")
                current_floor = scheduled_context.floor if scheduled_context else (floor.name if floor else "")

                history_manager = HistoryManager(
                    runtime_options=self.runtime_options,
                )
                retrieval_method = RetrievalHelper.retrieval_method(self.runtime_options)
                memory_limit = 0 if scheduled_request else get_setting_value(CONF_NUM_MEMORIES_TO_EXTRACT, self.runtime_options)
                needs_embedding = retrieval_method != RETRIEVAL_METHOD_LEXICAL or memory_limit > 0
                retrieval_query = (
                    scheduled_context.retrieval_query(user_input.text)
                    if scheduled_context else user_input.text
                )
                # A single request-scoped future is shared by device, tool and
                # memory retrieval. Lexical-only mode avoids it unless memory
                # recall, which is always vector-based, is enabled.
                query_embedding = QueryEmbedding(
                    lambda: self._async_embed_retrieval_text(
                        RetrievalHelper.build_retrieval_text(retrieval_query)
                    )
                ) if needs_embedding else []
                continuity = ContinuityContext() if scheduled_request else await self._async_build_continuity_context(
                    history_manager, chat_log,
                )
                log_debug_payload(
                    _logger, "conversation.retrieval_plan",
                    conversation_id=user_input.conversation_id,
                    user_text=user_input.text, retrieval_query=retrieval_query,
                    retrieval_method=retrieval_method,
                    configured_device_range=(
                        get_setting_value(CONF_MIN_DEVICES_TO_EXTRACT, self.runtime_options),
                        get_setting_value(CONF_MAX_DEVICES_TO_EXTRACT, self.runtime_options),
                    ),
                    configured_tool_range=(
                        get_setting_value(CONF_MIN_TOOLS_TO_EXTRACT, self.runtime_options),
                        get_setting_value(CONF_MAX_TOOLS_TO_EXTRACT, self.runtime_options),
                    ),
                    memory_limit=memory_limit, current_area=current_area,
                    current_floor=current_floor, scheduled_request=scheduled_request,
                    continuity=continuity,
                )

                # Rank devices, tools, and memories independently. A bad device
                # shortlist must never suppress the correct capability.
                async with asyncio.TaskGroup() as retrieval_tasks:
                    memory_task = retrieval_tasks.create_task(self._async_retrieve_memories(query_embedding, memory_limit))
                    min_devices, max_devices, min_tools, max_tools = self._configured_retrieval_ranges(
                        self.runtime_options
                    )
                    tool_retrieval_query = RetrievalHelper.build_tool_search_query(
                        retrieval_query, "", []
                    )
                    device_task = retrieval_tasks.create_task(self._async_retrieve_devices(
                        query_embedding,
                        retrieval_query,
                        n_devices=min_devices,
                        continuity=continuity,
                        current_area=current_area,
                        current_floor=current_floor,
                        max_devices=max_devices,
                    ))
                    if llm_api:
                        tool_task = retrieval_tasks.create_task(self._async_retrieve_tools(
                            query_embedding,
                            tool_retrieval_query,
                            n_tools=min_tools,
                            continuity=continuity,
                            devices=[],
                            max_tools=max_tools,
                        ))
                    else:
                        tool_task = None

                retrieved_memories = memory_task.result()
                retrieved_devices = device_task.result()
                retrieved_tools = tool_task.result() if tool_task else []
                log_debug_payload(
                    _logger, "conversation.retrieval_result",
                    conversation_id=user_input.conversation_id,
                    retrieval_query=retrieval_query,
                    tool_retrieval_query=tool_retrieval_query,
                    devices=retrieved_devices, tools=retrieved_tools,
                    memories=retrieved_memories, continuity=continuity,
                )

                log_timing(
                    f"Step 2 retrieved {len(retrieved_devices)} devices, "
                    f"{len(retrieved_tools)} tools and "
                    f"{len(retrieved_memories)} memories"
                )

                retrieved_tools = self._ensure_required_tools_exposed(retrieved_tools, llm_api)
                retrieved_tools = self._exclude_prohibited_scheduled_request_tools(retrieved_tools, scheduled_request)

                device_list = []
                for device in retrieved_devices:
                    st = self.hass.states.get(device.id)
                    if st is None:
                        continue

                    device.state = st.state
                    device.attributes = Device.clean_attributes(st.attributes)
                    device_list.append(device)

                candidate_context = self._candidate_context_from_devices(device_list)
                log_debug_payload(
                    _logger, "conversation.prompt_context",
                    conversation_id=user_input.conversation_id,
                    devices=device_list, memories=retrieved_memories,
                    continuity=continuity, candidate_context=candidate_context,
                )
                if isinstance(llm_api, RAGentAugmentedAPIInstance):
                    llm_api.set_search_context(
                        latest_request=retrieval_query,
                        area=current_area,
                        floor=current_floor,
                        candidates=candidate_context,
                    )

                system_prompt_content = await self._async_render_system_prompt(
                    device_list,
                    retrieved_memories,
                    area,
                    floor,
                    continuity=continuity,
                    scheduled_request=scheduled_request,
                    scheduled_context=scheduled_context,
                )
                log_timing("timing: system prompt rendering")
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
                if _logger.isEnabledFor(logging.DEBUG):
                    device_chars = len(json.dumps(
                        [device.to_dict() for device in device_list],
                        ensure_ascii=False,
                        default=str,
                    ))
                    memory_chars = len(json.dumps(
                        [memory.to_dict() for memory in retrieved_memories],
                        ensure_ascii=False,
                        default=str,
                    ))
                    history_chars = sum(
                        len(str(getattr(message, "content", "") or ""))
                        for message in history_manager.message_history[1:-1]
                    )
                    _logger.debug(f"RAGent prompt breakdown: system={len(system_prompt_content)} chars, devices={device_chars} chars, memories={memory_chars} chars, continuity_turns={len(continuity.selected_turn_keys)}, history={history_chars} chars")

                result = await self._async_prompt_model(
                    llm_api,
                    user_input,
                    retrieved_tools,
                    chat_log,
                    history_manager,
                    candidate_context,
                    retrieval_query,
                    scheduled_request
                )
                log_timing("model and tool processing")
                return result
        except Exception as err:
            _logger.error(f"Unexpected error in async_process: {err}")
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
