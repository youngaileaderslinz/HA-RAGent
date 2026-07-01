from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import timedelta
from typing import Any, List, Tuple, Dict

from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, JsonObjectType
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import chat_session, intent, llm
from homeassistant.helpers.template import Template
from homeassistant.helpers.llm import ToolInput, LLMContext
from homeassistant.helpers import area_registry as ar, device_registry as dr, floor_registry as fr
from homeassistant.util import dt as dt_util
from voluptuous_openapi import convert

from custom_components.ha_ragent.src.models.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding

from .ragent_entity import RAGentEntity
from .ragent_config_entry import RAGentConfigEntry
from .search_tool_api import augment_api_with_search_tool
from ..models.device import Device
from ..const import RAGENT_SEARCH_DEVICES_TOOL_NAME, RAGENT_SEARCH_TOOLS_TOOL_NAME

from ..const import (
    CONF_NUM_DEVICES_TO_EXTRACT,
    CONF_NUM_TOOLS_TO_EXTRACT,
    CONF_PROMPT,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    DEFAULT_NUM_DEVICES_TO_EXTRACT,
    DEFAULT_NUM_TOOLS_TO_EXTRACT,
    DEFAULT_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
    DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DOMAIN,
    PERSONA_PROMPTS,
    CURRENT_DATE_PROMPT,
    DEVICES_PROMPT,
    AREAS_PROMPT,
    CONVERSATION_PRIORITY_PROMPT,
    USER_INSTRUCTION,
    DEVICE_CONTROL_PROMPT,
    TOOL_REGEX_PATTERN
)

from ..utils import (
    get_placeholder_translation,
    clean_device_attributes
)

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

    def _filter_retrieval_history(
        self,
        chat_log: conversation.ChatLog,
    ) -> List[conversation.Content]:
        """Return history items that are safe and useful for retrieval."""
        remember_time_minutes = self.runtime_options.get(
            CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
            DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
        )
        remember_num_interactions = self.runtime_options.get(
            CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
            DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
        )

        keep_history = bool(remember_time_minutes) or bool(remember_num_interactions)
        if not keep_history:
            return []

        raw_history = list(chat_log.content)[:-1]
        filtered_history: List[conversation.Content] = []
        previous_msg: conversation.Content | None = None

        for msg in raw_history:
            if isinstance(msg, conversation.SystemContent):
                previous_msg = msg
                continue

            if isinstance(msg, conversation.ToolResultContent) and "failed" in getattr(msg, "tool_result", {}):
                if filtered_history and previous_msg is filtered_history[-1]:
                    filtered_history.pop()
                previous_msg = msg
                continue

            filtered_history.append(msg)
            previous_msg = msg

        if remember_time_minutes:
            now = dt_util.utcnow()
            cutoff = now - timedelta(minutes=remember_time_minutes)
            filtered_history = [
                msg for msg in filtered_history if getattr(msg, "created_at", now) >= cutoff
            ]

        if remember_num_interactions and len(filtered_history) > (remember_num_interactions * 2):
            filtered_history = filtered_history[-(remember_num_interactions * 2):]

        return filtered_history

    def _filter_prompt_history(
        self,
        chat_log: conversation.ChatLog,
    ) -> List[conversation.Content]:
        """Return compact prompt history that favors user intent and real tool state."""
        filtered_history = self._filter_retrieval_history(chat_log)
        prompt_history: List[conversation.Content] = []

        for msg in filtered_history:
            if isinstance(msg, conversation.UserContent):
                prompt_history.append(msg)
                continue

            if isinstance(msg, conversation.ToolResultContent):
                prompt_history.append(msg)
                continue

        return prompt_history

    def _content_to_retrieval_text(self, msg: conversation.Content) -> str:
        """Convert chat content into text suitable for retrieval embeddings."""
        if isinstance(msg, (conversation.UserContent, conversation.AssistantContent)):
            return (msg.content or "").strip()

        if isinstance(msg, conversation.ToolResultContent):
            return f"{msg.tool_name} {msg.tool_result}".strip()

        return ""

    async def _async_embed_query(
        self,
        user_input: ConversationInput,
        retrieval_history: List[conversation.Content],
    ) -> List[float]:
        """Embed the current query plus older messages with logarithmic recency decay."""
        _logger.debug("RAG Step 1: Embedding user input with history-aware decay: %s", user_input.text)

        history_texts = [
            text for text in (self._content_to_retrieval_text(msg) for msg in retrieval_history) if text
        ]
        texts_to_embed = [*history_texts, user_input.text]

        try:
            embeddings = await asyncio.gather(
                *[
                    self.entry.embedder_backend.async_embed_text(dict(self.subentry.data), text)
                    for text in texts_to_embed
                ]
            )
        except Exception as err:
            _logger.error("Error embedding user input with history: %s", err, exc_info=True)
            return None

        valid_embeddings = [embedding for embedding in embeddings if embedding]
        if not valid_embeddings:
            return None

        weighted_embedding = [0.0] * len(valid_embeddings[0])
        total_weight = 0.0

        for idx, embedding in enumerate(valid_embeddings):
            recency_rank = len(valid_embeddings) - idx
            weight = 1.0 if recency_rank == 1 else 1.0 / math.log2(recency_rank + 1)

            if len(embedding) != len(weighted_embedding):
                _logger.warning(
                    "Skipping history embedding due to mismatched dimensions (%s != %s).",
                    len(embedding),
                    len(weighted_embedding),
                )
                continue

            weighted_embedding = [
                current + (value * weight)
                for current, value in zip(weighted_embedding, embedding)
            ]
            total_weight += weight

        if total_weight == 0:
            return None

        query_embedding = [value / total_weight for value in weighted_embedding]
        _logger.debug(
            "History-aware query embedded successfully with %s messages, embedding shape: %s",
            len(valid_embeddings),
            len(query_embedding),
        )
        return query_embedding

    async def _async_retrieve_devices(self, query_embedding: List[float], n_devices: int) -> List[Device]:
        """Retrieve relevant devices from vector database based on query embedding."""
        _logger.debug("RAG Step 2: Querying vector database for similar devices")
        collection_name = f"devices_{self.subentry_id}"
        _logger.debug(f"Collection name: {collection_name}, Query embedding dimension: {len(query_embedding)}")
        retrieved_devices = []
        try:
            retrieved_devices = await self.entry.vector_db_backend.async_retrieve_objects(
                object_type=DeviceEmbedding,
                config_subentry=dict(self.subentry.data),
                collection_name=collection_name,
                query_embedding=query_embedding,
                top_k=n_devices
            )
            _logger.debug(f"Retrieved {len(retrieved_devices)} relevant devices from vector database (collection: {collection_name})")
        except Exception as e:
            _logger.error(f"Error retrieving devices from vector DB: {e}", exc_info=True)
        
        return retrieved_devices

    async def _async_retrieve_tools(self, query_embedding: List[float], n_tools: int) -> List[LlmTool]:
        """Retrieve relevant tools from vector database based on query embedding."""
        _logger.debug("RAG Step 2: Querying vector database for similar tools")
        collection_name = f"tools_{self.subentry_id}"
        _logger.debug(f"Collection name: {collection_name}, Query embedding dimension: {len(query_embedding)}")
        retrieved_tools = []
        try:
            retrieved_tools = await self.entry.vector_db_backend.async_retrieve_objects(
                object_type=LlmToolEmbedding,
                config_subentry=dict(self.subentry.data),
                collection_name=collection_name,
                query_embedding=query_embedding,
                top_k=n_tools
            )
            _logger.debug(f"Retrieved {len(retrieved_tools)} relevant tools from vector database (collection: {collection_name})")
        except Exception as e:
            _logger.error("Error retrieving tools from vector DB: %s", e, exc_info=True)
        
        return retrieved_tools

    async def _async_render_template(self, template_str: str, devices: List[Device], area: ar.AreaEntry, floor: fr.FloorEntry) -> str:
        """Render a Jinja2 template string with the given context."""
        try:
            template = Template(template_str, self.hass)
            rendered = template.async_render({
                "device_list": devices,
                "area_list": list(set(device.area_name for device in devices if device.area_name)),
                "area_name": area.name if area else None,
                "floor_name": floor.name if floor else None,
            })
            return rendered
        except TemplateError as e:
            _logger.error(f"Template rendering error: {e}", exc_info=True)
            raise e

    async def _async_get_message_history(self, chat_log: conversation.ChatLog, user_input: ConversationInput, devices: List[Device], area: ar.AreaEntry, floor: fr.FloorEntry) -> List[conversation.Content]:
        """Build the prompt for the LLM, including retrieved device context."""
        raw_prompt = self.runtime_options.get(CONF_PROMPT, DEFAULT_PROMPT)
        remember_time_minutes = self.runtime_options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES)
        remember_num_interactions = self.runtime_options.get(CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS, DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS)

        try:
            system_prompt_content = await self._async_render_template(raw_prompt, devices, area, floor)
            system_prompt = conversation.SystemContent(content=system_prompt_content)
        except Exception as err:
            _logger.error(f"Error rendering prompt: {err}", exc_info=True)
            return None

        message_history = [system_prompt]
        message_history.extend(self._filter_prompt_history(chat_log))
        message_history.append(conversation.UserContent(content=user_input.text))

        return message_history

    def _parse_tool_calls(self, response_text: str) -> List[dict]:
        """Parse tool calls from LLM response."""
        parsed_calls = []
        
        _logger.debug("Parsing tool calls from LLM response.")
        for match in TOOL_REGEX_PATTERN.finditer(response_text):
            try:
                content = match.group(1).strip()
                first_brace = content.find('{')
                last_brace = content.rfind('}')
                if first_brace >= 0 and last_brace > first_brace:
                    json_str = content[first_brace:last_brace + 1]
                    tool_json = json.loads(json_str)
                    
                    parameters = tool_json.get("arguments", {})

                    if "name" in parameters and "." in parameters["name"]:
                        state = self.hass.states.get(parameters["name"])
                        parameters["name"] = state.attributes.get("friendly_name") if state else parameters["name"]

                    if "device_class" in parameters:
                        device_class = parameters.pop("device_class")
                        if "domain" not in parameters:
                            parameters["domain"] = device_class

                    if "floor" in parameters:
                        parameters.pop("floor")

                    parsed_calls.append({
                        "name": tool_json.get("tool"),
                        "parameters": parameters
                    })

                    _logger.debug(f"Parsed tool call from homeassistant block: {tool_json}")
            except (json.JSONDecodeError, AttributeError) as e:
                _logger.warning(f"Failed to parse homeassistant block JSON: {e}")

        return parsed_calls
    
    def _parse_tool_results(self, tool_result: JsonObjectType) -> Dict[str, Any]:
        """Parse tool results from LLM response."""
        if not isinstance(tool_result, dict):
            return {"result": tool_result}

        data = tool_result.get("data", {})
        success = data.get("success", [])
        failed = data.get("failed", [])
        parsed_result: Dict[str, Any] = {}
        
        success_ids = [x["id"] for x in success if x.get("type") == "entity"]
        if success_ids:
            parsed_result["success"] = success_ids

        failed_ids = [x["id"] for x in failed if x.get("type") == "entity"]
        if failed_ids:
            parsed_result["failed"] = failed_ids

        if parsed_result:
            return parsed_result

        return tool_result

    def _convert_api_tool(self, api_tool: Any, llm_api: llm.APIInstance | None) -> LlmTool | None:
        """Convert a Home Assistant LLM tool into the local tool schema."""
        tool_name = getattr(api_tool, "name", None)
        if not tool_name:
            return None

        parameters = {}
        if hasattr(api_tool, "parameters") and api_tool.parameters:
            try:
                parameters = convert(
                    api_tool.parameters,
                    custom_serializer=llm_api.custom_serializer if llm_api else None,
                )
            except Exception as err:
                _logger.warning("Could not convert parameters for tool %s: %s", tool_name, err)

        return LlmTool(
            name=tool_name,
            description=getattr(api_tool, "description", ""),
            parameters=parameters,
            metadata={},
        )

    def _ensure_runtime_search_tools_exposed(
        self,
        tool_list: List[LlmTool],
        llm_api: llm.APIInstance | None,
    ) -> List[LlmTool]:
        """Always expose runtime search tools to the model without embedding them."""
        if not llm_api or not hasattr(llm_api, "tools"):
            return tool_list

        seen_tool_names = {tool.name for tool in tool_list}
        required_tool_names = {
            RAGENT_SEARCH_DEVICES_TOOL_NAME,
            RAGENT_SEARCH_TOOLS_TOOL_NAME,
        }

        runtime_search_tools: list[LlmTool] = []
        for api_tool in llm_api.tools:
            tool_name = getattr(api_tool, "name", None)
            if tool_name not in required_tool_names or tool_name in seen_tool_names:
                continue

            converted_tool = self._convert_api_tool(api_tool, llm_api)
            if converted_tool:
                runtime_search_tools.append(converted_tool)
                seen_tool_names.add(tool_name)

        return [*tool_list, *runtime_search_tools]
    
    def _get_current_area(self, llm_context: LLMContext) -> ar.AreaEntry | None:
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

    async def _async_prompt_model(self, llm_api: llm.APIInstance, user_input: ConversationInput, tool_list: List[LlmTool], chat_log: conversation.ChatLog, message_history: List[conversation.Content]) -> ConversationResult:
        """Process a prompt through the RAGent."""
        max_tool_call_iterations = self.runtime_options.get(CONF_MAX_TOOL_CALL_ITERATIONS, DEFAULT_MAX_TOOL_CALL_ITERATIONS)

        formatted_messages = []
        last_formatted_index = 0
        tool_calls: List[Tuple[llm.ToolInput, Any]] = []

        for idx in range(max(1, max_tool_call_iterations)):
            _logger.debug(f"Generating response for {user_input.text}, iteration {idx + 1}/{max_tool_call_iterations}.")
            
            for i in range(last_formatted_index, len(message_history)):
                msg = message_history[i]
                if isinstance(msg, conversation.SystemContent):
                    formatted_messages.append({"role": "SYSTEM", "content": msg.content})
                elif isinstance(msg, conversation.UserContent):
                    formatted_messages.append({"role": "USER", "content": msg.content})
                elif isinstance(msg, conversation.AssistantContent):
                    formatted_messages.append({"role": "ASSISTANT", "content": msg.content})
                elif isinstance(msg, conversation.ToolResultContent):
                    formatted_messages.append({"role": "TOOL", "content": "{" + f"name: {msg.tool_name}, result: {msg.tool_result}" + "}"})

            last_formatted_index = len(message_history)

            tool_calls_in_iteration = []
            try:
                _logger.info(f"Sending prompt to LLM (Iteration {idx + 1}/{max_tool_call_iterations}).")
                _logger.debug("Full prompt sent to the LLM:\n%s", "\n".join(f"{m['role']}: {m['content']}" for m in formatted_messages))
                
                content_chunks = []
                async for chunk in self.entry.llm_backend.async_send_chat_request(dict(self.subentry.data), formatted_messages, tool_list):
                    content_chunks.append(chunk)
                assistant_content = "".join(content_chunks)

                _logger.debug("LLM response: %s", assistant_content)
                
                tool_calls_in_iteration = self._parse_tool_calls(assistant_content)
                
                message = conversation.AssistantContent(
                    agent_id=user_input.agent_id,
                    content=assistant_content,
                    tool_calls=tool_calls_in_iteration
                )
                message_history.append(message)
                
                if tool_calls_in_iteration and len(tool_calls_in_iteration) > 0:
                    _logger.info("Executing %d tool calls", len(tool_calls_in_iteration))
                    
                    for tool_call in tool_calls_in_iteration:
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("parameters", {})
                        _logger.debug(f"Executing tool: {tool_name} with args: {tool_args}.")
                        
                        tool_input = ToolInput(tool_name=tool_name, tool_args=tool_args)
                        try:
                            if llm_api:
                                tool_result = await llm_api.async_call_tool(tool_input)
                                _logger.debug(f"Tool result: {tool_result}.")
                                
                                tool_calls.append((tool_input, tool_result))
                                
                                tool_result_msg = conversation.ToolResultContent(
                                    agent_id=user_input.agent_id,
                                    tool_call_id=tool_input.id,
                                    tool_name=tool_name,
                                    tool_result=self._parse_tool_results(tool_result)
                                )
                                message_history.append(tool_result_msg)
                            else:
                                _logger.warning("LLM API not available, skipping tool execution for tool: %s", tool_name)
                                tool_result_msg = conversation.ToolResultContent(
                                    agent_id=user_input.agent_id,
                                    tool_call_id=tool_input.id,
                                    tool_name=tool_name,
                                    tool_result="Tool calling is not active on this instance instruct the user to activate it manually."
                                )
                                message_history.append(tool_result_msg)

                        except Exception as tool_err:
                            tool_result_msg = conversation.ToolResultContent(
                                agent_id=user_input.agent_id,
                                tool_call_id=tool_input.id,
                                tool_name=tool_name,
                                tool_result={"failed": f"{tool_name}: {str(tool_err)}"}
                            )
                            message_history.append(tool_result_msg)
                    
            except Exception as err:
                _logger.error(f"There was a problem talking to the backend: {err}")
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, f"Sorry, there was a problem talking to the backend.")
                return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

            if not tool_calls_in_iteration:
                break

            if idx + 1 == max_tool_call_iterations:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, f"Sorry, I ran out of attempts to handle your request")
                return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
            
            chat_log.content = message_history
            
        intent_response = intent.IntentResponse(language=user_input.language)
        if len(tool_calls) > 0:
            str_tools = [f"{input.tool_name}({', '.join(str(x) for x in input.tool_args.values())})" for input, response in tool_calls]
            tools_str = '\n'.join(str_tools)
            intent_response.async_set_card(title="Changes", content=f"Ran the following tools:\n{tools_str}")

        has_speech = False
        for cur_msg in reversed(message_history[1:]):
            if isinstance(cur_msg, conversation.AssistantContent) and cur_msg.content:
                intent_response.async_set_speech(cur_msg.content)
                has_speech = True
                break

        if not has_speech:
            intent_response.async_set_speech("I don't have anything to say right now")
            _logger.debug(message_history)

        return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
        

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process the user request"""
        try:
            with (
                chat_session.async_get_chat_session(self.hass, user_input.conversation_id) as session,
                conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
            ):
                llm_api: llm.APIInstance | None = None

                if self.runtime_options.get(CONF_LLM_HASS_API) != "none":
                    try:
                        llm_api = await llm.async_get_api(
                            self.hass,
                            self.runtime_options[CONF_LLM_HASS_API],
                            llm_context=user_input.as_llm_context(DOMAIN)
                        )
                        llm_api = augment_api_with_search_tool(self.hass, llm_api)
                    except HomeAssistantError as err:
                        _logger.error("Error getting LLM API: %s", err)
                        intent_response = intent.IntentResponse(language=user_input.language)
                        intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Error preparing LLM API.")
                        return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
                    
                # ensure this chat log has the LLM API instance
                chat_log.llm_api = llm_api

                retrieval_history = self._filter_retrieval_history(chat_log)
                query_embedding = await self._async_embed_query(user_input, retrieval_history)
                if not query_embedding:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Failed to embed user input.")
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                retrieve_devices_task = self._async_retrieve_devices(query_embedding, n_devices=self.runtime_options.get(CONF_NUM_DEVICES_TO_EXTRACT, DEFAULT_NUM_DEVICES_TO_EXTRACT))
                retrieve_tools_task = self._async_retrieve_tools(query_embedding, n_tools=self.runtime_options.get(CONF_NUM_TOOLS_TO_EXTRACT, DEFAULT_NUM_TOOLS_TO_EXTRACT)) if llm_api else None

                if retrieve_tools_task:
                    retrieved_devices, retrieved_tools = await asyncio.gather(retrieve_devices_task, retrieve_tools_task)
                else:
                    retrieved_devices = await retrieve_devices_task
                    retrieved_tools = []

                if not retrieved_devices:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Failed to retrieve relevant devices.")
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                retrieved_tools = self._ensure_runtime_search_tools_exposed(retrieved_tools, llm_api)

                if not retrieved_tools and llm_api:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Failed to retrieve relevant tools.")
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                device_list = []
                for device in retrieved_devices:
                    st = self.hass.states.get(device.id)
                    if st is None:
                        continue

                    device.state = st.state
                    device.attributes = clean_device_attributes(st.attributes)
                    device_list.append(device)
                
                area, floor = self._get_current_area(user_input.as_llm_context(DOMAIN))

                message_history = await self._async_get_message_history(chat_log, user_input, device_list, area, floor)
                if not message_history:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Template rendering failed.")
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
                
                return await self._async_prompt_model(llm_api, user_input, retrieved_tools, chat_log, message_history)
        except Exception as err:
            _logger.error("Unexpected error in async_process: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, f"Sorry, an unexpected error occurred.")
            return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

    @staticmethod
    def build_base_prompt_template(selected_language: str, prompt_template: str):
        """Build base prompt template from constants in specified language."""
        prompt_template = prompt_template.replace("<persona>", get_placeholder_translation(PERSONA_PROMPTS, selected_language))
        prompt_template = prompt_template.replace("<current_date>", get_placeholder_translation(CURRENT_DATE_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<area_prompt>", get_placeholder_translation(AREAS_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<devices>", get_placeholder_translation(DEVICES_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<areas>", get_placeholder_translation(AREAS_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<device_control_prompt>", get_placeholder_translation(DEVICE_CONTROL_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<conversation_priority_prompt>", get_placeholder_translation(CONVERSATION_PRIORITY_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<user_instruction>", get_placeholder_translation(USER_INSTRUCTION, selected_language))
        
        return prompt_template
