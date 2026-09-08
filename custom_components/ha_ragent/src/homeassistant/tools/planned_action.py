from __future__ import annotations

from datetime import datetime, timedelta
import logging
from collections.abc import Callable
from uuid import uuid4

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from custom_components.ha_ragent.src.const import (
    DOMAIN,
    RAGENT_PLANNED_ACTION_TOOL_NAME,
    RAGENT_SCHEDULED_ACTIONS,
    RAGENT_SCHEDULED_ACTION_CANCELLERS,
    RAGENT_SCHEDULED_REQUEST_PREFIX,
    RAGENT_SCHEDULED_CONTEXT_PREFIX,
    RAGENT_SCHEDULED_EXECUTION_CONTEXTS,
    TRANSLATION_ERROR_DESCRIPTION_EMPTY,
    TRANSLATION_ERROR_MINUTES_NOT_NUMBER,
    TRANSLATION_ERROR_MINUTES_RANGE,
)
from custom_components.ha_ragent.src.models.retrieval.scheduled_context import ScheduledContext
from custom_components.ha_ragent.src.translation import RAGentTranslations

_logger = logging.getLogger(__name__)

class RAGentPlannedActionTool(llm.Tool):
    name = RAGENT_PLANNED_ACTION_TOOL_NAME
    parameters = vol.Schema(
        {
            vol.Required("description"): str,
            vol.Required("minutes"): vol.All(
                vol.Coerce(int),
                vol.Range(min=1, max=1440),
            ),
        }
    )

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        subentry_id: str,
        agent_id: str,
        context: Context,
        language: str | None,
        device_id: str | None,
    ) -> None:
        self.hass = hass
        self.subentry_id = subentry_id
        self.agent_id = agent_id
        self.context = context
        self.language = language
        self.translations = RAGentTranslations(language or "en")
        self.device_id = device_id
        self.description = self.translations.tool(RAGENT_PLANNED_ACTION_TOOL_NAME)
        self._scheduling_context = ScheduledContext(subentry_id, agent_id)

    def set_scheduling_context(self, **context) -> None:
        self._scheduling_context = ScheduledContext.capture(
            subentry_id=self.subentry_id, agent_id=self.agent_id, **context,
        )

    async def _async_execute(self, _now: datetime, description: str, snapshot: ScheduledContext) -> None:
        """Send the due action back through the originating conversation agent."""
        execution_id = uuid4().hex
        contexts = self.hass.data.setdefault(DOMAIN, {}).setdefault(RAGENT_SCHEDULED_EXECUTION_CONTEXTS, {})
        contexts[execution_id] = snapshot
        try:
            result = await conversation.async_converse(
                hass=self.hass,
                text=f"{RAGENT_SCHEDULED_REQUEST_PREFIX}{RAGENT_SCHEDULED_CONTEXT_PREFIX}{execution_id}] {description}",
                conversation_id=None,
                context=self.context,
                language=self.language,
                agent_id=snapshot.agent_id,
                device_id=self.device_id,
            )
            if result.response.error_code is not None:
                _logger.error(f"Planned action failed for agent {self.agent_id}: {description}. Error: {result.response.error_code}")
            else:
                _logger.info(f"Executed planned action for agent {self.agent_id}: {description}")

        except Exception:
            _logger.error(f"Failed to execute planned action for agent {self.agent_id}: {description}", exc_info=True)
        finally:
            contexts.pop(execution_id, None)

    async def async_call(self, tool_input: llm.ToolInput, *args, **kwargs) -> dict[str, object]:
        """Schedule the requested action without blocking the conversation."""
        description = str(tool_input.tool_args.get("description", "")).strip()
        if not description:
            return {"error": self.translations.error(TRANSLATION_ERROR_DESCRIPTION_EMPTY)}

        try:
            minutes = int(tool_input.tool_args.get("minutes", 0))
        except (TypeError, ValueError):
            return {"error": self.translations.error(TRANSLATION_ERROR_MINUTES_NOT_NUMBER)}

        if not 1 <= minutes <= 1440:
            return {"error": self.translations.error(TRANSLATION_ERROR_MINUTES_RANGE)}

        # Freeze per timer: subsequent searches or schedules must not retarget it.
        snapshot = ScheduledContext.capture(
            subentry_id=self.subentry_id,
            agent_id=self.agent_id,
            request=self._scheduling_context.request,
            area=self._scheduling_context.area,
            floor=self._scheduling_context.floor,
            candidates=self._scheduling_context.candidates,
            messages=self._scheduling_context.messages,
        )
        execute_at = dt_util.utcnow() + timedelta(minutes=minutes)
        local_execute_at = dt_util.as_local(execute_at)
        human_execute_at = local_execute_at.strftime("%A, %B %d, %Y at %H:%M %Z").replace(" 0", " ")

        domain_data = self.hass.data.setdefault(DOMAIN, {})
        subentry_data = domain_data.setdefault(self.subentry_id, {})
        cancellers = subentry_data.setdefault(RAGENT_SCHEDULED_ACTION_CANCELLERS, set())
        actions = subentry_data.setdefault(RAGENT_SCHEDULED_ACTIONS, {})
        remove_canceller: Callable[[], None] | None = None

        async def execute_and_remove(now: datetime) -> None:
            if remove_canceller is not None:
                cancellers.discard(remove_canceller)
                actions.pop(remove_canceller, None)
            await self._async_execute(now, description, snapshot)

        remove_canceller = async_call_later(self.hass, minutes * 60, execute_and_remove)
        cancellers.add(remove_canceller)
        actions[remove_canceller] = {
            "description": description,
            "minutes": minutes,
            "execute_at": human_execute_at,
        }
        return {
            "success": True,
            "description": description,
            "minutes": minutes,
            "execute_at": human_execute_at,
        }
