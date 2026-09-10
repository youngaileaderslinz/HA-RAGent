from __future__ import annotations
from dataclasses import dataclass, field

from custom_components.ha_ragent.src.models.retrieval.target_group import TargetGroup

@dataclass
class ContinuityContext:
    selected_turn_keys: set[str] = field(default_factory=set)
    entities: dict[str, float] = field(default_factory=dict)
    tools: dict[str, float] = field(default_factory=dict)
    areas: dict[str, float] = field(default_factory=dict)
    domains: dict[str, float] = field(default_factory=dict)
    device_classes: dict[str, float] = field(default_factory=dict)
    actions: dict[str, float] = field(default_factory=dict)
    ambiguous_entities: dict[str, float] = field(default_factory=dict)
    target_groups: list[tuple[TargetGroup, float]] = field(default_factory=list)

    @staticmethod
    def _maximum(values: dict[str, float], candidates: list[str]) -> float:
        """Return the maximum value for a set of candidates in a dictionary."""
        return max((values.get(str(candidate).casefold(), 0.0) for candidate in candidates if candidate), default=0.0)

    def device_score(self, device: object) -> float:
        """Return a small continuity boost for a device candidate."""
        entity_id = str(getattr(device, "id", "") or "")
        area = str(getattr(device, "area_name", "") or "")
        domains = list(getattr(device, "domain", None) or [])
        device_class = str(getattr(device, "device_class", "") or "")
        return (
            1.5 * self._maximum(self.entities, [entity_id])
            + 0.75 * self._maximum(self.areas, [area])
            + 0.5 * self._maximum(self.domains, domains)
            + 0.5 * self._maximum(self.device_classes, [device_class])
            + 0.2 * self._maximum(self.ambiguous_entities, [entity_id])
        )

    def tool_score(self, tool: object) -> float:
        """Return a continuity boost for a tool candidate."""
        name = str(getattr(tool, "name", "") or "")
        action = str(getattr(tool, "canonical_action", "") or "")
        return self._maximum(self.tools, [name]) + 0.5 * self._maximum(self.actions, [action])

    def successful_target_score(self, device: object) -> float:
        """Score membership in a selected, previously successful target group."""
        entity_id = str(getattr(device, "id", "") or "").casefold()
        area = str(getattr(device, "area_name", "") or "").casefold()
        floor = str(getattr(device, "floor_name", "") or "").casefold()
        domains = { str(value).casefold() for value in (getattr(device, "domain", None) or []) }
        device_class = str(getattr(device, "device_class", "") or "").casefold()
        best = 0.0
        for group, weight in self.target_groups:
            group_entities = {value.casefold() for value in group.entities}
            if entity_id and entity_id in group_entities:
                best = max(best, weight)
                continue
            location_matches = (
                (not group.areas or area in {value.casefold() for value in group.areas})
                and (not group.floors or floor in {value.casefold() for value in group.floors})
            )
            type_matches = (
                (not group.domains or bool(domains & {value.casefold() for value in group.domains}))
                and (
                    not group.device_classes
                    or device_class in {value.casefold() for value in group.device_classes}
                )
            )
            if location_matches and type_matches and (group.areas or group.floors) and (
                group.domains or group.device_classes
            ):
                best = max(best, weight)
        return best
