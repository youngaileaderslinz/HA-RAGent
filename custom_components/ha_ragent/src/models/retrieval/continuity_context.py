from __future__ import annotations
from dataclasses import dataclass, field

from custom_components.ha_ragent.src.models.retrieval.target_group import TargetGroup

@dataclass
class ContinuityContext:
    selected_turn_keys: set[str] = field(default_factory=set)
    entities: dict[str, float] = field(default_factory=dict)
    tools: dict[str, float] = field(default_factory=dict)
    areas: dict[str, float] = field(default_factory=dict)
    floors: dict[str, float] = field(default_factory=dict)
    domains: dict[str, float] = field(default_factory=dict)
    device_classes: dict[str, float] = field(default_factory=dict)
    actions: dict[str, float] = field(default_factory=dict)
    ambiguous_entities: dict[str, float] = field(default_factory=dict)
    target_groups: list[tuple[TargetGroup, float]] = field(default_factory=list)

    @staticmethod
    def _maximum(values: dict[str, float], candidates: list[str]) -> float:
        """Return the maximum value for a set of candidates in a dictionary."""
        return max((values.get(str(candidate).casefold(), 0.0) for candidate in candidates if candidate), default=0.0)

    def entity_score(self, device: object) -> float:
        """Return continuity evidence tied to a successfully resolved entity."""
        entity_id = str(getattr(device, "id", "") or "")
        return 1.5 * self._maximum(self.entities, [entity_id])

    def ambiguous_entity_score(self, device: object) -> float:
        """Return weak evidence from failed or still-unresolved entity targets."""
        entity_id = str(getattr(device, "id", "") or "")
        return 0.2 * self._maximum(self.ambiguous_entities, [entity_id])

    def area_score(self, device: object) -> float:
        """Return weaker location continuity without implying entity identity."""
        area = str(getattr(device, "area_name", "") or "")
        floor = str(getattr(device, "floor_name", "") or "")
        return (
            0.35 * self._maximum(self.areas, [area])
            + 0.2 * self._maximum(self.floors, [floor])
        )

    def successful_target_score(self, device: object) -> float:
        """Confirm only exact entity membership in a successful target group."""
        entity_id = str(getattr(device, "id", "") or "").casefold()
        best = 0.0
        for group, weight in self.target_groups:
            group_entities = {value.casefold() for value in group.entities}
            if entity_id and entity_id in group_entities:
                best = max(best, weight)
        return best
