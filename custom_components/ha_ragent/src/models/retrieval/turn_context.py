from __future__ import annotations
from dataclasses import dataclass

from custom_components.ha_ragent.src.models.retrieval.target_group import TargetGroup

@dataclass
class TurnContext:
    key: str
    text: str
    entities: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    areas: tuple[str, ...] = ()
    floors: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    device_classes: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    ambiguous_entities: tuple[str, ...] = ()
    target_groups: tuple[TargetGroup, ...] = ()
    created_at: float | None = None

    def to_embedding_text(self) -> str:
        values = [
            self.text,
            *self.entities,
            *self.tools,
            *self.areas,
            *self.floors,
            *self.domains,
            *self.device_classes,
            *self.actions,
            *self.ambiguous_entities,
        ]
        values.extend(
            "target group: " + " ".join((
                *group.entities,
                *group.areas,
                *group.floors,
                *group.domains,
                *group.device_classes,
                group.tool,
                group.action,
            ))
            for group in self.target_groups
        )
        return " | ".join(value for value in values if value)

    @property
    def has_canonical_context(self) -> bool:
        return bool(
            self.entities or self.tools or self.areas or self.floors
            or self.domains or self.device_classes
        )
