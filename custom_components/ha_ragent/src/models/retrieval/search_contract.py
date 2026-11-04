from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def opaque_id(*parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SearchFilters:
    """Validated AND constraints; string fields use exact normalized matching."""

    values: dict

    @classmethod
    def parse(cls, raw: object) -> SearchFilters:
        if not isinstance(raw, dict):
            raise ValueError("filters must be an object")
        fields = {"area", "floor", "domain", "integration", "entity_type", "device_type", "state"}
        if raw.keys() - fields:
            raise ValueError("Unknown filter field")
        values = {}
        for key, value in sorted(raw.items()):
            if key == "state":
                if isinstance(value, str):
                    value = {"operator": "eq", "value": value}
                if not isinstance(value, dict) or set(value) != {"operator", "value"}:
                    raise ValueError("state requires a value or {operator, value}")
                operator = value["operator"]
                target = value["value"]
                if not isinstance(operator, str) or operator not in {"eq", "ne", "gt", "gte", "lt", "lte"}:
                    raise ValueError("Invalid state operator")
                if isinstance(target, bool) or not isinstance(target, (str, int, float)):
                    raise ValueError("Invalid state value")
                if operator not in {"eq", "ne"}:
                    try:
                        target = float(target)
                    except (ValueError, OverflowError):
                        raise ValueError("Numeric state operator requires a finite number") from None
                    if not math.isfinite(target):
                        raise ValueError("Numeric state operator requires a finite number")
                elif isinstance(target, str):
                    if not target.strip() or len(target) > 256:
                        raise ValueError("State value must be nonempty and at most 256 characters")
                    target = normalized(target)
                else:
                    try:
                        finite = math.isfinite(target)
                    except OverflowError:
                        finite = False
                    if not finite:
                        raise ValueError("State value must be finite")
                values[key] = {"operator": operator, "value": target}
            else:
                if not isinstance(value, str) or not value.strip() or len(value) > 256:
                    raise ValueError(f"{key} must be a nonempty string of at most 256 characters")
                values[key] = normalized(value)
        return cls(values)

    def matches(self, candidate: dict) -> bool:
        for key, target in self.values.items():
            actual = candidate.get(key)
            if key == "state":
                if actual is None:
                    return False
                operator, value = target["operator"], target["value"]
                if operator in {"eq", "ne"} and isinstance(value, str):
                    equal = normalized(str(actual)) == value
                    if equal != (operator == "eq"):
                        return False
                else:
                    try:
                        number = float(actual)
                    except (TypeError, ValueError, OverflowError):
                        return False
                    if not math.isfinite(number):
                        return False
                    if not {"eq": number == value, "ne": number != value, "gt": number > value,
                            "gte": number >= value, "lt": number < value, "lte": number <= value}[operator]:
                        return False
            else:
                actual = actual if isinstance(actual, (list, tuple, set)) else [actual]
                if not any(isinstance(item, str) and normalized(item) == target for item in actual):
                    return False
        return True
