"""Conservative interpretation of JSON-schema target constraints.

This deliberately answers only whether a field is safely restrictive.  It is
not a general schema validator: an unknown construct (including ``$ref``) is
neutral, so it can never be used to exclude a tool.
"""

from __future__ import annotations

from typing import Any


def constrained_values(schema: object) -> set[str] | None:
    """Return accepted literals when *all* paths restrict the value.

    ``allOf`` intersects restrictions; ``anyOf``/``oneOf`` take a union only
    when each alternative is itself known to be restrictive.  This makes a
    branch such as ``anyOf: [enum(light), string]`` unrestricted.
    """
    if not isinstance(schema, dict) or "$ref" in schema:
        return None
    direct: set[str] | None = None
    if "const" in schema and isinstance(schema["const"], (str, int, float)):
        direct = {str(schema["const"]).casefold()}
    elif isinstance(schema.get("enum"), list):
        direct = {str(value).casefold() for value in schema["enum"]}

    for keyword in ("allOf", "anyOf", "oneOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list) or not variants:
            continue
        values = [constrained_values(variant) for variant in variants]
        if keyword == "allOf":
            known = [value for value in values if value is not None]
            if not known:
                continue
            combined = set.intersection(*known)
        else:
            if any(value is None for value in values):
                return None
            combined = set().union(*(value or set() for value in values))
        direct = combined if direct is None else direct & combined
    return direct


def root_property_values(schema: object, field: str) -> set[str]:
    """Return safe root property restrictions, excluding unknown nesting."""
    if not isinstance(schema, dict) or "$ref" in schema:
        return set()
    properties = schema.get("properties")
    value = properties.get(field) if isinstance(properties, dict) else None
    return constrained_values(value) or set()
