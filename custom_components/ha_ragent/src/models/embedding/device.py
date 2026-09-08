import json
from typing import List, Dict, Any
from dataclasses import dataclass

from custom_components.ha_ragent.src.const import DEVICE_ATTRIBUTES_MAX_JSON_LENGTH, DEVICE_ATTRIBUTES_TO_EXCLUDE
from custom_components.ha_ragent.src.models.base.serializeable_model import SerializableModel
from custom_components.ha_ragent.src.models.base.embeddable_model import EmbeddableModel

@dataclass
class Device(SerializableModel, EmbeddableModel):
    id: str
    friendly_name: str
    area_name: str
    floor_name: str
    domain: List[str] = None
    device_labels: List[str] = None
    services: List[str] = None
    aliases: List[str] = None
    unit_of_measurement: str = None
    device_class: str = None

    # Loaded from current state not used for embedding
    state: str = None
    attributes: Dict[str, Any] = None
    area_aliases: List[str] = None
    floor_aliases: List[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the device."""
        return {
            "device_id": self.id,
            "friendly_name": self.friendly_name,
            "area_name": self.area_name,
            "floor_name": self.floor_name,
            "area_aliases": self.area_aliases,
            "floor_aliases": self.floor_aliases,
            "domain": self.domain,
            "device_labels": self.device_labels,
            "services": self.services,
            "aliases": self.aliases,
            "unit_of_measurement": self.unit_of_measurement,
            "device_class": self.device_class
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Device':
        """Create a Device instance from a dictionary."""
        return cls(
            id=data.get("device_id", ""),
            friendly_name=data.get("friendly_name", ""),
            area_name=data.get("area_name", ""),
            floor_name=data.get("floor_name", ""),
            area_aliases=data.get("area_aliases"),
            floor_aliases=data.get("floor_aliases"),
            domain=data.get("domain", None),
            device_labels=data.get("device_labels", None),
            services=data.get("services", None),
            aliases=data.get("aliases", None),
            unit_of_measurement=data.get("unit_of_measurement", None),
            device_class=data.get("device_class", None)
        )

    def to_embedding_text(self) -> str:
        """Return a string representation of the device for embedding purposes."""
        parts = [ f"Device ID: {self.id}" ]
        self.append_if_exists(parts, "Friendly Name", self.friendly_name)
        self.append_if_exists(parts, "Aliases", self.aliases)
        self.append_if_exists(parts, "Area", self.area_name)
        self.append_if_exists(parts, "Floor", self.floor_name)
        self.append_if_exists(parts, "Area Aliases", self.area_aliases)
        self.append_if_exists(parts, "Floor Aliases", self.floor_aliases)
        self.append_if_exists(parts, "Domain", self.domain)
        self.append_if_exists(parts, "Device Labels", self.device_labels)
        self.append_if_exists(parts, "Unit of Measurement", self.unit_of_measurement)
        self.append_if_exists(parts, "Device Class", self.device_class)

        return " | ".join(parts)

    @staticmethod
    def clean_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
        """Cleans the device attributes by removing excluded keys and those with large JSON representations."""
        cleaned_attributes = attributes.copy()
        for key, value in attributes.items():
            if key in DEVICE_ATTRIBUTES_TO_EXCLUDE:
                cleaned_attributes.pop(key)

            try:
                json_value = json.dumps(value)
                if len(json_value) > DEVICE_ATTRIBUTES_MAX_JSON_LENGTH:
                    cleaned_attributes.pop(key)
            except (TypeError, OverflowError):
                cleaned_attributes.pop(key)

        return cleaned_attributes
