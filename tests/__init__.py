import importlib.util
import sys
import types
from pathlib import Path


if importlib.util.find_spec("homeassistant") is None:
    integration_path = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_ragent"
    )
    package_name = "custom_components.ha_ragent"
    if package_name not in sys.modules:
        integration_package = types.ModuleType(package_name)
        integration_package.__path__ = [str(integration_path)]
        integration_package.__package__ = package_name
        sys.modules[package_name] = integration_package
