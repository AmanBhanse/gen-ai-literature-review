# config_module/__init__.py
"""Configuration module - settings and constants."""

from config_module.settings import Settings
from config_module.constants import Constants, default_constants

# Validate settings on import
Settings.validate()

__all__ = [
    "Settings",
    "Constants",
    "default_constants",
]
