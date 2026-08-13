"""Configuration management with schema validation and external configurability."""

import json
import yaml
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Type, TypeVar
from ..core.errors import ConfigError
from ..core.logging import logger

C = TypeVar("C", bound="Config")


class ConfigValidator:
    """Validates configuration values against a schema."""

    @staticmethod
    def check_type(value: Any, expected_type: type, name: str, path: str = "") -> None:
        if not isinstance(value, expected_type):
            raise ConfigError(
                f"Config '{name}' at '{path}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    @staticmethod
    def check_positive(value: Union[int, float], name: str, path: str = "") -> None:
        if value <= 0:
            raise ConfigError(f"Config '{name}' at '{path}' must be positive, got {value}")

    @staticmethod
    def check_non_negative(value: Union[int, float], name: str, path: str = "") -> None:
        if value < 0:
            raise ConfigError(f"Config '{name}' at '{path}' must be non-negative, got {value}")

    @staticmethod
    def check_range(value: Union[int, float], lo: Union[int, float], hi: Union[int, float], name: str, path: str = "") -> None:
        if value < lo or value > hi:
            raise ConfigError(f"Config '{name}' at '{path}' must be in [{lo}, {hi}], got {value}")

    @staticmethod
    def check_in(value: Any, options: List[Any], name: str, path: str = "") -> None:
        if value not in options:
            raise ConfigError(f"Config '{name}' at '{path}' must be one of {options}, got {value}")


class Config:
    """Base configuration class with serialization, validation, and merging."""

    _schema: Dict[str, Dict[str, Any]] = {}

    def __init__(self, **kwargs):
        self._init_defaults()
        self._validate_and_set(kwargs)

    def _init_defaults(self) -> None:
        for field_name, field_spec in self._schema.items():
            if "default" in field_spec:
                default = field_spec["default"]
                setattr(self, field_name, copy.deepcopy(default))
            elif "required" in field_spec and field_spec["required"]:
                pass  # Will be set in validate_and_set
            else:
                setattr(self, field_name, None)

    def _validate_and_set(self, kwargs: Dict[str, Any]) -> None:
        for key, value in kwargs.items():
            if key not in self._schema:
                raise ConfigError(f"Unknown configuration field: '{key}'")
            spec = self._schema[key]
            if spec.get("validate"):
                spec["validate"](value, key)
            setattr(self, key, value)

        for field_name, field_spec in self._schema.items():
            if field_spec.get("required", False):
                if getattr(self, field_name, None) is None:
                    raise ConfigError(f"Required configuration field '{field_name}' was not provided")

    def validate_complete(self) -> None:
        """Validate all fields have acceptable values."""
        for field_name, field_spec in self._schema.items():
            value = getattr(self, field_name, None)
            if field_spec.get("required", False) and value is None:
                raise ConfigError(f"Required config field '{field_name}' is missing")
            if value is not None and "validate" in field_spec:
                field_spec["validate"](value, field_name)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        result = {}
        for field_name in self._schema:
            value = getattr(self, field_name, None)
            if hasattr(value, "to_dict"):
                result[field_name] = value.to_dict()
            elif isinstance(value, Config):
                result[field_name] = value.to_dict()
            else:
                result[field_name] = value
        return result

    @classmethod
    def from_dict(cls: Type[C], data: Dict[str, Any]) -> C:
        """Create configuration from a dictionary."""
        subconfigs = {}
        for field_name, field_spec in cls._schema.items():
            if field_name in data:
                sub_type = field_spec.get("type")
                if sub_type and isinstance(sub_type, type) and issubclass(sub_type, Config):
                    if isinstance(data[field_name], dict):
                        subconfigs[field_name] = sub_type.from_dict(data[field_name])
        kwargs = {k: v for k, v in data.items() if k not in subconfigs}
        kwargs.update(subconfigs)
        return cls(**kwargs)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON or YAML file."""
        path = Path(path)
        data = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml"):
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls: Type[C], path: Union[str, Path]) -> C:
        """Load configuration from a JSON or YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls.from_dict(data)

    def merge(self, other: "Config") -> "Config":
        """Merge another configuration into this one (in-place)."""
        for field_name in self._schema:
            other_val = getattr(other, field_name, None)
            if other_val is not None:
                current = getattr(self, field_name, None)
                if isinstance(current, Config) and isinstance(other_val, Config):
                    current.merge(other_val)
                else:
                    setattr(self, field_name, other_val)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()})"
