from typing import Dict, Type

from glass.systems.base import SystemUnderTest

_REGISTRY: Dict[str, Type[SystemUnderTest]] = {}


def register(name: str):
    def decorator(cls: Type[SystemUnderTest]):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_system_class(name: str) -> Type[SystemUnderTest]:
    if name not in _REGISTRY:
        raise ValueError(f"System '{name}' not found in registry. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]
