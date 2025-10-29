from typing import Any, Callable

from agents import liascript_generator

_REGISTRY: dict[str, Callable[[], Any]] = {
    "liascript-generator": liascript_generator.build_agent
}


def list_agents():
    return sorted(_REGISTRY.keys())


def load_agent(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown agent '{name}'. Available: {list_agents()}")
    return _REGISTRY[name]()
