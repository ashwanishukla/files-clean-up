from __future__ import annotations

from ..environment import SafeCleanupEnvironment

try:
    from openenv.core.env_server import Environment as OpenEnvBaseEnvironment
except ImportError:  # pragma: no cover - optional runtime path
    class OpenEnvBaseEnvironment:  # type: ignore[no-redef]
        pass


class OpenEnvSafeCleanupEnvironment(OpenEnvBaseEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, seed: int = 11, max_steps: int = 24) -> None:
        self._core = SafeCleanupEnvironment(seed=seed, max_steps=max_steps)

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs):
        task_id = kwargs.get("task_id")
        return self._core.reset(task_id=task_id)

    def step(self, action, timeout_s: float | None = None, **kwargs):
        if hasattr(action, "model_dump"):
            payload = action.model_dump()
        elif isinstance(action, dict):
            payload = action
        else:
            payload = dict(action.__dict__)
        return self._core.step(payload)

    @property
    def state(self):
        return self._core.state()
