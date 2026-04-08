from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import FilesCleanUpAction, FilesCleanUpObservation
from safe_cleanup_env.environment import SafeCleanupEnvironment


class FilesCleanUpEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._core = SafeCleanupEnvironment()
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> FilesCleanUpObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        payload = self._core.reset(kwargs.get("task_id") or "desktop_cleanup_easy")
        return self._observation(payload)

    def step(self, action: FilesCleanUpAction) -> FilesCleanUpObservation:  # type: ignore[override]
        result = self._core.step(action.model_dump(exclude_none=True))
        self._state.step_count = result["observation"]["step_count"]
        return self._observation(
            result["observation"],
            reward=result["reward"],
            done=result["done"],
            metadata=result["info"],
        )

    @property
    def state(self) -> State:
        return self._state

    @staticmethod
    def _observation(
        payload: dict,
        reward: float | None = 0.0,
        done: bool | None = None,
        metadata: dict | None = None,
    ) -> FilesCleanUpObservation:
        clean_payload = dict(payload)
        payload_done = clean_payload.pop("done", False)
        payload_reward = clean_payload.pop("reward", None)
        payload_metadata = clean_payload.pop("metadata", {})
        return FilesCleanUpObservation(
            **clean_payload,
            done=payload_done if done is None else done,
            reward=payload_reward if reward is None else reward,
            metadata=payload_metadata if metadata is None else metadata,
        )
