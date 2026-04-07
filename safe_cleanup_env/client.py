from __future__ import annotations

from .environment import SafeCleanupEnvironment
from .models import (
    SafeCleanupAction,
    SafeCleanupObservation,
    SafeCleanupState,
    SafeCleanupStepResult,
)


class SafeCleanupEnvClient:
    def __init__(self, env: SafeCleanupEnvironment | None = None) -> None:
        self._env = env or SafeCleanupEnvironment()

    def reset(self, task_id: str | None = None) -> SafeCleanupObservation:
        return SafeCleanupObservation.from_payload(self._env.reset(task_id))

    def step(self, action: SafeCleanupAction) -> SafeCleanupStepResult:
        return SafeCleanupStepResult.from_payload(self._env.step(action.to_payload()))

    def state(self) -> SafeCleanupState:
        return SafeCleanupState.from_payload(self._env.state())
