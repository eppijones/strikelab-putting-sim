from __future__ import annotations

from .runtime_service import RuntimeService


class GameService:
    def __init__(self, runtime: RuntimeService):
        self._runtime = runtime

    @property
    def runtime(self):
        return self._runtime.get_app()

    @property
    def game_logic(self):
        return self.runtime.game_logic
