from __future__ import annotations

from .runtime_service import RuntimeService


class StatsService:
    def __init__(self, runtime: RuntimeService):
        self._runtime = runtime

    @property
    def runtime(self):
        return self._runtime.get_app()

    @property
    def session_manager(self):
        return self.runtime.session_manager
