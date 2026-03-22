from __future__ import annotations

from .runtime_service import RuntimeService


class DrillService:
    def __init__(self, runtime: RuntimeService):
        self._runtime = runtime

    @property
    def runtime(self):
        return self._runtime.get_app()

    @property
    def drill_manager(self):
        return self.runtime.drill_manager
