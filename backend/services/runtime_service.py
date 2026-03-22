from __future__ import annotations

from typing import Any, Optional


class RuntimeService:
    """Small façade around the live runtime instance."""

    def __init__(self, app_instance: Optional[Any] = None, manage_lifecycle: bool = True):
        self._app_instance = app_instance
        self.manage_lifecycle = manage_lifecycle

    def set_app(self, app_instance: Any) -> None:
        self._app_instance = app_instance

    def clear_app(self) -> None:
        self._app_instance = None

    @property
    def app_instance(self) -> Optional[Any]:
        return self._app_instance

    def get_app(self) -> Any:
        if self._app_instance is None:
            raise RuntimeError("App not initialized")
        return self._app_instance

    def has_app(self) -> bool:
        return self._app_instance is not None

    def start(self) -> None:
        if self.manage_lifecycle and self._app_instance is not None:
            self._app_instance.start()

    def stop(self) -> None:
        if self.manage_lifecycle and self._app_instance is not None:
            self._app_instance.stop()
