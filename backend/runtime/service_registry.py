from __future__ import annotations

from ..dependencies import AppServices, build_services
from ..services.runtime_service import RuntimeService


def create_service_registry(runtime: RuntimeService) -> AppServices:
    return build_services(runtime)
