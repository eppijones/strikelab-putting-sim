# StrikeLab Putting Simulator

Real-time golf putting simulator with:

- Python / FastAPI / OpenCV backend
- React / TypeScript / Three.js frontend
- Protected tracking core kept intact
- Thin composition root plus modular routers, services, runtime, and websocket layers

## Project Structure

```text
backend/
  main.py             Thin backend entrypoint
  legacy_main.py      Compatibility module containing existing runtime logic
  app_factory.py      FastAPI app creation and lifespan wiring
  dependencies.py     Shared dependency accessors
  routers/            HTTP route modules
  runtime/            Runtime orchestration boundaries
  services/           Injected service facades
  ws/                 WebSocket transport and protocol builders
  protocol/           Versioned protocol models
  cameras/            Camera abstractions and adapters

frontend/
  src/
    contexts/         Compatibility provider facade
    hooks/            Split websocket / health / REST / UI hooks
    types/            Shared frontend protocol and state types
    config/           Backend URL configuration

tests/
  test_regression.py  Replay-based tracker regression checks
  test_hardening.py   Existing backend hardening checks
  test_api_http.py    Route parity smoke tests
  test_api_ws.py      WebSocket v1/v2 contract smoke tests

tools/
  benchmark_ws_payload.py
  benchmark_backend_runtime.py
```

## Setup

### Backend

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### Frontend

```powershell
cd frontend
npm install
```

## Run

### Full app from repo root

Starts backend first, waits for `/api/health`, then starts the frontend dev server.

```powershell
python main.py
```

### Backend only

```powershell
python main.py --no-frontend
```

### Backend module directly

```powershell
python -m backend.main
```

### Webcam mode

```powershell
python main.py --webcam
```

### Replay mode

```powershell
python main.py --replay "path\\to\\video.mp4"
```

### Debug logging

```powershell
python main.py --debug
```

## Frontend Backend URLs

The frontend now uses centralized config in `frontend/src/config/backend.ts`.

Optional env vars:

```powershell
VITE_BACKEND_HTTP_URL=http://localhost:8000
VITE_BACKEND_WS_URL=ws://localhost:8000/ws
```

## Validation

```powershell
pytest tests/test_api_http.py tests/test_api_ws.py tests/test_hardening.py -q
cd frontend
npm run build
```

## Notes

- WebSocket v2 exists behind feature flags in `backend/config.py`.
- Existing `/api/*` routes remain unchanged.
- Existing `/ws` behavior remains the default unless v2 is enabled.
- Protected tracking files were not modified as part of this refactor.
