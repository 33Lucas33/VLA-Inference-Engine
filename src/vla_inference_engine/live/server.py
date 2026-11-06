"""Local browser API for one interactive LIBERO session."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from vla_inference_engine.benchmark.runtime import project_root
from vla_inference_engine.live.catalog import models, tasks
from vla_inference_engine.live.events import Selection
from vla_inference_engine.live.session import SessionManager

ROOT = project_root()
STATIC = Path(__file__).parent / "static"
manager = SessionManager(ROOT)
viewers: dict[str, int] = {}


async def pause_unwatched(session_id: str) -> None:
    await asyncio.sleep(3)
    if viewers.get(session_id, 0):
        return
    try:
        session = manager.get(session_id)
        if session.snapshot()["state"] == "RUNNING":
            session.send("pause", uuid.uuid4().hex)
    except (ValueError, BrokenPipeError):
        pass


async def index(request):
    return FileResponse(STATIC / "index.html")


async def asset(request):
    name = request.path_params["name"]
    if name not in {"app.js", "style.css"}:
        return JSONResponse({"error": "Unknown asset"}, status_code=404)
    return FileResponse(STATIC / name)


async def catalog(request):
    return JSONResponse({"tasks": tasks(), "models": models(ROOT)})


async def create(request):
    try:
        selection = Selection.parse(await request.json())
        session = await asyncio.to_thread(manager.create, selection)
        return JSONResponse(session.snapshot(), status_code=201)
    except (ValueError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def status(request):
    try:
        return JSONResponse(manager.get(request.path_params["session_id"]).snapshot())
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)


async def command(request):
    try:
        session = manager.get(request.path_params["session_id"])
        payload = await request.json()
        name = str(payload["command"])
        if name == "reset":
            return JSONResponse((await asyncio.to_thread(manager.reset, session.id)).snapshot())
        command_id = str(payload["command_id"])
        return JSONResponse(session.send(name, command_id))
    except (ValueError, KeyError, TypeError, BrokenPipeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def stream(ws: WebSocket):
    await ws.accept()
    session_id = ws.path_params["session_id"]
    viewers[session_id] = viewers.get(session_id, 0) + 1
    try:
        session = manager.get(session_id)
        previous = -1
        while True:
            snapshot = session.snapshot()
            if snapshot["event_seq"] != previous:
                await ws.send_json(snapshot)
                previous = snapshot["event_seq"]
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass
    except ValueError:
        await ws.close()
    finally:
        viewers[session_id] = max(0, viewers.get(session_id, 1) - 1)
        if viewers[session_id] == 0:
            asyncio.create_task(pause_unwatched(session_id))


@asynccontextmanager
async def lifespan(_app):
    yield
    await asyncio.to_thread(manager.close)


app = Starlette(lifespan=lifespan, routes=[
    Route("/", index), Route("/static/{name}", asset),
    Route("/api/catalog", catalog), Route("/api/session", create, methods=["POST"]),
    Route("/api/session/{session_id}", status),
    Route("/api/session/{session_id}/command", command, methods=["POST"]),
    WebSocketRoute("/api/session/{session_id}/stream", stream),
])
