"""
GSAS-II API Gateway
Proxies traffic between FastAPI (API) and Streamlit (UI).
Handles the single-port requirement for Hugging Face Spaces.
"""
import uvicorn
import subprocess
import time
import os
import sys
import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState
from pathlib import Path

app = FastAPI()

# Configuration
API_PORT = 8000
UI_PORT = 8501
GATEWAY_PORT = int(os.getenv("PORT", 7860))
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import the API app
sys.path.append(str(PROJECT_ROOT / "scripts"))
from api_app import app as api_app

# Mount API
app.mount("/api", api_app)

# Helper for proxying
client = httpx.AsyncClient(base_url=f"http://localhost:{UI_PORT}")

@app.on_event("startup")
def startup():
    """Start UI and API processes"""
    print(f"Starting Streamlit on port {UI_PORT}...")
    subprocess.Popen([
        "streamlit", "run", "app.py", 
        "--server.port", str(UI_PORT), 
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ], cwd=str(PROJECT_ROOT))
    
    # Wait for Streamlit to start
    # In a real gateway, we'd poll or just handle 503s
    time.sleep(2)

@app.middleware("http")
async def proxy_to_ui(request: Request, call_next):
    """Proxy non-API requests to Streamlit"""
    if request.url.path.startswith("/api") or request.url.path.startswith("/docs") or request.url.path.startswith("/openapi.json"):
        return await call_next(request)
    
    # Proxy the request to Streamlit
    path = request.url.path
    if request.query_params:
        path += f"?{request.query_params}"
    
    # Simple proxying using httpx
    # Note: Streamlit uses WebSockets for the main app logic, 
    # so we MUST handle the websocket separately.
    
    try:
        req = client.build_request(
            request.method,
            path,
            headers=request.headers.raw,
            content=request.stream(),
        )
        res = await client.send(req, stream=True)
        return StreamingResponse(
            res.aiter_raw(),
            status_code=res.status_code,
            headers=res.headers,
            background=None,
        )
    except Exception as e:
        print(f"Proxy error: {e}")
        return StreamingResponse(
            iter([b"Streamlit is starting up or unavailable. Please refresh in a moment."]),
            status_code=503
        )

# WebSocket Proxy for Streamlit
import websockets
import asyncio

@app.websocket("/_stcore/stream")
async def websocket_proxy(websocket: WebSocket):
    await websocket.accept()
    async with websockets.connect(f"ws://localhost:{UI_PORT}/_stcore/stream") as target_ws:
        async def forward_to_ui():
            try:
                while True:
                    data = await websocket.receive_bytes()
                    await target_ws.send(data)
            except:
                pass

        async def forward_to_client():
            try:
                while True:
                    data = await target_ws.recv()
                    await websocket.send_bytes(data)
            except:
                pass

        await asyncio.gather(forward_to_ui(), forward_to_client())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)
