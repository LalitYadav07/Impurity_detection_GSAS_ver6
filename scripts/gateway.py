"""
GSAS-II API Gateway
Proxies traffic between FastAPI (API) and Streamlit (UI).
Handles the single-port requirement for Hugging Face Spaces.
"""
import uvicorn
import subprocess
import os
import sys
import httpx
import websockets
import asyncio
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import StreamingResponse
from pathlib import Path
from contextlib import asynccontextmanager

# Configuration
API_PORT = 8000
UI_PORT = 8501
GATEWAY_PORT = int(os.getenv("PORT", 7860))
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import the API app
sys.path.append(str(PROJECT_ROOT / "scripts"))
from api_app import app as api_app

# Helper for proxying
client = httpx.AsyncClient(base_url=f"http://localhost:{UI_PORT}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start Streamlit as a background process"""
    print(f"Starting Streamlit on port {UI_PORT}...")
    proc = subprocess.Popen([
        "streamlit", "run", "app.py", 
        "--server.port", str(UI_PORT), 
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableXsrfProtection", "false"
    ], cwd=str(PROJECT_ROOT))
    
    yield
    
    proc.terminate()

app = FastAPI(lifespan=lifespan)

# Mount API
app.mount("/api", api_app)

@app.middleware("http")
async def proxy_to_ui(request: Request, call_next):
    """Proxy non-API requests to Streamlit"""
    # Skip proxy for API, Docs, and OpenAPI
    if request.url.path.startswith("/api") or \
       request.url.path.startswith("/docs") or \
       request.url.path.startswith("/redoc") or \
       request.url.path.startswith("/openapi.json"):
        return await call_next(request)
    
    # Proxy the request to Streamlit
    path = request.url.path
    if request.query_params:
        path += f"?{request.query_params}"
    
    try:
        # Build request to internal Streamlit
        # We strip some headers that might cause issues in double-proxied environments
        headers = dict(request.headers)
        headers.pop("host", None) 
        headers.pop("content-length", None)
        
        req = client.build_request(
            request.method,
            path,
            headers=headers,
            content=request.stream(),
        )
        res = await client.send(req, stream=True)
        
        # Strip hop-by-hop headers
        res_headers = dict(res.headers)
        res_headers.pop("content-length", None)
        res_headers.pop("transfer-encoding", None)
        res_headers.pop("connection", None)

        return StreamingResponse(
            res.aiter_raw(),
            status_code=res.status_code,
            headers=res_headers,
        )
    except Exception as e:
        print(f"Proxy error: {e}")
        return StreamingResponse(
            iter([b"Streamlit is starting up or unavailable. Please refresh in a moment."]),
            status_code=503
        )

@app.websocket("/_stcore/stream")
async def websocket_proxy(websocket: WebSocket):
    await websocket.accept()
    
    # Attempt to pass through some headers (Streamlit might check Origin)
    headers = []
    orig = websocket.headers.get("origin")
    if orig:
        headers.append(("Origin", orig))

    try:
        async with websockets.connect(
            f"ws://localhost:{UI_PORT}/_stcore/stream",
            extra_headers=headers
        ) as target_ws:
            
            async def forward_to_ui():
                try:
                    while True:
                        msg = await websocket.receive()
                        if "text" in msg:
                            await target_ws.send(msg["text"])
                        elif "bytes" in msg:
                            await target_ws.send(msg["bytes"])
                except:
                    pass

            async def forward_to_client():
                try:
                    async for msg in target_ws:
                        if isinstance(msg, str):
                            await websocket.send_text(msg)
                        else:
                            await websocket.send_bytes(msg)
                except:
                    pass

            await asyncio.gather(forward_to_ui(), forward_to_client())
    except Exception as e:
        print(f"WebSocket Proxy error: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)
