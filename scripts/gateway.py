"""
GSAS-II API Gateway v7.1.4
Robust Proxy for FastAPI (API) and Streamlit (UI) on Hugging Face Spaces.
"""
import uvicorn
import subprocess
import os
import sys
import httpx
import websockets
import asyncio
import logging
import time
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse
from pathlib import Path
from contextlib import asynccontextmanager

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

# --- CONFIGURATION ---
UI_PORT = 8501
GATEWAY_PORT = 7860  # Force 7860 for Hugging Face
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- STREAMLIT STARTUP (EAGER) ---
logger.info("Eagerly starting Streamlit...")
streamlit_proc = subprocess.Popen([
    "streamlit", "run", "app.py",
    "--server.port", str(UI_PORT),
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false",
    "--server.enableWebsocketCompression", "false",
    "--browser.gatherUsageStats", "false"
], cwd=str(PROJECT_ROOT))

# --- API APP DELAYED IMPORT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure Streamlit has a head start
    logger.info("FastAPI Lifespan started. Gateway is ready.")
    yield
    logger.info("Shutting down Gateway...")
    streamlit_proc.terminate()

app = FastAPI(lifespan=lifespan)

# Import API app inside to avoid top-level side effects
sys.path.append(str(PROJECT_ROOT / "scripts"))
try:
    from api_app import app as api_app
    app.mount("/api", api_app)
except Exception as e:
    logger.error(f"Failed to mount API app: {e}")

# HTTP Proxy Client
client = httpx.AsyncClient(base_url=f"http://localhost:{UI_PORT}", timeout=None)

# --- ROUTES ---

@app.get("/health")
async def gateway_health():
    return {"status": "gateway_running", "streamlit_pid": streamlit_proc.pid}

@app.get("/", response_class=HTMLResponse)
async def root_proxy(request: Request):
    """
    Directly handle root to satisfy health checks and avoid loops.
    We proxy it manually to ensure it works.
    """
    try:
        res = await client.get("/")
        return HTMLResponse(content=res.text, status_code=res.status_code)
    except:
        return HTMLResponse(content="<h1>GSAS-II Pipeline is starting...</h1><script>setTimeout(()=>location.reload(), 2000)</script>", status_code=503)

@app.middleware("http")
async def proxy_http_to_ui(request: Request, call_next):
    path = request.url.path
    
    # Do not proxy API, Health, or Docs
    if any(path.startswith(prefix) for prefix in ["/api", "/docs", "/redoc", "/openapi.json", "/health"]):
        return await call_next(request)
    
    if path == "/":
        return await call_next(request)

    # Proxy everything else
    query = str(request.query_params)
    url_path = f"{path}?{query}" if query else path
    
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("connection", None)
    
    try:
        req = client.build_request(
            request.method,
            url_path,
            headers=headers,
            content=request.stream()
        )
        res = await client.send(req, stream=True)
        
        res_headers = dict(res.headers)
        res_headers.pop("content-length", None)
        res_headers.pop("transfer-encoding", None)
        res_headers.pop("connection", None)
        
        return StreamingResponse(
            res.aiter_raw(),
            status_code=res.status_code,
            headers=res_headers
        )
    except Exception as e:
        return StreamingResponse(
            iter([b"Linking to Streamlit..."]),
            status_code=503
        )

@app.websocket("/_stcore/stream")
async def proxy_websocket_to_ui(websocket: WebSocket):
    await websocket.accept()
    
    headers = []
    origin = websocket.headers.get("origin")
    if origin:
        headers.append(("Origin", origin))
    
    try:
        async with websockets.connect(
            f"ws://localhost:{UI_PORT}/_stcore/stream",
            extra_headers=headers,
            open_timeout=10
        ) as target_ws:
            
            async def forward_to_ui():
                try:
                    while True:
                        data = await websocket.receive()
                        if "text" in data:
                            await target_ws.send(data["text"])
                        elif "bytes" in data:
                            await target_ws.send(data["bytes"])
                        elif data.get("type") == "websocket.disconnect":
                            break
                except:
                    pass

            async def forward_to_client():
                try:
                    async for message in target_ws:
                        if isinstance(message, str):
                            await websocket.send_text(message)
                        else:
                            await websocket.send_bytes(message)
                except:
                    pass

            await asyncio.gather(forward_to_ui(), forward_to_client())
            
    except Exception as e:
        logger.error(f"WS Proxy Error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT, log_level="info")
