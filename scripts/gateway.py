"""
GSAS-II API Gateway v7.1.3
Unified Gateway for FastAPI (API) and Streamlit (UI).
Optimized for Hugging Face Spaces.
"""
import uvicorn
import subprocess
import os
import sys
import httpx
import websockets
import asyncio
import logging
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pathlib import Path
from contextlib import asynccontextmanager

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

# Configuration
UI_PORT = 8501
GATEWAY_PORT = int(os.getenv("PORT", 7860))
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import the API app
sys.path.append(str(PROJECT_ROOT / "scripts"))
from api_app import app as api_app

# HTTP Proxy Client
client = httpx.AsyncClient(base_url=f"http://localhost:{UI_PORT}", timeout=None)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the Gateway and Streamlit"""
    logger.info(f"Starting Streamlit on internal port {UI_PORT}...")
    
    # We use a very permissive configuration for Streamlit to work behind double proxies
    cmd = [
        "streamlit", "run", "app.py",
        "--server.port", str(UI_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false"
    ]
    
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    
    yield
    
    logger.info("Shutting down Streamlit...")
    proc.terminate()
    proc.wait()

app = FastAPI(lifespan=lifespan)

# Mount API Endpoints
app.mount("/api", api_app)

@app.middleware("http")
async def proxy_http_to_ui(request: Request, call_next):
    """Middleware to proxy all non-API HTTP traffic to Streamlit"""
    path = request.url.path
    
    # Do not proxy API or Docs
    if any(path.startswith(prefix) for prefix in ["/api", "/docs", "/redoc", "/openapi.json"]):
        return await call_next(request)
    
    # Prepare proxied request
    query = str(request.query_params)
    url_path = f"{path}?{query}" if query else path
    
    # Clean headers
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("connection", None)
    
    try:
        req = client.build_request(
            request.method,
            url_path,
            headers=headers,
            content=request.stream()
        )
        res = await client.send(req, stream=True)
        
        # Clean response headers
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
        logger.error(f"HTTP Proxy Error: {e}")
        return StreamingResponse(
            iter([b"Streamlit is starting... please wait."]),
            status_code=503
        )

@app.websocket("/_stcore/stream")
async def proxy_websocket_to_ui(websocket: WebSocket):
    """Bi-directional WebSocket proxy for Streamlit's real-time engine"""
    await websocket.accept()
    logger.info("WebSocket connection established with client")
    
    # Attempt to pass through the Origin header
    headers = []
    origin = websocket.headers.get("origin")
    if origin:
        headers.append(("Origin", origin))
    
    try:
        async with websockets.connect(
            f"ws://localhost:{UI_PORT}/_stcore/stream",
            extra_headers=headers
        ) as target_ws:
            logger.info("WebSocket connection established with internal Streamlit")
            
            async def forward_to_ui():
                try:
                    while True:
                        # FastAPI receives messages as dicts
                        data = await websocket.receive()
                        if "text" in data:
                            await target_ws.send(data["text"])
                        elif "bytes" in data:
                            await target_ws.send(data["bytes"])
                        elif data.get("type") == "websocket.disconnect":
                            break
                except Exception as e:
                    logger.debug(f"WS Forward to UI closed: {e}")

            async def forward_to_client():
                try:
                    async for message in target_ws:
                        if isinstance(message, str):
                            await websocket.send_text(message)
                        else:
                            await websocket.send_bytes(message)
                except Exception as e:
                    logger.debug(f"WS Forward to Client closed: {e}")

            # Run both forwarding tasks concurrently
            await asyncio.gather(forward_to_ui(), forward_to_client())
            
    except Exception as e:
        logger.error(f"WebSocket Proxy Error: {e}")
    finally:
        logger.info("WebSocket connection closed")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT, log_level="info")
