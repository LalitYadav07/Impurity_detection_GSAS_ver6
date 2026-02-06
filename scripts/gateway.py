"""
GSAS-II API Gateway v7.1.6
Safe-Mode Proxy for Hugging Face Spaces.
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

# --- FORCE LOGGING TO STDOUT ---
print("GATEWAY [INIT]: Starting v7.1.6...", flush=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("gateway")
logger.info("Initializing Gateway Logger...")

# --- CONFIGURATION ---
UI_PORT = 8501
GATEWAY_PORT = int(os.getenv("PORT", 7860))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger.info(f"Project Root: {PROJECT_ROOT}")

# Global state
streamlit_proc = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global streamlit_proc
    logger.info("Lifespan: Starting background tasks...")
    
    # Eager Streamlit startup
    logger.info("Lifespan: Launching Streamlit...")
    try:
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
        logger.info(f"Streamlit started with PID {streamlit_proc.pid}")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to launch Streamlit: {e}")

    yield

    if streamlit_proc:
        logger.info("Lifespan: Terminating Streamlit...")
        streamlit_proc.terminate()
        streamlit_proc.wait()
    logger.info("Lifespan: Shutdown complete.")

app = FastAPI(lifespan=lifespan)

# --- API MOUNTING (Defensive) ---
logger.info("Gateway: Mounting API routes...")
sys.path.append(str(PROJECT_ROOT / "scripts"))
try:
    from api_app import app as api_app
    app.mount("/api", api_app)
    logger.info("API App mounted successfully at /api")
except Exception as e:
    logger.critical(f"FATAL: Could not import or mount api_app.py: {e}")
    # Don't crash the whole gateway, just serve an error on /api
    @app.get("/api/{path:path}")
    async def api_error(path: str):
        return {"error": "API failed to load", "detail": str(e)}

# HTTP Proxy Client
client = httpx.AsyncClient(base_url=f"http://localhost:{UI_PORT}", timeout=None)

# --- CORE ROUTES ---

@app.get("/health")
async def health():
    st_status = "alive" if streamlit_proc and streamlit_proc.poll() is None else "dead"
    return {
        "gateway": "running",
        "streamlit": st_status,
        "pid": os.getpid(),
        "st_pid": streamlit_proc.pid if streamlit_proc else None
    }

@app.get("/", response_class=HTMLResponse)
async def gateway_root(request: Request):
    try:
        res = await client.get("/")
        return HTMLResponse(content=res.text, status_code=res.status_code)
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>System Starting...</h1><p>Please wait while the GSAS-II pipeline initializes.</p><script>setTimeout(()=>location.reload(), 3000)</script>",
            status_code=503
        )

@app.middleware("http")
async def proxy_middleware(request: Request, call_next):
    path = request.url.path
    if any(path.startswith(p) for p in ["/api", "/docs", "/redoc", "/openapi.json", "/health"]):
        return await call_next(request)
    if path == "/":
        return await call_next(request)

    # Proxy to Streamlit
    try:
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("connection", None)
        
        url = request.url.path
        if request.query_params:
            url += f"?{request.query_params}"
            
        req = client.build_request(
            request.method,
            url,
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
        return StreamingResponse(iter([b"Gateway connecting..."]), status_code=503)

@app.websocket("/_stcore/stream")
async def websocket_proxy(websocket: WebSocket):
    await websocket.accept()
    logger.info("WS: Client connected")
    
    headers = []
    origin = websocket.headers.get("origin")
    if origin:
        headers.append(("Origin", origin))
    
    try:
        async with websockets.connect(
            f"ws://localhost:{UI_PORT}/_stcore/stream",
            extra_headers=headers,
            open_timeout=20
        ) as target_ws:
            logger.info("WS: Internal Streamlit connected")
            
            async def from_client():
                try:
                    while True:
                        msg = await websocket.receive()
                        if "text" in msg:
                            await target_ws.send(msg["text"])
                        elif "bytes" in msg:
                            await target_ws.send(msg["bytes"])
                        elif msg.get("type") == "websocket.disconnect":
                            break
                except: pass

            async def to_client():
                try:
                    async for msg in target_ws:
                        if isinstance(msg, str):
                            await websocket.send_text(msg)
                        else:
                            await websocket.send_bytes(msg)
                except: pass

            await asyncio.gather(from_client(), to_client())
    except Exception as e:
        logger.error(f"WS: Error: {e}")
    finally:
        logger.info("WS: Client disconnected")
        try: await websocket.close()
        except: pass

if __name__ == "__main__":
    logger.info(f"Starting Uvicorn on port {GATEWAY_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT, log_level="info", access_log=False)
