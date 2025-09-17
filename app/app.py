import os
import io
import time
import uuid
import json
import base64
import cv2
import numpy as np
from typing import Optional, Dict, Set

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except Exception:
    pass
from settings import *
from embedder import load_embedder, get_embedding
from embedder import get_embedder_info
from retinaface_wrapper import detect_and_align
from retinaface_wrapper import detect_with_bbox_and_align
from stream_manager import StreamManager

# ===== App init =====
app = FastAPI(title="FaceVerify", version="1.0")

if CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

# Resolve STATIC_DIR to absolute path if needed (relative to repo root)
_RESOLVED_STATIC_DIR = STATIC_DIR
if _RESOLVED_STATIC_DIR and not os.path.isabs(_RESOLVED_STATIC_DIR):
    _RESOLVED_STATIC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", _RESOLVED_STATIC_DIR))
# Fallback: use repo's frontend/dist if available
if not _RESOLVED_STATIC_DIR or not os.path.isdir(_RESOLVED_STATIC_DIR):
    _fallback_dist = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
    if os.path.isdir(_fallback_dist):
        _RESOLVED_STATIC_DIR = _fallback_dist

# Mount static UI early
if _RESOLVED_STATIC_DIR and os.path.isdir(_RESOLVED_STATIC_DIR):
    print(f"Serving static UI from: {_RESOLVED_STATIC_DIR}")
    assets_dir = os.path.join(_RESOLVED_STATIC_DIR, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    index_path = os.path.join(_RESOLVED_STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        @app.get("/")
        def root_index():
            return FileResponse(index_path)

EMBEDDER = load_embedder(DEVICE)

# ===== WebSocket connections =====
class WSManager:
    def __init__(self):
        self.channels: Dict[str, Set[WebSocket]] = {}

    async def connect(self, key: str, ws: WebSocket):
        await ws.accept()
        self.channels.setdefault(key, set()).add(ws)

    def disconnect(self, key: str, ws: WebSocket):
        try:
            self.channels.get(key, set()).discard(ws)
        except Exception:
            pass

    async def broadcast(self, key: str, data: dict):
        dead = []
        for ws in list(self.channels.get(key, set())):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(key, ws)

ws_manager = WSManager()

# Callback for stream manager → websockets
async def _broadcast_async(sid: str, payload: dict):
    await ws_manager.broadcast(sid, payload)

def _on_score(sid: str, payload: dict):
    import asyncio
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast_async(sid, payload), loop)

STREAMS = StreamManager(on_score=_on_score, max_streams=MAX_STREAMS)

# ===== Helpers =====
def _ensure_img(f: UploadFile) -> np.ndarray:
    if f.content_type not in ALLOWED_IMG_MIMES:
        raise HTTPException(415, "Only JPEG/PNG images are supported")
    raw = f.file.read()
    if len(raw) > IMG_MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {IMG_MAX_MB}MB")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image payload")
    return img


def _scale(sim: float) -> int:
    return int(round(((sim + 1.0) / 2.0) * 9999))


def _save_audit(kind: str, payload: bytes, ext: str, retention: int) -> str:
    os.makedirs(RETENTION_ROOT, exist_ok=True)
    jid = str(uuid.uuid4())
    path = os.path.join(RETENTION_ROOT, f"{jid}.{ext}")
    with open(path, "wb") as f:
        f.write(payload)
    with open(path + ".meta", "w") as m:
        m.write(f"expires_at={time.time() + retention*24*3600}\n")
        m.write(f"kind={kind}\n")
    return jid


def _cleanup_expired():
    if not os.path.isdir(RETENTION_ROOT):
        return
    now = time.time()
    for name in os.listdir(RETENTION_ROOT):
        if name.endswith(".meta"):
            meta = os.path.join(RETENTION_ROOT, name)
            try:
                with open(meta) as m:
                    lines = dict(l.strip().split("=", 1) for l in m if "=" in l)
                if float(lines.get("expires_at", "0")) < now:
                    data_file = meta[:-5]
                    for p in (meta, data_file):
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except Exception:
                            pass
            except Exception:
                continue

# ===== Health =====
@app.get(f"{API_PREFIX}/healthz")
def healthz():
    _cleanup_expired()
    return {"ok": True}

@app.get(f"{API_PREFIX}/readyz")
def readyz():
    return {"ok": True}

# ===== Image verify =====
@app.post(f"{API_PREFIX}/verify-images")
async def verify_images(
    img1: UploadFile = File(...),
    img2: UploadFile = File(...),
    keep_for_audit: bool = Form(False),
    retention_days: int = Form(0),
    profile: str = Form("1in10k"),
):
    t0 = time.time()
    if keep_for_audit and retention_days not in ALLOWED_RETENTION:
        raise HTTPException(400, f"retention_days must be one of {sorted(ALLOWED_RETENTION)}")
    im1 = _ensure_img(img1)
    im2 = _ensure_img(img2)
    f1 = detect_and_align(im1)
    f2 = detect_and_align(im2)
    if f1 is None or f2 is None:
        raise HTTPException(422, "No face detected in one or both images")
    e1 = get_embedding(f1)
    e2 = get_embedding(f2)
    sim = float(np.dot(e1, e2))  # cosine because embeddings are L2-normalized
    thr = THRESHOLDS.get(profile, THRESHOLDS["1in10k"])
    print(f"similarity: {sim}, threshold: {thr}")
    decision = "match" if sim >= thr else "no_match"
    if keep_for_audit:
        _, b1 = cv2.imencode(".jpg", im1)
        _, b2 = cv2.imencode(".jpg", im2)
        _save_audit("image", b1.tobytes(), "jpg", retention_days)
        _save_audit("image", b2.tobytes(), "jpg", retention_days)
    return {
        "similarity_raw": sim,
        "similarity_scaled": _scale(sim),
        "decision": decision,
        "threshold_used": thr,
        "latency_ms": int((time.time() - t0) * 1000),
    }

# ===== Video verify =====
class VideoCfg(BaseModel):
    sampling_fps: float = 3.0
    topk: int = 3
    profile: str = "1in10k"

JOBS: Dict[str, dict] = {}

@app.post(f"{API_PREFIX}/verify-video")
async def verify_video(
    video: UploadFile = File(...),
    ref_image: UploadFile = File(...),
    keep_for_audit: bool = Form(False),
    retention_days: int = Form(0),
    cfg: VideoCfg = VideoCfg(),
):
    # Require a reference image for video similarity
    if "video" not in (video.content_type or ""):
        raise HTTPException(415, "Only video/* files accepted")
    if ref_image is None:
        raise HTTPException(400, "ref_image is required for video verification")
    if keep_for_audit and retention_days not in ALLOWED_RETENTION:
        raise HTTPException(400, f"retention_days must be one of {sorted(ALLOWED_RETENTION)}")

    raw = await video.read()
    job_id = str(uuid.uuid4())
    path = f"/tmp/{job_id}.mp4"
    with open(path, "wb") as f:
        f.write(raw)

    im = _ensure_img(ref_image)
    ref_face = detect_and_align(im)
    if ref_face is None:
        raise HTTPException(422, "No face detected in ref_image")
    ref_emb = get_embedding(ref_face)

    import threading

    def _process():
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(fps / max(cfg.sampling_fps, 0.5))))
        frame_id, results = 0, []
        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_id % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                face = detect_and_align(frame)
                if face is not None:
                    emb = get_embedding(face)
                    sim = float(np.dot(emb, ref_emb))
                    results.append(
                        {"t_sec": frame_id / fps, "similarity_raw": sim, "similarity_scaled": _scale(sim)}
                    )
            frame_id += 1
        cap.release()
        results.sort(key=lambda x: x["similarity_raw"], reverse=True)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["topk"] = results[: cfg.topk]
        # Do not delete the video immediately; preview endpoint may need it.
        # A periodic cleanup can remove old files later.
        # try:
        #     os.remove(path)
        # except Exception:
        #     pass
        if keep_for_audit:
            os.makedirs(RETENTION_ROOT, exist_ok=True)
            with open(os.path.join(RETENTION_ROOT, f"{job_id}.json"), "w") as f:
                json.dump({"created": time.time(), "results": JOBS[job_id]}, f)

    JOBS[job_id] = {"status": "running", "video_path": path, "ref_emb": ref_emb.tolist()}
    threading.Thread(target=_process, daemon=True).start()
    return {"job_id": job_id, "status": "queued"}

@app.get(f"{API_PREFIX}/jobs/{{job_id}}")
def get_job(job_id: str):
    return JOBS.get(job_id, {"status": "not_found"})

from fastapi.responses import StreamingResponse

def _mjpeg_generator(video_path: str, ref_emb: np.ndarray):
    cap = cv2.VideoCapture(video_path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bbox, face = detect_with_bbox_and_align(frame)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            color = (255, 0, 0)
            label = ""
            if face is not None:
                emb = get_embedding(face)
                sim = float(np.dot(emb, ref_emb))
                score = int(round(((sim + 1.0) / 2.0) * 100))
                color = (0, 200, 0) if sim >= THRESHOLDS.get("1in10k", 0.28) else (0, 0, 255)
                label = f"{score}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        jpg = buf.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
    cap.release()

@app.get(f"{API_PREFIX}/preview-video/{{job_id}}")
def preview_video(job_id: str):
    job = JOBS.get(job_id)
    if not job or "video_path" not in job or "ref_emb" not in job:
        raise HTTPException(404, "Job not found or not ready for preview")
    ref_emb = np.array(job["ref_emb"], dtype=np.float32)
    return StreamingResponse(_mjpeg_generator(job["video_path"], ref_emb), media_type='multipart/x-mixed-replace; boundary=frame')

# ===== Streams =====
@app.get(f"{API_PREFIX}/streams")
def list_streams():
    return {"items": STREAMS.list(), "capacity": MAX_STREAMS}

@app.post(f"{API_PREFIX}/streams")
def add_stream(rtsp_url: str = Form(...), label: str = Form(""), sampling_fps: float = Form(3.0), ref_image: Optional[UploadFile] = File(None)):
    if STREAMS.count() >= MAX_STREAMS:
        raise HTTPException(409, f"Max streams ({MAX_STREAMS}) reached. Delete one before adding.")
    ref_emb = None
    if ref_image is not None:
        im = _ensure_img(ref_image)
        face = detect_and_align(im)
        if face is None:
            raise HTTPException(422, "No face detected in ref_image")
        ref_emb = get_embedding(face)
    sid = str(uuid.uuid4())
    STREAMS.start(sid, rtsp_url, label=label, ref_emb=ref_emb, sampling_fps=sampling_fps)
    return {"stream_id": sid, "label": label, "sampling_fps": sampling_fps}

@app.delete(f"{API_PREFIX}/streams/{{stream_id}}")
def del_stream(stream_id: str):
    STREAMS.stop(stream_id)
    return {"ok": True}

# ===== WebSocket for live scores =====
@app.websocket(f"{API_PREFIX}/ws/streams/{{stream_id}}")
async def ws_stream(ws: WebSocket, stream_id: str):
    await ws_manager.connect(stream_id, ws)
    try:
        while True:
            # Keep-alive: client may send pings; we ignore payload
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(stream_id, ws)

@app.get(f"{API_PREFIX}/info")
def api_info():
    return {
        "api": {
            "version": "1.0",
            "profiles": list(THRESHOLDS.keys()),
            "max_streams": MAX_STREAMS,
        },
        "model": get_embedder_info(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)