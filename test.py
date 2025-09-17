# FaceVerify Service — Full Stack (FastAPI + React, No Docker)

A production-ready template to run your **face similarity** service 24×7 on an SSH GPU server (PyTorch runtime, RetinaFace detection/alignment, your embedder), plus a friendly **React UI**. No Docker required (we use `systemd`). Optional audit retention (15/30/45/90 days). Live RTSP (max 5 streams) with WebSocket updates.

---

## 0) Features

* **Image vs Image**: two uploads → cosine similarity → **0–9999** scaled score + **1:10K FAR** decision (threshold configurable).
* **Video vs Reference**: upload a video and a reference image → top-K moments with highest similarity.
* **Live RTSP**: register up to **5 streams**; subscribe to live scores via WebSocket; graceful error when capacity exceeded.
* **Retention**: store nothing unless user ticks **Keep for Audit** and picks **15/30/45/90 days**.
* **Serving UI**: React app (Vite + Tailwind + Recharts) served by the same FastAPI process (no Nginx needed, though sample Nginx config is included if you prefer).
* **24×7**: `systemd` unit provided.

> **Note on embedder**: You’ll plug in your embedder in `app/embedder.py` (TorchScript or direct PyTorch). A tiny reference TorchScript interface and export notes are provided. Threshold `0.55` for 1:10K FAR is a placeholder—**recalibrate** on your eval pairs.

---

## 1) Repository Layout

```
faceverify/
  app/
    app.py
    settings.py
    embedder.py
    retinaface_wrapper.py
    stream_manager.py
    requirements.txt
    .env.example
  frontend/
    package.json
    vite.config.ts
    index.html
    postcss.config.js
    tailwind.config.js
    src/
      main.tsx
      App.tsx
      components/
        Card.tsx
        Tabs.tsx
        UploadBox.tsx
        ScoreDial.tsx
        StreamViewer.tsx
        Toast.tsx
      styles.css
  deploy/
    faceverify.service
    nginx.sample.conf
  README-DEPLOY.md
```

---

## 2) Backend Code (FastAPI)

### `app/settings.py`

```python
import os

# ====== Core Config ======
API_PREFIX = "/api"
RETENTION_ROOT = os.getenv("RETENTION_ROOT", "/srv/faceverify/data")
STATIC_DIR = os.getenv("STATIC_DIR", "")  # path to built frontend (optional)
MAX_STREAMS = int(os.getenv("MAX_STREAMS", "5"))

# Threshold for ~1:10k FAR (PLACEHOLDER — recalibrate on your data)
THRESHOLDS = {"1in10k": float(os.getenv("THRESHOLD_1IN10K", "0.55"))}

# Upload limits / validation
IMG_MAX_MB = int(os.getenv("IMG_MAX_MB", "8"))
ALLOWED_IMG_MIMES = {"image/jpeg", "image/png"}

# Audit retention choices
ALLOWED_RETENTION = {15, 30, 45, 90}

# Device selection
DEVICE = os.getenv("DEVICE", "cuda:0")

# CORS (front and API on same origin → keep strict; otherwise, add your domain)
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if os.getenv("CORS_ALLOW_ORIGINS") else []
```

### `app/embedder.py`

```python
"""
Embedder interface.

Plug your model at the TODOs below. Two options:
- Direct PyTorch nn.Module (state_dict) — simplest during dev
- TorchScript model via EMBEDDER_TORCHSCRIPT (recommended for deploy)

Model contract: input is CHW float32 [0..1], aligned face (e.g., 112x112),
output is 1D embedding vector. We L2-normalize before cosine similarity.
"""
import os
import numpy as np
import torch
from typing import Optional

_DEVICE = None
_MODEL = None

EMBEDDER_TORCHSCRIPT = os.getenv("EMBEDDER_TORCHSCRIPT", "")  # e.g., /srv/faceverify/models/embedder.ts


def load_embedder(device: str = "cuda:0"):
    global _DEVICE, _MODEL
    _DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")

    if EMBEDDER_TORCHSCRIPT and os.path.isfile(EMBEDDER_TORCHSCRIPT):
        _MODEL = torch.jit.load(EMBEDDER_TORCHSCRIPT, map_location=_DEVICE)
        _MODEL.eval()
    else:
        # TODO: Replace with your model load (state_dict)
        # Example:
        # from my_model_def import IResNet50
        # _MODEL = IResNet50(num_features=512)
        # _MODEL.load_state_dict(torch.load("/path/to/weights.pth", map_location=_DEVICE))
        # _MODEL.eval().to(_DEVICE)
        raise RuntimeError(
            "No embedder configured. Set EMBEDDER_TORCHSCRIPT to a TorchScript file or load your model in embedder.py"
        )

    # Warmup
    with torch.inference_mode():
        dummy = torch.randn(1, 3, 112, 112, device=_DEVICE)
        _ = _MODEL(dummy)
    return _MODEL


def get_embedding(face_chw_float01: np.ndarray) -> np.ndarray:
    """Run inference on aligned face (CHW float32 [0..1]). Returns L2-normalized embedding (np.float32)."""
    if _MODEL is None:
        raise RuntimeError("Embedder not loaded. Call load_embedder() at startup.")
    x = torch.from_numpy(face_chw_float01).unsqueeze(0).to(_DEVICE)
    with torch.inference_mode():
        emb = _MODEL(x).detach().float().cpu().numpy()[0]
    n = np.linalg.norm(emb) + 1e-9
    return (emb / n).astype(np.float32)
```

### `app/retinaface_wrapper.py`

```python
"""RetinaFace detection + 5-pt landmark alignment via insightface."""
import numpy as np
import cv2
import insightface
from skimage import transform as trans

_DET = None

# Reference template for 5-point alignment (ArcFace)
# (x, y) for left-eye, right-eye, nose, left-mouth, right-mouth in 112x112 space
_ARCFACE_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _estimate_norm(lmk: np.ndarray, image_size: int = 112) -> np.ndarray:
    assert lmk.shape == (5, 2)
    dst = _ARCFACE_5PTS.copy()
    if image_size != 112:
        dst *= (image_size / 112.0)
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M.astype(np.float32)


def init_detector():
    global _DET
    if _DET is None:
        app = insightface.app.FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        _DET = app
    return _DET


def detect_and_align(bgr: np.ndarray, image_size: int = 112) -> np.ndarray | None:
    """
    Returns aligned face as CHW float32 in [0,1] or None if no face found.
    Picks the highest-confidence face.
    """
    det = init_detector()
    faces = det.get(bgr)
    if not faces:
        return None
    # Pick best face
    face = max(faces, key=lambda f: f.det_score)
    lmk = face.landmark.astype(np.float32)  # (5,2)
    M = _estimate_norm(lmk, image_size=image_size)
    aligned = cv2.warpAffine(bgr, M, (image_size, image_size))
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return chw
```

### `app/stream_manager.py`

```python
import cv2
import threading
import time
import numpy as np
from typing import Dict, Callable
from retinaface_wrapper import detect_and_align
from embedder import get_embedding

class StreamManager:
    def __init__(self, on_score: Callable[[str, dict], None], max_streams: int):
        self.max_streams = max_streams
        self.on_score = on_score
        self.lock = threading.Lock()
        self.streams: Dict[str, dict] = {}  # id -> {thread, stop, url, label, sampling_fps, last_score}

    def list(self):
        with self.lock:
            return [
                {
                    "stream_id": sid,
                    "label": st.get("label", ""),
                    "sampling_fps": st.get("sampling_fps", 3.0),
                    "last_score": st.get("last_score"),
                }
                for sid, st in self.streams.items()
            ]

    def count(self):
        with self.lock:
            return len(self.streams)

    def start(self, stream_id: str, url: str, label: str = "", ref_emb=None, sampling_fps: float = 3.0):
        with self.lock:
            if len(self.streams) >= self.max_streams:
                raise RuntimeError(f"Max streams ({self.max_streams}) reached")
            stop_ev = threading.Event()
            t = threading.Thread(
                target=self._loop,
                args=(stream_id, url, label, ref_emb, sampling_fps, stop_ev),
                daemon=True,
            )
            self.streams[stream_id] = {
                "thread": t,
                "stop": stop_ev,
                "url": url,
                "label": label,
                "sampling_fps": sampling_fps,
                "last_score": None,
            }
            t.start()

    def stop(self, stream_id: str):
        with self.lock:
            st = self.streams.get(stream_id)
            if not st:
                return
            st["stop"].set()
            del self.streams[stream_id]

    def _loop(self, sid, url, label, ref_emb, sampling_fps, stop_ev):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.on_score(sid, {"event": "error", "message": "Failed to open RTSP"})
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(fps / max(sampling_fps, 0.5))))
        idx = 0
        while not stop_ev.is_set():
            ok = cap.grab()
            if not ok:
                self.on_score(sid, {"event": "error", "message": "Stream ended"})
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                face = detect_and_align(frame)
                if face is not None and ref_emb is not None:
                    emb = get_embedding(face)
                    sim = float(np.dot(emb, ref_emb))  # cosine because emb is L2-normalized
                    score = int(round(((sim + 1.0) / 2.0) * 9999))
                    self.streams[sid]["last_score"] = score
                    self.on_score(sid, {"event": "score", "t": time.time(), "similarity_raw": sim, "score": score})
            idx += 1
        cap.release()
```

### `app/app.py`

```python
import os
import io
import time
import uuid
import json
import base64
import cv2
import numpy as np
from typing import Optional, Dict, Set

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel

from settings import *
from embedder import load_embedder, get_embedding
from retinaface_wrapper import detect_and_align
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
    keep_for_audit: bool = False,
    retention_days: int = 0,
    profile: str = "1in10k",
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
    keep_for_audit: bool = False,
    retention_days: int = 0,
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
        JOBS[job_id] = {"status": "done", "topk": results[: cfg.topk]}
        try:
            os.remove(path)
        except Exception:
            pass
        if keep_for_audit:
            os.makedirs(RETENTION_ROOT, exist_ok=True)
            with open(os.path.join(RETENTION_ROOT, f"{job_id}.json"), "w") as f:
                json.dump({"created": time.time(), "results": JOBS[job_id]}, f)

    JOBS[job_id] = {"status": "running"}
    threading.Thread(target=_process, daemon=True).start()
    return {"job_id": job_id, "status": "queued"}

@app.get(f"{API_PREFIX}/jobs/{{job_id}}")
def get_job(job_id: str):
    return JOBS.get(job_id, {"status": "not_found"})

# ===== Streams =====
@app.get(f"{API_PREFIX}/streams")
def list_streams():
    return {"items": STREAMS.list(), "capacity": MAX_STREAMS}

@app.post(f"{API_PREFIX}/streams")
def add_stream(rtsp_url: str = Body(...), label: str = Body(""), sampling_fps: float = Body(3.0), ref_image: Optional[UploadFile] = File(None)):
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

# ===== Static UI (optional) =====
if STATIC_DIR and os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
```

### `app/requirements.txt`

```
fastapi==0.115.*
uvicorn[standard]==0.30.*
numpy==1.26.*
opencv-python-headless==4.10.*
pydantic==2.9.*
python-multipart==0.0.9
python-dotenv==1.0.*
insightface==0.7.*
onnxruntime-gpu==1.18.*
scikit-image==0.24.*
torch==2.3.*+cu121 ; extra-index-url https://download.pytorch.org/whl/cu121
```

### `app/.env.example`

```
# Backend settings
DEVICE=cuda:0
RETENTION_ROOT=/srv/faceverify/data
STATIC_DIR=/srv/faceverify/frontend/dist
MAX_STREAMS=5

# Threshold override (optional after calibration)
# THRESHOLD_1IN10K=0.57

# Embedder
# Path to TorchScript model (required unless you modify embedder.py to load your weights)
EMBEDDER_TORCHSCRIPT=/srv/faceverify/models/your_embedder.ts
```

---

## 3) Frontend Code (React + Vite + Tailwind + Recharts)

**Assumes API is reachable at the same origin** and under `/api`. If you deploy API elsewhere, set `VITE_API_BASE` env in the frontend.

### `frontend/package.json`

```json
{
  "name": "faceverify-ui",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "recharts": "^2.12.7"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.13",
    "typescript": "^5.5.4",
    "vite": "^5.4.2"
  }
}
```

### `frontend/vite.config.ts`

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173
  },
  build: {
    outDir: 'dist'
  }
})
```

### `frontend/postcss.config.js`

```js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

### `frontend/tailwind.config.js`

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

### `frontend/index.html`

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>FaceVerify</title>
  </head>
  <body class="bg-slate-50">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

### `frontend/src/styles.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### `frontend/src/main.tsx`

```tsx
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

### `frontend/src/components/Card.tsx`

```tsx
import { ReactNode } from 'react'

export default function Card({ title, children }: { title: string, children: ReactNode }) {
  return (
    <div className="bg-white shadow-sm rounded-2xl p-5 border border-slate-200">
      <h2 className="text-xl font-semibold mb-4 text-slate-800">{title}</h2>
      {children}
    </div>
  )
}
```

### `frontend/src/components/Tabs.tsx`

```tsx
import React from 'react'

type Tab = { key: string; label: string }

export default function Tabs({ tabs, active, onChange }: { tabs: Tab[], active: string, onChange: (k: string) => void }) {
  return (
    <div className="flex gap-2 mb-6">
      {tabs.map(t => (
        <button key={t.key}
          onClick={() => onChange(t.key)}
          className={`px-4 py-2 rounded-full border ${active === t.key ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'}`}>
          {t.label}
        </button>
      ))}
    </div>
  )
}
```

### `frontend/src/components/UploadBox.tsx`

```tsx
import { useRef } from 'react'

export default function UploadBox({ label, onFile }: { label: string, onFile: (f: File) => void }) {
  const ref = useRef<HTMLInputElement>(null)
  return (
    <div className="border-2 border-dashed border-slate-300 rounded-xl p-6 text-center cursor-pointer hover:bg-slate-50" onClick={() => ref.current?.click()}>
      <p className="text-slate-600">{label}</p>
      <input ref={ref} type="file" accept="image/*,video/*" className="hidden" onChange={e => {
        const f = e.target.files?.[0]; if (f) onFile(f)
      }} />
    </div>
  )
}
```

### `frontend/src/components/ScoreDial.tsx`

```tsx
export default function ScoreDial({ score, decision }: { score: number, decision: string }) {
  const color = decision === 'match' ? 'text-emerald-600' : 'text-rose-600'
  return (
    <div className="text-center">
      <div className={`text-6xl font-bold ${color}`}>{score}</div>
      <div className="uppercase tracking-wide text-slate-500">score (0–9999)</div>
      <div className={`mt-2 font-semibold ${color}`}>{decision}</div>
    </div>
  )
}
```

### `frontend/src/components/StreamViewer.tsx`

```tsx
import { useEffect, useRef, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export default function StreamViewer({ streamId }: { streamId: string }) {
  const [data, setData] = useState<{ t: number; score: number }[]>([])
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const ws = new WebSocket(`${location.origin.replace('http', 'ws')}/api/ws/streams/${streamId}`)
    wsRef.current = ws
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.event === 'score') {
        setData(prev => {
          const next = [...prev, { t: msg.t, score: msg.score }]
          return next.slice(-100)
        })
      }
    }
    ws.onclose = () => { /* noop */ }
    return () => ws.close()
  }, [streamId])

  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
          <XAxis dataKey="t" tickFormatter={(v) => new Date(v * 1000).toLocaleTimeString()} hide={true} />
          <YAxis domain={[0, 9999]} />
          <Tooltip labelFormatter={(v) => new Date(Number(v) * 1000).toLocaleTimeString()} />
          <Line type="monotone" dataKey="score" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
```

### `frontend/src/components/Toast.tsx`

```tsx
import { useEffect } from 'react'

export default function Toast({ text, onClose }: { text: string, onClose: () => void }) {
  useEffect(() => { const t = setTimeout(onClose, 3000); return () => clearTimeout(t) }, [onClose])
  return (
    <div className="fixed bottom-4 right-4 bg-slate-800 text-white px-4 py-2 rounded-lg shadow-lg">{text}</div>
  )
}
```

### `frontend/src/App.tsx`

```tsx
import { useEffect, useState } from 'react'
import Card from './components/Card'
import Tabs from './components/Tabs'
import UploadBox from './components/UploadBox'
import ScoreDial from './components/ScoreDial'
import StreamViewer from './components/StreamViewer'
import Toast from './components/Toast'

const API = import.meta.env.VITE_API_BASE || ''

export default function App() {
  const [tab, setTab] = useState<'img'|'video'|'live'>('img')
  const [toast, setToast] = useState<string>('')

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h1 className="text-3xl font-bold text-slate-800 mb-2">FaceVerify</h1>
      <p className="text-slate-600 mb-6">Compare faces across images, video, and RTSP streams.</p>

      <Tabs tabs={[{key:'img',label:'Images'},{key:'video',label:'Video vs Ref'},{key:'live',label:'Live RTSP'}]} active={tab} onChange={(k)=>setTab(k as any)} />

      {tab==='img' && <ImagesPanel setToast={setToast} />}
      {tab==='video' && <VideoPanel setToast={setToast} />}
      {tab==='live' && <LivePanel setToast={setToast} />}

      {toast && <Toast text={toast} onClose={()=>setToast('')} />}
    </div>
  )
}

function ImagesPanel({ setToast }: { setToast: (s:string)=>void }) {
  const [img1, setImg1] = useState<File | null>(null)
  const [img2, setImg2] = useState<File | null>(null)
  const [keep, setKeep] = useState(false)
  const [retention, setRetention] = useState(15)
  const [res, setRes] = useState<any>(null)
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    if (!img1 || !img2) return setToast('Please select both images')
    setBusy(true)
    const fd = new FormData()
    fd.append('img1', img1)
    fd.append('img2', img2)
    fd.append('keep_for_audit', String(keep))
    fd.append('retention_days', String(keep ? retention : 0))
    try {
      const r = await fetch(`${API}/api/verify-images`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(await r.text())
      const j = await r.json()
      setRes(j)
    } catch (e:any) {
      setToast(e.message || 'Request failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card title="Image 1">
        <UploadBox label={img1? img1.name : 'Click to upload'} onFile={setImg1} />
      </Card>
      <Card title="Image 2">
        <UploadBox label={img2? img2.name : 'Click to upload'} onFile={setImg2} />
      </Card>
      <Card title="Options">
        <div className="flex items-center gap-3">
          <input id="keep" type="checkbox" checked={keep} onChange={e=>setKeep(e.target.checked)} />
          <label htmlFor="keep">Keep for Audit</label>
          {keep && (
            <select className="ml-4 border rounded px-2 py-1" value={retention} onChange={e=>setRetention(Number(e.target.value))}>
              {[15,30,45,90].map(d=> <option key={d} value={d}>{d} days</option>)}
            </select>
          )}
        </div>
        <button className={`mt-4 px-4 py-2 rounded bg-blue-600 text-white ${busy?'opacity-50':''}`} onClick={submit} disabled={busy}>Compare</button>
      </Card>
      <Card title="Result">
        {res ? <ScoreDial score={res.similarity_scaled} decision={res.decision} /> : <div className="text-slate-500">No result yet</div>}
      </Card>
    </div>
  )
}

function VideoPanel({ setToast }: { setToast: (s:string)=>void }) {
  const [video, setVideo] = useState<File | null>(null)
  const [ref, setRef] = useState<File | null>(null)
  const [keep, setKeep] = useState(false)
  const [retention, setRetention] = useState(15)
  const [job, setJob] = useState<any>(null)
  const [topk, setTopk] = useState<any[]>([])
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    if (!video || !ref) return setToast('Please select a video and a reference image')
    setBusy(true)
    const fd = new FormData()
    fd.append('video', video)
    fd.append('ref_image', ref)
    fd.append('keep_for_audit', String(keep))
    fd.append('retention_days', String(keep ? retention : 0))
    try {
      const r = await fetch(`${API}/api/verify-video`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(await r.text())
      const j = await r.json()
      setJob(j)
      poll(j.job_id)
    } catch (e:any) {
      setToast(e.message || 'Request failed')
    } finally {
      setBusy(false)
    }
  }

  const poll = async (id: string) => {
    const t = setInterval(async () => {
      const r = await fetch(`${API}/api/jobs/${id}`)
      const j = await r.json()
      if (j.status === 'done') {
        clearInterval(t)
        setTopk(j.topk || [])
      }
    }, 1500)
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card title="Video">
        <UploadBox label={video? video.name : 'Click to upload a video'} onFile={setVideo} />
      </Card>
      <Card title="Reference Image">
        <UploadBox label={ref? ref.name : 'Click to upload a face image'} onFile={setRef} />
      </Card>
      <Card title="Options">
        <div className="flex items-center gap-3">
          <input id="keepv" type="checkbox" checked={keep} onChange={e=>setKeep(e.target.checked)} />
          <label htmlFor="keepv">Keep for Audit</label>
          {keep && (
            <select className="ml-4 border rounded px-2 py-1" value={retention} onChange={e=>setRetention(Number(e.target.value))}>
              {[15,30,45,90].map(d=> <option key={d} value={d}>{d} days</option>)}
            </select>
          )}
        </div>
        <button className={`mt-4 px-4 py-2 rounded bg-blue-600 text-white ${busy?'opacity-50':''}`} onClick={submit} disabled={busy}>Submit</button>
      </Card>
      <Card title="Top Matches">
        {topk.length ? (
          <ul className="list-disc ml-5 text-slate-700">
            {topk.map((x,i)=> <li key={i}>t={x.t_sec.toFixed(2)}s — score {x.similarity_scaled}</li>)}
          </ul>
        ) : <div className="text-slate-500">No results yet</div>}
      </Card>
    </div>
  )
}

function LivePanel({ setToast }: { setToast: (s:string)=>void }) {
  const [items, setItems] = useState<any[]>([])
  const [label, setLabel] = useState('')
  const [url, setUrl] = useState('')
  const [fps, setFps] = useState(3)
  const [selected, setSelected] = useState<string>('')

  const refresh = async () => {
    const r = await fetch(`${API}/api/streams`)
    const j = await r.json()
    setItems(j.items||[])
  }
  useEffect(()=>{ refresh() }, [])

  const add = async () => {
    if (!url) return setToast('Enter RTSP URL')
    const fd = new FormData()
    fd.append('rtsp_url', url)
    fd.append('label', label)
    fd.append('sampling_fps', String(fps))
    const r = await fetch(`${API}/api/streams`, { method: 'POST', body: fd })
    if (r.status === 409) {
      setToast('Max streams (5) reached. Delete one before adding.')
      return
    }
    if (!r.ok) { setToast('Failed to add stream'); return }
    setUrl(''); setLabel('');
    await refresh()
  }

  const delStream = async (id: string) => {
    await fetch(`${API}/api/streams/${id}`, { method: 'DELETE' })
    await refresh()
    if (selected === id) setSelected('')
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <Card title="Add Stream">
        <div className="space-y-3">
          <input className="w-full border rounded px-3 py-2" placeholder="Label (optional)" value={label} onChange={e=>setLabel(e.target.value)} />
          <input className="w-full border rounded px-3 py-2" placeholder="rtsp://user:pass@host:port/…" value={url} onChange={e=>setUrl(e.target.value)} />
          <div className="flex items-center gap-2">
            <label>Sampling FPS</label>
            <input className="w-24 border rounded px-2 py-1" type="number" min={1} max={10} value={fps} onChange={e=>setFps(Number(e.target.value))} />
          </div>
          <button className="px-4 py-2 rounded bg-blue-600 text-white" onClick={add}>Add</button>
        </div>
      </Card>
      <Card title="Streams">
        <ul className="divide-y">
          {items.map((it:any)=> (
            <li key={it.stream_id} className="py-2 flex items-center justify-between">
              <div>
                <div className="font-medium">{it.label || it.stream_id.slice(0,8)}</div>
                <div className="text-sm text-slate-500">last score: {it.last_score ?? '-'}</div>
              </div>
              <div className="flex gap-2">
                <button className="px-3 py-1 border rounded" onClick={()=>setSelected(it.stream_id)}>View</button>
                <button className="px-3 py-1 border rounded text-rose-600" onClick={()=>delStream(it.stream_id)}>Delete</button>
              </div>
            </li>
          ))}
        </ul>
      </Card>
      <Card title="Live Scores">
        {selected ? <StreamViewer streamId={selected} /> : <div className="text-slate-500">Select a stream</div>}
      </Card>
    </div>
  )
}
```

---

## 4) `deploy/` — Services & (optional) Nginx

### `deploy/faceverify.service`

```ini
[Unit]
Description=FaceVerify API + Static UI
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/srv/faceverify/app
EnvironmentFile=/srv/faceverify/app/.env
ExecStart=/srv/faceverify/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1
Restart=always
RestartSec=5
StandardOutput=append:/srv/faceverify/logs/app.log
StandardError=append:/srv/faceverify/logs/app.err

[Install]
WantedBy=multi-user.target
```

### `deploy/nginx.sample.conf` (only if you want Nginx in front)

```nginx
server {
  listen 80;
  server_name _;

  # If you terminate TLS elsewhere, keep 80 for LAN. For Internet, add certs here.

  location /api/ {
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_pass http://127.0.0.1:8080;
  }

  location / {
    root /srv/faceverify/frontend/dist;
    try_files $uri /index.html;
  }
}
```

---

## 5) `README-DEPLOY.md` — Step-by-Step Guide

### A) Server prep (Ubuntu 22.04+, NVIDIA driver installed)

```bash
sudo apt update && sudo apt install -y python3.10-venv ffmpeg nginx git
sudo mkdir -p /srv/faceverify/{app,frontend,logs,data,models}
cd /srv/faceverify
```

### B) Backend setup

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Copy backend files into /srv/faceverify/app
# (Use scp/rsync/git clone; ensure the files from this repo are placed accordingly)

pip install --upgrade pip
pip install -r app/requirements.txt

# Put your TorchScript model at /srv/faceverify/models/your_embedder.ts
# (Or modify app/embedder.py to load your state_dict)

cp app/.env.example app/.env
# Edit app/.env to point to your STATIC_DIR and EMBEDDER_TORCHSCRIPT
```

### C) Frontend build

```bash
cd /srv/faceverify/frontend
npm i
npm run build

# Copy build to the path in STATIC_DIR (e.g., /srv/faceverify/frontend/dist — already the default)
# No Nginx needed: FastAPI serves the static UI because STATIC_DIR is set in .env
```

### D) Run as a service (24×7)

```bash
sudo cp /srv/faceverify/deploy/faceverify.service /etc/systemd/system/faceverify.service
sudo systemctl daemon-reload
sudo systemctl enable --now faceverify

# Check
curl http://localhost:8080/api/healthz
# → {"ok": true}
```

### E) (Optional) Put Nginx in front

If you prefer Nginx to serve the frontend and reverse-proxy `/api` to the backend:

```bash
sudo cp /srv/faceverify/deploy/nginx.sample.conf /etc/nginx/sites-available/faceverify
sudo ln -s /etc/nginx/sites-available/faceverify /etc/nginx/sites-enabled/faceverify
sudo nginx -t && sudo systemctl reload nginx
```

Navigate to `http://<server-ip>/`.

### F) RTSP notes

* Provide RTSP URL in the UI. If credentials are present, prefer limited-scope users.
* If you need to compare against a particular person, add a **reference image** when creating the stream (Live tab → optional `ref_image` input can be added similarly to Video; or extend UI with a file field posting to `/api/streams`).
* For stability/latency, consider GStreamer pipelines later.

### G) Threshold calibration (IMPORTANT)

1. Assemble genuine & imposter pairs from your org.
2. Compute cosine scores using this service offline or a notebook.
3. Pick threshold that yields desired FAR (e.g., 1:10k) and set `THRESHOLD_1IN10K` in `.env`.
4. Restart service: `sudo systemctl restart faceverify`.

### H) Troubleshooting

* **Embedder not configured**: ensure `EMBEDDER_TORCHSCRIPT` points to a valid TorchScript file.
* **CUDA not used**: verify `nvidia-smi`, Torch CUDA build (`python -c "import torch; print(torch.cuda.is_available())"`).
* **RTSP fails**: check firewall, URL correctness, try `ffplay <rtsp>` from the server.
* **Memory**: reduce sampling FPS or image size; ensure no other heavy GPU jobs are running.

---

## 6) TorchScript Export Snippet (example)

Use this in your training repo to export your embedder (adjust to your network):

```python
import torch
from my_model_def import IResNet50  # your class

model = IResNet50(num_features=512)
ckpt = torch.load('weights.pth', map_location='cpu')
model.load_state_dict(ckpt)
model.eval()

example = torch.randn(1,3,112,112)
traced = torch.jit.trace(model, example)
traced.save('/srv/faceverify/models/your_embedder.ts')
```

---

## 7) What to tweak next

* **Auth**: add OAuth2 proxy later (Google/Microsoft SSO). All API routes are under `/api` → easy to protect.
* **Batching**: for bursty traffic on `/verify-images`, consider a small in-process queue and batch inference.
* **GStreamer**: swap OpenCV backend for RTSP (`cv2.CAP_GSTREAMER`) for lower latency.
* **Observability**: add Prometheus `/metrics` and Grafana dashboard.
* **Error visuals**: add preview thumbnails and richer toasts.

---

**You’re set.** Copy these files to your server following the guide above and you’ll have a clean, friendly web tool your org can use for image, video, and live-stream face similarity — fast and secure by default (no persistence unless requested).
