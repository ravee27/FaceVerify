# FaceVerify + Weapon Detection (FastAPI + React)

A simple web UI and REST API to:
- Face Verify: compare faces across images, video, and live RTSP (embeddings + cosine similarity)
- Weapon Detection: detect weapons in images, video uploads, or live RTSP (YOLOv11L)

## 1) Prerequisites
- Python 3.10+ (with venv)
- Node.js 18+
- ffmpeg installed on the server (for video/RTSP)
- NVIDIA GPU optional (CPU works; InsightFace may use CPU provider if CUDA not present)

## 2) Setup
```bash
# Create and activate venv
python3 -m venv face_verify_venv
source face_verify_venv/bin/activate

# Install backend dependencies
pip install --upgrade pip
pip install -r app/requirements.txt

# Build frontend
cd frontend
npm i
npm run build
```

## 3) Configure models
- Face Verify (state_dict backbone):
  - Place face model at `models/face/model.pt`
  - Or set env: `EMBEDDER_STATE_DICT=/absolute/path/to/model.pt`
  - Backbone arch (default `r100`): `EMBEDDER_ARCH=r100`
- Weapon Detection (YOLOv11L):
  - Place YOLO model at `models/gun/yolo_11L_70K.pt`
  - Or set env: `GUN_MODEL_PATH=/absolute/path/to/yolo_11L_70K.pt`

Optional UI serving (static):
- Set `STATIC_DIR=/Users/<you>/Documents/GitHub/FaceVerify/frontend/dist` in `app/.env` or environment

Example `.env` (in `app/.env`):
```
DEVICE=cuda:0
STATIC_DIR=/Users/<you>/Documents/GitHub/FaceVerify/frontend/dist
EMBEDDER_STATE_DICT=/Users/<you>/Documents/GitHub/FaceVerify/models/face/model.pt
EMBEDDER_ARCH=r100
GUN_MODEL_PATH=/Users/<you>/Documents/GitHub/FaceVerify/models/gun/yolo_11L_70K.pt
```

## 4) Run the server
```bash
# From repo root
source face_verify_venv/bin/activate
cd app
uvicorn app:app --host 0.0.0.0 --port 8080
# Open http://localhost:8080/
```

## 5) API Examples
Replace `BASE=http://localhost:8080` if needed.

### Health
```bash
curl -s $BASE/api/healthz
```

### Face Verify
- Image vs Image:
```bash
curl -X POST $BASE/api/verify-images \
  -F img1=@/path/a.jpg \
  -F img2=@/path/b.jpg \
  -F keep_for_audit=false \
  -F retention_days=0
```
- Video vs Reference:
```bash
jid=$(curl -s -X POST $BASE/api/verify-video \
  -F video=@/path/clip.mp4 \
  -F ref_image=@/path/ref.jpg \
  -F keep_for_audit=false \
  -F retention_days=0 | jq -r .job_id)
# Poll results
curl -s $BASE/api/jobs/$jid
# Live annotated sample (face) preview is available under /api/preview-video/{job_id}
```

### Weapon Detection
- Image:
```bash
curl -X POST $BASE/api/weapon/detect-image \
  -F img=@/path/image.jpg \
  -F conf=0.35
```
- Video (upload + preview):
```bash
jid=$(curl -s -X POST $BASE/api/weapon/detect-video \
  -F video=@/path/clip.mp4 \
  -F conf=0.35 | jq -r .job_id)
# Open MJPEG preview in a browser
xdg-open "$BASE/api/weapon/preview-video/$jid" 2>/dev/null || open "$BASE/api/weapon/preview-video/$jid"
```
- Live RTSP (preview only):
```bash
# Open this URL in a browser
$BASE/api/weapon/preview-rtsp?rtsp_url=rtsp://user:pass@host:554/stream&conf=0.35
```

## 6) UI
- The server can serve the UI at `/` when `STATIC_DIR` points to the built `frontend/dist`.
- The UI offers two modes in the header:
  - Face Verify: Images, Video vs Ref, Live RTSP tabs
  - Weapon Detection: Images, Video, Live RTSP tabs
- A collapsible “API Endpoints” card at the bottom shows example curl commands.

## 7) Notes
- Thresholds: Face Verify uses a configurable profile (default `1in10k`). Calibrate `THRESHOLD_1IN10K` in `.env` to match your data.
- Cleanup: Uploaded videos are kept for preview; add a periodic cleanup if needed.
- GPU/CPU: InsightFace may fall back to CPU. Ultralytics YOLO will use CUDA if available.

## 8) Systemd (optional)
See `deploy/faceverify.service` and adjust paths, user, and `.env` before enabling the service.
