import os


# ====== Core Config ======
API_PREFIX = "/api"
RETENTION_ROOT = os.getenv("RETENTION_ROOT", "/srv/faceverify/data")
STATIC_DIR = os.getenv("STATIC_DIR", "") # path to built frontend (optional)
MAX_STREAMS = int(os.getenv("MAX_STREAMS", "5"))


# Threshold for ~1:10k FAR (PLACEHOLDER — recalibrate on your data)
THRESHOLDS = {"1in10k": float(os.getenv("THRESHOLD_1IN10K", "0.28"))}


# Upload limits / validation
IMG_MAX_MB = int(os.getenv("IMG_MAX_MB", "8"))
ALLOWED_IMG_MIMES = {"image/jpeg", "image/png", "image/JPEG", "image/PNG"}


# Audit retention choices
ALLOWED_RETENTION = {15, 30, 45, 90}


# Device selection
DEVICE = os.getenv("DEVICE", "cuda:0")


# CORS (front and API on same origin → keep strict; otherwise, add your domain)
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if os.getenv("CORS_ALLOW_ORIGINS") else []