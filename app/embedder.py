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

try:
    from backbones import get_model  # r100/r50 etc.
except Exception:
    get_model = None  # will error later if needed


_DEVICE = None
_MODEL = None

# Autoload .env for robustness if app hasn't loaded it yet
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except Exception:
    pass

# Preferred: TorchScript path
EMBEDDER_TORCHSCRIPT = os.getenv("EMBEDDER_TORCHSCRIPT", "")
# State-dict path + architecture (e.g., r100)
EMBEDDER_STATE_DICT = os.getenv("EMBEDDER_STATE_DICT", "")
EMBEDDER_ARCH = os.getenv("EMBEDDER_ARCH", "r100")

# Fallback to repo default state_dict if env not provided
if not EMBEDDER_STATE_DICT:
    _repo_root = os.path.dirname(os.path.dirname(__file__))
    _default_sd = os.path.join(_repo_root, "models", "face", "model.pt")
    if os.path.isfile(_default_sd):
        EMBEDDER_STATE_DICT = _default_sd

print(f"EMBEDDER_TORCHSCRIPT: {EMBEDDER_TORCHSCRIPT}")
print(f"EMBEDDER_STATE_DICT: {EMBEDDER_STATE_DICT}")
print(f"EMBEDDER_ARCH: {EMBEDDER_ARCH}")

def _load_from_torchscript(device: torch.device) -> bool:
    global _MODEL
    candidates = [
        EMBEDDER_TORCHSCRIPT,
        "/srv/faceverify/models/your_embedder.ts",
        "/srv/faceverify/models/model.ts",
        "/srv/faceverify/models/model.pt",
    ]
    for p in candidates:
        if not p:
            continue
        if not os.path.isfile(p):
            print(f"Embedder: file not found: {p}")
            continue
        print(f"Embedder: trying TorchScript load: {p}")
        try:
            m = torch.jit.load(p, map_location=device)
            m.eval()
            _MODEL = m
            print(f"Embedder: loaded TorchScript from {p}")
            return True
        except Exception as e:
            print(f"Embedder: TorchScript load failed for {p}: {e}")
    return False


def _load_from_state_dict(device: torch.device) -> bool:
    global _MODEL
    if not EMBEDDER_STATE_DICT:
        return False
    if not os.path.isfile(EMBEDDER_STATE_DICT):
        print(f"Embedder: state_dict file not found: {EMBEDDER_STATE_DICT}")
        return False
    if get_model is None:
        raise RuntimeError("backbones.get_model not available. Install your backbone module or export TorchScript.")
    print(f"Embedder: loading backbone {EMBEDDER_ARCH} from state_dict {EMBEDDER_STATE_DICT}")
    net = get_model(EMBEDDER_ARCH, fp16=False)
    sd = torch.load(EMBEDDER_STATE_DICT, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print(f"Embedder: warning — missing {len(missing)} keys, first 10: {missing[:10]}")
    if unexpected:
        print(f"Embedder: warning — unexpected {len(unexpected)} keys, first 10: {unexpected[:10]}")
    try:
        net = net.to(device)
    except Exception:
        pass
    _MODEL = net.eval()
    print("Embedder: loaded nn.Module via state_dict")
    return True


def load_embedder(device: str = "cuda:0"):
    global _DEVICE, _MODEL
    _DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Embedder: torch.cuda.is_available(): {torch.cuda.is_available()}, requested device: {device}, using device: {_DEVICE}")

    # Try state_dict first if provided, else TorchScript
    loaded = False
    if EMBEDDER_STATE_DICT:
        loaded = _load_from_state_dict(_DEVICE)
    if not loaded:
        loaded = _load_from_torchscript(_DEVICE)
    if not loaded:
        raise RuntimeError("No embedder configured. Set EMBEDDER_STATE_DICT + EMBEDDER_ARCH or EMBEDDER_TORCHSCRIPT.")

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


def get_embedder_info() -> dict:
    info = {
        "arch": EMBEDDER_ARCH,
        "torchscript": bool(EMBEDDER_TORCHSCRIPT),
        "state_dict": bool(EMBEDDER_STATE_DICT),
        "device": str(_DEVICE) if _DEVICE is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
        "torch": torch.__version__,
    }
    # Try to include model class name
    if _MODEL is not None:
        info["model_class"] = _MODEL.__class__.__name__
    return info