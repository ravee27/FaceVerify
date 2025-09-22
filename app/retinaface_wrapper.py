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


# def init_detector():
#     global _DET
#     if _DET is None:
#         app = insightface.app.FaceAnalysis(name="buffalo_l")
#         app.prepare(ctx_id=0, det_size=(640, 640))
#         _DET = app
#     return _DET

def init_detector():
    global _DET
    if _DET is None:
        import os
        try:
            import onnxruntime as ort
        except Exception:
            ort = None

        name = os.getenv("INSIGHTFACE_NAME", "buffalo_l")  # e.g., buffalo_l (accurate) or buffalo_sc (faster)
        det_w = int(os.getenv("INSIGHTFACE_DET_W", "640"))
        det_h = int(os.getenv("INSIGHTFACE_DET_H", "640"))

        providers = None
        ctx_id = 0
        if ort is not None:
            avail = ort.get_available_providers()
            print(f"InsightFace/ONNX providers available: {avail}")
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0
            else:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1

        app = insightface.app.FaceAnalysis(name=name, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
        _DET = app
    return _DET


def detect_and_align(bgr: np.ndarray, image_size: int = 112) -> np.ndarray | None:
    """
    Returns aligned face as CHW float32 in [0,1] or None if no face/landmarks found.
    Picks the highest-confidence face.
    """
    det = init_detector()
    faces = det.get(bgr)
    if not faces:
        return None
    # Pick best face
    face = max(faces, key=lambda f: getattr(f, 'det_score', 0.0))
    # InsightFace may expose 5-point landmarks as 'kps' or 'landmark'
    lmk = getattr(face, 'kps', None)
    if lmk is None:
        lmk = getattr(face, 'landmark', None)
    if lmk is None:
        return None
    lmk = np.array(lmk, dtype=np.float32).reshape(-1, 2)
    if lmk.shape[0] < 5:
        return None
    lmk5 = lmk[:5]
    try:
        M = _estimate_norm(lmk5, image_size=image_size)
    except Exception:
        return None
    aligned = cv2.warpAffine(bgr, M, (image_size, image_size))
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return chw


def detect_with_bbox_and_align(bgr: np.ndarray, image_size: int = 112):
    """
    Returns (bbox, aligned_face_chw) where bbox is (x1,y1,x2,y2) ints in the input image space.
    Returns (None, None) if no face is found.
    """
    det = init_detector()
    faces = det.get(bgr)
    if not faces:
        return None, None
    face = max(faces, key=lambda f: getattr(f, 'det_score', 0.0))
    bbox = getattr(face, 'bbox', None)
    if bbox is None:
        return None, None
    bbox = np.array(bbox, dtype=np.int32).tolist()
    lmk = getattr(face, 'kps', None)
    if lmk is None:
        lmk = getattr(face, 'landmark', None)
    if lmk is None:
        return bbox, None
    lmk = np.array(lmk, dtype=np.float32).reshape(-1, 2)
    if lmk.shape[0] < 5:
        return bbox, None
    lmk5 = lmk[:5]
    try:
        M = _estimate_norm(lmk5, image_size=image_size)
    except Exception:
        return bbox, None
    aligned = cv2.warpAffine(bgr, M, (image_size, image_size))
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return bbox, chw