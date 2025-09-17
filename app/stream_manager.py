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