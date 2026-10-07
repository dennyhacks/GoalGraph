"""Video / audio I/O helpers (OpenCV + ffmpeg)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    info = VideoInfo(str(path), cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def iter_frames(path: str, fps: float | None = None, start: float = 0.0,
                end: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, BGR frame) sampled at ``fps`` (None = native)."""
    cap = cv2.VideoCapture(str(path))
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = 1 if not fps else max(1, int(round(native / fps)))
    idx = int(start * native)
    if idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = idx / native
        if end is not None and t > end:
            break
        if (idx % step) == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield t, frame
        idx += 1
    cap.release()


def read_frame(path: str, t: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * native)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def extract_audio(video: str, wav: str, sr: int = 16000) -> str:
    Path(wav).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-ac", "1", "-ar", str(sr), "-vn", str(wav)],
                   check=True)
    return wav


def load_wav(wav: str, sr: int = 16000) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def has_audio(video: str) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def write_clip(video: str, start: float, end: float, out: str, boxes: dict | None = None,
               label: str | None = None) -> str:
    """Cut [start, end] to ``out`` (H.264 so browsers can play it).

    ``boxes`` optionally maps timestamp -> (x, y, w, h, caption) to burn in.
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, start)
    if not boxes and not label:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video), "-t",
                        f"{end - start:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-movflags", "+faststart", str(out)], check=True)
        return out
    info = probe(video)
    tmp = str(Path(out).with_suffix(".tmp.mp4"))
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
    keys = sorted(boxes) if boxes else []
    for t, frame in iter_frames(video, None, start, end):
        if keys:
            k = min(keys, key=lambda x: abs(x - t))
            if abs(k - t) < 0.5:
                x, y, w, h, cap = boxes[k]
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 3)
                if cap:
                    cv2.putText(frame, cap, (int(x), max(20, int(y) - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                (0, 255, 255), 2, cv2.LINE_AA)
        if label:
            cv2.putText(frame, label, (20, info.height - 24), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-ss", f"{start:.3f}", "-i", str(video), "-t",
                    f"{end - start:.3f}", "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out)],
                   check=True)
    Path(tmp).unlink(missing_ok=True)
    return out


def fmt_time(t: float | None) -> str:
    if t is None:
        return "--:--"
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:04.1f}"
