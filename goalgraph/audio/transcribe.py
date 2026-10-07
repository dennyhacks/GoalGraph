"""Audio branch: commentary transcription (faster-whisper, word timestamps)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def transcribe(wav: str, cache: str | None = None, model_size: str = "small.en",
               compute_type: str = "int8") -> list[dict]:
    """Return a list of sentences: {start, end, text, words:[{w, start, end, p}]}.

    Results are cached to JSON because Whisper is the slowest CPU step.
    """
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    try:
        from faster_whisper import WhisperModel
    except ImportError:  # pragma: no cover
        log.warning("faster-whisper not installed; audio branch disabled")
        return []
    from ..video_io import load_wav
    audio_arr = load_wav(wav, sr=16000)
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
    segments, _ = model.transcribe(
        audio_arr, language="en", word_timestamps=True, vad_filter=True, beam_size=5,
        initial_prompt="Football commentary. Goal, corner, foul, yellow card, red card, substitution, "
                       "kick off, half time, full time, number nine, the keeper saves.")
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3),
                          "p": round(w.probability, 3)})
    sentences = split_sentences(words)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(sentences, indent=1))
    return sentences


def split_sentences(words: list[dict], max_gap: float = 1.2) -> list[dict]:
    """Group words into sentences on terminal punctuation or long pauses."""
    out, cur = [], []

    def flush():
        if cur:
            out.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                        "text": " ".join(w["w"] for w in cur).strip(), "words": list(cur)})
            cur.clear()

    for w in words:
        if cur and w["start"] - cur[-1]["end"] > max_gap:
            flush()
        cur.append(w)
        if w["w"].endswith((".", "!", "?")):
            flush()
    flush()
    return out
