"""Speech-to-text using faster-whisper."""

import os
from dataclasses import dataclass, field

from faster_whisper import WhisperModel

# "base" runs on a modest CPU in roughly real time and is adequate for
# clear two-person speech. "small" is noticeably better on veterinary
# vocabulary and roughly three times slower; "medium" needs more RAM than
# most free tiers allow. Set via env so the trade-off can be measured
# rather than guessed at.
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

_model: WhisperModel | None = None


@dataclass
class TranscriptionResult:
    text: str
    segments: list[dict] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None


def load_model() -> WhisperModel:
    global _model
    if _model is None:
        # int8 quantisation cuts memory roughly fourfold with a small
        # accuracy cost — the difference between fitting on a free tier
        # and not.
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_file(path: str) -> TranscriptionResult:
    model = load_model()

    segments, info = model.transcribe(
        path,
        # Voice activity detection drops silence, which matters here:
        # consultations have long pauses while the vet examines the
        # animal, and Whisper is prone to hallucinating text over silence.
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        # Consultations are in English for now. Forcing the language
        # avoids misdetection on short or noisy recordings; remove this
        # to support multilingual consultations.
        language="en",
    )

    collected: list[dict] = []
    parts: list[str] = []

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        parts.append(text)
        collected.append(
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text,
                # No speaker diarisation yet. Separating vet from owner
                # needs pyannote, which requires a Hugging Face token and
                # considerably more compute. Timings are kept so a
                # proposed field can be traced back to its moment in the
                # audio, which is most of the value.
                "speaker": None,
            }
        )

    return TranscriptionResult(
        text=" ".join(parts),
        segments=collected,
        language=info.language,
        duration=info.duration,
    )
