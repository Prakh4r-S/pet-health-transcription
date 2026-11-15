"""Speech-to-text using faster-whisper."""

import threading

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from pydantic import BaseModel

import config

_model: WhisperModel | None = None
_model_lock = threading.Lock()


class AudioDecodeError(Exception):
    """The recording could not be decoded as audio."""


class Segment(BaseModel):
    start: float
    end: float
    text: str
    # No speaker diarisation yet. Separating vet from owner needs
    # pyannote, which requires a Hugging Face token and considerably more
    # compute. Timings are kept so a proposed field can be traced back to
    # its moment in the audio, which is most of the value.
    speaker: str | None = None


class TranscriptionResult(BaseModel):
    text: str
    segments: list[Segment] = []
    language: str | None = None
    duration: float | None = None


def load_model() -> WhisperModel:
    global _model
    if _model is None:
        # Locked because startup warms the model on a background thread
        # while requests may already be arriving; without it two threads
        # would each load a copy of the weights.
        with _model_lock:
            if _model is None:
                # int8 quantisation cuts memory roughly fourfold with a
                # small accuracy cost — the difference between fitting on
                # a small instance and not.
                _model = WhisperModel(
                    config.whisper_model(), device="cpu", compute_type="int8"
                )
    return _model


def is_model_loaded() -> bool:
    return _model is not None


def transcribe_file(path: str) -> TranscriptionResult:
    """Raises AudioDecodeError if the file is not decodable audio."""
    import av

    model = load_model()

    # Decoded here rather than inside model.transcribe() so that a bad
    # upload can be told apart from a fault in the model: the first is
    # the caller's problem, the second is ours.
    try:
        audio = decode_audio(path, sampling_rate=model.feature_extractor.sampling_rate)
    except (av.error.FFmpegError, IndexError) as exc:
        # IndexError: a valid container with no audio stream in it.
        raise AudioDecodeError("Recording could not be decoded as audio") from exc

    segments, info = model.transcribe(
        audio,
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

    collected: list[Segment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        collected.append(
            Segment(start=round(segment.start, 2), end=round(segment.end, 2), text=text)
        )

    return TranscriptionResult(
        text=" ".join(s.text for s in collected),
        segments=collected,
        language=info.language,
        duration=info.duration,
    )
