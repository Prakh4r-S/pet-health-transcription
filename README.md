# Pet Health transcription service

Transcribes veterinary consultation recordings and proposes a structured
clinical note. Built as the ML component of a pet healthcare application.

## What it does

`POST /transcribe` takes a signed URL to an audio recording and returns:

- **transcript** — full text, via `faster-whisper` running on CPU
- **segments** — timings, so a proposed field can be traced back to the
  moment in the audio it came from
- **extraction** — a structured proposal: presenting complaint,
  examination, assessment, plan, and any medications named

Everything returned is a *proposal*. The vet reviews each field in the
web application and accepts, edits, or discards it. Nothing this service
produces reaches a clinical record without a human approving it.

## Design notes

**Null over guessing.** The extraction prompt requires that anything not
explicitly stated in the transcript comes back as null. An empty field is
correct and useful; an invented dose is dangerous.

**Evidence for medications.** Each proposed drug carries the phrase it
was heard in, so the vet can check a dose against what was actually said
rather than trusting the extraction.

**Voice activity detection.** Consultations contain long pauses while the
vet examines the animal, and Whisper is prone to hallucinating text over
silence. VAD filtering removes those gaps before transcription.

**Swappable extraction model.** `LLM_PROVIDER` selects between Gemini,
Anthropic, and a local Ollama model. All three receive the same prompt
and go through the same parser, so extraction quality can be compared
across providers on the same transcripts.

## Limitations

- **No speaker diarisation.** Vet and owner are not distinguished.
  Adding this needs `pyannote`, a Hugging Face token, and considerably
  more compute.
- **English only**, currently forced to avoid language misdetection on
  short or noisy recordings.
- **Synchronous.** Long recordings hold the request open; a job queue
  would be the right answer at any real volume.

## Configuration

| Variable | Purpose |
| --- | --- |
| `SERVICE_API_KEY` | Shared secret; requests must carry it as a bearer token |
| `GEMINI_API_KEY` | Extraction model credentials |
| `EXTRACTION_MODEL` | Model name, e.g. `gemini-3.6-flash` |
| `WHISPER_MODEL` | `tiny`, `base`, `small`, `medium` (deployment uses `tiny` to fit 512 MB) |
| `LLM_PROVIDER` | `gemini` (default), `anthropic`, `ollama` |

This service holds no user context and makes no authorisation decisions
of its own — the calling application establishes that the requester is
the vet on the appointment. The shared secret only ensures requests come
from that application.
