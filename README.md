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

**One model client, measurable output.** Extraction and chat share a
single Gemini client with a per-request timeout. `eval/run_eval.py`
scores extraction against hand-written expected notes, so a change of
model or prompt can be compared on the same transcripts rather than
judged by eye. `LLM_PROVIDER=mock` runs everything with no API key.

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
| `SERVICE_API_KEY` | Shared secret; requests must carry it as a bearer token. Required — the service refuses to start without it |
| `GEMINI_API_KEY` | Model credentials. Required unless `LLM_PROVIDER=mock` |
| `EXTRACTION_MODEL` | Model for note extraction (default `gemini-3.6-flash`) |
| `CHAT_MODEL` | Model for `/chat` (defaults to `EXTRACTION_MODEL`) |
| `LLM_PROVIDER` | `gemini` (default) or `mock`, which returns canned output without calling any API |
| `LLM_TIMEOUT_SECONDS` | Deadline for one model request (default `60`) |
| `WHISPER_MODEL` | `tiny`, `base`, `small`, `medium` (default `base`). The Docker image contains only the model it was built with and runs offline, so change it together with the `WHISPER_MODEL` build arg |
| `ALLOWED_AUDIO_HOSTS` | Comma-separated hosts recordings may be fetched from; `*.` prefix matches subdomains. Default `*.public.blob.vercel-storage.com,*.blob.vercel-storage.com` |
| `ALLOW_LOCAL_AUDIO` | `true` permits plain HTTP and private/loopback hosts. Local testing only — never set in production |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | Whisper runs allowed at once per instance (default `2`); further requests queue |
| `LOG_LEVEL` | Log verbosity (default `INFO`) |

`.env.example` lists the same variables for local development.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /transcribe` | Transcribe a recording and propose a note |
| `POST /chat` | Answer an owner's question from their records |
| `GET /health` | Liveness — the process is up |
| `GET /ready` | Readiness — `200` once the Whisper model is loaded, `503` while it warms up |

Upstream failures are reported as such: `502` when the language model
returns an error, `504` when it times out, `422` when the recording
cannot be decoded as audio.

## Logging

Every request gets an ID, returned as `X-Request-ID` (a caller-supplied
one is kept if it is short and plain). Logs are one JSON object per line
with the route, status, `duration_ms`, and — where they apply —
`audio_duration_s`, `whisper_model`, and `llm_model`.

Transcripts, questions, record context, model output, audio URLs, and
API keys are never logged.

## Deploying to Cloud Run

```bash
PROJECT_ID=my-project ./deploy.sh
```

This builds the image with Cloud Build and deploys it with 2 vCPU, 2 GiB,
concurrency 10, a 300-second request timeout, startup CPU boost, and a
startup probe on `/ready`. `SERVICE_API_KEY` and `GEMINI_API_KEY` are
read from Secret Manager; the script's header describes the one-time
setup. `service.yaml` is the same configuration in declarative form.

To run the image locally:

```bash
docker build -t pet-health-transcription .
```

```bash
docker run --rm -p 8080:8080 --env-file .env pet-health-transcription
```

This service holds no user context and makes no authorisation decisions
of its own — the calling application establishes that the requester is
the vet on the appointment. The shared secret only ensures requests come
from that application.
