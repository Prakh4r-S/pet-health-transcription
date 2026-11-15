# No OS-level ffmpeg: faster-whisper decodes audio through PyAV, whose
# wheel bundles its own libav* libraries (including the Opus and Vorbis
# decoders that webm recordings need). It never shells out to a binary.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# "base" with int8 quantisation: noticeably better than "tiny" on drug
# names, which is the vocabulary that matters most here, and still close
# to real time on 2 vCPU. Must match the default in config.py.
ARG WHISPER_MODEL=base
ENV WHISPER_MODEL=${WHISPER_MODEL}

# Bake the weights into the image, as the user that will read them, so
# the cache under HF_HOME is owned by appuser. Otherwise every cold start
# would download the model before it could serve anything.
USER appuser
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

# From here on the model cache is all there is. Without this, each cold
# start still calls the Hugging Face Hub to check for a newer revision —
# a slow network round-trip, and an outage there would become an outage
# here. It also means a WHISPER_MODEL that was not baked in fails loudly
# instead of quietly downloading at runtime.
ENV HF_HUB_OFFLINE=1

COPY --chown=appuser:appuser . .

# Cloud Run injects PORT; 8080 is its default and the local fallback.
EXPOSE 8080

# exec so uvicorn is PID 1 and receives SIGTERM directly on shutdown.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
