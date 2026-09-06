# ffmpeg is not optional: faster-whisper shells out to it to decode
# webm/opus, and without it every request fails at the decode step with
# an error that looks like a model problem.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Render's free instance has 512 MB. "tiny" (~39M parameters, int8
# quantised) fits with room for the Python runtime; "base" is borderline
# and "small" will be killed mid-transcription by the OOM reaper, which
# presents as an unexplained 502 rather than a clear error.
#
# The cost is real: "tiny" is noticeably worse on drug names, which is
# the vocabulary that matters most here. Worth measuring rather than
# assuming — run the same audio through both locally and compare.
ARG WHISPER_MODEL=tiny
ENV WHISPER_MODEL=${WHISPER_MODEL}

# Bake the weights into the image rather than downloading on first
# request. Otherwise the first consultation after every deploy waits
# several minutes, and a cold-started instance may time out before the
# download finishes.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

COPY . .

# Render injects PORT and expects the process to bind to it.
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
