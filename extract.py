"""
Turns a consultation transcript into a structured proposed note.

The hard requirement is that the model does not invent clinical detail.
A fabricated dose in a prescription field is dangerous in a way a wrong
summary is not, so the prompt, the schema, and the post-processing all
push toward returning nothing rather than guessing.

The provider sits behind one function so it can be swapped, or compared.
Measuring extraction quality across providers on the same transcripts is
a more interesting thing to be able to report than picking one and
hoping.
"""

import json
import os

from pydantic import BaseModel, Field

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
MODEL = os.environ.get("EXTRACTION_MODEL", "gemini-2.5-flash")


class ProposedMedication(BaseModel):
    drug: str
    dose: str | None = None
    frequency: str | None = None
    duration_days: int | None = None
    instructions: str | None = None
    # The phrase this came from, so the vet can check it against what was
    # actually said rather than trusting the extraction.
    evidence: str | None = None


class ProposedNote(BaseModel):
    presenting_complaint: str | None = None
    examination: str | None = None
    assessment: str | None = None
    plan: str | None = None
    follow_up: str | None = None
    medications: list[ProposedMedication] = Field(default_factory=list)
    # Things the model was unsure about, surfaced rather than hidden.
    uncertain: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = """\
You extract structured clinical notes from veterinary consultation transcripts.

You are drafting a proposal for a qualified vet to review. You are not \
writing the record. Follow these rules exactly:

1. Use only what is stated in the transcript. Never infer a diagnosis, \
dose, or instruction that was not said aloud.
2. If a field was not discussed, return null for it. An empty field is \
correct and useful; an invented one is dangerous.
3. For medications, record only drugs explicitly named. If a dose or \
frequency was not stated, leave it null rather than supplying a typical \
value. Include the exact phrase the drug was mentioned in as evidence.
4. Transcripts are automatic and contain errors. If a drug name or \
number is garbled or ambiguous, put it in "uncertain" and leave the \
field null.
5. Write in the vet's clinical register, concise and factual. Add no \
hedging, recommendations, or anything not in the transcript.

Return only a JSON object of this shape, with no prose or code fences:

{
  "presenting_complaint": string | null,
  "examination": string | null,
  "assessment": string | null,
  "plan": string | null,
  "follow_up": string | null,
  "medications": [
    {
      "drug": string,
      "dose": string | null,
      "frequency": string | null,
      "duration_days": integer | null,
      "instructions": string | null,
      "evidence": string | null
    }
  ],
  "uncertain": [string]
}
"""


def extract_note(transcript: str) -> ProposedNote:
    user_prompt = f"<transcript>\n{transcript}\n</transcript>"

    if PROVIDER == "gemini":
        raw = _generate_gemini(SYSTEM_PROMPT, user_prompt)
    elif PROVIDER == "anthropic":
        raw = _generate_anthropic(SYSTEM_PROMPT, user_prompt)
    elif PROVIDER == "ollama":
        raw = _generate_ollama(SYSTEM_PROMPT, user_prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")

    return _parse(raw)


def _generate_gemini(system: str, user: str) -> str:
    import time

    from google import genai
    from google.genai import types

    # Reads GEMINI_API_KEY from the environment.
    client = genai.Client()

    config = types.GenerateContentConfig(
        system_instruction=system,
        # Near-zero temperature: this is extraction, and variation
        # between runs on the same audio is a defect, not a feature.
        temperature=0,
        max_output_tokens=4000,
        # Constrains the decoder to emit valid JSON, which removes a
        # whole class of parse failure before it happens.
        response_mime_type="application/json",
    )

    # The free tier returns 503 under load often enough that a single
    # attempt is not a fair test of the model. Retry with backoff so an
    # evaluation run measures extraction quality rather than Google's
    # capacity on the afternoon you happened to run it.
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=user, config=config
            )
            text = _text_from(response)
            if text.strip():
                return text
            last_error = RuntimeError("Model returned no text")
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
            if "503" not in str(exc) and "UNAVAILABLE" not in str(exc):
                raise
        time.sleep(2 ** attempt)

    raise RuntimeError(f"Gemini failed after retries: {last_error}")


def _text_from(response) -> str:
    """
    Pulls text out of a response that may also contain thinking parts.

    Newer Gemini models return reasoning alongside the answer, and
    `response.text` warns when it has to skip non-text parts. Reading the
    parts directly avoids the warning and, more importantly, avoids
    returning an empty string when the model spent its whole output
    budget thinking.
    """
    if getattr(response, "text", None):
        return response.text

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "thought", False):
                continue
            if getattr(part, "text", None):
                parts.append(part.text)
    return "".join(parts)


def _generate_anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def _generate_ollama(system: str, user: str) -> str:
    """Local model, for running the pipeline with no API at all."""
    import httpx

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    response = httpx.post(
        f"{host}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _parse(text: str) -> ProposedNote:
    """
    Parses the model's reply, degrading to an empty proposal rather than
    raising.

    A malformed response should leave the vet writing their note by hand,
    which is the status quo — not break the review page.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return ProposedNote(uncertain=["The transcript could not be parsed into a note."])

    try:
        note = ProposedNote.model_validate(data)
    except Exception:
        return ProposedNote(uncertain=["The proposed note did not match the expected shape."])

    # A medication with neither dose nor frequency is barely a
    # prescription. Keep it, but make sure the vet is told to check it
    # rather than accepting it blind.
    for med in note.medications:
        if med.dose is None and med.frequency is None:
            note.uncertain.append(f"{med.drug}: no dose or frequency was stated.")

    return note
