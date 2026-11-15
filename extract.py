"""
Turns a consultation transcript into a structured proposed note.

The hard requirement is that the model does not invent clinical detail.
A fabricated dose in a prescription field is dangerous in a way a wrong
summary is not, so the prompt, the schema, and the post-processing all
push toward returning nothing rather than guessing.

The model call itself lives in llm.py; this module owns the prompt, the
schema, and the parsing.
"""

import json

from pydantic import BaseModel, Field

import config
import llm


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
    """
    Raises llm.LLMError (or its LLMTimeoutError subclass) if the model
    cannot be reached. A reply that arrives but cannot be parsed degrades
    to an empty proposal instead — see _parse.
    """
    if config.llm_provider() == "mock":
        # No API call: an empty proposal, the same thing a real model
        # should return for a transcript with nothing clinical in it.
        return _parse("{}")

    raw = llm.generate_json(
        model=config.extraction_model(),
        system=SYSTEM_PROMPT,
        contents=f"<transcript>\n{transcript}\n</transcript>",
        # Zero temperature: this is extraction, and variation between
        # runs on the same audio is a defect, not a feature.
        temperature=0,
        max_output_tokens=4000,
        attempts=4,
    )
    return _parse(raw)


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
