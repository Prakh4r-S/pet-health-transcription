"""
Answers an owner's questions about their own pets' records.

This is deliberately not a veterinary advice service. It answers from a
context block the calling application assembled from that owner's data,
and it refuses anything that would amount to diagnosis or treatment
advice. The distinction matters: "when is Bruno's rabies due" is a
lookup, "what's wrong with Bruno" is a consultation, and conflating them
would be both useless and unsafe.
"""

import os

from pydantic import BaseModel

MODEL = os.environ.get("CHAT_MODEL", os.environ.get("EXTRACTION_MODEL", "gemini-3.6-flash"))


class ChatTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    context: str
    history: list[ChatTurn] = []


class ChatResponse(BaseModel):
    answer: str
    # True when the question was clinical rather than a records lookup.
    # The application uses this to surface a "book a consultation" action
    # rather than leaving the person with a refusal and nowhere to go.
    suggests_consultation: bool


SYSTEM_PROMPT = """\
You help a pet owner understand their own pets' health records in a \
veterinary application.

WHAT YOU KNOW
Everything you know is in the <records> block. It is that owner's data, \
already filtered to what they may see. You have no other source.

RULES

1. Answer only from <records>. If the answer is not there, say plainly \
that it is not in the records. Never fill a gap with general knowledge \
about pets, breeds, or medicine — a plausible invention is worse than an \
admission, because the owner cannot tell them apart.

2. Say which record you used. "Bruno's last consultation on 4 March" \
rather than an unattributed assertion.

3. You do not practise veterinary medicine. Do not diagnose, do not \
suggest treatments, do not interpret symptoms, do not say whether \
something is serious, and do not advise on medication doses even when \
the dose is in the records — reading back what was prescribed is fine, \
advising on it is not.

4. When asked something clinical, say that it needs a vet, and point at \
booking a consultation. Be warm about it rather than curt; someone \
asking whether their dog is unwell is worried, not misusing the tool.

5. If the question suggests an emergency — collapse, difficulty \
breathing, seizure, suspected poisoning, severe bleeding, bloating with \
retching — say clearly and immediately that this needs urgent in-person \
veterinary attention now, not an online consultation. Do this first, \
before anything else.

6. Be brief. Two or three sentences usually. This is a lookup, not an \
essay.

OUTPUT
Return JSON only, no prose or code fences:

{
  "answer": string,
  "suggests_consultation": boolean
}

Set suggests_consultation to true when the question is clinical, when \
the records show something that warrants a vet's attention, or when you \
had to decline.
"""


def answer_question(request: ChatRequest) -> ChatResponse:
    from google import genai
    from google.genai import types

    client = genai.Client()

    # History is included so follow-ups work ("what about her?"), but the
    # records block is re-sent every turn rather than relying on the
    # model remembering it. Context that matters should not depend on
    # attention over a long conversation.
    conversation = []
    for turn in request.history[-6:]:
        conversation.append(
            types.Content(
                role="user" if turn.role == "user" else "model",
                parts=[types.Part(text=turn.content)],
            )
        )
    conversation.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=f"<records>\n{request.context}\n</records>\n\n"
                    f"Question: {request.question}"
                )
            ],
        )
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=conversation,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=1500,
            response_mime_type="application/json",
        ),
    )

    return _parse(_text_from(response))


def _text_from(response) -> str:
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


def _parse(text: str) -> ChatResponse:
    import json

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    try:
        data = json.loads(cleaned)
        return ChatResponse(
            answer=str(data["answer"]),
            suggests_consultation=bool(data.get("suggests_consultation", False)),
        )
    except Exception:
        # Failing closed: an unparseable response becomes a referral to a
        # vet rather than a guess or a blank screen.
        return ChatResponse(
            answer=(
                "I couldn't answer that reliably. If it's about your pet's health, "
                "booking a consultation is the right next step."
            ),
            suggests_consultation=True,
        )
