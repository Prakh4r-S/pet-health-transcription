"""/chat: input validation, history handling, and failure behaviour."""

import httpx
import pytest
from google.genai import errors

from conftest import AUTH

RECORDS = "Bruno (dog): rabies booster due 4 March 2027."
FALLBACK = (
    "I couldn't answer that reliably. If it's about your pet's health, "
    "booking a consultation is the right next step."
)


def post_chat(client, **body):
    body.setdefault("question", "When is Bruno's booster due?")
    body.setdefault("context", RECORDS)
    return client.post("/chat", json=body, headers=AUTH)


# --- Validation ------------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_question_is_422(client, question):
    response = post_chat(client, question=question)
    assert response.status_code == 422
    assert response.json() == {"detail": "No question given"}


def test_question_length_limit(client):
    assert post_chat(client, question="x" * 4000).status_code == 200
    assert post_chat(client, question="x" * 4001).status_code == 422


def test_context_size_limit(client):
    assert post_chat(client, context="x" * 40_000).status_code == 200

    response = post_chat(client, context="x" * 40_001)
    assert response.status_code == 413
    assert response.json() == {"detail": "Context too large"}


@pytest.mark.parametrize("role", ["system", "model", "User", ""])
def test_unknown_history_role_is_422(client, role):
    response = post_chat(client, history=[{"role": role, "content": "hello"}])
    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["question", "context"])
def test_required_fields(client, missing):
    body = {"question": "q", "context": "c"}
    del body[missing]
    assert client.post("/chat", json=body, headers=AUTH).status_code == 422


def test_history_is_optional(client):
    response = post_chat(client)
    assert response.status_code == 200
    assert set(response.json()) == {"answer", "suggests_consultation"}


# --- What is sent to the model ---------------------------------------------


def sent_turns(fake):
    """The conversation as (role, text) pairs, as the model would see it."""
    return [(c.role, c.parts[0].text) for c in fake.calls[0]["contents"]]


def test_only_the_last_six_history_turns_are_sent(client, gemini):
    fake = gemini('{"answer": "4 March 2027.", "suggests_consultation": false}')
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(10)
    ]

    response = post_chat(client, history=history)

    assert response.status_code == 200
    turns = sent_turns(fake)
    # Six turns of history plus the new question.
    assert len(turns) == 7
    assert [text for _, text in turns[:-1]] == [f"turn {i}" for i in range(4, 10)]
    # "assistant" is translated to the role name Gemini expects.
    assert [role for role, _ in turns[:-1]] == ["user", "model"] * 3


def test_short_history_is_sent_whole(client, gemini):
    fake = gemini('{"answer": "a", "suggests_consultation": false}')

    post_chat(client, history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])

    assert [text for _, text in sent_turns(fake)[:-1]] == ["hi", "hello"]


def test_records_are_resent_with_every_question(client, gemini):
    fake = gemini('{"answer": "a", "suggests_consultation": false}')

    post_chat(client, question="And her?", history=[{"role": "user", "content": "earlier"}])

    role, text = sent_turns(fake)[-1]
    assert role == "user"
    assert text == f"<records>\n{RECORDS}\n</records>\n\nQuestion: And her?"


def test_chat_model_is_configurable(client, gemini, monkeypatch):
    fake = gemini('{"answer": "a", "suggests_consultation": false}')
    assert post_chat(client).status_code == 200
    assert fake.calls[0]["model"] == "gemini-3.6-flash"

    monkeypatch.setenv("CHAT_MODEL", "some-other-model")
    post_chat(client)
    assert fake.calls[1]["model"] == "some-other-model"


def test_every_model_call_has_a_timeout(client, gemini, monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    fake = gemini('{"answer": "a", "suggests_consultation": false}')

    post_chat(client)

    assert fake.calls[0]["config"].http_options.timeout == 12500  # milliseconds


# --- Replies ---------------------------------------------------------------


def test_model_answer_is_returned(client, gemini):
    gemini('{"answer": "Due 4 March 2027.", "suggests_consultation": false}')

    assert post_chat(client).json() == {
        "answer": "Due 4 March 2027.",
        "suggests_consultation": False,
    }


def test_code_fenced_json_is_still_understood(client, gemini):
    gemini('```json\n{"answer": "Due 4 March 2027.", "suggests_consultation": true}\n```')

    assert post_chat(client).json() == {
        "answer": "Due 4 March 2027.",
        "suggests_consultation": True,
    }


@pytest.mark.parametrize(
    "reply",
    [
        "Bruno's booster is due in March.",       # prose, not JSON
        '{"answer": "cut off mid-',               # truncated
        '{"suggests_consultation": false}',       # no answer field
        '["not", "an", "object"]',
    ],
)
def test_unparseable_reply_fails_closed(client, gemini, reply):
    gemini(reply)

    response = post_chat(client)

    # Not an error and not a guess: a referral to a vet.
    assert response.status_code == 200
    assert response.json() == {"answer": FALLBACK, "suggests_consultation": True}


@pytest.mark.parametrize(
    "failure, expected_status, expected_detail",
    [
        (httpx.ReadTimeout("slow"), 504, "Language model timed out"),
        (errors.ServerError(504, {}), 504, "Language model timed out"),
        (errors.ServerError(500, {}), 502, "Language model request failed"),
        (errors.ClientError(429, {}), 502, "Language model request failed"),
        (httpx.ConnectError("down"), 502, "Language model could not be reached"),
        ("", 502, "Language model is unavailable"),  # empty reply
    ],
)
def test_model_failures_map_to_gateway_errors(client, gemini, failure, expected_status, expected_detail):
    gemini(failure)

    response = post_chat(client)

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


def test_upstream_error_text_is_not_passed_to_the_caller(client, gemini):
    gemini(errors.ServerError(500, {"error": {"message": "internal detail from provider"}}))

    assert "internal detail" not in post_chat(client).text
