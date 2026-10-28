import pytest

from src.predict import SentimentPredictor

POSITIVE = "This movie completely blew me away, brilliant acting."
NEGATIVE = "Boring, predictable and a total waste of time."


# 1. API contract: the doors exist and answer in the right shape
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_contract(client):
    r = client.post("/predict", json={"text": POSITIVE})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"label", "score", "confidence"}
    assert body["label"] in {"POSITIVE", "NEGATIVE"}
    assert 0.0 <= body["score"] <= 1.0
    assert 0.5 <= body["confidence"] <= 1.0


# 2. Model loads: the model, tokenizer and config open and match
def test_model_loads():
    p = SentimentPredictor(model_dir="models")
    assert p.model.output_shape == (None, 1)
    assert p.config["MAX_LEN"] == 200
    assert p.config["VOCAB_SIZE"] == 10000


# 3. Input schema: the bouncer rejects bad input with 422
@pytest.mark.parametrize(
    "payload",
    [{"text": 123}, {"text": ""}, {"text": "    "}, {}, {"review": "good"}],
)
def test_invalid_input_rejected(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


# 4. Edge cases: weird but valid input must not crash
@pytest.mark.parametrize(
    "text",
    [
        "great movie " * 1500,  # ~3000 words, far longer than MAX_LEN=200
        "!!! 123 ???",          # becomes empty after cleaning
    ],
)
def test_edge_cases(client, text):
    r = client.post("/predict", json={"text": text})
    assert r.status_code == 200


# 5. Deterministic: same input gives the same output, and the labels make sense
def test_deterministic_prediction(client):
    r1 = client.post("/predict", json={"text": POSITIVE}).json()
    r2 = client.post("/predict", json={"text": POSITIVE}).json()
    assert r1 == r2
    assert r1["label"] == "POSITIVE"

    neg = client.post("/predict", json={"text": NEGATIVE}).json()
    assert neg["label"] == "NEGATIVE"