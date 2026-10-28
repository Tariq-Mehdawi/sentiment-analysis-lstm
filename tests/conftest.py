import pytest
from fastapi.testclient import TestClient

from src.app.main import app


@pytest.fixture(scope="session")
def client():
    # "with" runs the lifespan, so the model is loaded once for all tests
    with TestClient(app) as c:
        yield c