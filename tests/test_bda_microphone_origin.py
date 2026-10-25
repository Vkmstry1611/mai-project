"""The recorder should only appear on browser origins that can request a mic."""
from bda.dashboard import microphone_origin_available


def test_microphone_origin_available():
    assert microphone_origin_available("http://127.0.0.1:8501/")
    assert microphone_origin_available("http://localhost:8501/")
    assert microphone_origin_available("https://visionbridge.example/")
    assert not microphone_origin_available("http://192.168.1.4:8501/")
