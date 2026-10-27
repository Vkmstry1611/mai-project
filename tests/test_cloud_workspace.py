"""Cloud adapter keeps the MAI workspace connected to the DVC-selected model."""

import json
import unittest
from unittest.mock import patch

from PIL import Image

from mlops.cloud_workspace import CloudBlipSession


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"answer": "stop", "model_sha256": "abc123",
                           "latency_seconds": 1.0}).encode()


class _Opener:
    def open(self, request, timeout):
        assert request.full_url == "http://127.0.0.1:8000/predict"
        assert timeout == 120
        payload = json.loads(request.data)
        assert payload["question"] == "What word is on the sign?"
        assert payload["image_base64"]
        return _Response()


class CloudWorkspaceTest(unittest.TestCase):
    def test_blip_option_calls_cloud_localhost_api(self):
        session = CloudBlipSession()
        with patch("mlops.cloud_workspace.build_opener", return_value=_Opener()):
            result = session.request(Image.new("RGB", (16, 16), "red"),
                                     "What word is on the sign?")
        self.assertEqual(result[0], "stop")
        self.assertEqual(result[3]["model_sha256"], "abc123")
        self.assertEqual(result[3]["backend"], "AWS EC2 CPU")

    def test_private_ocr_is_not_claimed(self):
        with self.assertRaisesRegex(ValueError, "does not include"):
            CloudBlipSession().request(Image.new("RGB", (16, 16)),
                                       "Read this", experimental=True)


if __name__ == "__main__":
    unittest.main()
