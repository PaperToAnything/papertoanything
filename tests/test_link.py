import json
import os
import unittest

from papertoanything.link import LAB_URL, extract_payload, from_url, payload_to_dict, spec_to_payload, to_url
from papertoanything.spec import ModelSpec

VEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors", "link.json")


class LinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(VEC, encoding="utf-8") as fh:
            cls.v = json.load(fh)

    def test_vector_encode(self):
        self.assertEqual(json.dumps(self.v["spec"], separators=(",", ":"), ensure_ascii=False), self.v["json"])
        self.assertEqual(spec_to_payload(self.v["spec"]), self.v["payload"])
        self.assertEqual(self.v["fragment"], "#s=" + self.v["payload"])
        self.assertTrue(self.v["url"].startswith(LAB_URL + "#s="))

    def test_vector_decode(self):
        self.assertEqual(payload_to_dict(self.v["payload"]), self.v["spec"])
        self.assertEqual(from_url(self.v["url"]).to_dict(), self.v["spec"])

    def test_decodes_browser_encoder_output(self):
        self.assertNotEqual(self.v["payloadNode"], self.v["payload"])  # different bytes...
        self.assertEqual(payload_to_dict(self.v["payloadNode"]), self.v["spec"])  # ...same spec

    def test_no_padding_and_url_safe(self):
        p = self.v["payload"]
        for c in "+/=":
            self.assertNotIn(c, p)

    def test_extract_forms(self):
        p = self.v["payload"]
        for form in (self.v["url"], "#s=" + p, "s=" + p, p, self.v["url"] + "&x=1", p + "=="):
            self.assertEqual(from_url(form).to_dict(), self.v["spec"])
        with self.assertRaises(ValueError):
            extract_payload("#a=1&b=2")

    def test_round_trip_modelspec(self):
        spec = ModelSpec.from_dict(self.v["spec"])
        self.assertEqual(from_url(to_url(spec)), spec)

    def test_env_override(self):
        os.environ["PTA_LAB_URL"] = "http://localhost:5173/"
        try:
            self.assertTrue(to_url(self.v["spec"]).startswith("http://localhost:5173/#s="))
        finally:
            del os.environ["PTA_LAB_URL"]

    def test_bomb_guard(self):
        from papertoanything.link import b64url_encode, deflate_raw, inflate_raw

        blob = deflate_raw(b"[" + b"0," * 100000 + b"0]")
        with self.assertRaises(ValueError):
            inflate_raw(blob, limit=1000)
        self.assertEqual(len(inflate_raw(blob)), 200003)
        self.assertTrue(b64url_encode(b"\xff\xfe"))


if __name__ == "__main__":
    unittest.main()
