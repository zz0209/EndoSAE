import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import acquire_realcolon_pilot as acquisition


class AcquisitionResumeTests(unittest.TestCase):
    def test_status_replace_retries_a_transient_windows_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            path.write_text('{"old": true}')
            original_replace = acquisition.os.replace
            attempts = []

            def locked_once(source, destination):
                attempts.append(destination)
                if len(attempts) == 1:
                    raise PermissionError("transient reader")
                return original_replace(source, destination)

            with patch.object(acquisition.os, "replace", side_effect=locked_once), patch.object(acquisition.time, "sleep"):
                acquisition.atomic_json(path, {"progress": 1})
            self.assertEqual(json.loads(path.read_text()), {"progress": 1})
            self.assertEqual(len(attempts), 2)

    def test_complete_partial_is_verified_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"downloaded archive fixture"
            name = "001-001_frames.tar.gz"
            (root / (name + ".partial")).write_bytes(payload)
            record = {"name": name, "bytes": len(payload), "md5": hashlib.md5(payload).hexdigest(),
                      "download_url": "https://example.invalid/not-called"}
            with patch.object(acquisition.urllib.request, "urlopen", side_effect=AssertionError("network must not be used")):
                result = acquisition.download(record, root)
            self.assertEqual((root / name).read_bytes(), payload)
            self.assertFalse((root / (name + ".partial")).exists())
            self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
