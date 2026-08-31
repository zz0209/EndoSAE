import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.checkpoint_preflight import (
    CheckpointPreflightError,
    PreflightLimits,
    preflight_checkpoint,
)


class CheckpointPreflightTests(unittest.TestCase):
    def test_zip_metadata_passes_without_reading_pickle_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pth"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("archive/data.pkl", b"not-executed")
                archive.writestr("archive/data/0", b"tensor-bytes")
            result = preflight_checkpoint(path)
            self.assertEqual(result.container_format, "zip")
            self.assertTrue(result.contains_pickle_named_member)
            self.assertEqual(result.member_count, 2)

    def test_raw_pickle_like_bytes_stop_as_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.pth"
            path.write_bytes(b"\x80\x04cposix\nsystem\n.")
            with self.assertRaisesRegex(CheckpointPreflightError, "raw-pickle"):
                preflight_checkpoint(path)

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../escape.pkl", b"x")
            with self.assertRaisesRegex(CheckpointPreflightError, "unsafe archive"):
                preflight_checkpoint(path)

    def test_zip_bomb_ratio_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratio.zip"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("archive/data/0", b"0" * 100_000)
            limits = PreflightLimits(max_compression_ratio=5.0)
            with self.assertRaisesRegex(CheckpointPreflightError, "compression ratio"):
                preflight_checkpoint(path, limits)

    def test_tar_regular_members_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.tar"
            with tarfile.open(path, "w") as archive:
                payload = b"metadata-only"
                info = tarfile.TarInfo("archive/data.pkl")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            result = preflight_checkpoint(path)
            self.assertEqual(result.container_format, "tar")
            self.assertTrue(result.contains_pickle_named_member)

    def test_tar_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "link.tar"
            with tarfile.open(path, "w") as archive:
                info = tarfile.TarInfo("archive/link")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                archive.addfile(info)
            with self.assertRaisesRegex(CheckpointPreflightError, "link/device/FIFO"):
                preflight_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
