import logging
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hydrantic.utils import download_url, extract_archive, unzip_file


def fake_response(*chunks: bytes) -> MagicMock:
    """Builds a context-manager mock that yields `chunks` then EOF.

    :param chunks: The chunks the fake response should return.
    :return: The mock to patch `urlopen` with."""

    response = MagicMock()
    response.read.side_effect = [*chunks, b""]
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


class TestDownloadUrl:
    """Tests the download helper."""

    def test_downloads_and_writes(self, tmp_download_path: Path) -> None:
        with patch("urllib.request.urlopen", return_value=fake_response(b"payload")):
            path = download_url("https://example.com/file.txt", tmp_download_path)

        assert path == tmp_download_path / "file.txt"
        assert path.read_bytes() == b"payload"

    def test_joins_multiple_chunks(self, tmp_download_path: Path) -> None:
        with patch("urllib.request.urlopen", return_value=fake_response(b"abc", b"def")):
            path = download_url("https://example.com/file.bin", tmp_download_path)

        assert path.read_bytes() == b"abcdef"

    def test_explicit_filename(self, tmp_download_path: Path) -> None:
        with patch("urllib.request.urlopen", return_value=fake_response(b"x")):
            path = download_url("https://example.com/file.txt", tmp_download_path, "other.txt")

        assert path.name == "other.txt"

    def test_strips_query_string(self, tmp_download_path: Path) -> None:
        with patch("urllib.request.urlopen", return_value=fake_response(b"x")):
            path = download_url("https://example.com/f.txt?token=abc", tmp_download_path)

        assert path.name == "f.txt"

    def test_creates_missing_folder(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested"
        with patch("urllib.request.urlopen", return_value=fake_response(b"x")):
            path = download_url("https://example.com/f.txt", target)

        assert path.exists()

    def test_skips_an_existing_file(self, tmp_download_path: Path) -> None:
        existing = tmp_download_path / "file.txt"
        existing.write_bytes(b"already here")

        with patch("urllib.request.urlopen") as urlopen:
            path = download_url("https://example.com/file.txt", tmp_download_path)

        urlopen.assert_not_called()
        assert path.read_bytes() == b"already here"

    def test_logs_progress(self, tmp_download_path: Path, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="hydrantic.utils"):
            with patch("urllib.request.urlopen", return_value=fake_response(b"x")):
                download_url("https://example.com/file.txt", tmp_download_path)

        assert "Downloading" in caplog.text

    def test_verifies_certificates_by_default(self, tmp_download_path: Path) -> None:
        # The old implementation passed ssl._create_unverified_context(), silently disabling
        # certificate verification for every download the library made.
        with patch("urllib.request.urlopen", return_value=fake_response(b"x")) as urlopen:
            download_url("https://example.com/f.txt", tmp_download_path)

        assert urlopen.call_args.kwargs["context"] is None

    def test_failure_leaves_no_partial_file(self, tmp_download_path: Path) -> None:
        response = MagicMock()
        response.read.side_effect = [b"half", OSError("connection reset")]
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=response):
            with pytest.raises(OSError, match="connection reset"):
                download_url("https://example.com/file.txt", tmp_download_path)

        assert list(tmp_download_path.iterdir()) == []


class TestUnzipFile:
    """Tests the zip extraction helper."""

    @pytest.fixture
    def archive(self, tmp_path: Path) -> Path:
        path = tmp_path / "data.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("inner.txt", "content")
        return path

    def test_extracts(self, archive: Path, tmp_path: Path) -> None:
        target = tmp_path / "out"
        folder = unzip_file(archive, target)

        assert folder == target
        assert (target / "inner.txt").read_text() == "content"

    def test_keeps_the_archive_by_default(self, archive: Path, tmp_path: Path) -> None:
        unzip_file(archive, tmp_path / "out")
        assert archive.exists()

    def test_removes_the_archive_on_request(self, archive: Path, tmp_path: Path) -> None:
        unzip_file(archive, tmp_path / "out", remove_zip=True)
        assert not archive.exists()

    def test_extract_archive_handles_zip(self, archive: Path, tmp_path: Path) -> None:
        target = tmp_path / "out2"
        extract_archive(archive, target)
        assert (target / "inner.txt").read_text() == "content"


class TestRemovedHelpers:
    """Tests that the string-import registry is gone."""

    def test_import_from_string_is_removed(self) -> None:
        import hydrantic.utils as utils

        assert not hasattr(utils, "import_from_string")

    def test_utils_is_importable_directly(self) -> None:
        from hydrantic.utils import download_url as imported

        assert imported is download_url
