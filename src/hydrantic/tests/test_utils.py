import pytest
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from ..utils.utils import import_from_string, download_url


class TestImportFromString:
    """Test import_from_string utility function."""

    def test_import_builtin_module(self):
        """Test importing a built-in module object."""
        json_module = import_from_string("json.dumps")
        import json

        assert json_module is json.dumps

    def test_import_standard_library_class(self):
        """Test importing a class from standard library."""
        path_class = import_from_string("pathlib.Path")
        from pathlib import Path

        assert path_class is Path

    def test_import_torch_object(self):
        """Test importing a PyTorch object."""
        nn_module = import_from_string("torch.nn.Module")
        import torch.nn

        assert nn_module is torch.nn.Module

    def test_import_invalid_module(self):
        """Test that importing non-existent module raises error."""
        with pytest.raises(ModuleNotFoundError):
            import_from_string("nonexistent.module.Object")

    def test_import_invalid_attribute(self):
        """Test that importing non-existent attribute raises error."""
        with pytest.raises(AttributeError):
            import_from_string("json.NonExistentClass")

    def test_import_nested_attribute(self):
        """Test importing nested attribute."""
        functional = import_from_string("torch.nn.functional.relu")
        import torch.nn.functional as F

        assert functional is F.relu


class TestDownloadUrl:
    """Test download_url utility function."""

    def test_download_with_explicit_filename(self, tmp_download_path):
        """Test downloading a file with explicit filename.

        :param tmp_download_path: Temporary download directory fixture."""
        url = "https://example.com/test.txt"
        mock_data = b"test content"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.side_effect = [mock_data, b""]
            mock_urlopen.return_value = mock_response

            result_path = download_url(url, str(tmp_download_path), "custom.txt")

            assert result_path == os.path.join(str(tmp_download_path), "custom.txt")
            assert os.path.exists(result_path)

    def test_download_infers_filename(self, tmp_download_path):
        """Test downloading a file with inferred filename.

        :param tmp_download_path: Temporary download directory fixture."""
        url = "https://example.com/data.csv"
        mock_data = b"col1,col2\n1,2"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.side_effect = [mock_data, b""]
            mock_urlopen.return_value = mock_response

            result_path = download_url(url, str(tmp_download_path))

            assert result_path == os.path.join(str(tmp_download_path), "data.csv")
            assert os.path.exists(result_path)

    def test_download_creates_folder(self, tmp_path):
        """Test that download creates the target folder if it doesn't exist.

        :param tmp_path: Pytest temporary path fixture."""
        new_folder = tmp_path / "new" / "nested" / "folder"
        url = "https://example.com/file.txt"
        mock_data = b"content"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.side_effect = [mock_data, b""]
            mock_urlopen.return_value = mock_response

            result_path = download_url(url, str(new_folder), "test.txt")

            assert os.path.exists(new_folder)
            assert os.path.exists(result_path)

    def test_download_skips_existing_file(self, tmp_download_path, capsys):
        """Test that download skips if file already exists.

        :param tmp_download_path: Temporary download directory fixture.
        :param capsys: Pytest fixture to capture stdout/stderr."""
        existing_file = tmp_download_path / "existing.txt"
        existing_file.write_text("existing content")

        url = "https://example.com/test.txt"
        result_path = download_url(url, str(tmp_download_path), "existing.txt")

        captured = capsys.readouterr()
        assert "already exists" in captured.out
        assert result_path == str(existing_file)
        assert existing_file.read_text() == "existing content"

    def test_download_url_with_query_params(self, tmp_download_path):
        """Test downloading URL with query parameters strips them from filename.

        :param tmp_download_path: Temporary download directory fixture."""
        url = "https://example.com/data.csv?version=1&key=abc"
        mock_data = b"data"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.side_effect = [mock_data, b""]
            mock_urlopen.return_value = mock_response

            result_path = download_url(url, str(tmp_download_path))

            assert result_path == os.path.join(str(tmp_download_path), "data.csv")
            assert "?" not in os.path.basename(result_path)

    def test_download_handles_large_file(self, tmp_download_path):
        """Test downloading file in chunks.

        :param tmp_download_path: Temporary download directory fixture."""
        url = "https://example.com/large.bin"
        chunk_size = 10 * 1024 * 1024
        chunk1 = b"x" * chunk_size
        chunk2 = b"y" * chunk_size

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.side_effect = [chunk1, chunk2, b""]
            mock_urlopen.return_value = mock_response

            result_path = download_url(url, str(tmp_download_path))

            assert os.path.exists(result_path)
            file_size = os.path.getsize(result_path)
            assert file_size == len(chunk1) + len(chunk2)

    def test_download_prints_progress(self, tmp_download_path, capsys):
        """Test that download prints progress message.

        :param tmp_download_path: Temporary download directory fixture.
        :param capsys: Pytest fixture to capture stdout/stderr."""
        url = "https://example.com/file.txt"
        mock_data = b"content"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.side_effect = [mock_data, b""]
            mock_urlopen.return_value = mock_response

            download_url(url, str(tmp_download_path))

            captured = capsys.readouterr()
            assert "Downloading" in captured.out
            assert url in captured.out
