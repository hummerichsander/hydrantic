import logging
import os
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 10 * 1024 * 1024


def download_url(
    url: str,
    folder: str | Path,
    filename: str | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> Path:
    """Downloads a file to a folder, skipping the download if it is already there.

    :param url: URL to download from.
    :param folder: Folder to download into.
    :param filename: Name to save the file as, derived from the URL if omitted.
    :param ssl_context: TLS context to use. The default context verifies certificates; pass an
        explicit context only if you knowingly need to relax that.
    :return: Path to the downloaded file."""

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    folder = Path(folder)
    path = folder / filename

    if path.exists():
        logger.info("%s already exists, skipping download.", path)
        return path

    logger.info("Downloading %s to %s", url, path)
    folder.mkdir(parents=True, exist_ok=True)

    # Download to a temporary name and rename on success, so an interrupted download cannot
    # leave a truncated file that the `path.exists()` check above would later treat as complete.
    partial = path.with_name(f"{path.name}.part")
    try:
        with (
            urllib.request.urlopen(url, context=ssl_context) as response,
            open(partial, "wb") as file,
        ):
            while chunk := response.read(_CHUNK_SIZE):
                file.write(chunk)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    os.replace(partial, path)
    return path


def unzip_file(zip_path: str | Path, folder: str | Path, remove_zip: bool = False) -> Path:
    """Extracts a zip archive into a folder.

    :param zip_path: Path to the zip file.
    :param folder: Folder to extract into.
    :param remove_zip: Whether to delete the archive afterwards.
    :return: Path to the folder the archive was extracted into."""

    zip_path, folder = Path(zip_path), Path(folder)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(folder)
    logger.info("Extracted %s into %s", zip_path, folder)

    if remove_zip:
        zip_path.unlink()
        logger.info("Removed %s", zip_path)

    return folder


def extract_archive(
    archive_path: str | Path, folder: str | Path, remove_archive: bool = False
) -> Path:
    """Extracts any archive format `shutil` supports, including tar and zip.

    :param archive_path: Path to the archive.
    :param folder: Folder to extract into.
    :param remove_archive: Whether to delete the archive afterwards.
    :return: Path to the folder the archive was extracted into."""

    archive_path, folder = Path(archive_path), Path(folder)

    shutil.unpack_archive(archive_path, folder)
    logger.info("Extracted %s into %s", archive_path, folder)

    if remove_archive:
        archive_path.unlink()
        logger.info("Removed %s", archive_path)

    return folder
