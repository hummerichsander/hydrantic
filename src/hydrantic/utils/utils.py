from typing import Any
from importlib import import_module

import os
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional

import fsspec


def import_from_string(string: str) -> Any:
    """Imports an object from a string.

    :param string: String representation of an object to import.
    :return: The imported object."""

    module_name, name = string.rsplit(".", 1)
    module = import_module(module_name)
    object_ = getattr(module, name)
    return object_


def download_url(url: str, folder: str, filename: Optional[str] = None) -> str:
    """Downloads a file from an URL to the specified folder.
    Taken and adapted from pytorch_geometric:
        https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/download.html#download_url

    :param url: URL to download the file from.
    :param folder: Folder to download the file to.
    :param filename: Filename to save the file as.
    :return: The path to the downloaded file."""

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if os.path.exists(path):
        print("File {} already exists. Skipping download.".format(path))
        return path

    print("Downloading {} to {}".format(url, path))
    os.makedirs(folder, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with fsspec.open(path, "wb") as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path
