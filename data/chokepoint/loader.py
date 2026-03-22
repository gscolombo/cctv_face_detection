import requests
from bs4 import BeautifulSoup
from pprint import pprint
from tqdm import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool

import os
from shutil import rmtree
from pathlib import Path
import multiprocessing as mp
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

BASE_URL = "https://arma.sourceforge.net/chokepoint/"

DOWNLOAD_PATH = Path(os.path.abspath(os.path.dirname(__file__)), "raw")

CROPPED_FACE_IMAGES_DIRNAME = "cropped_face_images"
ORIGINAL_FILES_DIRNAME = "original_files"

BLOCK_SIZE = 1024

URL_LIST = List[List[str]]

PREFIX = "P2"
TO_DOWNLOAD = ["P2E_S2", "P2L_S2", "P2E_S1", "P2L_S1"]


def download_file(dirpath: Path, url: str, tqdm_func, global_tqdm):
    req = requests.get(url, stream=True)

    content_length = int(req.headers.get('content-length', 0))

    if content_length == 0:
        return False

    with tqdm_func(total=content_length,
                   unit="B",
                   unit_scale=True,
                   dynamic_ncols=True,
                   desc=dirpath.name,
                   leave=False) as pbar:
        try:
            with open(dirpath, "wb") as file:
                for chunk in req.iter_content(chunk_size=BLOCK_SIZE):
                    file.write(chunk)
                    pbar.update(len(chunk))
        except Exception as e:
            if dirpath.exists():
                rmtree(dirpath)
            return (False, dirpath.name, e)

    global_tqdm.update(1)

    return True, None, None


def scrape_urls() -> URL_LIST:
    urls = []

    res = requests.get(BASE_URL)

    soup = BeautifulSoup(res.content, "html.parser")

    for a in soup.select("a[name='download'] ~ ol li a"):
        url = a.get("href")
        filename: str = url.split("/")[-1]

        if filename.startswith(PREFIX):
            if filename.split(".", maxsplit=1)[0] in TO_DOWNLOAD:
                urls.append([ORIGINAL_FILES_DIRNAME, url])
            elif "S" not in filename:
                urls.append([CROPPED_FACE_IMAGES_DIRNAME, url])

    return urls
