import requests
from bs4 import BeautifulSoup
from pprint import pprint
from tqdm import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool
import multiprocessing as mp

import os
from shutil import rmtree
from pathlib import Path
import multiprocessing as mp
from typing import List, Tuple, Iterable


DOWNLOAD_PATH = Path(os.path.abspath(os.path.dirname(__file__)), "raw")

URL_LIST = List[List[str]]


def download_file(dirpath: Path, url: str, tqdm_func, global_tqdm):
    req = requests.get(url, stream=True)

    content_length = int(req.headers.get("content-length", 0))

    if content_length == 0:
        return False

    with tqdm_func(
        total=content_length,
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        desc=dirpath.name,
        leave=False,
    ) as pbar:
        try:
            with open(dirpath, "wb") as file:
                for chunk in req.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))
        except Exception as e:
            if dirpath.exists():
                rmtree(dirpath)
            return (False, dirpath, e)

    global_tqdm.update(1)

    return True


def scrape_urls(_filter: Iterable[str]) -> URL_LIST:
    urls = []

    res = requests.get(os.environ["CHOKEPOINT_DS_URL"])
    soup = BeautifulSoup(res.content, "html.parser")

    for a in soup.select("a[name='download'] ~ ol li a"):
        url = a.get("href")
        filename: str = url.split("/")[-1].strip()

        if filename in _filter:
            if "S" not in filename:
                urls.append([os.environ["CROPPED_FACE_IMAGES_DIRNAME"], url])
            else:
                urls.append([ps.environ["ORIGINAL_FILES_DIRNAME"], url])

    return urls


def on_download_end(*args):
    ok = args[0]
    if not ok:
        _, path, err = args
        with open(DOWNLOAD_PATH.parent, "a+") as log:
            log.write(f"ERROR@{datetime.now()}")
            log.write(f"\nError during downloading of file {path.stem}")
            log.write(err)

    os.remove(path)

def extract_data():
    # Check if data is already available locally
    data_loaded = DOWNLOAD_PATH.exists()

    if data_loaded:
        while y := input("Download path found. Download again? (y-Y|n-N) "):
            if y in ["y", "Y", "n", "N"]:
                if y in ["y", "Y"]:
                    rmtree(DOWNLOAD_PATH)
                    break
                else:
                    return
            else:   
                print("Invalid option.")

    # Prepare file system
    print("Creating storage folders for raw data")

    os.makedirs(DOWNLOAD_PATH.joinpath(ORIGINAL_FILES_DIRNAME), exist_ok=True)
    os.makedirs(DOWNLOAD_PATH.joinpath(CROPPED_FACE_IMAGES_DIRNAME), exist_ok=True)

    # Scrape files URLs
    files = list(
        map(lambda x: x.strip(), os.environ["CHOKEPOINT_FILES_TO_DOWNLOAD"].split(","))
    )

    urls = scrape_urls(files)

    if len(urls) == 0:
        print("No URL found. Aborting.")
        exit(1)

    print("URLs selected: ")
    for _, url in urls:
        print(f" -  {url}")

    for l in urls:
        l[0] = Path(DOWNLOAD_PATH, l[0], l[1].split("/")[-1])

    # Get user confirmation to download data
    while (_continue := input("Continue? [y-Y, n-N] ")) not in ["y", "Y", "n", "N"]:
        print("Invalid option.")

    if _continue in ["y", "Y"]:
        # Download data in parallel
        num_cores = mp.cpu_count()
        process_count = min(num_cores // 2, len(urls)) or 1

        pool = TqdmMultiProcessPool(process_count)

        tasks = [(download_file, url) for url in urls]

        print(f"\nDownloading data using {process_count} out of {num_cores} vCPUs\n")

        with tqdm(
            total=len(tasks), desc="Downloaded files", unit="file", dynamic_ncols=True
        ) as global_progress:
            pool.map(global_progress, tasks, on_download_end, on_download_end)

        print(f"{files_downloaded} out of {len(tasks)} files successfully downloaded.")
    else:
        rmtree(DOWNLOAD_PATH)
