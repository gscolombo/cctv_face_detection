import requests
from bs4 import BeautifulSoup
from pprint import pprint
from tqdm import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool

import os
from pathlib import Path
import multiprocessing as mp
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

BASE_URL = "https://arma.sourceforge.net/chokepoint/"

DOWNLOAD_PATH = Path(os.path.abspath(os.path.dirname(__file__)), "raw")

CROPPED_FACE_IMAGES_DIRNAME = "cropped_face_images"
ORIGINAL_FILES_DIRNAME = "original_files"

NUM_CORES = mp.cpu_count()
BLOCK_SIZE = 1024

URL_LIST = List[List[str]]

def download_file(dirpath: Path, url: str, tqdm_func, global_tqdm):
    req = requests.get(url, stream=True)
    
    content_length = int(req.headers.get('content-length', 0))
    
    if content_length == 0:
        return False
        
    with tqdm_func(total=content_length, 
                   unit="B", 
                   unit_scale=True, 
                   dynamic_ncols=True,
                   desc=dirpath.name) as pbar:
        try: 
            with open(dirpath, "wb") as file:
                for chunk in req.iter_content(chunk_size=BLOCK_SIZE):
                    file.write(chunk)
                    pbar.update(len(chunk))
        except Exception as e:
            logger.error(e)
            return
    
    global_tqdm.update(1)
    
    return dirpath
    
def download_error(result):
    raise RuntimeError("Could not download file")

def download_complete(dirpath: str):
    return

def scrape_urls() -> URL_LIST:
    urls = []

    res = requests.get(BASE_URL)

    soup = BeautifulSoup(res.content, "html.parser")

    for a in soup.select("a[name='download'] ~ ol li a"):
        url = a.get("href")
        filename: str = url.split("/")[-1]

        if "S" in filename:
            urls.append([ORIGINAL_FILES_DIRNAME, url])
            continue

        if not "groundtruth" in filename:
            urls.append([CROPPED_FACE_IMAGES_DIRNAME, url])
            
    return urls


if __name__ == "__main__":    
    os.makedirs(DOWNLOAD_PATH.joinpath(ORIGINAL_FILES_DIRNAME), exist_ok=True)
    os.makedirs(DOWNLOAD_PATH.joinpath(CROPPED_FACE_IMAGES_DIRNAME), exist_ok=True)
    
    urls: URL_LIST = scrape_urls()
    
    for l in urls:
        l[0] = Path(DOWNLOAD_PATH, l[0], l[1].split("/")[-1])
        

    process_count = NUM_CORES // 2 or 1

    pool = TqdmMultiProcessPool(process_count)
    tasks = [(download_file, url) for url in urls]
    
    print("Starting bulk file download")
    print(f"Using {process_count} out of {NUM_CORES} vCPUs\n")
    
    with tqdm(total=len(tasks), 
              desc="Downloaded files\n",
              unit="file", 
              dynamic_ncols=True) as global_progress:
        pool.map(global_progress, tasks, download_error, download_complete)