import os
from pathlib import Path

from tqdm import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool
import multiprocessing as mp

from loader import (scrape_urls, download_file,
                    URL_LIST, ORIGINAL_FILES_DIRNAME,
                    CROPPED_FACE_IMAGES_DIRNAME, DOWNLOAD_PATH)
from processor import process_data, BASE_PATH

NUM_CORES = mp.cpu_count()

files_downloaded = 0
files_processed = 0


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def on_processing_end(ok):
    global files_processed

    if ok:
        files_processed += 1


def on_download_end(result):
    global files_downloaded

    ok, path, err = result
    if ok:
        files_downloaded += 1
    else:
        print(f"\nError during download of file {path}")
        print(err, end="")
        print("\033[2A", end="\r")


def extract_data():
    print("Creating storage folders for raw data")

    os.makedirs(DOWNLOAD_PATH.joinpath(ORIGINAL_FILES_DIRNAME), exist_ok=True)
    os.makedirs(DOWNLOAD_PATH.joinpath(
        CROPPED_FACE_IMAGES_DIRNAME), exist_ok=True)

    print("Gathering URLs")
    urls: URL_LIST = scrape_urls()

    if len(urls) == 0:
        print("No URL found. Aborting.")
        exit(1)

    print("URLs selected: ")
    for _, url in urls:
        print(f" -  {url}")

    for l in urls:
        l[0] = Path(DOWNLOAD_PATH, l[0], l[1].split("/")[-1])

    process_count = min(NUM_CORES // 2, len(urls)) or 1

    pool = TqdmMultiProcessPool(process_count)

    to_download = [(download_file, url) for url in urls]

    print(
        f"\nDownloading data using {process_count} out of {NUM_CORES} vCPUs\n")

    print("Starting bulk file download")

    with tqdm(total=len(to_download),
              desc="Downloaded files",
              unit="file",
              dynamic_ncols=True) as global_progress:
        pool.map(global_progress, to_download,
                 lambda _: None, on_download_end)

    clear_terminal()
    print(f"{files_downloaded} out of {len(to_download)} files successfully downloaded.")


if __name__ == "__main__":
    clear_terminal()

    data_loaded = DOWNLOAD_PATH.exists()

    if data_loaded:
        while (y := input("Download path found. Download again? (y-Y|n-N) ")):
            if y in ["y", "Y", "n", "N"]:
                if y in ["y", "Y"]:
                    extract_data()
                else:
                    break
            else:
                print("Invalid input.")
    else:
        extract_data()

    to_processing = [(process_data, (_dir.name == ORIGINAL_FILES_DIRNAME, path))
                     for _dir in DOWNLOAD_PATH.iterdir()
                     for path in _dir.iterdir()]

    if len(to_processing) > 0:
        print("\nStarting file processing")

        process_count = NUM_CORES * 3 // 4 or 1
        pool = TqdmMultiProcessPool(process_count)

        with tqdm(total=len(to_processing),
                  desc="Processed files",
                  unit="file",
                  dynamic_ncols=True) as global_progress:
            pool.map(global_progress, to_processing,
                     lambda _: None, on_processing_end)

        os.rmdir(BASE_PATH.joinpath("tmp"))
    else:
        print("No file to process.")
