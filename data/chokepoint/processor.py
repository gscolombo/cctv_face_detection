import cv2 as cv
from tqdm import tqdm
import numpy as np

import os
import sys
import tarfile as tar
import io
from pathlib import Path
from pprint import pprint
from shutil import rmtree
import logging

logger = logging.getLogger(__name__)

BASE_PATH = Path(os.path.abspath(os.path.dirname(__file__)))
OUTPUT_PATH = os.path.join(BASE_PATH, "videos")

CODEC_FOURCC = cv.VideoWriter.fourcc(*'FFV1')
VIDEO_FORMAT = ".avi"

FRAME_WIDTH = 800
FRAME_HEIGHT = 600
FRAME_RATE = 30.0

    
def compose_video(name: str, frames: list, tqdm):
    output = cv.VideoWriter(filename=os.path.join(OUTPUT_PATH, f"{name}{VIDEO_FORMAT}"),
                            apiPreference=cv.CAP_FFMPEG,
                            fourcc=CODEC_FOURCC,
                            fps=FRAME_RATE,
                            frameSize=(FRAME_WIDTH, FRAME_HEIGHT))

    with tqdm(frames,
              desc=f"Composing video {name}",
              unit="frame",
              leave=False) as pbar:
        for frame in frames:
            output.write(cv.imread(frame))
            pbar.update(1)

    output.release()


def compose_chokepoint_video(path: Path, tqdm_func=None):
    _tqdm = tqdm if tqdm_func is None else tqdm_func
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    tmp_folder = BASE_PATH.joinpath("tmp")
    tmp_folder.mkdir(exist_ok=True)

    with tar.open(path, mode="r:xz") as xz:
        print(f"Extracting {path.name}", end="\r")
        xz.extractall(path=tmp_folder, filter='data')

        for _tarfile in xz:
            inner_tar_path = tmp_folder.joinpath(_tarfile.name)

            with tar.open(inner_tar_path, mode="r:xz") as inner_xz:
                print(f"Extracting {inner_tar_path.name}", end="")
                print("\033[1A", end="\r")

                inner_xz.extractall(path=tmp_folder, filter='data',
                                    members=[m for m in inner_xz if m.name.endswith('.jpg')])
                os.remove(inner_tar_path)

                frame_folders = [Path(tmp_folder, folder) for folder in os.listdir(tmp_folder)
                                 if folder.startswith(_tarfile.name.split(".")[0])]

                for folder in frame_folders:
                    compose_video(folder.name,
                                  [Path(folder, img)
                                   for img in sorted(os.listdir(folder))],
                                  tqdm_func)

                    rmtree(folder)
                    

def process_data(compose: bool, path: Path, tqdm_func, global_tqdm):
    try:
        if compose:
            compose_chokepoint_video(path, tqdm_func)
        else:
            return True

        global_tqdm.update(1)
        return True
    except Exception as e:
        logger.error(e)
        return
