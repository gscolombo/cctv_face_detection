import cv2 as cv
from tqdm import tqdm
import numpy as np

import os
import tarfile as tar
import io
from pathlib import Path
from pprint import pprint
from shutil import rmtree

BASE_PATH = Path(os.path.abspath(os.path.dirname(__file__)))
OUTPUT_PATH = os.path.join(BASE_PATH, "videos")

CODEC_FOURCC = cv.VideoWriter.fourcc(*'FFV1')
VIDEO_FORMAT = ".avi"

FRAME_WIDTH = 800
FRAME_HEIGHT = 600
FRAME_RATE = 30.0


def compose_chokepoint_video(path: Path, tqdm_func=None):
    _tqdm = tqdm if tqdm_func is None else tqdm_func
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    tmp_folder = BASE_PATH.joinpath("tmp")

    with tar.open(path, mode="r:xz") as xz:
        xz.extractall(path=tmp_folder, filter='data')

        for _tarfile in os.listdir(tmp_folder):
            video_name = _tarfile.split(".")[0]

            frame_folder = tmp_folder.joinpath(video_name)

            output = cv.VideoWriter(filename=os.path.join(OUTPUT_PATH, f"{video_name}{VIDEO_FORMAT}"),
                                    apiPreference=cv.CAP_FFMPEG,
                                    fourcc=CODEC_FOURCC,
                                    fps=FRAME_RATE,
                                    frameSize=(FRAME_WIDTH, FRAME_HEIGHT))

            with tar.open(tmp_folder.joinpath(_tarfile), mode="r:xz") as inner_xz:
                inner_xz.extractall(path=tmp_folder, filter='data',
                                    members=[m for m in inner_xz if m.name.endswith('.jpg')])

                frame_files = os.listdir(frame_folder)
                frame_files.sort()
                
                for file in _tqdm(frame_files, desc=f"Composing video {video_name}", unit="frame"):
                    frame_path = Path(frame_folder, file)
                    output.write(cv.imread(frame_path))

                output.release()            
                
        
        rmtree(tmp_folder)      