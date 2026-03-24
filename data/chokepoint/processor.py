import cv2 as cv
from tqdm import tqdm
import numpy as np
from pymongo import MongoClient
from deepface.modules.representation import represent
from deepface.modules.datastore import register
from deepface.modules.exceptions import DuplicateEntryError

import os
import tarfile as tar
import io
from pathlib import Path
from shutil import rmtree
from PIL import Image
from datetime import datetime

BASE_PATH = Path(os.path.abspath(os.path.dirname(__file__)))
OUTPUT_PATH = os.path.join(BASE_PATH, "videos")

CODEC_FOURCC = cv.VideoWriter.fourcc(*"FFV1")
VIDEO_FORMAT = ".avi"

FRAME_WIDTH = 800
FRAME_HEIGHT = 600
FRAME_RATE = 30.0

MODEL_NAME = "SFace"

db_client = MongoClient(os.environ["MONGODB_URI"])


def compose_video(name: str, frames: list):
    try:
        output = cv.VideoWriter(
            filename=os.path.join(OUTPUT_PATH, name),
            apiPreference=cv.CAP_FFMPEG,
            fourcc=CODEC_FOURCC,
            fps=FRAME_RATE,
            frameSize=(FRAME_WIDTH, FRAME_HEIGHT),
        )

        with tqdm(frames, desc=name, unit="frame", leave=False) as pbar:
            for frame in frames:
                output.write(cv.imread(frame))
                pbar.update(1)

        output.release()
    except Exception as e:
        with open(BASE_PATH.joinpath("logs.txt", "a+")) as log:
            log.write(f"ERROR@{datetime.now()}")
            log.write(f"\nError during processing of file {name}")
            log.write(e)


def compose_chokepoint_videos(path: Path):
    # Prepare file system
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    tmp_folder = BASE_PATH.joinpath("tmp")
    tmp_folder.mkdir(exist_ok=True)

    # Open and extract tar file contents to temporary storage
    with tar.open(path, mode="r:xz") as xz:
        print(f"Extracting {path.stem}")
        xz.extractall(path=tmp_folder, filter="data")

        for _tarfile in xz:
            inner_tar_path = tmp_folder.joinpath(_tarfile.name)
            video_name = f"{inner_tar_path.stem.split(".")[0]}{VIDEO_FORMAT}"

            if video_name in os.listdir(OUTPUT_PATH):
                print(f"Video {video_name} already composed. Skipping.")
                os.remove(inner_tar_path)
                continue

            # Open and extract inner tar files to temporary storage
            with tar.open(inner_tar_path, mode="r:xz") as inner_xz:

                print(f"Extracting {inner_tar_path.stem}")
                inner_xz.extractall(
                    path=tmp_folder,
                    filter="data",
                    members=[
                        m
                        for m in inner_xz
                        if m.name.endswith(".jpg")  # Filter for .jpg files
                    ],
                )
                os.remove(inner_tar_path)

                frame_folders = [
                    tmp_folder.joinpath(folder)
                    for folder in os.listdir(tmp_folder)
                    if folder.startswith(_tarfile.name.split(".")[0])
                ]

                for folder in frame_folders:
                    compose_video(
                        video_name,
                        [folder.joinpath(img) for img in sorted(os.listdir(folder))],
                    )

                    rmtree(folder)


def populate_db(path: Path, _filter: list[str]):
    # Use database connection pool
    global db_client

    db = db_client.get_database(os.environ.get("DB_NAME"))
    persons_coll = db.get_collection("persons")

    # Prepare data for storage
    data: list[Person] = []

    tmp_folder = BASE_PATH.joinpath("tmp")
    tmp_folder.mkdir(exist_ok=True)

    with tar.open(path, mode="r:xz") as xz:
        print(f"Extracting {path.name}")
        xz.extractall(
            path=tmp_folder,
            filter="data",
            members=[
                m
                for m in xz
                if m.name.endswith(".pgm") and any([f in m.name for f in _filter])
            ],
        )

        folders: list[Path] = [
            folder
            for folder in tmp_folder.iterdir()
            if os.path.basename(xz.name).split(".")[0] in os.path.basename(folder)
        ]

        # Create embeddings for each id in each video
        last_id = 0
        frame_interval = 5  #

        for folder in tqdm(
            folders, desc=f"{Path(xz.name).stem}", unit="folder", leave=False
        ):
            for _id in tqdm(
                os.listdir(folder), desc=f"ID {folder.stem}", unit="id", leave=False
            ):
                f = -1

                images = [
                    np.array(Image.open(path, formats=["PPM"]).convert("RGB"))
                    for path in Path(folder, _id).iterdir()
                    if (f := f + 1) % frame_interval
                ]

                embeddings = []
                kwargs = {
                    "model_name": MODEL_NAME,
                    "enforce_detection": False,
                    "detector_backend": "skip",
                    "database_type": "mongo",
                    "connection": db_client,
                }

                try:
                    n = register(images, **kwargs)[
                        "inserted"
                    ]  # Store embeddings in DeepFace format

                    # Relate the inserted embeddings IDs with the person ID
                    persons_coll.update_one(
                        {"_id": _id},
                        {
                            "$push": {
                                "embedding_id": {
                                    "$each": list(range(last_id, last_id + n))
                                }
                            }
                        },
                        upsert=True,
                    )

                    last_id += n
                except DuplicateEntryError as e:
                    continue

            rmtree(folder)
