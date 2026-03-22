from deepface.modules.detection import extract_faces
from deepface.modules.exceptions import FaceNotDetected

import numpy as np
import cv2 as cv
import pyarrow as pa
import pyarrow.parquet as pq

import os
import resource
import psutil
from pathlib import Path
import sys

from datetime import datetime
from pprint import pprint
import argparse

from typing import Union, List, Dict, Any, cast

TEMP_DIR = Path(os.environ["CAMERA_TEMP_DIR"])
DETECTOR_MODEL = "yolov8n"

# Avoid full memory depletion (limited to 90% of available memory)
_, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (int(
    psutil.virtual_memory()[1] * 0.95), hard))


def prepare_fs(name: str):
    annotations_folder = Path(TEMP_DIR, name)
    annotations_folder.mkdir(exist_ok=True, parents=True)

    return annotations_folder


def face_recon(path: Union[str, Path], show: bool = False, debug: bool = False):

    annot_path = prepare_fs(os.path.basename(path).split(".")[0])

    vidcap = cv.VideoCapture(path)

    width = int(vidcap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_count = 0

    detected_faces = []

    schema = pa.schema([
        ("face", pa.list_(pa.float64())),
        ("facial_area", pa.struct([
            ("x", pa.int64()),
            ("y", pa.int64()),
            ("w", pa.int64()),
            ("h", pa.int64()),
            ("left_eye", pa.list_(pa.int64(), 2)),
            ("right_eye", pa.list_(pa.int64(), 2))
        ])),
        ("confidence", pa.float64()),
        ("ts", pa.timestamp('us'))
    ])

    while vidcap.isOpened():
        success, image = vidcap.read()

        if success:
            frame_count += 1
            
            detected_faces = extract_faces(img_path=image,
                                            detector_backend=DETECTOR_MODEL,
                                            enforce_detection=False)  # Detect faces in frame
            
            # Include bounding box for every face detected in axis, if any
            if len(detected_faces) > 0:
                for i, face in enumerate(detected_faces):
                    if face["confidence"] > 0.9:
                        # Set dict as column-oriented data
                        face_co = face.copy()

                        face_co["face"] = [face_co["face"].flatten()]
                        face_co["facial_area"] = [face_co["facial_area"]]
                        face_co["confidence"] = [face_co["confidence"]]
                        face_co["ts"] = [datetime.now()]

                        uuid = f"{i}_{frame_count}"

                        annot_table = pa.Table.from_pydict(face_co, schema)
                        pq.write_table(annot_table, annot_path.joinpath(uuid + ".parquet"))

            if show or debug:
                for face in detected_faces:
                    if face["confidence"] > 0:
                        x, y, w, h, _, _ = face["facial_area"].values()

                        if debug:
                            print(f"Face detected in frame {frame_count}:")
                            print(f"Position: ({x}, {y}) | Size: {w}x{h} | Confidence: {face['confidence']:.0%}\n")

                        if show:
                            cv.rectangle(image, rec=(x, y, w, h),
                                            color=(0, 255, 0), thickness=2)

                if show:
                    cv.imshow('frame', image)

                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break

                detected_faces = []
        else:
            break

    vidcap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("path")
    arg_parser.add_argument("-s", "--show", action="store_true", default=False)
    arg_parser.add_argument("-d", "--debug", action="store_true", default=False)

    args = arg_parser.parse_args()

    if not os.path.exists(args.path):
        print("Path not found.")
        exit(1)

    face_recon(**vars(args))
