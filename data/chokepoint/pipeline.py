import os
from shutil import rmtree

os.environ["CHOKEPOINT_DS_URL"] = "https://arma.sourceforge.net/chokepoint/"
os.environ["CHOKEPOINT_FILES_TO_DOWNLOAD"] = "P1E_S1.tar.xz, P1E.tar.xz"
os.environ["ORIGINAL_FILES_DIRNAME"] = "original_files"
os.environ["CROPPED_FACE_IMAGES_DIRNAME"] = "cropped_face_images"
os.environ["MONGODB_URI"] = "mongodb://root:root@localhost:27017/"
os.environ["DB_NAME"] = "test"

from processor import compose_chokepoint_videos, populate_db, BASE_PATH
from loader import extract_data, DOWNLOAD_PATH

from utils import clear_terminal



if __name__ == "__main__":
    clear_terminal()
    extract_data()
    clear_terminal()

    print("Processing original files")
    for path in DOWNLOAD_PATH.joinpath(os.environ["ORIGINAL_FILES_DIRNAME"]).iterdir():
        compose_chokepoint_videos(path)

    print("Processing cropped face images")
    for path in DOWNLOAD_PATH.joinpath(
        os.environ["CROPPED_FACE_IMAGES_DIRNAME"]
    ).iterdir():
        populate_db(path, ["P1E_S1"])

    clear_terminal()
    rmtree(BASE_PATH.joinpath("tmp"))
