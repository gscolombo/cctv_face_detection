import cv2 as cv

from tqdm import tqdm
import os

INPUT_PATH=os.path.abspath(os.path.dirname(__file__))
OUTPUT_PATH=os.path.join(INPUT_PATH, "videos")

CODEC_FOURCC = cv.VideoWriter.fourcc(*'FFV1')
VIDEO_FORMAT = ".avi"

FRAME_WIDTH = 800
FRAME_HEIGHT = 600
FRAME_RATE = 30.0

def compose_chokepoint_videos():
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
        
    frame_folders: list[str] = [folder for folder in os.listdir(INPUT_PATH) 
                                if os.path.isdir(os.path.join(INPUT_PATH, folder)) and
                                folder != "videos"]

    for folder in frame_folders:
        output = cv.VideoWriter(filename=os.path.join(OUTPUT_PATH, f"{folder}{VIDEO_FORMAT}"), 
                                apiPreference=cv.CAP_FFMPEG,
                                fourcc=CODEC_FOURCC, 
                                fps=FRAME_RATE, 
                                frameSize=(FRAME_WIDTH, FRAME_HEIGHT))
        
        frames: list[str] = [frame for frame in os.listdir(os.path.join(INPUT_PATH, folder)) 
                             if frame.endswith(".jpg")]
        frames.sort()
        
        for frame in tqdm(frames, desc=f"Composing video {folder}: ", unit="frame"):
            frame_path = os.path.join(INPUT_PATH, folder, frame)
            output.write(cv.imread(frame_path))
        
        output.release()
        
if __name__ == "__main__":
    compose_chokepoint_videos()