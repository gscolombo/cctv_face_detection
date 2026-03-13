from deepface.modules.streaming import grab_facial_areas
from deepface.modules.detection import extract_faces

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import cv2 as cv
from tqdm import tqdm

import os
import io
import resource
import psutil

CCTV_VIDEO_FOLDER = "./data/chokepoint/videos"
DETECTOR_MODEL="yolov11n"
OUTPUT_PATH="./data/results/"

CODEC_FOURCC = cv.VideoWriter.fourcc(*'FFV1')
VIDEO_FORMAT = ".avi"

# Avoid full memory depletion (limited to 90% of available memory)
_, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (int(psutil.virtual_memory()[1] * 0.9), hard))
    

def face_recon():
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    
    files = os.listdir(CCTV_VIDEO_FOLDER)

    if len(files) > 0:
        for file in files:
            vidcap = cv.VideoCapture(os.path.join(CCTV_VIDEO_FOLDER, file))

            fps = vidcap.get(cv.CAP_PROP_FPS)
            height = int(vidcap.get(cv.CAP_PROP_FRAME_HEIGHT))
            width = int(vidcap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_count = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
            file_path = os.path.join(OUTPUT_PATH, os.path.basename(file).split(".")[0] + "_out" + VIDEO_FORMAT)
            
            output = cv.VideoWriter(filename=file_path, 
                                    apiPreference=cv.CAP_FFMPEG,
                                    fourcc=CODEC_FOURCC, 
                                    fps=fps, 
                                    frameSize=(width, height))
                    
            
            clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(8,6))
            
            while vidcap.isOpened():
                success, image = vidcap.read()
                
                if success:
                    frame = cv.cvtColor(image, cv.COLOR_BGR2GRa)
                    
                    faces = grab_facial_areas(img=image,
                                              detector_backend=DETECTOR_MODEL,
                                              threshold=int(width * 0.01)) # Detect faces in frame
                    
                    # Include bounding box for every face detected in axis, if any
                    if len(faces) > 0:
                        for (x, y, w, h, _, _) in faces:
                            cv.rectangle(image, rec=(x, y, w, h), color=(0, 255, 0), thickness=2)
                    
                    cv.imshow('frame', image)  
                    
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
                
            output.release()
            vidcap.release()
            cv.destroyAllWindows()
            
if __name__ == "__main__":
    face_recon()