FROM python:3.12-trixie

RUN useradd -m annon

# Install OpenCV dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Install uv package manager
RUN pip install uv

COPY . .
RUN uv pip install -r requirements.txt --system

USER annon
RUN mkdir /home/annon/.deepface && mkdir /home/annon/.deepface/weights