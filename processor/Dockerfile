FROM spark:4.0.1-python3

USER root

COPY . .

# Install OpenCV dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install -r requirements.txt
RUN mkdir /home/spark && mkdir /home/spark/.deepface && mkdir /home/spark/.deepface/weights && chown -R spark /home/spark

ENV HOME /home/spark

USER spark

ENTRYPOINT /opt/spark/bin/spark-submit \
 --master local[*] \
 --driver-memory 4g \
 --executor-memory 4g \
  streaming.py