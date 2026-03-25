FROM spark:4.0.1-python3

COPY . .

ENTRYPOINT /opt/spark/bin/spark-submit \
 --master local[*] \
 --driver-memory 4g \
 --executor-memory 4g \
  streaming.py