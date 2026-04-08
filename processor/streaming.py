from schemas import *
from time import sleep
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.logger import PySparkLogger

import pandas as core_pd
import pyspark.pandas as pd
import numpy as np
import pyarrow as pa

from deepface.modules.datastore import search


from datetime import datetime
import logging

logger = PySparkLogger.getLogger("CameraProcessor")
logger.setLevel(logging.INFO)


def search_vector_db(df_it):
    search_kwargs = {
        "enforce_detection": False,
        "model_name": "SFace",
        "detector_backend": "skip",
        "k": 1,
        "database_type": "mongo",
    }

    cols = [
        "_id",
        "id",
        "img_name",
        "model_name",
        "search_method",
        "confidence",
        "distance_metric",
        "distance",
    ]

    for df in df_it:
        results = []

        for _, row in df.iterrows():
            try:
                img = np.reshape(row["face"], (row['h'], row['w'], 3))

                search_result = search(img, **search_kwargs)

                if search_result:
                    search_result = core_pd.concat(search_result)[cols].copy()

                    search_result["video"] = row["video"]
                    search_result["ts"] = row["ts"]

                    for col in search_result:
                        if col not in ["confidence", "distance", "ts"]:
                            search_result[col] = search_result[col].astype(str)

                    yield search_result
            except Exception as e:
                logger.error(e)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CameraProcessor").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    while not os.path.exists(os.environ["DATA_PATH"]):
        logger.info("Waiting for data...")
        sleep(1)

    data = (
        spark.readStream.schema(input_schema)
        .options(maxFilesPerTrigger=100)
        .parquet(os.environ["DATA_PATH"])
    )

    faces = data.withColumns({
        "h": 'facial_area.h',
        "w": 'facial_area.w'
    }).drop("facial_area")

    search_results = (
        faces
        .select("face", "h", "w", "video", "ts")
        .mapInPandas(search_vector_db, result_schema)
        .filter("confidence > 75")
    )

    query = (
        search_results
        .writeStream
        .outputMode("append")
        .format("console")
        .start()
    )

    logger.info("Streaming started, waiting for termination")
    query.awaitTermination()
