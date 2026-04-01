from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.logger import PySparkLogger

import pyspark.pandas as pd
import numpy as np

from deepface.modules.representation import represent
from deepface.modules.datastore import search


import logging

logger = PySparkLogger.getLogger("CameraProcessor")
logger.setLevel(logging.INFO)

import os
from time import sleep

schema = StructType(
    [
        StructField("face", ArrayType(DoubleType())),
        StructField(
            "facial_area",
            StructType(
                [
                    StructField("x", LongType()),
                    StructField("y", LongType()),
                    StructField("w", LongType()),
                    StructField("h", LongType()),
                    StructField("left_eye", ArrayType(LongType())),
                    StructField("right_eye", ArrayType(LongType())),
                ]
            ),
        ),
        StructField("confidence", DoubleType()),
        StructField("ts", TimestampType()),
    ]
)


def _search(data: np.ndarray, **kwargs):
    img, video, ts = data
    result = search(np.stack(img), **kwargs)

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

    if result:
        result = result[0][cols]
        result["video"] = video
        result["ts"] = ts

        return pd.DataFrame(result[result["confidence"] >= 75.0])


def search_vector_db(batch_df: DataFrame, batch_id: int):
    # Create embeddings
    search_kwargs = {
        "enforce_detection": False,
        "model_name": "SFace",
        "detector_backend": "skip",
        "k": 3,
        "database_type": "mongo",
    }

    df: pd.DataFrame = batch_df.toPandas()

    search_results = (
        df[["face_image", "video", "ts"]]
        .apply(_search, axis=1, **search_kwargs)
        .to_numpy()
    )

    full_search_results = pd.concat(search_results).reset_index(drop=True)

    print(full_search_results)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CameraProcessor").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.ansi.enabled", False)

    while not os.path.exists(os.environ["DATA_PATH"]):
        logger.info("Waiting for data...")
        sleep(1)

    data = (
        spark.readStream.schema(schema)
        .options(maxFilesPerTrigger=10)
        .parquet(os.environ["DATA_PATH"])
    )

    # Restore image pixel matrix
    faces = data.withColumn(
        "face_image",
        F.expr(
            """transform(sequence(0, facial_area.h - 1), i -> slice(
                    transform(sequence(0, facial_area.w * facial_area.h - 1), j -> slice(face, j*3 + 1, 3)),
                    i*facial_area.w + 1, facial_area.w
                )
            )
            """
        ),
    ).drop("face", "facial_area")

    query = (
        faces.writeStream.outputMode("append")
        .format("console")
        .foreachBatch(search_vector_db)
        .start()
    )

    logger.info("Streaming started, waiting for termination")
    query.awaitTermination()
