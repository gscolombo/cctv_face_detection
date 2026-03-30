from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.logger import PySparkLogger

import pyspark.pandas as pd
import numpy as np

from deepface.modules.representation import represent

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


def search_embedding(batch_df: DataFrame, batch_id: int):
    # Create embeddings
    represent_kwargs = {
        "enforce_detection": False,
        "model_name": "SFace",
        "detector_backend": "skip",
        "max_faces": 1,
    }

    df = batch_df.toPandas()

    df["embeddings"] = (
        df["face_image"]
        .map(np.stack)
        .apply(represent, **represent_kwargs)
        .map(lambda d: d[0]["embedding"])
    )

    print(df)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CameraProcessor").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    while not os.path.exists(os.environ["DATA_PATH"]):
        logger.info("Waiting for data...")
        sleep(1)

    data = (
        spark.readStream.schema(schema)
        .options(maxFilesPerTrigger=50)
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
        .foreachBatch(search_embedding)
        .start()
    )

    logger.info("Streaming started, waiting for termination")
    query.awaitTermination()
