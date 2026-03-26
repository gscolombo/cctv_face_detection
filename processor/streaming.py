from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.logger import PySparkLogger

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

if __name__ == "__main__":
    spark = SparkSession.builder.appName("CameraProcessor").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    while not os.path.exists(os.environ["DATA_PATH"]):
        logger.info("Waiting for data...")
        sleep(1)

    data = (
        spark.readStream.schema(schema)
        .options(maxFilesPerTrigger=100)
        .parquet(os.environ["DATA_PATH"])
    )

    faces = data.withColumn(
        "face_image",
        F.expr(
            "transform(sequence(0, facial_area.w * facial_area.h), i -> slice(face, i*3 + 1, 3))"
        ),
    ).drop("face", "facial_area")

    query = faces.writeStream.outputMode("append").format("console").start()

    logger.info("Streaming started, waiting for termination")
    query.awaitTermination()
