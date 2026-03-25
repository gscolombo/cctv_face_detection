from pyspark.sql import SparkSession
from pyspark.sql.types import *
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

    faces = (
        spark.readStream.schema(schema)
        .options(maxFilesPerTrigger=100)
        .parquet(os.environ["DATA_PATH"])
        .writeStream.outputMode("append")
        .format("console")
        .start()
    )

    logger.info("Streaming started, waiting for termination")
    faces.awaitTermination()