from pyspark.sql.types import *

input_schema = StructType(
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

result_schema = StructType([
    StructField("_id", StringType()),
    StructField("id", StringType()),
    StructField("img_name", StringType()),
    StructField("model_name", StringType()),
    StructField("search_method", StringType()),
    StructField("confidence", DoubleType()),
    StructField("distance_metric", StringType()),
    StructField("distance", DoubleType()),
    StructField("video", StringType()),
    StructField("ts", TimestampType()),
])