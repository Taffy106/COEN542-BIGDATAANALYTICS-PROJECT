from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim

spark = SparkSession.builder \
    .appName("Stage1-Clean") \
    .master("local[2]") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/socialmedia.raw_posts") \
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017/socialmedia.cleaned_posts") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

df = spark.read.format("mongodb").load()
print("Raw records:", df.count())

clean_df = df.withColumn("text_clean", lower(col("text"))) \
             .withColumn("text_clean", regexp_replace(col("text_clean"), r"http\S+", "")) \
             .withColumn("text_clean", regexp_replace(col("text_clean"), r"[^a-z0-9\s#@]", "")) \
             .withColumn("text_clean", trim(col("text_clean"))) \
             .dropDuplicates(["id"]) \
             .na.drop(subset=["text_clean"])

print("Cleaned records:", clean_df.count())
clean_df.show(5, truncate=80)

clean_df.write.format("mongodb") \
    .option("connection.uri", "mongodb://localhost:27017/socialmedia.cleaned_posts") \
    .mode("overwrite").save()

print("Stage 1 done -- wrote to socialmedia.cleaned_posts")
spark.stop()