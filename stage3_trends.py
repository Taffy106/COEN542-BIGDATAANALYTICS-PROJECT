from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, to_timestamp, explode, split, window, count

spark = SparkSession.builder \
    .appName("Stage3-Trends") \
    .master("local[2]") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/socialmedia.processed_posts") \
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017/socialmedia.trends") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

scored_df = spark.read.format("mongodb").load()
print("Processed records loaded:", scored_df.count())

date_stripped_df = scored_df.withColumn(
    "date_clean", regexp_replace(col("date"), r"\s(PDT|PST|EDT|EST|CDT|CST|MDT|MST|UTC|GMT)\s", " ")
).withColumn(
    "date_ts", to_timestamp(col("date_clean"), "EEE MMM dd HH:mm:ss yyyy")
)

hashtag_df = date_stripped_df.withColumn(
    "hashtags", split(col("text_clean"), " ")
).select("date_ts", explode(col("hashtags")).alias("token")) \
 .filter(col("token").startswith("#")) \
 .filter(col("date_ts").isNotNull())

trend_df = hashtag_df.groupBy(
    window(col("date_ts"), "1 day"), col("token")
).agg(count("*").alias("mentions")) \
 .orderBy(col("mentions").desc())

trend_count = trend_df.count()
print("Trend row count:", trend_count)
trend_df.show(20, truncate=False)

if trend_count > 0:
    trend_df.write.format("mongodb") \
        .option("connection.uri", "mongodb://localhost:27017/socialmedia.trends") \
        .mode("overwrite").save()
    print("Stage 3 done -- wrote hashtag trends to socialmedia.trends")
else:
    stopwords = {"the","a","an","and","or","but","is","are","was","were","to","of","in",
                 "on","for","it","i","you","my","me","im","its","this","that","be","at",
                 "so","just","not","have","has","with","your","as","by"}
    keyword_df = date_stripped_df.withColumn(
        "words", split(col("text_clean"), " ")
    ).select("date_ts", explode(col("words")).alias("token")) \
     .filter(~col("token").startswith("#")) \
     .filter(~col("token").startswith("@")) \
     .filter(col("token") != "") \
     .filter(~col("token").isin(list(stopwords))) \
     .filter(col("date_ts").isNotNull())

    keyword_trend_df = keyword_df.groupBy(
        window(col("date_ts"), "1 day"), col("token")
    ).agg(count("*").alias("mentions")) \
     .orderBy(col("mentions").desc())

    keyword_trend_df.write.format("mongodb") \
        .option("connection.uri", "mongodb://localhost:27017/socialmedia.trends") \
        .mode("overwrite").save()
    print("Stage 3 done -- hashtags too sparse, wrote keyword trends instead")

spark.stop()