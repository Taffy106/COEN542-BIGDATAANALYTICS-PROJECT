from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import StringType, FloatType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

spark = SparkSession.builder \
    .appName("Stage2-Sentiment") \
    .master("local[2]") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/socialmedia.cleaned_posts") \
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017/socialmedia.processed_posts") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

clean_df = spark.read.format("mongodb").load()
print("Cleaned records loaded:", clean_df.count())

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update({
    "meh": -1.0, "lol": 0.5, "smh": -0.5, "fml": -2.0, "yasss": 1.5,
})

def score_sentiment(text):
    if not text:
        return 0.0
    return analyzer.polarity_scores(text)["compound"]

def label_sentiment_binary(score):
    return "positive" if score >= 0 else "negative"

score_udf = udf(score_sentiment, FloatType())
label_udf = udf(label_sentiment_binary, StringType())

scored_df = clean_df.withColumn("sentiment_score", score_udf(col("text"))) \
                     .withColumn("sentiment_label", label_udf(col("sentiment_score")))

print("=== Sentiment label distribution ===")
scored_df.groupBy("sentiment_label").count().show()

# Accuracy check against Sentiment140's ground truth
eval_df = scored_df.withColumn(
    "true_label",
    when(col("label") == "0", "negative").when(col("label") == "4", "positive")
).filter(col("source") == "twitter") \
 .filter(col("true_label").isNotNull())

correct = eval_df.filter(col("sentiment_label") == col("true_label")).count()
total = eval_df.count()
print(f"Binary Accuracy: {correct/total:.3f}")

scored_df.write.format("mongodb") \
    .option("connection.uri", "mongodb://localhost:27017/socialmedia.processed_posts") \
    .mode("overwrite").save()

print("Stage 2 done -- wrote to socialmedia.processed_posts")
spark.stop()