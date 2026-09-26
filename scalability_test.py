import time, csv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def score_sentiment(text):
    if not text:
        return 0.0
    return analyzer.polarity_scores(text)["compound"]

# Adjust this list to match your actual core count (check with `nproc` first).
# Keep it short -- each entry restarts a whole Spark session, which is the
# slowest part of this test on a laptop.
CORE_COUNTS = [1, 2]          # e.g. change to [1, 2, 4] only if nproc shows 4+
FRACTIONS = [0.25, 1.0]       # dropped 0.5 -- 25% and 100% already show the trend

results = []

for cores in CORE_COUNTS:
    spark = SparkSession.builder \
        .appName(f"ScalabilityTest-{cores}cores") \
        .master(f"local[{cores}]") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
        .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/socialmedia.cleaned_posts") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    score_udf = udf(score_sentiment, FloatType())
    # Read from cleaned_posts (already deduped/cleaned by stage1) instead of
    # raw_posts -- smaller, and avoids redoing the cleaning step every time.
    df = spark.read.format("mongodb").load().na.drop(subset=["text"])

    for fraction in FRACTIONS:
        sample_df = df.sample(withReplacement=False, fraction=fraction, seed=42)
        start = time.time()
        processed_count = sample_df.withColumn("sentiment_score", score_udf(col("text"))).count()
        elapsed = time.time() - start
        throughput = processed_count / elapsed if elapsed > 0 else 0
        row = {
            "cores": cores,
            "fraction": fraction,
            "records": processed_count,
            "seconds": round(elapsed, 2),
            "throughput": round(throughput, 2),
        }
        results.append(row)
        print(row)

    spark.stop()  # release memory fully before the next core-count session

with open("scalability_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Saved scalability_results.csv (one file, covers both data-size and core-count scaling)")