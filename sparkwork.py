from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("COEN542-BigDataPipeline") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/socialmedia.raw_posts") \
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017/socialmedia.processed_posts") \
    .getOrCreate()

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

df = spark.read.format("mongodb").load()
df.printSchema()
df.show(5)
print("Total records:", df.count())

from pyspark.sql.functions import col, lower, regexp_replace, trim, when

clean_df = df.withColumn("text_clean", lower(col("text"))) \
             .withColumn("text_clean", regexp_replace(col("text_clean"), r"http\S+", "")) \
             .withColumn("text_clean", regexp_replace(col("text_clean"), r"[^a-z0-9\s#@]", "")) \
             .withColumn("text_clean", trim(col("text_clean"))) \
             .dropDuplicates(["id"]) \
             .na.drop(subset=["text_clean"])

clean_df.show(5, truncate=80)

# ---------------- PHASE 7: Sentiment scoring (VADER, BINARY ONLY) ----------------
# PIVOT: dropped the 3-way (positive/negative/neutral) classification entirely.
# Sentiment140's ground truth only ever contains positive/negative -- there is
# no neutral class in the data at all. Scoring 3-way meant every "neutral" VADER
# call was automatically wrong against ground truth, which dragged accuracy down
# for reasons that had nothing to do with whether VADER's judgment was reasonable.
# Going strictly binary removes that mismatch and gives an apples-to-apples
# comparison against the dataset's actual labeling scheme.
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, FloatType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

# Score on the ORIGINAL "text" column, not "text_clean" -- text_clean strips
# punctuation/case, which removes signal VADER relies on (e.g. "!!!", "SO GOOD").
scored_df = clean_df.withColumn("sentiment_score", score_udf(col("text"))) \
                     .withColumn("sentiment_label", label_udf(col("sentiment_score")))

print("=== Sentiment label distribution (binary, full dataset) ===")
scored_df.groupBy("sentiment_label").count().show()

# Sentiment140: 0=negative, 4=positive
eval_df = scored_df.withColumn(
    "true_label",
    when(col("label") == "0", "negative").when(col("label") == "4", "positive")
).filter(col("source") == "twitter") \
 .filter(col("true_label").isNotNull())

correct = eval_df.filter(col("sentiment_label") == col("true_label")).count()
total = eval_df.count()
accuracy = correct / total
print(f"Binary Accuracy: {accuracy:.3f}")

from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

label_indexer = StringIndexer(inputCol="true_label", outputCol="true_label_idx", stringOrderType="alphabetAsc")
pred_indexer = StringIndexer(inputCol="sentiment_label", outputCol="pred_label_idx", stringOrderType="alphabetAsc")

indexed_df = label_indexer.fit(eval_df).transform(eval_df)
indexed_df = pred_indexer.fit(eval_df).transform(indexed_df)

evaluator = MulticlassClassificationEvaluator(
    labelCol="true_label_idx", predictionCol="pred_label_idx", metricName="f1"
)
f1_score = evaluator.evaluate(indexed_df)
print(f"Binary F1 Score: {f1_score:.3f}")

# Confidence-bucketed accuracy -- shows VADER is more reliable when it's more
# confident, a genuinely useful finding to report alongside the raw number.
from pyspark.sql.functions import abs as spark_abs

confidence_df = eval_df.withColumn(
    "confidence_bucket",
    when(spark_abs(col("sentiment_score")) >= 0.5, "high")
    .when(spark_abs(col("sentiment_score")) >= 0.2, "medium")
    .otherwise("low")
).withColumn("is_correct", (col("sentiment_label") == col("true_label")).cast("int"))

print("=== Accuracy by confidence bucket ===")
confidence_df.groupBy("confidence_bucket").agg({"is_correct": "avg", "*": "count"}).show()

eval_df.select("text", "sentiment_score", "sentiment_label", "true_label").show(10, truncate=60)

print(f"""
=== Summary ===
Binary Accuracy: {accuracy:.3f}
Binary F1:       {f1_score:.3f}

Go, Bhayani & Huang (2009) reported >80% accuracy using a SUPERVISED classifier
(Naive Bayes) trained directly on Sentiment140's own labels. VADER here is an
UNTRAINED, generic lexicon applied with no knowledge of this dataset's specific
labeling conventions -- a lower score is expected and is itself a valid finding,
not an error: it demonstrates the accuracy trade-off between a zero-training
lexicon approach and a supervised approach trained on the target data.
""")

# ---------------- PHASE 8: Trend analysis ----------------
from pyspark.sql.functions import explode, split, window, count, to_timestamp

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

print("=== Trend results ===")
trend_df.show(20, truncate=False)
print("Trend row count:", trend_df.count())

stopwords = {"the","a","an","and","or","but","is","are","was","were","to","of","in",
             "on","for","it","i","you","my","me","im","its","this","that","be","at",
             "so","just","not","have","has","with","its","your","of","as","by"}
stop_udf_list = list(stopwords)

keyword_df = date_stripped_df.withColumn(
    "words", split(col("text_clean"), " ")
).select("date_ts", explode(col("words")).alias("token")) \
 .filter(~col("token").startswith("#")) \
 .filter(~col("token").startswith("@")) \
 .filter(col("token") != "") \
 .filter(~col("token").isin(stop_udf_list)) \
 .filter(col("date_ts").isNotNull())

keyword_trend_df = keyword_df.groupBy(
    window(col("date_ts"), "1 day"), col("token")
).agg(count("*").alias("mentions")) \
 .orderBy(col("mentions").desc())

print("=== Keyword trend results ===")
keyword_trend_df.show(20, truncate=False)

# ---------------- Write results to MongoDB ----------------
scored_df.select("_id", "date", "id", "label", "source", "text", "user",
                  "sentiment_label", "sentiment_score") \
    .write.format("mongodb") \
    .option("connection.uri", "mongodb://localhost:27017/socialmedia.processed_posts") \
    .mode("overwrite").save()

if trend_df.count() > 0:
    trend_df.write.format("mongodb") \
        .option("connection.uri", "mongodb://localhost:27017/socialmedia.trends") \
        .mode("overwrite").save()
    print("Wrote hashtag trends to MongoDB.")
else:
    keyword_trend_df.write.format("mongodb") \
        .option("connection.uri", "mongodb://localhost:27017/socialmedia.trends") \
        .mode("overwrite").save()
    print("Hashtags were too sparse -- wrote keyword trends to MongoDB instead.")


import time, csv

results = []
for fraction in [0.25, 0.5, 1.0]:
    df_sample = df.sample(withReplacement=False, fraction=fraction, seed=42)
    start = time.time()
    processed = df_sample.withColumn("sentiment_score", score_udf(col("text"))).count()
    elapsed = time.time() - start
    throughput = processed / elapsed
    results.append({"fraction": fraction, "records": processed, "seconds": elapsed, "throughput": throughput})
    print(results[-1])

with open("scalability_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)



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

results = []

for cores in [1, 2, 4]:
    spark = SparkSession.builder \
        .appName(f"ScalabilityTest-{cores}cores") \
        .master(f"local[{cores}]") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
        .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/socialmedia.raw_posts") \
        .getOrCreate()

    score_udf = udf(score_sentiment, FloatType())
    df = spark.read.format("mongodb").load().na.drop(subset=["text"])

    for fraction in [0.25, 0.5, 1.0]:
        sample_df = df.sample(withReplacement=False, fraction=fraction, seed=42)
        start = time.time()
        processed_count = sample_df.withColumn("sentiment_score", score_udf(col("text"))).count()
        elapsed = time.time() - start
        throughput = processed_count / elapsed if elapsed > 0 else 0
        results.append({
            "cores": cores,
            "fraction": fraction,
            "records": processed_count,
            "seconds": round(elapsed, 2),
            "throughput": round(throughput, 2)
        })
        print(results[-1])

    spark.stop()  # important: fully stop before starting the next core-count session

with open("scalability_results_cores.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Saved scalability_results_cores.csv")