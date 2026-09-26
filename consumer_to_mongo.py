from kafka import KafkaConsumer
from pymongo import MongoClient
import json

client = MongoClient("mongodb://localhost:27017/")
db = client["socialmedia"]
collection = db["raw_posts"]

consumer = KafkaConsumer(
    'social-posts',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

for msg in consumer:
    collection.insert_one(msg.value)
    print("Inserted:", msg.value.get("id"))