import json, time, csv
from kafka import KafkaProducer

print("Connecting to Kafka...")
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
print("Connected. Opening CSV...")

try:
    with open('data/raw/Sentiment140.csv', encoding='latin-1') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            record = {
                "source": "twitter", "label": row[0], "id": row[1],
                "date": row[2], "user": row[4], "text": row[5]
            }
            producer.send('social-posts', value=record)
            print("Sent:", record["id"])
            time.sleep(0.1)
    producer.flush()
    print("Done.")
except Exception as e:
    print("ERROR:", e)