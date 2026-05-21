import pandas as pd
import random
from datetime import datetime, timedelta

rows = []

start = datetime(2026, 5, 21, 19, 0, 0)

for i in range(500):
    patient_id = random.randint(151179, 151250)

    if random.random() < 0.15:
        hr = random.randint(95, 120)
        spo2 = random.randint(85, 92)
        temp = round(random.uniform(38.0, 39.5), 1)
    else:
        hr = random.randint(68, 90)
        spo2 = random.randint(95, 99)
        temp = round(random.uniform(36.5, 37.5), 1)

    timestamp = start + timedelta(seconds=i)

    rows.append([
        patient_id,
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        hr,
        spo2,
        temp
    ])

df = pd.DataFrame(rows, columns=[
    "patient_id",
    "timestamp",
    "heart_rate",
    "spo2",
    "temperature"
])

df.to_csv("patient_data.csv", index=False)

print("patient_data.csv generated successfully!")
