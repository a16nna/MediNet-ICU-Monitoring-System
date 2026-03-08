# data_loader.py
import pandas as pd
import numpy as np

def load_eicu(path='data/vitalPeriodic.csv.gz'):
    print("Loading eICU — all patients...")
    df = pd.read_csv(path, compression='gzip')

    df = df.rename(columns={
        'heartrate':         'heart_rate',
        'sao2':              'spo2',
        'temperature':       'temperature',
        'observationoffset': 'timestamp',
        'patientunitstayid': 'patient_id'   # ← keep patient ID
    })

    cols = ['patient_id', 'timestamp', 'heart_rate', 'spo2', 'temperature']
    df = df[cols].dropna()

    df = df[
        df['heart_rate'].between(30, 200) &
        df['spo2'].between(70, 100) &
        df['temperature'].between(34, 42)
    ]

    df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)
    df.to_csv('data/icu_data.csv', index=False)
    n_patients = df['patient_id'].nunique()
    print(f"✅ Saved {len(df)} rows from {n_patients} patients")
    return df

def load_csv():
    df = pd.read_csv('data/icu_data.csv')
    return df.reset_index(drop=True)

def get_patient_ids():
    df = pd.read_csv('data/icu_data.csv', usecols=['patient_id'])
    return df['patient_id'].unique().tolist()

def load_patient(patient_id):
    df = load_csv()
    return df[df['patient_id'] == patient_id].reset_index(drop=True)

if __name__ == '__main__':
    load_eicu()