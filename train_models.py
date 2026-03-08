# train_models.py
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from signal_processing import detect_anomalies, moving_average
from data_loader import load_csv

os.makedirs('models', exist_ok=True)

# ── Logistic Regression ──────────────────────────────────────
def train_logistic(df):
    print("\n── Training Logistic Regression ──")
    df = detect_anomalies(df)

    df['ma_hr']   = moving_average(df['heart_rate'], 5)
    df['ma_spo2'] = moving_average(df['spo2'], 5)
    df['hr_diff'] = df['heart_rate'].diff().fillna(0)

    feature_cols = ['heart_rate', 'spo2', 'temperature', 'ma_hr', 'ma_spo2', 'hr_diff']
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values
    y = df['anomaly'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Critical']))

    joblib.dump(model,        'models/logistic_model.pkl')
    joblib.dump(scaler,       'models/logistic_scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    print("✅ Logistic model saved")
    return model, scaler, feature_cols

# ── LSTM ─────────────────────────────────────────────────────
SEQ_LEN    = 30
PRED_STEPS = 10

def create_sequences(series, seq_len=SEQ_LEN, pred_steps=PRED_STEPS):
    X, y = [], []
    arr = series.values
    for i in range(len(arr) - seq_len - pred_steps):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len:i+seq_len+pred_steps])
    return np.array(X), np.array(y)

from sklearn.neural_network import MLPRegressor

SEQ_LEN    = 30
PRED_STEPS = 10

def train_lstm(df):
    print("\n── Training MLP Regressor (HR Prediction) ──")
    hr = df['heart_rate'].dropna().reset_index(drop=True).values

    hr_min, hr_max = hr.min(), hr.max()
    hr_norm = (hr - hr_min) / (hr_max - hr_min)

    X, y = [], []
    for i in range(len(hr_norm) - SEQ_LEN - PRED_STEPS):
        X.append(hr_norm[i:i+SEQ_LEN])
        y.append(hr_norm[i+SEQ_LEN:i+SEQ_LEN+PRED_STEPS])
    X, y = np.array(X), np.array(y)

    model = MLPRegressor(hidden_layer_sizes=(64, 32),
                         max_iter=200, random_state=42, verbose=True)
    model.fit(X, y)

    joblib.dump(model, 'models/lstm_model.pkl')
    np.save('models/lstm_norm.npy', [hr_min, hr_max])
    print("✅ MLP model saved")
    return model, hr_min, hr_max
if __name__ == '__main__':
    df = load_csv()
    train_logistic(df)
    train_lstm(df)
    print("\n✅ All models trained and saved!")