# predict.py
import numpy as np
import joblib

_lr_model  = joblib.load('models/logistic_model.pkl')
_lr_scaler = joblib.load('models/logistic_scaler.pkl')
_feat_cols = joblib.load('models/feature_cols.pkl')
_mlp       = joblib.load('models/lstm_model.pkl')
_norm      = np.load('models/lstm_norm.npy')

HR_BUFFER = []

def predict_status(hr, spo2, temp, ma_hr=None, ma_spo2=None, hr_diff=0):
    ma_hr   = ma_hr   if ma_hr   is not None else hr
    ma_spo2 = ma_spo2 if ma_spo2 is not None else spo2
    X = np.array([[hr, spo2, temp, ma_hr, ma_spo2, hr_diff]])
    X_scaled = _lr_scaler.transform(X)
    label = _lr_model.predict(X_scaled)[0]
    prob  = _lr_model.predict_proba(X_scaled)[0][1]
    return ('Critical' if label == 1 else 'Normal'), round(float(prob), 3)
# Add this function at the bottom of predict.py
def reset_buffer():
    global HR_BUFFER
    HR_BUFFER = []
    print("✅ HR buffer reset for new patient")
def predict_next_hr(current_hr):
    global HR_BUFFER
    HR_BUFFER.append(current_hr)
    if len(HR_BUFFER) > 30:
        HR_BUFFER.pop(0)
    if len(HR_BUFFER) < 30:
        return [round(current_hr, 1)] * 10

    hr_min, hr_max = _norm
    seq = np.array(HR_BUFFER)
    seq_norm = (seq - hr_min) / (hr_max - hr_min + 1e-8)
    pred_norm = _mlp.predict([seq_norm])[0]
    pred = pred_norm * (hr_max - hr_min) + hr_min
    return [round(float(v), 1) for v in pred]