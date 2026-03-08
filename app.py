# app.py
import time, threading, collections
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

ALERT_LOG = collections.deque(maxlen=50)
df = None

def load_everything():
    global df
    print("Loading data...")
    from data_loader import load_csv
    from signal_processing import moving_average, detect_anomalies
    df = load_csv()
    df = detect_anomalies(df)
    df['ma_hr']   = moving_average(df['heart_rate'], 5).fillna(df['heart_rate'])
    df['ma_spo2'] = moving_average(df['spo2'], 5).fillna(df['spo2'])
    df['hr_diff'] = df['heart_rate'].diff().fillna(0)
    print("✅ Data loaded!")
    threading.Thread(target=stream_data, daemon=True).start()

def stream_data():
    from predict import predict_status, predict_next_hr, reset_buffer
    print("✅ Streaming started!")
    time.sleep(1)
    i = 0
    prev_hr = None
    last_patient = None

    while True:
        try:
            global current_patient_id

            # Filter by patient if one is selected
            if current_patient_id is not None:
                patient_df = df[df['patient_id'] == current_patient_id].reset_index(drop=True)
                if len(patient_df) == 0:
                    time.sleep(1)
                    continue
                # Reset buffer when switching patients
                if last_patient != current_patient_id:
                    reset_buffer()
                    last_patient = current_patient_id
                    i = 0
                    prev_hr = None
                row = patient_df.iloc[i % len(patient_df)]
            else:
                row = df.iloc[i % len(df)]

            hr   = float(row['heart_rate'])
            spo2 = float(row['spo2'])
            temp = float(row['temperature'])
            ma_hr   = float(row['ma_hr'])
            ma_spo2 = float(row['ma_spo2'])
            hr_diff = hr - prev_hr if prev_hr else 0
            prev_hr = hr

            status, crit_prob = predict_status(hr, spo2, temp, ma_hr, ma_spo2, hr_diff)
            next_hr = predict_next_hr(hr)

            payload = {
                'heart_rate':  round(hr, 1),
                'spo2':        round(spo2, 1),
                'temperature': round(temp, 2),
                'ma_hr':       round(ma_hr, 1),
                'next_hr':     next_hr,
                'status':      status,
                'crit_prob':   crit_prob,
                'anomaly':     bool(row['anomaly']),
                'reason':      str(row['anomaly_reason']).strip(),
                'timestamp':   str(row.get('timestamp', i)),
                'patient_id':  int(current_patient_id) if current_patient_id else 'ALL',
            }

            if status == 'Critical':
                ALERT_LOG.appendleft({
                    'time': payload['timestamp'],
                    'msg': f"HR:{hr} SpO₂:{spo2} Temp:{temp} | {payload['reason']}"
                })

            socketio.emit('vitals', payload)
            i += 1
            time.sleep(1)

        except Exception as e:
            print(f"Stream error: {e}")
            i += 1
            time.sleep(1)

current_patient_id = None

@app.route('/set_patient/<int:patient_id>')
def set_patient(patient_id):
    global current_patient_id
    current_patient_id = None if patient_id == 0 else patient_id
    return jsonify({'status': 'ok', 'patient_id': current_patient_id})

@app.route('/current_patient')
def current_patient():
    return jsonify({'patient_id': current_patient_id})
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/test')
def test():
    return jsonify({'status': 'ok', 'rows': len(df) if df is not None else 0})
@app.route('/alerts')
def alerts():
    return jsonify(list(ALERT_LOG))
@app.route('/sleep/<int:patient_id>')
def sleep_analysis(patient_id):
    from data_loader import load_csv
    from signal_processing import detect_sleep_windows, sleep_summary
    full_df = load_csv()
    if 'patient_id' not in full_df.columns:
        return jsonify({'error': 'No patient_id column'})
    patient_df = full_df[full_df['patient_id'] == patient_id].reset_index(drop=True)
    if len(patient_df) == 0:
        return jsonify({'error': 'Patient not found'})
    
    patient_df, acf = detect_sleep_windows(patient_df)
    summary = sleep_summary(patient_df)
    
    return jsonify({
        'patient_id': patient_id,
        'summary':    summary,
        'acf':        acf.tolist(),
        'hr':         patient_df['heart_rate'].tolist()[:300],
        'sleep_mask': patient_df['likely_sleep'].tolist()[:300],
        'record_count': len(patient_df)
    })

@app.route('/patient_stats')
def patient_stats():
    from data_loader import load_csv
    from signal_processing import detect_anomalies
    full_df = load_csv()
    if 'patient_id' not in full_df.columns:
        return jsonify([])
    full_df = detect_anomalies(full_df)
    stats = []
    for pid, grp in list(full_df.groupby('patient_id'))[:50]:
        stats.append({
            'id':           int(pid),
            'count':        len(grp),
            'avg_hr':       round(grp['heart_rate'].mean(), 1),
            'avg_spo2':     round(grp['spo2'].mean(), 1),
            'avg_temp':     round(grp['temperature'].mean(), 2),
            'anomaly_rate': round(grp['anomaly'].mean() * 100, 1)
        })
    return jsonify(stats)
@app.route('/patients')
def patients():
    from data_loader import get_patient_ids
    return jsonify(get_patient_ids()[:50].tolist() if hasattr(get_patient_ids()[:50], 'tolist') else get_patient_ids()[:50])
if __name__ == '__main__':
    threading.Thread(target=load_everything, daemon=True).start()
    print("Starting server at http://127.0.0.1:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)