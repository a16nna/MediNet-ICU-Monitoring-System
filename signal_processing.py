# signal_processing.py
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=4.0, fs=10.0, order=3):
    nyq = fs / 2
    low, high = lowcut / nyq, highcut / nyq
    high = min(high, 0.99)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def moving_average(series, window=5):
    return series.rolling(window=window, center=True).mean()

def autocorrelation(series, max_lag=60):
    series = series.dropna()
    n = len(series)
    mean = series.mean()
    var = series.var()
    result = []
    for lag in range(max_lag):
        if n - lag <= 0:
            break
        cov = ((series[:n-lag].values - mean) * (series[lag:].values - mean)).mean()
        result.append(cov / var if var > 0 else 0)
    return np.array(result)

THRESHOLDS = {
    'heart_rate':  {'low': 40,  'high': 120},
    'spo2':        {'low': 90,  'high': 101},
    'temperature': {'low': 35,  'high': 39.5},
}

def detect_anomalies(df):
    df = df.copy()
    df['anomaly'] = False
    df['anomaly_reason'] = ''
    for col, bounds in THRESHOLDS.items():
        if col not in df.columns:
            continue
        mask = (df[col] < bounds['low']) | (df[col] > bounds['high'])
        df.loc[mask, 'anomaly'] = True
        df.loc[mask, 'anomaly_reason'] += col + ' '
    return df

def temp_variation(series, window=10):
    return series.rolling(window=window).std()

def oxygen_trend(series, window=10):
    trends = []
    for i in range(len(series)):
        if i < window:
            trends.append(0.0)
        else:
            y = series.iloc[i-window:i].values
            x = np.arange(window)
            slope = np.polyfit(x, y, 1)[0]
            trends.append(round(slope, 4))
    return pd.Series(trends, index=series.index)
def detect_sleep_windows(df, window=30):
    df = df.copy().reset_index(drop=True)
    hr = df['heart_rate']
    
    # Need at least window*2 rows for meaningful analysis
    if len(df) < window:
        df['likely_sleep'] = False
        acf = np.zeros(10)
        return df, acf

    roll_mean = hr.rolling(window=window, center=True).mean()
    roll_std  = hr.rolling(window=window, center=True).std()
    df['likely_sleep'] = (roll_mean < hr.mean()) & (roll_std < roll_std.mean())
    df['likely_sleep'] = df['likely_sleep'].fillna(False)
    
    max_lag = min(120, len(hr) // 2)
    acf = autocorrelation(hr, max_lag=max_lag)
    return df, acf

def sleep_summary(df):
    total = len(df)
    if total == 0 or 'likely_sleep' not in df.columns:
        return {'sleep_pct':0,'awake_pct':0,'avg_hr_sleep':0,'avg_hr_awake':0}
    
    sleep = df['likely_sleep'].sum()
    sleep_hr = df[df['likely_sleep']]['heart_rate']
    awake_hr = df[~df['likely_sleep']]['heart_rate']
    
    return {
        'sleep_pct':    round(float(sleep / total * 100), 1) if total > 0 else 0,
        'awake_pct':    round(float((total-sleep) / total * 100), 1) if total > 0 else 0,
        'avg_hr_sleep': round(float(sleep_hr.mean()), 1) if len(sleep_hr) > 0 else 0,
        'avg_hr_awake': round(float(awake_hr.mean()), 1) if len(awake_hr) > 0 else 0,
    }
if __name__ == '__main__':
    from data_loader import load_csv
    df = load_csv()
    df = detect_anomalies(df)
    print(f"Anomalies: {df['anomaly'].sum()} / {len(df)}")
    print(df[df['anomaly']].head(5))