# generate_graphs.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from data_loader import load_csv
from signal_processing import (moving_average, autocorrelation,
                                detect_anomalies, temp_variation, oxygen_trend)

os.makedirs('static/graphs', exist_ok=True)

STYLE = {
    'bg':    '#0a0f1e',
    'grid':  '#1e3a5f',
    'hr':    '#00e5ff',
    'ma':    '#ffd700',
    'anomaly': '#ff1744',
    'text':  '#ccddee',
}

def style_ax(ax, title):
    ax.set_facecolor(STYLE['bg'])
    ax.tick_params(colors=STYLE['text'])
    ax.xaxis.label.set_color(STYLE['text'])
    ax.yaxis.label.set_color(STYLE['text'])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE['grid'])
    ax.grid(color=STYLE['grid'], linestyle='--', alpha=0.5)
    ax.set_title(title, color=STYLE['text'], fontsize=12, pad=10)

def plot_hr_trend(df):
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=STYLE['bg'])
    s = df.head(300)
    ax.plot(s.index, s['heart_rate'], color=STYLE['hr'], linewidth=1.2, label='Heart Rate')
    ma = moving_average(s['heart_rate'], 10)
    ax.plot(s.index, ma, color=STYLE['ma'], linewidth=2, linestyle='--', label='Moving Avg (10)')
    style_ax(ax, 'Heart Rate Time-Series with Moving Average')
    ax.set_ylabel('BPM')
    ax.legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    plt.tight_layout()
    plt.savefig('static/graphs/hr_trend.png', dpi=150, facecolor=STYLE['bg'])
    plt.close()
    print("✅ hr_trend.png")

def plot_anomalies(df):
    df = detect_anomalies(df)
    s = df.head(300).copy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=STYLE['bg'])
    vitals = [('heart_rate','BPM'), ('spo2','%'), ('temperature','°C')]
    colors = [STYLE['hr'], '#00ff88', '#ff9800']
    for ax, (col, unit), color in zip(axes, vitals, colors):
        ax.plot(s.index, s[col], color=color, linewidth=1.2, label=col)
        anom = s[s['anomaly']]
        ax.scatter(anom.index, anom[col], color=STYLE['anomaly'], s=30, zorder=5, label='Anomaly')
        style_ax(ax, f'{col.replace("_"," ").title()} — Anomaly Detection')
        ax.set_ylabel(unit)
        ax.legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    plt.tight_layout()
    plt.savefig('static/graphs/anomaly_detection.png', dpi=150, facecolor=STYLE['bg'])
    plt.close()
    print("✅ anomaly_detection.png")

def plot_autocorrelation(df):
    acf = autocorrelation(df['heart_rate'].head(500), max_lag=60)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=STYLE['bg'])
    ax.bar(range(len(acf)), acf, color=STYLE['hr'], alpha=0.7)
    ax.axhline(0,    color=STYLE['text'], linewidth=0.8)
    ax.axhline(0.2,  color=STYLE['ma'], linestyle='--', linewidth=1, label='Significance ±0.2')
    ax.axhline(-0.2, color=STYLE['ma'], linestyle='--', linewidth=1)
    style_ax(ax, 'Autocorrelation of Heart Rate (Sleep Pattern Detection)')
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    plt.tight_layout()
    plt.savefig('static/graphs/autocorrelation.png', dpi=150, facecolor=STYLE['bg'])
    plt.close()
    print("✅ autocorrelation.png")

def plot_temp_variation(df):
    s = df.head(300)
    tvar = temp_variation(s['temperature'], 10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), facecolor=STYLE['bg'])
    ax1.plot(s.index, s['temperature'], color='#ff9800', linewidth=1.2)
    style_ax(ax1, 'Temperature Over Time')
    ax1.set_ylabel('°C')
    ax2.fill_between(s.index, tvar, alpha=0.6, color='#ff9800')
    style_ax(ax2, 'Temperature Variation (Rolling Std Dev)')
    ax2.set_ylabel('Std Dev')
    plt.tight_layout()
    plt.savefig('static/graphs/temp_variation.png', dpi=150, facecolor=STYLE['bg'])
    plt.close()
    print("✅ temp_variation.png")

def plot_oxygen_trend(df):
    s = df.head(300)
    trend = oxygen_trend(s['spo2'], 10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), facecolor=STYLE['bg'])
    ax1.plot(s.index, s['spo2'], color='#00ff88', linewidth=1.2)
    style_ax(ax1, 'SpO₂ Over Time')
    ax1.set_ylabel('%')
    ax2.fill_between(s.index, trend, where=(trend >= 0), alpha=0.5, color='#00ff88', label='Rising')
    ax2.fill_between(s.index, trend, where=(trend < 0),  alpha=0.5, color=STYLE['anomaly'], label='Falling')
    style_ax(ax2, 'Oxygen Trend (Slope per 10 readings)')
    ax2.legend(facecolor=STYLE['bg'], labelcolor=STYLE['text'])
    plt.tight_layout()
    plt.savefig('static/graphs/oxygen_trend.png', dpi=150, facecolor=STYLE['bg'])
    plt.close()
    print("✅ oxygen_trend.png")

if __name__ == '__main__':
    df = load_csv()
    plot_hr_trend(df)
    plot_anomalies(df)
    plot_autocorrelation(df)
    plot_temp_variation(df)
    plot_oxygen_trend(df)
    print("\n✅ All graphs saved to static/graphs/")