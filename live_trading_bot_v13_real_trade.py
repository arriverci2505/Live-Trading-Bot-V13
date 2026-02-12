"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER BOT v13 - LIVE TRADING DASHBOARD (STREAMLIT + PYTORCH)        ║
║  VERSION: v13.2 TITAN - FIX KRAKEN INVALID ARGUMENTS                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import streamlit.components.v1 as components
import joblib
import time
from datetime import datetime, timedelta
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import norm
import warnings
import logging

# Cấu hình log
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

LIVE_CONFIG = {
    'exchange': 'kraken',           
    'symbol': 'BTC/USDT',            # CCXT sẽ tự chuyển đổi cho Kraken
    'timeframe': '15m',
    'limit': 720,                    # Tăng limit để đủ dữ liệu cho các chỉ báo dài hạn
    'model_path': './BTC-USDT_MONSTER_model.pt',
    'sequence_length': 60,
    'temperature': 0.7,              
    'entry_percentile': 85.0,        
    'rolling_window': 200,           
    'rolling_min_periods': 50,
    
    'config': {
        'input_dim': 29,         
        'hidden_dim': 256,       
        'num_lstm_layers': 2,
        'num_transformer_layers': 2,
        'num_heads': 4,
        'se_reduction_ratio': 16,
        'dropout': 0.35,
        'num_classes': 3,
        'use_positional_encoding': True,
    },
    'adx_threshold_trending': 25,
    'adx_threshold_ranging': 20,
    'atr_multiplier_sl': 4.0,        
    'atr_multiplier_tp': 20.0,       
    'leverage': 5,
    'risk_per_trade': 0.02,          
    'refresh_interval': 60,          
}

# ════════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE (GIỮ NGUYÊN TOÀN BỘ)
# ════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        batch_size, seq_len, channels = x.size()
        squeeze = x.mean(dim=1)
        excitation = torch.sigmoid(self.fc2(torch.relu(self.fc1(squeeze))))
        return x * excitation.unsqueeze(1)

class HybridTransformerLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_classes = config['num_classes']
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=config['num_heads'],
            dim_feedforward=self.hidden_dim * 4, dropout=config['dropout'], batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_transformer_layers'])
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim,
                            num_layers=config['num_lstm_layers'], batch_first=True, bidirectional=True)
        self.se_block = SEBlock(self.hidden_dim * 2, config['se_reduction_ratio'])
        self.final_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim * 2, num_heads=config['num_heads'], batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x, _ = self.lstm(x)
        x = self.se_block(x)
        attn_output, _ = self.final_attention(x, x, x)
        x = attn_output[:, -1, :]
        return self.classifier(x)

# ════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING (GIỮ NGUYÊN TOÀN BỘ LOGIC CỦA BẠN)
# ════════════════════════════════════════════════════════════════════════════

def calculate_fractional_diff(series: pd.Series, d: float = 0.5, threshold: float = 0.01) -> pd.Series:
    x = series.dropna().values
    n = len(x)
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold: break
        weights.append(weight)
        k += 1
    weights = np.array(weights)
    width = len(weights)
    output = np.full(n, np.nan)
    for i in range(width - 1, n):
        output[i] = np.dot(weights[::-1], x[i - width + 1:i + 1])
    return pd.Series(output, index=series.dropna().index).reindex(series.index)

def calculate_fourier_features(series: pd.Series, n_components: int = 5) -> pd.DataFrame:
    valid_data = series.dropna()
    if len(valid_data) < 20: return pd.DataFrame(index=series.index)
    detrended = scipy_signal.detrend(valid_data.values)
    fft_vals = np.fft.fft(detrended)
    fft_freq = np.fft.fftfreq(len(detrended))
    pos_mask = fft_freq > 0
    magnitudes = np.abs(fft_vals[pos_mask])
    top_indices = np.argsort(magnitudes)[-n_components:]
    features = {}
    t = np.arange(len(series))
    for i, idx in enumerate(top_indices):
        freq = fft_freq[pos_mask][idx]
        features[f'fourier_sin_{i+1}'] = np.sin(2 * np.pi * freq * t)
        features[f'fourier_cos_{i+1}'] = np.cos(2 * np.pi * freq * t)
    return pd.DataFrame(features, index=series.index)

def enrich_features_v13(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # BB
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_width'] = (4 * std20) / sma20
    df['BB_position'] = (df['Close'] - (sma20 - 2*std20)) / (4 * std20)
    
    # Advanced
    df['frac_diff_close'] = calculate_fractional_diff(df['Close'], d=0.5)
    fourier = calculate_fourier_features(df['Close'], 5)
    df = pd.concat([df, fourier], axis=1)
    
    # Imbalance & Entropy
    price_range = (df['High'] - df['Low']).replace(0, 1e-9)
    df['volume_imbalance'] = ((df['Close'] - df['Open']) / price_range) * df['Volume']
    
    def _ent(x):
        p = np.histogram(x, bins=10)[0]
        p = p[p>0] / p.sum()
        return -np.sum(p * np.log2(p))
    df['entropy'] = df['Close'].rolling(14).apply(_ent, raw=True)
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # ADX & Regime
    plus_dm = np.where((df['High'].diff() > df['Low'].shift(1)-df['Low']), np.maximum(df['High'].diff(), 0), 0)
    minus_dm = np.where((df['Low'].shift(1)-df['Low'] > df['High'].diff()), np.maximum(df['Low'].shift(1)-df['Low'], 0), 0)
    pdi = 100 * (pd.Series(plus_dm).rolling(14).mean() / df['ATR'])
    mdi = 100 * (pd.Series(minus_dm).rolling(14).mean() / df['ATR'])
    df['ADX'] = (100 * abs(pdi-mdi)/(pdi+mdi)).rolling(14).mean()
    
    df['SMA_distance'] = (df['Close'].rolling(20).mean() - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
    df['regime_trending'] = (df['ADX'] > 25).astype(int)
    df['regime_uptrend'] = ((df['SMA_distance'] > 0) & (df['regime_trending'] == 1)).astype(int)
    df['regime_downtrend'] = ((df['SMA_distance'] < 0) & (df['regime_trending'] == 1)).astype(int)
    
    # Indicators
    delta = df['Close'].diff()
    u = (delta.where(delta > 0, 0)).rolling(14).mean()
    d = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + u/d))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    vol = df['Close'].pct_change().rolling(20).std()
    df['volatility_zscore'] = (vol - vol.rolling(100).mean()) / vol.rolling(100).std()
    df['RSI_vol_adj'] = df['RSI'] / (vol * 100)
    df['ROC_vol_adj'] = (df['Close'].pct_change(10) * 100) / (vol * 100)
    
    return df.ffill().fillna(0)

def apply_rolling_normalization(df, feature_cols):
    df = df.copy()
    for col in feature_cols:
        m = df[col].rolling(200, min_periods=50).mean()
        s = df[col].rolling(200, min_periods=50).std().replace(0, 1e-9)
        df[col] = ((df[col] - m) / s).clip(-5, 5)
    return df.fillna(0)

# ════════════════════════════════════════════════════════════════════════════
# 4. TRADING LOGIC
# ════════════════════════════════════════════════════════════════════════════

def get_signal_info(probs, price, atr, config):
    idx = np.argmax(probs)
    conf = probs[idx]
    signal = "NEUTRAL"
    if idx == 1 and conf > 0.5: signal = "BUY"
    if idx == 2 and conf > 0.5: signal = "SELL"
    
    sl_dist = atr * config['atr_multiplier_sl']
    tp_dist = atr * config['atr_multiplier_tp']
    tp = price + tp_dist if signal == "BUY" else price - tp_dist if signal == "SELL" else price
    sl = price - sl_dist if signal == "BUY" else price + sl_dist if signal == "SELL" else price
    
    return signal, conf, tp, sl

# ════════════════════════════════════════════════════════════════════════════
# 5. STREAMLIT UI & MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="MONSTER v13.2", layout="wide")

@st.cache_resource
def init_exchange(exchange_name):
    # Fix cho Kraken: CCXT xử lý tên cặp tiền tự động
    return getattr(ccxt, exchange_name)({'enableRateLimit': True})

@st.cache_resource
def load_assets(path, config):
    ckpt = torch.load(path, map_location='cpu')
    model = HybridTransformerLSTM(ckpt.get('config', config['config']))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    # Đảm bảo đúng 29 features
    features = ckpt.get('feature_cols', [
        'log_return', 'ATR', 'BB_width', 'BB_position', 'frac_diff_close',
        'fourier_sin_1', 'fourier_sin_2', 'fourier_sin_3', 'fourier_sin_4', 'fourier_sin_5',
        'fourier_cos_1', 'fourier_cos_2', 'fourier_cos_3', 'fourier_cos_4', 'fourier_cos_5',
        'volume_imbalance', 'entropy', 'volume_ratio', 'ADX', 'SMA_distance',
        'regime_trending', 'regime_uptrend', 'regime_downtrend', 'RSI', 'MACD',
        'MACD_signal', 'volatility_zscore', 'RSI_vol_adj', 'ROC_vol_adj'
    ])
    return model, features

def main():
    st.title("🚀 MONSTER BOT v13.2 TITAN")
    
    exchange = init_exchange(LIVE_CONFIG['exchange'])
    try:
        model, feature_cols = load_assets(LIVE_CONFIG['model_path'], LIVE_CONFIG)
    except Exception as e:
        st.error(f"Lỗi load model: {e}"); st.stop()

    # Dashboard Layout
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Chart")
        components.html(f'<div style="height:500px;"><div id="tv" style="height:100%;"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"autosize":true,"symbol":"KRAKEN:BTCUSDT","interval":"15","theme":"dark","container_id":"tv"}});</script></div>', height=520)
    
    with col2:
        st.subheader("AI Prediction")
        placeholder = st.empty()

    while True:
        try:
            # Fetch Data - Sửa lỗi Invalid Arguments
            # Kraken yêu cầu limit tối đa khoảng 720 cho OHLCV
            ohlcv = exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], LIVE_CONFIG['timeframe'], limit=LIVE_CONFIG['limit'])
            df = pd.DataFrame(ohlcv, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            
            # Feature Engineering
            df_enriched = enrich_features_v13(df)
            df_norm = apply_rolling_normalization(df_enriched, feature_cols)
            
            # Prediction
            X = torch.FloatTensor(df_norm[feature_cols].tail(60).values).unsqueeze(0)
            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits / LIVE_CONFIG['temperature'], dim=-1).numpy()[0]
            
            # Info
            last_price = df['Close'].iloc[-1]
            last_atr = df_enriched['ATR'].iloc[-1]
            sig, conf, tp, sl = get_signal_info(probs, last_price, last_atr, LIVE_CONFIG)
            
            # Update UI
            with placeholder.container():
                color = "#00ff88" if sig=="BUY" else "#ff4b4b" if sig=="SELL" else "#888"
                st.markdown(f"""
                    <div style="background:{color}22; padding:20px; border:2px solid {color}; border-radius:10px; text-align:center;">
                        <h1 style="color:{color}; margin:0;">{sig}</h1>
                        <h2 style="margin:0;">{conf:.1%} Confidence</h2>
                        <hr>
                        <p>Price: <b>${last_price:,.2f}</b></p>
                        <p>TP: <span style="color:#00ff88;">${tp:,.2f}</span> | SL: <span style="color:#ff4b4b;">${sl:,.2f}</span></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("---")
                st.write(f"ADX: {df_enriched['ADX'].iloc[-1]:.1f}")
                st.progress(float(probs[1]), "Buy Probability")
                st.progress(float(probs[2]), "Sell Probability")

            time.sleep(LIVE_CONFIG['refresh_interval'])
            
        except Exception as e:
            st.error(f"Main Loop Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
