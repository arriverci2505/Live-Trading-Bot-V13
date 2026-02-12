"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER BOT v13 - LIVE TRADING DASHBOARD (STREAMLIT + PYTORCH)        ║
║  VERSION: v13.1 TITAN - REAL TRADE READY                                ║
║                                                                          ║
║  Features:                                                              ║
║  ✅ PyTorch Model Loading (HybridTransformerLSTM)                       ║
║  ✅ Advanced Feature Engineering (Frac Diff, Fourier, Rolling Z-Score) ║
║  ✅ Temperature Scaling (T=0.7) + Adaptive Thresholding (P85)          ║
║  ✅ Market Regime Detection (ADX-based)                                ║
║  ✅ Dynamic TP/SL (ATR Adaptive: 4x SL, 20x TP)                        ║
║  ✅ Advanced Exit Manager Integration (Trailing Stop & Profit Lock)     ║
║  ✅ Real-time TradingView Chart                                        ║
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

# Cấu hình log và cảnh báo
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

LIVE_CONFIG = {
    # Exchange & Symbol
    'exchange': 'kraken',           
    'symbol': 'BTC/USDT',
    'timeframe': '15m',
    'limit': 500,                    # Đủ cho EMA 200 + rolling window
    
    # Model Path
    'model_path': './BTC-USDT_MONSTER_model.pt',
    
    # v13 Settings
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
    
    # Market Regime
    'adx_threshold_trending': 25,
    'adx_threshold_ranging': 20,
    
    # Risk Management (Advanced Exit Manager)
    'atr_multiplier_sl': 4.0,        
    'atr_multiplier_tp': 20.0,       
    'profit_lock_levels': [
        (1.2, 0.5), # Khi đạt 1.2% profit, khóa 0.5%
        (2.0, 1.0), # Khi đạt 2.0% profit, khóa 1.0%
        (3.0, 1.5), # Khi đạt 3.0% profit, khóa 1.5%
    ],
    
    # Trading
    'leverage': 5,
    'risk_per_trade': 0.02,          
    
    # UI Update
    'refresh_interval': 60,          
}

# ════════════════════════════════════════════════════════════════════════════
# 2. PYTORCH MODEL ARCHITECTURE
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
            d_model=self.hidden_dim,
            nhead=config['num_heads'],
            dim_feedforward=self.hidden_dim * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_transformer_layers'])
        
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=config['num_lstm_layers'],
            batch_first=True,
            bidirectional=True
        )
        
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
# 3. ADVANCED FEATURE ENGINEERING (v13)
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
    result = pd.Series(output, index=series.dropna().index)
    return result.reindex(series.index)

def calculate_fourier_features(series: pd.Series, n_components: int = 5) -> pd.DataFrame:
    detrended = scipy_signal.detrend(series.dropna().values)
    fft_vals = np.fft.fft(detrended)
    fft_freq = np.fft.fftfreq(len(detrended))
    positive_freqs = fft_freq > 0
    magnitudes = np.abs(fft_vals[positive_freqs])
    top_indices = np.argsort(magnitudes)[-n_components:]
    features = {}
    for i, idx in enumerate(top_indices):
        freq = fft_freq[positive_freqs][idx]
        t = np.arange(len(series))
        features[f'fourier_sin_{i+1}'] = np.sin(2 * np.pi * freq * t)
        features[f'fourier_cos_{i+1}'] = np.cos(2 * np.pi * freq * t)
    return pd.DataFrame(features, index=series.index)

def calculate_volume_order_imbalance(df: pd.DataFrame) -> pd.Series:
    price_range = (df['High'] - df['Low']).replace(0, np.nan)
    imbalance = ((df['Close'] - df['Open']) / price_range) * df['Volume']
    return imbalance.ffill().fillna(0)

def calculate_entropy(series: pd.Series, window: int = 14) -> pd.Series:
    def _entropy(x):
        x = x[~np.isnan(x)]
        if len(x) == 0: return 0
        counts = np.bincount((x * 100).astype(int) - (x * 100).astype(int).min())
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    return series.rolling(window).apply(_entropy, raw=True).ffill().fillna(0)

def enrich_features_v13(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    logger.info(f"🔧 Feature Engineering v13 (Input: {len(df)} rows)")
    
    # --- BASIC ---
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['BB_width'] = (4 * std_20) / sma_20
    df['BB_position'] = (df['Close'] - (sma_20 - 2*std_20)) / (4 * std_20)
    
    # --- ADVANCED ---
    df['frac_diff_close'] = calculate_fractional_diff(df['Close'], d=0.5)
    df = pd.concat([df, calculate_fourier_features(df['Close'], n_components=5)], axis=1)
    df['volume_imbalance'] = calculate_volume_order_imbalance(df)
    df['entropy'] = calculate_entropy(df['Close'], window=14)
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # --- REGIME (ADX) ---
    period = 14
    plus_dm = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    minus_dm = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    atr_smooth = df['ATR'].rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean().values / atr_smooth)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean().values / atr_smooth)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(period).mean()
    
    # SMA Distance
    df['SMA_distance'] = (df['Close'].rolling(20).mean() - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
    df['regime_trending'] = (df['ADX'] > 25).astype(int)
    df['regime_uptrend'] = ((df['SMA_distance'] > 0) & (df['regime_trending'] == 1)).astype(int)
    df['regime_downtrend'] = ((df['SMA_distance'] < 0) & (df['regime_trending'] == 1)).astype(int)
    
    # --- INDICATORS ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    volatility = df['Close'].pct_change().rolling(20).std()
    df['volatility_zscore'] = (volatility - volatility.rolling(100).mean()) / volatility.rolling(100).std()
    df['RSI_vol_adj'] = df['RSI'] / (volatility * 100)
    df['ROC_vol_adj'] = (df['Close'].pct_change(10) * 100) / (volatility * 100)
    
    # Xử lý dữ liệu trống
    df.fillna(0, inplace=True)
    return df

def apply_rolling_normalization(df: pd.DataFrame, feature_cols: list, config: dict) -> pd.DataFrame:
    df = df.copy()
    window, min_p = config.get('rolling_window', 200), config.get('rolling_min_periods', 50)
    for col in feature_cols:
        if col in df.columns:
            m = df[col].rolling(window=window, min_periods=min_p).mean()
            s = df[col].rolling(window=window, min_periods=min_p).std().replace(0, np.nan)
            df[col] = ((df[col] - m) / s).ffill().fillna(0).clip(-5, 5)
    return df

# ════════════════════════════════════════════════════════════════════════════
# 4. ADAPTIVE LOGIC & MARKET REGIME
# ════════════════════════════════════════════════════════════════════════════

def apply_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)

def detect_market_regime(adx: float, sma_distance: float, config: dict) -> dict:
    is_trending = adx > config.get('adx_threshold_trending', 25)
    trend_dir = "UP" if sma_distance > 0.005 else "DOWN" if sma_distance < -0.005 else "NEUTRAL"
    
    if is_trending:
        regime = f"TRENDING_{trend_dir}"
    else:
        regime = "RANGING" if adx < config.get('adx_threshold_ranging', 20) else "VOLATILE"
        
    return {'regime': regime, 'is_trending': is_trending, 'trend_direction': trend_dir, 'adx_value': adx}

def calculate_dynamic_tp_sl(current_price: float, atr: float, signal: str, config: dict) -> dict:
    sl_dist = atr * config.get('atr_multiplier_sl', 4.0)
    tp_dist = atr * config.get('atr_multiplier_tp', 20.0)
    
    if signal == "BUY":
        tp, sl = current_price + tp_dist, current_price - sl_dist
    elif signal == "SELL":
        tp, sl = current_price - tp_dist, current_price + sl_dist
    else:
        tp = sl = current_price
        
    return {'tp_price': tp, 'sl_price': sl, 'tp_distance': tp_dist, 'sl_distance': sl_dist, 
            'reward_risk': tp_dist/sl_dist if sl_dist > 0 else 0}

# ════════════════════════════════════════════════════════════════════════════
# 5. LOAD ASSETS & UI
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_and_assets(config):
    model_path = Path(config['model_path'])
    if not model_path.exists():
        st.error(f"❌ Model file not found at {model_path}"); st.stop()
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint.get('config', config['config'])
    
    # FALLBACK FEATURE COLS (Đúng 29 features cho v13)
    feature_cols = checkpoint.get('feature_cols', [
        'log_return', 'ATR', 'BB_width', 'BB_position', 'frac_diff_close',
        'fourier_sin_1', 'fourier_sin_2', 'fourier_sin_3', 'fourier_sin_4', 'fourier_sin_5',
        'fourier_cos_1', 'fourier_cos_2', 'fourier_cos_3', 'fourier_cos_4', 'fourier_cos_5',
        'volume_imbalance', 'entropy', 'volume_ratio', 'ADX', 'SMA_distance',
        'regime_trending', 'regime_uptrend', 'regime_downtrend', 'RSI', 'MACD',
        'MACD_signal', 'volatility_zscore', 'RSI_vol_adj', 'ROC_vol_adj'
    ])
    
    model = HybridTransformerLSTM(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, feature_cols

@st.cache_resource
def get_exchange(name):
    ex = getattr(ccxt, name)({'enableRateLimit': True})
    if name == 'binance': ex.options['defaultType'] = 'future'
    return ex

# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Monster Bot v13 - TITAN", layout="wide")
st.markdown("<style>.stMetric { background-color: #1c2128; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }</style>", unsafe_allow_html=True)

def main():
    st.sidebar.title("🤖 Monster Bot v13")
    st.sidebar.info("Model: HybridTransformerLSTM\nStatus: 🟢 Live")
    
    temp = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7)
    refresh = st.sidebar.number_input("Refresh (sec)", 10, 300, 60)
    
    model, feature_cols = load_model_and_assets(LIVE_CONFIG)
    exchange = get_exchange(LIVE_CONFIG['exchange'])
    
    col_chart, col_signal = st.columns([1, 1.3])
    with col_signal:
        st.subheader("🤖 AI Prediction")
        sig_box = st.empty(); met_box = st.empty(); reg_box = st.empty(); status = st.empty()

    with col_chart:
        st.subheader("📊 Market View")
        components.html(f'<div style="height:600px;"><div id="tv_chart" style="height:100%;"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"autosize": true,"symbol": "BINANCE:BTCUSDT","interval": "15","timezone": "Asia/Ho_Chi_Minh","theme": "dark","style": "1","locale": "en","toolbar_bg": "#0e1117","container_id": "tv_chart"}});</script></div>', height=650)

    last_update = 0
    while True:
        if time.time() - last_update < refresh:
            time.sleep(1); continue
            
        try:
            status.caption("⏳ Fetching data...")
            ohlcv = exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], LIVE_CONFIG['timeframe'], LIVE_CONFIG['limit'])
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df_enriched = enrich_features_v13(df, LIVE_CONFIG)
            df_norm = apply_rolling_normalization(df_enriched, feature_cols, LIVE_CONFIG)
            
            # Predict
            X = torch.FloatTensor(df_norm[feature_cols].tail(60).values).unsqueeze(0)
            with torch.no_grad():
                probs = apply_temperature_scaling(model(X), temp).numpy()[0]
            
            # Signal logic
            dominant = np.argmax(probs)
            conf = probs[dominant]
            signal = "BUY" if dominant == 1 and conf > 0.5 else "SELL" if dominant == 2 and conf > 0.5 else "NEUTRAL"
            
            # UI Updates
            color = "#00ff88" if signal=="BUY" else "#ff4b4b" if signal=="SELL" else "#f1c40f"
            sig_box.markdown(f'<div style="padding:25px; border-radius:15px; text-align:center; background:{color}15; border:2px solid {color};"><h1 style="color:{color};">{signal} ({conf:.1%})</h1><h2 style="color:white;">BTC: ${df["Close"].iloc[-1]:,.2f}</h2></div>', unsafe_allow_html=True)
            
            with met_box.container():
                p_cols = st.columns(3)
                p_cols[0].metric("Neutral", f"{probs[0]:.1%}")
                p_cols[1].metric("Buy", f"{probs[1]:.1%}")
                p_cols[2].metric("Sell", f"{probs[2]:.1%}")
                
                info = calculate_dynamic_tp_sl(df['Close'].iloc[-1], df_enriched['ATR'].iloc[-1], signal, LIVE_CONFIG)
                t_cols = st.columns(3)
                t_cols[0].metric("Take Profit", f"${info['tp_price']:,.2f}")
                t_cols[1].metric("Stop Loss", f"${info['sl_price']:,.2f}")
                t_cols[2].metric("R:R Ratio", f"{info['reward_risk']:.2f}x")
            
            reg = detect_market_regime(df_enriched['ADX'].iloc[-1], df_enriched['SMA_distance'].iloc[-1], LIVE_CONFIG)
            reg_box.info(f"Regime: {reg['regime']} | ADX: {reg['adx_value']:.1f}")
            
            status.caption(f"⏱️ Last update: {datetime.now().strftime('%H:%M:%S')}")
            last_update = time.time()
            
        except Exception as e:
            status.error(f"❌ Error: {e}"); time.sleep(10)

if __name__ == "__main__":
    main()
