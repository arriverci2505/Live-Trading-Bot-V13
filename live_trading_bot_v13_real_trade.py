"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER BOT v13 - HOÀN THIỆN: LAYOUT + FILTERS + SIMULATOR            ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import streamlit.components.v1 as components
import time
from datetime import datetime
from pathlib import Path
from scipy import signal as scipy_signal
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & ARCHITECTURE (Giữ nguyên các class Model của bạn)
# ════════════════════════════════════════════════════════════════════════════

LIVE_CONFIG = {
    'exchange': 'kraken', # Dùng Kraken để tránh lỗi 451 của Binance trên Cloud
    'symbol': 'BTC/USDT',
    'timeframe': '15m',
    'limit': 500,
    'model_path': './BTC-USDT_MONSTER_model.pt',
    'sequence_length': 60,
    'temperature': 0.7,
    'entry_percentile': 85.0,
    'rolling_window': 200,
    'atr_multiplier_sl': 4.0,
    'atr_multiplier_tp': 20.0,
    'leverage': 5,
    'refresh_interval': 60,
    'min_confidence': 0.75,
    'use_trend_filter': True,
    'min_adx': 20
}

# [Đoạn này giữ nguyên các class: PositionalEncoding, SEBlock, HybridTransformerLSTM của bạn]
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
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        squeeze = x.mean(dim=1)
        excitation = torch.sigmoid(self.fc2(torch.relu(self.fc1(squeeze))))
        return x * excitation.unsqueeze(1)

class HybridTransformerLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim, self.hidden_dim = config['input_dim'], config['hidden_dim']
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=config['num_heads'], batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_transformer_layers'])
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=config['num_lstm_layers'], batch_first=True, bidirectional=True)
        self.se_block = SEBlock(self.hidden_dim * 2)
        self.final_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim * 2, num_heads=config['num_heads'], batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, config['num_classes']))
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x, _ = self.lstm(x)
        x = self.se_block(x)
        attn, _ = self.final_attention(x, x, x)
        return self.classifier(attn[:, -1, :])

# [Các hàm enrich_features_v13, calculate_fractional_diff... giữ nguyên từ code cũ của bạn]
# Lưu ý: Đảm bảo các hàm này có sẵn trong file của bạn

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (Đã sửa để nhận config)
# ════════════════════════════════════════════════════════════════════════════

def predict_v13(model, df_normalized, feature_cols, config):
    try:
        X_raw = df_normalized[feature_cols].tail(config['sequence_length']).values
        X_tensor = torch.FloatTensor(X_raw).unsqueeze(0)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits / config['temperature'], dim=-1).cpu().numpy()[0]
        
        dominant_class = np.argmax(probs)
        conf = probs[dominant_class]
        
        sig = "NEUTRAL"
        if dominant_class == 1: sig = "BUY"
        elif dominant_class == 2: sig = "SELL"
        
        return sig, conf, probs
    except:
        return "NEUTRAL", 0.0, [0.33, 0.33, 0.33]

@st.cache_resource
def load_model_and_assets(path):
    checkpoint = torch.load(path, map_location='cpu')
    model = HybridTransformerLSTM(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['feature_cols']

@st.cache_resource
def get_exchange(name):
    return ccxt.kraken({'enableRateLimit': True}) if name == 'kraken' else ccxt.binance()

# ════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Monster Bot v13", layout="wide")
    
    # --- SIDEBAR SETTINGS ---
    st.sidebar.title("🤖 MONSTER BOT v13")
    
    st.sidebar.subheader("🎮 Trading Mode")
    is_auto_trade = st.sidebar.toggle("Bật Giao Dịch Giả Lập", value=False)
    
    st.sidebar.subheader("⚙️ Chiến Thuật TP/SL")
    atr_sl = st.sidebar.slider("Cắt lỗ (ATR x)", 1.0, 8.0, 4.0)
    atr_tp = st.sidebar.slider("Chốt lời (ATR x)", 5.0, 40.0, 20.0)
    
    st.sidebar.subheader("🔍 Bộ Lọc Độ Chính Xác")
    min_conf = st.sidebar.slider("Độ tự tin tối thiểu (%)", 50, 95, 75)
    use_trend_filter = st.sidebar.toggle("Lọc Xu Hướng (SMA 200)", value=True)
    min_adx = st.sidebar.slider("Sức mạnh (Min ADX)", 10, 50, 20)
    
    st.sidebar.subheader("🛠️ Thông Số")
    temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
    refresh_sec = st.sidebar.number_input("Cập nhật (giây)", 10, 300, 60)

    # Cập nhật config từ UI
    LIVE_CONFIG.update({
        'temperature': temp, 'atr_multiplier_sl': atr_sl, 'atr_multiplier_tp': atr_tp,
        'refresh_interval': refresh_sec, 'min_confidence': min_conf/100,
        'use_trend_filter': use_trend_filter, 'min_adx': min_adx
    })

    # Load Assets
    model, feature_cols = load_model_and_assets(LIVE_CONFIG['model_path'])
    exchange = get_exchange(LIVE_CONFIG['exchange'])
    
    # --- LAYOUT (Signal TRÁI, Chart PHẢI) ---
    col_signal, col_chart = st.columns([1, 1.8])

    with col_signal:
        st.markdown("### 🤖 AI Prediction")
        signal_container = st.empty()
        prob_container = st.empty()
        status_container = st.empty()
        trade_log_container = st.empty()

    with col_chart:
        st.markdown("### 📊 Market View")
        tv_html = """<div style="height:600px;"><div id="tv_chart" style="height:100%;"></div>
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({"autosize":true,"symbol":"BINANCE:BTCUSDT","interval":"15","theme":"dark","container_id":"tv_chart"});</script></div>"""
        components.html(tv_html, height=620)

    # --- MAIN LOOP ---
    last_update = 0
    while True:
        if time.time() - last_update < LIVE_CONFIG['refresh_interval']:
            time.sleep(1); continue
            
        try:
            # Fetch & Process
            ohlcv = exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], timeframe='15m', limit=400)
            df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
            # [Giả sử bạn đã có hàm enrich_features_v13 và apply_rolling_normalization ở trên]
            df_enriched = enrich_features_v13(df, LIVE_CONFIG) 
            df_norm = apply_rolling_normalization(df_enriched, feature_cols, LIVE_CONFIG)
            
            # Prediction
            sig, conf, probs = predict_v13(model, df_norm, feature_cols, LIVE_CONFIG)
            
            # --- BỘ LỌC NÂNG CAO ---
            price = df['Close'].iloc[-1]
            sma200 = df['Close'].rolling(200).mean().iloc[-1]
            adx = df_enriched['ADX'].iloc[-1]
            
            if conf < LIVE_CONFIG['min_confidence']: sig = "NEUTRAL"
            if adx < LIVE_CONFIG['min_adx']: sig = "NEUTRAL"
            if LIVE_CONFIG['use_trend_filter']:
                if sig == "BUY" and price < sma200: sig = "NEUTRAL"
                if sig == "SELL" and price > sma200: sig = "NEUTRAL"

            # UI Update
            color = "#2ecc71" if sig=="BUY" else "#e74c3c" if sig=="SELL" else "#f1c40f"
            with signal_container.container():
                st.markdown(f"""<div style="padding:20px; border:2px solid {color}; border-radius:10px; background:{color}10; text-align:center;">
                    <h1 style="color:{color};">{sig}</h1>
                    <h3>Price: ${price:,.2f} | Conf: {conf:.1%}</h3>
                    <p>ADX: {adx:.1f} | Trend: {'OK' if sig!='NEUTRAL' else 'Wait'}</p>
                </div>""", unsafe_allow_html=True)
            
            # Simulator Toast
            if is_auto_trade and sig != "NEUTRAL":
                st.toast(f"🚀 Đã vào lệnh {sig} giả lập tại {price}", icon="🤖")

            status_container.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
            last_update = time.time()
            
        except Exception as e:
            st.error(f"Lỗi: {e}"); time.sleep(10)

if __name__ == "__main__":
    main()
