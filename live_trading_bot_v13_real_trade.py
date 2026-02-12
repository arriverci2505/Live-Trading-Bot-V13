"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER BOT v13.6 - TITAN INTERACTIVE                                   ║
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
from scipy import signal as scipy_signal
import warnings
import logging

# Cấu hình log và cảnh báo
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 1. MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        s = x.mean(dim=1)
        e = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * e.unsqueeze(1)

class HybridTransformerLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.input_proj = nn.Linear(config['input_dim'], self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=config['num_heads'], 
            dim_feedforward=self.hidden_dim * 4, dropout=config.get('dropout', 0.3), batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_transformer_layers'])
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=config['num_lstm_layers'], batch_first=True, bidirectional=True)
        self.se_block = SEBlock(self.hidden_dim * 2, config.get('se_reduction_ratio', 16))
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ReLU(), 
            nn.Linear(self.hidden_dim, config['num_classes'])
        )
    def forward(self, x):
        x = self.pos_encoding(self.input_proj(x))
        x = self.transformer(x)
        x, _ = self.lstm(x)
        return self.classifier(self.se_block(x)[:, -1, :])

# ════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (FIXED: Added SMA200)
# ════════════════════════════════════════════════════════════════════════════

def enrich_features_v13(df):
    df = df.copy()
    # Basic Features
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # SMA 200 - Cần thiết cho bộ lọc xu hướng ở Sidebar
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # ADX Calculation
    p = 14
    plus_dm = np.where((df['High'].diff() > df['Low'].shift(1)-df['Low']), np.maximum(df['High'].diff(), 0), 0)
    minus_dm = np.where((df['Low'].shift(1)-df['Low'] > df['High'].diff()), np.maximum(df['Low'].shift(1)-df['Low'], 0), 0)
    pdi = 100 * (pd.Series(plus_dm).rolling(p).mean() / df['ATR'])
    mdi = 100 * (pd.Series(minus_dm).rolling(p).mean() / df['ATR'])
    df['ADX'] = (100 * abs(pdi-mdi)/(pdi+mdi)).rolling(p).mean()
    
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

# ════════════════════════════════════════════════════════════════════════════
# 3. MAIN INTERFACE & LOGIC
# ════════════════════════════════════════════════════════════════════════════

LIVE_CONFIG = {
    'exchange': 'kraken', 'symbol': 'BTC/USDT', 'timeframe': '15m',
    'config': {'input_dim': 29, 'hidden_dim': 256, 'num_lstm_layers': 2, 'num_transformer_layers': 2, 'num_heads': 4, 'num_classes': 3}
}

@st.cache_resource
def load_monster_model():
    model = HybridTransformerLSTM(LIVE_CONFIG['config'])
    model.eval()
    return model

def main():
    st.set_page_config(page_title="Monster Bot v13.6 Interactive", layout="wide")

    # --- SIDEBAR SETTINGS ---
    st.sidebar.title("🤖 MONSTER BOT v13.6")
    
    st.sidebar.subheader("🎮 Trading Mode")
    is_auto_trade = st.sidebar.toggle("Bật Giao Dịch Giả Lập", value=False)
    
    st.sidebar.subheader("⚙️ Chiến Thuật TP/SL")
    ui_atr_sl = st.sidebar.slider("Cắt lỗ (ATR x)", 1.0, 8.0, 4.0)
    ui_atr_tp = st.sidebar.slider("Chốt lời (ATR x)", 5.0, 40.0, 20.0)
    
    st.sidebar.subheader("🔍 Bộ Lọc Độ Chính Xác")
    ui_min_conf = st.sidebar.slider("Độ tự tin tối thiểu (%)", 50, 95, 75)
    ui_use_trend = st.sidebar.toggle("Lọc Xu Hướng (SMA 200)", value=True)
    ui_min_adx = st.sidebar.slider("Sức mạnh (Min ADX)", 10, 50, 20)
    
    st.sidebar.subheader("🛠️ Thông Số AI")
    ui_temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
    ui_refresh = st.sidebar.number_input("Cập nhật (giây)", 10, 300, 60)

    # --- LAYOUT ---
    col_signal, col_chart = st.columns([1, 1.8])

    with col_signal:
        st.markdown("### 🤖 AI Prediction")
        signal_container = st.empty()
        metrics_container = st.empty()
        status_container = st.empty()

    with col_chart:
        st.markdown("### 📊 Market View")
        tv_html = f"""<div style="height:620px;"><div id="tv_v13" style="height:100%;"></div>
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{"autosize":true,"symbol":"KRAKEN:BTCUSDT","interval":"15","theme":"dark","container_id":"tv_v13","timezone":"Asia/Ho_Chi_Minh"}});</script></div>"""
        components.html(tv_html, height=640)

    # Khởi tạo Assets
    exchange = ccxt.kraken({'enableRateLimit': True})
    model = load_monster_model()
    last_update = 0

    # Main Loop
    while True:
        current_time = time.time()
        if current_time - last_update < ui_refresh:
            time.sleep(1)
            continue
            
        try:
            # 1. Fetch & Enrich
            ohlcv = exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], timeframe='15m', limit=400)
            df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
            df_enriched = enrich_features_v13(df)
            
            # 2. Predict (Sử dụng ui_temp từ Sidebar)
            # Mock Probs (Thay bằng logic model(X) thật của bạn)
            logits = torch.randn(1, 3) 
            probs = torch.softmax(logits / ui_temp, dim=-1).detach().numpy()[0]
            dom_idx = np.argmax(probs)
            conf = probs[dom_idx]
            
            raw_sig = "BUY" if dom_idx == 1 else "SELL" if dom_idx == 2 else "NEUTRAL"
            
            # 3. Apply Filters from Sidebar
            price = df['Close'].iloc[-1]
            adx = df_enriched['ADX'].iloc[-1]
            sma200 = df_enriched['SMA200'].iloc[-1]
            atr = df_enriched['ATR'].iloc[-1]
            
            final_sig = raw_sig
            reason = "Đạt điều kiện"
            
            if conf < (ui_min_conf / 100):
                final_sig = "NEUTRAL"; reason = "Conf thấp"
            elif adx < ui_min_adx:
                final_sig = "NEUTRAL"; reason = "ADX yếu"
            elif ui_use_trend:
                if raw_sig == "BUY" and price < sma200: final_sig = "NEUTRAL"; reason = "Dưới SMA200"
                if raw_sig == "SELL" and price > sma200: final_sig = "NEUTRAL"; reason = "Trên SMA200"

            # 4. Update UI
            color = "#00ff88" if final_sig=="BUY" else "#ff4b4b" if final_sig=="SELL" else "#f1c40f"
            with signal_container.container():
                st.markdown(f"""
                    <div style="padding:20px; border:2px solid {color}; border-radius:15px; background:{color}10; text-align:center;">
                        <h1 style="color:{color}; margin:0;">{final_sig}</h1>
                        <p>Lý do: {reason}</p>
                        <hr style="border-color:#333">
                        <h3>BTC: ${price:,.2f}</h3>
                        <p>Confidence: {conf:.1%} | ADX: {adx:.1f}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if final_sig != "NEUTRAL":
                    tp = price + (atr * ui_atr_tp) if final_sig == "BUY" else price - (atr * ui_atr_tp)
                    sl = price - (atr * ui_atr_sl) if final_sig == "BUY" else price + (atr * ui_atr_sl)
                    st.success(f"🎯 TP: {tp:,.1f} | 🛡️ SL: {sl:,.1f}")
                    if is_auto_trade:
                        st.toast(f"🚀 [TRADE ẢO] Đã vào lệnh {final_sig}!", icon="🤖")

            status_container.caption(f"✅ Last update: {datetime.now().strftime('%H:%M:%S')}")
            last_update = current_time
            
        except Exception as e:
            st.error(f"Lỗi: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()

