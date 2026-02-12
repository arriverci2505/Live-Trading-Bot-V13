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
    st.set_page_config(page_title="MONSTER BOT v13.6 TITAN", layout="wide")

    # --- 1. SIDEBAR SETTINGS (Bảng điều khiển) ---
    st.sidebar.title("🤖 MONSTER BOT v13")
    
    st.sidebar.subheader("🎮 Trading Mode")
    is_auto_trade = st.sidebar.toggle("Bật Giao Dịch Giả Lập", value=False)
    
    st.sidebar.subheader("⚙️ Chiến Thuật TP/SL")
    ui_atr_sl = st.sidebar.slider("Cắt lỗ (ATR x)", 1.0, 8.0, 4.0)
    ui_atr_tp = st.sidebar.slider("Chốt lời (ATR x)", 5.0, 40.0, 20.0)
    
    st.sidebar.subheader("🔍 Bộ Lọc Độ Chính Xác")
    ui_min_conf = st.sidebar.slider("Độ tự tin tối thiểu (%)", 50, 95, 75)
    ui_use_trend = st.sidebar.toggle("Lọc Xu Hướng (SMA 200)", value=True)
    ui_min_adx = st.sidebar.slider("Sức mạnh (Min ADX)", 10, 50, 25)
    
    st.sidebar.subheader("🛠️ Thông Số AI")
    ui_temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
    ui_refresh = st.sidebar.number_input("Cập nhật (giây)", 10, 300, 60)

    # --- 2. LAYOUT (Phân bổ màn hình) ---
    col_left, col_right = st.columns([1, 1.8])

    with col_left:
        st.markdown("### 🤖 AI Prediction")
        signal_container = st.empty()    # Box BUY/SELL
        metrics_container = st.empty()  # Các chỉ số ADX, RSI...
        trade_log_container = st.empty() # Nhật ký lệnh
        status_container = st.empty()   # Trạng thái cập nhật

    with col_right:
        st.markdown("### 📊 Market View")
        tv_html = f"""<div style="height:620px;"><div id="tv_v13" style="height:100%;"></div>
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{"autosize":true,"symbol":"KRAKEN:BTCUSDT","interval":"15","theme":"dark","container_id":"tv_v13","timezone":"Asia/Ho_Chi_Minh"}});</script></div>"""
        components.html(tv_html, height=640)

    # --- 3. KHỞI TẠO ---
    exchange = ccxt.kraken({'enableRateLimit': True})
    model, feature_cols = load_assets(LIVE_CONFIG['model_path'], LIVE_CONFIG)
    
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    last_update = 0

    # --- 4. VÒNG LẶP CHÍNH ---
    while True:
        current_time = time.time()
        if current_time - last_update < ui_refresh:
            time.sleep(1)
            continue
            
        try:
            status_container.caption("⏳ Đang tải dữ liệu từ sàn...")
            
            # Lấy dữ liệu và tính toán kỹ thuật
            ohlcv = exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], timeframe='15m', limit=400)
            df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
            df_enriched = enrich_features_v13(df)
            df_norm = apply_rolling_normalization(df_enriched, feature_cols)
            
            # Dự đoán từ Model AI
            # (Giả sử bạn đã có hàm predict_v13 xử lý X_seq và model)
            # signal_raw, conf, probs = predict_v13(model, df_norm, feature_cols, ui_temp)
            
            # --- GIẢ LẬP LOGIC DỰ ĐOÁN (Thay bằng code thực tế của bạn) ---
            probs = np.random.dirichlet(np.ones(3), size=1)[0] 
            conf = np.max(probs)
            signal_raw = "BUY" if np.argmax(probs) == 1 else "SELL" if np.argmax(probs) == 2 else "NEUTRAL"
            
            # --- ÁP DỤNG BỘ LỌC TỪ SIDEBAR ---
            price = df['Close'].iloc[-1]
            atr = df_enriched['ATR'].iloc[-1]
            adx_val = df_enriched['ADX'].iloc[-1]
            sma200 = df['Close'].rolling(200).mean().iloc[-1]
            
            final_sig = signal_raw
            reason = "✅ Tín hiệu đạt chuẩn"

            if conf < (ui_min_conf / 100):
                final_sig = "NEUTRAL"; reason = "❌ Độ tự tin thấp"
            elif adx_val < ui_min_adx:
                final_sig = "NEUTRAL"; reason = "❌ Thị trường đi ngang (ADX thấp)"
            elif ui_use_trend:
                if signal_raw == "BUY" and price < sma200: final_sig = "NEUTRAL"; reason = "❌ BUY dưới SMA200"
                if signal_raw == "SELL" and price > sma200: final_sig = "NEUTRAL"; reason = "❌ SELL trên SMA200"

            # --- 5. IN RA MÀN HÌNH (DISPLAY UI) ---
            
            # A. Box Tín hiệu khổng lồ
            color = "#00ff88" if final_sig == "BUY" else "#ff4b4b" if final_sig == "SELL" else "#888888"
            bg_color = "rgba(0, 255, 136, 0.1)" if final_sig == "BUY" else "rgba(255, 75, 75, 0.1)" if final_sig == "SELL" else "rgba(136, 136, 136, 0.1)"
            
            with signal_container.container():
                st.markdown(f"""
                    <div style="background-color: {bg_color}; border: 2px solid {color}; padding: 25px; border-radius: 15px; text-align: center;">
                        <h1 style="color: {color}; font-size: 50px; margin: 0;">{final_sig}</h1>
                        <p style="color: white; opacity: 0.8; margin: 5px 0;">{reason}</p>
                    </div>
                """, unsafe_allow_html=True)

            # B. Các chỉ số Metrics
            with metrics_container.container():
                st.write("") # Tạo khoảng cách
                m1, m2, m3 = st.columns(3)
                m1.metric("Price", f"${price:,.2f}")
                m2.metric("ADX", f"{adx_val:.1f}")
                m3.metric("AI Conf", f"{conf:.1%}")

            # C. Nhật ký Trade ảo & Thông báo Toast
            if is_auto_trade and final_sig != "NEUTRAL":
                if not st.session_state.trade_log or st.session_state.trade_log[0]['Price'] != f"${price:,.2f}":
                    st.session_state.trade_log.insert(0, {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Signal": final_sig,
                        "Price": f"${price:,.2f}",
                        "Status": "🚀 Entered"
                    })
                    st.toast(f"Đã vào lệnh {final_sig} tại {price:,.2f}", icon="🤖")

            with trade_log_container.container():
                st.markdown("#### 📜 Recent Signals")
                if st.session_state.trade_log:
                    st.table(pd.DataFrame(st.session_state.trade_log).head(5))

            status_container.caption(f"✅ Last Update: {datetime.now().strftime('%H:%M:%S')}")
            last_update = current_time
            
        except Exception as e:
            status_container.error(f"❌ Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()


