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
# 2. FEATURE ENGINEERING & NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════

def enrich_features_v13(df):
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # ADX Simple
    p = 14
    plus_dm = np.where((df['High'].diff() > df['Low'].shift(1)-df['Low']), np.maximum(df['High'].diff(), 0), 0)
    minus_dm = np.where((df['Low'].shift(1)-df['Low'] > df['High'].diff()), np.maximum(df['Low'].shift(1)-df['Low'], 0), 0)
    pdi = 100 * (pd.Series(plus_dm).rolling(p).mean() / df['ATR'])
    mdi = 100 * (pd.Series(minus_dm).rolling(p).mean() / df['ATR'])
    df['ADX'] = (100 * abs(pdi-mdi)/(pdi+mdi)).rolling(p).mean()
    
    df['SMA_distance'] = (df['Close'].rolling(20).mean() - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
    
    # Placeholder cho các Fourier và các cột khác để đủ 29 dims
    for i in range(1, 6):
        df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * df.index / 100)
        df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * df.index / 100)
    
    df['BB_width'] = (df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()) - (df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std())
    df['BB_position'] = (df['Close'] - (df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std())) / df['BB_width']
    df['frac_diff_close'] = df['Close'].diff()
    df['volume_imbalance'] = df['Volume'].diff()
    df['entropy'] = df['Close'].rolling(10).std()
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    df['regime_trending'] = (df['ADX'] > 25).astype(int)
    df['regime_uptrend'] = ((df['SMA_distance'] > 0) & (df['regime_trending'] == 1)).astype(int)
    df['regime_downtrend'] = ((df['SMA_distance'] < 0) & (df['regime_trending'] == 1)).astype(int)
    
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

def apply_rolling_normalization(df, cols):
    df_norm = df.copy()
    for col in cols:
        if col in df_norm.columns:
            mean = df_norm[col].rolling(window=100, min_periods=1).mean()
            std = df_norm[col].rolling(window=100, min_periods=1).std()
            df_norm[col] = (df_norm[col] - mean) / (std + 1e-8)
    return df_norm.fillna(0)

# ════════════════════════════════════════════════════════════════════════════
# 3. MAIN INTERFACE & LOGIC
# ════════════════════════════════════════════════════════════════════════════

# THÊM CÁC KEY THIẾU VÀO LIVE_CONFIG
LIVE_CONFIG = {
    'exchange': 'kraken', 
    'symbol': 'BTC/USDT', 
    'timeframe': '15m',
    'sequence_length': 30,
    'atr_multiplier_sl': 4.0,
    'atr_multiplier_tp': 20.0,
    'adx_threshold_trending': 25,
    'temperature': 0.7,
    'refresh_interval': 60,
    'config': {'input_dim': 29, 'hidden_dim': 256, 'num_lstm_layers': 2, 'num_transformer_layers': 2, 'num_heads': 4, 'num_classes': 3}
}

@st.cache_resource
def load_monster_model():
    model = HybridTransformerLSTM(LIVE_CONFIG['config'])
    # Trong thực tế bạn sẽ load weight ở đây: model.load_state_dict(torch.load('path.pt'))
    model.eval()
    return model

def main():
    st.set_page_config(page_title="MONSTER BOT v13.7 TITAN", layout="wide")

    # --- CSS TÙY CHỈNH CHO GIAO DIỆN ĐẸP ---
    st.markdown("""
        <style>
        .stMetric {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #333;
        }
        .stMetric:hover {
            border-color: #555;
        }
        /* Chỉnh màu chữ Neutral cho sáng */
        .neutral-box {
            color: #FFFFFF !important;
            border-color: #555555 !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- 1. SIDEBAR SETTINGS (Đã chỉnh step nhảy số chẵn) ---
    st.sidebar.title("🤖 MONSTER BOT v13")
    
    st.sidebar.subheader("🎮 Trading Mode")
    is_auto_trade = st.sidebar.toggle("Bật Giao Dịch Giả Lập", value=False)
    
    st.sidebar.subheader("⚙️ Chiến Thuật TP/SL")
    # Thêm step=0.5 để trượt: 1.0 -> 1.5 -> 2.0
    ui_atr_sl = st.sidebar.slider("Cắt lỗ (ATR x)", 1.0, 8.0, 
                                  float(LIVE_CONFIG.get('atr_multiplier_sl', 4.0)), step=0.5)
    ui_atr_tp = st.sidebar.slider("Chốt lời (ATR x)", 5.0, 40.0, 
                                  float(LIVE_CONFIG.get('atr_multiplier_tp', 20.0)), step=0.5)
    
    st.sidebar.subheader("🔍 Bộ Lọc Độ Chính Xác")
    ui_min_conf = st.sidebar.slider("Độ tự tin tối thiểu (%)", 50, 95, 75, step=5)
    ui_use_trend = st.sidebar.toggle("Lọc Xu Hướng (SMA 200)", value=True)
    ui_min_adx = st.sidebar.slider("Sức mạnh (Min ADX)", 10, 50, 
                                   int(LIVE_CONFIG.get('adx_threshold_trending', 25)), step=1)
    
    st.sidebar.subheader("🛠️ Thông Số AI")
    # Step=0.1 để chỉnh Temperature mịn hơn
    ui_temp = st.sidebar.slider("Temperature", 0.1, 1.5, 
                                float(LIVE_CONFIG.get('temperature', 0.7)), step=0.1)
    ui_refresh = st.sidebar.number_input("Cập nhật (giây)", 10, 300, 
                                         int(LIVE_CONFIG.get('refresh_interval', 60)))

    # --- 2. LAYOUT ---
    col_left, col_right = st.columns([1, 1.8])

    with col_left:
        st.markdown("### 🤖 AI Prediction")
        signal_container = st.empty()     # Box BUY/SELL
        st.write("") # Khoảng cách
        metrics_container = st.empty()   # Chỉ số
        st.write("")
        trade_log_container = st.empty() # Nhật ký
        status_container = st.empty()

    with col_right:
        st.markdown("### 📊 Market View")
        tv_html = f"""<div style="height:620px; border-radius:10px; overflow:hidden;"><div id="tv_chart_v13" style="height:100%;"></div>
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{"autosize":true,"symbol":"KRAKEN:BTCUSDT","interval":"15","theme":"dark","container_id":"tv_chart_v13","timezone":"Asia/Ho_Chi_Minh","toolbar_bg": "#111"}});</script></div>"""
        components.html(tv_html, height=640)

    # --- 3. KHỞI TẠO ---
    try:
        model = load_monster_model()
        exchange = ccxt.kraken({'enableRateLimit': True})
        # (Giữ nguyên danh sách feature_cols của bạn)
        feature_cols = [
            'log_return', 'ATR', 'BB_width', 'BB_position', 'frac_diff_close',
            'fourier_sin_1', 'fourier_sin_2', 'fourier_sin_3', 'fourier_sin_4', 'fourier_sin_5',
            'fourier_cos_1', 'fourier_cos_2', 'fourier_cos_3', 'fourier_cos_4', 'fourier_cos_5',
            'volume_imbalance', 'entropy', 'volume_ratio', 'ADX', 'SMA_distance',
            'regime_trending', 'regime_uptrend', 'regime_downtrend', 'RSI', 'MACD',
            'MACD_signal', 'volatility_zscore', 'RSI_vol_adj', 'ROC_vol_adj'
        ]
        
        if 'trade_log' not in st.session_state:
            st.session_state.trade_log = []
            
    except Exception as e:
        st.error(f"❌ Khởi tạo lỗi: {e}")
        return

    last_update = 0

    # --- 4. VÒNG LẶP CHÍNH ---
    while True:
        current_time = time.time()
        if current_time - last_update < ui_refresh:
            time.sleep(1)
            continue
            
        try:
            status_container.caption("⏳ Đang quét dữ liệu...")
            
            # Fetch & Process
            ohlcv = exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], timeframe=LIVE_CONFIG['timeframe'], limit=400)
            df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
            df_enriched = enrich_features_v13(df)
            df_norm = apply_rolling_normalization(df_enriched, feature_cols)
            
            # AI Predict
            X_last = df_norm[feature_cols].tail(LIVE_CONFIG['sequence_length']).values
            X_tensor = torch.FloatTensor(X_last).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.softmax(logits / ui_temp, dim=-1).numpy()[0]
            
            conf = np.max(probs)
            raw_idx = np.argmax(probs)
            raw_sig = "BUY" if raw_idx == 1 else "SELL" if raw_idx == 2 else "NEUTRAL"
            
            # Logic lọc
            price = df['Close'].iloc[-1]
            atr = df_enriched['ATR'].iloc[-1]
            adx_val = df_enriched['ADX'].iloc[-1]
            sma200 = df_enriched['SMA200'].iloc[-1]
            
            final_sig = raw_sig
            reason = "✅ Tín hiệu đạt chuẩn"

            if conf < (ui_min_conf / 100):
                final_sig = "NEUTRAL"; reason = f"Low Confidence ({conf:.0%})"
            elif adx_val < ui_min_adx:
                final_sig = "NEUTRAL"; reason = f"Weak Trend (ADX {adx_val:.1f})"
            elif ui_use_trend:
                if raw_sig == "BUY" and price < sma200: final_sig = "NEUTRAL"; reason = "Price < SMA200"
                if raw_sig == "SELL" and price > sma200: final_sig = "NEUTRAL"; reason = "Price > SMA200"

            # --- 5. HIỂN THỊ (ĐÃ SỬA MÀU VÀ STYLE) ---
            
            # Xử lý màu sắc
            if final_sig == "BUY":
                color = "#00ff88" # Xanh Neon
                bg_color = "rgba(0, 255, 136, 0.1)"
                border = "2px solid #00ff88"
                icon = "🟢"
            elif final_sig == "SELL":
                color = "#ff4b4b" # Đỏ Neon
                bg_color = "rgba(255, 75, 75, 0.1)"
                border = "2px solid #ff4b4b"
                icon = "🔴"
            else: # NEUTRAL
                color = "#FFFFFF" # Trắng sáng (Sửa lỗi đen thui)
                bg_color = "rgba(255, 255, 255, 0.05)"
                border = "2px dashed #666"
                icon = "⚪"

            # In Box Tín Hiệu
            with signal_container.container():
                st.markdown(f"""
                    <div style="
                        background-color: {bg_color}; 
                        border: {border}; 
                        padding: 30px; 
                        border-radius: 15px; 
                        text-align: center;
                        box-shadow: 0 0 15px {bg_color};
                    ">
                        <h1 style="color: {color}; font-size: 60px; margin: 0; font-family: 'Helvetica Neue', sans-serif; font-weight: 700;">
                            {icon} {final_sig}
                        </h1>
                        <p style="color: #cccccc; font-size: 18px; margin-top: 10px;">{reason}</p>
                    </div>
                """, unsafe_allow_html=True)

            # In Metrics (Cột thông số)
            with metrics_container.container():
                m1, m2, m3 = st.columns(3)
                # Format số tiền và chỉ số gọn gàng
                m1.metric("💰 BTC Price", f"${price:,.2f}")
                m2.metric("📉 ADX Strength", f"{adx_val:.1f}")
                m3.metric("🤖 AI Confidence", f"{conf:.1%}")

            # Xử lý Trade Ảo
            if is_auto_trade and final_sig != "NEUTRAL":
                if not st.session_state.trade_log or st.session_state.trade_log[0]['Price'] != f"${price:,.2f}":
                    tp = price + (atr * ui_atr_tp) if final_sig == "BUY" else price - (atr * ui_atr_tp)
                    sl = price - (atr * ui_atr_sl) if final_sig == "BUY" else price + (atr * ui_atr_sl)
                    
                    st.session_state.trade_log.insert(0, {
                        "⏰ Time": datetime.now().strftime("%H:%M"),
                        "Signal": final_sig,
                        "💵 Price": f"${price:,.0f}", # Bỏ số lẻ thập phân cho gọn
                        "🎯 TP": f"${tp:,.0f}",
                        "🛑 SL": f"${sl:,.0f}"
                    })
                    st.toast(f"Đã vào lệnh {final_sig} giá {price:,.0f}", icon="🚀")

            # In Nhật Ký (Bảng đẹp hơn)
            with trade_log_container.container():
                if st.session_state.trade_log:
                    st.markdown("#### 📜 Live Trade Log")
                    st.dataframe(
                        pd.DataFrame(st.session_state.trade_log).head(5),
                        use_container_width=True,
                        hide_index=True
                    )

            status_container.caption(f"✅ Last Update: {datetime.now().strftime('%H:%M:%S')}")
            last_update = current_time
            
        except Exception as e:
            status_container.error(f"⚠️ System Warning: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()

