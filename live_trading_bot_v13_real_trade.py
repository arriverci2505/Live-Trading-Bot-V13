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
    st.set_page_config(page_title="MONSTER BOT v14.6 - FULL OPTION", layout="wide")

    # --- 1. CSS & AUDIO SCRIPT ---
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        .signal-card { padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 15px; }
        .value-text { font-size: 22px; font-weight: bold; font-family: 'Consolas', monospace; }
        /* Tùy chỉnh bảng Log */
        [data-testid="stDataFrame"] { border: 1px solid #444; border-radius: 10px; }
        </style>
        
        <audio id="audio-alert">
          <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
        </audio>
        <script>
        function playAlert() {
          var audio = document.getElementById("audio-alert");
          if (audio) { audio.play(); }
        }
        </script>
    """, unsafe_allow_html=True)

    # --- 2. SIDEBAR ---
    st.sidebar.title("🚀 TITAN COMMANDER")
    
    with st.sidebar.expander("🎯 SIÊU BỘ LỌC AI", expanded=True):
        ui_temp = st.slider("Temperature", 0.1, 1.5, 0.5, step=0.1)
        ui_buy_threshold = st.slider("Ngưỡng MUA", 0.3, 0.8, 0.45, step=0.05)
        ui_sell_threshold = st.slider("Ngưỡng BÁN", 0.3, 0.8, 0.45, step=0.05)

    with st.sidebar.expander("📊 BỘ LỌC THỊ TRƯỜNG", expanded=False):
        ui_adx_min = st.slider("ADX Tối thiểu", 10, 50, 20)
        ui_adx_max = st.slider("ADX Tối đa", 50, 100, 100)
        ui_use_dynamic = st.toggle("Trend động (SMA200)", value=True)

    with st.sidebar.expander("🛡️ QUẢN TRỊ LỆNH & EXIT", expanded=False):
        ui_atr_sl = st.slider("Cắt lỗ (ATR x)", 1.0, 8.0, 4.0, step=0.5)
        ui_atr_tp = st.slider("Chốt lời (ATR x)", 5.0, 40.0, 20.0, step=0.5)
        ui_trailing_act = st.slider("Kích hoạt Trailing (ATR x)", 2.0, 10.0, 4.0)
        ui_time_barrier = st.number_input("Time Barrier (Bars)", 12, 200, 48)

    if st.sidebar.button("🗑️ RESET NHẬT KÝ"):
        st.session_state.trade_log = []
        st.rerun()

    # --- 3. LAYOUT (Giữ nguyên cấu hình 1.2 : 1.8) ---
    col_left, col_right = st.columns([1.2, 1.8])
    with col_left:
        st.subheader("🤖 TÍN HIỆU")
        signal_placeholder = st.empty()
        setup_placeholder = st.empty()
        st.markdown("---")
        st.subheader("📜 LOG CHI TIẾT")
        log_placeholder = st.empty()

    with col_right:
        st.subheader("📊 BIỂU ĐỒ")
        # [TradingView Widget giữ nguyên]
        # ...
        
    # --- 4. LOGIC XỬ LÝ (SỬ DỤNG THÔNG SỐ TỪ SIDEBAR) ---
    # (Phần Load Model và Dữ liệu giữ nguyên)

    while True:
        try:
            # ... [Lấy dữ liệu và tính toán Features giữ nguyên] ...

            # AI Predict với Temperature từ Sidebar
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.softmax(logits / ui_temp, dim=-1).numpy()[0]
            
            p_neutral, p_buy, p_sell = probs[0], probs[1], probs[2]
            
            # Logic quyết định dựa trên Threshold Sidebar
            raw_sig = "NEUTRAL"
            if p_buy > ui_buy_threshold: raw_sig = "BUY"
            elif p_sell > ui_sell_threshold: raw_sig = "SELL"
            
            conf = max(p_buy, p_sell) if raw_sig != "NEUTRAL" else p_neutral
            
            # Lấy thông số thực tế
            price = df['Close'].iloc[-1]
            atr = df_enriched['ATR'].iloc[-1]
            adx = df_enriched['ADX'].iloc[-1]
            sma200 = df_enriched['SMA200'].iloc[-1]

            # Bộ lọc ADX và Trend từ Sidebar
            final_sig = raw_sig
            reason = "TÍN HIỆU ĐẠT CHUẨN"
            
            if adx < ui_adx_min or adx > ui_adx_max:
                final_sig = "NEUTRAL"; reason = f"ADX ngoài vùng ({adx:.1f})"
            elif ui_use_dynamic:
                if raw_sig == "BUY" and price < sma200: final_sig = "NEUTRAL"; reason = "Dưới SMA200"
                if raw_sig == "SELL" and price > sma200: final_sig = "NEUTRAL"; reason = "Trên SMA200"

            # --- 5. HIỂN THỊ VÀ GHI LOG (Giữ nguyên bản v14.6) ---
            # ... (Phần hiển thị card và ghi log) ...
            # Thêm thông tin Trailing Stop vào Log để dễ theo dõi
            if final_sig != "NEUTRAL":
                trailing_price = price + (atr * ui_trailing_act) if final_sig == "BUY" else price - (atr * ui_trailing_act)
                # (Lưu vào log...)

            time.sleep(60)
            st.rerun()

        except Exception as e:
            st.error(f"Lỗi: {e}"); time.sleep(10)

if __name__ == "__main__":
    main()









