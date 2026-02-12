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
    st.set_page_config(page_title="MONSTER BOT v13.9 - PRO TRADER", layout="wide")

    # --- 1. CSS CHO GIAO DIỆN "THỰC CHIẾN" ---
    st.markdown("""
        <style>
        /* Ẩn bớt padding thừa của Streamlit */
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        
        /* Style cho Box Tín Hiệu Chính */
        .signal-box {
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        
        /* Style cho Box Setup Lệnh (Entry/SL/TP) */
        .setup-box {
            background-color: #161a25;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 15px;
            margin-top: 10px;
        }
        
        /* Chữ số to rõ */
        .big-number {
            font-size: 28px;
            font-weight: 800;
            font-family: 'Consolas', monospace;
        }
        
        /* Các dòng chú thích điều kiện */
        .condition-pass { color: #00FF00; font-weight: bold; }
        .condition-fail { color: #FF0000; font-weight: bold; }
        .condition-wait { color: #FFFF00; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    # --- 2. SIDEBAR (CHỈ GIỮ CÁI CẦN THIẾT) ---
    st.sidebar.title("🎛️ CONTROL PANEL")
    
    st.sidebar.subheader("1. Quản lý vốn (Risk)")
    ui_atr_sl = st.sidebar.slider("Hệ số Cắt lỗ (SL)", 1.0, 5.0, 2.0, step=0.5, help="SL = Giá - (ATR x Hệ số)")
    ui_atr_tp = st.sidebar.slider("Hệ số Chốt lời (TP)", 2.0, 10.0, 4.0, step=0.5, help="TP = Giá + (ATR x Hệ số)")
    
    st.sidebar.subheader("2. Bộ lọc tín hiệu")
    ui_min_conf = st.sidebar.slider("Độ tin cậy AI (%)", 60, 95, 75, step=5)
    ui_use_trend = st.sidebar.checkbox("Chỉ đánh thuận xu hướng (SMA200)", value=True)
    
    # Nút dừng/chạy bot
    if 'run_bot' not in st.session_state: st.session_state.run_bot = True
    if st.sidebar.button("🛑 DỪNG BOT" if st.session_state.run_bot else "▶️ CHẠY BOT"):
        st.session_state.run_bot = not st.session_state.run_bot

    # --- 3. KHỞI TẠO HỆ THỐNG ---
    # Layout: Trái (Tín hiệu & Thông số) - Phải (Biểu đồ)
    col_signal, col_chart = st.columns([1, 2])

    with col_signal:
        main_signal_area = st.empty()  # Nơi hiển thị Box Tín Hiệu to đùng
        setup_area = st.empty()        # Nơi hiển thị SL/TP khi có lệnh
        checklist_area = st.empty()    # Nơi hiển thị lý do vào lệnh

    with col_chart:
        # Chart TradingView đơn giản, gọn nhẹ
        tv_html = f"""<div style="height:600px; border: 2px solid #333; border-radius:10px; overflow:hidden;">
        <div id="tv_chart_v13" style="height:100%;"></div>
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{"autosize":true,"symbol":"KRAKEN:BTCUSDT","interval":"15","theme":"dark","container_id":"tv_chart_v13","timezone":"Asia/Ho_Chi_Minh","hide_side_toolbar":false,"details":false}});</script></div>"""
        components.html(tv_html, height=605)

    # Load Model & Config
    try:
        model = load_monster_model()
        exchange = ccxt.kraken({'enableRateLimit': True})
        # List features cũ (vẫn cần để chạy model, nhưng không hiển thị)
        feature_cols = [
            'log_return', 'ATR', 'BB_width', 'BB_position', 'frac_diff_close',
            'fourier_sin_1', 'fourier_sin_2', 'fourier_sin_3', 'fourier_sin_4', 'fourier_sin_5',
            'fourier_cos_1', 'fourier_cos_2', 'fourier_cos_3', 'fourier_cos_4', 'fourier_cos_5',
            'volume_imbalance', 'entropy', 'volume_ratio', 'ADX', 'SMA_distance',
            'regime_trending', 'regime_uptrend', 'regime_downtrend', 'RSI', 'MACD',
            'MACD_signal', 'volatility_zscore', 'RSI_vol_adj', 'ROC_vol_adj'
        ]
    except Exception as e:
        st.error(f"Lỗi khởi tạo: {e}")
        return

    last_update = 0

    # --- 4. VÒNG LẶP CHÍNH ---
    while st.session_state.run_bot:
        current_time = time.time()
        # Refresh mỗi 60s (cố định cho đỡ rối)
        if current_time - last_update < 60:
            time.sleep(1)
            continue
            
        try:
            # 4.1 Lấy dữ liệu & Tính toán
            ohlcv = exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], timeframe='15m', limit=400)
            df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
            df_enriched = enrich_features_v13(df)
            df_norm = apply_rolling_normalization(df_enriched, feature_cols)
            
            # 4.2 AI Dự đoán
            X_last = df_norm[feature_cols].tail(LIVE_CONFIG['sequence_length']).values
            X_tensor = torch.FloatTensor(X_last).unsqueeze(0)
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.softmax(logits / 0.7, dim=-1).numpy()[0] # Temp cố định 0.7 cho ổn định
            
            # 4.3 Xử lý Logic Tín hiệu
            conf = np.max(probs)
            raw_idx = np.argmax(probs)
            ai_sig = "BUY" if raw_idx == 1 else "SELL" if raw_idx == 2 else "NEUTRAL"
            
            # Lấy thông số kỹ thuật hiện tại
            price = df['Close'].iloc[-1]
            atr = df_enriched['ATR'].iloc[-1]
            adx = df_enriched['ADX'].iloc[-1]
            sma200 = df_enriched['SMA200'].iloc[-1]
            
            # Logic lọc (Filter)
            final_sig = ai_sig
            status_msg = "SẴN SÀNG"
            
            # Các điều kiện check
            is_conf_ok = conf >= (ui_min_conf / 100)
            is_trend_ok = True
            if ui_use_trend:
                if ai_sig == "BUY" and price < sma200: is_trend_ok = False
                if ai_sig == "SELL" and price > sma200: is_trend_ok = False
            
            # Quyết định cuối cùng
            if not is_conf_ok:
                final_sig = "NEUTRAL"
                status_msg = "Độ tin cậy thấp"
            elif not is_trend_ok:
                final_sig = "NEUTRAL"
                status_msg = "Ngược xu hướng SMA200"
            elif adx < 20:
                final_sig = "NEUTRAL"
                status_msg = "Thị trường đi ngang (Sideway)"

            # --- 5. HIỂN THỊ GIAO DIỆN ---
            
            # 5.1 Cấu hình màu sắc & Viền
            if final_sig == "BUY":
                box_color = "#00FF00"  # Xanh Neon
                bg_color = "rgba(0, 50, 0, 0.8)"
                icon = "🟢 MUA NGAY"
                border = "4px solid #00FF00"
            elif final_sig == "SELL":
                box_color = "#FF0000"  # Đỏ Neon
                bg_color = "rgba(50, 0, 0, 0.8)"
                icon = "🔴 BÁN NGAY"
                border = "4px solid #FF0000"
            else:
                box_color = "#FFFF00"  # Vàng
                bg_color = "rgba(50, 50, 0, 0.3)"
                icon = "⚠️ CHỜ TÍN HIỆU"
                border = "3px dashed #FFFF00"

            # 5.2 Hiển thị Box Tín Hiệu Chính
            with main_signal_area.container():
                st.markdown(f"""
                <div class="signal-box" style="border: {border}; background-color: {bg_color};">
                    <h2 style="color: {box_color}; margin:0; font-size: 40px; text-transform: uppercase; letter-spacing: 2px;">
                        {icon}
                    </h2>
                    <p style="color: white; margin-top: 5px; font-size: 18px;">Giá hiện tại: <b>${price:,.2f}</b></p>
                </div>
                """, unsafe_allow_html=True)

            # 5.3 Hiển thị Setup Lệnh (Chỉ hiện khi BUY/SELL)
            if final_sig != "NEUTRAL":
                # Tính toán SL/TP
                sl = price - (atr * ui_atr_sl) if final_sig == "BUY" else price + (atr * ui_atr_sl)
                tp = price + (atr * ui_atr_tp) if final_sig == "BUY" else price - (atr * ui_atr_tp)
                
                with setup_area.container():
                    st.markdown(f"""
                    <div class="setup-box">
                        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                            <span style="color:#aaa;">ENTRY</span>
                            <span class="big-number" style="color:white;">${price:,.0f}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px;">
                            <span style="color:#aaa;">STOP LOSS</span>
                            <span class="big-number" style="color:#ff4b4b;">${sl:,.0f}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:#aaa;">TAKE PROFIT</span>
                            <span class="big-number" style="color:#00ff88;">${tp:,.0f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                setup_area.empty() # Xóa box setup nếu đang Neutral

            # 5.4 Hiển thị Checklist (Lý do/Bộ lọc)
            with checklist_area.container():
                # Tạo icon check/cross
                conf_icon = "✅" if is_conf_ok else "❌"
                trend_icon = "✅" if is_trend_ok else "❌"
                adx_icon = "✅" if adx >= 20 else "⚠️"
                
                st.markdown("### 📋 Điều Kiện Vào Lệnh")
                st.markdown(f"""
                * **AI Dự đoán:** {raw_idx} ({conf:.0%}) {conf_icon}
                * **Xu hướng (SMA200):** {'Thuận' if is_trend_ok else 'Ngược'} {trend_icon}
                * **Sức mạnh (ADX):** {adx:.1f} {adx_icon}
                """)
                
                if final_sig == "NEUTRAL":
                    st.info(f"💡 **Trạng thái:** {status_msg}. Hãy kiên nhẫn chờ.")

            last_update = current_time
            
        except Exception as e:
            st.error(f"Lỗi vòng lặp: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()




