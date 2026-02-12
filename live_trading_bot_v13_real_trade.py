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
    st.set_page_config(page_title="TITAN INTEL TERMINAL v15.2", layout="wide")

    # --- 1. INTELLIGENCE TERMINAL CSS ---
    st.markdown("""
        <style>
        /* Import font chữ hacker chuyên dụng */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

        .stApp { 
            background-color: #050505; 
            font-family: 'JetBrains Mono', monospace; 
        }

        /* Hiệu ứng chữ phát sáng (Neon Glow) */
        .glow-text {
            color: #00FF41; /* Màu xanh Matrix */
            text-shadow: 0 0 5px #00FF41, 0 0 10px #00FF41;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Box Tín hiệu phong cách tình báo */
        .signal-card { 
            padding: 20px; 
            border-radius: 5px; 
            text-align: center; 
            border: 1px solid #00FF41;
            background: rgba(0, 255, 65, 0.05);
            box-shadow: inset 0 0 15px rgba(0, 255, 65, 0.2);
            margin-bottom: 15px;
        }

        /* Bảng Log phong cách mã lệnh */
        [data-testid="stDataFrame"] { 
            border: 1px solid #333; 
            filter: brightness(0.8) sepia(1) hue-rotate(70deg); /* Ép toàn bộ bảng sang màu xanh */
        }

        /* Chỉnh lại font cho Sidebar và các Header */
        h1, h2, h3, p, span, label {
            font-family: 'JetBrains Mono', monospace !important;
            color: #00FF41 !important;
        }

        .trade-setup {
            background: #000;
            border: 1px dashed #00FF41;
            padding: 10px;
            font-size: 14px;
        }
        
        /* Loại bỏ các đường kẻ thừa của Streamlit để giao diện sạch hơn */
        hr { border: 0.5px solid #111; }
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

    # --- 2. SIDEBAR (STYLIZED) ---
    st.sidebar.markdown("<h2 class='glow-text'>SYSTEM ACCESS</h2>", unsafe_allow_html=True)
    
    with st.sidebar.expander("CORE PARAMETERS", expanded=True):
        ui_temp = st.slider("Signal Temp", 0.1, 1.5, 0.5)
        ui_buy_threshold = st.slider("Buy Limit", 0.3, 0.8, 0.45)
        ui_sell_threshold = st.slider("Sell Limit", 0.3, 0.8, 0.45)

    with st.sidebar.expander("REGIME ANALYSIS", expanded=True):
        ui_adx_min = st.slider("Min Volatility", 10, 50, 25)
        ui_use_dynamic = st.toggle("SMA Filter", value=True)

    with st.sidebar.expander("EXTRACTION PROTOCOL", expanded=False):
        ui_atr_sl = st.slider("Hard Stop (ATR)", 1.0, 10.0, 4.0)
        ui_atr_tp = st.slider("Target Exit (ATR)", 5.0, 50.0, 20.0)
    
    if st.sidebar.button("TERMINATE LOGS"):
        st.session_state.trade_log = []
        st.rerun()

    # --- 3. MAIN LAYOUT ---
    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.markdown("<h3 class='glow-text'>> AI_ANALYSIS_STREAM</h3>", unsafe_allow_html=True)
        signal_placeholder = st.empty()
        setup_placeholder = st.empty()
        st.markdown("<h3 class='glow-text'>> AUDIT_TRAIL</h3>", unsafe_allow_html=True)
        log_placeholder = st.empty()

    with col_right:
        st.markdown("<h3 class='glow-text'>> LIVE_SATELLITE_FEED</h3>", unsafe_allow_html=True)
        # Ép TradingView sang theme Dark nhất có thể
        tv_html = f"""<div style="height:750px; border: 1px solid #00FF41; border-radius:5px; overflow:hidden; filter: grayscale(1) contrast(1.2) brightness(0.8) sepia(1) hue-rotate(70deg);">
        <div id="tv_chart_v15" style="height:100%;"></div>
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{"autosize":true,"symbol":"KRAKEN:BTCUSDT","interval":"15","theme":"dark","container_id":"tv_chart_v15","style":"1","enable_publishing":false,"hide_side_toolbar":false,"allow_symbol_change":true}});</script></div>"""
        components.html(tv_html, height=760)

    # --- 4. DATA INITIALIZATION ---
    if 'trade_log' not in st.session_state: st.session_state.trade_log = []
    if 'last_signal_time' not in st.session_state: st.session_state.last_signal_time = ""

    try:
        model = load_monster_model()
        exchange = ccxt.kraken({'enableRateLimit': True})
        feature_cols = ['log_return', 'ATR', 'BB_width', 'BB_position', 'frac_diff_close','fourier_sin_1', 'fourier_sin_2', 'fourier_sin_3', 'fourier_sin_4', 'fourier_sin_5','fourier_cos_1', 'fourier_cos_2', 'fourier_cos_3', 'fourier_cos_4', 'fourier_cos_5','volume_imbalance', 'entropy', 'volume_ratio', 'ADX', 'SMA_distance','regime_trending', 'regime_uptrend', 'regime_downtrend', 'RSI', 'MACD','MACD_signal', 'volatility_zscore', 'RSI_vol_adj', 'ROC_vol_adj']
    except Exception as e:
        st.error(f"Initialization Error: {e}"); return

    # --- 5. MAIN LOOP ---
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=400)
            df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
            df_enriched = enrich_features_v13(df)
            df_norm = apply_rolling_normalization(df_enriched, feature_cols)
            
            X_last = df_norm[feature_cols].tail(30).values
            X_tensor = torch.FloatTensor(X_last).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.softmax(logits / ui_temp, dim=-1).numpy()[0]
            
            p_neutral, p_buy, p_sell = probs[0], probs[1], probs[2]
            
            raw_sig = "NEUTRAL"
            if p_buy > ui_buy_threshold: raw_sig = "BUY"
            elif p_sell > ui_sell_threshold: raw_sig = "SELL"
            
            price = df['Close'].iloc[-1]
            atr = df_enriched['ATR'].iloc[-1]
            adx = df_enriched['ADX'].iloc[-1]
            sma200 = df_enriched['SMA200'].iloc[-1]
            conf = max(p_buy, p_sell) if raw_sig != "NEUTRAL" else p_neutral

            final_sig = raw_sig
            reason = "OPTIMAL CONDITION"
            
            if adx < ui_adx_min or adx > ui_adx_max:
                final_sig = "NEUTRAL"; reason = f"Weak ADX ({adx:.1f})"
            elif ui_use_dynamic:
                if raw_sig == "BUY" and price < sma200: final_sig = "NEUTRAL"; reason = "Below SMA200 (Bearish)"
                if raw_sig == "SELL" and price > sma200: final_sig = "NEUTRAL"; reason = "Above SMA200 (Bullish)"

            # 5.1 Signal Visualization
            color = "#00FF00" if final_sig == "BUY" else "#FF0000" if final_sig == "SELL" else "#FFFF00"
            bg = f"rgba({255 if final_sig=='SELL' else 0}, {255 if final_sig=='BUY' else 0 if final_sig=='SELL' else 255}, 0, 0.1)"

            with signal_placeholder.container():
                st.markdown(f"""
                "<div class='signal-card'><h1 class='glow-text'>{final_sig}
                    <h1 style="color:{color}; margin:0; font-size:50px; letter-spacing:2px;">{final_sig}</h1>
                    <p style="color:white; margin-top:5px; font-weight:bold;">BTC: $ {price:,.1f} | AI CONF: {conf:.1%}</p>
                    <p style="color:#aaa; font-size:14px; margin:0; text-transform:uppercase;">{reason}</p>
                </div>
                """, unsafe_allow_html=True)

            # 5.2 Order Setup & Audit Log
            if final_sig != "NEUTRAL":
                sl_val = price - (atr * ui_atr_sl) if final_sig == "BUY" else price + (atr * ui_atr_sl)
                tp_val = price + (atr * ui_atr_tp) if final_sig == "BUY" else price - (atr * ui_atr_tp)
                rr = abs(tp_val - price) / abs(price - sl_val)
                
                with setup_placeholder.container():
                    st.markdown(f"""
                    <div class="trade-setup">
                        <span style="color:#00FF88; font-weight:bold;">TP: $ {tp_val:,.1f}</span> | 
                        <span style="color:#FF4B4B; font-weight:bold;">SL: $ {sl_val:,.1f}</span> | 
                        <span style="color:#FFFF00; font-weight:bold;">R:R: 1:{rr:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)

                current_min = datetime.now().strftime("%H:%M")
                if st.session_state.last_signal_time != current_min:
                    st.session_state.last_signal_time = current_min
                    st.session_state.trade_log.insert(0, {
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Action": final_sig,
                        "Price": f"{price:,.1f}",
                        "TP": f"{tp_val:,.1f}",
                        "SL": f"{sl_val:,.1f}",
                        "R:R": f"1:{rr:.1f}",
                        "Conf": f"{conf:.1%}"
                    })
                    components.html("<script>playAlert();</script>", height=0)
            else:
                setup_placeholder.empty()

            # 5.3 Audit Log Refresh
            with log_placeholder.container():
                if st.session_state.trade_log:
                    st.dataframe(pd.DataFrame(st.session_state.trade_log).head(20), use_container_width=True, hide_index=True)

            time.sleep(60)
            st.rerun()

        except Exception as e:
            st.error(f"System Error: {e}"); time.sleep(10)

if __name__ == "__main__":
    main()
            
if __name__ == "__main__":
    main()












