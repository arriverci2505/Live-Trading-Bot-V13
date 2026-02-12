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

def backup_trade_log(new_entry):
    file_name = "titan_audit_trail.csv"
    df_new = pd.DataFrame([new_entry])
    if not os.path.isfile(file_name):
        df_new.to_csv(file_name, index=False)
    else:
        df_new.to_csv(file_name, mode='a', header=False, index=False)
        
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
    st.set_page_config(page_title="TITAN INTEL TERMINAL v15.8", layout="wide")

    # --- 1. VINTAGE TERMINAL CSS (FIXED OVERLAP & ENHANCED GLOW) ---
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;700&display=swap');
        
        .stApp { background-color: #030a03; }

        /* Hiệu ứng chữ lân quang phát sáng - CRT EFFECT */
        .crt-glow {
            color: #00FF41 !important;
            font-family: 'Fira Code', monospace !important;
            text-shadow: 
                0 0 5px rgba(0, 255, 65, 1),
                0 0 10px rgba(0, 255, 65, 0.6);
            letter-spacing: 1px;
        }

        /* Card tín hiệu: Radar Style */
        .signal-card { 
            padding: 25px; 
            border: 2px solid #00FF41; 
            background: rgba(0, 30, 0, 0.4);
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.3);
            text-align: center;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        /* Trade Setup: Command Prompt Style */
        .trade-setup { 
            background: #000; border-left: 4px solid #00FF41; 
            padding: 12px; margin-top: 10px;
            font-family: 'Fira Code', monospace;
            color: #00FF41;
        }

        /* Sidebar: QUAN TRỌNG - Sửa lỗi trùng chữ */
        [data-testid="stSidebar"] {
            background-color: #010801 !important;
            border-right: 1px solid #004400;
        }
        
        /* Chỉ định rõ ràng màu nhãn để không bị CSS khác đè lên */
        .stSlider label, .stToggle label, .stSelectbox label, [data-testid="stWidgetLabel"] p {
            color: #008800 !important; 
            font-family: 'Fira Code', monospace !important;
            font-size: 13px !important;
            text-shadow: none !important;
        }

        /* Bảng log */
        [data-testid="stDataFrame"] {
            border: 1px solid #003300;
            filter: sepia(100%) hue-rotate(80deg) brightness(85%);
        }
        
        hr { border: 0.5px solid #002200; }
        </style>
    """, unsafe_allow_html=True)

    # --- 2. SIDEBAR (ĐỦ 4 PHẦN SETTING) ---
    st.sidebar.markdown("<h2 class='crt-glow' style='font-size:22px;'>TERMINAL_CONFIG</h2>", unsafe_allow_html=True)
    
    # Mục 1: AI core
    with st.sidebar.expander("🤖 OPERATIONAL_PARAMS", expanded=True):
        ui_temp = st.slider("Signal_Temp", 0.1, 1.5, 0.5)
        ui_buy_threshold = st.slider("Buy_Threshold", 0.3, 0.8, 0.45)
        ui_sell_threshold = st.slider("Sell_Threshold", 0.3, 0.8, 0.45)

    # Mục 2: Market Filter
    with st.sidebar.expander("📡 RADAR_FILTERS", expanded=True):
        ui_adx_min = st.slider("Min_ADX_Level", 10, 50, 20)
        ui_adx_max = st.slider("Max_ADX_Level", 50, 100, 100)
        ui_use_dynamic = st.toggle("Activate_SMA_Filter", value=True)

    # Mục 3: Risk Management (PHẦN BỊ THIẾU 1)
    with st.sidebar.expander("⚖️ EXTRACTION_PROTOCOL", expanded=True):
        ui_atr_sl = st.slider("Hard_Stop (ATR)", 1.0, 10.0, 4.0)
        ui_atr_tp = st.slider("Target_Exit (ATR)", 5.0, 50.0, 20.0)

    # Mục 4: Advanced Exit (PHẦN BỊ THIẾU 2)
    with st.sidebar.expander("🛡️ ADVANCED_EXIT", expanded=True):
        ui_use_profit_lock = st.toggle("Enable_Profit_Lock", value=True)
        st.sidebar.caption("Auto-Move SL: 2.5% -> 0.5% | 5% -> 3%")

    # --- 3. MAIN LAYOUT ---
    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.markdown("<div class='crt-glow' style='font-size:16px;'>[STATUS: ANALYZING...]</div>", unsafe_allow_html=True)
        signal_placeholder = st.empty()
        setup_placeholder = st.empty()
        st.markdown("<div class='crt-glow' style='font-size:16px; margin-top:20px;'>[HISTORICAL_AUDIT_LOG]</div>", unsafe_allow_html=True)
        log_placeholder = st.empty()

    with col_right:
        st.markdown("<div class='crt-glow' style='font-size:16px;'>[LIVE_SATELLITE_FEED]</div>", unsafe_allow_html=True)
        tv_html = f"""<div style="height:750px; border: 2px solid #004400; border-radius:5px; overflow:hidden; filter: brightness(0.7) contrast(1.2) sepia(100%) hue-rotate(70deg);">
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
                if raw_sig == "BUY" and price < sma200: final_sig = "NEUTRAL"; reason = "Below SMA200"
                if raw_sig == "SELL" and price > sma200: final_sig = "NEUTRAL"; reason = "Above SMA200"

# --- 5.1 SIGNAL VISUALIZATION (CRT GLOW EFFECT) ---
            sig_color = "#00FF41" if final_sig == "BUY" else "#FF0000" if final_sig == "SELL" else "#FFFF00"
            glow_style = f"text-shadow: 0 0 20px {sig_color}, 0 0 30px {sig_color}; color: {sig_color} !important;"

            with signal_placeholder.container():
                st.markdown(f"""
                <div class='signal-card' style='border-color: {sig_color};'>
                    <div style='{glow_style} font-size:60px; font-weight:bold; font-family:Fira Code;'>{final_sig}</div>
                    <div class='crt-glow' style='font-size:20px; color:white !important;'>UNIT_PRICE: ${price:,.1f}</div>
                    <div class='crt-glow' style='font-size:14px; opacity:0.7;'>CONFIDENCE_LEVEL: {conf:.1% | STATUS: {reason}}</div>
                </div>
                """, unsafe_allow_html=True)

            # --- 5.2 ORDER SETUP & DYNAMIC CALCULATION ---
            if final_sig != "NEUTRAL":
                sl_val = price - (atr * ui_atr_sl) if final_sig == "BUY" else price + (atr * ui_atr_sl)
                tp_val = price + (atr * ui_atr_tp) if final_sig == "BUY" else price - (atr * ui_atr_tp)
                rr = abs(tp_val - price) / abs(price - sl_val)
                
                with setup_placeholder.container():
                    st.markdown(f"""
                    <div class="trade-setup">
                        <div class="crt-glow">> INITIATING_TARGET: {tp_val:,.1f}</div>
                        <div class="crt-glow">> STOP_LOSS_LIMIT:  {sl_val:,.1f}</div>
                        <div class="crt-glow">> RISK_REWARD_RATIO: 1:{rr:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Chặn trùng lặp log trong cùng 1 phút
                current_min = datetime.now().strftime("%H:%M")
                if st.session_state.last_signal_time != current_min:
                    st.session_state.last_signal_time = current_min
                    new_entry = {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Action": final_sig,
                        "Price": f"{price:,.1f}",
                        "Target": f"{tp_val:,.1f}",
                        "Stop": f"{sl_val:,.1f}",
                        "AI%": f"{conf:.1%}"
                    }
                    st.session_state.trade_log.insert(0, new_entry)
                    
                    # Auto-Backup ra file CSV để ko mất dữ liệu khi tắt bot
                    try:
                        file_name = "titan_audit_trail.csv"
                        pd.DataFrame([new_entry]).to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
                    except: pass
                    
                    # Âm thanh cảnh báo
                    components.html("<script>playAlert();</script>", height=0)
            else:
                setup_placeholder.empty()

            # --- 5.3 DASHBOARD ANALYTICS (PHẦN MỚI THÊM CHO TIỆN) ---
            with log_placeholder.container():
                if st.session_state.trade_log:
                    df_log = pd.DataFrame(st.session_state.trade_log)
                    
                    # Bảng thống kê mini phía trên Log
                    cols = st.columns(3)
                    cols[0].metric("TOTAL_SCAN", len(df_log))
                    cols[1].metric("LAST_ACTION", final_sig)
                    cols[2].metric("SYSTEM_UPTIME", datetime.now().strftime("%H:%M"))

                    # Hiển thị bảng Log chính
                    st.dataframe(
                        df_log.head(15), 
                        use_container_width=True, 
                        hide_index=True
                    )

            # --- VÒNG LẶP CHỜ QUÉT LẦN TIẾP THEO ---
            time.sleep(60)
            st.rerun()

        except Exception as e:
            st.error(f"SYSTEM CRITICAL ERROR: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
            




















