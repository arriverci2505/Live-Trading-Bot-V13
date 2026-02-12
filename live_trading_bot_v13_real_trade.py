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
    st.set_page_config(page_title="MONSTER BOT v14.5 - PRO LOG", layout="wide")

    # --- 1. CSS & AUDIO SCRIPT ---
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        .signal-card { padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 15px; }
        .value-text { font-size: 24px; font-weight: bold; font-family: 'Consolas', monospace; }
        /* Tùy chỉnh bảng Log cho đẹp */
        [data-testid="stDataFrame"] { border: 1px solid #444; border-radius: 10px; }
        </style>
        
        <audio id="audio-alert">
          <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
        </audio>
        <script>
        function playAlert() {
          var audio = document.getElementById("audio-alert");
          audio.play();
        }
        </script>
    """, unsafe_allow_html=True)

    # --- 2. SIDEBAR ---
    st.sidebar.title("🛠️ SETTINGS")
    ui_atr_sl = st.sidebar.slider("Cắt lỗ (SL) x", 1.0, 5.0, 2.0, step=0.5)
    ui_atr_tp = st.sidebar.slider("Chốt lời (TP) x", 2.0, 15.0, 4.0, step=0.5)
    ui_min_conf = st.sidebar.slider("Độ tự tin AI (%)", 50, 95, 75, step=5)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ XÓA NHẬT KÝ"):
        st.session_state.trade_log = []
        st.rerun()

    # --- 3. LAYOUT ---
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🤖 TÍN HIỆU")
        signal_placeholder = st.empty()
        setup_placeholder = st.empty()
        
    with col_right:
        st.subheader("📜 CHI TIẾT NHẬT KÝ TRADE ẢO")
        log_placeholder = st.empty()

    # --- 4. KHỞI TẠO ---
    if 'trade_log' not in st.session_state: st.session_state.trade_log = []
    # Biến để kiểm tra lệnh mới để phát âm thanh
    if 'last_signal_time' not in st.session_state: st.session_state.last_signal_time = ""

    try:
        model = load_monster_model()
        exchange = ccxt.kraken({'enableRateLimit': True})
        feature_cols = ['log_return', 'ATR', 'BB_width', 'BB_position', 'frac_diff_close','fourier_sin_1', 'fourier_sin_2', 'fourier_sin_3', 'fourier_sin_4', 'fourier_sin_5','fourier_cos_1', 'fourier_cos_2', 'fourier_cos_3', 'fourier_cos_4', 'fourier_cos_5','volume_imbalance', 'entropy', 'volume_ratio', 'ADX', 'SMA_distance','regime_trending', 'regime_uptrend', 'regime_downtrend', 'RSI', 'MACD','MACD_signal', 'volatility_zscore', 'RSI_vol_adj', 'ROC_vol_adj']
    except Exception as e:
        st.error(f"Lỗi: {e}"); return

    # --- 5. VÒNG LẶP ---
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=400)
            df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
            df_enriched = enrich_features_v13(df)
            df_norm = apply_rolling_normalization(df_enriched, feature_cols)
            
            # Predict
            X_last = df_norm[feature_cols].tail(30).values
            X_tensor = torch.FloatTensor(X_last).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(X_tensor) / 0.7, dim=-1).numpy()[0]
            
            conf = np.max(probs)
            ai_sig = "BUY" if np.argmax(probs) == 1 else "SELL" if np.argmax(probs) == 2 else "NEUTRAL"
            
            price = df['Close'].iloc[-1]
            atr = df_enriched['ATR'].iloc[-1]
            sma200 = df_enriched['SMA200'].iloc[-1]
            
            # Filter
            final_sig = ai_sig
            if conf < (ui_min_conf/100) or (ai_sig == "BUY" and price < sma200) or (ai_sig == "SELL" and price > sma200):
                final_sig = "NEUTRAL"

            # --- HIỂN THỊ TÍN HIỆU ---
            color = "#00FF00" if final_sig == "BUY" else "#FF0000" if final_sig == "SELL" else "#FFFF00"
            border = f"4px solid {color}"
            bg = f"rgba({0 if final_sig!='SELL' else 255}, {255 if final_sig=='BUY' else 0 if final_sig=='SELL' else 255}, 0, 0.1)"

            with signal_placeholder.container():
                st.markdown(f"""
                <div class="signal-card" style="background:{bg}; border:{border};">
                    <h1 style="color:{color}; margin:0; font-size:50px;">{final_sig}</h1>
                    <p style="color:white; margin-top:5px;">BTC: $ {price:,.1f} | Conf: {conf:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            # --- XỬ LÝ LỆNH VÀ LOG CHI TIẾT ---
            if final_sig != "NEUTRAL":
                sl = price - (atr * ui_atr_sl) if final_sig == "BUY" else price + (atr * ui_atr_sl)
                tp = price + (atr * ui_atr_tp) if final_sig == "BUY" else price - (atr * ui_atr_tp)
                rr = abs(tp - price) / abs(price - sl)
                profit_est = abs(tp - price)
                
                # Hiển thị box setup nhanh
                with setup_placeholder.container():
                    st.markdown(f"""
                    <div style="background:#161a25; padding:15px; border:1px solid #444; border-radius:10px;">
                        <div style="color:#00FF88;">🎯 TP: $ {tp:,.1f}</div>
                        <div style="color:#FF4B4B;">🛑 SL: $ {sl:,.1f}</div>
                        <div style="color:#FFFF00;">⚖️ R:R: 1:{rr:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Kiểm tra nếu là lệnh mới (dựa trên phút hiện tại) để ghi log và phát âm thanh
                current_min = datetime.now().strftime("%H:%M")
                if st.session_state.last_signal_time != current_min:
                    st.session_state.last_signal_time = current_min
                    
                    # Thêm vào log với đầy đủ thông số
                    st.session_state.trade_log.insert(0, {
                        "🕒 Giờ": datetime.now().strftime("%H:%M:%S"),
                        "📈 Lệnh": final_sig,
                        "💵 Vào giá": f"{price:,.1f}",
                        "🎯 Chốt lời": f"{tp:,.1f}",
                        "🛑 Cắt lỗ": f"{sl:,.1f}",
                        "💰 Lãi dự kiến": f"+{profit_est:,.1f}$",
                        "⚖️ R:R": f"1:{rr:.1f}",
                        "🤖 Độ tự tin": f"{conf:.1%}"
                    })
                    
                    # PHÁT ÂM THANH
                    components.html("<script>playAlert();</script>", height=0)
                    st.toast(f"PHÁT HIỆN LỆNH {final_sig}!", icon="🔔")
            else:
                setup_placeholder.empty()

            # Hiển thị Bảng Log chi tiết
            with log_placeholder.container():
                if st.session_state.trade_log:
                    df_log = pd.DataFrame(st.session_state.trade_log)
                    st.dataframe(df_log, use_container_width=True, hide_index=True)
                else:
                    st.info("Đang quét thị trường... Lệnh mới sẽ xuất hiện tại đây kèm âm báo.")

            time.sleep(60)
            st.rerun()

        except Exception as e:
            st.error(f"Lỗi: {e}"); time.sleep(10)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()







