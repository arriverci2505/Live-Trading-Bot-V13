"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER BOT v13 - LIVE TRADING DASHBOARD (STREAMLIT + PYTORCH)        ║
║                                                                          ║
║  Features:                                                              ║
║  ✅ PyTorch Model Loading (HybridTransformerLSTM)                       ║
║  ✅ Advanced Feature Engineering (Frac Diff, Fourier, Rolling Z-Score) ║
║  ✅ Temperature Scaling (T=0.7) + Adaptive Thresholding (P85)          ║
║  ✅ Market Regime Detection (ADX-based)                                ║
║  ✅ Dynamic TP/SL (ATR Adaptive: 4x SL, 20x TP)                        ║
║  ✅ Advanced Exit Manager Integration                                  ║
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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

LIVE_CONFIG = {
    # Exchange & Symbol
    'exchange': 'binance',           # binance, kraken, etc.
    'symbol': 'BTC/USDT',
    'timeframe': '15m',
    'limit': 500,                    # Enough for EMA 200 + rolling window
    
    # Model Paths
    'model_path': './models/BTC-USDT_MONSTER_model.pt',
    'feature_cols_path': './models/BTC-USDT_feature_cols.txt',
    
    # v13 Settings
    'sequence_length': 60,
    'temperature': 0.7,              # Temperature scaling
    'entry_percentile': 85.0,        # Top 15% probabilities
    'rolling_window': 200,           # Rolling Z-score window
    'rolling_min_periods': 50,
    
    # Market Regime
    'adx_threshold_trending': 25,
    'adx_threshold_ranging': 20,
    
    # Risk Management
    'atr_multiplier_sl': 4.0,        # Stop Loss = 4x ATR
    'atr_multiplier_tp': 20.0,       # Take Profit = 20x ATR
    'profit_lock_levels': [
        (1.2, 0.5),
        (2.0, 1.0),
        (3.0, 1.5),
    ],
    
    # Trading
    'leverage': 5,
    'risk_per_trade': 0.02,          # 2% of capital
    
    # UI Update
    'refresh_interval': 60,          # Seconds between updates
}

# ════════════════════════════════════════════════════════════════════════════
# 2. PYTORCH MODEL ARCHITECTURE (Copy from v13)
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
        
        # Embedding
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Positional Encoding
        if config.get('use_positional_encoding', True):
            self.pos_encoder = PositionalEncoding(self.hidden_dim)
        else:
            self.pos_encoder = None
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config['num_heads'],
            dim_feedforward=self.hidden_dim * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_transformer_layers']
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=config['num_lstm_layers'],
            batch_first=True,
            dropout=config['dropout'] if config['num_lstm_layers'] > 1 else 0
        )
        
        # SE Block
        self.se_block = SEBlock(self.hidden_dim, config['se_reduction_ratio'])
        
        # Classification Head
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        # Projection
        x = self.input_projection(x)
        
        # Positional Encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # SE Block
        x = self.se_block(x)
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# ════════════════════════════════════════════════════════════════════════════
# 3. ADVANCED FEATURE ENGINEERING (v13)
# ════════════════════════════════════════════════════════════════════════════

def calculate_fractional_diff(series: pd.Series, d: float = 0.5, threshold: float = 0.01) -> pd.Series:
    """Fractional Differentiation with memory preservation."""
    x = series.dropna().values
    n = len(x)
    
    # Calculate weights
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1
    
    weights = np.array(weights)
    width = len(weights)
    
    # Apply fractional diff
    output = np.full(n, np.nan)
    for i in range(width - 1, n):
        output[i] = np.dot(weights[::-1], x[i - width + 1:i + 1])
    
    result = pd.Series(output, index=series.dropna().index)
    return result.reindex(series.index)

def calculate_fourier_features(series: pd.Series, n_components: int = 5) -> pd.DataFrame:
    """Extract dominant frequencies using FFT."""
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
    """Volume Order Imbalance."""
    price_range = df['High'] - df['Low']
    price_range = price_range.replace(0, np.nan)
    
    imbalance = ((df['Close'] - df['Open']) / price_range) * df['Volume']
    return imbalance.ffill().fillna(0)

def calculate_entropy(series: pd.Series, window: int = 14) -> pd.Series:
    """Shannon Entropy."""
    def _entropy(x):
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return 0
        counts = np.bincount((x * 100).astype(int) - (x * 100).astype(int).min())
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    return series.rolling(window).apply(_entropy, raw=True).ffill().fillna(0)

def enrich_features_v13(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Advanced feature engineering matching v13 training pipeline.
    """
    df = df.copy()
    
    logger.info(f"🔧 Feature Engineering v13 (Input: {len(df)} rows)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # BASIC FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['BB_upper'] = sma_20 + (2 * std_20)
    df['BB_lower'] = sma_20 - (2 * std_20)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / sma_20
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # ═══════════════════════════════════════════════════════════════════════
    # FRACTIONAL DIFFERENTIATION
    # ═══════════════════════════════════════════════════════════════════════
    
    logger.info("   ⚡ Fractional Differentiation...")
    df['frac_diff_close'] = calculate_fractional_diff(df['Close'], d=0.5, threshold=0.01)
    
    # ═══════════════════════════════════════════════════════════════════════
    # FOURIER TRANSFORM
    # ═══════════════════════════════════════════════════════════════════════
    
    logger.info("   ⚡ Fourier Transform...")
    df_fourier = calculate_fourier_features(df['Close'], n_components=5)
    df = pd.concat([df, df_fourier], axis=1)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MICROSTRUCTURE FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    
    logger.info("   ⚡ Microstructure Features...")
    df['volume_imbalance'] = calculate_volume_order_imbalance(df)
    df['entropy'] = calculate_entropy(df['Close'], window=14)
    
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    
    # ═══════════════════════════════════════════════════════════════════════
    # MARKET REGIME (ADX)
    # ═══════════════════════════════════════════════════════════════════════
    
    logger.info("   ⚡ Market Regime (ADX)...")
    period = 14
    df['plus_dm'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0),
        0
    )
    df['minus_dm'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )
    
    atr_smooth = df['ATR'].rolling(period).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(period).mean() / atr_smooth)
    df['minus_di'] = 100 * (df['minus_dm'].rolling(period).mean() / atr_smooth)
    
    dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['ADX'] = dx.rolling(period).mean()
    
    # SMA Distance
    df['SMA_short'] = df['Close'].rolling(20).mean()
    df['SMA_long'] = df['Close'].rolling(50).mean()
    df['SMA_distance'] = (df['SMA_short'] - df['SMA_long']) / df['SMA_long']
    
    df['regime_trending'] = (df['ADX'] > 25).astype(int)
    df['regime_uptrend'] = ((df['SMA_distance'] > 0) & (df['regime_trending'] == 1)).astype(int)
    df['regime_downtrend'] = ((df['SMA_distance'] < 0) & (df['regime_trending'] == 1)).astype(int)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ADDITIONAL INDICATORS
    # ═══════════════════════════════════════════════════════════════════════
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # Volatility Z-score
    returns = df['Close'].pct_change()
    volatility = returns.rolling(20).std()
    vol_mean = volatility.rolling(100).mean()
    vol_std = volatility.rolling(100).std()
    df['volatility_zscore'] = (volatility - vol_mean) / vol_std.replace(0, np.nan)
    
    # Volatility-adjusted indicators
    df['RSI_vol_adj'] = df['RSI'] / (volatility * 100).replace(0, np.nan)
    roc = df['Close'].pct_change(10) * 100
    df['ROC_vol_adj'] = roc / (volatility * 100).replace(0, np.nan)
    
    # ═══════════════════════════════════════════════════════════════════════
    # WARM-UP HANDLING
    # ═══════════════════════════════════════════════════════════════════════
    
    critical_indicators = ['ATR', 'ADX', 'volatility_zscore', 'SMA_long', 'frac_diff_close']
    
    warmup_periods = {}
    for col in critical_indicators:
        if col in df.columns:
            first_valid = df[col].first_valid_index()
            if first_valid is not None:
                warmup_periods[col] = df.index.get_loc(first_valid)
    
    if warmup_periods:
        max_warmup = max(warmup_periods.values())
        df = df.iloc[max_warmup:].copy()
    
    df = df.ffill()
    df.fillna(0, inplace=True)
    
    logger.info(f"   ✓ Features ready: {len(df)} rows retained")
    
    return df

def apply_rolling_normalization(df: pd.DataFrame, feature_cols: list, config: dict) -> pd.DataFrame:
    """
    Rolling Z-Score Normalization (v13).
    """
    df = df.copy()
    
    window = config.get('rolling_window', 200)
    min_periods = config.get('rolling_min_periods', 50)
    
    logger.info(f"🔄 Rolling Z-Score (window={window})...")
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        rolling_mean = df[col].rolling(window=window, min_periods=min_periods).mean()
        rolling_std = df[col].rolling(window=window, min_periods=min_periods).std()
        rolling_std = rolling_std.replace(0, np.nan)
        
        df[col] = (df[col] - rolling_mean) / rolling_std
        df[col] = df[col].ffill().fillna(0)
        df[col] = df[col].clip(-5, 5)
    
    logger.info(f"   ✓ Normalization complete")
    
    return df

# ════════════════════════════════════════════════════════════════════════════
# 4. ADAPTIVE CONFIDENCE & TEMPERATURE SCALING
# ════════════════════════════════════════════════════════════════════════════

def apply_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature scaling for probability calibration."""
    return torch.softmax(logits / temperature, dim=-1)

def determine_signal(probs: np.ndarray, buy_threshold: float, sell_threshold: float) -> tuple:
    """
    Determine trading signal based on probabilities.
    
    Returns:
        (signal: str, confidence: float, dominant_class: int)
    """
    prob_neutral, prob_buy, prob_sell = probs
    
    # Find dominant class
    dominant_class = np.argmax(probs)
    confidence = probs[dominant_class]
    
    if dominant_class == 1 and prob_buy >= buy_threshold:
        signal = "BUY"
    elif dominant_class == 2 and prob_sell >= sell_threshold:
        signal = "SELL"
    else:
        signal = "NEUTRAL"
    
    return signal, confidence, dominant_class

# ════════════════════════════════════════════════════════════════════════════
# 5. MARKET REGIME DETECTOR
# ════════════════════════════════════════════════════════════════════════════

def detect_market_regime(adx: float, sma_distance: float, config: dict) -> dict:
    """
    Detect current market regime.
    
    Returns:
        {
            'regime': str,  # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
            'is_trending': bool,
            'trend_direction': str,  # UP, DOWN, NEUTRAL
            'adx_strength': str,  # WEAK, MEDIUM, STRONG
        }
    """
    adx_trending = config.get('adx_threshold_trending', 25)
    adx_ranging = config.get('adx_threshold_ranging', 20)
    
    # ADX strength
    if adx > adx_trending:
        is_trending = True
        adx_strength = "STRONG"
    elif adx > adx_ranging:
        is_trending = True
        adx_strength = "MEDIUM"
    else:
        is_trending = False
        adx_strength = "WEAK"
    
    # Trend direction
    if sma_distance > 0.02:
        trend_direction = "UP"
    elif sma_distance < -0.02:
        trend_direction = "DOWN"
    else:
        trend_direction = "NEUTRAL"
    
    # Regime classification
    if is_trending and trend_direction == "UP":
        regime = "TRENDING_UP"
    elif is_trending and trend_direction == "DOWN":
        regime = "TRENDING_DOWN"
    elif not is_trending:
        regime = "RANGING"
    else:
        regime = "VOLATILE"
    
    return {
        'regime': regime,
        'is_trending': is_trending,
        'trend_direction': trend_direction,
        'adx_strength': adx_strength,
        'adx_value': adx
    }

# ════════════════════════════════════════════════════════════════════════════
# 6. DYNAMIC TP/SL CALCULATOR
# ════════════════════════════════════════════════════════════════════════════

def calculate_dynamic_tp_sl(current_price: float, atr: float, 
                            signal: str, config: dict) -> dict:
    """
    Calculate dynamic TP/SL based on ATR.
    
    Returns:
        {
            'tp_price': float,
            'sl_price': float,
            'tp_distance': float,
            'sl_distance': float,
            'reward_risk': float,
            'profit_locks': list
        }
    """
    atr_sl = config.get('atr_multiplier_sl', 4.0)
    atr_tp = config.get('atr_multiplier_tp', 20.0)
    
    sl_distance = atr * atr_sl
    tp_distance = atr * atr_tp
    
    if signal == "BUY":
        tp_price = current_price + tp_distance
        sl_price = current_price - sl_distance
    elif signal == "SELL":
        tp_price = current_price - tp_distance
        sl_price = current_price + sl_distance
    else:
        # NEUTRAL
        tp_price = current_price
        sl_price = current_price
    
    reward_risk = tp_distance / sl_distance if sl_distance > 0 else 0
    
    # Calculate profit lock levels
    profit_locks = []
    for profit_threshold, lock_level in config.get('profit_lock_levels', []):
        if signal == "BUY":
            lock_price = current_price + (current_price * lock_level / 100)
        elif signal == "SELL":
            lock_price = current_price - (current_price * lock_level / 100)
        else:
            lock_price = current_price
        
        profit_locks.append({
            'threshold_pct': profit_threshold,
            'lock_pct': lock_level,
            'lock_price': lock_price
        })
    
    return {
        'tp_price': tp_price,
        'sl_price': sl_price,
        'tp_distance': tp_distance,
        'sl_distance': sl_distance,
        'reward_risk': reward_risk,
        'profit_locks': profit_locks
    }

# ════════════════════════════════════════════════════════════════════════════
# 7. STREAMLIT UI SETUP
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Monster Bot v13 - Live Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { 
        background-color: #1c2128; 
        border: 1px solid #30363d; 
        border-radius: 10px; 
        padding: 15px; 
    }
    .signal-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    .regime-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 8. LOAD MODEL & ASSETS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_and_assets(config):
    """Load PyTorch model and feature columns."""
    
    model_path = Path(config['model_path'])
    
    if not model_path.exists():
        st.error(f"❌ Model not found: {model_path}")
        st.stop()
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get model config
        model_config = checkpoint.get('config', {
            'input_dim': 30,
            'hidden_dim': 128,
            'num_lstm_layers': 2,
            'num_transformer_layers': 2,
            'num_heads': 4,
            'se_reduction_ratio': 16,
            'dropout': 0.35,
            'num_classes': 3,
            'use_positional_encoding': True,
        })
        
        # Get feature columns
        feature_cols = checkpoint.get('feature_cols', [])
        
        # Build model
        model = HybridTransformerLSTM(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"✅ Model loaded: {len(feature_cols)} features")
        
        return model, feature_cols, model_config
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# ════════════════════════════════════════════════════════════════════════════
# 9. EXCHANGE CONNECTION
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_exchange(exchange_name: str):
    """Initialize CCXT exchange."""
    try:
        if exchange_name.lower() == 'binance':
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}  # For futures
            })
        elif exchange_name.lower() == 'kraken':
            exchange = ccxt.kraken({'enableRateLimit': True})
        else:
            exchange = ccxt.binance({'enableRateLimit': True})
        
        logger.info(f"✅ Connected to {exchange_name}")
        return exchange
        
    except Exception as e:
        st.error(f"❌ Exchange connection failed: {e}")
        st.stop()

# ════════════════════════════════════════════════════════════════════════════
# 10. MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════

def main():
    # Sidebar
    st.sidebar.title("🤖 Monster Bot v13")
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Model:** HybridTransformerLSTM  
    **Version:** v13 TITAN  
    **Status:** 🟢 Live
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    
    # User settings
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 
                                    float(LIVE_CONFIG['temperature']), 0.1)
    entry_percentile = st.sidebar.slider("Entry Percentile", 70.0, 95.0, 
                                        float(LIVE_CONFIG['entry_percentile']), 1.0)
    refresh_interval = st.sidebar.number_input("Refresh (sec)", 10, 300, 
                                              LIVE_CONFIG['refresh_interval'])
    
    # Update config
    LIVE_CONFIG['temperature'] = temperature
    LIVE_CONFIG['entry_percentile'] = entry_percentile
    LIVE_CONFIG['refresh_interval'] = refresh_interval
    
    # Load model
    model, feature_cols, model_config = load_model_and_assets(LIVE_CONFIG)
    exchange = get_exchange(LIVE_CONFIG['exchange'])
    
    # Layout
    col_chart, col_signal = st.columns([1.3, 1])
    
    # TradingView Chart
    with col_chart:
        st.markdown("### 📊 Market View")
        tv_html = f"""
        <div style="height:600px;">
            <div id="tv_chart" style="height:100%;"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({{
                "autosize": true,
                "symbol": "BINANCE:BTCUSDT",
                "interval": "15",
                "timezone": "Asia/Ho_Chi_Minh",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#0e1117",
                "enable_publishing": false,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "container_id": "tv_chart"
            }});
            </script>
        </div>
        """
        components.html(tv_html, height=650)
    
    # Signal Panel
    with col_signal:
        st.markdown("### 🤖 AI Prediction")
        signal_container = st.empty()
        metrics_container = st.empty()
        regime_container = st.empty()
        status_container = st.empty()
    
    # Main loop
    last_update = 0
    
    while True:
        current_time = time.time()
        
        # Check if should update
        if current_time - last_update < refresh_interval:
            time.sleep(1)
            continue
        
        try:
            status_container.caption(f"⏳ Fetching data...")
            
            # Fetch OHLCV
            ohlcv = exchange.fetch_ohlcv(
                LIVE_CONFIG['symbol'],
                timeframe=LIVE_CONFIG['timeframe'],
                limit=LIVE_CONFIG['limit']
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Feature engineering
            df_enriched = enrich_features_v13(df, LIVE_CONFIG)
            
            # Check minimum data
            if len(df_enriched) < LIVE_CONFIG['sequence_length']:
                status_container.warning(f"⚠️ Not enough data: {len(df_enriched)} < {LIVE_CONFIG['sequence_length']}")
                time.sleep(10)
                continue
            
            # Apply rolling normalization
            df_normalized = apply_rolling_normalization(df_enriched, feature_cols, LIVE_CONFIG)
            
            # Prepare input
            X_raw = df_normalized[feature_cols].tail(LIVE_CONFIG['sequence_length']).values
            X_tensor = torch.FloatTensor(X_raw).unsqueeze(0)
            
            # Predict with temperature scaling
            with torch.no_grad():
                logits = model(X_tensor)
                probs = apply_temperature_scaling(logits, temperature)
                probs_np = probs.cpu().numpy()[0]
            
            # Calibrate thresholds (adaptive)
            # In production, these should be calibrated on historical data
            # For now, use fixed percentile-based thresholds
            buy_threshold = 0.50  # Simplified for live demo
            sell_threshold = 0.50
            
            # Determine signal
            signal, confidence, dominant_class = determine_signal(
                probs_np, buy_threshold, sell_threshold
            )
            
            # Get current market data
            current_price = df['Close'].iloc[-1]
            current_atr = df_enriched['ATR'].iloc[-1] if 'ATR' in df_enriched.columns else current_price * 0.01
            current_adx = df_enriched['ADX'].iloc[-1] if 'ADX' in df_enriched.columns else 25.0
            current_sma_dist = df_enriched['SMA_distance'].iloc[-1] if 'SMA_distance' in df_enriched.columns else 0.0
            
            # Market regime
            regime_info = detect_market_regime(current_adx, current_sma_dist, LIVE_CONFIG)
            
            # Calculate TP/SL
            tp_sl_info = calculate_dynamic_tp_sl(current_price, current_atr, signal, LIVE_CONFIG)
            
            # ═══════════════════════════════════════════════════════════════
            # UPDATE UI
            # ═══════════════════════════════════════════════════════════════
            
            # Signal colors
            if signal == "BUY":
                if confidence > 0.70:
                    color, icon, label = "#00ff88", "🔥", "STRONG BUY"
                else:
                    color, icon, label = "#2ecc71", "📈", "BUY"
            elif signal == "SELL":
                if confidence > 0.70:
                    color, icon, label = "#ff4b4b", "💀", "STRONG SELL"
                else:
                    color, icon, label = "#e74c3c", "📉", "SELL"
            else:
                color, icon, label = "#f1c40f", "⚖️", "NEUTRAL"
            
            # Signal Box
            with signal_container.container():
                st.markdown(f"""
                    <div class="signal-box" style="background-color:{color}15; border: 2px solid {color};">
                        <h1 style="color:{color}; margin:0; font-size: 40px;">{icon} {label}</h1>
                        <h2 style="color:white; margin:10px 0;">BTC: ${current_price:,.2f}</h2>
                        <p style="color:{color}; font-weight:bold; font-size: 18px;">
                            Confidence: {confidence:.1%}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Probabilities
                st.markdown("**Probability Distribution:**")
                prob_cols = st.columns(3)
                prob_cols[0].metric("Neutral", f"{probs_np[0]:.1%}")
                prob_cols[1].metric("Buy", f"{probs_np[1]:.1%}", 
                                   delta=None if probs_np[1] < 0.5 else "Strong")
                prob_cols[2].metric("Sell", f"{probs_np[2]:.1%}",
                                   delta=None if probs_np[2] < 0.5 else "Strong")
            
            # TP/SL Metrics
            with metrics_container.container():
                st.markdown("---")
                st.markdown("**Dynamic TP/SL (ATR-based):**")
                
                tp_sl_cols = st.columns(3)
                tp_sl_cols[0].metric("Take Profit", f"${tp_sl_info['tp_price']:,.2f}",
                                    delta=f"+{tp_sl_info['tp_distance']:.2f}")
                tp_sl_cols[1].metric("Stop Loss", f"${tp_sl_info['sl_price']:,.2f}",
                                    delta=f"-{tp_sl_info['sl_distance']:.2f}")
                tp_sl_cols[2].metric("R:R Ratio", f"{tp_sl_info['reward_risk']:.2f}x")
                
                # Profit Locks
                if tp_sl_info['profit_locks']:
                    st.markdown("**Profit Lock Levels:**")
                    for lock in tp_sl_info['profit_locks']:
                        st.caption(f"• At +{lock['threshold_pct']:.1f}% → Lock {lock['lock_pct']:.1f}% (${lock['lock_price']:,.2f})")
            
            # Market Regime
            with regime_container.container():
                st.markdown("---")
                st.markdown("**Market Regime:**")
                
                # Regime badge
                regime_color = {
                    'TRENDING_UP': '#2ecc71',
                    'TRENDING_DOWN': '#e74c3c',
                    'RANGING': '#f1c40f',
                    'VOLATILE': '#9b59b6'
                }.get(regime_info['regime'], '#95a5a6')
                
                st.markdown(f"""
                    <span class="regime-badge" style="background-color:{regime_color}20; color:{regime_color}; border: 1px solid {regime_color};">
                        {regime_info['regime']}
                    </span>
                    <span class="regime-badge" style="background-color:#30363d; color:#8b949e;">
                        ADX: {regime_info['adx_value']:.1f} ({regime_info['adx_strength']})
                    </span>
                """, unsafe_allow_html=True)
                
                regime_cols = st.columns(2)
                regime_cols[0].caption(f"Trending: {'✅' if regime_info['is_trending'] else '❌'}")
                regime_cols[1].caption(f"Direction: {regime_info['trend_direction']}")
                
                # Trading recommendation
                if signal != "NEUTRAL":
                    if regime_info['is_trending']:
                        st.success("✅ Good conditions for trend-following trade")
                    else:
                        st.warning("⚠️ Ranging market - use tight stops")
            
            # Status
            now = datetime.now()
            status_container.caption(f"⏱️ Last update: {now.strftime('%H:%M:%S')}")
            
            last_update = current_time
            
        except Exception as e:
            status_container.error(f"❌ Error: {e}")
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(10)
        
        time.sleep(1)

# ════════════════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()