# ⚡ QUICK START - Live Dashboard v13

## 📦 BẠN CHỈ CẦN 1 FILE DUY NHẤT!

```
✅ BTC-USDT_MONSTER_model.pt  (Checkpoint PyTorch)
❌ KHÔNG CẦN file .txt riêng!
❌ KHÔNG CẦN scaler .pkl!
```

---

## 🎯 CẤU TRÚC CHECKPOINT

File `.pt` chứa **ĐẦY ĐỦ** mọi thứ:

```python
checkpoint = {
    'model_state_dict': {...},      # Model weights
    'config': {                     # Architecture config
        'input_dim': 30,
        'hidden_dim': 128,
        'num_lstm_layers': 2,
        'num_transformer_layers': 2,
        'num_heads': 4,
        'se_reduction_ratio': 16,
        'dropout': 0.35,
        'num_classes': 3,
        'use_positional_encoding': True,
    },
    'feature_cols': [               # ← Feature names (BUILT-IN!)
        'log_return',
        'ATR',
        'BB_width',
        'frac_diff_close',
        'fourier_sin_1',
        # ... (30 features total)
    ]
}
```

**Dashboard tự động đọc `feature_cols` từ checkpoint!**

---

## 🚀 SETUP (3 BƯỚC)

### **1. Cài packages:**
```bash
pip install streamlit torch ccxt pandas numpy scipy
```

### **2. Đặt model vào đúng chỗ:**
```
project/
├── live_trading_dashboard_v13_TITAN.py
└── models/
    └── BTC-USDT_MONSTER_model.pt  ← File này THÔI!
```

### **3. Chạy:**
```bash
streamlit run live_trading_dashboard_v13_TITAN.py
```

Mở browser: `http://localhost:8501`

---

## ⚙️ NẾUCHƯA CÓ CHECKPOINT ĐÚNG FORMAT

### **Từ model đã train (v13):**

```python
import torch

# Giả sử bạn có:
# - model (HybridTransformerLSTM instance)
# - CONFIG (dict với các settings)
# - feature_cols (list các tên feature)

# Save checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'input_dim': len(feature_cols),
        'hidden_dim': 128,
        'num_lstm_layers': 2,
        'num_transformer_layers': 2,
        'num_heads': 4,
        'se_reduction_ratio': 16,
        'dropout': 0.35,
        'num_classes': 3,
        'use_positional_encoding': True,
    },
    'feature_cols': feature_cols  # List: ['log_return', 'ATR', ...]
}

torch.save(checkpoint, 'models/BTC-USDT_MONSTER_model.pt')
```

### **Từ training script v13:**

Nếu bạn đang dùng `live_trading_bot_v13.py`, trong phần training có đoạn:

```python
# Save best model
best_checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'config': CONFIG,            # ← Config tự động có
    'feature_cols': feature_cols # ← Features tự động có
}
torch.save(best_checkpoint, f'{model_path}/best_model.pt')
```

Chỉ cần copy file `best_model.pt` này sang `models/BTC-USDT_MONSTER_model.pt`!

---

## 🔍 VERIFY CHECKPOINT

### **Kiểm tra checkpoint có đúng format không:**

```python
import torch

checkpoint = torch.load('models/BTC-USDT_MONSTER_model.pt', map_location='cpu')

print("✅ Checkpoint keys:")
print(checkpoint.keys())
# Expected: dict_keys(['model_state_dict', 'config', 'feature_cols', ...])

print("\n✅ Config:")
print(checkpoint['config'])
# Expected: {'input_dim': 30, 'hidden_dim': 128, ...}

print("\n✅ Features:")
print(len(checkpoint['feature_cols']), "features")
print(checkpoint['feature_cols'][:5])  # First 5
# Expected: ['log_return', 'ATR', 'BB_width', ...]
```

**Nếu thiếu `feature_cols`:**

```python
# Fix bằng cách thêm vào
checkpoint = torch.load('old_model.pt')

# Thêm feature_cols (list 30 features của bạn)
checkpoint['feature_cols'] = [
    'log_return', 'ATR', 'BB_width', 'BB_position',
    'frac_diff_close',
    'fourier_sin_1', 'fourier_cos_1',
    'fourier_sin_2', 'fourier_cos_2',
    'fourier_sin_3', 'fourier_cos_3',
    'fourier_sin_4', 'fourier_cos_4',
    'fourier_sin_5', 'fourier_cos_5',
    'RSI_vol_adj', 'ROC_vol_adj',
    'volume_imbalance', 'entropy',
    'volume_ratio',
    'ADX', 'SMA_distance',
    'regime_trending', 'regime_uptrend', 'regime_downtrend',
    'RSI', 'MACD', 'MACD_signal',
    'volatility_zscore'
]

# Save lại
torch.save(checkpoint, 'models/BTC-USDT_MONSTER_model.pt')
```

---

## ❓ FAQ

### **Q: Tôi cần file `scaler_BTC-USDT.pkl` không?**
**A:** ❌ KHÔNG! v13 dùng **Rolling Z-Score** (tính real-time), không dùng scaler cố định.

### **Q: Tôi cần file `BTC-USDT_feature_cols.txt` không?**
**A:** ❌ KHÔNG! Feature names đã nằm trong checkpoint `.pt` rồi!

### **Q: Tại sao v13 không dùng scaler?**
**A:** Global scaler gây **covariate shift** khi giá BTC thay đổi (30k→90k). Rolling Z-Score **tự adapt** theo giá hiện tại!

### **Q: File checkpoint `.pt` nặng bao nhiêu?**
**A:** Khoảng 5-10MB (model + config + features). Nhẹ hơn TensorFlow nhiều!

### **Q: Tôi có thể dùng model từ Google Drive không?**
**A:** Được! Chỉ cần update path:
```python
'model_path': '/content/drive/MyDrive/models/BTC-USDT_MONSTER_model.pt'
```

---

## 🎯 CHECKLIST

Trước khi chạy dashboard:

- [ ] ✅ Python 3.8+ installed
- [ ] ✅ Packages installed: `pip install streamlit torch ccxt pandas numpy scipy`
- [ ] ✅ Có file `BTC-USDT_MONSTER_model.pt`
- [ ] ✅ Checkpoint chứa `model_state_dict`, `config`, `feature_cols`
- [ ] ✅ Path trong `LIVE_CONFIG['model_path']` đúng
- [ ] ✅ Internet connection OK (để lấy data từ Binance)

**Run:**
```bash
streamlit run live_trading_dashboard_v13_TITAN.py
```

**Dashboard mở tại:** `http://localhost:8501` 🚀

---

## 💡 TÓM TẮT

```
┌─────────────────────────────────────────────┐
│  1 FILE .pt = Model + Config + Features    │
│  ✅ Đủ để chạy dashboard                   │
│  ✅ Không cần scaler                        │
│  ✅ Không cần file .txt riêng               │
└─────────────────────────────────────────────┘
```

**Đơn giản vậy thôi!** 🎯

---

## 📞 NEED HELP?

**Checkpoint format sai?**
→ Xem phần "VERIFY CHECKPOINT" ở trên

**Dashboard không load model?**
→ Check path: `LIVE_CONFIG['model_path']`

**Missing features?**
→ Thêm `feature_cols` vào checkpoint (xem hướng dẫn trên)

**Other issues?**
→ Xem `LIVE_DASHBOARD_v13_GUIDE.md` (detailed guide)

---

**Happy Trading!** 📈💰
