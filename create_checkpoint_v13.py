"""
╔══════════════════════════════════════════════════════════════════════════╗
║  CREATE CHECKPOINT - Convert trained model to dashboard-ready format   ║
╚══════════════════════════════════════════════════════════════════════════╝

Use this script to create a proper checkpoint from your trained v13 model.

Input:  Trained model (either .pt or in-memory)
Output: BTC-USDT_MONSTER_model.pt (ready for dashboard)
"""

import torch
import torch.nn as nn
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (Same as training)
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
        
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        if config.get('use_positional_encoding', True):
            self.pos_encoder = PositionalEncoding(self.hidden_dim)
        else:
            self.pos_encoder = None
        
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
        
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=config['num_lstm_layers'],
            batch_first=True,
            dropout=config['dropout'] if config['num_lstm_layers'] > 1 else 0
        )
        
        self.se_block = SEBlock(self.hidden_dim, config['se_reduction_ratio'])
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.input_projection(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x, _ = self.lstm(x)
        x = self.se_block(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMNS (v13 Standard)
# ════════════════════════════════════════════════════════════════════════════

FEATURE_COLS_V13 = [
    # Basic
    'log_return',
    'ATR',
    'BB_width',
    'BB_position',
    
    # Fractional Diff
    'frac_diff_close',
    
    # Fourier (5 components x 2)
    'fourier_sin_1', 'fourier_cos_1',
    'fourier_sin_2', 'fourier_cos_2',
    'fourier_sin_3', 'fourier_cos_3',
    'fourier_sin_4', 'fourier_cos_4',
    'fourier_sin_5', 'fourier_cos_5',
    
    # Volatility-adjusted
    'RSI_vol_adj',
    'ROC_vol_adj',
    
    # Microstructure
    'volume_imbalance',
    'entropy',
    'volume_ratio',
    
    # Market Regime
    'ADX',
    'SMA_distance',
    'regime_trending',
    'regime_uptrend',
    'regime_downtrend',
    
    # Traditional
    'RSI',
    'MACD',
    'MACD_signal',
    'volatility_zscore'
]

# ════════════════════════════════════════════════════════════════════════════
# METHOD 1: From Existing Checkpoint (with missing fields)
# ════════════════════════════════════════════════════════════════════════════

def upgrade_checkpoint(old_checkpoint_path, output_path):
    """
    Upgrade old checkpoint to dashboard-ready format.
    
    Args:
        old_checkpoint_path: Path to old .pt file
        output_path: Path to save new checkpoint
    """
    print("📦 Loading old checkpoint...")
    checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    
    # Check what's inside
    print(f"   Keys: {list(checkpoint.keys())}")
    
    # Add missing fields
    if 'config' not in checkpoint:
        print("⚠️  Adding default config...")
        checkpoint['config'] = {
            'input_dim': 30,
            'hidden_dim': 128,
            'num_lstm_layers': 2,
            'num_transformer_layers': 2,
            'num_heads': 4,
            'se_reduction_ratio': 16,
            'dropout': 0.35,
            'num_classes': 3,
            'use_positional_encoding': True,
        }
    
    if 'feature_cols' not in checkpoint:
        print("⚠️  Adding feature columns...")
        checkpoint['feature_cols'] = FEATURE_COLS_V13
    
    # Save upgraded checkpoint
    print(f"💾 Saving to {output_path}...")
    torch.save(checkpoint, output_path)
    
    print("✅ Done! Checkpoint ready for dashboard.")
    print(f"   Features: {len(checkpoint['feature_cols'])}")
    print(f"   Config: {checkpoint['config']['input_dim']}D input")

# ════════════════════════════════════════════════════════════════════════════
# METHOD 2: From Trained Model Object
# ════════════════════════════════════════════════════════════════════════════

def create_checkpoint_from_model(model, feature_cols, output_path):
    """
    Create checkpoint from trained model instance.
    
    Args:
        model: Trained HybridTransformerLSTM instance
        feature_cols: List of feature names
        output_path: Where to save checkpoint
    """
    print("📦 Creating checkpoint from model...")
    
    # Extract config from model
    config = {
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'num_classes': model.num_classes,
        'num_lstm_layers': 2,  # Update if different
        'num_transformer_layers': 2,  # Update if different
        'num_heads': 4,  # Update if different
        'se_reduction_ratio': 16,
        'dropout': 0.35,
        'use_positional_encoding': model.pos_encoder is not None,
    }
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'feature_cols': feature_cols
    }
    
    # Save
    print(f"💾 Saving to {output_path}...")
    torch.save(checkpoint, output_path)
    
    print("✅ Done! Checkpoint ready for dashboard.")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")

# ════════════════════════════════════════════════════════════════════════════
# METHOD 3: From Scratch (for testing)
# ════════════════════════════════════════════════════════════════════════════

def create_dummy_checkpoint(output_path):
    """
    Create a dummy checkpoint for testing dashboard (untrained model).
    
    WARNING: This model is NOT trained! Only for UI testing.
    """
    print("📦 Creating dummy checkpoint (untrained)...")
    
    config = {
        'input_dim': 30,
        'hidden_dim': 128,
        'num_lstm_layers': 2,
        'num_transformer_layers': 2,
        'num_heads': 4,
        'se_reduction_ratio': 16,
        'dropout': 0.35,
        'num_classes': 3,
        'use_positional_encoding': True,
    }
    
    # Create model
    model = HybridTransformerLSTM(config)
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'feature_cols': FEATURE_COLS_V13
    }
    
    # Save
    print(f"💾 Saving to {output_path}...")
    torch.save(checkpoint, output_path)
    
    print("✅ Done! Dummy checkpoint created.")
    print("⚠️  WARNING: This is an UNTRAINED model! For testing only.")

# ════════════════════════════════════════════════════════════════════════════
# VERIFY CHECKPOINT
# ════════════════════════════════════════════════════════════════════════════

def verify_checkpoint(checkpoint_path):
    """
    Verify checkpoint has all required fields.
    """
    print(f"🔍 Verifying checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check required keys
    required_keys = ['model_state_dict', 'config', 'feature_cols']
    missing = [k for k in required_keys if k not in checkpoint]
    
    if missing:
        print(f"❌ Missing keys: {missing}")
        return False
    
    print("✅ All required keys present")
    
    # Check config
    config = checkpoint['config']
    print(f"\n📊 Config:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # Check features
    features = checkpoint['feature_cols']
    print(f"\n📋 Features ({len(features)}):")
    print(f"   {features[:5]}...")
    print(f"   ...")
    print(f"   {features[-5:]}")
    
    # Try loading model
    try:
        model = HybridTransformerLSTM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✅ Model loads successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"\n❌ Model load failed: {e}")
        return False
    
    print("\n✅ Checkpoint is valid!")
    return True

# ════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("CHECKPOINT CREATOR - Monster Bot v13")
    print("="*80)
    
    # Example 1: Upgrade old checkpoint
    print("\n1️⃣  Upgrade old checkpoint:")
    print("   upgrade_checkpoint('old_model.pt', 'models/BTC-USDT_MONSTER_model.pt')")
    
    # Example 2: Create from model
    print("\n2️⃣  Create from trained model:")
    print("   create_checkpoint_from_model(model, FEATURE_COLS_V13, 'models/BTC-USDT_MONSTER_model.pt')")
    
    # Example 3: Create dummy (testing)
    print("\n3️⃣  Create dummy for testing:")
    print("   create_dummy_checkpoint('models/BTC-USDT_MONSTER_model.pt')")
    
    # Example 4: Verify
    print("\n4️⃣  Verify checkpoint:")
    print("   verify_checkpoint('models/BTC-USDT_MONSTER_model.pt')")
    
    print("\n" + "="*80)
    
    # Interactive mode
    if len(sys.argv) > 1:
        action = sys.argv[1]
        
        if action == "upgrade":
            if len(sys.argv) < 4:
                print("Usage: python create_checkpoint.py upgrade <input.pt> <output.pt>")
            else:
                upgrade_checkpoint(sys.argv[2], sys.argv[3])
        
        elif action == "dummy":
            if len(sys.argv) < 3:
                print("Usage: python create_checkpoint.py dummy <output.pt>")
            else:
                create_dummy_checkpoint(sys.argv[2])
        
        elif action == "verify":
            if len(sys.argv) < 3:
                print("Usage: python create_checkpoint.py verify <checkpoint.pt>")
            else:
                verify_checkpoint(sys.argv[2])
        
        else:
            print(f"Unknown action: {action}")
            print("Available: upgrade, dummy, verify")
    
    else:
        print("\nUsage:")
        print("  python create_checkpoint.py upgrade old_model.pt new_model.pt")
        print("  python create_checkpoint.py dummy test_model.pt")
        print("  python create_checkpoint.py verify model.pt")

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  QUICK COMMANDS                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

# Upgrade existing checkpoint
python create_checkpoint.py upgrade old.pt models/BTC-USDT_MONSTER_model.pt

# Create dummy for testing
python create_checkpoint.py dummy models/BTC-USDT_MONSTER_model.pt

# Verify checkpoint
python create_checkpoint.py verify models/BTC-USDT_MONSTER_model.pt

"""
