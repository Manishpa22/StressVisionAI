"""
StressVision AI — Create Model & Export to TF.js Compatible Format
Builds the 1D CNN, trains on synthetic data, exports as SavedModel + H5,
and creates TF.js-compatible JSON weights manually (no tensorflowjs pip needed).
"""

import os
import sys
import json
import struct
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    Dropout, Dense, Input, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam

# Match notebook exactly
WINDOW_SIZE = 3500  # 5 sec * 700 Hz
NUM_CHANNELS = 8    # ECG, EDA, EMG, Resp, Temp, ACCx, ACCy, ACCz


def build_model():
    """Exact 1D CNN architecture from the notebook."""
    model = Sequential([
        Input(shape=(WINDOW_SIZE, NUM_CHANNELS)),
        
        Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def generate_synthetic_data(n_samples=200):
    """Generate synthetic training data that mimics stressed / not-stressed patterns."""
    np.random.seed(42)
    X, y = [], []
    
    for i in range(n_samples):
        is_stressed = i >= n_samples // 2
        window = np.zeros((WINDOW_SIZE, NUM_CHANNELS), dtype=np.float32)
        t = np.arange(WINDOW_SIZE) / 700.0
        
        if is_stressed:
            hr = np.random.uniform(100, 160)
            eda_base = np.random.uniform(0.3, 0.8)
            emg_amp = np.random.uniform(0.4, 0.9)
            resp_rate = np.random.uniform(22, 35)
            temp_val = np.random.uniform(0.2, 0.6)
            acc_amp = np.random.uniform(0.3, 0.7)
        else:
            hr = np.random.uniform(55, 85)
            eda_base = np.random.uniform(-0.5, 0.1)
            emg_amp = np.random.uniform(0.0, 0.2)
            resp_rate = np.random.uniform(12, 18)
            temp_val = np.random.uniform(-0.3, 0.1)
            acc_amp = np.random.uniform(0.0, 0.15)
        
        freq = hr / 60
        phase = t * freq * 2 * np.pi
        window[:, 0] = np.sin(phase) * 0.5 + np.random.normal(0, 0.05, WINDOW_SIZE)
        window[:, 1] = eda_base + 0.1 * np.sin(t * 0.5) + np.random.normal(0, 0.02, WINDOW_SIZE)
        window[:, 2] = emg_amp * np.random.normal(0, 0.3, WINDOW_SIZE)
        window[:, 3] = 0.5 * np.sin(t * resp_rate / 60 * 2 * np.pi) + np.random.normal(0, 0.03, WINDOW_SIZE)
        window[:, 4] = temp_val + np.random.normal(0, 0.01, WINDOW_SIZE)
        window[:, 5] = acc_amp * np.random.normal(0, 0.2, WINDOW_SIZE)
        window[:, 6] = acc_amp * np.random.normal(0, 0.2, WINDOW_SIZE)
        window[:, 7] = acc_amp * np.random.normal(0, 0.2, WINDOW_SIZE)
        
        for ch in range(NUM_CHANNELS):
            mean = window[:, ch].mean()
            std = window[:, ch].std()
            if std > 0:
                window[:, ch] = (window[:, ch] - mean) / std
        
        X.append(window)
        y.append(1.0 if is_stressed else 0.0)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def export_tfjs_manual(model, output_dir):
    """
    Manually export Keras model to TF.js Layers format.
    Creates model.json + group1-shardN.bin files compatible with tf.loadLayersModel().
    """
    os.makedirs(output_dir, exist_ok=True)
    
    weights = model.get_weights()
    weight_specs = []
    weight_data = b''
    
    layer_index = 0
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue
        for i, w in enumerate(layer_weights):
            w_flat = w.flatten().astype(np.float32)
            weight_data += w_flat.tobytes()
            
            name = f"{layer.name}/{['kernel', 'bias', 'gamma', 'beta', 'moving_mean', 'moving_variance'][i] if i < 6 else f'weight_{i}'}"
            
            weight_specs.append({
                'name': name,
                'shape': list(w.shape),
                'dtype': 'float32'
            })
    
    # Write binary weights
    shard_path = os.path.join(output_dir, 'group1-shard1of1.bin')
    with open(shard_path, 'wb') as f:
        f.write(weight_data)
    
    # Build model topology
    model_config = json.loads(model.to_json())
    
    # Build model.json
    model_json = {
        'format': 'layers-model',
        'generatedBy': 'keras v' + tf.keras.__version__,
        'convertedBy': 'StressVision AI Manual Exporter',
        'modelTopology': model_config,
        'weightsManifest': [{
            'paths': ['group1-shard1of1.bin'],
            'weights': weight_specs
        }]
    }
    
    json_path = os.path.join(output_dir, 'model.json')
    with open(json_path, 'w') as f:
        json.dump(model_json, f)
    
    total_mb = len(weight_data) / (1024 * 1024)
    print(f"  model.json: {os.path.getsize(json_path)} bytes")
    print(f"  group1-shard1of1.bin: {total_mb:.2f} MB")
    print(f"  Total weights: {len(weight_specs)} tensors")


def main():
    print("=" * 60)
    print("StressVision AI — Model Creation & TF.js Export")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Build
    print("\n[1/4] Building 1D CNN architecture...")
    model = build_model()
    model.summary()
    
    # 2. Generate & Train
    print("\n[2/4] Generating synthetic data & training...")
    X, y = generate_synthetic_data(200)
    print(f"  Data: X={X.shape}, y={y.shape}")
    print(f"  Stressed: {int(y.sum())}, Not Stressed: {int(len(y) - y.sum())}")
    
    model.fit(X, y, epochs=15, batch_size=16, validation_split=0.2, verbose=1)
    
    # 3. Save H5
    h5_path = os.path.join(base_dir, 'stress_model.h5')
    print(f"\n[3/4] Saving Keras model to: {h5_path}")
    model.save(h5_path)
    
    # 4. Export TF.js
    tfjs_dir = os.path.join(base_dir, 'tfjs_model')
    print(f"\n[4/4] Exporting TF.js model to: {tfjs_dir}/")
    export_tfjs_manual(model, tfjs_dir)
    
    # Also save scaler params (identity for synthetic data)
    scaler_params = {
        'mean': [0.0] * NUM_CHANNELS,
        'scale': [1.0] * NUM_CHANNELS,
        'channels': ['ECG', 'EDA', 'EMG', 'Resp', 'Temp', 'ACC_x', 'ACC_y', 'ACC_z']
    }
    scaler_path = os.path.join(base_dir, 'scaler_params.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"\n  Scaler params saved to: {scaler_path}")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Files created:")
    print(f"  1. {h5_path}")
    print(f"  2. {tfjs_dir}/model.json")
    print(f"  3. {tfjs_dir}/group1-shard1of1.bin")
    print(f"  4. {scaler_path}")
    print("\nThe web app will now use these for real TF.js inference!")
    print("=" * 60)


if __name__ == '__main__':
    main()
