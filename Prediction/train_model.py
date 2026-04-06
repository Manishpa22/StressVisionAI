"""
StressVision AI — Full Training Script with WESAD Dataset
Replicates the exact training pipeline from the Jupyter notebook.

Usage:
  1. Download WESAD dataset and place in ./WESAD/ folder
  2. pip install tensorflow tensorflowjs scikit-learn numpy
  3. python train_model.py

The WESAD dataset should have folders: S2, S3, S4, ..., S17
Each containing a .pkl file with sensor data.
"""

import os
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    Dropout, Dense, Input, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ===== Configuration =====
WESAD_PATH = os.path.join(os.path.dirname(__file__), 'WESAD')
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
FS = 700
WINDOW_SECONDS = 5
WINDOW_SIZE = FS * WINDOW_SECONDS  # 3500

# Channel names matching the notebook
CHANNELS = ['ECG', 'EDA', 'EMG', 'Resp', 'Temp', 'ACC_x', 'ACC_y', 'ACC_z']
NUM_CHANNELS = len(CHANNELS)


def load_subject_data(subject_id):
    """Load data for a single subject from WESAD .pkl file."""
    pkl_path = os.path.join(WESAD_PATH, f'S{subject_id}', f'S{subject_id}.pkl')
    
    if not os.path.exists(pkl_path):
        print(f"  Warning: {pkl_path} not found, skipping")
        return None, None
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Extract chest sensor data (700Hz sampling rate)
    chest = data['signal']['chest']
    labels = data['label']
    
    # Combine channels: ECG, EDA, EMG, Resp, Temp, ACC(x,y,z)
    ecg = chest['ECG'].flatten()
    eda = chest['EDA'].flatten()
    emg = chest['EMG'].flatten()
    resp = chest['Resp'].flatten()
    temp = chest['Temp'].flatten()
    acc = chest['ACC']  # shape: (n, 3)
    
    # Ensure matching lengths
    min_len = min(len(ecg), len(eda), len(emg), len(resp), len(temp), len(acc))
    
    signals = np.column_stack([
        ecg[:min_len],
        eda[:min_len],
        emg[:min_len],
        resp[:min_len],
        temp[:min_len],
        acc[:min_len, 0],
        acc[:min_len, 1],
        acc[:min_len, 2]
    ])
    
    labels = labels[:min_len].flatten()
    
    return signals, labels


def create_windows(signals, labels, window_size=WINDOW_SIZE):
    """Create non-overlapping windows from continuous signal data."""
    n_windows = len(signals) // window_size
    X_windows = []
    y_windows = []
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        
        window_signals = signals[start:end]
        window_labels = labels[start:end]
        
        # Use majority label for the window
        label_counts = np.bincount(window_labels.astype(int), minlength=5)
        majority_label = np.argmax(label_counts)
        
        # Skip transient/undefined labels (0)
        if majority_label == 0:
            continue
        
        # Binary mapping: label 2 = Stressed (1), labels 1,3,4 = Not Stressed (0)
        binary_label = 1.0 if majority_label == 2 else 0.0
        
        X_windows.append(window_signals)
        y_windows.append(binary_label)
    
    return X_windows, y_windows


def build_model():
    """Build the exact 1D CNN architecture from the notebook."""
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


def export_to_tfjs(h5_path):
    """Convert saved Keras model to TensorFlow.js format."""
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("Installing tensorflowjs...")
        os.system('pip install tensorflowjs')
        import tensorflowjs as tfjs
    
    output_dir = os.path.join(os.path.dirname(__file__), 'tfjs_model')
    os.makedirs(output_dir, exist_ok=True)
    
    model = tf.keras.models.load_model(h5_path)
    tfjs.converters.save_keras_model(model, output_dir)
    print(f"TF.js model saved to: {output_dir}/")


def save_scaler_params(scaler):
    """Save StandardScaler parameters for use in the web app."""
    params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    import json
    params_path = os.path.join(os.path.dirname(__file__), 'scaler_params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Scaler parameters saved to: {params_path}")


def main():
    print("=" * 60)
    print("StressVision AI — Full WESAD Training Pipeline")
    print("=" * 60)
    
    if not os.path.exists(WESAD_PATH):
        print(f"\nERROR: WESAD dataset not found at: {WESAD_PATH}")
        print("Please download the WESAD dataset and place it in the WESAD/ folder.")
        print("Download from: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Un")
        return
    
    # 1. Load all subjects
    print("\n[1/5] Loading WESAD data...")
    all_X = []
    all_y = []
    
    for sid in SUBJECTS:
        print(f"  Loading Subject S{sid}...", end=' ')
        signals, labels = load_subject_data(sid)
        if signals is None:
            continue
        
        X_win, y_win = create_windows(signals, labels)
        all_X.extend(X_win)
        all_y.extend(y_win)
        print(f"{len(X_win)} windows")
    
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    
    print(f"\nTotal: {len(X)} windows")
    print(f"Stressed: {int(y.sum())}, Not Stressed: {int(len(y) - y.sum())}")
    
    # 2. Normalize
    print("\n[2/5] Normalizing with StandardScaler...")
    original_shape = X.shape
    X_flat = X.reshape(-1, NUM_CHANNELS)
    
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(original_shape)
    
    save_scaler_params(scaler)
    
    # 3. Split
    print("\n[3/5] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 4. Train
    print("\n[4/5] Training model...")
    model = build_model()
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    
    # 5. Save & Export
    print("\n[5/5] Saving model...")
    h5_path = os.path.join(os.path.dirname(__file__), 'stress_model.h5')
    model.save(h5_path)
    print(f"Keras model saved to: {h5_path}")
    
    export_to_tfjs(h5_path)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("The web app will now use the trained model for inference.")
    print("=" * 60)


if __name__ == '__main__':
    main()
