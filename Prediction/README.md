# 🧠 StressVision AI — Real-Time Stress Detection

A web-based stress detection dashboard powered by a **1D Convolutional Neural Network (CNN)** trained on the WESAD dataset. The model runs directly in the browser using **TensorFlow.js** — no backend server required.

![StressVision AI Demo](https://img.shields.io/badge/Accuracy-99%25-00f5d4?style=for-the-badge&logo=tensorflow&logoColor=white)
![Model](https://img.shields.io/badge/Model-1D%20CNN-fee440?style=for-the-badge)
![Channels](https://img.shields.io/badge/Sensors-8%20Channels-f72585?style=for-the-badge)

## ✨ Features

- **Real TensorFlow.js Inference** — Actual neural network predictions in the browser
- **8 Physiological Sensors** — ECG, EDA (Skin Conductance), EMG, Respiration, Temperature, Accelerometer (X/Y/Z)
- **Live Waveform Visualization** — Real-time Canvas API signal rendering
- **Animated Stress Gauge** — Visual 0-100% stress indicator
- **Preset Scenarios** — Quick test with Relaxed, Moderate, Stressed, or Random inputs
- **Dark Cyberpunk Theme** — Premium medical-monitor aesthetic

## 🚀 Live Demo

👉 [Open StressVision AI](https://yourusername.github.io/StressVisionAI/)

## 🏗️ Model Architecture

| Parameter | Value |
|-----------|-------|
| **Type** | 1D Convolutional Neural Network |
| **Dataset** | WESAD (Wearable Stress and Affect Detection) |
| **Input** | 8 channels × 3500 samples (5 sec @ 700Hz) |
| **Layers** | Conv1D → BatchNorm → MaxPool → Dense |
| **Parameters** | 268,865 |
| **Accuracy** | 99.09% |
| **Output** | Binary (Stressed / Not Stressed) |

## 📡 Sensor Channels

| # | Channel | Description | Unit |
|---|---------|-------------|------|
| 1 | ECG | Electrocardiogram (Heart Rate) | BPM |
| 2 | EDA | Electrodermal Activity (Skin Conductance) | µS |
| 3 | EMG | Electromyography (Muscle Activity) | % |
| 4 | Resp | Respiration Rate | BrPM |
| 5 | Temp | Body Temperature | °C |
| 6 | ACC_X | Accelerometer X-axis | g |
| 7 | ACC_Y | Accelerometer Y-axis | g |
| 8 | ACC_Z | Accelerometer Z-axis | g |

## 🛠️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/StressVisionAI.git
cd StressVisionAI

# Start a local server (needed for TF.js model loading)
python -m http.server 8080

# Open in browser
# http://localhost:8080
```

## 📁 Project Structure

```
├── index.html           # Main web application
├── index.css            # Dark theme with glassmorphism
├── app.js               # TF.js inference engine + UI logic
├── scaler_params.json   # StandardScaler normalization params
├── tfjs_model/
│   ├── model.json           # Model topology for TF.js
│   └── group1-shard1of1.bin # Model weights (1.03 MB)
└── README.md
```

## 🔬 How It Works

1. **User adjusts sensor sliders** (or selects a preset scenario)
2. **5-second data window generated** — 3500 samples across 8 channels
3. **StandardScaler normalization** applied per-channel
4. **TF.js runs forward pass** through the 1D CNN in the browser
5. **Sigmoid output** interpreted as stress probability (>50% = Stressed)
6. **Results displayed** with animated gauge and feature analysis

## 📚 References

- [WESAD Dataset](https://doi.org/10.1145/3242969.3242985) — Schmidt et al., 2018
- [TensorFlow.js](https://www.tensorflow.org/js)

## 📄 License

MIT License — Free to use, modify, and distribute.

---

*Built with ❤️ using TensorFlow, Keras, and TensorFlow.js*
