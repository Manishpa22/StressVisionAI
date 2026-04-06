// ===== StressVision AI — Real TF.js Inference Application =====

// -- State --
const state = {
    monitoring: false,
    animFrame: null,
    modelLoaded: false,
    model: null,
    sensors: {
        heartRate: 72,
        eda: 5.0,
        emg: 15,
        respRate: 16,
        temp: 36.6,
        accX: 0.0,
        accY: 0.0,
        accZ: 0.0
    },
    waveforms: {
        ecg: { data: [], color: '#00f5d4', offset: 0 },
        eda: { data: [], color: '#fee440', offset: 0 },
        emg: { data: [], color: '#f72585', offset: 0 },
        resp: { data: [], color: '#7209b7', offset: 0 }
    },
    prediction: null,
    scalerParams: null
};

// Model constants (match notebook)
const WINDOW_SIZE = 3500;
const NUM_CHANNELS = 8;
const FS = 700;

// -- Presets --
const presets = {
    relaxed: { heartRate: 65, eda: 3.0, emg: 8, respRate: 14, temp: 36.5, accX: 0.1, accY: -0.1, accZ: 0.05 },
    moderate: { heartRate: 95, eda: 12.0, emg: 40, respRate: 22, temp: 37.2, accX: 1.5, accY: 0.8, accZ: -0.5 },
    stressed: { heartRate: 145, eda: 28.0, emg: 78, respRate: 32, temp: 38.0, accX: 4.2, accY: -3.1, accZ: 2.8 }
};

// -- Initialize --
document.addEventListener('DOMContentLoaded', () => {
    initWaveforms();
    syncSlidersToState();
    loadModel();
});

// ===== TF.js Model Loading =====
async function loadModel() {
    const statusEl = document.getElementById('model-status');
    if (statusEl) {
        statusEl.textContent = 'Loading model...';
        statusEl.className = 'model-status loading';
    }

    try {
        // Load model
        state.model = await tf.loadLayersModel('./tfjs_model/model.json');
        state.modelLoaded = true;
        console.log('✅ TF.js model loaded successfully');
        console.log('   Input shape:', state.model.inputs[0].shape);
        console.log('   Output shape:', state.model.outputs[0].shape);

        if (statusEl) {
            statusEl.textContent = '✅ Model loaded — Real inference ready';
            statusEl.className = 'model-status loaded';
        }

        // Load scaler params
        try {
            const resp = await fetch('./scaler_params.json');
            state.scalerParams = await resp.json();
            console.log('✅ Scaler params loaded:', state.scalerParams.channels);
        } catch (e) {
            console.warn('⚠️ Scaler params not found, using defaults');
            state.scalerParams = {
                mean: new Array(NUM_CHANNELS).fill(0),
                scale: new Array(NUM_CHANNELS).fill(1)
            };
        }

        // Warm up with a dummy prediction
        const dummy = tf.zeros([1, WINDOW_SIZE, NUM_CHANNELS]);
        const warmup = state.model.predict(dummy);
        warmup.dispose();
        dummy.dispose();
        console.log('✅ Model warmed up');

    } catch (err) {
        console.error('❌ Failed to load TF.js model:', err);
        state.modelLoaded = false;
        if (statusEl) {
            statusEl.textContent = '⚠️ Model not found — Using simulated mode';
            statusEl.className = 'model-status fallback';
        }
    }
}

// ===== Generate 5s Window from Sensor Values =====
function generateSignalWindow() {
    const s = state.sensors;
    const t = new Float32Array(WINDOW_SIZE);
    for (let i = 0; i < WINDOW_SIZE; i++) t[i] = i / FS;

    // Create (WINDOW_SIZE, NUM_CHANNELS) array
    const window = new Float32Array(WINDOW_SIZE * NUM_CHANNELS);

    for (let i = 0; i < WINDOW_SIZE; i++) {
        const ti = t[i];

        // Channel 0: ECG — generate realistic QRS complex pattern
        const hrFreq = s.heartRate / 60;
        const phase = ti * hrFreq * 2 * Math.PI;
        const cycle = (phase % (2 * Math.PI)) / (2 * Math.PI);
        let ecgVal;
        if (cycle < 0.05) ecgVal = 0.1 * Math.sin(cycle * Math.PI * 20);
        else if (cycle < 0.08) ecgVal = -0.15;
        else if (cycle < 0.10) ecgVal = 0.9 * Math.sin((cycle - 0.08) * Math.PI * 25);
        else if (cycle < 0.14) ecgVal = -0.2 * Math.sin((cycle - 0.10) * Math.PI * 12.5);
        else if (cycle < 0.25) ecgVal = 0.15 * Math.sin((cycle - 0.14) * Math.PI * 9);
        else ecgVal = 0.02 * Math.sin(ti * 3);
        // Scale by heart rate intensity
        ecgVal *= (s.heartRate / 72);
        window[i * NUM_CHANNELS + 0] = ecgVal + (Math.random() - 0.5) * 0.05;

        // Channel 1: EDA — smooth slowly varying signal
        const edaVal = (s.eda / 40) * 0.6 + 0.15 * Math.sin(ti * 0.5) + 0.08 * Math.sin(ti * 1.3);
        window[i * NUM_CHANNELS + 1] = edaVal + (Math.random() - 0.5) * 0.03;

        // Channel 2: EMG — high frequency noise proportional to muscle tension
        const emgAmp = s.emg / 100;
        window[i * NUM_CHANNELS + 2] = emgAmp * (
            Math.sin(ti * 47) * 0.4 +
            Math.sin(ti * 93) * 0.3 +
            Math.sin(ti * 157) * 0.2 +
            (Math.random() - 0.5) * 0.3
        );

        // Channel 3: Respiration — sinusoidal breathing pattern
        const respFreq = s.respRate / 60;
        window[i * NUM_CHANNELS + 3] = 0.6 * Math.sin(ti * respFreq * 2 * Math.PI) + 
                                         0.1 * Math.sin(ti * respFreq * 4 * Math.PI) +
                                         (Math.random() - 0.5) * 0.03;

        // Channel 4: Temperature — near-constant with tiny variation
        const tempNorm = (s.temp - 36.8) / 2.0;
        window[i * NUM_CHANNELS + 4] = tempNorm + (Math.random() - 0.5) * 0.01;

        // Channel 5: Accelerometer X
        window[i * NUM_CHANNELS + 5] = (s.accX / 10) * (0.5 + (Math.random() - 0.5) * 0.4);

        // Channel 6: Accelerometer Y
        window[i * NUM_CHANNELS + 6] = (s.accY / 10) * (0.5 + (Math.random() - 0.5) * 0.4);

        // Channel 7: Accelerometer Z
        window[i * NUM_CHANNELS + 7] = (s.accZ / 10) * (0.5 + (Math.random() - 0.5) * 0.4);
    }

    // Apply StandardScaler normalization per channel (like the notebook)
    for (let ch = 0; ch < NUM_CHANNELS; ch++) {
        let sum = 0, sumSq = 0;
        for (let i = 0; i < WINDOW_SIZE; i++) {
            const val = window[i * NUM_CHANNELS + ch];
            sum += val;
            sumSq += val * val;
        }
        const mean = sum / WINDOW_SIZE;
        const variance = (sumSq / WINDOW_SIZE) - (mean * mean);
        const std = Math.sqrt(Math.max(variance, 1e-10));

        for (let i = 0; i < WINDOW_SIZE; i++) {
            window[i * NUM_CHANNELS + ch] = (window[i * NUM_CHANNELS + ch] - mean) / std;
        }
    }

    return window;
}

// ===== Real TF.js Inference =====
async function runInference() {
    const signalData = generateSignalWindow();

    if (state.modelLoaded && state.model) {
        // REAL TF.js INFERENCE
        const inputTensor = tf.tensor3d(signalData, [1, WINDOW_SIZE, NUM_CHANNELS]);
        const prediction = state.model.predict(inputTensor);
        const probValue = (await prediction.data())[0];

        // Clean up tensors
        inputTensor.dispose();
        prediction.dispose();

        const stressScore = Math.round(probValue * 100);
        const isStressed = probValue > 0.5;
        const confidence = isStressed ? probValue : (1 - probValue);

        return {
            stressScore,
            isStressed,
            confidence: Math.min(0.99, confidence),
            probRaw: probValue,
            mode: 'TF.js Real Inference'
        };
    } else {
        // FALLBACK: Simulated prediction
        return computeSimulatedPrediction();
    }
}

function computeSimulatedPrediction() {
    const s = state.sensors;
    const features = {};

    features.heartRate = clamp01(mapRange(s.heartRate, 60, 160, 0, 1));
    features.eda = clamp01(mapRange(s.eda, 5, 30, 0, 1));
    features.emg = clamp01(mapRange(s.emg, 20, 80, 0, 1));
    features.respRate = clamp01(mapRange(s.respRate, 18, 35, 0, 1));
    const tempDev = Math.abs(s.temp - 36.8);
    features.temp = clamp01(mapRange(tempDev, 0.5, 3, 0, 1));
    const accMag = Math.sqrt(s.accX ** 2 + s.accY ** 2 + s.accZ ** 2);
    features.acc = clamp01(mapRange(accMag, 1, 8, 0, 1));

    const weights = { heartRate: 0.25, eda: 0.25, emg: 0.20, respRate: 0.15, temp: 0.08, acc: 0.07 };
    let stressScore = 0;
    Object.keys(weights).forEach(key => { stressScore += features[key] * weights[key]; });
    stressScore = clamp01(stressScore + (Math.random() - 0.5) * 0.05);

    return {
        stressScore: Math.round(stressScore * 100),
        isStressed: stressScore > 0.45,
        confidence: Math.min(0.99, 0.80 + stressScore * 0.18),
        probRaw: stressScore,
        features,
        mode: 'Simulated (No model loaded)'
    };
}


// ===== UI Functions =====
function initWaveforms() {
    const canvases = ['canvas-ecg', 'canvas-eda', 'canvas-emg', 'canvas-resp'];
    canvases.forEach(id => {
        const canvas = document.getElementById(id);
        if (canvas) {
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            const ctx = canvas.getContext('2d');
            ctx.scale(dpr, dpr);
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
        }
    });
}

function syncSlidersToState() {
    Object.keys(state.sensors).forEach(key => {
        const slider = document.getElementById('slider-' + key);
        const display = document.getElementById('display-' + key);
        if (slider) slider.value = state.sensors[key];
        if (display) display.textContent = formatSensorValue(key, state.sensors[key]);
    });
}

function formatSensorValue(key, value) {
    const v = parseFloat(value);
    if (key === 'temp' || key === 'eda') return v.toFixed(1);
    if (key === 'accX' || key === 'accY' || key === 'accZ') return v.toFixed(1);
    return Math.round(v).toString();
}

function updateSensor(name, value) {
    state.sensors[name] = parseFloat(value);
    const display = document.getElementById('display-' + name);
    if (display) display.textContent = formatSensorValue(name, value);
}

// -- Presets --
function applyPreset(name) {
    let values;
    if (name === 'random') {
        values = {
            heartRate: randInt(50, 180),
            eda: randFloat(0.5, 35, 1),
            emg: randInt(2, 95),
            respRate: randInt(10, 38),
            temp: randFloat(35, 40, 1),
            accX: randFloat(-8, 8, 1),
            accY: randFloat(-8, 8, 1),
            accZ: randFloat(-8, 8, 1)
        };
    } else {
        values = presets[name];
    }
    if (!values) return;

    Object.keys(values).forEach(key => {
        const slider = document.getElementById('slider-' + key);
        if (slider) animateSlider(slider, parseFloat(slider.value), values[key], key);
    });
}

function animateSlider(slider, from, to, key) {
    const duration = 600;
    const start = performance.now();
    function tick(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = easeOutCubic(progress);
        const current = from + (to - from) * eased;
        slider.value = current;
        updateSensor(key, current);
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
function randInt(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }
function randFloat(min, max, d) { return parseFloat((Math.random() * (max - min) + min).toFixed(d)); }

// -- Monitoring --
function toggleMonitor() {
    state.monitoring = !state.monitoring;
    const btn = document.getElementById('btn-start');
    const btnAnalyze = document.getElementById('btn-analyze');

    if (state.monitoring) {
        btn.innerHTML = '<span class="btn-icon">⏸</span> Stop Monitoring';
        btnAnalyze.disabled = false;
        startWaveformAnimation();
    } else {
        btn.innerHTML = '<span class="btn-icon">▶</span> Start Monitoring';
        if (state.animFrame) cancelAnimationFrame(state.animFrame);
    }
}

function startWaveformAnimation() {
    let lastTime = 0;
    function animate(timestamp) {
        if (!state.monitoring) return;
        const dt = timestamp - lastTime;
        if (dt > 30) {
            lastTime = timestamp;
            drawAllWaveforms(timestamp);
            updateWaveformValues();
        }
        state.animFrame = requestAnimationFrame(animate);
    }
    state.animFrame = requestAnimationFrame(animate);
}

function drawAllWaveforms(time) {
    drawWaveform('canvas-ecg', time, '#00f5d4', generateECGWave);
    drawWaveform('canvas-eda', time, '#fee440', generateEDAWave);
    drawWaveform('canvas-emg', time, '#f72585', generateEMGWave);
    drawWaveform('canvas-resp', time, '#7209b7', generateRespWave);
}

function drawWaveform(canvasId, time, color, generator) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width, h = rect.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 0.5;
    for (let y = 0; y < h; y += 20) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Waveform
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (let i = 0; i < w; i++) {
        const t = (time / 1000) + (i / w) * 4;
        const val = generator(t);
        const y = h / 2 - val * (h * 0.35);
        if (i === 0) ctx.moveTo(i, y); else ctx.lineTo(i, y);
    }
    ctx.stroke();
    ctx.strokeStyle = color + '40';
    ctx.lineWidth = 4;
    ctx.stroke();
}

function generateECGWave(t) {
    const hr = state.sensors.heartRate;
    const freq = hr / 60;
    const phase = t * freq * Math.PI * 2;
    const cycle = (phase % (Math.PI * 2)) / (Math.PI * 2);
    if (cycle < 0.05) return 0.1 * Math.sin(cycle * Math.PI * 20);
    if (cycle < 0.08) return -0.15;
    if (cycle < 0.10) return 0.9 * Math.sin((cycle - 0.08) * Math.PI * 25);
    if (cycle < 0.14) return -0.2 * Math.sin((cycle - 0.10) * Math.PI * 12.5);
    if (cycle < 0.25) return 0.15 * Math.sin((cycle - 0.14) * Math.PI * 9);
    return 0.02 * Math.sin(t * 3 + Math.random() * 0.02);
}
function generateEDAWave(t) {
    const eda = state.sensors.eda;
    return (eda / 40) * 0.6 + 0.15 * Math.sin(t * 0.5) + 0.08 * Math.sin(t * 1.3) + 0.03 * Math.random();
}
function generateEMGWave(t) {
    const emg = state.sensors.emg;
    const amp = emg / 100;
    return amp * (Math.sin(t * 47) * 0.4 + Math.sin(t * 93) * 0.3 + Math.sin(t * 157) * 0.2 + (Math.random() - 0.5) * 0.3);
}
function generateRespWave(t) {
    const rr = state.sensors.respRate;
    const f = rr / 60;
    return 0.6 * Math.sin(t * f * Math.PI * 2) + 0.1 * Math.sin(t * f * Math.PI * 4);
}

function updateWaveformValues() {
    document.getElementById('ecg-value').textContent = state.sensors.heartRate + ' BPM';
    document.getElementById('eda-value').textContent = parseFloat(state.sensors.eda).toFixed(1) + ' µS';
    document.getElementById('emg-value').textContent = state.sensors.emg + '%';
    document.getElementById('resp-value').textContent = state.sensors.respRate + ' BrPM';
}

// ===== Stress Analysis — REAL TF.js Inference =====
async function analyzeStress() {
    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');
    const resultCard = document.getElementById('result-card');
    const gaugeLabel = document.getElementById('gauge-label');
    const confDiv = document.getElementById('result-confidence');
    const analysisCard = document.getElementById('analysis-card');

    // Start animation
    statusIcon.textContent = '⚙️';
    statusIcon.classList.add('analyzing');
    statusText.textContent = state.modelLoaded ? 'Running TF.js inference...' : 'Analyzing signals...';
    statusText.style.color = '';
    gaugeLabel.textContent = 'Processing...';
    gaugeLabel.style.color = '';
    resultCard.className = 'result-card';
    confDiv.style.display = 'none';
    analysisCard.style.display = 'none';

    // Run inference with slight delay for UX
    await new Promise(resolve => setTimeout(resolve, 800));
    const result = await runInference();
    console.log('Prediction result:', result);
    displayResult(result);
}

function displayResult(result) {
    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');
    const resultCard = document.getElementById('result-card');
    const gaugeLabel = document.getElementById('gauge-label');
    const gaugeValue = document.getElementById('gauge-value');
    const gaugeArc = document.getElementById('gauge-arc');
    const confDiv = document.getElementById('result-confidence');
    const confFill = document.getElementById('conf-fill');
    const confValue = document.getElementById('conf-value');
    const analysisCard = document.getElementById('analysis-card');
    const featureBars = document.getElementById('feature-bars');
    const inferenceBadge = document.getElementById('inference-mode');

    statusIcon.classList.remove('analyzing');

    // Show inference mode
    if (inferenceBadge) {
        inferenceBadge.textContent = result.mode;
        inferenceBadge.style.display = 'block';
    }

    if (result.isStressed) {
        statusIcon.textContent = '🔴';
        statusText.textContent = 'STRESSED';
        statusText.style.color = '#ef4444';
        resultCard.className = 'result-card stressed';
        gaugeLabel.textContent = 'Stress Detected';
        gaugeLabel.style.color = '#ef4444';
    } else {
        statusIcon.textContent = '🟢';
        statusText.textContent = 'NOT STRESSED';
        statusText.style.color = '#10b981';
        resultCard.className = 'result-card relaxed';
        gaugeLabel.textContent = 'Relaxed State';
        gaugeLabel.style.color = '#10b981';
    }

    // Animate gauge
    gaugeValue.textContent = '0';
    animateGauge(result.stressScore);

    const arcLength = 251.3;
    const dashOffset = arcLength - (result.stressScore / 100) * arcLength;
    gaugeArc.style.transition = 'stroke-dashoffset 1.5s ease';
    gaugeArc.setAttribute('stroke-dashoffset', dashOffset);

    // Confidence
    confDiv.style.display = 'flex';
    const confPct = Math.round(result.confidence * 100);
    confValue.textContent = confPct + '%';
    confFill.style.width = confPct + '%';
    confFill.style.background = result.isStressed
        ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
        : 'linear-gradient(90deg, #10b981, #00f5d4)';

    // Show sensor contribution analysis
    analysisCard.style.display = 'block';
    featureBars.innerHTML = '';

    const sensorAnalysis = [
        { name: 'Heart Rate (ECG)', value: Math.min(100, Math.round((state.sensors.heartRate / 200) * 100)), color: '#00f5d4' },
        { name: 'Skin Conductance (EDA)', value: Math.min(100, Math.round((state.sensors.eda / 40) * 100)), color: '#fee440' },
        { name: 'Muscle Activity (EMG)', value: state.sensors.emg, color: '#f72585' },
        { name: 'Respiration Rate', value: Math.min(100, Math.round((state.sensors.respRate / 40) * 100)), color: '#7209b7' },
        { name: 'Body Temperature', value: Math.min(100, Math.round(Math.abs(state.sensors.temp - 36.8) / 5 * 100)), color: '#ff6b35' },
        { name: 'Movement (ACC)', value: Math.min(100, Math.round(Math.sqrt(state.sensors.accX**2 + state.sensors.accY**2 + state.sensors.accZ**2) / 10 * 100)), color: '#4cc9f0' }
    ];

    sensorAnalysis.forEach((s, idx) => {
        const item = document.createElement('div');
        item.className = 'feature-bar-item';
        item.innerHTML = `
            <div class="feature-bar-header">
                <span class="feature-bar-name">${s.name}</span>
                <span class="feature-bar-val">${s.value}%</span>
            </div>
            <div class="feature-bar-track">
                <div class="feature-bar-fill" style="width: 0%; background: ${s.color}"></div>
            </div>
        `;
        featureBars.appendChild(item);
        requestAnimationFrame(() => {
            setTimeout(() => {
                item.querySelector('.feature-bar-fill').style.width = s.value + '%';
            }, 100 + idx * 80);
        });
    });

    state.prediction = result;
}

function animateGauge(target) {
    const el = document.getElementById('gauge-value');
    const duration = 1200;
    const start = performance.now();
    function tick(now) {
        const progress = Math.min((now - start) / duration, 1);
        el.textContent = Math.round(target * easeOutCubic(progress));
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

// -- Utilities --
function clamp01(v) { return Math.max(0, Math.min(1, v)); }
function mapRange(value, inMin, inMax, outMin, outMax) {
    return ((value - inMin) / (inMax - inMin)) * (outMax - outMin) + outMin;
}

window.addEventListener('resize', () => { initWaveforms(); });
