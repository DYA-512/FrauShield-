import os
import zipfile
import io
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Global variables for model and data cache
model = None
scaler = None
df_cache = None
feature_columns = []

DATA_ZIP_PATH = "creditcard_2023_csv.zip"
CSV_FILENAME = "creditcard_2023.csv"

def load_and_train_model():
    global model, scaler, df_cache, feature_columns
    print("Initializing FraudShield ML pipeline...")
    
    if not os.path.exists(DATA_ZIP_PATH):
        # Create dummy synthetic data if dataset zip is missing so app still runs out-of-the-box
        print(f"Warning: {DATA_ZIP_PATH} not found. Generating synthetic fallback data...")
        np.random.seed(42)
        n_samples = 5000
        data = {
            'amount': np.random.exponential(scale=50.0, size=n_samples),
            'velocity': np.random.poisson(lam=1.5, size=n_samples),
            'distance_from_home': np.random.gamma(shape=2, scale=10, size=n_samples),
            'device_risk_score': np.random.uniform(0, 1, size=n_samples),
            'is_vpn': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
        }
        df = pd.DataFrame(data)
    else:
        print(f"Extracting and loading dataset from {DATA_ZIP_PATH}...")
        with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as z:
            # Look for the csv file inside the zip
            csv_name = next((name for name in z.namelist() if name.endswith('.csv')), CSV_FILENAME)
            with z.open(csv_name) as f:
                # Read a subset for fast initial load and responsiveness
                df = pd.read_csv(f, nrows=10000)
        
        # Clean column names if necessary
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
        # Map dataset columns to standard feature set if standard names aren't present
        if 'amount' not in df.columns and 'v1' in df.columns:
            # Using typical Kaggle creditcard dataset format columns (V1-V28, Amount)
            feature_candidates = [c for c in df.columns if c.startswith('v') or c == 'amount']
            df = df[feature_candidates].copy()
            if 'amount' not in df.columns:
                df['amount'] = np.random.uniform(10, 500, len(df))
        elif 'amount' not in df.columns:
            df['amount'] = np.random.uniform(10, 500, len(df))

    # Select numerical features for Isolation Forest
    feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'class' and col != 'id']
    
    # Keep a small sample cache for reference transactions and analytics
    df_cache = df[feature_columns].dropna().head(500).copy()
    
    X = df_cache.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest for anomaly detection
    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(X_scaled)
    print("FraudShield ML Model successfully trained and ready!")

# HTML Template embedded for single-file adherence
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FraudShield | Stripe Radar Intelligence</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-950 text-slate-100 font-sans antialiased min-h-screen flex flex-col">
    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="bg-indigo-600 p-2.5 rounded-xl shadow-lg shadow-indigo-500/30">
                    <i class="fa-solid fa-shield-halved text-white text-xl"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                        FraudShield <span class="text-xs px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 font-mono">Stripe Radar Engine</span>
                    </h1>
                    <p class="text-xs text-slate-400">Real-time ML Anomaly Detection & Risk Scoring</p>
                </div>
            </div>
            <div class="flex items-center gap-3">
                <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Engine Active
                </span>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 py-8 flex-grow grid grid-cols-1 lg:grid-cols-3 gap-8 w-full">
        <!-- Control Panel / Inputs -->
        <div class="lg:col-span-1 bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
            <div>
                <h2 class="text-base font-semibold text-white mb-1 flex items-center gap-2">
                    <i class="fa-solid fa-sliders text-indigo-400"></i> Transaction Parameters
                </h2>
                <p class="text-xs text-slate-400 mb-6">Adjust parameters to simulate live incoming payment requests.</p>
                
                <form id="fraudForm" class="space-y-4">
                    <div id="dynamicInputs" class="space-y-4">
                        <!-- Populated dynamically via JS based on model features -->
                    </div>
                </form>
            </div>

            <div class="mt-8 pt-6 border-t border-slate-800">
                <button type="button" id="analyzeBtn" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-3 px-4 rounded-xl shadow-lg shadow-indigo-600/30 transition duration-200 flex items-center justify-center gap-2">
                    <i class="fa-solid fa-bolt"></i> Evaluate Risk Score
                </button>
            </div>
        </div>

        <!-- Results & Radar Display -->
        <div class="lg:col-span-2 space-y-6">
            <!-- Score Card -->
            <div id="resultCard" class="bg-slate-900/80 border border-slate-800 rounded-2xl p-8 shadow-xl relative overflow-hidden flex flex-col items-center justify-center text-center min-h-[280px]">
                <div class="absolute top-0 left-0 w-full h-1 bg-slate-800" id="topIndicator"></div>
                <div id="initialState" class="space-y-3">
                    <div class="w-16 h-16 bg-slate-800/60 rounded-full flex items-center justify-center mx-auto text-slate-500 text-2xl">
                        <i class="fa-solid fa-radar"></i>
                    </div>
                    <h3 class="text-lg font-medium text-slate-300">Ready for Analysis</h3>
                    <p class="text-sm text-slate-500 max-w-sm">Configure transaction attributes on the left and click evaluate to run real-time isolation forest scoring.</p>
                </div>
                
                <div id="loadingState" class="hidden space-y-3">
                    <div class="w-12 h-12 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
                    <p class="text-sm text-slate-400">Scraping patterns & analyzing feature isolation...</p>
                </div>

                <div id="scoreState" class="hidden w-full space-y-6">
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
                        <div class="bg-slate-950/60 border border-slate-800/80 p-4 rounded-xl text-center">
                            <span class="text-xs text-slate-400 block mb-1">Risk Score</span>
                            <span id="riskScoreVal" class="text-3xl font-black font-mono">0%</span>
                        </div>
                        <div class="bg-slate-950/60 border border-slate-800/80 p-4 rounded-xl text-center">
                            <span class="text-xs text-slate-400 block mb-1">Radar Status</span>
                            <span id="radarStatusVal" class="text-lg font-bold">NORMAL</span>
                        </div>
                        <div class="bg-slate-950/60 border border-slate-800/80 p-4 rounded-xl text-center">
                            <span class="text-xs text-slate-400 block mb-1">Action Suggested</span>
                            <span id="actionVal" class="text-sm font-semibold text-emerald-400">Allow Payment</span>
                        </div>
                    </div>
                    <div class="text-left bg-slate-950/40 p-4 rounded-xl border border-slate-800/50">
                        <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Model Decision Breakdown</h4>
                        <p id="decisionReason" class="text-sm text-slate-300"></p>
                    </div>
                </div>
            </div>

            <!-- Recent Analytics Log -->
            <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
                <h3 class="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                    <i class="fa-solid fa-clock-rotate-left text-indigo-400"></i> Live Radar Transaction Feed
                </h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs text-slate-300">
                        <thead class="bg-slate-950 text-slate-400 uppercase font-mono tracking-wider">
                            <tr>
                                <th class="p-3 rounded-l-lg">Timestamp</th>
                                <th class="p-3">Primary Amount / Metric</th>
                                <th class="p-3">Risk Score</th>
                                <th class="p-3 rounded-r-lg">Status</th>
                            </tr>
                        </thead>
                        <tbody id="logTableBody" class="divide-y divide-slate-800/60">
                            <tr>
                                <td colspan="4" class="p-4 text-center text-slate-500 italic">No transactions evaluated in this session yet.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-800 bg-slate-900/40 py-6 text-center text-xs text-slate-500">
        FraudShield Engine &bull; Powered by Scikit-Learn & Flask
    </footer>

    <script>
        let featureNames = [];

        async function fetchFeatures() {
            const res = await fetch('/api/features');
            const data = await res.json();
            featureNames = data.features;
            
            const container = document.getElementById('dynamicInputs');
            container.innerHTML = '';
            
            featureNames.forEach((feat, index) => {
                const defaultVal = data.defaults[feat] !== undefined ? data.defaults[feat] : 50.0;
                const div = document.createElement('div');
                div.className = "space-y-1.5";
                div.innerHTML = `
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider">${feat}</label>
                    <input type="number" step="any" name="${feat}" value="${defaultVal}" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 transition">
                `;
                container.appendChild(div);
            });
        }

        document.getElementById('analyzeBtn').addEventListener('click', async () => {
            const form = document.getElementById('fraudForm');
            const formData = new FormData(form);
            const payload = {};
            formData.forEach((val, key) => {
                payload[key] = parseFloat(val) || 0;
            });

            document.getElementById('initialState').classList.add('hidden');
            document.getElementById('scoreState').classList.add('hidden');
            document.getElementById('loadingState').classList.remove('hidden');
            document.getElementById('topIndicator').className = "absolute top-0 left-0 w-full h-1 bg-indigo-500 animate-pulse";

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();

                setTimeout(() => {
                    document.getElementById('loadingState').classList.add('hidden');
                    document.getElementById('scoreState').classList.remove('hidden');

                    const scoreValEl = document.getElementById('riskScoreVal');
                    const radarStatusEl = document.getElementById('radarStatusVal');
                    const actionEl = document.getElementById('actionVal');
                    const reasonEl = document.getElementById('decisionReason');
                    const topIndicator = document.getElementById('topIndicator');

                    scoreValEl.innerText = result.risk_score + '%';
                    
                    if (result.is_fraud) {
                        radarStatusEl.innerText = "HIGH RISK / FRAUD";
                        radarStatusEl.className = "text-lg font-bold text-rose-400";
                        actionEl.innerText = "Block / Require 3DS";
                        actionEl.className = "text-sm font-semibold text-rose-400";
                        topIndicator.className = "absolute top-0 left-0 w-full h-1 bg-rose-500 shadow-lg shadow-rose-500/50";
                    } else {
                        radarStatusEl.innerText = "NORMAL";
                        radarStatusEl.className = "text-lg font-bold text-emerald-400";
                        actionEl.innerText = "Allow Payment";
                        actionEl.className = "text-sm font-semibold text-emerald-400";
                        topIndicator.className = "absolute top-0 left-0 w-full h-1 bg-emerald-500 shadow-lg shadow-emerald-500/50";
                    }

                    reasonEl.innerText = result.explanation;
                    addLogEntry(result);
                }, 400);

            } catch (err) {
                console.error(err);
                document.getElementById('loadingState').classList.add('hidden');
                document.getElementById('initialState').classList.remove('hidden');
                alert('Error analyzing transaction.');
            }
        });

        function addLogEntry(res) {
            const tbody = document.getElementById('logTableBody');
            if (tbody.querySelector('td[colspan]')) {
                tbody.innerHTML = '';
            }
            const row = document.createElement('tr');
            const timeStr = new Date().toLocaleTimeString();
            const badgeClass = res.is_fraud ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
            const statusText = res.is_fraud ? 'Blocked' : 'Approved';

            row.innerHTML = `
                <td class="p-3 font-mono text-slate-400">${timeStr}</td>
                <td class="p-3 font-mono">${res.primary_amount}</td>
                <td class="p-3 font-mono font-bold">${res.risk_score}%</td>
                <td class="p-3"><span class="px-2 py-0.5 rounded text-xs border ${badgeClass}">${statusText}</span></td>
            `;
            tbody.insertBefore(row, tbody.firstChild);
        }

        window.onload = fetchFeatures;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/features')
def get_features():
    global feature_columns, df_cache
    defaults = {}
    if df_cache is not None and len(df_cache) > 0:
        mean_vals = df_cache.mean().to_dict()
        defaults = {k: round(float(v), 2) for k, v in mean_vals.items()}
    return jsonify({
        "features": feature_columns,
        "defaults": defaults
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    global model, scaler, feature_columns
    data = request.get_json()
    
    # Extract values in correct feature order
    input_vector = [float(data.get(feat, 0.0)) for feat in feature_columns]
    X_input = np.array([input_vector])
    
    X_scaled = scaler.transform(X_input)
    # Isolation forest decision function: negative scores represent anomalies/outliers
    score_raw = model.decision_function(X_scaled)[0]
    prediction = model.predict(X_scaled)[0] # -1 for anomaly, 1 for normal
    
    # Map score_raw to a risk percentage from 0 to 100
    # Typically decision function ranges roughly from -0.3 to 0.3
    normalized_risk = int(np.clip((0.2 - score_raw) / 0.4 * 100, 2, 98))
    is_fraud = bool(prediction == -1 or normalized_risk > 65)

    primary_amt = data.get('amount', list(data.values())[0] if data else 0.0)

    if is_fraud:
        explanation = f"Anomaly detected by Stripe Radar Isolation Forest. The combination of feature metrics deviates significantly from baseline distributions (Decision score: {score_raw:.3f})."
    else:
        explanation = f"Transaction profile matches normal behavioral patterns within safe statistical bounds (Decision score: {score_raw:.3f})."

    return jsonify({
        "risk_score": normalized_risk,
        "is_fraud": is_fraud,
        "primary_amount": f"${primary_amt:.2f}" if 'amount' in data else f"{primary_amt:.2f}",
        "explanation": explanation
    })

if __name__ == '__main__':
    load_and_train_model()
    app.run(debug=True, port=5000)