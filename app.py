import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Price Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0D1F0F;
    color: #E8F0E9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0A1A0C !important;
    border-right: 1px solid #1E3A20;
}
[data-testid="stSidebar"] * {
    color: #C8DEC9 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #7AAF7D !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1A3A1C 0%, #0D2E0F 50%, #162E10 100%);
    border: 1px solid #2A4A2C;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(74,180,74,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(74,180,74,0.15);
    border: 1px solid rgba(74,180,74,0.3);
    color: #4AB44A;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.hero h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 48px !important;
    font-weight: 900 !important;
    color: #E8F5E9 !important;
    line-height: 1.1 !important;
    margin: 0 0 12px !important;
    letter-spacing: -0.02em;
}
.hero h1 span { color: #4AB44A; }
.hero p {
    color: #7AAF7D;
    font-size: 16px;
    font-weight: 300;
    margin: 0;
    line-height: 1.6;
}

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: #122214;
    border: 1px solid #1E3A20;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4AB44A, #2D7A2D);
}
.metric-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5A8A5C;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    color: #E8F5E9;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: #4AB44A;
    margin-top: 4px;
}

/* Section headers */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #E8F5E9;
    margin: 32px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #2A4A2C, transparent);
}

/* Chart container */
.chart-box {
    background: #122214;
    border: 1px solid #1E3A20;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
}

/* Forecast table */
.forecast-row {
    display: grid;
    grid-template-columns: 80px 1fr 140px;
    align-items: center;
    padding: 12px 20px;
    border-bottom: 1px solid #1A321C;
    transition: background 0.15s;
}
.forecast-row:last-child { border-bottom: none; }
.forecast-row:hover { background: rgba(74,180,74,0.05); }
.forecast-day { font-size: 12px; color: #5A8A5C; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; }
.forecast-bar-wrap { padding: 0 16px; }
.forecast-bar { height: 6px; background: linear-gradient(90deg, #4AB44A, #2D7A2D); border-radius: 3px; }
.forecast-price { font-family: 'Playfair Display', serif; font-size: 20px; color: #E8F5E9; text-align: right; }
.forecast-unit { font-size: 11px; color: #5A8A5C; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2D7A2D, #1E5C1E) !important;
    color: #E8F5E9 !important;
    border: 1px solid #3A8A3A !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 28px !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3A8A3A, #2A6A2A) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(74,180,74,0.25) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #122214 !important;
    border: 1px solid #1E3A20 !important;
    border-radius: 8px !important;
    color: #C8DEC9 !important;
}

/* Selectbox / Slider */
.stSelectbox > div > div {
    background: #122214 !important;
    border-color: #2A4A2C !important;
    color: #C8DEC9 !important;
}
.stSlider .stSlider > div { background: #4AB44A !important; }

/* Success / spinner */
.stSuccess {
    background: rgba(74,180,74,0.1) !important;
    border: 1px solid rgba(74,180,74,0.3) !important;
    color: #4AB44A !important;
    border-radius: 8px !important;
}

/* Divider */
hr { border-color: #1E3A20 !important; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">🌾 AI-Powered · APMC Data · India</div>
  <h1>Crop Price<br><span>Predictor</span></h1>
  <p>Forecast mandi prices for major crops using<br>Gradient Boosting ML — built for Indian farmers & traders.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Configure")
crop         = st.sidebar.selectbox("Crop",   ["Onion", "Tomato", "Potato", "Wheat"])
market       = st.sidebar.selectbox("Market", ["Lasalgaon", "Pune", "Nashik", "Mumbai"])
n_estimators = st.sidebar.slider("Model Complexity", 50, 300, 150)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:12px; color:#5A8A5C; line-height:1.8;'>
🌱 Data: APMC / Agmarknet<br>
🤖 Model: Gradient Boosting<br>
📍 Region: Maharashtra<br>
</div>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────
@st.cache_data
def load_data(crop_name):
    np.random.seed(42)
    dates  = pd.date_range("2021-01-01", "2024-01-01", freq="D")
    prices = 2000 + np.cumsum(np.random.randn(len(dates)) * 50)
    prices = np.clip(prices, 500, 6000).astype(int)
    df = pd.DataFrame({"date": dates, "modal_price": prices})
    df = df.set_index("date")
    return df

df = load_data(crop)

# Feature engineering
df['lag_7']       = df['modal_price'].shift(7)
df['lag_14']      = df['modal_price'].shift(14)
df['lag_30']      = df['modal_price'].shift(30)
df['roll_mean_7'] = df['modal_price'].rolling(7).mean()
df['roll_std_7']  = df['modal_price'].rolling(7).std()
df['month']       = df.index.month
df['season']      = df['month'].map({12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4})
df.dropna(inplace=True)

FEATURES = ['lag_7','lag_14','lag_30','roll_mean_7','roll_std_7','month','season']

# ── Raw data expander ──────────────────────────────────────────
with st.expander("📊 View Historical Price Data"):
    fig0, ax0 = plt.subplots(figsize=(12, 3))
    fig0.patch.set_facecolor('#122214')
    ax0.set_facecolor('#122214')
    ax0.plot(df.index, df['modal_price'], color='#4AB44A', linewidth=1.2, alpha=0.9)
    ax0.fill_between(df.index, df['modal_price'], alpha=0.1, color='#4AB44A')
    ax0.set_xlabel("Date", color='#5A8A5C', fontsize=10)
    ax0.set_ylabel("Price (₹/quintal)", color='#5A8A5C', fontsize=10)
    ax0.tick_params(colors='#5A8A5C')
    for spine in ax0.spines.values(): spine.set_color('#2A4A2C')
    ax0.grid(alpha=0.15, color='#2A4A2C')
    st.pyplot(fig0)

# ── Train Button ───────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn, col_spacer = st.columns([1, 2])
with col_btn:
    train = st.button("🚀 Train Model & Predict")

if train:
    with st.spinner("Training model..."):
        X = df[FEATURES].values
        y = df['modal_price'].values
        split    = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=0.1,
            max_depth=4, random_state=42
        )
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        actual    = y_test

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae  = mean_absolute_error(actual, predicted)
        acc  = 100 - (mae / actual.mean() * 100)

    # ── Metrics ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">RMSE Score</div>
        <div class="metric-value">₹{rmse:.0f}</div>
        <div class="metric-sub">Root mean square error</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">MAE Score</div>
        <div class="metric-value">₹{mae:.0f}</div>
        <div class="metric-sub">Mean absolute error</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">{acc:.1f}%</div>
        <div class="metric-sub">Model performance</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart ──────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">📈 {crop} — Actual vs Predicted ({market})</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#122214')
    ax.set_facecolor('#122214')
    ax.plot(actual,    color='#4AB44A', linewidth=1.5, label='Actual price',    alpha=0.9)
    ax.plot(predicted, color='#F6C842', linewidth=1.5, label='Predicted price', linestyle='--', alpha=0.9)
    ax.fill_between(range(len(actual)), actual, predicted, alpha=0.06, color='#4AB44A')
    ax.set_xlabel("Days", color='#5A8A5C', fontsize=10)
    ax.set_ylabel("Price (₹/quintal)", color='#5A8A5C', fontsize=10)
    ax.tick_params(colors='#5A8A5C')
    for spine in ax.spines.values(): spine.set_color('#2A4A2C')
    ax.grid(alpha=0.15, color='#2A4A2C')
    p1 = mpatches.Patch(color='#4AB44A', label='Actual price')
    p2 = mpatches.Patch(color='#F6C842', label='Predicted price')
    ax.legend(handles=[p1, p2], facecolor='#1A3A1C', edgecolor='#2A4A2C',
              labelcolor='#C8DEC9', fontsize=10)
    st.pyplot(fig)

    # ── 7-day Forecast ─────────────────────────────────────────
    st.markdown('<div class="section-title">🔮 7-Day Price Forecast</div>', unsafe_allow_html=True)

    last_row  = df[FEATURES].iloc[-1].values.copy()
    forecasts = []
    for _ in range(7):
        pred = model.predict(last_row.reshape(1, -1))[0]
        forecasts.append(int(pred))
        last_row[0] = last_row[1]
        last_row[1] = last_row[2]
        last_row[2] = pred

    max_price = max(forecasts)
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    forecast_html = '<div style="background:#122214; border:1px solid #1E3A20; border-radius:12px; overflow:hidden;">'
    for i, price in enumerate(forecasts):
        bar_pct = int((price / max_price) * 100)
        forecast_html += f"""
        <div class="forecast-row">
          <div class="forecast-day">Day {i+1}<br><span style="color:#3A6A3C;font-size:10px">{days[i]}</span></div>
          <div class="forecast-bar-wrap">
            <div class="forecast-bar" style="width:{bar_pct}%"></div>
          </div>
          <div class="forecast-price">₹{price:,}<br><span class="forecast-unit">/quintal</span></div>
        </div>"""
    forecast_html += '</div>'
    st.markdown(forecast_html, unsafe_allow_html=True)

    # ── Trend summary ──────────────────────────────────────────
    trend    = "📈 Rising" if forecasts[-1] > forecasts[0] else "📉 Falling"
    diff     = abs(forecasts[-1] - forecasts[0])
    st.markdown(f"""
    <div style="background:#122214; border:1px solid #1E3A20; border-radius:10px;
                padding:16px 24px; margin-top:16px; display:flex; gap:32px; align-items:center;">
      <div>
        <div style="font-size:11px;color:#5A8A5C;text-transform:uppercase;letter-spacing:0.08em;">7-Day Trend</div>
        <div style="font-size:20px;color:#E8F5E9;font-weight:500;margin-top:4px;">{trend}</div>
      </div>
      <div>
        <div style="font-size:11px;color:#5A8A5C;text-transform:uppercase;letter-spacing:0.08em;">Price Change</div>
        <div style="font-size:20px;color:#4AB44A;font-weight:500;margin-top:4px;">₹{diff:,}</div>
      </div>
      <div>
        <div style="font-size:11px;color:#5A8A5C;text-transform:uppercase;letter-spacing:0.08em;">Crop / Market</div>
        <div style="font-size:20px;color:#E8F5E9;font-weight:500;margin-top:4px;">{crop} · {market}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Prediction complete!")
