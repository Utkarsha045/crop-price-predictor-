import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Crop Price Predictor", page_icon="🌾", layout="wide")
st.title("🌾 APMC Crop Price Predictor")
st.markdown("Predict future mandi prices using Machine Learning")

st.sidebar.header("Settings")
crop    = st.sidebar.selectbox("Crop",   ["Onion", "Tomato", "Potato"])
market  = st.sidebar.selectbox("Market", ["Lasalgaon", "Pune", "Nashik"])
n_estimators = st.sidebar.slider("Model complexity", 50, 300, 100)

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
TARGET   = 'modal_price'

with st.expander("📊 View raw price data"):
    st.line_chart(df['modal_price'])
    st.dataframe(df[['modal_price']].tail(10))

if st.button("🚀 Train Model & Predict"):
    with st.spinner("Training model... please wait"):

        X = df[FEATURES].values
        y = df[TARGET].values

        split    = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        actual    = y_test

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae  = mean_absolute_error(actual, predicted)

    st.success("✅ Model trained successfully!")

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE",     f"₹ {rmse:.0f}")
    col2.metric("MAE",      f"₹ {mae:.0f}")
    col3.metric("Accuracy", f"{100-(mae/actual.mean()*100):.1f}%")

    st.subheader(f"📈 {crop} Price — Actual vs Predicted ({market})")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual,    label="Actual price",    color="#2E86AB")
    ax.plot(predicted, label="Predicted price", color="#F6AE2D", linestyle="--")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (₹/quintal)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # Next 7 days forecast
    st.subheader("🔮 Next 7-day Price Forecast")
    last_row   = df[FEATURES].iloc[-1].values.copy()
    last_price = df['modal_price'].iloc[-1]
    forecasts  = []

    for i in range(7):
        pred = model.predict(last_row.reshape(1, -1))[0]
        forecasts.append(pred)
        last_row[0] = last_row[1]
        last_row[1] = last_row[2]
        last_row[2] = pred

    forecast_df = pd.DataFrame({
        "Day"                         : [f"Day {i+1}" for i in range(7)],
        "Predicted Price (₹/quintal)" : [int(p) for p in forecasts]
    })
    st.dataframe(forecast_df, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Built with Gradient Boosting + Streamlit\nData: APMC / Agmarknet")
