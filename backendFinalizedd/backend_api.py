from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timezone
import logging
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.getenv("MODEL_PATH", "rf_btc_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
ACCURACY_NOTE = "Approximately 60.87%"  


def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        app.logger.info("Model and scaler loaded.")
        return model, scaler
    except Exception as e:
        app.logger.error(f"Failed to load artifacts: {e}")
        raise

def fetch_btc_yfinance(days=90):
    try:
        df = yf.download("BTC-USD", period=f"{days}d", interval="1d", auto_adjust=False, progress=False)
        df = df.rename(columns=str.title)  
        df = df[["Close", "Volume"]].dropna()
        if df.empty:
            raise ValueError("Empty dataframe from yfinance.")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception as e:
        app.logger.warning(f"yfinance failed: {e}")
        return None

def fetch_btc_coingecko(days=90):
    
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        vols = data.get("total_volumes", [])
        if not prices or not vols:
            raise ValueError("Missing prices or volumes in CoinGecko response.")

        dfp = pd.DataFrame(prices, columns=["ts", "Close"])
        dfv = pd.DataFrame(vols, columns=["ts", "Volume"])
        df = pd.merge(dfp, dfv, on="ts", how="inner")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("ts").sort_index()
        return df[["Close", "Volume"]].dropna()
    except Exception as e:
        app.logger.error(f"CoinGecko fallback failed: {e}")
        return None

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_features(df: pd.DataFrame):
    
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["RSI14"] = compute_rsi(df["Close"], period=14)
    df["Volatility21"] = df["Return"].rolling(window=21).std()

    latest = df.dropna().iloc[-1]  
    features = np.array([
        latest["Close"],
        latest["RSI14"],
        latest["Volume"],
        latest["Volatility21"]
    ], dtype=float)
    return features

def get_latest_features():
    df = fetch_btc_yfinance(days=90)
    if df is None:
        df = fetch_btc_coingecko(days=90)
    if df is None or len(df) < 30:
        raise RuntimeError("Unable to fetch sufficient BTC data from any source.")
    return compute_features(df)


model, scaler = load_artifacts()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model": "RandomForest (direction up/down)",
        "features": ["Close", "RSI14", "Volume", "Volatility21"],
        "needs_recent_days": 21,
        "accuracy_note": ACCURACY_NOTE
    })

@app.route("/predict", methods=["GET"])
def predict():
    
    try:
        features = get_latest_features()
        features_scaled = scaler.transform(features.reshape(1, -1))
        pred = model.predict(features_scaled)[0]  
        label = "Up" if int(pred) == 1 else "Down"
        return jsonify({
            "status": "success",
            "prediction": label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accuracy": ACCURACY_NOTE
        })
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route("/historical", methods=["GET"])
def historical():
    
    try:
        df = fetch_btc_yfinance(days=30)  
        if df is None:
            df = fetch_btc_coingecko(days=30)

        if df is None:
            raise RuntimeError("Unable to fetch historical data")

        
        recent_data = df.tail(10)

        
        prices = recent_data["Close"].values.tolist()
        timestamps = recent_data.index.to_pydatetime()  
        timestamps = [ts.isoformat() for ts in timestamps]

        return jsonify({
            "status": "success",
            "prices": prices,
            "timestamps": timestamps,
            "count": len(prices)
        })
    except Exception as e:
        app.logger.exception("Historical data fetch failed")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000, debug=True)
