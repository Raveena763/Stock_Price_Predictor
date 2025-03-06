import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    return scaled_data, scaler

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    st.title("Stock Price Prediction using LSTM")
    ticker = st.text_input("Enter NSE Stock Ticker (e.g., RELIANCE.NS):", "RELIANCE.NS")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    
    if st.button("Predict"):
        with st.spinner("Fetching Data..."):
            data = fetch_stock_data(ticker, start_date, end_date)
            st.write("Stock Data:", data.tail())
            
            scaled_data, scaler = prepare_data(data)
            seq_length = 60
            x, y = create_sequences(scaled_data, seq_length)
            x = x.reshape((x.shape[0], x.shape[1], 1))
            
            model = build_lstm_model((x.shape[1], 1))
            model.fit(x, y, epochs=10, batch_size=32, verbose=1)
            
            test_data = scaled_data[-seq_length:]
            test_data = test_data.reshape((1, seq_length, 1))
            prediction = model.predict(test_data)
            predicted_price = scaler.inverse_transform(prediction)[0][0]
            
            st.success(f"Predicted Stock Price: ₹{predicted_price:.2f}")
            
            plt.figure(figsize=(10,5))
            plt.plot(data.index, data['Close'], label='Actual Price')
            plt.axhline(y=predicted_price, color='r', linestyle='--', label='Predicted Price')
            plt.legend()
            st.pyplot(plt)

if __name__ == "__main__":
    main()
