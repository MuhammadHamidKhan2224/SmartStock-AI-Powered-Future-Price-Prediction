import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Title of the App
st.title("📈 Stock Price Predictor App")

# User Input for Stock Ticker
stock = st.text_input('Enter the Stock ID (e.g., ENGRO.KA)', 'ENGRO.KA')

# Set Start and End Dates for Historical Data
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)

# Fetch Stock Data
engro = yf.download(stock, start, end)

# Check if Data is Available
if engro.empty:
    st.error("⚠️ No data found for this stock ticker! Please try another.")
    st.stop()

# Load Pre-Trained Model
model_path = "./Latest_stock_price_model.keras"  # Adjust path if needed
model = load_model(model_path)

# Display Stock Data
st.subheader('📊 Stock Data (Last 10 Records)')
st.write(engro.tail(10))

# Moving Averages
engro['MA_for_250_days'] = engro['Close'].rolling(250).mean()
engro['MA_for_100_days'] = engro['Close'].rolling(100).mean()

# Function to Plot Graphs
def plot_graph(figsize, actual_data, predicted_data=None, title="Stock Data"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(actual_data, label="Actual Prices", color="blue")
    if predicted_data is not None:
        ax.plot(predicted_data, label="Predicted Prices", color="orange")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# Plot Moving Averages
st.subheader('📈 Moving Averages (250 Days)')
plot_graph((15,6), engro['Close'], engro['MA_for_250_days'], "Close Price & 250-Day MA")

st.subheader('📉 Moving Averages (100 Days)')
plot_graph((15,6), engro['Close'], engro['MA_for_100_days'], "Close Price & 100-Day MA")

# Scale the Data for Model Prediction
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(engro[['Close']])

# Prepare Data for LSTM
x_test, y_test = [], []
for i in range(100, len(scaled_data)):
    x_test.append(scaled_data[i-100:i])
    y_test.append(scaled_data[i])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make Predictions
predictions = model.predict(x_test)

# Inverse Transform Predictions
inv_pred = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)

# Create DataFrame for Display
plotting_data = pd.DataFrame(
    {'Original Test Data': inv_y_test.flatten(), 'Predicted': inv_pred.flatten()},
    index=engro.index[-len(inv_pred):]  # Align index correctly
)

# Display Predictions
st.subheader("📊 Original vs Predicted Stock Prices")
st.write(plotting_data.tail(10))  # Show last 10 records

# Plot Actual vs Predicted Prices
st.subheader('📉 Original Close Price vs Predicted Close Price')
plot_graph((15,6), plotting_data['Original Test Data'], plotting_data['Predicted'], "Actual vs Predicted Prices")
