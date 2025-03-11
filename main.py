import pandas as pd
import keras
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, R2Score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('spx_d.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
min_date = df['Date'].min()
df['Date'] = (df['Date'] - min_date).dt.days

dates_to_mark = ['2001-09-11', '2020-03-11', '1987-10-19', '2020-01-31', '2022-02-24']
dates_to_mark = [(pd.to_datetime(date) - min_date).days for date in dates_to_mark]
df['one_hot'] = df['Date'].apply(lambda x: 1 if x in dates_to_mark else 0)

features = ['Date', 'Open', 'High', 'Low', 'Volume', 'one_hot']
target = ['Close']

X = df.loc[:, features].values
y = df.loc[:, target].values

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

def create_dataset(X, y, look_back=1):
    X_seq, y_seq = [], []
    for i in range(len(X) - look_back):
        X_seq.append(X[i:(i + look_back)])
        y_seq.append(y[i + look_back])
    return np.array(X_seq), np.array(y_seq)

look_back = 30
X_seq, y_seq = create_dataset(X_scaled, y_scaled, look_back)

split_index = int(len(X_seq) * 0.75)
X_train_seq, X_test_seq = X_seq[:split_index], X_seq[split_index:]
y_train_seq, y_test_seq = y_seq[:split_index], y_seq[split_index:]

batch_size = 96
train_data = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq)).batch(batch_size, drop_remainder=True)
test_data = tf.data.Dataset.from_tensor_slices((X_test_seq, y_test_seq)).batch(batch_size, drop_remainder=True)

tf.random.set_seed(42)
model = keras.Sequential([
    layers.LSTM(512, activation='relu', input_shape=(look_back, X_scaled.shape[1]), return_sequences=True),
    layers.LSTM(256, activation='relu', return_sequences=True),
    layers.LSTM(128, activation='relu', return_sequences=True),
    layers.LSTM(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=[MeanSquaredError(), MeanAbsoluteError(), R2Score()])

history = model.fit(train_data, epochs=10, validation_data=test_data, verbose=1)

loss, mse, mae, r2 = model.evaluate(test_data)
print(f"Test loss: {loss}")
print(f"Test MAE: {mae}")
print(f"Test R² Score: {r2}")

y_pred = model.predict(X_test_seq)
r2 = sklearn.metrics.r2_score(y_test_seq, y_pred)
print(f"R² Score: {r2}")

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=window, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(span=window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI"] = calculate_rsi(df["Close"], window=14)

features = ['Date', 'Open', 'High', 'Low', 'Volume', 'one_hot', 'RSI']
X = df.loc[:, features].values
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_seq, y_seq = create_dataset(X_scaled, y_scaled, look_back)

split_index = int(len(X_seq) * 0.75)
X_train_seq, X_test_seq = X_seq[:split_index], X_seq[split_index:]
y_train_seq, y_test_seq = y_seq[:split_index], y_seq[split_index:]

model2 = keras.Sequential([
    layers.LSTM(64, activation='relu', input_shape=(look_back, X_scaled.shape[1])),
    layers.Dense(1)
])
model2.compile(optimizer='adam', loss='mean_squared_error',
               metrics=[MeanSquaredError(), MeanAbsoluteError(), R2Score()])

history2 = model2.fit(X_train_seq, y_train_seq, epochs=10,
                      validation_data=(X_test_seq, y_test_seq), batch_size=batch_size, verbose=1)

y_pred2 = model2.predict(X_test_seq)
y_pred2_inv = scaler_y.inverse_transform(y_pred2)
real_prices = scaler_y.inverse_transform(y_test_seq)

dates_test = df.iloc[split_index+look_back:split_index+look_back+len(y_test_seq)]['Date']
plt.figure(figsize=(12, 6))
plt.plot(dates_test, real_prices, label='Real Close Price', color='blue')
plt.plot(dates_test, y_pred2_inv, label='Predicted Close Price', color='orange')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction vs Actual Prices')
plt.xticks(rotation=45)
plt.legend()
plt.show()
