import pandas as pd
import numpy as np
import keras
from keras import layers
import sklearn.metrics
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, R2Score
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

n_input = 60  # liczba dni wstecznych do przewidywania
batch_size = 32  # rozmiar batcha

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

lr_schedule1 = CosineDecay(initial_learning_rate=0.001, decay_steps=1000, alpha=0.00001)
lr_schedule2 = CosineDecay(initial_learning_rate=0.001, decay_steps=1000, alpha=0.00001)

df = pd.read_csv('^spx_d.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
min_date = df['Date'].min()
df['Date'] = (df['Date'] - min_date).dt.days

def convert_to_days(date_str, min_date):
    return (pd.to_datetime(date_str) - min_date).days

dates_to_mark = ['2001-09-11', '2020-03-11', '1987-10-19', '2020-01-31', '2022-02-24', '2008-09-15']
dates_to_mark = [(pd.to_datetime(date) - min_date).days for date in dates_to_mark]
df['one_hot'] = df['Date'].apply(lambda x: 1 if x in dates_to_mark else 0)

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=window, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(span=window, adjust=False).mean()
    rs = avg_gain / (avg_loss + np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI"] = calculate_rsi(df["Close"], window=14)

features_with_onehot = ['Date', 'Open', 'High', 'Low', 'Volume', 'one_hot', 'RSI']
features_without_onehot = ['Date', 'Open', 'High', 'Low', 'Volume', 'RSI']
target = ['Close']

X = df.loc[:, features_with_onehot].values
y = df.loc[:, target].values
X_without_onehot = df.loc[:, features_without_onehot].values

scaler_X = StandardScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
X_scaled_without_onehot = scaler_X.fit_transform(X_without_onehot)
y_scaled = scaler_y.fit_transform(y)

split_index = int(len(X_scaled) * 0.75)
X_train_seq, X_test_seq = X_scaled[:split_index], X_scaled[split_index:]
X_train_seq_without_onehot, X_test_seq_without_onehot = X_scaled_without_onehot[:split_index], X_scaled_without_onehot[split_index:]
y_train_seq, y_test_seq = y_scaled[:split_index], y_scaled[split_index:]


train_generator = TimeseriesGenerator(X_train_seq, y_train_seq, length=n_input, batch_size=batch_size)
test_generator = TimeseriesGenerator(X_test_seq, y_test_seq, length=n_input, batch_size=batch_size)
train_generator_without_onehot = TimeseriesGenerator(X_train_seq_without_onehot, y_train_seq, length=n_input, batch_size=batch_size)
test_generator_without_onehot = TimeseriesGenerator(X_test_seq_without_onehot, y_test_seq, length=n_input, batch_size=batch_size)

# Model z one-hot encoding
model1 = keras.Sequential([
    layers.LSTM(512, activation='tanh', return_sequences=True, input_shape=(n_input, X_scaled.shape[1])),
    layers.Dropout(0.3),
    layers.LSTM(256, activation='tanh', return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(128, activation='tanh', return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(64, activation='tanh'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model2 = Sequential([
    Bidirectional(LSTM(256, activation='tanh', return_sequences=True), input_shape=(n_input, X_scaled.shape[1])),
    Dropout(0.3),
    Bidirectional(LSTM(128, activation='tanh', return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64, activation='tanh')),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Model bez one-hot encoding
model2_without_onehot = Sequential([
    Bidirectional(LSTM(256, activation='tanh', return_sequences=True), input_shape=(n_input, X_scaled_without_onehot.shape[1])),
    Dropout(0.3),
    Bidirectional(LSTM(128, activation='tanh', return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64, activation='tanh')),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

optimizer1 = AdamW(learning_rate=lr_schedule1, weight_decay=0.001)
optimizer2 = AdamW(learning_rate=lr_schedule2, weight_decay=0.001)

model1.compile(optimizer=optimizer1, loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), R2Score()])
model2.compile(optimizer=optimizer2, loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), R2Score()])
model2_without_onehot.compile(optimizer=optimizer2, loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), R2Score()])

history1 = model1.fit(train_generator, epochs=10, validation_data=test_generator,
                     callbacks=[early_stopping, reduce_lr], verbose=1)
history2 = model2.fit(train_generator, epochs=10, validation_data=test_generator,
                     callbacks=[early_stopping, reduce_lr], verbose=1)
history2_without_onehot = model2_without_onehot.fit(train_generator_without_onehot, epochs=10,
                                                   validation_data=test_generator_without_onehot,
                                                   callbacks=[early_stopping, reduce_lr], verbose=1)

y_pred1 = model1.predict(test_generator)
y_pred2 = model2.predict(test_generator)
y_pred2_without_onehot = model2_without_onehot.predict(test_generator_without_onehot)

y_pred1_inv = scaler_y.inverse_transform(y_pred1)
y_pred2_inv = scaler_y.inverse_transform(y_pred2)
y_pred2_inv_without_onehot = scaler_y.inverse_transform(y_pred2_without_onehot)
real_prices = scaler_y.inverse_transform(y_test_seq[n_input:])

dates_test = df.iloc[split_index + n_input:split_index + n_input + len(y_test_seq[n_input:])]['Date']
dates_test = min_date + pd.to_timedelta(dates_test, unit='D')

plt.figure(figsize=(15, 8))
plt.plot(dates_test, real_prices, label='Real Price', color='blue', linewidth=2)
plt.plot(dates_test, y_pred1_inv, label='LSTM Model', color='orange', linewidth=2)
plt.plot(dates_test, y_pred2_inv, label='Bidirectional LSTM with One-Hot', color='red', linewidth=2)
plt.plot(dates_test, y_pred2_inv_without_onehot, label='Bidirectional LSTM without One-Hot', color='green', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Comparison of Different Models')
plt.legend()
plt.grid(True)
plt.show()


# Wykresy pokazujące porównanie z one hot encoding, bez one hot encoding oraz z rzeczywistymi wartościami
# Wykres dla WTC
filtered_df = df.loc[
    (df['Date'] >= convert_to_days('2001-09-01', min_date)) &
    (df['Date'] < convert_to_days('2001-12-11', min_date))
]
plt.figure(figsize=(15, 8))
dates_test_filtered = min_date + pd.to_timedelta(filtered_df['Date'], unit='D')
real_prices_filtered = scaler_y.inverse_transform(y_test_seq[:len(filtered_df)])
plt.plot(dates_test_filtered, real_prices_filtered, label='Real Price', color='blue', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv[:len(filtered_df)], label='With One-Hot Encoding', color='red', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv_without_onehot[:len(filtered_df)], label='Without One-Hot Encoding', color='green', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction for WTC')
plt.legend()
plt.grid(True)
plt.show()

# Wykres dla Black Monday
filtered_df2 = df.loc[
    (df['Date'] >= convert_to_days('1987-10-08', min_date)) &
    (df['Date'] < convert_to_days('1988-01-19', min_date))
]
plt.figure(figsize=(15, 8))
dates_test_filtered = min_date + pd.to_timedelta(filtered_df2['Date'], unit='D')
real_prices_filtered2 = scaler_y.inverse_transform(y_test_seq[:len(filtered_df2)])
plt.plot(dates_test_filtered, real_prices_filtered2, label='Real Price', color='blue', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv[:len(filtered_df2)], label='With One-Hot Encoding', color='red', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv_without_onehot[:len(filtered_df2)], label='Without One-Hot Encoding', color='green', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction for Black Monday')
plt.legend()
plt.grid(True)
plt.show()

# Wykres dla Covid-19
filtered_df3 = df.loc[
    (df['Date'] >= convert_to_days('2020-03-01', min_date)) &
    (df['Date'] < convert_to_days('2020-06-11', min_date))
]
plt.figure(figsize=(15, 8))
dates_test_filtered = min_date + pd.to_timedelta(filtered_df3['Date'], unit='D')
real_prices_filtered3 = scaler_y.inverse_transform(y_test_seq[:len(filtered_df3)])
plt.plot(dates_test_filtered, real_prices_filtered3, label='Real Price', color='blue', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv[:len(filtered_df3)], label='With One-Hot Encoding', color='red', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv_without_onehot[:len(filtered_df3)], label='Without One-Hot Encoding', color='green', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction for Covid-19')
plt.legend()
plt.grid(True)
plt.show()

# Wykres dla Brexitu
filtered_df4 = df.loc[
    (df['Date'] >= convert_to_days('2020-01-20', min_date)) &
    (df['Date'] < convert_to_days('2020-04-30', min_date))
]
plt.figure(figsize=(15, 8))
dates_test_filtered = min_date + pd.to_timedelta(filtered_df4['Date'], unit='D')
real_prices_filtered4 = scaler_y.inverse_transform(y_test_seq[:len(filtered_df4)])
plt.plot(dates_test_filtered, real_prices_filtered4, label='Real Price', color='blue', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv[:len(filtered_df4)], label='With One-Hot Encoding', color='red', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv_without_onehot[:len(filtered_df4)], label='Without One-Hot Encoding', color='green', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction for Brexit')
plt.legend()
plt.grid(True)
plt.show()

# Wykres dla wojny na Ukrainie
filtered_df5 = df.loc[
    (df['Date'] >= convert_to_days('2022-02-13', min_date)) &
    (df['Date'] < convert_to_days('2022-02-24', min_date))
]
plt.figure(figsize=(15, 8))
dates_test_filtered = min_date + pd.to_timedelta(filtered_df5['Date'], unit='D')
real_prices_filtered5 = scaler_y.inverse_transform(y_test_seq[:len(filtered_df5)])
plt.plot(dates_test_filtered, real_prices_filtered5, label='Real Price', color='blue', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv[:len(filtered_df5)], label='With One-Hot Encoding', color='red', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv_without_onehot[:len(filtered_df5)], label='Without One-Hot Encoding', color='green', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction for War in Ukraine')
plt.legend()
plt.grid(True)
plt.show()

# Wykres dla upadku Lehman Brothers
filtered_df6 = df.loc[
    (df['Date'] >= convert_to_days('2008-09-05', min_date)) &
    (df['Date'] < convert_to_days('2008-12-06', min_date))
]
plt.figure(figsize=(15, 8))
dates_test_filtered = min_date + pd.to_timedelta(filtered_df6['Date'], unit='D')
real_prices_filtered6 = scaler_y.inverse_transform(y_test_seq[:len(filtered_df6)])
plt.plot(dates_test_filtered, real_prices_filtered6, label='Real Price', color='blue', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv[:len(filtered_df6)], label='With One-Hot Encoding', color='red', linewidth=2)
plt.plot(dates_test_filtered, y_pred2_inv_without_onehot[:len(filtered_df6)], label='Without One-Hot Encoding', color='green', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction for the fall of Lehman Brothers')
plt.legend()
plt.grid(True)
plt.show()

# Metryki porównawcze
print("\nPorównanie metryk dla obu wersji:")
print("With One-Hot Encoding:")
print(f"R2 Score: {model2.evaluate(test_generator, verbose=0)[2]}")
print(f"Mean Squared Error: {model2.evaluate(test_generator, verbose=0)[0]}")
print(f"Mean Absolute Error: {model2.evaluate(test_generator, verbose=0)[1]}")

print("\nWithout One-Hot Encoding:")
print(f"R2 Score: {model2_without_onehot.evaluate(test_generator_without_onehot, verbose=0)[2]}")
print(f"Mean Squared Error: {model2_without_onehot.evaluate(test_generator_without_onehot, verbose=0)[0]}")
print(f"Mean Absolute Error: {model2_without_onehot.evaluate(test_generator_without_onehot, verbose=0)[1]}")

df2 = pd.read_csv('^dji_d.csv')
df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
min_date_dji = df2['Date'].min()
df2['Date'] = (df2['Date'] - min_date_dji).dt.days

X_dji = df2.loc[:, features_with_onehot].values
y_dji = df2.loc[:, target].values

scaler_X_dji = StandardScaler()
scaler_y_dji = MinMaxScaler()
X_dji_scaled = scaler_X_dji.fit_transform(X_dji)
y_dji_scaled = scaler_y_dji.fit_transform(y_dji)

split_index_dji = int(len(X_dji_scaled) * 0.75)
X_train_dji, X_test_dji = X_dji_scaled[:split_index_dji], X_dji_scaled[split_index_dji:]
y_train_dji, y_test_dji = y_dji_scaled[:split_index_dji], y_dji_scaled[split_index_dji:]

train_generator_dji = TimeseriesGenerator(X_train_dji, y_train_dji, length=n_input, batch_size=batch_size)
test_generator_dji = TimeseriesGenerator(X_test_dji, y_test_dji, length=n_input, batch_size=batch_size)

model3 = keras.Sequential([
    layers.LSTM(512, activation='tanh', return_sequences=True, input_shape=(n_input, X_dji_scaled.shape[1])),
    layers.Dropout(0.3),
    layers.LSTM(256, activation='tanh', return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(128, activation='tanh', return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(64, activation='tanh'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

lr_schedule4 = CosineDecay(initial_learning_rate=0.0005, decay_steps=1000, alpha=0.00001)
optimizer4 = AdamW(learning_rate=lr_schedule4, weight_decay=0.001)
model3.compile(optimizer=optimizer4, loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), R2Score()])

history4 = model3.fit(train_generator_dji, epochs=20, validation_data=test_generator_dji, 
                     callbacks=[early_stopping, reduce_lr], verbose=1)

y_pred_dji = model3.predict(test_generator_dji)
y_pred_dji_inv = scaler_y_dji.inverse_transform(y_pred_dji)
real_prices_dji = scaler_y_dji.inverse_transform(y_test_dji[n_input:])

dates_test_dji = df2.iloc[split_index_dji + n_input:split_index_dji + n_input + len(y_test_dji[n_input:])]['Date']
dates_test_dji = min_date_dji + pd.to_timedelta(dates_test_dji, unit='D')

# Wykres porównawczy DJIA
plt.figure(figsize=(12, 6))
plt.plot(dates_test_dji, real_prices_dji, label='Real DJI Price', color='blue')
plt.plot(dates_test_dji, y_pred_dji_inv, label='Predicted DJI Price', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('DJI Stock Price Prediction')
plt.legend()
plt.show()

# Covid-19 DJIA
filtered_df_dji = df2.loc[
    (df2['Date'] >= convert_to_days('2020-03-01', min_date_dji)) &
    (df2['Date'] < convert_to_days('2020-06-11', min_date_dji))
]

plt.figure(figsize=(10,6))
dates_test_filtered_dji = min_date_dji + pd.to_timedelta(filtered_df_dji['Date'], unit='D')
plt.plot(dates_test_filtered_dji, real_prices_dji[:len(filtered_df_dji)], label='Real DJI Prices', color='brown')
plt.plot(dates_test_filtered_dji, y_pred_dji_inv[:len(filtered_df_dji)], label='Predicted DJI Prices', color='yellow')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('DJI Stock Price Prediction for Covid-19')
plt.legend()
plt.show()

