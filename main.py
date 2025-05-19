import pandas as pd
import numpy as np
import keras
from keras import layers
import sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, SimpleRNN
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, R2Score
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from scipy import stats
import os

# Wczytywanie danych SPX
df = pd.read_csv('^spx_d.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
min_date = df['Date'].min()
df['Date'] = (df['Date'] - min_date).dt.days  # Przekształcanie daty do liczby dni

# Wczytywanie danych DJIA
df2 = pd.read_csv('^dji_d.csv')
df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
min_date_dji = df2['Date'].min()
df2['Date'] = (df2['Date'] - min_date_dji).dt.days  # Przekształcanie daty do liczby dni

# Konfiguracja callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

callbacks = [early_stopping, reduce_lr]

def mse_entropy(data, m=5, tau=2, scale=1):
    """
    Oblicza entropię MSE (Multiscale Entropy) dla danych czasowych.
    
    Parametry:
    data: array-like, dane wejściowe
    m: int, długość wzorca
    tau: int, opóźnienie czasowe
    scale: int, skala czasowa
    
    Zwraca:
    float: wartość entropii MSE
    """
    scaled_data = data[::scale]
    returns = np.diff(scaled_data) / scaled_data[:-1]
    returns = (returns - np.mean(returns)) / np.std(returns)
    
    patterns = []
    for i in range(len(returns) - (m-1)*tau):
        pattern = returns[i:i + m*tau:tau]
        if len(pattern) == m:
            patterns.append(pattern)
    
    if not patterns:
        return 0
    
    patterns = np.array(patterns)
    permutations = []
    
    for pattern in patterns:
        sorted_indices = np.argsort(pattern)
        perm = np.zeros_like(sorted_indices)
        perm[sorted_indices] = np.arange(len(sorted_indices))
        permutations.append(perm)
    
    unique_perms, counts = np.unique(permutations, axis=0, return_counts=True)
    probs = counts / len(permutations)
    entropy = -np.sum(probs * np.log(probs))
    
    return entropy

def market_complexity(close_prices, window=20):
    epsilon = 1e-10  # Dodanie definicji epsilon na początku funkcji
    complexities = []
    close_prices = np.array(close_prices)
    
    for i in range(window, len(close_prices)):
        window_data = close_prices[i-window:i]
        if len(window_data) > 1:
            # Obliczamy logarytmiczne stopy zwrotu
            log_returns = np.log(window_data[1:] / window_data[:-1])
            log_returns = log_returns[np.isfinite(log_returns)]
            
            if len(log_returns) > 0:
                # Złożoność jako kombinacja różnych metryk
                volatility = np.std(log_returns)
                mean_return = np.abs(np.mean(log_returns))
                skewness = stats.skew(log_returns) if len(log_returns) > 2 else 0
                kurtosis = stats.kurtosis(log_returns) if len(log_returns) > 2 else 0
                
                # Zabezpieczenie przed zerowymi wartościami
                volatility = max(volatility, epsilon)
                mean_return = max(mean_return, epsilon)
                
                # Nowa formuła złożoności
                complexity = (
                    volatility * 
                    (1 + np.abs(mean_return)) * 
                    (1 + np.abs(skewness)) * 
                    (1 + np.abs(kurtosis)) * 
                    np.log1p(len(log_returns))
                )
                complexities.append(complexity)
            else:
                complexities.append(epsilon)
        else:
            complexities.append(epsilon)
    
    # Normalizacja wyników
    complexities = np.array(complexities)
    if np.std(complexities) > epsilon:
        complexities = (complexities - np.mean(complexities)) / np.std(complexities)
    
    return complexities

# Definicja skal i list do przechowywania entropii
scales = [1, 2, 4, 8, 16]
spx_entropies = []
dji_entropies = []

# Obliczenie entropii dla surowych danych
for scale in scales:
    spx_entropies.append(mse_entropy(df['Close'].values, m=5, tau=2, scale=scale))
    dji_entropies.append(mse_entropy(df2['Close'].values, m=5, tau=2, scale=scale))

# Wykres entropii MSE
plt.figure(figsize=(12, 6))
plt.plot(scales, spx_entropies, 'b-o', label='SPX')
plt.plot(scales, dji_entropies, 'r-o', label='DJIA')
plt.xlabel('Skala czasowa')
plt.ylabel('Entropia MSE')
plt.title('Entropia MSE dla różnych skal czasowych')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Analiza statystyczna dla entropii MSE
print("\nAnaliza entropii MSE:")
print("\nSPX:")
print(f"Średnia entropia: {np.mean(spx_entropies):.4f}")
print(f"Odchylenie standardowe: {np.std(spx_entropies):.4f}")
print(f"Maksymalna entropia: {np.max(spx_entropies):.4f}")
print(f"Minimalna entropia: {np.min(spx_entropies):.4f}")

print("\nDJIA:")
print(f"Średnia entropia: {np.mean(dji_entropies):.4f}")
print(f"Odchylenie standardowe: {np.std(dji_entropies):.4f}")
print(f"Maksymalna entropia: {np.max(dji_entropies):.4f}")
print(f"Minimalna entropia: {np.min(dji_entropies):.4f}")

# Analiza dla podokresów
periods = [
    ('Cały okres', 0, len(df)),
    ('Pierwsza połowa', 0, len(df)//2),
    ('Druga połowa', len(df)//2, len(df)),
    ('Ostatni rok', -252, len(df))
]

print("\nAnaliza dla podokresów:")
for period_name, start, end in periods:
    period_spx = df['Close'].values[start:end]
    period_dji = df2['Close'].values[start:end]
    
    print(f"\n{period_name}:")
    print("Skala\tSPX\t\tDJIA\t\tRóżnica")
    print("-" * 50)
    
    for scale in scales:
        spx_ent = mse_entropy(period_spx, m=5, tau=2, scale=scale)
        dji_ent = mse_entropy(period_dji, m=5, tau=2, scale=scale)
        diff = abs(spx_ent - dji_ent)
        print(f"{scale}\t{spx_ent:.4f}\t\t{dji_ent:.4f}\t\t{diff:.4f}")

# Obliczanie złożoności dla SPX i DJIA
spx_complexity = np.array(market_complexity(df['Close'].values))
dji_complexity = np.array(market_complexity(df2['Close'].values))

# Wypisanie wyników złożoności
print(f"\nŚrednia złożoność SPX: {np.mean(spx_complexity):.4f}")
print(f"Średnia złożoność DJIA: {np.mean(dji_complexity):.4f}")

# Funkcja do obliczania metryk błędu
def calculate_error_metrics(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

# Analiza dla podokresów z metrykami błędu
print("\nAnaliza metryk błędu dla podokresów:")
all_metrics = []

for period_name, start, end in periods:
    period_spx = spx_complexity[start:end]
    period_dji = dji_complexity[start:end]
    
    # Obliczanie metryk błędu dla danego okresu
    metrics = calculate_error_metrics(period_spx, period_dji)
    all_metrics.append(metrics)
    
    print(f"\n{period_name}:")
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

# Obliczanie średnich metryk dla wszystkich okresów
print("\nŚrednie metryki błędu dla wszystkich okresów:")
avg_mse = np.mean([m['MSE'] for m in all_metrics])
avg_mae = np.mean([m['MAE'] for m in all_metrics])
avg_rmse = np.mean([m['RMSE'] for m in all_metrics])
avg_mape = np.mean([m['MAPE'] for m in all_metrics])

print(f"Średnie MSE: {avg_mse:.6f}")
print(f"Średnie MAE: {avg_mae:.6f}")
print(f"Średnie RMSE: {avg_rmse:.6f}")
print(f"Średnie MAPE: {avg_mape:.2f}%")


# Interpretacja wyników
print("\nInterpretacja wyników:")
print("1. Porównanie indeksów:")
if np.mean(np.abs(np.array(spx_entropies) - np.array(dji_entropies))) < 0.1:
    print("   - Indeksy wykazują podobny poziom entropii w różnych skalach czasowych")
else:
    print("   - Indeksy wykazują różny poziom entropii w różnych skalach czasowych")
    if np.mean(spx_entropies) > np.mean(dji_entropies):
        print("   - SPX wykazuje większą entropię niż DJIA")
    else:
        print("   - DJIA wykazuje większą entropię niż SPX")

print("\n2. Analiza skal czasowych:")
max_spx_scale = scales[np.argmax(spx_entropies)]
max_dji_scale = scales[np.argmax(dji_entropies)]
print(f"   - Maksymalna entropia SPX występuje w skali {max_spx_scale}")
print(f"   - Maksymalna entropia DJIA występuje w skali {max_dji_scale}")

print("\n3. Wnioski:")
if np.mean(spx_entropies) > 2 and np.mean(dji_entropies) > 2:
    print("   - Oba indeksy wykazują wysoką entropię")
    print("   - Trudno przewidywać ich zachowanie w różnych skalach czasowych")
elif np.mean(spx_entropies) < 1 and np.mean(dji_entropies) < 1:
    print("   - Oba indeksy wykazują niską entropię")
    print("   - Wzorce są bardziej przewidywalne w różnych skalach czasowych")
else:
    print("   - Indeksy wykazują umiarkowaną entropię")
    print("   - Wzorce są częściowo przewidywalne w różnych skalach czasowych")

# Wyświetlenie wyników
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][20:len(spx_complexity)+20], spx_complexity, label='SPX', alpha=0.7)  # Dostosowanie indeksów
plt.plot(df2['Date'][20:len(dji_complexity)+20], dji_complexity, label='DJIA', alpha=0.7)  # Dostosowanie indeksów
plt.xlabel('Data')
plt.ylabel('Złożoność')
plt.title('Złożoność rynku SPX i DJIA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Analiza statystyczna
print("\nAnaliza złożoności rynku:")
print("\nSPX:")
print(f"Średnia złożoność: {spx_complexity.mean():.4f}")
print(f"Odchylenie standardowe: {spx_complexity.std():.4f}")
print(f"Maksymalna złożoność: {spx_complexity.max():.4f}")
print(f"Minimalna złożoność: {spx_complexity.min():.4f}")

print("\nDJIA:")
print(f"Średnia złożoność: {dji_complexity.mean():.4f}")
print(f"Odchylenie standardowe: {dji_complexity.std():.4f}")
print(f"Maksymalna złożoność: {dji_complexity.max():.4f}")
print(f"Minimalna złożoność: {dji_complexity.min():.4f}")



# Analiza dla podokresów
periods = [
    ('Cały okres', 0, len(spx_complexity)),
    ('Pierwsza połowa', 0, len(spx_complexity)//2),
    ('Druga połowa', len(spx_complexity)//2, len(spx_complexity)),
    ('Ostatni rok', -252, len(spx_complexity))
]

print("\nAnaliza dla podokresów:")
for period_name, start, end in periods:
    period_spx = spx_complexity[start:end]
    period_dji = dji_complexity[start:end]
    
    print(f"\n{period_name}:")
    print(f"SPX - średnia złożoność: {period_spx.mean():.4f}")
    print(f"DJIA - średnia złożoność: {period_dji.mean():.4f}")
    print(f"Różnica: {abs(period_spx.mean() - period_dji.mean()):.4f}")

# Obliczenie stóp zwrotu - poprawiona wersja
spx_returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
dji_returns = np.log(df2['Close'] / df2['Close'].shift(1)).fillna(0)

# Obliczenie zmienności (volatility) na dłuższym oknie
window = 20  # zwiększamy okno do 20 dni
spx_vol = spx_returns.rolling(window=window).std().fillna(method='bfill')
dji_vol = dji_returns.rolling(window=window).std().fillna(method='bfill')

# Normalizacja stóp zwrotu przez zmienność z dodanym małym epsilon
epsilon = 1e-10
spx_normalized = (spx_returns / (spx_vol + epsilon)).fillna(0)
dji_normalized = (dji_returns / (dji_vol + epsilon)).fillna(0)

# Obliczenie miary złożoności dla różnych skal czasowych
scales = [5, 10, 20, 40, 60]  # zmienione skale czasowe
spx_complexities = []
dji_complexities = []

for scale in scales:
    # Obliczanie złożoności dla każdej skali z większym oknem
    spx_complex = np.mean(market_complexity(spx_normalized.values, window=scale))
    dji_complex = np.mean(market_complexity(dji_normalized.values, window=scale))
    spx_complexities.append(spx_complex)
    dji_complexities.append(dji_complex)

# Konwersja list na numpy arrays przed rysowaniem
spx_complexities = np.array(spx_complexities)
dji_complexities = np.array(dji_complexities)

# Wykres miary złożoności
plt.figure(figsize=(12, 6))
plt.plot(scales, spx_complexities, 'b-o', label='SPX')
plt.plot(scales, dji_complexities, 'r-o', label='DJIA')
plt.xlabel('Skala czasowa')
plt.ylabel('Miar złożoności')
plt.title('Miar złożoności dla znormalizowanych stóp zwrotu')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Interpretacja wyników
print("\nInterpretacja wyników:")
print("\n1. Miar złożoności dla stóp zwrotu:")
print("   - Wartości dla SPX:", [f"{x:.4f}" for x in spx_complexities])
print("   - Wartości dla DJIA:", [f"{x:.4f}" for x in dji_complexities])
print("   - Różnice między indeksami:", [f"{abs(s-d):.4f}" for s, d in zip(spx_complexities, dji_complexities)])

# Analiza zmienności
print("\n2. Analiza zmienności:")
print(f"   - Średnia zmienność SPX: {spx_vol.mean():.4f}")
print(f"   - Średnia zmienność DJIA: {dji_vol.mean():.4f}")
print(f"   - Stosunek zmienności (SPX/DJIA): {spx_vol.mean()/dji_vol.mean():.4f}")

import matplotlib.pyplot as plt
from scipy import stats

# Funkcja do obliczania macierzy korelacji krzyżowej
def cross_correlation_matrix(x, y, max_lag, window_size=100):  # Zwiększone okno czasowe
    matrix = np.zeros((2 * max_lag + 1, 2 * max_lag + 1))
    lags = range(-max_lag, max_lag + 1)

    # Obliczenie logarytmicznych stóp zwrotu z większą precyzją
    x_returns = np.log(x[1:] / x[:-1])
    y_returns = np.log(y[1:] / y[:-1])

    # Usuwanie wartości odstających
    x_mean, x_std = np.mean(x_returns), np.std(x_returns)
    y_mean, y_std = np.mean(y_returns), np.std(y_returns)
    
    x_returns = np.where(np.abs(x_returns - x_mean) <= 3 * x_std, x_returns, np.nan)
    y_returns = np.where(np.abs(y_returns - y_mean) <= 3 * y_std, y_returns, np.nan)
    
    # Normalizacja z obsługą wartości NaN
    x_returns = (x_returns - np.nanmean(x_returns)) / np.nanstd(x_returns)
    y_returns = (y_returns - np.nanmean(y_returns)) / np.nanstd(y_returns)

    for i, lag1 in enumerate(lags):
        for j, lag2 in enumerate(lags):
            x_shifted, y_shifted = shift_series(x_returns, y_returns, lag1, lag2)
            
            if len(x_shifted) > window_size:
                # Używamy mniejszego kroku dla większej dokładności
                step = window_size // 4
                rolling_corrs = []
                
                for k in range(0, len(x_shifted) - window_size + 1, step):
                    window_x = x_shifted[k:k + window_size]
                    window_y = y_shifted[k:k + window_size]
                    
                    if len(window_x) == window_size and len(window_y) == window_size:
                        # Usuwanie NaN przed obliczeniem korelacji
                        valid_idx = ~np.isnan(window_x) & ~np.isnan(window_y)
                        if np.sum(valid_idx) > window_size * 0.8:  # Minimum 80% ważnych danych
                            try:
                                corr = np.corrcoef(window_x[valid_idx], window_y[valid_idx])[0, 1]
                                if np.isfinite(corr):
                                    rolling_corrs.append(corr)
                            except:
                                continue
                
                if rolling_corrs:
                    # Ważona średnia korelacji
                    weights = np.linspace(0.5, 1.0, len(rolling_corrs))
                    matrix[i, j] = np.average(rolling_corrs, weights=weights)
                else:
                    matrix[i, j] = 0
            else:
                valid_idx = ~np.isnan(x_shifted) & ~np.isnan(y_shifted)
                if np.sum(valid_idx) > len(x_shifted) * 0.8:
                    try:
                        corr = np.corrcoef(x_shifted[valid_idx], y_shifted[valid_idx])[0, 1]
                        matrix[i, j] = corr if np.isfinite(corr) else 0
                    except:
                        matrix[i, j] = 0
                else:
                    matrix[i, j] = 0

    return matrix

    # Obliczenie zwrotów (returns)
    x_returns = np.diff(x) / x[:-1]
    y_returns = np.diff(y) / y[:-1]

    # Normalizacja zwrotów
    x_returns = (x_returns - np.mean(x_returns)) / np.std(x_returns)
    y_returns = (y_returns - np.mean(y_returns)) / np.std(y_returns)

    for i, lag1 in enumerate(lags):
        for j, lag2 in enumerate(lags):
            x_shifted, y_shifted = shift_series(x_returns, y_returns, lag1, lag2)
            if len(x_shifted) > window_size:
                rolling_corr = [
                    np.corrcoef(x_shifted[k:k + window_size], y_shifted[k:k + window_size])[0, 1]
                    for k in range(len(x_shifted) - window_size + 1)
                ]
                matrix[i, j] = np.mean(rolling_corr)
            else:
                matrix[i, j] = np.corrcoef(x_shifted, y_shifted)[0, 1]
    return matrix

# Funkcja pomocnicza do przesuwania serii czasowych
def shift_series(x, y, lag1, lag2):
    if lag1 < 0:
        x = x[-lag1:]
        y = y[:len(x)]
    elif lag1 > 0:
        x = x[:-lag1]
        y = y[lag1:len(x) + lag1]

    if lag2 < 0:
        x = x[-lag2:]
        y = y[:len(x)]
    elif lag2 > 0:
        x = x[:-lag2]
        y = y[lag2:len(x) + lag2]

    return x, y

# Funkcja do obliczania korelacji krzyżowej dla opóźnień
def simple_cross_correlation(x, y, max_lag=30):
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    for lag in lags:
        x_shifted, y_shifted = shift_series(x, y, lag, 0)
        min_len = min(len(x_shifted), len(y_shifted))
        x_shifted = x_shifted[:min_len]
        y_shifted = y_shifted[:min_len]
        correlations.append(np.corrcoef(x_shifted, y_shifted)[0, 1])
    return correlations, lags

# Obliczenie macierzy korelacji krzyżowej
# Zwiększenie zakresu opóźnień
max_lag = 60  # Zwiększony zakres opóźnień

# Obliczenie macierzy korelacji krzyżowej
correlation_matrix = cross_correlation_matrix(df['Close'].values, df2['Close'].values, max_lag=max_lag)

# Wykres macierzy korelacji krzyżowej
plt.figure(figsize=(15, 12))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Współczynnik korelacji')
plt.title('Macierz korelacji krzyżowej SPX-DJIA')
plt.xlabel('Opóźnienie DJIA')
plt.ylabel('Opóźnienie SPX')
plt.xticks(range(0, 2 * max_lag + 1, 10), range(-max_lag, max_lag + 1, 10))
plt.yticks(range(0, 2 * max_lag + 1, 10), range(-max_lag, max_lag + 1, 10))

# Dodanie siatki
plt.grid(True, alpha=0.2)

# Dodanie linii przekątnej
plt.axline([max_lag, max_lag], [0, 0], color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# Test istotności korelacji
def correlation_significance_test(corr, length):
    t_stat = corr * np.sqrt((length - 2) / (1 - corr ** 2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), length - 2))
    return t_stat, p_value

# Obliczenie korelacji krzyżowej i lagów przed tworzeniem wykresów
correlations, lags = simple_cross_correlation(df['Close'].values, df2['Close'].values, max_lag=30)

# Wykres korelacji krzyżowej
plt.figure(figsize=(12, 8))
plt.plot(lags, correlations, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)

# Dodanie poziomych linii dla poziomów istotności
plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.2, label='Umiarkowana korelacja')
plt.axhline(y=-0.5, color='g', linestyle='--', alpha=0.2)
plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.2, label='Silna korelacja')
plt.axhline(y=-0.7, color='r', linestyle='--', alpha=0.2)

plt.xlabel('Opóźnienie (dni)')
plt.ylabel('Współczynnik korelacji')
plt.title('Korelacja krzyżowa między SPX a DJIA')
plt.grid(True, alpha=0.3)

# Dodanie adnotacji dla maksymalnej korelacji
max_corr_idx = np.argmax(np.abs(correlations))
max_corr_lag = lags[max_corr_idx]
max_corr_val = correlations[max_corr_idx]
plt.annotate(f'Max korelacja: {max_corr_val:.3f}\nOpóźnienie: {max_corr_lag} dni',
             xy=(max_corr_lag, max_corr_val),
             xytext=(10, 10),
             textcoords='offset points',
             ha='left',
             va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.legend()
plt.tight_layout()
plt.show()

# Dodanie pionowych linii dla dat czarnych łabędzi
black_swan_dates = [
    '2001-09-11',  # Atak na WTC
    '2020-03-11',  # Pandemia COVID-19
    '1987-10-19',  # Czarny Poniedziałek
    '2020-01-31',  # Brexit
    '2022-02-24',  # Wojna na Ukrainie
    '2008-09-15'   # Upadek Lehman Brothers
]

# Konwersja dat na format datetime
black_swan_dates = [pd.to_datetime(date) for date in black_swan_dates]
black_swan_dates_days = [(date - min_date).days for date in black_swan_dates]


# Dodanie linii i etykiet dla czarnych łabędzi
for i, (date, date_days) in enumerate(zip(black_swan_dates, black_swan_dates_days)):
    if date_days in df['Date'].values:
        plt.axvline(x=date_days, color='red', linestyle='-', alpha=0.3)
        
        # Znajdź odpowiednią cenę dla etykiety
        base_price = df.loc[df['Date'] == date_days, 'Close'].values[0]
        
        # Specjalne przesunięcie dla daty z Brexitu (indeks 3 w liście)
        if i == 3:  # Brexit
            price = base_price + 3000  # Bardzo duże przesunięcie dla Brexitu
        else:
            price = base_price
        
        plt.text(date_days, price, date.strftime('%Y-%m-%d'), 
                rotation=90, verticalalignment='bottom', 
                horizontalalignment='right', fontsize=8)

# Formatowanie osi X
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.xlabel('Data')
plt.ylabel('Cena zamknięcia')
plt.title('Porównanie cen zamknięcia SPX i DJIA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Figura 2: Wykresy korelacji
plt.figure(figsize=(20, 15))

# Wykres 1: Scatter plot
plt.subplot(2, 2, 1)
plt.scatter(df2['Close'], df['Close'], alpha=0.5)
plt.xlabel('DJIA Close Price')
plt.ylabel('SPX Close Price')
plt.title('Korelacja między DJIA a SPX')
plt.grid(True)

# Wykres 2: Histogram różnic przyrostów
plt.subplot(2, 2, 2)
returns_diff = df['Close'].pct_change().dropna() - df2['Close'].pct_change().dropna()
plt.hist(returns_diff, bins=50, alpha=0.7, density=True)
plt.xlabel('Różnica przyrostów (SPX - DJIA)')
plt.ylabel('Gęstość')
plt.title('Rozkład różnic przyrostów')
plt.grid(True)

# Dodanie krzywej normalnej
x = np.linspace(returns_diff.min(), returns_diff.max(), 100)
p = stats.norm.pdf(x, returns_diff.mean(), returns_diff.std())
plt.plot(x, p, 'k', linewidth=2)


# Obliczenie korelacji krzyżowej i lagów przed tworzeniem wykresów
correlations, lags = simple_cross_correlation(df['Close'].values, df2['Close'].values, max_lag=30)
max_corr = np.max(np.abs(correlations))

# Wykres korelacji krzyżowej
plt.figure(figsize=(12, 8))
plt.plot(lags, correlations, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)

# Dodanie poziomych linii dla poziomów istotności
plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.2, label='Umiarkowana korelacja')
plt.axhline(y=-0.5, color='g', linestyle='--', alpha=0.2)
plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.2, label='Silna korelacja')
plt.axhline(y=-0.7, color='r', linestyle='--', alpha=0.2)

plt.xlabel('Opóźnienie (dni)')
plt.ylabel('Współczynnik korelacji')
plt.title('Korelacja krzyżowa między SPX a DJIA')
plt.grid(True, alpha=0.3)

# Dodanie adnotacji dla maksymalnej korelacji
max_corr_idx = np.argmax(np.abs(correlations))
max_corr_lag = lags[max_corr_idx]
max_corr_val = correlations[max_corr_idx]
plt.annotate(f'Max korelacja: {max_corr_val:.3f}\nOpóźnienie: {max_corr_lag} dni',
             xy=(max_corr_lag, max_corr_val),
             xytext=(10, 10),
             textcoords='offset points',
             ha='left',
             va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.legend()
plt.tight_layout()
plt.show()

# Analiza siły korelacji
if max_corr > 0.7:
    print("Bardzo silna korelacja między indeksami")
elif max_corr > 0.5:
    print("Silna korelacja między indeksami")
elif max_corr > 0.3:
    print("Umiarkowana korelacja między indeksami")
else:
    print("Słaba korelacja między indeksami")


# Dodatkowa analiza korelacji
print("\nAnaliza korelacji dla różnych opóźnień:")
for lag in [-10, -5, 0, 5, 10]:
    if lag < 0:
        x_shifted = spx_normalized[-lag:]
        y_shifted = dji_normalized[:len(x_shifted)]
    elif lag > 0:
        x_shifted = spx_normalized[:-lag]
        y_shifted = dji_normalized[lag:len(x_shifted)+lag]
    else:
        x_shifted = spx_normalized
        y_shifted = dji_normalized[:len(spx_normalized)]
    
    # Ensure equal lengths
    min_len = min(len(x_shifted), len(y_shifted))
    x_shifted = x_shifted[:min_len]
    y_shifted = y_shifted[:min_len]
    
    # Calculate correlation
    corr = np.corrcoef(x_shifted, y_shifted)[0,1]
    print(f"Opóźnienie {lag} dni: {corr:.4f}")

# Analiza rozkładu różnic przyrostów
print("\nAnaliza rozkładu różnic przyrostów:")
print(f"Średnia różnica przyrostów: {returns_diff.mean():.6f}")
print(f"Odchylenie standardowe różnic: {returns_diff.std():.6f}")
print(f"Maksymalna różnica: {returns_diff.max():.6f}")
print(f"Minimalna różnica: {returns_diff.min():.6f}")
print(f"Skośność: {stats.skew(returns_diff):.4f}")
print(f"Kurtoza: {stats.kurtosis(returns_diff):.4f}")

plt.tight_layout()
plt.show()

# Dodatkowa analiza
print("\nDodatkowa analiza:")
print(f"Średnia różnica cen: {returns_diff.mean():.2f}")
print(f"Odchylenie standardowe różnic: {returns_diff.std():.2f}")
print(f"Maksymalna różnica: {returns_diff.max():.2f}")
print(f"Minimalna różnica: {returns_diff.min():.2f}")

# Definicja zmiennych dla modeli
n_input = 60  # Długość sekwencji wejściowej
batch_size = 32  # Rozmiar batcha
features_with_onehot = ['Close']  # Lista cech
target = 'Close'  # Zmienna docelowa

# Funkcja pomocnicza do konwersji dat
def convert_to_days(date_str, min_date):
    date = pd.to_datetime(date_str)
    return (date - min_date).days

# Przygotowanie danych X i y
X = df[features_with_onehot].values.reshape(-1, 1)
X_without_onehot = df[['Close']].values.reshape(-1, 1)
y = df[target].values.reshape(-1, 1)

# ====== SEKCJA 1: PRZYGOTOWANIE DANYCH DLA MODELI ======

# Przygotowanie danych do treningu
sequence_length = 60  # Długość sekwencji
features = ['Close']  # Używamy tylko ceny zamknięcia

# Przygotowanie danych SPX
spx_data = df[features].values
spx_scaler = MinMaxScaler()
spx_scaled = spx_scaler.fit_transform(spx_data)

# Przygotowanie danych DJIA
dji_data = df2[features].values
dji_scaler = MinMaxScaler()
dji_scaled = dji_scaler.fit_transform(dji_data)

# Przygotowanie danych X i y dla modeli
X = df[features].values
X_without_onehot = df[['Close']].values
y = df[target].values

# Skalowanie danych
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_scaled_without_onehot = scaler_X.fit_transform(X_without_onehot)
y_scaled = scaler_y.fit_transform(y)

# Podział na zbiory treningowe i walidacyjne
train_size = int(len(spx_scaled) * 0.8)
val_size = len(spx_scaled) - train_size

# Generatory danych
train_generator = TimeseriesGenerator(
    spx_scaled[:train_size], 
    dji_scaled[:train_size],
    length=sequence_length,
    batch_size=32
)

val_generator = TimeseriesGenerator(
    spx_scaled[train_size:], 
    dji_scaled[train_size:],
    length=sequence_length,
    batch_size=32
)

# Przygotowanie danych do modeli
def prepare_data(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Normalizacja danych
scaler_spx = MinMaxScaler()
scaler_dji = MinMaxScaler()
spx_scaled = scaler_spx.fit_transform(df[['Close']].values)
dji_scaled = scaler_dji.fit_transform(df2[['Close']].values)

# Parametry modeli
look_back = 30
batch_size = 64
epochs = 20
n_features = 1

# Przygotowanie danych dla SPX i DJIA
X_spx, y_spx = prepare_data(spx_scaled, look_back)
X_dji, y_dji = prepare_data(dji_scaled, look_back)

# Podział na zbiory treningowe i testowe
train_size_spx = int(len(X_spx) * 0.8)
X_train_spx, X_test_spx = X_spx[:train_size_spx], X_spx[train_size_spx:]
y_train_spx, y_test_spx = y_spx[:train_size_spx], y_spx[train_size_spx:]

train_size_dji = int(len(X_dji) * 0.8)
X_train_dji, X_test_dji = X_dji[:train_size_dji], X_dji[train_size_dji:]
y_train_dji, y_test_dji = y_dji[:train_size_dji], y_dji[train_size_dji:]

# Dodatkowe skalowanie danych dla modeli
X = df[features].values
X_without_onehot = df[['Close']].values
y = df[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_scaled_without_onehot = scaler_X.fit_transform(X_without_onehot)
y_scaled = scaler_y.fit_transform(y)

# ====== SEKCJA 2: MODEL LSTM ======
print("\n=== Model LSTM ===")
# Model LSTM dla SPX
lstm_model = keras.Sequential([
    layers.Bidirectional(layers.LSTM(256, activation='tanh', return_sequences=True), 
                        input_shape=(n_input, X_scaled.shape[1])),
    layers.Dropout(0.2),
    layers.Bidirectional(layers.LSTM(128, activation='tanh')),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Konfiguracja harmonogramu uczenia
initial_learning_rate = 0.001
decay_steps = 1000
lr_schedule = CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps
)

# Konfiguracja optymalizatora z harmonogramem
optimizer = AdamW(
    learning_rate=lr_schedule,
    weight_decay=0.01
)

# Kompilacja modelu
lstm_model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae', 'mape', MeanSquaredError(), MeanAbsoluteError()]
)

# Dodaj po definicji modeli, przed treningiem
def save_trained_model(model, model_name):
    """Zapisuje model do pliku"""
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save(f'models/{model_name}.h5')
    print(f"Model {model_name} został zapisany")

def load_or_train_model(model, model_name, train_data, val_data, epochs, initial_epoch=0):
    """Wczytuje istniejący model lub trenuje nowy"""
    model_path = f'models/{model_name}.h5'
    
    if os.path.exists(model_path):
        print(f"Wczytywanie istniejącego modelu {model_name}...")
        loaded_model = load_model(model_path, compile=True)
        print(f"Kontynuowanie treningu modelu {model_name} od epoki {initial_epoch}...")
        history = loaded_model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=1
        )
        return loaded_model, history
    else:
        print(f"Trenowanie nowego modelu {model_name}...")
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        save_trained_model(model, model_name)
        return model, history

# Zmodyfikuj sekcję treningu LSTM
print("\n=== Model LSTM ===")
lstm_model, lstm_history = load_or_train_model(
    lstm_model,
    'lstm_model',
    train_generator,
    val_generator,
    epochs=20,
    initial_epoch=0
)

# ====== SEKCJA 3: MODEL RNN ======
print("\n=== Model RNN ===")
# Model RNN dla SPX
sequence_length = 60  # Długość sekwencji
n_features = 1  # Liczba cech (cena zamknięcia)

# Przygotowanie danych dla RNN
def prepare_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Przygotowanie sekwencji dla SPX
X_train_spx_seq, y_train_spx_seq = prepare_sequences(X_train_spx, sequence_length)
X_test_spx_seq, y_test_spx_seq = prepare_sequences(X_test_spx, sequence_length)

# Reshape danych do wymaganego formatu [samples, time steps, features]
X_train_spx_seq = X_train_spx_seq.reshape((X_train_spx_seq.shape[0], X_train_spx_seq.shape[1], 1))
X_test_spx_seq = X_test_spx_seq.reshape((X_test_spx_seq.shape[0], X_test_spx_seq.shape[1], 1))

# Sprawdzenie wymiarów
print(f"Wymiary X_train_spx_seq: {X_train_spx_seq.shape}")
print(f"Wymiary y_train_spx_seq: {y_train_spx_seq.shape}")

# Model RNN
rnn_model = keras.Sequential([
    layers.SimpleRNN(128, activation='tanh', return_sequences=True, input_shape=(n_input, X_scaled.shape[1])),
    layers.Dropout(0.2),
    layers.SimpleRNN(64, activation='tanh'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Kompilacja modelu RNN
rnn_model.compile(
    optimizer=AdamW(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae', MeanSquaredError()]
)

# Zmodyfikuj sekcję treningu RNN
print("\n=== Model RNN ===")
rnn_model, rnn_history = load_or_train_model(
    rnn_model,
    'rnn_model',
    X_train_spx_seq,
    y_train_spx_seq,
    epochs=20,
    initial_epoch=0
)

# ====== SEKCJA 4: MODEL BLSTM ======
print("\n=== Model BLSTM ===")
# Model BLSTM dla SPX
blstm_model = keras.Sequential([
    layers.Bidirectional(layers.LSTM(128, activation='relu', return_sequences=True), input_shape=(look_back, n_features)),
    layers.Dropout(0.2),
    layers.Bidirectional(layers.LSTM(64, activation='relu', return_sequences=True)),
    layers.Dropout(0.2),
    layers.Bidirectional(layers.LSTM(32, activation='relu')),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

# Kompilacja modelu BLSTM
blstm_model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae', 'mape', MeanSquaredError(), MeanAbsoluteError()]
)

# Zmodyfikuj sekcję treningu BLSTM
print("\n=== Model BLSTM ===")
blstm_model, blstm_history = load_or_train_model(
    blstm_model,
    'blstm_model',
    X_train_spx,
    y_train_spx,
    epochs=20,
    initial_epoch=0
)

# ====== SEKCJA 5: WYKRESY I ANALIZA WYNIKÓW ======
# Predykcje
lstm_pred_spx = lstm_model.predict(X_test_spx)
rnn_pred_spx = rnn_model.predict(X_test_spx)
blstm_pred_spx = blstm_model.predict(X_test_spx)

# Wykresy dla SPX
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(lstm_history.history['loss'], label='LSTM - Strata treningowa')
plt.plot(lstm_history.history['val_loss'], label='LSTM - Strata walidacyjna')
plt.plot(rnn_history.history['loss'], label='RNN - Strata treningowa')
plt.plot(rnn_history.history['val_loss'], label='RNN - Strata walidacyjna')
plt.plot(blstm_history.history['loss'], label='BLSTM - Strata treningowa')
plt.plot(blstm_history.history['val_loss'], label='BLSTM - Strata walidacyjna')
plt.title('Porównanie strat podczas treningu - SPX')
plt.xlabel('Epoka')
plt.ylabel('Strata (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(scaler_spx.inverse_transform(lstm_pred_spx), label='Predykcja LSTM')
plt.plot(scaler_spx.inverse_transform(rnn_pred_spx), label='Predykcja RNN')
plt.plot(scaler_spx.inverse_transform(blstm_pred_spx), label='Predykcja BLSTM')
plt.plot(scaler_spx.inverse_transform(y_test_spx), label='Rzeczywiste wartości', alpha=0.5)
plt.title('Porównanie predykcji modeli - SPX')
plt.xlabel('Czas')
plt.ylabel('Cena')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Wykresy dla DJIA
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(lstm_history.history['loss'], label='LSTM - Strata treningowa')
plt.plot(lstm_history.history['val_loss'], label='LSTM - Strata walidacyjna')
plt.plot(rnn_history.history['loss'], label='RNN - Strata treningowa')
plt.plot(rnn_history.history['val_loss'], label='RNN - Strata walidacyjna')
plt.title('Porównanie strat podczas treningu - DJIA')
plt.xlabel('Epoka')
plt.ylabel('Strata (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(scaler_dji.inverse_transform(lstm_pred_dji), label='Predykcja LSTM')
plt.plot(scaler_dji.inverse_transform(rnn_pred_dji), label='Predykcja RNN')
plt.plot(scaler_dji.inverse_transform(y_test_dji), label='Rzeczywiste wartości', alpha=0.5)
plt.title('Porównanie predykcji modeli - DJIA')
plt.xlabel('Czas')
plt.ylabel('Cena')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Obliczenie metryk dla każdego modelu
print("\nPorównanie metryk modeli dla SPX:")
print("\nModel LSTM:")
print(f"MSE: {mean_squared_error(y_test_spx, lstm_pred_spx):.6f}")
print(f"MAE: {mean_absolute_error(y_test_spx, lstm_pred_spx):.6f}")
print(f"R2: {r2_score(y_test_spx, lstm_pred_spx):.6f}")

print("\nModel RNN:")
print(f"MSE: {mean_squared_error(y_test_spx, rnn_pred_spx):.6f}")
print(f"MAE: {mean_absolute_error(y_test_spx, rnn_pred_spx):.6f}")
print(f"R2: {r2_score(y_test_spx, rnn_pred_spx):.6f}")

print("\nModel BLSTM:")
print(f"MSE: {mean_squared_error(y_test_spx, blstm_pred_spx):.6f}")
print(f"MAE: {mean_absolute_error(y_test_spx, blstm_pred_spx):.6f}")
print(f"R2: {r2_score(y_test_spx, blstm_pred_spx):.6f}")

print("\nPorównanie metryk modeli dla DJIA:")
print("\nModel LSTM:")
print(f"MSE: {mean_squared_error(y_test_dji, lstm_pred_dji):.6f}")
print(f"MAE: {mean_absolute_error(y_test_dji, lstm_pred_dji):.6f}")
print(f"R2: {r2_score(y_test_dji, lstm_pred_dji):.6f}")

print("\nModel RNN:")
print(f"MSE: {mean_squared_error(y_test_dji, rnn_pred_dji):.6f}")
print(f"MAE: {mean_absolute_error(y_test_dji, rnn_pred_dji):.6f}")
print(f"R2: {r2_score(y_test_dji, rnn_pred_dji):.6f}")

scaler_X = StandardScaler()
scaler_y = StandardScaler()
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

# Definicja modelu1
model1 = keras.Sequential([
    layers.LSTM(128, activation='tanh', return_sequences=True, input_shape=(n_input, X_scaled.shape[1])),
    layers.Dropout(0.2),
    layers.LSTM(64, activation='tanh'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Definicja optimizer2
optimizer2 = AdamW(
    learning_rate=0.001,
    weight_decay=0.01
)

# Kompilacja modelu1
model1.compile(
    optimizer=optimizer2,
    loss='mean_squared_error',
    metrics=[MeanSquaredError(), MeanAbsoluteError(), R2Score()]
)

# Definicja modelu LSTM bez one-hot encoding
lstm_model_without_onehot = keras.Sequential([
    layers.Bidirectional(layers.LSTM(256, activation='tanh', return_sequences=True), 
                        input_shape=(n_input, X_scaled_without_onehot.shape[1])),
    layers.Dropout(0.2),
    layers.Bidirectional(layers.LSTM(128, activation='tanh')),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Definicja modelu RNN bez one-hot encoding
rnn_model_without_onehot = keras.Sequential([
    layers.SimpleRNN(128, activation='tanh', return_sequences=True, input_shape=(n_input, X_scaled_without_onehot.shape[1])),
    layers.Dropout(0.2),
    layers.SimpleRNN(64, activation='tanh'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Kompilacja modeli bez one-hot encoding
lstm_model_without_onehot.compile(
    optimizer=optimizer2,
    loss='mean_squared_error',
    metrics=[MeanSquaredError(), MeanAbsoluteError(), R2Score()]
)

rnn_model_without_onehot.compile(
    optimizer=optimizer2,
    loss='mean_squared_error',
    metrics=[MeanSquaredError(), MeanAbsoluteError(), R2Score()]
)

# Predykcje
y_pred1 = model1.predict(test_generator)
y_pred2 = lstm_model.predict(test_generator)
y_pred2_without_onehot = lstm_model_without_onehot.predict(test_generator_without_onehot)
y_pred4 = rnn_model.predict(test_generator)
y_pred4_without_onehot = rnn_model_without_onehot.predict(test_generator_without_onehot)

y_pred1_inv = scaler_y.inverse_transform(y_pred1)
y_pred2_inv = scaler_y.inverse_transform(y_pred2)
y_pred2_inv_without_onehot = scaler_y.inverse_transform(y_pred2_without_onehot)
y_pred4_inv = scaler_y.inverse_transform(y_pred4)
y_pred4_inv_without_onehot = scaler_y.inverse_transform(y_pred4_without_onehot)
real_prices = scaler_y.inverse_transform(y_test_seq[n_input:])

# Daty
dates_test = df.iloc[split_index + n_input:split_index + n_input + len(y_test_seq[n_input:])]['Date']
dates_test = min_date + pd.to_timedelta(dates_test, unit='D')

# Porównanie wyników
plt.figure(figsize=(15, 8))
plt.plot(dates_test, real_prices, label='Real Price', color='blue', linewidth=2)
plt.plot(dates_test, y_pred1_inv, label='LSTM Model', color='orange', linewidth=2)
plt.plot(dates_test, y_pred2_inv, label='Bidirectional LSTM with One-Hot', color='red', linewidth=2)
plt.plot(dates_test, y_pred2_inv_without_onehot, label='Bidirectional LSTM without One-Hot', color='green', linewidth=2)
plt.plot(dates_test, y_pred4_inv, label='RNN with One-Hot', color='purple', linewidth=2)
plt.plot(dates_test, y_pred4_inv_without_onehot, label='RNN without One-Hot', color='brown', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Comparison of Different Models')
plt.legend()
plt.grid(True)
plt.show()

# Wykresy pokazujące porównanie z one hot encoding, bez one hot encoding oraz z rzeczywistymi wartościami
# Analiza dla okresów czarnych łabędzi
# Wykres dla WTC
filtered_df = df.loc[
    (df['Date'] >= pd.Timestamp('2001-09-01').value // 10**9) &
    (df['Date'] < pd.Timestamp('2001-12-11').value // 10**9)
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
    (df['Date'] >= pd.Timestamp('1987-10-08').value // 10**9) &
    (df['Date'] < pd.Timestamp('1988-01-19').value // 10**9)
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
    (df['Date'] >= pd.Timestamp('2020-03-01').value // 10**9) &
    (df['Date'] < pd.Timestamp('2020-06-11').value // 10**9)
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
    (df['Date'] >= pd.Timestamp('2020-01-20').value // 10**9) &
    (df['Date'] < pd.Timestamp('2020-04-30').value // 10**9)
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
    (df['Date'] >= pd.Timestamp('2022-02-13').value // 10**9) &
    (df['Date'] < pd.Timestamp('2022-02-24').value // 10**9)
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
    (df['Date'] >= pd.Timestamp('2008-09-05').value // 10**9) &
    (df['Date'] < pd.Timestamp('2008-12-06').value // 10**9)
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
print("\nPorównanie metryk dla wszystkich modeli:")
print("\nLSTM with One-Hot Encoding:")
print(f"R2 Score: {model1.evaluate(test_generator, verbose=0)[2]}")
print(f"Mean Squared Error: {model1.evaluate(test_generator, verbose=0)[0]}")
print(f"Mean Absolute Error: {model1.evaluate(test_generator, verbose=0)[1]}")

print("\nBidirectional LSTM with One-Hot Encoding:")
print(f"R2 Score: {lstm_model.evaluate(test_generator, verbose=0)[2]}")
print(f"Mean Squared Error: {lstm_model.evaluate(test_generator, verbose=0)[0]}")
print(f"Mean Absolute Error: {lstm_model.evaluate(test_generator, verbose=0)[1]}")

print("\nBidirectional LSTM without One-Hot Encoding:")
print(f"R2 Score: {lstm_model_without_onehot.evaluate(test_generator_without_onehot, verbose=0)[2]}")
print(f"Mean Squared Error: {lstm_model_without_onehot.evaluate(test_generator_without_onehot, verbose=0)[0]}")
print(f"Mean Absolute Error: {lstm_model_without_onehot.evaluate(test_generator_without_onehot, verbose=0)[1]}")

print("\nRNN with One-Hot Encoding:")
print(f"R2 Score: {rnn_model.evaluate(test_generator, verbose=0)[2]}")
print(f"Mean Squared Error: {rnn_model.evaluate(test_generator, verbose=0)[0]}")
print(f"Mean Absolute Error: {rnn_model.evaluate(test_generator, verbose=0)[1]}")

print("\nRNN without One-Hot Encoding:")
print(f"R2 Score: {rnn_model_without_onehot.evaluate(test_generator_without_onehot, verbose=0)[2]}")
print(f"Mean Squared Error: {rnn_model_without_onehot.evaluate(test_generator_without_onehot, verbose=0)[0]}")
print(f"Mean Absolute Error: {rnn_model_without_onehot.evaluate(test_generator_without_onehot, verbose=0)[1]}")

X_dji = df2.loc[:, features_with_onehot].values
y_dji = df2.loc[:, target].values

scaler_X_dji = StandardScaler()
scaler_y_dji = StandardScaler()
X_dji_scaled = scaler_X_dji.fit_transform(X_dji)
y_dji_scaled = scaler_y_dji.fit_transform(y_dji)

split_index_dji = int(len(X_dji_scaled) * 0.75)
X_train_dji, X_test_dji = X_dji_scaled[:split_index_dji], X_dji_scaled[split_index_dji:]
y_train_dji, y_test_dji = y_dji_scaled[:split_index_dji], y_dji_scaled[split_index_dji:]

train_generator_dji = TimeseriesGenerator(X_train_dji, y_train_dji, length=n_input, batch_size=batch_size)
test_generator_dji = TimeseriesGenerator(X_test_dji, y_test_dji, length=n_input, batch_size=batch_size)

model3 = keras.Sequential([
    layers.LSTM(128, activation='tanh', return_sequences=True, input_shape=(n_input, X_dji_scaled.shape[1])),
    layers.Dropout(0.1),
    layers.LSTM(64, activation='tanh'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

lr_schedule4 = CosineDecay(initial_learning_rate=0.01, decay_steps=200, alpha=0.0001)
optimizer4 = AdamW(learning_rate=lr_schedule4, weight_decay=0.001)
model3.compile(optimizer=optimizer4, loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), R2Score()])

history4 = model3.fit(train_generator_dji, epochs=10, validation_data=test_generator_dji, 
                     callbacks=[reduce_lr], verbose=1)

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



