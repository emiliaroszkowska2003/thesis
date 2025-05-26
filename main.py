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
from tensorflow.keras.losses import MeanSquaredError as MSELoss
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

# Definicje niestandardowych metryk
@keras.utils.register_keras_serializable()
class CustomMSEMetric(MeanSquaredError):
    def __init__(self, name='custom_mse', **kwargs):
        super().__init__(name=name, **kwargs)

@keras.utils.register_keras_serializable()
class CustomMAEMetric(MeanAbsoluteError):
    def __init__(self, name='custom_mae', **kwargs):
        super().__init__(name=name, **kwargs)

@keras.utils.register_keras_serializable()
class CustomR2Metric(R2Score):
    def __init__(self, name='custom_r2', **kwargs):
        super().__init__(name=name, **kwargs)

# Definicje parametrów i zmiennych
n_input = 60  # Długość sekwencji wejściowej
batch_size = 32  # Rozmiar batcha
n_features = 1  # Liczba cech (w tym przypadku tylko cena zamknięcia)
look_back = 60  # Okno czasowe do analizy

# Definicja cech i targetu
features = ['Close']  # Używamy tylko ceny zamknięcia
target = 'Close'

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

# Definicja okresów czarnych łabędzi z dokładnymi datami
black_swan_periods = [
    ('WTC', '2001-09-11', '2001-09-11'),  # Dokładna data ataku
    ('Black Monday', '1987-10-19', '1987-10-19'),  # Dokładna data krachu
    ('Covid-19', '2020-02-24', '2020-03-23'),  # Początek pandemii
    ('Brexit', '2016-06-23', '2016-06-24'),  # Referendum
    ('Ukraine War', '2022-02-24', '2022-02-24'),  # Początek inwazji
    ('Lehman Brothers', '2008-09-15', '2008-09-15')  # Data bankructwa
]

# Konwersja dat na timestamp
df['Date'] = pd.to_datetime(df['Date'], unit='D')
df2['Date'] = pd.to_datetime(df2['Date'], unit='D')

# Inicjalizacja kolumny black_swan
df['black_swan'] = 0
df2['black_swan'] = 0

# Oznaczenie okresów czarnych łabędzi
for _, start_date, end_date in black_swan_periods:
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    df.loc[mask, 'black_swan'] = 1
    
    mask2 = (df2['Date'] >= pd.to_datetime(start_date)) & (df2['Date'] <= pd.to_datetime(end_date))
    df2.loc[mask2, 'black_swan'] = 1

# Dodanie kolumny black_swan do cech
features = ['Close', 'black_swan']
features_with_onehot = ['Close', 'black_swan']

# Aktualizacja przygotowania danych
X = df[features].values
X_without_onehot = df[['Close']].values.reshape(-1, 1)
y = df[target].values.reshape(-1, 1)

# Skalowanie danych
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_scaled_without_onehot = scaler_X.fit_transform(X_without_onehot)
y_scaled = scaler_y.fit_transform(y)

# Konfiguracja optymalizatora
initial_learning_rate = 0.001
decay_steps = 1000
lr_schedule = CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps
)

optimizer = AdamW(
    learning_rate=lr_schedule,
    weight_decay=0.01
)

optimizer2 = AdamW(
    learning_rate=0.001,
    weight_decay=0.01
)

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
    # Konwersja danych do 1D jeśli są 2D
    if len(data.shape) > 1:
        data = data.flatten()
        
    scaled_data = data[::scale]
    # Upewniamy się, że mamy odpowiednie wymiary
    scaled_data = scaled_data.reshape(-1)
    returns = np.diff(scaled_data) / scaled_data[:-1]
    returns = returns.reshape(-1)
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
    period_spx = df['Close'].values[start:end].reshape(-1, 1)
    period_dji = df2['Close'].values[start:end].reshape(-1, 1)
    
    print(f"\n{period_name}:")
    print("Skala\tSPX\t\tDJIA\t\tRóżnica")
    print("-" * 50)
    
    for scale in scales:
        spx_ent = mse_entropy(period_spx, m=5, tau=2, scale=scale)
        dji_ent = mse_entropy(period_dji, m=5, tau=2, scale=scale)
        diff = abs(spx_ent - dji_ent)
        print(f"{scale}\t{spx_ent:.4f}\t\t{dji_ent:.4f}\t\t{diff:.4f}")

# Obliczanie złożoności dla SPX i DJIA
spx_complexity = np.array(market_complexity(df['Close'].values.reshape(-1, 1)))
dji_complexity = np.array(market_complexity(df2['Close'].values.reshape(-1, 1)))

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

# Normalizacja stóp zwrotu
spx_returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
dji_returns = np.log(df2['Close'] / df2['Close'].shift(1)).fillna(0)

# Obliczenie zmienności (volatility) na dłuższym oknie
window = 20
spx_vol = spx_returns.rolling(window=window).std().fillna(method='bfill')
dji_vol = dji_returns.rolling(window=window).std().fillna(method='bfill')

# Normalizacja stóp zwrotu przez zmienność
epsilon = 1e-10  # Mała wartość do uniknięcia dzielenia przez zero
spx_normalized = (spx_returns / (spx_vol + epsilon)).fillna(0)
dji_normalized = (dji_returns / (dji_vol + epsilon)).fillna(0)

def analyze_lag_correlations(x, y, max_lag=30):
    """
    Analizuje korelację między dwoma szeregami czasowymi dla różnych opóźnień.
    
    Parametry:
    x, y: array-like, szeregi czasowe do porównania
    max_lag: int, maksymalne opóźnienie do analizy
    
    Zwraca:
    tuple: (korelacje, opóźnienia)
    """
    # Obliczenie stóp zwrotu
    x_returns = np.diff(x.flatten()) / x[:-1].flatten()
    y_returns = np.diff(y.flatten()) / y[:-1].flatten()
    
    # Normalizacja stóp zwrotu
    x_returns = (x_returns - np.mean(x_returns)) / np.std(x_returns)
    y_returns = (y_returns - np.mean(y_returns)) / np.std(y_returns)
    
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            x_shifted = x_returns[-lag:]
            y_shifted = y_returns[:len(x_shifted)]
        elif lag > 0:
            x_shifted = x_returns[:-lag]
            y_shifted = y_returns[lag:]
        else:
            x_shifted = x_returns
            y_shifted = y_returns
            
        min_len = min(len(x_shifted), len(y_shifted))
        x_shifted = x_shifted[:min_len]
        y_shifted = y_shifted[:min_len]
        
        corr = np.corrcoef(x_shifted, y_shifted)[0, 1]
        correlations.append(corr)
    
    return np.array(correlations), np.array(lags)

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


# Analiza korelacji dla różnych opóźnień
print("\nAnaliza korelacji dla różnych opóźnień:")
correlations, lags = analyze_lag_correlations(
    df['Close'].values.reshape(-1, 1),
    df2['Close'].values.reshape(-1, 1),
    max_lag=30
)

# Wykres korelacji dla opóźnień
plt.figure(figsize=(12, 6))
plt.plot(lags, correlations, 'b-', linewidth=2, label='Korelacja')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)

# Dodanie poziomych linii dla poziomów istotności
plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.2, label='Umiarkowana korelacja')
plt.axhline(y=-0.5, color='g', linestyle='--', alpha=0.2)
plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.2, label='Silna korelacja')
plt.axhline(y=-0.7, color='r', linestyle='--', alpha=0.2)

plt.xlabel('Opóźnienie (dni)')
plt.ylabel('Współczynnik korelacji')
plt.title('Korelacja między SPX a DJIA dla różnych opóźnień')
plt.grid(True, alpha=0.3)

# Znajdź maksymalną korelację dodatnią i ujemną
max_pos_corr_idx = np.argmax(correlations)
max_neg_corr_idx = np.argmin(correlations)
max_pos_corr = correlations[max_pos_corr_idx]
max_neg_corr = correlations[max_neg_corr_idx]
max_pos_lag = lags[max_pos_corr_idx]
max_neg_lag = lags[max_neg_corr_idx]

# Dodaj adnotacje dla maksymalnych korelacji
plt.annotate(f'Max korelacja dodatnia: {max_pos_corr:.3f}\nOpóźnienie: {max_pos_lag} dni',
             xy=(max_pos_lag, max_pos_corr),
             xytext=(10, 10),
             textcoords='offset points',
             ha='left',
             va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.annotate(f'Max korelacja ujemna: {max_neg_corr:.3f}\nOpóźnienie: {max_neg_lag} dni',
             xy=(max_neg_lag, max_neg_corr),
             xytext=(10, -10),
             textcoords='offset points',
             ha='left',
             va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.legend()
plt.tight_layout()
plt.show()

# Wyświetl szczegółową analizę korelacji
print("\nSzczegółowa analiza korelacji:")
print(f"Maksymalna korelacja dodatnia: {max_pos_corr:.4f} (opóźnienie: {max_pos_lag} dni)")
print(f"Maksymalna korelacja ujemna: {max_neg_corr:.4f} (opóźnienie: {max_neg_lag} dni)")
print(f"Średnia wartość bezwzględna korelacji: {np.mean(np.abs(correlations)):.4f}")
print(f"Odchylenie standardowe korelacji: {np.std(correlations):.4f}")

# Analiza dla wybranych opóźnień
selected_lags = [-10, -5, 0, 5, 10]
print("\nKorelacja dla wybranych opóźnień:")
for lag in selected_lags:
    idx = np.where(lags == lag)[0][0]
    print(f"Opóźnienie {lag} dni: {correlations[idx]:.4f}")

# Obliczenie różnic przyrostów
spx_returns = df['Close'].pct_change().dropna()
dji_returns = df2['Close'].pct_change().dropna()
returns_diff = spx_returns - dji_returns

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

# Funkcja do zapisywania modelu
def save_model_and_history(model, history, model_name):
    try:
        # Tworzenie katalogu jeśli nie istnieje
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Zapisywanie modelu
        model_path = f'models/{model_name}_model.keras'
        model.save(model_path)
        print(f"Model {model_name} zapisany w {model_path}")
        
        # Zapisywanie historii
        if history is not None:
            history_path = f'models/{model_name}_history.npy'
            if isinstance(history, dict):
                np.save(history_path, history)
                print(f"Historia treningu zapisana w {history_path}")
            else:
                print(f"Ostrzeżenie: Historia treningu dla {model_name} nie jest słownikiem")
        else:
            print(f"Ostrzeżenie: Brak historii treningu dla {model_name}")
            
    except Exception as e:
        print(f"Błąd podczas zapisywania modelu {model_name}: {str(e)}")
        raise  # Rzucamy błąd dalej, aby program się zatrzymał

# Funkcja do wczytywania modelu i historii
def load_model_and_history(model_name):
    try:
        # Sprawdzanie czy istnieje katalog models
        if not os.path.exists('models'):
            print(f"Katalog 'models' nie istnieje. Tworzę nowy model {model_name}.")
            return None, None
            
        # Próba wczytania modelu
        model_path = f'models/{model_name}_model.keras'
        if not os.path.exists(model_path):
            print(f"Nie znaleziono modelu {model_name}, tworzę nowy.")
            return None, None
        
        print(f"Wczytywanie modelu {model_name}...")
        model = load_model(model_path, compile=False)
        
        # Wczytywanie historii
        history_path = f'models/{model_name}_history.npy'
        history = None
        if os.path.exists(history_path):
            try:
                history = np.load(history_path, allow_pickle=True).item()
                if isinstance(history, dict) and 'loss' in history:
                    print(f"Znaleziono historię treningu dla {model_name} z {len(history['loss'])} epokami.")
                else:
                    print(f"Historia treningu dla {model_name} jest nieprawidłowa.")
                    history = None
            except Exception as e:
                print(f"Błąd podczas wczytywania historii dla {model_name}: {str(e)}")
                history = None
        
        # Kompilacja modelu
        print(f"Kompilacja modelu {model_name}...")
        if 'lstm' in model_name:
            optimizer = AdamW(learning_rate=initial_learning_rate, weight_decay=0.01)
        elif 'rnn' in model_name:
            optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
        elif 'blstm' in model_name:
            optimizer = AdamW(learning_rate=initial_learning_rate, weight_decay=0.01)
        else:
            optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
            
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=[CustomMSEMetric(), CustomMAEMetric()]
        )
        
        return model, history
        
    except Exception as e:
        print(f"Krytyczny błąd podczas wczytywania modelu {model_name}: {str(e)}")
        return None, None

# Sprawdzanie poprawności danych
def validate_data(X, y, name=""):
    if np.isnan(X).any() or np.isinf(X).any():
        print(f"Ostrzeżenie: Wykryto wartości NaN lub Inf w danych wejściowych {name}")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isnan(y).any() or np.isinf(y).any():
        print(f"Ostrzeżenie: Wykryto wartości NaN lub Inf w danych wyjściowych {name}")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y

# Przygotowanie danych do treningu
train_size = int(len(X_scaled) * 0.8)
X_train = X_scaled[:train_size]
X_test = X_scaled[train_size:]
y_train = y_scaled[:train_size]
y_test = y_scaled[train_size:]

# Walidacja danych
X_train, y_train = validate_data(X_train, y_train, "treningowych")
X_test, y_test = validate_data(X_test, y_test, "testowych")

# Przygotowanie generatorów danych
try:
    train_generator = TimeseriesGenerator(X_train, y_train, length=n_input, batch_size=batch_size)
    test_generator = TimeseriesGenerator(X_test, y_test, length=n_input, batch_size=batch_size)
except Exception as e:
    print(f"Błąd podczas tworzenia generatorów danych: {str(e)}")
    raise

# Przygotowanie generatorów danych dla modeli bez one-hot encoding
try:
    X_train_without_onehot = X_scaled_without_onehot[:train_size]
    X_test_without_onehot = X_scaled_without_onehot[train_size:]
    X_train_without_onehot, y_train = validate_data(X_train_without_onehot, y_train, "treningowych bez one-hot")
    X_test_without_onehot, y_test = validate_data(X_test_without_onehot, y_test, "testowych bez one-hot")
    
    train_generator_without_onehot = TimeseriesGenerator(X_train_without_onehot, y_train, length=n_input, batch_size=batch_size)
    test_generator_without_onehot = TimeseriesGenerator(X_test_without_onehot, y_test, length=n_input, batch_size=batch_size)
except Exception as e:
    print(f"Błąd podczas tworzenia generatorów danych bez one-hot encoding: {str(e)}")
    raise

# Konfiguracja callbacks z dodatkowymi zabezpieczeniami
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# Funkcja do bezpiecznego trenowania modelu
def train_model_safely(model, train_gen, test_gen, model_name, initial_epoch=0):
    try:
        # Sprawdzenie czy model jest skompilowany
        if not hasattr(model, 'optimizer') or model.optimizer is None:
            print(f"Model {model_name} nie jest skompilowany. Kompiluję...")
            if 'lstm' in model_name:
                new_optimizer = AdamW(
                    learning_rate=initial_learning_rate,
                    weight_decay=0.01
                )
                model.compile(
                    optimizer=new_optimizer,
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                )
            elif 'rnn' in model_name:
                new_optimizer = AdamW(
                    learning_rate=0.001,
                    weight_decay=0.01
                )
                model.compile(
                    optimizer=new_optimizer,
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                )
            elif 'blstm' in model_name:
                new_optimizer = AdamW(
                    learning_rate=initial_learning_rate,
                    weight_decay=0.01
                )
                model.compile(
                    optimizer=new_optimizer,
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                )

        print(f"Trenowanie modelu {model_name} od epoki {initial_epoch}...")
        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=20,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=initial_epoch
        )
        return history.history
    except Exception as e:
        print(f"Błąd podczas treningu modelu {model_name}: {str(e)}")
        print("Próba ponownego treningu od początku...")
        try:
            # Ponowna kompilacja modelu przed próbą ponownego treningu
            if 'lstm' in model_name:
                new_optimizer = AdamW(
                    learning_rate=initial_learning_rate,
                    weight_decay=0.01
                )
                model.compile(
                    optimizer=new_optimizer,
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                )
            elif 'rnn' in model_name:
                new_optimizer = AdamW(
                    learning_rate=0.001,
                    weight_decay=0.01
                )
                model.compile(
                    optimizer=new_optimizer,
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                )
            elif 'blstm' in model_name:
                new_optimizer = AdamW(
                    learning_rate=initial_learning_rate,
                    weight_decay=0.01
                )
                model.compile(
                    optimizer=new_optimizer,
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                )

            history = model.fit(
                train_gen,
                validation_data=test_gen,
                epochs=20,
                callbacks=callbacks,
                verbose=1
            )
            return history.history
        except Exception as e:
            print(f"Krytyczny błąd podczas treningu modelu {model_name}: {str(e)}")
            raise

# Trenowanie modeli z kontynuacją
print("\nTrenowanie modelu LSTM...")
lstm_model, lstm_history = load_model_and_history('lstm')
if lstm_model is None:
    print("Tworzenie nowego modelu LSTM...")
    lstm_model = keras.Sequential([
        layers.Bidirectional(layers.LSTM(256, activation='tanh', return_sequences=True), 
                            input_shape=(n_input, X_scaled.shape[1])),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(128, activation='tanh')),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    new_optimizer = AdamW(
        learning_rate=initial_learning_rate,
        weight_decay=0.01
    )
    lstm_model.compile(
        optimizer=new_optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    )
    lstm_history = train_model_safely(lstm_model, train_generator, test_generator, 'LSTM')
else:
    print("Kontynuacja treningu LSTM od ostatniego stanu...")
    initial_epoch = len(lstm_history['loss']) if lstm_history and 'loss' in lstm_history else 0
    if initial_epoch >= 20:
        print(f"Model LSTM już został wytrenowany przez {initial_epoch} epok. Pomijam trening.")
    else:
        lstm_history = train_model_safely(lstm_model, train_generator, test_generator, 'LSTM', 
                                        initial_epoch=initial_epoch)
save_model_and_history(lstm_model, lstm_history, 'lstm')

print("\nTrenowanie modelu RNN...")
rnn_model, rnn_history = load_model_and_history('rnn')
if rnn_model is None:
    print("Tworzenie nowego modelu RNN...")
    rnn_model = keras.Sequential([
        layers.SimpleRNN(128, activation='tanh', return_sequences=True, input_shape=(n_input, X_scaled.shape[1])),
        layers.Dropout(0.2),
        layers.SimpleRNN(64, activation='tanh'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    new_optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.01
    )
    rnn_model.compile(
        optimizer=new_optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    )
    rnn_history = train_model_safely(rnn_model, train_generator, test_generator, 'RNN')
else:
    print("Kontynuacja treningu RNN od ostatniego stanu...")
    rnn_history = train_model_safely(rnn_model, train_generator, test_generator, 'RNN', 
                                    initial_epoch=len(rnn_history['loss']))
save_model_and_history(rnn_model, rnn_history, 'rnn')

print("\nTrenowanie modelu BLSTM...")
blstm_model, blstm_history = load_model_and_history('blstm')
if blstm_model is None:
    print("Tworzenie nowego modelu BLSTM...")
    blstm_model = keras.Sequential([
        layers.Bidirectional(layers.LSTM(128, activation='relu', return_sequences=True), 
                            input_shape=(n_input, X_scaled.shape[1])),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(64, activation='relu', return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32, activation='relu')),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    new_optimizer = AdamW(
        learning_rate=initial_learning_rate,
        weight_decay=0.01
    )
    blstm_model.compile(
        optimizer=new_optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    )
    blstm_history = train_model_safely(blstm_model, train_generator, test_generator, 'BLSTM')
else:
    print("Kontynuacja treningu BLSTM od ostatniego stanu...")
    initial_epoch = len(blstm_history['loss']) if blstm_history and 'loss' in blstm_history else 0
    blstm_history = train_model_safely(blstm_model, train_generator, test_generator, 'BLSTM', 
                                     initial_epoch=initial_epoch)
save_model_and_history(blstm_model, blstm_history, 'blstm')

# Trenowanie modeli bez one-hot encoding
print("\nTrenowanie modelu RNN bez one-hot encoding...")
rnn_model_without_onehot, rnn_history_without_onehot = load_model_and_history('rnn_without_onehot')
if rnn_model_without_onehot is None:
    print("Tworzenie nowego modelu RNN bez one-hot encoding...")
    rnn_model_without_onehot = keras.Sequential([
        layers.SimpleRNN(128, activation='tanh', return_sequences=True, input_shape=(n_input, X_scaled_without_onehot.shape[1])),
        layers.Dropout(0.2),
        layers.SimpleRNN(64, activation='tanh'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    new_optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.01
    )
    rnn_model_without_onehot.compile(
        optimizer=new_optimizer,
        loss='mean_squared_error',
        metrics=[CustomMSEMetric(), CustomMAEMetric()]
    )
    rnn_history_without_onehot = train_model_safely(rnn_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'RNN_without_onehot')
else:
    print("Sprawdzanie historii treningu RNN bez one-hot encoding...")
    if rnn_history_without_onehot is not None and isinstance(rnn_history_without_onehot, dict) and 'loss' in rnn_history_without_onehot:
        epochs_trained = len(rnn_history_without_onehot['loss'])
        print(f"Model został już wytrenowany przez {epochs_trained} epok")
        
        if epochs_trained >= 20:
            print("Model jest już w pełni wytrenowany (20 epok). Pomijam trening.")
            # Zachowujemy istniejącą historię
            save_model_and_history(rnn_model_without_onehot, rnn_history_without_onehot, 'rnn_without_onehot')
        else:
            print(f"Kontynuacja treningu od epoki {epochs_trained}")
            rnn_history_without_onehot = train_model_safely(rnn_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'RNN_without_onehot', 
                                                    initial_epoch=epochs_trained)
    else:
        print("Nie znaleziono historii treningu, rozpoczynam od początku")
        rnn_history_without_onehot = train_model_safely(rnn_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'RNN_without_onehot')
save_model_and_history(rnn_model_without_onehot, rnn_history_without_onehot, 'rnn_without_onehot')

print("\nTrenowanie modelu LSTM bez one-hot encoding...")
lstm_model_without_onehot, lstm_history_without_onehot = load_model_and_history('lstm_without_onehot')
if lstm_model_without_onehot is None:
    print("Tworzenie nowego modelu LSTM bez one-hot encoding...")
    lstm_model_without_onehot = keras.Sequential([
        layers.Bidirectional(layers.LSTM(256, activation='tanh', return_sequences=True), 
                            input_shape=(n_input, X_scaled_without_onehot.shape[1])),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(128, activation='tanh')),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    new_optimizer = AdamW(
        learning_rate=initial_learning_rate,
        weight_decay=0.01
    )
    lstm_model_without_onehot.compile(
        optimizer=new_optimizer,
        loss='mean_squared_error',
        metrics=[CustomMSEMetric(), CustomMAEMetric()]
    )
    lstm_history_without_onehot = train_model_safely(lstm_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'LSTM_without_onehot')
else:
    print("Sprawdzanie historii treningu LSTM bez one-hot encoding...")
    if lstm_history_without_onehot is not None and isinstance(lstm_history_without_onehot, dict) and 'loss' in lstm_history_without_onehot:
        epochs_trained = len(lstm_history_without_onehot['loss'])
        print(f"Model został już wytrenowany przez {epochs_trained} epok")
        
        if epochs_trained >= 20:
            print("Model jest już w pełni wytrenowany (20 epok). Pomijam trening.")
            # Zachowujemy istniejącą historię
            save_model_and_history(lstm_model_without_onehot, lstm_history_without_onehot, 'lstm_without_onehot')
        else:
            print(f"Kontynuacja treningu od epoki {epochs_trained}")
            lstm_history_without_onehot = train_model_safely(lstm_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'LSTM_without_onehot', 
                                                    initial_epoch=epochs_trained)
    else:
        print("Nie znaleziono historii treningu, rozpoczynam od początku")
        lstm_history_without_onehot = train_model_safely(lstm_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'LSTM_without_onehot')
save_model_and_history(lstm_model_without_onehot, lstm_history_without_onehot, 'lstm_without_onehot')

print("\nTrenowanie modelu RNN bez one-hot encoding...")
rnn_model_without_onehot, rnn_history_without_onehot = load_model_and_history('rnn_without_onehot')
if rnn_model_without_onehot is None:
    print("Tworzenie nowego modelu RNN bez one-hot encoding...")
    rnn_model_without_onehot = keras.Sequential([
        layers.SimpleRNN(128, activation='tanh', return_sequences=True, input_shape=(n_input, X_scaled_without_onehot.shape[1])),
        layers.Dropout(0.2),
        layers.SimpleRNN(64, activation='tanh'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    new_optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.01
    )
    rnn_model_without_onehot.compile(
        optimizer=new_optimizer,
        loss='mean_squared_error',
        metrics=[CustomMSEMetric(), CustomMAEMetric()]
    )
    rnn_history_without_onehot = train_model_safely(rnn_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'RNN_without_onehot')
else:
    print("Kontynuacja treningu RNN bez one-hot encoding od ostatniego stanu...")
    initial_epoch = 0
    if rnn_history_without_onehot and isinstance(rnn_history_without_onehot, dict) and 'loss' in rnn_history_without_onehot:
        initial_epoch = len(rnn_history_without_onehot['loss'])
        print(f"Znaleziono historię treningu z {initial_epoch} epokami")
    else:
        print("Nie znaleziono historii treningu, rozpoczynam od początku")
    
    if initial_epoch >= 20:
        print(f"Model RNN bez one-hot encoding już został wytrenowany przez {initial_epoch} epok. Pomijam trening.")
    else:
        rnn_history_without_onehot = train_model_safely(rnn_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'RNN_without_onehot', 
                                                initial_epoch=initial_epoch)
save_model_and_history(rnn_model_without_onehot, rnn_history_without_onehot, 'rnn_without_onehot')

print("\nTrenowanie modelu BLSTM bez one-hot encoding...")
blstm_model_without_onehot, blstm_history_without_onehot = load_model_and_history('blstm_without_onehot')
if blstm_model_without_onehot is None:
    print("Tworzenie nowego modelu BLSTM bez one-hot encoding...")
    blstm_model_without_onehot = keras.Sequential([
        layers.Bidirectional(layers.LSTM(128, activation='relu', return_sequences=True), 
                            input_shape=(n_input, X_scaled_without_onehot.shape[1])),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(64, activation='relu', return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32, activation='relu')),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    new_optimizer = AdamW(
        learning_rate=initial_learning_rate,
        weight_decay=0.01
    )
    blstm_model_without_onehot.compile(
        optimizer=new_optimizer,
        loss='mean_squared_error',
        metrics=[CustomMSEMetric(), CustomMAEMetric()]
    )
    blstm_history_without_onehot = train_model_safely(blstm_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'BLSTM_without_onehot')
else:
    print("Sprawdzanie historii treningu BLSTM bez one-hot encoding...")
    if blstm_history_without_onehot is not None and isinstance(blstm_history_without_onehot, dict) and 'loss' in blstm_history_without_onehot:
        epochs_trained = len(blstm_history_without_onehot['loss'])
        print(f"Model został już wytrenowany przez {epochs_trained} epok")
        
        if epochs_trained >= 20:
            print("Model jest już w pełni wytrenowany (20 epok). Pomijam trening.")
            # Zachowujemy istniejącą historię
            save_model_and_history(blstm_model_without_onehot, blstm_history_without_onehot, 'blstm_without_onehot')
        else:
            print(f"Kontynuacja treningu od epoki {epochs_trained}")
            blstm_history_without_onehot = train_model_safely(blstm_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'BLSTM_without_onehot', 
                                                    initial_epoch=epochs_trained)
    else:
        print("Nie znaleziono historii treningu, rozpoczynam od początku")
        blstm_history_without_onehot = train_model_safely(blstm_model_without_onehot, train_generator_without_onehot, test_generator_without_onehot, 'BLSTM_without_onehot')
save_model_and_history(blstm_model_without_onehot, blstm_history_without_onehot, 'blstm_without_onehot')

# Generowanie predykcji dla SPX
print("\nGenerowanie predykcji dla SPX...")
lstm_pred = lstm_model.predict(test_generator)
rnn_pred = rnn_model.predict(test_generator)
blstm_pred = blstm_model.predict(test_generator)

# Przekształcenie predykcji SPX z powrotem do oryginalnej skali
lstm_pred = scaler_y.inverse_transform(lstm_pred)
rnn_pred = scaler_y.inverse_transform(rnn_pred)
blstm_pred = scaler_y.inverse_transform(blstm_pred)
y_test_orig = scaler_y.inverse_transform(y_test[n_input:])

# Przygotowanie dat dla wykresu SPX
dates_spx = df['Date'].values[train_size + n_input:train_size + n_input + len(y_test[n_input:])]
dates_spx = pd.to_datetime(dates_spx, unit='D')

# Przygotowanie danych dla DJIA
X_dji = df2[features].values
X_dji_scaled = scaler_X.transform(X_dji)
y_dji = df2[target].values.reshape(-1, 1)
y_dji_scaled = scaler_y.transform(y_dji)

# Podział danych DJIA
train_size_dji = int(len(X_dji_scaled) * 0.8)
X_train_dji = X_dji_scaled[:train_size_dji]
X_test_dji = X_dji_scaled[train_size_dji:]
y_train_dji = y_dji_scaled[:train_size_dji]
y_test_dji = y_dji_scaled[train_size_dji:]

# Generatory danych dla DJIA
train_generator_dji = TimeseriesGenerator(X_train_dji, y_train_dji, length=n_input, batch_size=batch_size)
test_generator_dji = TimeseriesGenerator(X_test_dji, y_test_dji, length=n_input, batch_size=batch_size)

# Generowanie predykcji dla DJIA
print("\nGenerowanie predykcji dla DJIA...")
lstm_pred_dji = lstm_model.predict(test_generator_dji)
rnn_pred_dji = rnn_model.predict(test_generator_dji)
blstm_pred_dji = blstm_model.predict(test_generator_dji)

# Przekształcenie predykcji DJIA z powrotem do oryginalnej skali
lstm_pred_dji = scaler_y.inverse_transform(lstm_pred_dji)
rnn_pred_dji = scaler_y.inverse_transform(rnn_pred_dji)
blstm_pred_dji = scaler_y.inverse_transform(blstm_pred_dji)
y_test_orig_dji = scaler_y.inverse_transform(y_test_dji[n_input:])

# Przygotowanie dat dla wykresu DJIA
dates_dji = df2['Date'].values[train_size_dji + n_input:train_size_dji + n_input + len(y_test_dji[n_input:])]
dates_dji = pd.to_datetime(dates_dji, unit='D')

# Wykresy porównawcze
plt.figure(figsize=(15, 20))

# Wykres 1: Porównanie strat podczas treningu
plt.subplot(2, 1, 1)

# Bezpieczne rysowanie historii treningu
def plot_training_history(history, label_prefix):
    if history and isinstance(history, dict):
        if 'loss' in history:
            plt.plot(history['loss'], label=f'{label_prefix} - Strata treningowa')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label=f'{label_prefix} - Strata walidacyjna')

# Rysowanie historii dla każdego modelu
if lstm_history and isinstance(lstm_history, dict):
    plot_training_history(lstm_history, 'LSTM')
if rnn_history and isinstance(rnn_history, dict):
    plot_training_history(rnn_history, 'RNN')
if blstm_history and isinstance(blstm_history, dict):
    plot_training_history(blstm_history, 'BLSTM')

plt.title('Porównanie strat podczas treningu')
plt.xlabel('Epoka')
plt.ylabel('Strata (MSE)')
plt.legend()
plt.grid(True)

# Wykres 2: Porównanie predykcji dla SPX
plt.subplot(3, 1, 2)
plt.plot(dates_spx, y_test_orig, label='Rzeczywiste wartości SPX', color='black', linewidth=2)
plt.plot(dates_spx, lstm_pred, label='Predykcja LSTM', color='blue', alpha=0.7)
plt.plot(dates_spx, rnn_pred, label='Predykcja RNN', color='red', alpha=0.7)
plt.plot(dates_spx, blstm_pred, label='Predykcja BLSTM', color='green', alpha=0.7)
plt.title('Porównanie predykcji modeli dla indeksu SPX')
plt.xlabel('Data')
plt.ylabel('Cena zamknięcia SPX')
plt.legend()
plt.grid(True)

# Wykres 3: Porównanie predykcji dla DJIA
plt.subplot(3, 1, 3)
plt.plot(dates_dji, y_test_orig_dji, label='Rzeczywiste wartości DJIA', color='black', linewidth=2)
plt.plot(dates_dji, lstm_pred_dji, label='Predykcja LSTM', color='blue', alpha=0.7)
plt.plot(dates_dji, rnn_pred_dji, label='Predykcja RNN', color='red', alpha=0.7)
plt.plot(dates_dji, blstm_pred_dji, label='Predykcja BLSTM', color='green', alpha=0.7)
plt.title('Porównanie predykcji modeli dla indeksu DJIA')
plt.xlabel('Data')
plt.ylabel('Cena zamknięcia DJIA')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Funkcja do bezpiecznego obliczania metryk
def calculate_metrics_safely(y_true, y_pred, model_name):
    try:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n{model_name}:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R2: {r2:.6f}")
    except Exception as e:
        print(f"Błąd podczas obliczania metryk dla {model_name}: {str(e)}")

# Obliczenie metryk dla obu indeksów
print("\nMetryki dla modeli (SPX):")
calculate_metrics_safely(y_test_orig, lstm_pred, "LSTM")
calculate_metrics_safely(y_test_orig, rnn_pred, "RNN")
calculate_metrics_safely(y_test_orig, blstm_pred, "BLSTM")

print("\nMetryki dla modeli (DJIA):")
calculate_metrics_safely(y_test_orig_dji, lstm_pred_dji, "LSTM")
calculate_metrics_safely(y_test_orig_dji, rnn_pred_dji, "RNN")
calculate_metrics_safely(y_test_orig_dji, blstm_pred_dji, "BLSTM")

# Funkcja do tworzenia wykresów dla czarnych łabędzi
def plot_black_swan_predictions(event_name, start_date, end_date, df_spx, df_dji, models_dict, scaler_y):
    # Rozszerzamy okres o miesiąc przed i po
    start_date = pd.to_datetime(start_date) - pd.DateOffset(months=1)
    end_date = pd.to_datetime(end_date) + pd.DateOffset(months=1)
    
    # Filtrowanie danych dla SPX
    mask_spx = (df_spx['Date'] >= start_date) & (df_spx['Date'] <= end_date)
    filtered_dates_spx = df_spx.loc[mask_spx, 'Date']
    filtered_data_spx = df_spx.loc[mask_spx]
    
    # Filtrowanie danych dla DJIA
    mask_dji = (df_dji['Date'] >= start_date) & (df_dji['Date'] <= end_date)
    filtered_dates_dji = df_dji.loc[mask_dji, 'Date']
    filtered_data_dji = df_dji.loc[mask_dji]
    
    if len(filtered_dates_spx) == 0 or len(filtered_dates_dji) == 0:
        print(f"Brak danych dla okresu {event_name}")
        return
    
    # Przygotowanie danych do predykcji
    X_filtered_spx = filtered_data_spx[features].values
    X_filtered_spx_scaled = scaler_X.transform(X_filtered_spx)
    
    X_filtered_dji = filtered_data_dji[features].values
    X_filtered_dji_scaled = scaler_X.transform(X_filtered_dji)
    
    # Generowanie predykcji dla każdego modelu
    predictions_spx = {}
    predictions_dji = {}
    
    for model_name, model in models_dict.items():
        try:
            # Predykcje dla SPX
            pred_spx = model.predict(X_filtered_spx_scaled)
            predictions_spx[model_name] = scaler_y.inverse_transform(pred_spx)
            
            # Predykcje dla DJIA
            pred_dji = model.predict(X_filtered_dji_scaled)
            predictions_dji[model_name] = scaler_y.inverse_transform(pred_dji)
            
        except Exception as e:
            print(f"Błąd podczas generowania predykcji dla modelu {model_name}: {str(e)}")
            continue
    
    if not predictions_spx or not predictions_dji:
        print(f"Nie udało się wygenerować predykcji dla {event_name}")
        return
    
    # Tworzenie wykresów
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Wykres dla SPX
    ax1.plot(filtered_dates_spx, filtered_data_spx['Close'], 
             label='Rzeczywiste wartości SPX', color='black', linewidth=2)
    
    colors = {'lstm': 'blue', 'rnn': 'red', 'blstm': 'green'}
    for model_name, pred in predictions_spx.items():
        if 'without_onehot' in model_name:
            linestyle = '--'
            alpha = 0.7
        else:
            linestyle = '-'
            alpha = 1.0
        
        base_model = model_name.split('_')[0]
        ax1.plot(filtered_dates_spx, pred, 
                label=f'{model_name.upper()} SPX', 
                color=colors[base_model],
                linestyle=linestyle,
                alpha=alpha)
    
    # Wykres dla DJIA
    ax2.plot(filtered_dates_dji, filtered_data_dji['Close'], 
             label='Rzeczywiste wartości DJIA', color='black', linewidth=2)
    
    for model_name, pred in predictions_dji.items():
        if 'without_onehot' in model_name:
            linestyle = '--'
            alpha = 0.7
        else:
            linestyle = '-'
            alpha = 1.0
        
        base_model = model_name.split('_')[0]
        ax2.plot(filtered_dates_dji, pred, 
                label=f'{model_name.upper()} DJIA', 
                color=colors[base_model],
                linestyle=linestyle,
                alpha=alpha)
    
    # Dodanie pionowej linii dla daty czarnego łabędzia
    event_date = pd.to_datetime(start_date) + pd.DateOffset(months=1)
    ax1.axvline(x=event_date, color='red', linestyle='--', alpha=0.5, 
                label='Data czarnego łabędzia')
    ax2.axvline(x=event_date, color='red', linestyle='--', alpha=0.5, 
                label='Data czarnego łabędzia')
    
    # Konfiguracja wykresów
    ax1.set_title(f'Predykcje dla {event_name} - SPX')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Cena zamknięcia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.set_title(f'Predykcje dla {event_name} - DJIA')
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Cena zamknięcia')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Słownik modeli
models_dict = {
    'lstm': lstm_model,
    'lstm_without_onehot': lstm_model_without_onehot,
    'rnn': rnn_model,
    'rnn_without_onehot': rnn_model_without_onehot,
    'blstm': blstm_model,
    'blstm_without_onehot': blstm_model_without_onehot
}

# Tworzenie wykresów dla każdego czarnego łabędzia
for event_name, start_date, end_date in black_swan_periods:
    print(f"\nGenerowanie wykresu dla {event_name}...")
    plot_black_swan_predictions(event_name, start_date, end_date, df, models_dict, scaler_y)








