"""
🚀 MARIA HELENA TRADING BOT - LSTM TRAINER
Google Colab Notebook (Convert to .ipynb)
"""

# ============================================================
# CÉLULA 1: Importar bibliotecas
# ============================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("✅ Bibliotecas importadas!")

# ============================================================
# CÉLULA 2: Carregar dados do GitHub/Drive
# ============================================================
# Opção A: Do GitHub
url = "https://raw.githubusercontent.com/seu-usuario/maria-helena/main/bitcoin_training_data.csv"
df = pd.read_csv(url)

# Opção B: Do Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# df = pd.read_csv('/content/drive/My Drive/maria-helena/bitcoin_training_data.csv')

print(f"✅ {len(df)} candles carregados!")
print(df.head())

# ============================================================
# CÉLULA 3: Preparar dados
# ============================================================
# Usar close price para treinar
data = df['close'].values.reshape(-1, 1)

# Normalizar
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Criar sequences de treino (60 dias anteriores -> próximo dia)
lookback = 60
X_train = []
y_train = []

for i in range(lookback, len(scaled_data)):
    X_train.append(scaled_data[i-lookback:i, 0])
    y_train.append(scaled_data[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape para LSTM [samples, timesteps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(f"✅ Dados preparados!")
print(f"   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")

# ============================================================
# CÉLULA 4: Criar modelo LSTM
# ============================================================
model = Sequential([
    LSTM(50, activation='relu', input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

print("✅ Modelo criado!")

# ============================================================
# CÉLULA 5: Treinar modelo
# ============================================================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("✅ Modelo treinado!")

# ============================================================
# CÉLULA 6: Plotar loss
# ============================================================
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# CÉLULA 7: Fazer predições
# ============================================================
# Predições no conjunto de treino
train_predict = model.predict(X_train)
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

# Calcular RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
print(f"Train RMSE: ${train_rmse:.2f}")

# ============================================================
# CÉLULA 8: Prever próximo preço
# ============================================================
# Usar últimos 60 candles pra prever o próximo
last_60_days = data[-60:]
last_60_days_scaled = scaler.transform(last_60_days)
X_test = np.array([last_60_days_scaled])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

next_price = model.predict(X_test)
next_price = scaler.inverse_transform(next_price)

print(f"🚀 Preço predito: ${next_price[0][0]:.2f}")
print(f"💰 Preço atual: ${data[-1][0]:.2f}")

# ============================================================
# CÉLULA 9: Salvar modelo
# ============================================================
model.save('maria_helena_lstm_model.h5')
print("✅ Modelo salvo como maria_helena_lstm_model.h5")

# ============================================================
# CÉLULA 10: Exportar pra servidor
# ============================================================
# Copiar arquivo do Colab pra GitHub/Drive
# ! cp maria_helena_lstm_model.h5 /content/drive/My\ Drive/
print("✅ Salve o arquivo: maria_helena_lstm_model.h5")

