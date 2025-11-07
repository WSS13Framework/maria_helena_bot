"""
🚀 MARIA HELENA TRADING BOT - LSTM TRAINER
Google Colab - Treinamento de Modelo de Deep Learning
Desenvolvedor: Marcos Sea (WSS13Framework)
Email: wss13.framework@gmail.com
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 MARIA HELENA TRADING BOT - LSTM TRAINER")
print("=" * 70)

# ============================================================
# CÉLULA 1: IMPORTS E CONFIGURAÇÃO
# ============================================================
print("\n1️⃣ IMPORTANDO BIBLIOTECAS...")
print("✅ Bibliotecas importadas!")
print(f"   TensorFlow: {tf.__version__}")

# ============================================================
# CÉLULA 2: CARREGAR DADOS
# ============================================================
print("\n2️⃣ CARREGANDO DADOS...")

# Opção A: Do GitHub (RECOMENDADO)
try:
    url = "https://raw.githubusercontent.com/WSS13Framework/maria_helena_bot/main/bitcoin_training_data.csv"
    df = pd.read_csv(url)
    print(f"✅ {len(df)} candles carregados do GitHub!")
except:
    print("❌ Erro ao carregar do GitHub. Use upload manual.")
    from google.colab import files
    uploaded = files.upload()
    df = pd.read_csv(list(uploaded.keys())[0])

print(f"\n📊 Dataset:")
print(f"   Total de candles: {len(df)}")
print(f"   Primeiras linhas:")
print(df.head(3))

# ============================================================
# CÉLULA 3: PREPARAR DADOS
# ============================================================
print("\n3️⃣ PREPARANDO DADOS...")

# Extrair preço de fechamento
data = df['close'].values.reshape(-1, 1)

print(f"📊 Dados originais:")
print(f"   Min: ${data.min():,.2f}")
print(f"   Max: ${data.max():,.2f}")
print(f"   Média: ${data.mean():,.2f}")

# Normalizar entre 0 e 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

print(f"✅ Dados normalizados!")

# Criar sequences (60 dias → próximo dia)
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

print(f"\n📈 Sequências criadas:")
print(f"   X_train shape: {X_train.shape} (amostras, dias, features)")
print(f"   y_train shape: {y_train.shape} (targets)")

# ============================================================
# CÉLULA 4: CRIAR MODELO LSTM
# ============================================================
print("\n4️⃣ CRIANDO MODELO LSTM...")

model = Sequential([
    LSTM(50, activation='relu', input_shape=(lookback, 1), name='LSTM_1'),
    Dropout(0.2, name='Dropout_1'),
    LSTM(50, activation='relu', name='LSTM_2'),
    Dropout(0.2, name='Dropout_2'),
    Dense(25, activation='relu', name='Dense_1'),
    Dense(1, name='Output')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("✅ Modelo criado!")
print("\n🧠 ARQUITETURA:")
model.summary()

# ============================================================
# CÉLULA 5: TREINAR MODELO
# ============================================================
print("\n5️⃣ TREINANDO MODELO... ⏳")
print("=" * 70)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("=" * 70)
print("✅ Modelo treinado com sucesso!")

# ============================================================
# CÉLULA 6: PLOTAR LOSS
# ============================================================
print("\n6️⃣ PLOTANDO RESULTADOS DO TREINAMENTO...")

plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"📊 Loss Final:")
print(f"   Train: {history.history['loss'][-1]:.6f}")
print(f"   Validation: {history.history['val_loss'][-1]:.6f}")

# ============================================================
# CÉLULA 7: FAZER PREDIÇÕES NO TREINO
# ============================================================
print("\n7️⃣ FAZENDO PREDIÇÕES NO TREINO...")

train_predict = model.predict(X_train, verbose=0)
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
train_mae = mean_absolute_error(y_train_actual, train_predict)

print(f"✅ Predições concluídas!")
print(f"📈 Métricas:")
print(f"   RMSE: ${train_rmse:,.2f}")
print(f"   MAE: ${train_mae:,.2f}")

# Plotar
plt.figure(figsize=(14, 6))
plt.plot(y_train_actual, label='Preço Real', linewidth=2)
plt.plot(train_predict, label='Preço Predito', linewidth=2, alpha=0.7)
plt.title('Maria Helena LSTM - Predições vs Real', fontsize=14, fontweight='bold')
plt.xlabel('Dias', fontsize=12)
plt.ylabel('Preço (USD)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# CÉLULA 8: PREVER PRÓXIMO PREÇO
# ============================================================
print("\n8️⃣ PREVENDO PRÓXIMO PREÇO...")

last_60_days = data[-60:]
last_60_days_scaled = scaler.transform(last_60_days)
X_test = np.array([last_60_days_scaled])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

next_price = model.predict(X_test, verbose=0)
next_price = scaler.inverse_transform(next_price)

current_price = data[-1][0]
change = next_price[0][0] - current_price
change_pct = (change / current_price) * 100

print("\n🚀 PREDIÇÃO DO PRÓXIMO PREÇO:")
print("=" * 70)
print(f"💰 Preço atual: ${current_price:,.2f}")
print(f"📈 Preço predito: ${next_price[0][0]:,.2f}")
print(f"📊 Mudança: ${change:,.2f} ({change_pct:+.2f}%)")
print("=" * 70)

if change_pct > 2:
    signal = "🟢 BUY (Preço pode subir)"
elif change_pct < -2:
    signal = "🔴 SELL (Preço pode cair)"
else:
    signal = "🟡 HOLD (Sem mudança significativa)"

print(f"\n⚡ SIGNAL: {signal}")

# ============================================================
# CÉLULA 9: SALVAR MODELO
# ============================================================
print("\n9️⃣ SALVANDO MODELO...")

model.save('maria_helena_lstm_model.h5')

import os
file_size = os.path.getsize('maria_helena_lstm_model.h5') / (1024 * 1024)

print(f"✅ Modelo salvo com sucesso!")
print(f"   Arquivo: maria_helena_lstm_model.h5")
print(f"   Tamanho: {file_size:.2f} MB")

# ============================================================
# CÉLULA 10: DOWNLOAD
# ============================================================
print("\n🔟 FAZENDO DOWNLOAD DO MODELO...")

try:
    from google.colab import files
    files.download('maria_helena_lstm_model.h5')
    print("✅ Download iniciado!")
except:
    print("⚠️ Modo local - Arquivo salvo: maria_helena_lstm_model.h5")

print("\n" + "=" * 70)
print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("=" * 70)

print("\n🚀 PRÓXIMOS PASSOS:")
print("1. Salvar: maria_helena_lstm_model.h5")
print("2. Upload pro servidor:")
print("   scp maria_helena_lstm_model.h5 root@server:/root/maria-helena-scripts/")
print("3. Rodar predições:")
print("   python3 /root/maria-helena-scripts/run_lstm_predictions.py")

print("\n📞 DESENVOLVEDOR:")
print("   Marcos Sea (WSS13Framework)")
print("   Email: wss13.framework@gmail.com")
print("   GitHub: github.com/WSS13Framework/maria_helena_bot")

