"""
🚀 MARIA HELENA TRADING BOT - LSTM TRAINER
Google Colab - Treinamento de Modelo de Deep Learning
Desenvolvedor: Marcos Sea (WSS13Framework)
Email: wss13.framework@gmail.com
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt
import sqlite3
import warnings
import sys

warnings.filterwarnings('ignore')

# Detectar se está no Google Colab ou ambiente local
IN_COLAB = 'google.colab' in sys.modules
if not IN_COLAB:
    # Modo headless para ambientes sem display (servidores/Linux)
    matplotlib.use('Agg')
    print("⚠️ Modo headless ativado - Gráficos serão salvos como imagens")
DB_PATH = "~/maria_helena_bot/maria_helena.sqlite"
FEATURES = ['close', 'high', 'low', 'volume', 'ema_200', 'sma_short', 'sma_long', 'rsi_14', 'atr_14', 'bb_upper', 'bb_lower', 'macd', 'macd_signal', 'donchian_high', 'donchian_low', 'obv']
TARGET_FEATURE = "close"


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
print("\n2️⃣ CARREGANDO DADOS ENRIQUECIDOS DO SQLITE...")

try:
    conn = sqlite3.connect(os.path.expanduser(DB_PATH))
    df = pd.read_sql_query(
        f"SELECT {', '.join(FEATURES)}, openTime FROM maria_helena_candles ORDER BY openTime ASC",
        conn
    )
    conn.close()
    
    df_clean = df.dropna(subset=FEATURES)
    
    if df_clean.empty:
        raise ValueError("DataFrame vazio após remover NaNs. Verifique os dados e o período dos indicadores.")
    
    print(f"✅ {len(df_clean)} candles enriquecidos carregados do SQLite!")
    print(f"   Primeiras linhas:")
    print(df_clean.head(3))

except Exception as e:
    print(f"❌ Erro ao carregar do SQLite: {str(e)}")
    print("Certifique-se de que 'calculate_indicators.py' foi executado e o DB existe.")
    print("Tentando carregar do GitHub (apenas 'close' sem indicadores, para fallback)...")
    url = "https://raw.githubusercontent.com/WSS13Framework/maria_helena_bot/main/bitcoin_training_data.csv"
    df_fallback = pd.read_csv(url)
    df_clean = df_fallback.dropna(subset=[TARGET_FEATURE])
    print(f"⚠️ {len(df_clean)} candles carregados do GitHub (apenas 'close'). Execute 'calculate_indicators.py' primeiro para ter todos os indicadores.")
    print(f"   Primeiras linhas:")
    print(df_clean.head(3))

print(f"\n📊 Dataset Final:")
print(f"   Total de candles processados: {len(df_clean)}")

# ============================================================
# CÉLULA 3: PREPARAR DADOS
# ============================================================
print("\n3️⃣ PREPARANDO DADOS COM MÚLTIPLAS FEATURES...")

data_features = df_clean[FEATURES].values

print(f"📊 Dados originais das features:")
for i, feature_name in enumerate(FEATURES):
    print(f"   {feature_name}: Min={data_features[:, i].min():,.2f}, Max={data_features[:, i].max():,.2f}")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_features = scaler.fit_transform(data_features)

print(f"✅ Todas as {len(FEATURES)} features normalizadas!")

target_close_prices = df_clean[TARGET_FEATURE].values.reshape(-1, 1)
target_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_target_close = target_scaler.fit_transform(target_close_prices)

lookback = 60
X_train = []
y_train = []

for i in range(lookback, len(scaled_data_features)):
    X_train.append(scaled_data_features[i-lookback:i, :])
    y_train.append(scaled_target_close[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], lookback, len(FEATURES)))

print(f"\n📈 Sequências criadas com {len(FEATURES)} features:")
print(f"   X_train shape: {X_train.shape} (amostras, lookback, features)")
print(f"   y_train shape: {y_train.shape} (targets)")

# ============================================================
# CÉLULA 4: CRIAR MODELO LSTM
# ============================================================
print("\n4️⃣ CRIANDO MODELO LSTM...")

model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(lookback, len(FEATURES)), name='LSTM_1'),
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
if IN_COLAB:
    plt.show()
else:
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    print("📊 Gráfico salvo: training_loss.png")

print(f"📊 Loss Final:")
print(f"   Train: {history.history['loss'][-1]:.6f}")
print(f"   Validation: {history.history['val_loss'][-1]:.6f}")

# ============================================================
# CÉLULA 7: FAZER PREDIÇÕES NO TREINO
# ============================================================
print("\n7️⃣ FAZENDO PREDIÇÕES NO TREINO...")

train_predict = model.predict(X_train, verbose=0)
train_predict = target_scaler.inverse_transform(train_predict)
y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1))

train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
train_mae = mean_absolute_error(y_train_actual, train_predict)

print(f"✅ Predições concluídas!")
print(f"📈 Métricas:")
print(f"   RMSE: ${train_rmse:,.2f}")
print(f"   MAE: ${train_mae:,.2f}")

plt.figure(figsize=(14, 6))
plt.plot(y_train_actual, label='Preço Real', linewidth=2)
plt.plot(train_predict, label='Preço Predito', linewidth=2, alpha=0.7)
plt.title('Maria Helena LSTM - Predições vs Real', fontsize=14, fontweight='bold')
plt.xlabel('Dias', fontsize=12)
plt.ylabel('Preço (USD)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
if IN_COLAB:
    plt.show()
else:
    plt.savefig('predictions_vs_real.png', dpi=150, bbox_inches='tight')
    print("📊 Gráfico salvo: predictions_vs_real.png")

# ============================================================
# CÉLULA 8: PREVER PRÓXIMO PREÇO
# ============================================================
print("\n8️⃣ PREVENDO PRÓXIMO PREÇO COM MULTI-FEATURES...")

try:
    conn_pred = sqlite3.connect(os.path.expanduser(DB_PATH))
    df_pred = pd.read_sql_query(
        f"SELECT {', '.join(FEATURES)}, openTime FROM maria_helena_candles ORDER BY openTime DESC LIMIT {lookback}",
        conn_pred
    )
    conn_pred.close()
    df_pred = df_pred.iloc[::-1]
    
    if len(df_pred) < lookback:
        raise ValueError(f"Não há dados suficientes no DB para prever. Necessário {lookback} candles, encontrado {len(df_pred)}.")

    last_lookback_features_values = df_pred[FEATURES].values
    last_lookback_scaled_features = scaler.transform(last_lookback_features_values)
    
    X_test_multi_feature = np.array([last_lookback_scaled_features])
    X_test_multi_feature = np.reshape(X_test_multi_feature, (1, lookback, len(FEATURES)))

    next_price_scaled_target = model.predict(X_test_multi_feature, verbose=0)
    next_price_actual = target_scaler.inverse_transform(next_price_scaled_target)[0][0]
    
    current_price = df_pred[TARGET_FEATURE].iloc[-1]
    change = next_price_actual - current_price
    change_pct = (change / current_price) * 100
    
    print(f"\n🔮 PREDIÇÃO DO PRÓXIMO PREÇO:")
    print(f"   Preço Atual: ${current_price:,.2f}")
    print(f"   Preço Predito: ${next_price_actual:,.2f}")
    print(f"   Mudança: ${change:,.2f} ({change_pct:+.2f}%)")

except Exception as e:
    print(f"❌ Erro na predição do próximo preço: {str(e)}")

# ============================================================
# CÉLULA 9: SALVAR MODELO
# ============================================================
print("\n9️⃣ SALVANDO MODELO...")

model.save('maria_helena_lstm_integrated_model.h5')
file_size = os.path.getsize('maria_helena_lstm_integrated_model.h5') / (1024 * 1024)

print(f"✅ Modelo salvo com sucesso!")
print(f"   Arquivo: maria_helena_lstm_integrated_model.h5")
print(f"   Tamanho: {file_size:.2f} MB")

# ============================================================
# CÉLULA 10: DOWNLOAD (GOOGLE COLAB)
# ============================================================
print("\n🔟 VERIFICANDO AMBIENTE...")

# Detectar se está no Google Colab
try:
    import sys
    if 'google.colab' in sys.modules:
        from google.colab import files
        files.download('maria_helena_lstm_integrated_model.h5')
        print("✅ Download iniciado (Google Colab)!")
    else:
        raise ImportError("Não está no Colab")
except (ImportError, ModuleNotFoundError):
    print("✅ Modo local - Arquivo salvo em:")
    print(f"   {os.path.abspath('maria_helena_lstm_integrated_model.h5')}")

print("\n" + "=" * 70)
print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("=" * 70)

print("\n🚀 PRÓXIMOS PASSOS:")
print("1. Salvar: maria_helena_lstm_integrated_model.h5")
print("2. Upload pro servidor:")
print("   scp maria_helena_lstm_integrated_model.h5 root@server:/root/maria-helena-scripts/")
print("3. Rodar predições:")
print("   python3 /root/maria-helena-scripts/run_lstm_predictions.py")

print("\n📞 DESENVOLVEDOR:")
print("   Marcos Sea (WSS13Framework)")
print("   Email: wss13.framework@gmail.com")
print("   GitHub: github.com/WSS13Framework/maria_helena_bot")