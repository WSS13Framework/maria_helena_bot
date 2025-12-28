"""
🚀 MARIA HELENA TRADING BOT - PREDIÇÃO LOCAL
Faz predições usando o modelo LSTM treinado
Desenvolvedor: Marcos Sea (WSS13Framework)
"""

import os
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
import joblib # Adicionado para carregar o scaler

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÕES
# ============================================================
DB_PATH = "~/maria_helena_bot/maria_helena.sqlite"
MODEL_PATH = "~/maria_helena_bot/maria_helena_lstm_integrated_model.h5"
# Define o caminho para o scaler salvo. Este arquivo deve ser gerado durante o treinamento do modelo.
SCALER_PATH = "~/maria_helena_bot/min_max_scaler.joblib" 
FEATURES = ['close', 'high', 'low', 'volume', 'ema_200', 'sma_short', 'sma_long', 
            'rsi_14', 'atr_14', 'bb_upper', 'bb_lower', 'macd', 'macd_signal', 
            'donchian_high', 'donchian_low', 'obv']
TARGET_FEATURE = "close"
LOOKBACK = 60

print("=" * 70)
print("🚀 MARIA HELENA TRADING BOT - PREDIÇÃO LOCAL")
print("=" * 70)
print(f"⏰ Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# CARREGAR MODELO
# ============================================================
print("\n1️⃣ CARREGANDO MODELO TREINADO...")

try:
    # Carregar modelo sem compilar (não precisa das métricas para predição)
    model = load_model(os.path.expanduser(MODEL_PATH), compile=False)
    print("✅ Modelo carregado com sucesso!")
    print(f"   Arquivo: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {str(e)}")
    print("   Execute primeiro: python3 maria_helena_lstm_integrated.py")
    exit(1)

# ============================================================
# CARREGAR DADOS DO SQLITE
# ============================================================
print("\n2️⃣ CARREGANDO DADOS DO SQLITE...")

try:
    conn = sqlite3.connect(os.path.expanduser(DB_PATH))
    
    # Buscar últimos dados suficientes para lookback
    df = pd.read_sql_query(
        f"""SELECT {', '.join(FEATURES)}, openTime 
            FROM maria_helena_candles 
            ORDER BY openTime DESC 
            LIMIT {LOOKBACK + 10}""",
        conn
    )
    conn.close()
    
    # Inverter para ordem cronológica
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Remover NaNs
    df_clean = df.dropna(subset=FEATURES)
    
    if len(df_clean) < LOOKBACK:
        raise ValueError(f"Dados insuficientes. Necessário {LOOKBACK}, encontrado {len(df_clean)}")
    
    print(f"✅ {len(df_clean)} candles carregados")
    print(f"   Período: {datetime.fromtimestamp(df_clean['openTime'].iloc[0]/1000).strftime('%Y-%m-%d %H:%M')}")
    print(f"          até {datetime.fromtimestamp(df_clean['openTime'].iloc[-1]/1000).strftime('%Y-%m-%d %H:%M')}")
    
except Exception as e:
    print(f"❌ Erro ao carregar dados: {str(e)}")
    print("   Execute primeiro: python3 calculate_indicators.py")
    exit(1)

# ============================================================
# PREPARAR DADOS PARA PREDIÇÃO
# ============================================================
print("\n3️⃣ PREPARANDO DADOS...")

# Pegar últimos LOOKBACK candles
last_data = df_clean.tail(LOOKBACK)[FEATURES].values

# --- INÍCIO DA ALTERAÇÃO: Carregar o scaler salvo ---
try:
    scaler = joblib.load(os.path.expanduser(SCALER_PATH))
    print(f"✅ Scaler carregado com sucesso de: {SCALER_PATH}")
except Exception as e:
    print(f"❌ Erro ao carregar scaler: {str(e)}")
    print("   Certifique-se de que o scaler foi salvo durante o treinamento do modelo.")
    print("   O scaler deve ser ajustado APENAS nos dados de treinamento e salvo.")
    exit(1)
# --- FIM DA ALTERAÇÃO ---

# Normalizar os últimos dados usando o scaler carregado
last_data_scaled = scaler.transform(last_data)

# Preparar para o modelo [1, lookback, features]
X_pred = np.reshape(last_data_scaled, (1, LOOKBACK, len(FEATURES)))

print(f"✅ Dados preparados")
print(f"   Shape: {X_pred.shape}")

# ============================================================
# FAZER PREDIÇÃO
# ============================================================
print("\n4️⃣ FAZENDO PREDIÇÃO...")

try:
    # Fazer predição (normalizada)
    prediction_scaled = model.predict(X_pred, verbose=0)
    
    # Desnormalizar apenas o preço (close)
    # Criar array com todas features zeradas, exceto close
    dummy_features = np.zeros((1, len(FEATURES)))
    close_index = FEATURES.index('close')
    dummy_features[0, close_index] = prediction_scaled[0, 0]
    
    # Desnormalizar
    prediction_unscaled = scaler.inverse_transform(dummy_features)
    predicted_price = prediction_unscaled[0, close_index]
    
    # Preço atual
    current_price = df_clean[TARGET_FEATURE].iloc[-1]
    
    # Calcular mudança
    change = predicted_price - current_price
    change_pct = (change / current_price) * 100
    
    print("✅ Predição concluída!")
    
except Exception as e:
    print(f"❌ Erro na predição: {str(e)}")
    exit(1)

# ============================================================
# ANÁLISE E RECOMENDAÇÃO
# ============================================================
print("\n" + "=" * 70)
print("📊 RESULTADO DA ANÁLISE")
print("=" * 70)

print(f"\n💰 PREÇOS:")
print(f"   Atual:    ${current_price:,.2f}")
print(f"   Predito:  ${predicted_price:,.2f}")
print(f"   Mudança:  ${change:+,.2f} ({change_pct:+.2f}%)")

# Determinar sinal
if change_pct > 0.5:
    signal = "🟢 COMPRA (BULLISH)"
    confidence = "Alta" if change_pct > 2 else "Média"
elif change_pct < -0.5:
    signal = "🔴 VENDA (BEARISH)"
    confidence = "Alta" if change_pct < -2 else "Média"
else:
    signal = "🟡 NEUTRO (HOLD)"
    confidence = "Baixa"

print(f"\n🎯 RECOMENDAÇÃO:")
print(f"   Sinal:      {signal}")
print(f"   Confiança:  {confidence}")

# Adicionar contexto de indicadores técnicos
print(f"\n📈 INDICADORES ATUAIS:")
print(f"   RSI-14:     {df_clean['rsi_14'].iloc[-1]:.2f}")
print(f"   MACD:       {df_clean['macd'].iloc[-1]:.2f}")
print(f"   EMA-200:    ${df_clean['ema_200'].iloc[-1]:,.2f}")

# Análise RSI
rsi = df_clean['rsi_14'].iloc[-1]
if rsi > 70:
    print(f"   ⚠️ RSI em sobrecompra ({rsi:.2f})")
elif rsi < 30:
    print(f"   ⚠️ RSI em sobrevenda ({rsi:.2f})")

# Análise tendência
if current_price > df_clean['ema_200'].iloc[-1]:
    print(f"   ✅ Preço acima da EMA-200 (tendência de alta)")
else:
    print(f"   ⚠️ Preço abaixo da EMA-200 (tendência de baixa)")

print("\n" + "=" * 70)
print("⚠️  AVISO: Esta é apenas uma predição baseada em machine learning.")
print("    Não é uma recomendação financeira. Faça sua própria análise!")
print("=" * 70)

print("\n📞 DESENVOLVEDOR:")
print("   Marcos Sea (WSS13Framework)")
print("   Email: wss13.framework@gmail.com")
print("   GitHub: github.com/WSS13Framework/maria_helena_bot")