import os
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib # Para salvar o scaler

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
DB_PATH = "~/maria_helena_bot/maria_helena.sqlite"
MODEL_SAVE_PATH = "~/maria_helena_bot/maria_helena_lstm_classifier_model.h5"
SCALER_SAVE_PATH = "~/maria_helena_bot/min_max_scaler.joblib" # Caminho para salvar o scaler
FEATURES = ['close', 'high', 'low', 'volume', 'ema_200', 'sma_short', 'sma_long',
            'rsi_14', 'atr_14', 'bb_upper', 'bb_lower', 'macd', 'macd_signal',
            'donchian_high', 'donchian_low', 'obv']
LOOKBACK = 60 # Número de candles passados para cada previsão
FUTURE_HORIZON_CANDLES = 10 # TEMPORÁRIO: Apenas para testar a lógica com os dados atuais
THRESHOLD_PERCENTUAL = 0.0001 # 0.01% - TEMPORÁRIO: Apenas para depuração

# Parâmetros do Modelo LSTM
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10 # Para EarlyStopping

print("=" * 70)
print("🚀 MARIA HELENA TRADING BOT - TREINAMENTO LSTM CLASSIFICADOR")
print("=" * 70)
print(f"⏰ Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# 1. CARREGAR DADOS DO SQLITE
# ============================================================
print("\n1️⃣ CARREGANDO DADOS DO SQLITE...")
try:
    conn = sqlite3.connect(os.path.expanduser(DB_PATH))
    # Carregar todos os dados necessários para criar features e target
    df = pd.read_sql_query(
        f"""SELECT {', '.join(FEATURES)}, openTime
            FROM maria_helena_candles
            ORDER BY openTime ASC""", # Ordem cronológica ascendente
        conn
    )
    conn.close()
    
    # Remover NaNs iniciais que podem vir dos indicadores
    df_original = df.copy() # Manter uma cópia para referência
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if df.empty:
        raise ValueError("Nenhum dado válido após remover NaNs. Verifique a coleta de dados.")

    print(f"✅ {len(df)} candles carregados e limpos.")
    print(f"   Período: {datetime.fromtimestamp(df['openTime'].iloc[0]/1000).strftime('%Y-%m-%d %H:%M')}")
    print(f"          até {datetime.fromtimestamp(df['openTime'].iloc[-1]/1000).strftime('%Y-%m-%d %H:%M')}")

except Exception as e:
    print(f"❌ Erro ao carregar dados: {str(e)}")
    exit(1)

# ============================================================
# 2. DEFINIR TARGET DE CLASSIFICAÇÃO BINÁRIA
# ============================================================
print("\n2️⃣ DEFININDO TARGET DE CLASSIFICAÇÃO BINÁRIA...")

# Criar a coluna de preço futuro
df['future_close'] = df['close'].shift(-FUTURE_HORIZON_CANDLES)

# Calcular a mudança percentual
df['price_change_pct'] = (df['future_close'] - df['close']) / df['close']

# Definir o target binário (1: Compra, 0: Venda). Descartamos os neutros.
# Usamos 0 e 1 para binary_crossentropy
df['target'] = np.nan # Inicializa com NaN

# Compra (1) se o preço subir acima do threshold
df.loc[df['price_change_pct'] > THRESHOLD_PERCENTUAL, 'target'] = 1

# Venda (0) se o preço cair abaixo do threshold negativo
df.loc[df['price_change_pct'] < -THRESHOLD_PERCENTUAL, 'target'] = 0

# Remove linhas onde o target é NaN (casos neutros ou onde não há future_close)
df_classified = df.dropna(subset=['target']).reset_index(drop=True)

# Converter target para int
df_classified['target'] = df_classified['target'].astype(int)

# Verificar balanceamento das classes
buy_count = df_classified[df_classified['target'] == 1].shape[0]
sell_count = df_classified[df_classified['target'] == 0].shape[0]
total_classified = df_classified.shape[0]

print(f"✅ Target defi used.
/home/sea/.local/lib/python3.11/site-packages/keras/src/export/tf2onnx_lib.py:8: FutureWarning: In the future `np.object` will be defined as the corresponding NumPy scalar.
  if not hasattr(np, "object"):
======================================================================
🚀 MARIA HELENA TRADING BOT - TREINAMENTO LSTM CLASSIFICADOR
======================================================================
⏰ Executado em: 2025-12-28 08:08:34

1️⃣ CARREGANDO DADOS DO SQLITE...
✅ 201 candles carregados e limpos.
   Período: 2025-12-27 15:25
          até 2025-12-28 08:05

2️⃣ DEFININDO TARGET DE CLASSIFICAÇÃO BINÁRIA...
✅ Target definido para 56 amostras.
   Classe 'Compra' (1): 37 (66.07%)
   Classe 'Venda' (0): 19 (33.93%)

3️⃣ PREPARANDO DADOS PARA LSTM...
✅ Scaler salvo com sucesso em: ~/maria_helena_bot/min_max_scaler.joblib
✅ Sequências criadas. X_sequences shape: (0,), y_sequences shape: (0,)

4️⃣ REALIZANDO SPLIT TEMPORAL...
✅ Dados divididos:
   Treino: 0 amostras
   Validação: 0 amostras
   Teste: 0 amostras

5️⃣ CONSTRUINDO E TREINANDO MODELO LSTM CLASSIFICADOR...
2025-12-28 08:08:34.141151: E external/local_xla/xla/stream_executor/cuda/cuda_platform.cc:51] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
/home/sea/.local/lib/python3.11/site-packages/keras/src/layers/rnn/rnn.py:199: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
   Iniciando treinamento...
Epoch 1/100
Traceback (most recent call last):
  File "/home/sea/maria_helena_bot/train_lstm_classifier.py", line 186, in <module>
  File "/home/sea/.local/lib/python3.11/site-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/sea/.local/lib/python3.11/site-packages/keras/src/models/functional.py", line 278, in _adjust_input_rank
    raise ValueError(
ValueError: Exception encountered when calling Sequential.call().

[1mInvalid input shape for input Tensor("data:0", shape=(32,), dtype=float32) with name 'keras_tensor' and path ''. Expected shape (None, 60, 16), but input has incompatible shape (32,)[0m

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(32,), dtype=float32)
  • training=True
  • mask=None
  • kwargs=<class 'inspect._empty'>
[?2004h]0;sea@pop-os: ~/maria_helena_bot[01;32msea@pop-os[00m:[01;34m~/maria_helena_bot[00m$ [H[2J]0;sea@pop-os: ~/maria_helena_bot[01;32msea@pop-os[00m:[01;34m~/maria_helena_bot[00m$ python train_lstm_classifier.py[K[H[2J]0;sea@pop-os: ~/maria_helena_bot[01;32msea@pop-os[00m:[01;34m~/maria_helena_bot[00m$ 
[?2004l[?2004h]0;sea@pop-os: ~/maria_helena_bot[01;32msea@pop-os[00m:[01;34m~/maria_helena_bot[00m$ python train_lstm_classifier.py
[?2004l2025-12-28 08:11:59.707178: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2025-12-28 08:11:59.818202: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-12-28 08:12:03.136602: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
/home/sea/.local/lib/python3.11/site-packages/keras/src/export/tf2onnx_lib.py:8: FutureWarning: In the future `np.object` will be defined as the corresponding NumPy scalar.
  if not hasattr(np, "object"):
======================================================================
🚀 MARIA HELENA TRADING BOT - TREINAMENTO LSTM CLASSIFICADOR
======================================================================
⏰ Executado em: 2025-12-28 08:12:04

1️⃣ CARREGANDO DADOS DO SQLITE...
✅ 201 candles carregados e limpos.
   Período: 2025-12-27 15:30
          até 2025-12-28 08:10

2️⃣ DEFININDO TARGET DE CLASSIFICAÇÃO BINÁRIA...
✅ Target definido para 178 amostras.
   Classe 'Compra' (1): 105 (58.99%)
   Classe 'Venda' (0): 73 (41.01%)

3️⃣ PREPARANDO DADOS PARA LSTM...
✅ Scaler salvo com sucesso em: ~/maria_helena_bot/min_max_scaler.joblib
✅ Sequências criadas. X_sequences shape: (118, 60, 16), y_sequences shape: (118,)

4️⃣ REALIZANDO SPLIT TEMPORAL...
✅ Dados divididos:
   Treino: 82 amostras, X_train shape: (82, 60, 16), y_train shape: (82,)
   Validação: 17 amostras, X_val shape: (17, 60, 16), y_val shape: (17,)
   Teste: 19 amostras, X_test shape: (19, 60, 16), y_test shape: (19,)

5️⃣ CONSTRUINDO E TREINANDO MODELO LSTM CLASSIFICADOR...
2025-12-28 08:12:04.978263: E external/local_xla/xla/stream_executor/cuda/cuda_platform.cc:51] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
/home/sea/.local/lib/python3.11/site-packages/keras/src/layers/rnn/rnn.py:199: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
   Iniciando treinamento...
Epoch 1/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m12s[0m 6s/step - accuracy: 0.3438 - loss: 0.7068[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 76ms/step - accuracy: 0.3906 - loss: 0.7041[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 82ms/step - accuracy: 0.4108 - loss: 0.7020WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 505ms/step - accuracy: 0.4512 - loss: 0.6978 - val_accuracy: 0.0000e+00 - val_loss: 0.8602
Epoch 2/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 160ms/step - accuracy: 0.6250 - loss: 0.6607[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 84ms/step - accuracy: 0.6094 - loss: 0.6662 [1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 71ms/step - accuracy: 0.5973 - loss: 0.6713[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 133ms/step - accuracy: 0.5732 - loss: 0.6816 - val_accuracy: 0.0000e+00 - val_loss: 0.9432
Epoch 3/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 114ms/step - accuracy: 0.5000 - loss: 0.6944[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 67ms/step - accuracy: 0.5391 - loss: 0.6804 [1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - accuracy: 0.5504 - loss: 0.6765[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 114ms/step - accuracy: 0.5732 - loss: 0.6685 - val_accuracy: 0.0000e+00 - val_loss: 0.8932
Epoch 4/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 131ms/step - accuracy: 0.5625 - loss: 0.6883[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 163ms/step - accuracy: 0.5703 - loss: 0.6876[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 165ms/step - accuracy: 0.5753 - loss: 0.6861WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 431ms/step - accuracy: 0.5854 - loss: 0.6830 - val_accuracy: 0.0000e+00 - val_loss: 0.8440
Epoch 5/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 215ms/step - accuracy: 0.6562 - loss: 0.6601[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 116ms/step - accuracy: 0.6406 - loss: 0.6634[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 111ms/step - accuracy: 0.6181 - loss: 0.6682[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 169ms/step - accuracy: 0.5732 - loss: 0.6778 - val_accuracy: 0.0000e+00 - val_loss: 0.8521
Epoch 6/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 100ms/step - accuracy: 0.6250 - loss: 0.6754[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 68ms/step - accuracy: 0.6172 - loss: 0.6698 [1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - accuracy: 0.6147 - loss: 0.6690[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 120ms/step - accuracy: 0.6098 - loss: 0.6673 - val_accuracy: 0.0000e+00 - val_loss: 0.8470
Epoch 7/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 110ms/step - accuracy: 0.5938 - loss: 0.6972[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 68ms/step - accuracy: 0.6250 - loss: 0.6858 [1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step - accuracy: 0.6362 - loss: 0.6818[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 123ms/step - accuracy: 0.6585 - loss: 0.6738 - val_accuracy: 0.0000e+00 - val_loss: 0.8480
Epoch 8/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 150ms/step - accuracy: 0.6875 - loss: 0.6640[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 87ms/step - accuracy: 0.6875 - loss: 0.6603 [1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 81ms/step - accuracy: 0.6738 - loss: 0.6636[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 184ms/step - accuracy: 0.6463 - loss: 0.6702 - val_accuracy: 0.0000e+00 - val_loss: 0.8809
Epoch 9/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 237ms/step - accuracy: 0.6562 - loss: 0.6556[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 115ms/step - accuracy: 0.6172 - loss: 0.6664[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 115ms/step - accuracy: 0.6025 - loss: 0.6702[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 311ms/step - accuracy: 0.5732 - loss: 0.6778 - val_accuracy: 0.0000e+00 - val_loss: 0.8922
Epoch 10/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m1s[0m 536ms/step - accuracy: 0.6250 - loss: 0.6688[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 184ms/step - accuracy: 0.6172 - loss: 0.6702[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 138ms/step - accuracy: 0.6228 - loss: 0.6677[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 224ms/step - accuracy: 0.6341 - loss: 0.6626 - val_accuracy: 0.0000e+00 - val_loss: 0.8960
Epoch 11/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 136ms/step - accuracy: 0.5312 - loss: 0.7330[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 94ms/step - accuracy: 0.5625 - loss: 0.7168 [1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 84ms/step - accuracy: 0.5823 - loss: 0.7059[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 151ms/step - accuracy: 0.6220 - loss: 0.6841 - val_accuracy: 0.0000e+00 - val_loss: 0.9194
Epoch 12/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 173ms/step - accuracy: 0.6875 - loss: 0.6530[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 130ms/step - accuracy: 0.6719 - loss: 0.6549[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 128ms/step - accuracy: 0.6593 - loss: 0.6601[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 259ms/step - accuracy: 0.6341 - loss: 0.6705 - val_accuracy: 0.0000e+00 - val_loss: 0.9663
Epoch 13/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 303ms/step - accuracy: 0.5938 - loss: 0.6673[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 143ms/step - accuracy: 0.5938 - loss: 0.6701[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 157ms/step - accuracy: 0.5991 - loss: 0.6684[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 314ms/step - accuracy: 0.6098 - loss: 0.6652 - val_accuracy: 0.0000e+00 - val_loss: 0.9185
Epoch 14/100
[1m1/3[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m1s[0m 519ms/step - accuracy: 0.5312 - loss: 0.6979[1m2/3[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 109ms/step - accuracy: 0.5781 - loss: 0.6837[1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 95ms/step - accuracy: 0.6009 - loss: 0.6750 [1m3/3[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 161ms/step - accuracy: 0.6463 - loss: 0.6574 - val_accuracy: 0.0000e+00 - val_loss: 0.8890
✅ Treinamento concluído! Modelo salvo em: ~/maria_helena_bot/maria_helena_lstm_classifier_model.h5

6️⃣ AVALIANDO MODELO NO CONJUNTO DE TESTE...
✅ Avaliação no conjunto de teste:
   Loss: 0.6930
   Accuracy: 0.5263
[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 651ms/step[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 683ms/step

======================================================================
✅ TREINAMENTO LSTM CLASSIFICADOR CONCLUÍDO COM SUCESSO!
   Pronto para integrar ao ensemble.
======================================================================
[?2004h]0;sea@pop-os: ~/maria_helena_bot[01;32msea@pop-os[00m:[01;34m~/maria_helena_bot[00m$ [7mgit fetch origin[27m
[7mgit checkout 2-feature-implementar-ensembles-de-modelos-para-predição[27m[A[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[Cgit fetch origin
git checkout 2-feature-implementar-ensembles-de-modelos-para-predição
[?2004lerror: pathspec '2-feature-implementar-ensembles-de-modelos-para-predição' did not match any file(s) known to git
[?2004h]0;sea@pop-os: ~/maria_helena_bot[01;32msea@pop-os[00m:[01;34m~/maria_helena_bot[00m$ nao[Kno [7m unify_and_import_raw_data.py Atualizado[27m[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C unify_and_import_raw_data.py Atualizado
[?2004l[?2004h[?1049h[22;0;0t[1;25r(B[m[4l[?7h[39;49m[?1h=[?1h=[?25l[39;49m(B[m[H[2J[23;71H(B[0;7m[ Novo arquivo ](B[m[23;58H(B[0;7m[ unify_and_import_raw_data.py -- 0 linha ](B[m[H(B[0;7m  [1/2]                                                        unify_and_import_raw_data.py                                                                  [1;156H(B[m[24d(B[0;7m^G(B[m Ajuda[24;18H(B[0;7m^O(B[m Gravar[24;35H(B[0;7m^W(B[m Onde está?    (B[0;7m^K(B[m Recortar[69G(B[0;7m^T(B[m Executar[86G(B[0;7m^C(B[m Local[24;103H(B[0;7mM-U(B[m Desfazer     (B[0;7mM-A(B[m Marcar[137G(B[0;7mM-](B[m Parênteses[25d(B[0;7m^X(B[m Fechar[25;18H(B[0;7m^R(B[m Ler o arq     (B[0;7m^\(B[m Substituir    (B[0;7m^U(B[m Colar[25;69H(B[0;7m^J(B[m Justificar    (B[0;7m^/(B[m Ir p/ linha   (B[0;7mM-E(B[m Refazer[120G(B[0;7mM-6(B[m Copiar[137G(B[0;7m^Q(B[m Onde estava[2d[?12l[?25h[?25l[23d[K[1;93H(B[0;7m*[156G(B[m[2dprint((B[0;1m[32m"="[39m(B[m * 70)[3dprint((B[0;1m[32m"🚀 INICIANDO UNIFICAÇÃO E IMPORTAÇÃO DE DADOS BRUTOS PARA SQLITE"[39m(B[m)[4dprint((B[0;1m[32m"="[39m(B[m * 70)[5dprint(f(B[0;1m[32m"⏰ Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"[39m(B[m)[7dunify_and_import_raw_data(RAW_CSV_FOLDER, DB_PATH)[9dprint((B[0;1m[32m"(B[0m[38;5;148m\n(B[0;1m[32m"[39m(B[m + (B[0;1m[32m"="[39m(B[m * 70)[10dprint((B[0;1m[32m"✅ PROCESSO DE UNIFICAÇÃO E IMPORTAÇÃO CONCLUÍDO."[39m(B[m)[11dprint((B[0;1m[32m"   Seu terreno está arrumado e a fundação está pronta!"[39m(B[m)[12dprint((B[0;1m[32m"="[39m(B[m * 70)[?12l[?25h[?25l[?12l[?25h[A[?25l[?12l[?25h[12d[?25l[24;18H         [24;35H     (B[0;7mM-D(B[m Formato DOS        [69G          (B[0;7mM-A(B[m Anexar     [24;103H[15X[24;118H(B[0;7mM-B(B[m Arquivo reserva[K[25;2H(B[0;7mC(B[m Cancelar[18G[22X[25;40H(B[0;7mM-M(B[m Formato Mac     [25;69H          (B[0;7mM-P(B[m Pré-anexar[25X[25;118H(B[0;7m^T(B[m Navegar[K[23d(B[0;7mNome do arquivo para salvar: unify_and_import_raw_data.py                                                                                                    [23;58H(B[m[?12l[?25h[?25l[23;70H[1K (B[0;7m[ Escrevendo... ](B[m[K[1;93H(B[0;7m [156G(B[m[23;68H(B[0;7m[ Escritas 167 linhas ](B[m[24;18H(B[0;7m^O(B[m Gravar[24;35H(B[0;7m^W(B[m Onde está?    (B[0;7m^K(B[m Recortar[69G(B[0;7m^T(B[m Executar      (B[0;7m^C(B[m Local[24;103H(B[0;7mM-U(B[m Desfazer     (B[0;7mM-A(B[m Marcar       (B[0;7mM-](B[m Parênteses[25;2H(B[0;7mX(B[m Fechar  [18G(B[0;7m^R(B[m Ler o arq     (B[0;7m^\(B[m Substituir    (B[0;7m^U(B[m Colar[25;69H(B[0;7m^J(B[m Justificar    (B[0;7m^/(B[m Ir p/ linha   (B[0;7mM-E(B[m Refazer      (B[0;7mM-6(B[m Copiar[137G(B[0;7m^Q(B[m Onde estava[?12l[?25h[12;16H[?25l[23;48H(B[0;7m[ linha 167/168 (99%), col 16/16 (100%), car 8829/8830 (99%) ](B[m[?12l[?25h[12;16H[?25l[23;110H[?12l[?25h[12;16H[?25l[1;93H(B[0;7m*[156G(B[m[13d[?12l[?25h[?25l[24;18H         [24;35H     (B[0;7mM-D(B[m Formato DOS        [69G          (B[0;7mM-A(B[m Anexar     [24;103H[15X[24;118H(B[0;7mM-B(B[m Arquivo reserva[K[25;2H(B[0;7mC(B[m Cancelar[18G[22X[25;40H(B[0;7mM-M(B[m Formato Mac     [25;69H          (B[0;7mM-P(B[m Pré-anexar[25X[25;118H(B[0;7m^T(B[m Navegar[K[23d(B[0;7mNome do arquivo para salvar: unify_and_import_raw_data.py                                                                                                    [23;58H(B[m[?12l[?25h[?25l[23;70H[1K (B[0;7m[ Escrevendo... ](B[m[K[1;93H(B[0;7m [156G(B[m[23;68H(B[0;7m[ Escritas 168 linhas ](B[m[24;18H(B[0;7m^O(B[m Gravar[24;35H(B[0;7m^W(B[m Onde está?    (B[0;7m^K(B[m Recortar[69G(B[0;7m^T(B[m Executar      (B[0;7m^C(B[m Local[24;103H(B[0;7mM-U(B[m Desfazer     (B[0;7mM-A(B[m Marcar       (B[0;7mM-](B[m Parênteses[25;2H(B[0;7mX(B[m Fechar  [18G(B[0;7m^R(B[m Ler o arq     (B[0;7m^\(B[m Substituir    (B[0;7m^U(B[m Colar[25;69H(B[0;7m^J(B[m Justificar    (B[0;7m^/(B[m Ir p/ linha   (B[0;7mM-E(B[m Refazer      (B[0;7mM-6(B[m Copiar[137G(B[0;7m^Q(B[m Onde estava[?12l[?25h[13d[?25l[1;4H(B[0;7m2[1;64H         Atualizado         [156G(B[m[23;67H(B[0;7m[ Atualizado -- 0 linha ](B[m[1;4H(B[0;7m1/1[156G(B[m[25;4HSair  [151G[2d[K[3d[K[4d[K[5d[K[7d[K[9d[K[10d[K[11d[K[12d[K[2d[?12l[?25h[?25l[?12l[?25h[?25l[?12l[?25h[?25l[?12l[?25h