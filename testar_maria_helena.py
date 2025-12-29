import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÕES ---
# Caminhos para o modelo e o scaler (AJUSTE CONFORME O SEU CASO)
MODEL_PATH = './maria_helena_lstm_retrained.keras'
SCALER_PATH = './scaler (1).joblib'

# As FEATURES DEVEM ser EXATAMENTE as mesmas usadas no treinamento
# e na mesma ordem!
FEATURES = ['close', 'high', 'low', 'volume', 'ema_200', 'sma_short', 'sma_long',
            'rsi_14', 'atr_14', 'bb_upper', 'bb_lower', 'macd', 'macd_signal',
            'donchian_high', 'donchian_low', 'obv']
LOOKBACK = 60 # O mesmo LOOKBACK usado no treinamento
TARGET_FEATURE_INDEX = FEATURES.index('close') # Índice da coluna 'close' na sua lista de FEATURES

# --- FUNÇÃO PRINCIPAL DE TESTE ---
def testar_modelo():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Modelo não encontrado em: {MODEL_PATH}")
        logger.info("Por favor, verifique o caminho do MODEL_PATH.")
        return
    if not os.path.exists(SCALER_PATH):
        logger.error(f"❌ Scaler não encontrado em: {SCALER_PATH}")
        logger.info("Por favor, verifique o caminho do SCALER_PATH.")
        return

    logger.info(f"Carregando modelo de: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✅ Modelo carregado com sucesso!")

    logger.info(f"Carregando scaler de: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    logger.info("✅ Scaler carregado com sucesso!")

    # --- Preparar dados de exemplo ---
    logger.info("Gerando dados de exemplo para predição...")
    # Criar um DataFrame de exemplo com o número correto de features e lookback
    # O modelo espera uma entrada 3D: (número de amostras, lookback, número de features)
    
    # Gerar dados aleatórios para simular 60 candles com 17 features
    # Em um cenário real, você buscaria os últimos 60 candles do seu banco de dados
    sample_data_raw = np.random.rand(LOOKBACK, len(FEATURES))
    sample_df = pd.DataFrame(sample_data_raw, columns=FEATURES)

    # Para a predição, é crucial que os dados sejam escalados com o MESMO scaler
    # usado no treinamento.
    # O scaler espera um array 2D (samples, features)
    sample_data_scaled = scaler.transform(sample_df)

    # O modelo LSTM espera uma entrada 3D (samples, lookback, features)
    # Como estamos prevendo uma única próxima vela, temos 1 amostra.
    sample_data_for_prediction = sample_data_scaled.reshape(1, LOOKBACK, len(FEATURES))

    logger.info(f"Dados de entrada para predição formatados: {sample_data_for_prediction.shape}")

    # --- Fazer a predição ---
    logger.info("Fazendo predição...")
    predicted_scaled_value = model.predict(sample_data_for_prediction)[0][0]
    logger.info(f"Predição (escalada): {predicted_scaled_value}")

    # --- Desfazer a normalização (Inverse Transform) ---
    logger.info("Desfazendo normalização da predição...")
    # O scaler foi treinado em todas as FEATURES. Para inverter a transformação de uma única feature,
    # precisamos criar um array com o mesmo número de features que o scaler espera.
    # Preenchemos as outras features com valores que não afetam a transformação da feature 'close'.
    # Uma estratégia comum é usar a média ou zeros, mas o mais seguro é preencher
    # com um valor qualquer (ex: 0) e colocar a predição no lugar correto.
    
    dummy_row_for_inverse = np.zeros((1, len(FEATURES)))
    dummy_row_for_inverse[0, TARGET_FEATURE_INDEX] = predicted_scaled_value
    
    predicted_original_value = scaler.inverse_transform(dummy_row_for_inverse)[0, TARGET_FEATURE_INDEX]

    logger.info(f"✅ Predição do preço 'close' (original): {predicted_original_value:.2f}")
    logger.info("\n--- Teste concluído com sucesso! ---")

if __name__ == "__main__":
    testar_modelo()