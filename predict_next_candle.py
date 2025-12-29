import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import sqlite3
import ta # Biblioteca para indicadores técnicos

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÕES GLOBAIS ---
MODEL_PATH = './maria_helena_lstm_retrained.keras'
SCALER_PATH = './scaler (1).joblib'
DB_PATH = './maria_helena.sqlite' # Caminho para o seu banco de dados SQLite

FEATURES = [
    'close', 'high', 'low', 'volume', 'ema_200', 'sma_short', 'sma_long',
    'rsi_14', 'atr_14', 'bb_upper', 'bb_lower', 'macd', 'macd_signal',
    'donchian_high', 'donchian_low', 'obv'
]
LOOKBACK = 60 # O mesmo LOOKBACK usado no treinamento
TARGET_FEATURE_INDEX = FEATURES.index('close') # Índice da coluna 'close'

# Variáveis globais para armazenar o modelo e o scaler após o carregamento inicial
model = None
scaler = None

# --- FUNÇÕES DE CARREGAMENTO E PRÉ-PROCESSAMENTO ---

def load_model_and_scaler():
    """Carrega o modelo e o scaler uma única vez."""
    global model, scaler
    if model is None:
        logger.info(f"Carregando modelo de: {MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("✅ Modelo carregado com sucesso!")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar o modelo de {MODEL_PATH}: {e}", exc_info=True)
            return False
    
    if scaler is None:
        logger.info(f"Carregando scaler de: {SCALER_PATH}")
        try:
            scaler = joblib.load(SCALER_PATH)
            logger.info("✅ Scaler carregado com sucesso!")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar o scaler de {SCALER_PATH}: {e}", exc_info=True)
            return False
    return True

def get_last_candles_from_db(num_candles: int) -> pd.DataFrame:
    """
    Busca os últimos 'num_candles' do banco de dados.
    Retorna um DataFrame com as colunas necessárias para calcular os indicadores.
    """
    if not os.path.exists(DB_PATH):
        logger.error(f"❌ Banco de dados não encontrado em: {DB_PATH}.")
        return pd.DataFrame()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Precisamos de 'open', 'high', 'low', 'close', 'volume' para calcular os indicadores.
            # E também um pouco mais de dados históricos (ex: 200 candles para EMA_200)
            # para que os indicadores sejam calculados corretamente.
            # Vamos pegar um número maior de candles para garantir que todos os indicadores
            # possam ser calculados, e depois pegamos os últimos 'num_candles' deles.
            query = f"""
                SELECT openTime, open, high, low, close, volume
                FROM maria_helena_candles
                ORDER BY openTime DESC
                LIMIT {num_candles + 200} -- Pega mais candles para cálculo de indicadores
            """
            df = pd.read_sql_query(query, conn)
        
        df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
        df.set_index('openTime', inplace=True)
        df.sort_index(inplace=True) # Garante que os dados estão em ordem cronológica ascendente
        
        logger.info(f"✅ Últimos {num_candles + 200} candles brutos carregados do DB.")
        return df
    except Exception as e:
        logger.error(f"❌ Erro ao buscar candles do DB: {e}", exc_info=True)
        return pd.DataFrame()

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos os indicadores técnicos necessários para o modelo.
    Esta função DEVE ser idêntica à que você usou no treinamento.
    """
    if df.empty:
        logger.warning("DataFrame vazio para cálculo de indicadores.")
        return df

    # Certifique-se de que as colunas numéricas estão no tipo correto
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # EMA 200
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)

    # SMA (Short e Long) - Ex: 20 e 50 períodos
    df['sma_short'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_long'] = ta.trend.sma_indicator(df['close'], window=50)

    # RSI (Relative Strength Index)
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

    # ATR (Average True Range)
    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # Bandas de Bollinger
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()

    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # Donchian Channel - CORREÇÃO AQUI! Implementação manual
    # O erro 'AttributeError' indica que a função ta.trend.donchian_channel_hband não existe na versão instalada.
    # Vamos calcular manualmente, que é mais robusto e garante consistência com o que o modelo espera.
    window_donchian = 20 # Usar a mesma janela que você usou no treinamento
    df['donchian_high'] = df['high'].rolling(window=window_donchian).max()
    df['donchian_low'] = df['low'].rolling(window=window_donchian).min()

    # On-Balance Volume (OBV)
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    logger.info("✅ Indicadores técnicos calculados.")
    return df

def predict_next_candle_close() -> float | None:
    """
    Função principal para prever o preço de fechamento do próximo candle.
    """
    if not load_model_and_scaler():
        logger.error("❌ Falha ao carregar modelo ou scaler. Não é possível fazer a predição.")
        return None

    logger.info(f"🔄 Buscando os últimos {LOOKBACK} candles para predição...")
    df_raw_candles = get_last_candles_from_db(LOOKBACK)
    if df_raw_candles.empty:
        logger.error("❌ Não foi possível obter dados suficientes do DB para predição.")
        return None

    # Calcular indicadores para os dados brutos
    df_with_indicators = calculate_all_indicators(df_raw_candles)
    
    # Remover NAs que surgem do cálculo dos indicadores (primeiras linhas)
    df_with_indicators.dropna(inplace=True)

    if len(df_with_indicators) < LOOKBACK:
        logger.error(f"❌ Dados insuficientes ({len(df_with_indicators)} linhas) após cálculo de indicadores e remoção de NAs para o lookback de {LOOKBACK}.")
        logger.error("Verifique se há dados suficientes no DB ou se os indicadores estão gerando muitos NAs.")
        return None

    # Pegar apenas os últimos 'LOOKBACK' candles com indicadores calculados
    # e apenas as FEATURES que o modelo espera
    data_for_prediction_df = df_with_indicators[FEATURES].tail(LOOKBACK)

    if len(data_for_prediction_df) != LOOKBACK:
        logger.error(f"❌ Não foi possível obter exatamente {LOOKBACK} candles com FEATURES completas para predição. Obtido: {len(data_for_prediction_df)}")
        return None

    logger.info(f"✅ Dados prontos para escalonamento. Shape: {data_for_prediction_df.shape}")

    # Escalar os dados
    scaled_data_for_prediction = scaler.transform(data_for_prediction_df)

    # Remodelar para o formato que o LSTM espera: (1, LOOKBACK, num_features)
    reshaped_data = scaled_data_for_prediction.reshape(1, LOOKBACK, len(FEATURES))
    logger.info(f"✅ Dados formatados para LSTM. Shape: {reshaped_data.shape}")

    # Fazer a predição
    logger.info("🔄 Fazendo predição do próximo preço 'close'...")
    predicted_scaled_value = model.predict(reshaped_data, verbose=0)[0][0]
    logger.info(f"✅ Predição (escalada): {predicted_scaled_value}")

    # Desfazer a normalização
    dummy_row_for_inverse = np.zeros((1, len(FEATURES)))
    dummy_row_for_inverse[0, TARGET_FEATURE_INDEX] = predicted_scaled_value
    predicted_original_value = scaler.inverse_transform(dummy_row_for_inverse)[0, TARGET_FEATURE_INDEX]

    logger.info(f"✅ Predição final do preço 'close' (original): {predicted_original_value:.2f}")
    return predicted_original_value

if __name__ == "__main__":
    # Exemplo de como usar a função
    predicted_price = predict_next_candle_close()
    if predicted_price is not None:
        logger.info(f"A Maria Helena prevê que o próximo preço de fechamento será: {predicted_price:.2f}")
    else:
        logger.error("A Maria Helena não conseguiu fazer uma previsão.")