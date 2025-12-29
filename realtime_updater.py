"""
🚀 MARIA HELENA TRADING BOT - ATUALIZADOR EM TEMPO REAL
Coleta dados da Binance continuamente e atualiza o SQLite.
Desenvolvedor: Marcos Sea (WSS13Framework)
"""

import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

# ============================================================
# CONFIGURAÇÕES
# ============================================================
SYMBOL = "BTCUSDT"
INTERVAL = "5m"  # Velas de 5 minutos
DB_PATH = os.path.expanduser("~/maria_helena_bot/maria_helena.sqlite")
UPDATE_INTERVAL = 300  # Atualizar a cada 5 minutos (300 segundos)
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
CANDLE_FETCH_LIMIT = 300  # Buscar 300 candles para garantir 250+ válidos após dropna

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("realtime_updater.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# ============================================================
# FUNÇÕES DE UTILIDADE
# ============================================================
def create_database_table() -> None:
    """Cria a tabela 'maria_helena_candles' no SQLite se ela não existir."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS maria_helena_candles (
                    openTime INTEGER PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    ema_200 REAL,
                    sma_short REAL,
                    sma_long REAL,
                    rsi_14 REAL,
                    atr_14 REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    macd REAL,
                    macd_signal REAL,
                    donchian_high REAL,
                    donchian_low REAL,
                    obv REAL
                )
            """)
            conn.commit()
        logger.info(f"Tabela 'maria_helena_candles' verificada/criada em {DB_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Erro ao criar/verificar tabela no banco de dados: {e}")
    except Exception as e:
        logger.error(f"Erro inesperado em create_database_table: {e}")

def get_candle_count() -> int:
    """Retorna a quantidade de candles no banco de dados."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM maria_helena_candles")
            count = cursor.fetchone()[0]
            return count
    except sqlite3.Error as e:
        logger.error(f"Erro ao obter contagem de candles do banco de dados: {e}")
        return 0
    except Exception as e:
        logger.error(f"Erro inesperado em get_candle_count: {e}")
        return 0

# ============================================================
# FUNÇÕES PRINCIPAIS
# ============================================================
def fetch_latest_candles(limit: int = CANDLE_FETCH_LIMIT) -> Optional[pd.DataFrame]:
    """
    Busca os últimos candles da Binance.
    Args:
        limit: Número máximo de candles a serem buscados.
    Returns:
        Um DataFrame do pandas com os dados dos candles, ou None em caso de erro.
    """
    try:
        params = {
            'symbol': SYMBOL,
            'interval': INTERVAL,
            'limit': limit
        }
        
        response = requests.get(BINANCE_API_URL, params=params, timeout=10)
        response.raise_for_status()  # Levanta HTTPError para códigos de status 4xx/5xx
        data = response.json()
        
        if not data:
            logger.warning("Nenhum dado de candle retornado pela Binance.")
            return None

        df = pd.DataFrame(data, columns=[
            'openTime', 'open', 'high', 'low', 'close', 'volume',
            'closeTime', 'quoteAssetVolume', 'numberOfTrades',
            'takerBuyBaseAssetVolume', 'takerBuyQuoteAssetVolume', 'ignore'
        ])
        
        # Converter tipos
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Use pd.to_numeric para lidar com erros
        
        df['openTime'] = pd.to_numeric(df['openTime'], errors='coerce').astype(int)
        
        # Remover linhas com valores NaN resultantes da conversão
        df.dropna(subset=['openTime', 'open', 'high', 'low', 'close', 'volume'], inplace=True)

        return df[['openTime', 'open', 'high', 'low', 'close', 'volume']]
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de requisição ao buscar dados da Binance: {e}")
        return None
    except ValueError as e:
        logger.error(f"Erro de valor ao processar dados da Binance (JSON ou conversão): {e}")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado em fetch_latest_candles: {e}")
        return None

def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calcula indicadores técnicos para o DataFrame de candles.
    Args:
        df: DataFrame contendo os candles (openTime, open, high, low, close, volume).
    Returns:
        DataFrame com os indicadores adicionados, ou None em caso de erro.
    """
    if df.empty:
        logger.warning("DataFrame vazio para cálculo de indicadores.")
        return None

    # Criar uma cópia para evitar SettingWithCopyWarning
    df_copy = df.copy()

    try:
        # EMA 200
        df_copy['ema_200'] = df_copy['close'].ewm(span=200, adjust=False).mean()
        
        # SMA Short (20) e Long (50)
        df_copy['sma_short'] = df_copy['close'].rolling(window=20).mean()
        df_copy['sma_long'] = df_copy['close'].rolling(window=50).mean()
        
        # RSI 14
        delta = df_copy['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_copy['rsi_14'] = 100 - (100 / (1 + rs))
        
        # ATR 14
        high_low = df_copy['high'] - df_copy['low']
        high_close = abs(df_copy['high'] - df_copy['close'].shift())
        low_close = abs(df_copy['low'] - df_copy['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df_copy['atr_14'] = true_range.rolling(14).mean()
        
        # Bollinger Bands
        sma_20 = df_copy['close'].rolling(window=20).mean()
        std_20 = df_copy['close'].rolling(window=20).std()
        df_copy['bb_upper'] = sma_20 + (std_20 * 2)
        df_copy['bb_lower'] = sma_20 - (std_20 * 2)
        
        # MACD
        exp1 = df_copy['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_copy['close'].ewm(span=26, adjust=False).mean()
        df_copy['macd'] = exp1 - exp2
        df_copy['macd_signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
        
        # Donchian Channel
        df_copy['donchian_high'] = df_copy['high'].rolling(window=20).max()
        df_copy['donchian_low'] = df_copy['low'].rolling(window=20).min()
        
        # OBV (On Balance Volume)
        # O cálculo do OBV pode ser sensível a NaNs iniciais, então garantimos que 'close' e 'volume' são numéricos
        df_copy['obv'] = (df_copy['volume'] * (~df_copy['close'].diff().le(0) * 2 - 1)).cumsum()
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Erro ao calcular indicadores: {e}")
        return None

def update_database(df: pd.DataFrame) -> bool:
    """
    Atualiza o banco de dados SQLite com novos candles e indicadores.
    Usa INSERT OR IGNORE para adicionar apenas novos registros, evitando duplicatas.
    Args:
        df: DataFrame contendo os candles com indicadores.
    Returns:
        True se a atualização for bem-sucedida, False caso contrário.
    """
    if df.empty:
        logger.warning("DataFrame vazio para atualização do banco de dados.")
        return False

    try:
        with sqlite3.connect(DB_PATH) as conn:
            # O 'if_exists='append'' combinado com 'openTime INTEGER PRIMARY KEY'
            # resultará em um IntegrityError para chaves duplicadas.
            # Para evitar isso, podemos usar um método mais explícito de INSERT OR IGNORE.
            
            # Alternativa 1: Inserir e ignorar erros de PK (mais simples com to_sql)
            # df.to_sql('maria_helena_candles', conn, if_exists='append', index=False)
            # logger.info("Dados inseridos/ignorados no banco de dados.")

            # Alternativa 2: Filtrar antes de inserir (mais performático para muitos dados)
            # Obter openTimes existentes no DB
            existing_open_times = pd.read_sql_query("SELECT openTime FROM maria_helena_candles", conn)['openTime'].tolist()
            
            # Filtrar o DataFrame para incluir apenas novos candles
            new_candles_df = df[~df['openTime'].isin(existing_open_times)]
            
            if not new_candles_df.empty:
                new_candles_df.to_sql('maria_helena_candles', conn, if_exists='append', index=False)
                logger.info(f"{len(new_candles_df)} novos candles inseridos no banco de dados.")
            else:
                logger.info("Nenhum novo candle para inserir no banco de dados.")
            
            conn.commit()
        return True
    
    except sqlite3.Error as e:
        logger.error(f"Erro de banco de dados ao atualizar: {e}")
        return False
    except Exception as e:
        logger.error(f"Erro inesperado em update_database: {e}")
        return False

def main_loop():
    """Loop principal de atualização em tempo real."""
    logger.info("=" * 70)
    logger.info("🚀 MARIA HELENA - ATUALIZADOR EM TEMPO REAL")
    logger.info("=" * 70)
    logger.info(f"📊 Símbolo: {SYMBOL}")
    logger.info(f"⏰ Intervalo: {INTERVAL}")
    logger.info(f"🔄 Atualização a cada: {UPDATE_INTERVAL} segundos")
    logger.info(f"💾 Banco de dados: {DB_PATH}")
    logger.info("=" * 70)
    logger.info("\n⏹️  Pressione Ctrl+C para parar\n")
    
    create_database_table() # Garante que a tabela exista ao iniciar

    iteration = 0
    
    while True:
        try:
            iteration += 1
            
            logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Atualização #{iteration}")
            
            # 1. Buscar dados da Binance
            logger.info("  └─ Buscando dados da Binance...")
            df = fetch_latest_candles() # Usa o CANDLE_FETCH_LIMIT configurado
            
            if df is None or df.empty:
                logger.warning("  ❌ Falhou ao buscar dados da Binance ou DataFrame vazio. Tentando novamente em 60s.")
                time.sleep(60)
                continue
            
            logger.info(f"  ✅ {len(df)} candles brutos recebidos.")
            
            # 2. Calcular indicadores
            logger.info("  └─ Calculando indicadores técnicos...")
            df_with_indicators = calculate_indicators(df)
            
            if df_with_indicators is None or df_with_indicators.empty:
                logger.warning("  ❌ Falhou ao calcular indicadores ou DataFrame vazio. Tentando novamente em 60s.")
                time.sleep(60)
                continue
            
            # Remover NaNs dos primeiros candles (onde indicadores não podem ser calculados)
            df_clean = df_with_indicators.dropna()
            
            if df_clean.empty:
                logger.warning("  ❌ DataFrame vazio após remover NaNs dos indicadores. Tentando novamente em 60s.")
                time.sleep(60)
                continue

            logger.info(f"  ✅ {len(df_clean)} candles válidos após cálculo de indicadores.")
            
            # 3. Atualizar banco de dados
            logger.info("  └─ Atualizando banco de dados...")
            success = update_database(df_clean)
            
            if not success:
                logger.error("  ❌ Falhou ao atualizar banco de dados. Tentando novamente em 60s.")
                time.sleep(60)
                continue
            
            total_candles = get_candle_count()
            logger.info(f"  ✅ Total de {total_candles} candles no DB.")
            
            # 4. Mostrar último preço
            if not df_clean.empty:
                last_price = df_clean['close'].iloc[-1]
                last_time = datetime.fromtimestamp(df_clean['openTime'].iloc[-1]/1000)
                logger.info(f"  └─ 💰 Último preço: ${last_price:,.2f} ({last_time.strftime('%H:%M:%S')})")
            else:
                logger.warning("  └─ Não há candles limpos para exibir o último preço.")
            
            # 5. Aguardar próxima atualização
            logger.info(f"  └─ ⏰ Próxima atualização em {UPDATE_INTERVAL}s...")
            logger.info("") # Linha em branco para separar logs
            
            time.sleep(UPDATE_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("\n\n⏹️  Parando atualizador...")
            logger.info("✅ Finalizado com sucesso!")
            break
        
        except Exception as e:
            logger.critical(f"\n❌ Erro crítico inesperado no loop principal: {e}", exc_info=True)
            logger.info("⏰ Tentando novamente em 60 segundos...")
            time.sleep(60)

# ============================================================
# EXECUTAR
# ============================================================
if __name__ == '__main__':
    main_loop()