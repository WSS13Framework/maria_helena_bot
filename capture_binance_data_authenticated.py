#!/usr/bin/env python3
import requests
import json
import sqlite3
import time
from datetime import datetime, timedelta
import logging
import os # Importar para verificar existência do arquivo de config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- FUNÇÃO: Carregar credenciais da Binance ---
def load_binance_credentials(config_path="~/.binance_config"):
    """
    Carrega as credenciais da Binance de um arquivo de configuração.
    Espera o formato:
    BINANCE_API_KEY="SUA_API_KEY"
    BINANCE_SECRET_KEY="SUA_SECRET_KEY"
    BINANCE_TESTNET=true/false
    BINANCE_API_NAME="SEU_NOME"
    """
    expanded_path = os.path.expanduser(config_path)
    credentials = {}
    
    if not os.path.exists(expanded_path):
        logging.error(f"❌ Erro: Arquivo de configuração da Binance não encontrado em '{expanded_path}'")
        return None

    try:
        with open(expanded_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'): # Ignora linhas vazias e comentários
                    key, value = line.split('=', 1)
                    credentials[key.strip()] = value.strip().strip('"') # Remove aspas
        
        # Validar se as chaves essenciais estão presentes
        if "BINANCE_API_KEY" not in credentials or "BINANCE_SECRET_KEY" not in credentials:
            logging.error("❌ Erro: BINANCE_API_KEY ou BINANCE_SECRET_KEY não encontradas no arquivo de configuração.")
            return None
            
        logging.info(f"✅ Credenciais da Binance carregadas de '{expanded_path}'.")
        return credentials

    except Exception as e:
        logging.error(f"❌ Erro ao ler arquivo de configuração da Binance '{expanded_path}': {str(e)}")
        return None

class BinanceCollector:
    ## LINHA CORRIGIDA: db_path agora tem um valor padrão no diretório do usuário
    def __init__(self, symbol="BTCUSDT", interval="5m", db_path="~/maria_helena_bot/maria_helena.sqlite", 
                 api_key=None, secret_key=None, testnet=False):
        self.symbol = symbol
        self.interval = interval
        ## NOVA LINHA: Expande o caminho do db_path para o diretório real do usuário
        self.db_path = os.path.expanduser(db_path) 
        self.api_url = "https://api.binance.com/api/v3/klines" # Endpoint público para klines
        
        # --- ATRIBUTOS: Chaves da API ---
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet

        if self.api_key and self.secret_key:
            logging.info("API Key e Secret Key carregadas na classe BinanceCollector.")
            if self.testnet:
                logging.info("Modo Testnet ativado.")
        else:
            logging.warning("⚠️ API Key e Secret Key não fornecidas. Operações autenticadas não serão possíveis.")

    ## NOVO MÉTODO: Para criar a tabela com a estrutura completa
    def _create_table_if_not_exists(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maria_helena_candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    openTime INTEGER UNIQUE,
                    closeTime INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    -- Colunas para indicadores (serão preenchidas pelo calculate_indicators.py)
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
            conn.close()
            logging.info(f"✅ Tabela 'maria_helena_candles' verificada/criada em '{self.db_path}'.")
        except Exception as e:
            logging.error(f"❌ Erro ao criar/verificar tabela no DB: {str(e)}")


    def fetch_latest_candle(self):
        """Busca o candle mais recente de 5 min (não requer autenticação)"""
        try:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "limit": 1
            }
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data:
                candle = data[0]
                return {
                    "openTime": int(candle[0]),
                    "closeTime": int(candle[6]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[7])
                }
        except Exception as e:
            logging.error(f"Erro ao buscar candle: {str(e)}")
        
        return None
    
    def fetch_historical_candles(self, limit=200):
        """Busca últimos N candles históricos (não requer autenticação)"""
        try:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "limit": limit
            }
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            
            candles = []
            for candle in response.json():
                candles.append({
                    "openTime": int(candle[0]),
                    "closeTime": int(candle[6]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[7])
                })
            
            return candles
        
        except Exception as e:
            logging.error(f"Erro ao buscar histórico: {str(e)}")
        
        return []
    
    def store_candle(self, candle):
        """Armazena candle no SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # A coluna 'id' será preenchida automaticamente por AUTOINCREMENT
            cursor.execute("""
                INSERT OR IGNORE INTO maria_helena_candles 
                (openTime, closeTime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                candle["openTime"],
                candle["closeTime"],
                candle["open"],
                candle["high"],
                candle["low"],
                candle["close"],
                candle["volume"]
            ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"✅ Candle armazenado: {self.symbol} @ {candle['close']}")
            return True
        
        except Exception as e:
            logging.error(f"❌ Erro ao armazenar candle: {str(e)}")
            return False
    
    def store_historical_candles(self, candles):
        """Armazena múltiplos candles"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for candle in candles:
                # A coluna 'id' será preenchida automaticamente por AUTOINCREMENT
                cursor.execute("""
                    INSERT OR IGNORE INTO maria_helena_candles 
                    (openTime, closeTime, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    candle["openTime"],
                    candle["closeTime"],
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"]
                ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"✅ {len(candles)} candles históricos armazenados")
            return True
        
        except Exception as e:
            logging.error(f"❌ Erro ao armazenar histórico: {str(e)}")
            return False

def main():
    # --- Carregar credenciais antes de inicializar o coletor ---
    binance_creds = load_binance_credentials()
    
    # Definir o caminho do DB aqui para ser usado em ambos os casos
    db_file_path = "~/maria_helena_bot/maria_helena.sqlite"

    if binance_creds:
        api_key = binance_creds.get("BINANCE_API_KEY")
        secret_key = binance_creds.get("BINANCE_SECRET_KEY")
        testnet = binance_creds.get("BINANCE_TESTNET", "false").lower() == "true" # Default para false
        
        ## LINHA CORRIGIDA: Passando o db_path para o construtor
        collector = BinanceCollector(api_key=api_key, secret_key=secret_key, testnet=testnet, 
                                     db_path=db_file_path)
    else:
        logging.warning("⚠️ Não foi possível carregar as credenciais. Iniciando BinanceCollector sem chaves API.")
        ## LINHA CORRIGIDA: Passando o db_path para o construtor
        collector = BinanceCollector(db_path=db_file_path) # Inicia sem chaves se o carregamento falhar
    
    ## NOVA LINHA: Criar a tabela antes de tentar armazenar qualquer dado
    collector._create_table_if_not_exists()

    logging.info("📊 Coletando 200 candles históricos...")
    historical = collector.fetch_historical_candles(limit=200)
    if historical:
        collector.store_historical_candles(historical)
    
    logging.info("📈 Coletando candle mais recente...")
    latest = collector.fetch_latest_candle()
    if latest:
        collector.store_candle(latest)
    
    logging.info("✅ Coleta concluída!")

if __name__ == "__main__":
    main()