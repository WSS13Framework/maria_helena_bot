#!/usr/bin/env python3
import requests
import sqlite3
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class KrakenCollector:
    def __init__(self, db_path="/root/.n8n/database.sqlite"):
        self.db_path = db_path
        # Kraken API pública (sem autenticação)
        self.api_url = "https://api.kraken.com/0/public"
        self.symbol = "XXBTZUSD"  # Bitcoin em USD
    
    def fetch_ohlc_5min(self):
        """Busca OHLC de 5 minutos em tempo real"""
        try:
            url = f"{self.api_url}/OHLC"
            params = {
                "pair": self.symbol,
                "interval": 5  # 5 minutos
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('error'):
                logging.error(f"❌ Erro Kraken: {data['error']}")
                return None
            
            # Kraken retorna dados em formato específico
            ohlc_data = data.get('result', {}).get(self.symbol, [])
            
            if not ohlc_data:
                logging.warning("⚠️ Nenhum dado OHLC recebido")
                return None
            
            # Último candle de 5 min
            latest = ohlc_data[-1]
            
            candle = {
                "openTime": int(latest[0] * 1000),
                "closeTime": int(latest[0] * 1000) + 300000,  # 5 min em ms
                "open": float(latest[1]),
                "high": float(latest[2]),
                "low": float(latest[3]),
                "close": float(latest[4]),
                "volume": float(latest[6])
            }
            
            logging.info(f"✅ Candle 5min recebido: BTC @ ${candle['close']:.2f}")
            return candle
        
        except Exception as e:
            logging.error(f"❌ Erro ao buscar OHLC 5min: {str(e)}")
            return None
    
    def fetch_historical_5min(self, limit=288):
        """Busca últimos N candles de 5 min (288 = 1 dia)"""
        try:
            url = f"{self.api_url}/OHLC"
            params = {
                "pair": self.symbol,
                "interval": 5,
                "since": int((datetime.now() - timedelta(hours=24)).timestamp())
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            ohlc_data = data.get('result', {}).get(self.symbol, [])
            
            candles = []
            for candle in ohlc_data[-limit:]:
                candles.append({
                    "openTime": int(candle[0] * 1000),
                    "closeTime": int(candle[0] * 1000) + 300000,
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[6])
                })
            
            logging.info(f"✅ {len(candles)} candles 5min históricos recebidos")
            return candles
        
        except Exception as e:
            logging.error(f"❌ Erro ao buscar histórico 5min: {str(e)}")
            return []
    
    def store_5min_candle(self, candle):
        """Armazena candle 5min em tabela separada"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Cria tabela se não existir
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maria_helena_candles_5min (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    openTime INTEGER UNIQUE,
                    closeTime INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                INSERT OR IGNORE INTO maria_helena_candles_5min 
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
            
            return True
        
        except Exception as e:
            logging.error(f"❌ Erro ao armazenar candle 5min: {str(e)}")
            return False
    
    def store_multiple_5min(self, candles):
        """Armazena múltiplos candles 5min"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Cria tabela se não existir
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maria_helena_candles_5min (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    openTime INTEGER UNIQUE,
                    closeTime INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            for candle in candles:
                cursor.execute("""
                    INSERT OR IGNORE INTO maria_helena_candles_5min 
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
            
            logging.info(f"✅ {len(candles)} candles 5min armazenados")
            return True
        
        except Exception as e:
            logging.error(f"❌ Erro ao armazenar múltiplos: {str(e)}")
            return False

def main():
    collector = KrakenCollector()
    
    logging.info("=" * 60)
    logging.info("🚀 COLETANDO DADOS 5MIN KRAKEN (TEMPO REAL)")
    logging.info("=" * 60)
    
    # Histórico 5min (últimas 24h)
    logging.info("📊 Buscando últimas 24h de candles 5min...")
    historical = collector.fetch_historical_5min(limit=288)
    if historical:
        collector.store_multiple_5min(historical)
    
    # Candle atual
    logging.info("📈 Buscando candle 5min atual...")
    latest = collector.fetch_ohlc_5min()
    if latest:
        collector.store_5min_candle(latest)
    
    logging.info("=" * 60)
    logging.info("✅ Coleta 5min concluída!")

if __name__ == "__main__":
    main()
