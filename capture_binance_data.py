#!/usr/bin/env python3
import requests
import json
import sqlite3
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BinanceCollector:
    def __init__(self, symbol="BTCUSDT", interval="5m", db_path="/root/.n8n/database.sqlite"):
        self.symbol = symbol
        self.interval = interval
        self.db_path = db_path
        self.api_url = "https://api.binance.com/api/v3/klines"
        
    def fetch_latest_candle(self):
        """Busca o candle mais recente de 5 min"""
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
        """Busca últimos N candles históricos"""
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
    collector = BinanceCollector()
    
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
