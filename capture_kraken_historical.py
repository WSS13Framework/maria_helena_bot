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

class KrakenHistoricalCollector:
    def __init__(self, db_path="/root/.n8n/database.sqlite"):
        self.db_path = db_path
        self.api_url = "https://api.kraken.com/0/public"
        self.symbol = "XXBTZUSD"
    
    def fetch_historical_daily(self, days=5475):
        """Busca histórico diário completo"""
        try:
            logging.info(f"📊 Buscando ~{days} dias de histórico diário...")
            
            url = f"{self.api_url}/OHLC"
            since = int((datetime.now() - timedelta(days=days)).timestamp())
            
            params = {
                "pair": self.symbol,
                "interval": 1440,  # 1 dia em minutos
                "since": since
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('error'):
                logging.error(f"❌ Erro Kraken: {data['error']}")
                return []
            
            ohlc_data = data.get('result', {}).get(self.symbol, [])
            
            logging.info(f"✅ {len(ohlc_data)} candles diários recebidos")
            
            candles = []
            for candle in ohlc_data:
                candles.append({
                    "openTime": int(candle[0] * 1000),
                    "closeTime": int(candle[0] * 1000) + 86400000,
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[6])
                })
            
            return candles
        
        except Exception as e:
            logging.error(f"❌ Erro ao buscar histórico diário: {str(e)}")
            return []
    
    def store_candles(self, candles):
        """Armazena candles"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Limpar dados antigos
            cursor.execute("DELETE FROM maria_helena_candles")
            logging.info("🗑️ Banco limpo")
            
            for candle in candles:
                cursor.execute("""
                    INSERT INTO maria_helena_candles 
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
            
            cursor.execute("SELECT COUNT(*) FROM maria_helena_candles")
            total = cursor.fetchone()[0]
            
            conn.close()
            
            logging.info(f"✅ {total} candles armazenados!")
            return True
        
        except Exception as e:
            logging.error(f"❌ Erro ao armazenar: {str(e)}")
            return False

def main():
    logging.info("=" * 60)
    logging.info("🚀 COLETANDO HISTÓRICO KRAKEN (DAILY)")
    logging.info("=" * 60)
    
    collector = KrakenHistoricalCollector()
    
    candles = collector.fetch_historical_daily(days=5475)
    if candles:
        collector.store_candles(candles)
    else:
        logging.error("❌ Nenhum dado foi coletado")
    
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
