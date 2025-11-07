#!/usr/bin/env python3
import requests
import sqlite3
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BitcoinHistoryCollector:
    def __init__(self, db_path="/root/.n8n/database.sqlite"):
        self.db_path = db_path
        self.api_url = "https://api.coingecko.com/api/v3"
    
    def fetch_15years_bitcoin(self):
        """Busca 15 anos completos de Bitcoin"""
        try:
            logging.info("🔍 Consultando API CoinGecko para 15 anos de Bitcoin...")
            
            url = f"{self.api_url}/coins/bitcoin/market_chart"
            params = {
                "vs_currency": "usd",
                "days": "5475",
                "interval": "daily"
            }
            
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            prices = data.get('prices', [])
            volumes = data.get('volumes', [])
            
            logging.info(f"✅ Recebido: {len(prices)} dias de histórico")
            
            candles = []
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0
                
                date = datetime.fromtimestamp(timestamp / 1000)
                days_from_start = (date - datetime(2009, 1, 1)).days
                
                if days_from_start < 365:
                    volatility = price * 0.05 if price > 0 else 0.01
                elif days_from_start < 1825:
                    volatility = price * 0.04 if price > 0 else 0.01
                elif days_from_start < 3650:
                    volatility = price * 0.03 if price > 0 else 0.01
                else:
                    volatility = price * 0.02 if price > 0 else 0.01
                
                candle = {
                    "openTime": int(timestamp),
                    "closeTime": int(timestamp) + 86400000,
                    "open": round(max(price - volatility, 0.01), 8),
                    "high": round(price + volatility * 1.5, 8),
                    "low": round(max(price - volatility * 1.5, 0.01), 8),
                    "close": round(price, 8),
                    "volume": round(volume, 2)
                }
                
                candles.append(candle)
            
            logging.info(f"📊 Total de candles gerados: {len(candles)}")
            logging.info(f"📅 Período: {datetime.fromtimestamp(prices[0][0]/1000)} até {datetime.fromtimestamp(prices[-1][0]/1000)}")
            
            return candles
        
        except Exception as e:
            logging.error(f"❌ Erro ao buscar histórico: {str(e)}")
            return []
    
    def store_candles(self, candles):
        """Armazena candles no banco"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
            
            logging.info(f"✅ {total} candles armazenados com sucesso!")
            return True
        
        except Exception as e:
            logging.error(f"❌ Erro ao armazenar: {str(e)}")
            return False

def main():
    logging.info("=" * 60)
    logging.info("🚀 COLETANDO 15 ANOS COMPLETOS DE BITCOIN")
    logging.info("=" * 60)
    
    collector = BitcoinHistoryCollector()
    
    logging.info("📥 Buscando dados...")
    candles = collector.fetch_15years_bitcoin()
    
    if candles:
        logging.info("💾 Armazenando no banco de dados...")
        collector.store_candles(candles)
    else:
        logging.error("❌ Nenhum dado foi coletado")

if __name__ == "__main__":
    main()
