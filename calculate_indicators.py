#!/usr/bin/env python3
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

class IndicatorCalculator:
    def __init__(self, db_path="/root/.n8n/database.sqlite"):
        self.db_path = db_path
    
    def calculate_ema(self, data, period=200):
        """Calcula EMA (Exponential Moving Average)"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, data, period=50):
        """Calcula SMA (Simple Moving Average)"""
        return data.rolling(window=period).mean()
    
    def calculate_rsi(self, data, period=14):
        """Calcula RSI (Relative Strength Index)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Calcula Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calcula MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line
    
    def calculate_atr(self, high, low, close, period=14):
        """Calcula ATR (Average True Range)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_obv(self, close, volume):
        """Calcula OBV (On Balance Volume)"""
        obv = volume.copy()
        obv[close < close.shift(1)] *= -1
        return obv.cumsum()
    
    def calculate_donchian_channels(self, high, low, period=20):
        """Calcula Donchian Channels"""
        donchian_high = high.rolling(window=period).max()
        donchian_low = low.rolling(window=period).min()
        return donchian_high, donchian_low
    
    def update_indicators(self):
        """Atualiza todos os indicadores no banco"""
        try:
            # Ler dados do banco
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                "SELECT id, close, high, low, volume FROM maria_helena_candles ORDER BY openTime ASC",
                conn
            )
            
            if len(df) < 60:
                logging.warning(f"⚠️ Apenas {len(df)} candles. Precisa de 60+ pra calcular indicadores.")
                conn.close()
                return False
            
            logging.info(f"📊 Calculando indicadores para {len(df)} candles...")
            
            # Calcular indicadores
            df['ema_200'] = self.calculate_ema(df['close'], period=200)
            df['sma_short'] = self.calculate_sma(df['close'], period=20)
            df['sma_long'] = self.calculate_sma(df['close'], period=50)
            df['rsi_14'] = self.calculate_rsi(df['close'], period=14)
            df['atr_14'] = self.calculate_atr(df['high'], df['low'], df['close'], period=14)
            
            # Bollinger Bands
            bb_upper, bb_lower = self.calculate_bollinger_bands(df['close'], period=20)
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            
            # MACD
            macd_line, signal_line = self.calculate_macd(df['close'], fast=12, slow=26, signal=9)
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            
            # Donchian Channels
            donchian_high, donchian_low = self.calculate_donchian_channels(df['high'], df['low'], period=20)
            df['donchian_high'] = donchian_high
            df['donchian_low'] = donchian_low
            
            # OBV
            df['obv'] = self.calculate_obv(df['close'], df['volume'])
            
            # Atualizar banco
            cursor = conn.cursor()
            
            for idx, row in df.iterrows():
                cursor.execute("""
                    UPDATE maria_helena_candles
                    SET 
                        ema_200 = ?,
                        sma_short = ?,
                        sma_long = ?,
                        rsi_14 = ?,
                        atr_14 = ?,
                        bb_upper = ?,
                        bb_lower = ?,
                        macd = ?,
                        macd_signal = ?,
                        donchian_high = ?,
                        donchian_low = ?,
                        obv = ?
                    WHERE id = ?
                """, (
                    row['ema_200'],
                    row['sma_short'],
                    row['sma_long'],
                    row['rsi_14'],
                    row['atr_14'],
                    row['bb_upper'],
                    row['bb_lower'],
                    row['macd'],
                    row['macd_signal'],
                    row['donchian_high'],
                    row['donchian_low'],
                    row['obv'],
                    row['id']
                ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"✅ {len(df)} candles atualizados com indicadores!")
            return True
        
        except Exception as e:
            logging.error(f"❌ Erro ao calcular indicadores: {str(e)}")
            return False

def main():
    calc = IndicatorCalculator()
    calc.update_indicators()

if __name__ == "__main__":
    main()
