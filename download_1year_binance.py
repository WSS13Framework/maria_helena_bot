#!/usr/bin/env python3
"""
Download de 1 ano de dados BTCUSDT da Binance
Intervalo: 15 minutos
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import sqlite3
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

SYMBOL = 'BTCUSDT'
INTERVAL = '15m'
DB_PATH = 'maria_helena.sqlite'
DAYS = 365

logging.info("=" * 70)
logging.info("📥 DOWNLOAD HISTÓRICO BINANCE")
logging.info("=" * 70)
logging.info(f"Par: {SYMBOL}")
logging.info(f"Intervalo: {INTERVAL}")
logging.info(f"Período: {DAYS} dias")

# Calcular timestamps
end_time = int(datetime.now().timestamp() * 1000)
start_time = int((datetime.now() - timedelta(days=DAYS)).timestamp() * 1000)

logging.info(f"De: {datetime.fromtimestamp(start_time/1000)}")
logging.info(f"Até: {datetime.fromtimestamp(end_time/1000)}")

# Download
all_candles = []
current_start = start_time
batch = 0
max_batches = 40  # ~40,000 candles (1 ano)

logging.info("\n📥 Iniciando download...")

while current_start < end_time and batch < max_batches:
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': SYMBOL,
        'interval': INTERVAL,
        'startTime': current_start,
        'limit': 1000
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            logging.error(f"HTTP Error: {response.status_code}")
            time.sleep(5)
            continue
        
        data = response.json()
        
        if not data or not isinstance(data, list):
            logging.warning("Resposta vazia ou inválida")
            break
        
        all_candles.extend(data)
        current_start = data[-1][0] + 1
        batch += 1
        
        logging.info(f"Batch {batch}/{max_batches}: {len(all_candles):,} candles")
        time.sleep(0.5)  # Rate limit
        
    except Exception as e:
        logging.error(f"Erro: {e}")
        time.sleep(5)

logging.info(f"\n✅ Download completo: {len(all_candles):,} candles")

if len(all_candles) == 0:
    logging.error("Nenhum dado baixado!")
    exit(1)

# Converter para DataFrame
logging.info("🔧 Processando dados...")

df = pd.DataFrame(all_candles, columns=[
    'openTime', 'open', 'high', 'low', 'close', 'volume',
    'closeTime', 'quoteVolume', 'trades', 'takerBuyBase', 'takerBuyQuote', 'ignore'
])

# Converter tipos
for col in ['openTime', 'closeTime']:
    df[col] = df[col].astype(int)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

# Selecionar colunas
df = df[['openTime', 'open', 'high', 'low', 'close', 'volume']]

# Remover duplicatas e ordenar
df = df.drop_duplicates(subset=['openTime']).sort_values('openTime').reset_index(drop=True)

logging.info(f"Após limpeza: {len(df):,} candles")
logging.info(f"Período: {datetime.fromtimestamp(df['openTime'].min()/1000)} → {datetime.fromtimestamp(df['openTime'].max()/1000)}")

# Salvar no banco (SUBSTITUIR dados antigos)
logging.info("\n💾 Salvando no banco...")

conn = sqlite3.connect(DB_PATH)
df.to_sql('maria_helena_candles', conn, if_exists='replace', index=False)

# Verificar
count = pd.read_sql_query("SELECT COUNT(*) as total FROM maria_helena_candles", conn)
conn.close()

logging.info(f"✅ {count['total'][0]:,} registros salvos em {DB_PATH}")

logging.info("\n" + "=" * 70)
logging.info("✅ SUCESSO!")
logging.info("=" * 70)
logging.info("\nPRÓXIMOS PASSOS:")
logging.info("1. python3 calculate_indicators.py")
logging.info("2. Fazer upload do banco para Google Drive")
logging.info("3. Treinar modelos no Colab")
