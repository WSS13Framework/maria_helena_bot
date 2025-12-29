import pandas as pd
import os
import re
from datetime import datetime

# ============================================================
# CONFIGURAÇÕES
# ============================================================
RAW_CSV_FOLDER = "~/Área de Trabalho/dados-csv"
ORIGINAL_FILENAME = "bitcoin_training_data.csv"
CLEANED_FILENAME = "bitcoin_training_data_cleaned.csv"

# ============================================================
# FUNÇÃO DE LIMPEZA E SALVAMENTO
# ============================================================
def clean_and_save_bitcoin_data(raw_folder, original_filename, cleaned_filename):
    raw_full_path = os.path.expanduser(raw_folder)
    original_file_path = os.path.join(raw_full_path, original_filename)
    cleaned_file_path = os.path.join(raw_full_path, cleaned_filename)

    print(f"Caminho do arquivo original: {original_file_path}")
    print(f"Caminho do arquivo limpo: {cleaned_file_path}")

    if not os.path.exists(original_file_path):
        print(f"❌ Erro: Arquivo original '{original_file_path}' não encontrado.")
        return

    print(f"\n🚀 Iniciando limpeza do arquivo: {original_filename}")

    data = []
    try:
        with open(original_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: # Pular linhas vazias
                    continue
                
                # Usar re.split para dividir por um ou mais espaços em branco/tabulações
                # e remover aspas duplas do timestamp
                parts = [p.strip().replace('"', '') for p in re.split(r'\s+', line) if p.strip()]
                
                if len(parts) >= 6: # Garante que há colunas suficientes para OHLCV
                    # Junta a data e a hora se estiverem separadas
                    timestamp_str = ""
                    remaining_parts = []

                    # Tenta detectar o padrão 'YYYY-MM-DD HH:MM:SS'
                    if len(parts) >= 2 and re.match(r'\d{4}-\d{2}-\d{2}', parts[0]) and re.match(r'\d{2}:\d{2}:\d{2}', parts[1]):
                        timestamp_str = f"{parts[0]} {parts[1]}"
                        remaining_parts = parts[2:]
                    else: # Assume que a primeira parte já é o timestamp completo
                        timestamp_str = parts[0]
                        remaining_parts = parts[1:]

                    # Adiciona o timestamp limpo e as outras 5 colunas (open, high, low, close, volume)
                    if len(remaining_parts) >= 5:
                        data.append([timestamp_str] + remaining_parts[:5])
                    else:
                        print(f"DEBUG: Linha '{line}' pulada: Partes insuficientes após processamento de OHLCV ({len(remaining_parts)}).")
                else:
                    print(f"DEBUG: Linha '{line}' pulada: Partes iniciais insuficientes ({len(parts)}).")
        
        if not data:
            print(f"⚠️ Aviso: '{original_filename}' não contém dados válidos ou suficientes após parsing. Nenhum dado para salvar.")
            return

        df = pd.DataFrame(data)
        df.columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
        
        # Converte para tipo numérico, forçando erros para NaN
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # Converte openTime para datetime e depois para timestamp em milissegundos (INTEGER)
        df['openTime'] = pd.to_datetime(df['openTime'], errors='coerce')
        df.dropna(subset=['openTime', 'open', 'high', 'low', 'close', 'volume'], inplace=True)
        df['openTime'] = (df['openTime'].astype(int) / 10**6).astype(int)

        if df.empty:
            print(f"⚠️ Aviso: DataFrame vazio após limpeza e conversão de tipos. Nenhum dado para salvar.")
            return

        # Salva o DataFrame limpo em um novo CSV com cabeçalho e separador de vírgula
        df.to_csv(cleaned_file_path, index=False)
        print(f"✅ {len(df)} linhas limpas e salvas em '{cleaned_file_path}'.")

    except Exception as e:
        print(f"❌ Erro ao processar/salvar '{original_filename}': {e}")

# ============================================================
# EXECUÇÃO DA LIMPEZA
# ============================================================
print("=" * 70)
print("🧹 INICIANDO LIMPEZA DE DADOS BRUTOS")
print("=" * 70)
print(f"⏰ Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

clean_and_save_bitcoin_data(RAW_CSV_FOLDER, ORIGINAL_FILENAME, CLEANED_FILENAME)

print("\n" + "=" * 70)
print("✅ PROCESSO DE LIMPEZA CONCLUÍDO.")
print("   Arquivo limpo pronto para importação!")
print("=" * 70)
