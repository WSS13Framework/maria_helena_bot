import pandas as pd
import sqlite3
import os
from datetime import datetime

# ============================================================
# CONFIGURAÇÕES
# ============================================================
DB_PATH = "~/maria_helena_bot/maria_helena.sqlite"
RAW_CSV_FOLDER = "~/Área de Trabalho/dados-csv" # Caminho para a pasta com seus CSVs brutos

# ============================================================
# FUNÇÃO DE IMPORTAÇÃO
# ============================================================
def unify_and_import_raw_data(csv_folder, db_path):
    db_full_path = os.path.expanduser(db_path)
    csv_full_folder = os.path.expanduser(csv_folder)

    print(f"Caminho expandido para a pasta de CSVs: {csv_full_folder}")

    if not os.path.exists(csv_full_folder):
        print(f"❌ Erro: Pasta de CSVs não encontrada em {csv_full_folder}")
        return

    conn = None
    try:
        conn = sqlite3.connect(db_full_path)
        cursor = conn.cursor()

        # Cria a tabela se não existir
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maria_helena_candles (
                openTime INTEGER PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        """)
        conn.commit()
        print(f"✅ Tabela 'maria_helena_candles' verificada/criada em {db_full_path}")

        total_imported_rows = 0
        
        for filename in os.listdir(csv_full_folder):
            csv_file_path = os.path.join(csv_full_folder, filename)

            # === Lógica para IGNORAR arquivos que não são OHLCV brutos ===
            if not filename.endswith(".csv"):
                print(f"⚠️ Aviso: Arquivo '{filename}' não é um CSV. Pulando.")
                continue
            if "with_signals" in filename:
                print(f"⚠️ Aviso: Arquivo '{filename}' contém dados com sinais (não OHLCV bruto). Pulando.")
                continue
            if filename.startswith('ticks_BTC-USDT_15m'):
                print(f"⚠️ Aviso: Arquivo '{filename}' contém dados processados (não OHLCV bruto). Pulando.")
                continue
            if filename == 'training.csv':
                print(f"⚠️ Aviso: Arquivo '{filename}' contém dados de texto/label (não OHLCV bruto). Pulando.")
                continue
            if filename.startswith('.~lock.'): # Ignora arquivos de lock do LibreOffice/outros
                print(f"⚠️ Aviso: Arquivo '{filename}' é um arquivo de lock. Pulando.")
                continue

            print(f"\nProcessando arquivo: {filename}")

            try:
                df_standardized = pd.DataFrame() # DataFrame para os dados padronizados

                # === Lógica Específica para bitcoin_training_data.csv (sem cabeçalho, leitura manual) ===
                if filename == 'bitcoin_training_data.csv':
                    print("   Detectado 'bitcoin_training_data.csv'. Lendo manualmente e mapeando por índice.")
                    
                    data = []
                    with open(csv_file_path, 'r') as f:
                        for line in f:
                            # Remove espaços em branco extras e divide a linha por qualquer espaço
                            # Filtra strings vazias resultantes de múltiplos espaços
                            parts = [p.strip() for p in line.strip().split()]
                            if len(parts) >= 6: # Garante que há colunas suficientes
                                data.append(parts)
                    
                    if not data:
                        print(f"⚠️ Aviso: 'bitcoin_training_data.csv' não contém dados válidos ou suficientes. Pulando este arquivo.")
                        continue

                    df = pd.DataFrame(data)

                    # Verifica se há colunas suficientes para OHLCV
                    if df.shape[1] < 6:
                        print(f"⚠️ Aviso: 'bitcoin_training_data.csv' não tem colunas suficientes para OHLCV (esperado 6+, encontrado {df.shape[1]}). Pulando este arquivo.")
                        continue

                    # Mapeia as colunas pela posição (índice)
                    # Converte para tipo numérico, forçando erros para NaN
                    df_standardized = df.iloc[:, [0, 1, 2, 3, 4, 5]].copy()
                    df_standardized.columns = ['openTime', 'open', 'high', 'low', 'close', 'volume']
                    df_standardized['open'] = pd.to_numeric(df_standardized['open'], errors='coerce')
                    df_standardized['high'] = pd.to_numeric(df_standardized['high'], errors='coerce')
                    df_standardized['low'] = pd.to_numeric(df_standardized['low'], errors='coerce')
                    df_standardized['close'] = pd.to_numeric(df_standardized['close'], errors='coerce')
                    df_standardized['volume'] = pd.to_numeric(df_standardized['volume'], errors='coerce')

                else:
                    # Se houver outros arquivos CSV brutos OHLCV no futuro, a lógica para eles viria aqui.
                    # Por enquanto, qualquer outro CSV que não seja 'bitcoin_training_data.csv' e não foi ignorado, será avisado.
                    print(f"⚠️ Aviso: Arquivo '{filename}' não é 'bitcoin_training_data.csv' e não foi explicitamente ignorado. Verifique se é um CSV de dados brutos OHLCV. Pulando por segurança.")
                    continue


                # === Tratamento de openTime (para todos os arquivos padronizados) ===
                # Garante que openTime seja um timestamp em milissegundos (INTEGER)
                if 'openTime' in df_standardized.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_standardized['openTime']):
                        df_standardized['openTime'] = (df_standardized['openTime'].astype(int) / 10**6).astype(int)
                    elif pd.api.types.is_string_dtype(df_standardized['openTime']):
                        df_standardized['openTime'] = pd.to_datetime(df_standardized['openTime'], errors='coerce')
                        df_standardized.dropna(subset=['openTime'], inplace=True)
                        df_standardized['openTime'] = (df_standardized['openTime'].astype(int) / 10**6).astype(int)
                    elif pd.api.types.is_numeric_dtype(df_standardized['openTime']):
                        df_standardized['openTime'] = df_standardized['openTime'].astype(int)
                    else:
                        print(f"⚠️ Aviso: Tipo de dado inesperado para 'openTime' no arquivo {filename}. Tentando converter para int.")
                        df_standardized['openTime'] = pd.to_numeric(df_standardized['openTime'], errors='coerce').astype(int)
                
                df_standardized.dropna(subset=['openTime'], inplace=True)

                # Remove duplicatas baseadas em openTime para evitar erros no UNIQUE PRIMARY KEY
                initial_rows = len(df_standardized)
                df_standardized.drop_duplicates(subset=['openTime'], inplace=True)
                if len(df_standardized) < initial_rows:
                    print(f"   Foram removidas {initial_rows - len(df_standardized)} duplicatas de 'openTime' no arquivo {filename}.")

                # === Inserção no Banco de Dados ===
                rows_before = pd.read_sql_query("SELECT COUNT(*) FROM maria_helena_candles", conn).iloc[0, 0]
                
                try:
                    df_standardized.to_sql('maria_helena_candles', conn, if_exists='append', index=False)
                    rows_after = pd.read_sql_query("SELECT COUNT(*) FROM maria_helena_candles", conn).iloc[0, 0]
                    imported_from_file = rows_after - rows_before
                    total_imported_rows += imported_from_file

                    print(f"✅ {imported_from_file} novas linhas importadas de {filename} para o SQLite.")
                except sqlite3.IntegrityError as ie:
                    print(f"❌ Erro de integridade ao importar {filename}: {ie}. Provavelmente, chaves primárias duplicadas que não foram removidas.")
                except Exception as insert_e:
                    print(f"❌ Erro na inserção de {filename} no SQLite: {insert_e}")

            except Exception as e:
                print(f"❌ Erro ao processar/importar {filename}: {e}")
        
        print(f"\n⭐ Importação de todos os CSVs brutos concluída. Total de {total_imported_rows} novas linhas adicionadas ao DB.")

    except sqlite3.Error as e:
        print(f"❌ Erro no SQLite: {e}")
    except Exception as e:
        print(f"❌ Erro geral: {e}")
    finally:
        if conn:
            conn.close()

# ============================================================
# EXECUÇÃO DA UNIFICAÇÃO E IMPORTAÇÃO
# ============================================================
print("=" * 70)
print("🚀 INICIANDO UNIFICAÇÃO E IMPORTAÇÃO DE DADOS BRUTOS PARA SQLITE")
print("=" * 70)
print(f"⏰ Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

unify_and_import_raw_data(RAW_CSV_FOLDER, DB_PATH)

print("\n" + "=" * 70)
print("✅ PROCESSO DE UNIFICAÇÃO E IMPORTAÇÃO CONCLUÍDO.")
print("   Seu terreno está arrumado e a fundação está pronta!")
print("=" * 70)