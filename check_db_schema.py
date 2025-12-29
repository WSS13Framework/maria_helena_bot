import sqlite3
import os

DB_PATH = os.path.expanduser("~/maria_helena_bot/maria_helena.sqlite")

def check_table_schema(db_path: str, table_name: str):
    """Conecta ao banco de dados e imprime o esquema da tabela."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Consulta para obter o esquema da tabela
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema = cursor.fetchall()

        if not schema:
            print(f"❌ Tabela '{table_name}' não encontrada no banco de dados '{db_path}'.")
            return

        print(f"✅ Esquema da tabela '{table_name}' no '{db_path}':")
        print("-" * 50)
        print(f"{'CID':<5} {'Name':<20} {'Type':<10} {'Not Null':<10} {'PK':<5}")
        print("-" * 50)
        for col in schema:
            cid, name, col_type, not_null, default_value, pk = col
            print(f"{cid:<5} {name:<20} {col_type:<10} {bool(not_null):<10} {bool(pk):<5}")
        print("-" * 50)

    except sqlite3.Error as e:
        print(f"❌ Erro ao acessar o banco de dados: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("🚀 Verificando esquema do banco de dados Maria Helena...")
    check_table_schema(DB_PATH, "maria_helena_candles")
    print("\n💡 Compare as colunas acima com a lista de FEATURES no seu dashboard.py.")


