import re
import os
import pandas as pd # Adicionado para garantir que pd esteja disponível para o generated script
import numpy as np # Adicionado para garantir que np esteja disponível para o generated script
from sklearn.preprocessing import MinMaxScaler # Adicionado para garantir que MinMaxScaler esteja disponível para o generated script

def generate_integrated_lstm_script(original_script_path, output_script_path):
    """
    Modifica o script original do LSTM para integrar o carregamento de dados
    enriquecidos do SQLite e o treinamento com múltiplas features (indicadores).
    """
    print(f"Iniciando integração de '{original_script_path}'...")
    
    with open(original_script_path, 'r') as f:
        content = f.read()

    # --- Configurações que serão injetadas no script ---
    DB_PATH = "~/maria_helena_bot/maria_helena.sqlite" # Ajustado para o novo caminho do DB
    # Lista completa de features a serem usadas pelo LSTM
    FEATURES = [
        'close', 'high', 'low', 'volume', 'ema_200', 'sma_short', 'sma_long',
        'rsi_14', 'atr_14', 'bb_upper', 'bb_lower', 'macd', 'macd_signal',
        'donchian_high', 'donchian_low', 'obv'
    ]
    TARGET_FEATURE = 'close' # A feature que o modelo vai prever

    if TARGET_FEATURE not in FEATURES:
        raise ValueError(f"TARGET_FEATURE '{TARGET_FEATURE}' deve estar presente na lista FEATURES.")

    # --- 1. Adicionar importação do sqlite3 e os ---
    print("  - Adicionando imports 'sqlite3' e 'os'...")
    # Adiciona sqlite3 após warnings
    content = re.sub(r'(import warnings)', r'import sqlite3\n\1', content, count=1)
    # Adiciona os após numpy
    content = re.sub(r'(import numpy as np)', r'import os\n\1', content, count=1)


    # --- 2. Inserir definições de DB_PATH, FEATURES e TARGET_FEATURE ---
    print("  - Inserindo configurações de DB_PATH e FEATURES...")
    insert_config = f"""
DB_PATH = "{DB_PATH}"
FEATURES = {FEATURES}
TARGET_FEATURE = "{TARGET_FEATURE}"
"""
    # Insere as configurações após a importação de warnings
    content = re.sub(r'(warnings.filterwarnings\(\'ignore\'\))', r'\1' + insert_config, content, count=1)

    # --- 3. Modificar CÉLULA 2: CARREGAR DADOS para usar SQLite ---
    print("  - Modificando CÉLULA 2: Carregamento de dados para SQLite...")
    # Regex mais robusta para pegar a CÉLULA 2 completa
    # Captura o cabeçalho e então tudo que vem depois até o próximo cabeçalho de CÉLULA ou fim do arquivo
    data_loading_block_pattern = re.compile(
        r'(# =+\n# CÉLULA 2: CARREGAR DADOS\n# =+\n).*?(?=\n# =+\n# CÉLULA 3: PREPARAR DADOS\n# =+|$)',
        re.DOTALL
    )

    new_data_loading_block_content = f"""print(\"\"\"\\n2️⃣ CARREGANDO DADOS ENRIQUECIDOS DO SQLITE...\"\"\")

try:
    conn = sqlite3.connect(os.path.expanduser(DB_PATH)) # Expande o caminho do DB
    # Seleciona todas as FEATURES e openTime para ordenação
    df = pd.read_sql_query(
        f"SELECT {{', '.join(FEATURES)}}, openTime FROM maria_helena_candles ORDER BY openTime ASC",
        conn
    )
    conn.close()
    
    # Remove linhas com NaN que surgem dos cálculos dos indicadores (primeiras N linhas)
    # Isso é crucial para garantir que todas as features tenham valores válidos
    df_clean = df.dropna(subset=FEATURES)
    
    if df_clean.empty:
        raise ValueError("DataFrame vazio após remover NaNs. Verifique os dados e o período dos indicadores.")
    
    print(f"✅ {{len(df_clean)}} candles enriquecidos carregados do SQLite!")
    print(f"   Primeiras linhas:")
    print(df_clean.head(3))

except Exception as e:
    print(f"❌ Erro ao carregar do SQLite: {{str(e)}}")
    print("Certifique-se de que 'calculate_indicators.py' foi executado e o DB existe.")
    print("Tentando carregar do GitHub (apenas 'close' sem indicadores, para fallback)...")
    url = "https://raw.githubusercontent.com/WSS13Framework/maria_helena_bot/main/bitcoin_training_data.csv"
    df_fallback = pd.read_csv(url)
    df_clean = df_fallback.dropna(subset=[TARGET_FEATURE]) # Fallback apenas com close
    print(f"⚠️ {{len(df_clean)}} candles carregados do GitHub (apenas 'close'). Execute 'calculate_indicators.py' primeiro para ter todos os indicadores.")
    print(f"   Primeiras linhas:")
    print(df_clean.head(3))

print(f\"\"\"\\n📊 Dataset Final:\"\"\")
print(f"   Total de candles processados: {{len(df_clean)}}")
"""
    # Substitui o conteúdo da CÉLULA 2, mantendo o cabeçalho
    content = data_loading_block_pattern.sub(r'\1' + new_data_loading_block_content, content, count=1)


    # --- 4. Modificar CÉLULA 3: PREPARAR DADOS para multi-features ---
    print("  - Modificando CÉLULA 3: Preparação de dados para multi-features...")
    data_prep_pattern = re.compile(
        r'(# =+\n# CÉLULA 3: PREPARAR DADOS\n# =+\n).*?(?=\n# =+\n# CÉLULA 4: CRIAR MODELO LSTM\n# =+|$)',
        re.DOTALL
    )

    new_data_prep_block_content = f"""print(\"\"\"\\n3️⃣ PREPARANDO DADOS COM MÚLTIPLAS FEATURES...\"\"\")

# Selecionar as features para o treinamento
data_features = df_clean[FEATURES].values

print(f"📊 Dados originais das features:")
for i, feature_name in enumerate(FEATURES):
    print(f"   {{feature_name}}: Min={{data_features[:, i].min():,.2f}}, Max={{data_features[:, i].max():,.2f}}")

# Normalizar todas as features entre 0 e 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_features = scaler.fit_transform(data_features)

print(f"✅ Todas as {{len(FEATURES)}} features normalizadas!")

# O target (y_train) ainda será o preço de fechamento do próximo candle
target_close_prices = df_clean[TARGET_FEATURE].values.reshape(-1, 1)
# Usar um scaler separado para o target para facilitar a inversão posterior
target_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_target_close = target_scaler.fit_transform(target_close_prices)

# Criar sequences (lookback dias → próximo dia)
lookback = 60
X_train = []
y_train = []

for i in range(lookback, len(scaled_data_features)):
    X_train.append(scaled_data_features[i-lookback:i, :]) # Todas as features para o lookback
    y_train.append(scaled_target_close[i, 0]) # Apenas o TARGET_FEATURE como target

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape para LSTM [samples, timesteps, features]
X_train = np.reshape(X_train, (X_train.shape[0], lookback, len(FEATURES)))

print(f\"\"\"\\n📈 Sequências criadas com {{len(FEATURES)}} features:\"\"\")
print(f"   X_train shape: {{X_train.shape}} (amostras, lookback, features)")
print(f"   y_train shape: {{y_train.shape}} (targets)")
"""
    content = data_prep_pattern.sub(r'\1' + new_data_prep_block_content, content, count=1)


    # --- 5. Modificar CÉLULA 4: CRIAR MODELO LSTM - input_shape e return_sequences ---
    print("  - Ajustando input_shape e return_sequences do modelo LSTM...")
    # Primeiro, ajusta o input_shape
    content = re.sub(r'input_shape=\(lookback, 1\)', r'input_shape=(lookback, len(FEATURES))', content, count=1)
    # Depois, adiciona return_sequences=True para a primeira camada LSTM que alimenta outra LSTM
    # Isso assume que a primeira LSTM é 'LSTM_1' e que ela é seguida por outra LSTM.
    content = re.sub(
        r"LSTM\(50, activation='relu', input_shape=\(lookback, len\(FEATURES\)\), name='LSTM_1'\),",
        r"LSTM(50, activation='relu', input_shape=(lookback, len(FEATURES)), return_sequences=True, name='LSTM_1'),",
        content,
        count=1
    )


    # --- 6. Modificar CÉLULA 7: FAZER PREDIÇÕES NO TREINO - inverse_transform para y_train_actual ---
    print("  - Ajustando inverse_transform para y_train_actual...")
    content = re.sub(r'y_train_actual = scaler\.inverse_transform\(y_train\.reshape\(-1, 1\)\)', r'y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1))', content, count=1)

    # --- 7. Modificar CÉLULA 8: PREVER PRÓXIMO PREÇO para multi-features ---
    print("  - Modificando CÉLULA 8: Predição do próximo preço para multi-features...")
    prediction_block_pattern = re.compile(
        r'(# =+\n# CÉLULA 8: PREVER PRÓXIMO PREÇO\n# =+\n).*?(?=\n# =+\n# CÉLULA 9: |$)', # Match até o próximo cabeçalho ou fim do arquivo
        re.DOTALL
    )

    new_prediction_block_content = f"""print(\"\"\"\\n8️⃣ PREVENDO PRÓXIMO PREÇO COM MULTI-FEATURES...\"\"\")

# Para prever o próximo preço, precisamos dos últimos 'lookback' candles com TODAS as features.
# df_clean já está ordenado por openTime.
# Vamos recarregar os últimos dados enriquecidos para garantir consistência.
try:
    conn_pred = sqlite3.connect(os.path.expanduser(DB_PATH))
    df_pred = pd.read_sql_query(
        f"SELECT {{', '.join(FEATURES)}}, openTime FROM maria_helena_candles ORDER BY openTime DESC LIMIT {{lookback}}",
        conn_pred
    )
    conn_pred.close()
    df_pred = df_pred.iloc[::-1] # Inverter para ordem cronológica ascendente
    
    if len(df_pred) &lt; lookback:
        raise ValueError(f"Não há dados suficientes no DB para prever. Necessário {{lookback}} candles, encontrado {{len(df_pred)}}.")

    last_lookback_features_values = df_pred[FEATURES].values

    # Normalizar as últimas features usando o scaler treinado para as features
    last_lookback_scaled_features = scaler.transform(last_lookback_features_values)
    
    # Reshape para o formato esperado pelo LSTM [1, lookback, num_features]
    X_test_multi_feature = np.array([last_lookback_scaled_features])
    X_test_multi_feature = np.reshape(X_test_multi_feature, (1, lookback, len(FEATURES)))

    # Fazer a predição
    next_price_scaled_target = model.predict(X_test_multi_feature, verbose=0)
    
    # Para inverter a escala do preço predito, usamos o target_scaler
    next_price_actual = target_scaler.inverse_transform(next_price_scaled_target)[0][0]
    
    current_price = df_pred[TARGET_FEATURE].iloc[-1] # Último preço de fechamento do df_pred
    change = next_price_actual - current_price
    change_pct = (change / current_price) * 100

except Exception as e:
    print(f"❌ Erro na predição do próximo preço: {{str(e)}}")
    next_price_actual = None
    current_price = None
    change = None
    change_pct = None

# 
# CÉLULA 9: SALVAR MODELO E PRÓXIMOS PASSOS
# 
print(\"\"\"\\n9️⃣ SALVAR MODELO E PRÓXIMOS PASSOS...\"\"\")

try:
    model.save('maria_helena_lstm_integrated_model.h5')
    file_size = os.path.getsize('maria_helena_lstm_integrated_model.h5')
    print(f"✅ Modelo salvo com sucesso!")
    print(f"   Arquivo: maria_helena_lstm_integrated_model.h5")
    print(f"   Tamanho: {{file_size / (1024*1024):.2f}} MB")

    # Verifica se está no ambiente Colab para download
    try:
        if 'google.colab' in str(get_ipython()):
            from google.colab import files
            files.download('maria_helena_lstm_integrated_model.h5')
            print("✅ Download do modelo iniciado (ambiente Colab).")
    except NameError: # get_ipython não definido em ambiente local
        pass 
    
    print("⚠️ Modo local - Arquivo salvo: maria_helena_lstm_integrated_model.h5")
    print(\"\"\"\\n⚙️ Próximos Passos:\"\"\")
    print("1. Salvar: maria_helena_lstm_integrated_model.h5")
    print("2. Fazer upload para o servidor:")
    print("   scp maria_helena_lstm_integrated_model.h5 root@server:/root/maria-helena-scripts/")
    print("3. Usar este modelo para predições em tempo real ou futuras análises.")

except Exception as e:
    print(f"❌ Erro ao salvar o modelo: {{str(e)}}")
"""
    # Substitui o conteúdo da CÉLULA 8 e adiciona a CÉLULA 9
    content = prediction_block_pattern.sub(r'\1' + new_prediction_block_content, content, count=1)

    # Remove as CÉLULAS 9 e 10 originais, que agora estão integradas na nova CÉLULA 9
    # Regex para remover CÉLULA 9 e CÉLULA 10 originais
    content = re.sub(
        r'\n# =+\n# CÉLULA 9: SALVAR MODELO\n# =+.*?print\("✅ Modelo salvo com sucesso!"\)',
        '',
        content,
        flags=re.DOTALL
    )
    content = re.sub(
        r'\n# =+\n# CÉLULA 10: DOWNLOAD\n# =+.*?print\("⚠️ Modo local - Arquivo salvo: maria_helena_lstm_model.h5"\)',
        '',
        content,
        flags=re.DOTALL
    )
    # Remove os "Próximos Passos" e "Desenvolvedor" finais duplicados
    content = re.sub(
        r'\nprint\("\\n" \+ "=" \* 70\)\nprint\("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!"\)\nprint\("=" \* 70\)\n\nprint\("\\n🚀 PRÓXIMOS PASSOS:"\)\nprint\("1\. Salvar: maria_helena_lstm_model\.h5"\)\nprint\("2\. Upload pro servidor:"\)\nprint\("   scp maria_helena_lstm_model\.h5 root@server:\/root\/maria-helena-scripts\/"\)\nprint\("3\. Rodar predições:"\)\nprint\("   python3 \/root\/maria-helena-scripts\/run_lstm_predictions\.py"\)\n\nprint\("\\n📞 DESENVOLVEDOR:"\)\nprint\("   Marcos Sea \(WSS13Framework\)"\)\nprint\("   Email: wss13\.framework@gmail\.com"\)\nprint\("   GitHub: github\.com\/WSS13Framework\/maria_helena_bot"\)',
        '',
        content,
        flags=re.DOTALL
    )


    with open(output_script_path, 'w') as f:
        f.write(content)

    print(f"\n✅ Script '{original_script_path}' modificado e salvo como '{output_script_path}' com sucesso!")
    print("Por favor, revise o novo script e teste-o cuidadosamente.")
    print("Certifique-se de que o 'calculate_indicators.py' foi executado e o banco de dados SQLite está populado com os indicadores.")

if __name__ == "__main__":
    original_script = "maria_helena_lstm_final.py"
    output_script = "maria_helena_lstm_integrated.py"
    
    # Verifica se o script original existe
    if not os.path.exists(original_script):
        print(f"❌ Erro: O arquivo '{original_script}' não foi encontrado no diretório atual.")
        print("Certifique-se de que você está no diretório '~/maria_helena_bot/' ou forneça o caminho correto.")
    else:
        generate_integrated_lstm_script(original_script, output_script)