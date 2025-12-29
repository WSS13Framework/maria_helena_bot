import os
import configparser

def load_binance_credentials(config_path=os.path.expanduser('~/.binance_config')):
    """
    Carrega as credenciais da Binance de um arquivo de configuração.

    Args:
        config_path (str): Caminho para o arquivo de configuração.

    Returns:
        tuple: (api_key, secret_key, testnet, api_name)
               Retorna None para chaves se não forem encontradas ou houver erro.
    """
    api_key = None
    secret_key = None
    testnet = False
    api_name = None

    if not os.path.exists(config_path):
        print(f"❌ Erro: Arquivo de configuração não encontrado em '{config_path}'.")
        print("Certifique-se de que suas chaves da Binance estão configuradas corretamente.")
        return None, None, None, None

    # O formato do seu arquivo é como um arquivo .env ou um arquivo de texto simples
    # Vamos ler linha por linha para extrair as variáveis
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"') # Remove as aspas se existirem

                    if key == "BINANCE_API_KEY":
                        api_key = value
                    elif key == "BINANCE_SECRET_KEY":
                        secret_key = value
                    elif key == "BINANCE_TESTNET":
                        testnet = (value.lower() == 'true')
                    elif key == "BINANCE_API_NAME":
                        api_name = value
        
        if not api_key or not secret_key:
            print(f"⚠️ Aviso: API Key ou Secret Key não encontradas em '{config_path}'.")
            return None, None, None, None

        print(f"✅ Credenciais da Binance carregadas de '{config_path}'.")
        return api_key, secret_key, testnet, api_name

    except Exception as e:
        print(f"❌ Erro ao ler o arquivo de configuração '{config_path}': {e}")
        return None, None, None, None

if __name__ == "__main__":
    # Exemplo de uso:
    api_key, secret_key, testnet, api_name = load_binance_credentials()

    if api_key and secret_key:
        print("\n--- Credenciais Carregadas (APENAS PARA TESTE) ---")
        print(f"API Key: {api_key[:5]}...{api_key[-5:]}") # Mostra apenas o início e fim por segurança
        print(f"Secret Key: {secret_key[:5]}...{secret_key[-5:]}") # Mostra apenas o início e fim por segurança
        print(f"Testnet: {testnet}")
        print(f"API Name: {api_name}")
        print("--------------------------------------------------")
        print("Agora você pode usar estas chaves para inicializar o cliente da Binance.")
    else:
        print("\nNão foi possível carregar as credenciais da Binance.")
