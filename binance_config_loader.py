# ~/maria_helena_bot/binance_config_loader.py

import os
import logging

# Configura o logger para exibir mensagens informativas
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_binance_credentials(config_path="~/.binance_config"):
    """
    Carrega as credenciais da Binance de um arquivo de configuração.

    Args:
        config_path (str): Caminho para o arquivo de configuração.
                           Por padrão, busca em ~/.binance_config.

    Returns:
        dict: Um dicionário contendo 'api_key', 'secret_key', 'testnet' e 'api_name'.
              Retorna None se o arquivo não for encontrado ou houver erro.
    """
    expanded_config_path = os.path.expanduser(config_path)
    credentials = {}

    if not os.path.exists(expanded_config_path):
        logging.error(f"❌ Erro: Arquivo de configuração não encontrado em '{expanded_config_path}'")
        return None

    try:
        with open(expanded_config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"') # Remove aspas duplas
                    
                    if key == "BINANCE_API_KEY":
                        credentials['api_key'] = value
                    elif key == "BINANCE_SECRET_KEY":
                        credentials['secret_key'] = value
                    elif key == "BINANCE_TESTNET":
                        credentials['testnet'] = value.lower() == 'true'
                    elif key == "BINANCE_API_NAME":
                        credentials['api_name'] = value
        
        if not all(k in credentials for k in ['api_key', 'secret_key', 'testnet', 'api_name']):
            logging.error(f"❌ Erro: Credenciais incompletas no arquivo '{expanded_config_path}'. Certifique-se de ter 'BINANCE_API_KEY', 'BINANCE_SECRET_KEY', 'BINANCE_TESTNET' e 'BINANCE_API_NAME'.")
            return None

        logging.info(f"✅ Credenciais da Binance carregadas de '{expanded_config_path}'.")
        # Para segurança, não logamos as chaves completas
        logging.info(f"   API Key: {credentials['api_key'][:5]}...{credentials['api_key'][-5:]}")
        logging.info(f"   Secret Key: {credentials['secret_key'][:5]}...{credentials['secret_key'][-5:]}")
        logging.info(f"   Testnet: {credentials['testnet']}")
        logging.info(f"   API Name: {credentials['api_name']}")
        
        return credentials

    except Exception as e:
        logging.error(f"❌ Erro ao ler o arquivo de configuração '{expanded_config_path}': {e}")
        return None

if __name__ == "__main__":
    # Exemplo de uso
    creds = load_binance_credentials()
    if creds:
        print("\n--- Credenciais Carregadas (APENAS PARA TESTE) ---")
        print(f"API Key: {creds['api_key'][:5]}...{creds['api_key'][-5:]}")
        print(f"Secret Key: {creds['secret_key'][:5]}...{creds['secret_key'][-5:]}")
        print(f"Testnet: {creds['testnet']}")
        print(f"API Name: {creds['api_name']}")
        print("--------------------------------------------------")
        print("Agora você pode usar estas chaves para inicializar o cliente da Binance.")
    else:
        print("Não foi possível carregar as credenciais.")
