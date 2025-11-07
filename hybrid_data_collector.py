#!/usr/bin/env python3
import subprocess
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_script(script_path, description):
    """Executa script e mostra resultado"""
    try:
        logging.info(f"▶️  {description}...")
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            logging.info(f"✅ {description} - OK")
            return True
        else:
            logging.error(f"❌ {description} - ERRO")
            logging.error(result.stderr)
            return False
    
    except Exception as e:
        logging.error(f"❌ Erro ao executar {description}: {str(e)}")
        return False

def main():
    logging.info("=" * 70)
    logging.info("🚀 SISTEMA HÍBRIDO - COLETA DADOS DIÁRIOS + 5MIN")
    logging.info("=" * 70)
    
    scripts = [
        ("/root/maria-helena-scripts/capture_15years_bitcoin.py", "📊 Coleta 15 anos (dados diários históricos)"),
        ("/root/maria-helena-scripts/capture_kraken_5min.py", "📈 Coleta Kraken 5min (tempo real)"),
        ("/root/maria-helena-scripts/calculate_indicators.py", "🔧 Calcula indicadores técnicos"),
    ]
    
    results = []
    for script, desc in scripts:
        result = run_script(script, desc)
        results.append((desc, result))
        logging.info("")
    
    # Resumo
    logging.info("=" * 70)
    logging.info("📋 RESUMO DA COLETA HÍBRIDA")
    logging.info("=" * 70)
    
    for desc, result in results:
        status = "✅" if result else "❌"
        logging.info(f"{status} {desc}")
    
    logging.info("=" * 70)

if __name__ == "__main__":
    main()
