#!/bin/bash

DB_PATH="/root/.n8n/database.sqlite"
PYTHON_ENV="/root/maria-helena-env/bin/python3"
SCRIPT_DIR="/root/maria-helena-scripts"

source /root/maria-helena-env/bin/activate

echo "🚀 Iniciando coleta de dados..."
$PYTHON_ENV $SCRIPT_DIR/capture_binance_data.py

echo ""
echo "🏥 Executando health check..."
$PYTHON_ENV $SCRIPT_DIR/health_check.py

echo ""
echo "✅ Coleta concluída!"
