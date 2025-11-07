#!/bin/bash

SCRIPT_DIR="/root/maria-helena-scripts"
PYTHON_ENV="/root/maria-helena-env/bin/python3"
LOG_FILE="/root/maria-helena-scripts/collection.log"

source /root/maria-helena-env/bin/activate

# Rodar coleta
$PYTHON_ENV $SCRIPT_DIR/capture_real_data.py >> $LOG_FILE 2>&1

# Health check
$PYTHON_ENV $SCRIPT_DIR/health_check.py >> $LOG_FILE 2>&1

echo "[$(date)] ✅ Coleta concluída" >> $LOG_FILE
