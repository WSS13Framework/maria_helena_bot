#!/bin/bash

echo "🚀 Preparando para push GitHub..."
echo "=================================="

cd /root/maria-helena-scripts

# Configurar git
git config user.email "wss13.framework@gmail.com"
git config user.name "Marcos Sea - WSS13Framework"

# Adicionar arquivos
git add *.py *.json *.csv *.ipynb *.md .gitignore

# Commit
git commit -m "🚀 Maria Helena Trading Bot v1.0 - Sistema Híbrido Completo

✅ Coleta de dados: Kraken (5min + Daily)
✅ Indicadores: EMA200, RSI14, MACD, ATR, Bollinger
✅ Cron: 11 tasks automáticas
✅ N8N: 4 workflows prontos
✅ LSTM: Notebook Colab pronto
✅ Dataset: 721 candles históricos

Desenvolvedor: Marcos Sea (WSS13Framework)
Email: wss13.framework@gmail.com"

# Push
echo ""
echo "📤 Push para GitHub..."
git push origin main

echo ""
echo "✅ Concluído!"
