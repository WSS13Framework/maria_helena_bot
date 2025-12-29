import logging
import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
from typing import List, Dict, Optional

# Configure logging for the dashboard
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("dashboard.log"), # Dashboard's own log file
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Suppress warnings from libraries like TensorFlow/Keras
warnings.filterwarnings('ignore')

# 
# CONFIGURAÇÕES
# 
DB_PATH = os.path.expanduser("~/maria_helena_bot/maria_helena.sqlite")
MODEL_PATH = os.path.expanduser("~/maria_helena_bot/maria_helena_lstm_integrated_model.h5")
REALTIME_UPDATER_LOG_PATH = os.path.expanduser("~/maria_helena_bot/realtime_updater.log") # Path to the updater's log file

FEATURES = ['close', 'high', 'low', 'volume', 'ema_200', 'sma_short', 'sma_long', 
            'rsi_14', 'atr_14', 'bb_upper', 'bb_lower', 'macd', 'macd_signal', 
            'donchian_high', 'donchian_low', 'obv']
TARGET_FEATURE = "close"
LOOKBACK = 60 # Number of past candles to consider for prediction

app = Flask(__name__)

# Carregar modelo na inicialização
logger.info("🚀 Carregando modelo LSTM...")
model = None
try:
    model = load_model(MODEL_PATH, compile=False)
    logger.info("✅ Modelo carregado!")
except Exception as e:
    logger.error(f"❌ Erro ao carregar modelo LSTM de '{MODEL_PATH}': {e}", exc_info=True)
    logger.warning("Predições não estarão disponíveis sem o modelo.")

# 
# FUNÇÕES DE UTILIDADE (para o log)
# 
def get_latest_log_entries(num_lines: int = 15) -> List[str]:
    """
    Lê as últimas 'num_lines' do arquivo de log do realtime_updater.
    Args:
        num_lines: O número de linhas a serem lidas do final do arquivo.
    Returns:
        Uma lista de strings, onde cada string é uma linha do log.
    """
    if not os.path.exists(REALTIME_UPDATER_LOG_PATH):
        logger.warning(f"Arquivo de log do atualizador não encontrado: {REALTIME_UPDATER_LOG_PATH}")
        return [f"Log do atualizador não disponível em {REALTIME_UPDATER_LOG_PATH}"]

    try:
        with open(REALTIME_UPDATER_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-num_lines:]]
    except Exception as e:
        logger.error(f"Erro ao ler o arquivo de log '{REALTIME_UPDATER_LOG_PATH}': {e}", exc_info=True)
        return [f"Erro ao ler log: {e}"]

# 
# FUNÇÕES DE PREDIÇÃO E HISTÓRICO
# 
def get_prediction() -> Optional[Dict]:
    """
    Faz predição do próximo preço usando o modelo LSTM.
    Returns:
        Um dicionário com os dados da predição e indicadores, ou None em caso de erro.
    """
    if model is None:
        logger.warning("Modelo LSTM não carregado. Não é possível fazer predições.")
        return None
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                f"""SELECT {', '.join(FEATURES)}, openTime 
                    FROM maria_helena_candles 
                    ORDER BY openTime DESC 
                    LIMIT {LOOKBACK + 50}""", # Fetch more data to ensure enough for LOOKBACK after dropna
                conn
            )
        
        df = df.iloc[::-1].reset_index(drop=True)
        df_clean = df.dropna(subset=FEATURES)

        if len(df_clean) < LOOKBACK: # CORRIGIDO
            logger.warning(f"Não há dados suficientes ({len(df_clean)} < {LOOKBACK}) no DB para predição.") # CORRIGIDO
            return None
        
        # Preparar dados para a predição
        last_data = df_clean.tail(LOOKBACK)[FEATURES].values
        
        # Escalar os dados. Idealmente, o scaler seria pré-treinado com o mesmo dataset do modelo.
        # Por simplicidade, escalamos com base nos dados mais recentes.
        all_data_for_scaler = df_clean[FEATURES].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(all_data_for_scaler)
        
        last_data_scaled = scaler.transform(last_data)
        X_pred = np.reshape(last_data_scaled, (1, LOOKBACK, len(FEATURES)))
        
        # Fazer predição
        prediction_scaled = model.predict(X_pred, verbose=0)
        
        # Inverter a escala da predição para o valor real
        dummy_features = np.zeros((1, len(FEATURES)))
        close_index = FEATURES.index('close')
        dummy_features[0, close_index] = prediction_scaled[0, 0]
        prediction_unscaled = scaler.inverse_transform(dummy_features)
        predicted_price = prediction_unscaled[0, close_index]
        
        # Dados atuais e cálculo de sinal
        current_price = df_clean[TARGET_FEATURE].iloc[-1]
        change = predicted_price - current_price
        change_pct = (change / current_price) * 100
        
        signal = "HOLD"
        confidence_score = 50.0 # Default para HOLD
        
        if change_pct > 0.5:
            signal = "BUY"
            if change_pct > 2:
                confidence_score = min(95.0, 75.0 + (change_pct - 2) * 5) # Alta confiança
            else:
                confidence_score = min(70.0, 50.0 + (change_pct - 0.5) * 10) # Média confiança
        elif change_pct < -0.5: # CORRIGIDO
            signal = "SELL"
            if change_pct < -2: # CORRIGIDO
                confidence_score = min(95.0, 75.0 + (abs(change_pct) - 2) * 5) # Alta confiança
            else:
                confidence_score = min(70.0, 50.0 + (abs(change_pct) - 0.5) * 10) # Média confiança
        
        confidence_score = round(confidence_score, 0) # Arredonda para inteiro

        insight_message = "Mercado está estável, sem forte viés direcional."
        insight_reasons = []
        
        # Lógica para mensagem de insight e razões
        if signal == "BUY":
            insight_message = "Forte momentum de alta detectado com condições técnicas favoráveis."
            if rsi < 70: # Não sobrecomprado # CORRIGIDO
                insight_reasons.append("RSI indica espaço para movimento de alta.")
            if macd > 0 and macd > df_clean['macd_signal'].iloc[-1]:
                insight_reasons.append("MACD mostra cruzamento de alta e momentum positivo.")
            if current_price > ema_200:
                insight_reasons.append("Preço acima da EMA 200, confirmando tendência de alta.")
            if change_pct > 2:
                insight_reasons.append("Mudança significativa de preço para cima prevista.")
                
        elif signal == "SELL":
            insight_message = "Pressão de baixa aumentando, observar movimento de queda."
            if rsi > 30: # Não sobrevendido
                insight_reasons.append("RSI indica espaço para movimento de baixa.")
            if macd < 0 and macd < df_clean['macd_signal'].iloc[-1]: # Cruzamento de baixa do MACD # CORRIGIDO
                insight_reasons.append("MACD mostra cruzamento de baixa e momentum negativo.")
            if current_price < ema_200: # CORRIGIDO
                insight_reasons.append("Preço abaixo da EMA 200, confirmando tendência de baixa.")
            if change_pct < -2: # CORRIGIDO
                insight_reasons.append("Mudança significativa de preço para baixo prevista.")
        else: # HOLD
            if abs(change_pct) < 0.2: # CORRIGIDO
                insight_message = "Ação de preço consolidando, aguardando sinais mais claros."
            else:
                insight_message = "Sentimento de mercado neutro, potencial para movimento lateral."
            if rsi >= 40 and rsi <= 60: # RSI neutro
                insight_reasons.append("RSI em território neutro, indicando forças equilibradas.")
            if abs(macd) < 0.1: # MACD próximo de zero # CORRIGIDO
                insight_reasons.append("MACD próximo de zero, sugerindo falta de forte momentum.")
            if abs(current_price - ema_200) / current_price < 0.005: # Preço próximo da EMA 200 # CORRIGIDO
                insight_reasons.append("Preço consolidando em torno da EMA 200, indicando indecisão.")


        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'change': float(change),
            'change_pct': float(change_pct),
            'signal': signal,
            'confidence_score': float(confidence_score),
            'rsi': float(rsi),
            'macd': float(macd),
            'ema_200': float(ema_200),
            'trend': 'Up' if current_price > ema_200 else 'Down',
            'candles_count': len(df_clean),
            'insight_message': insight_message,
            'insight_reasons': insight_reasons,
            'last_rsi': float(df_clean['rsi_14'].iloc[-1]),
            'last_macd': float(df_clean['macd'].iloc[-1]),
            'last_macd_signal': float(df_clean['macd_signal'].iloc[-1]), # Adicionado para o frontend
            'last_ema_200': float(df_clean['ema_200'].iloc[-1]),
            'last_atr_14': float(df_clean['atr_14'].iloc[-1]),
            'last_volume': float(df_clean['volume'].iloc[-1]),
            'last_bb_upper': float(df_clean['bb_upper'].iloc[-1]),
            'last_bb_lower': float(df_clean['bb_lower'].iloc[-1]),
            'last_donchian_high': float(df_clean['donchian_high'].iloc[-1]),
            'last_donchian_low': float(df_clean['donchian_low'].iloc[-1]),
            'last_obv': float(df_clean['obv'].iloc[-1]),
            'last_sma_short': float(df_clean['sma_short'].iloc[-1]),
            'last_sma_long': float(df_clean['sma_long'].iloc[-1]),
        }
    except sqlite3.Error as e:
        logger.error(f"Erro de banco de dados na predição: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Erro inesperado na predição: {e}", exc_info=True)
        return None

def get_historical_prices() -> Dict[str, List]:
    """
    Retorna o histórico de preços de fechamento para exibição em gráfico.
    Returns:
        Um dicionário com listas de timestamps e preços, ou listas vazias em caso de erro.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                f"""SELECT close, openTime 
                    FROM maria_helena_candles 
                    ORDER BY openTime DESC 
                    LIMIT 100""",
                conn
            )
        
        df = df.iloc[::-1].reset_index(drop=True)
        
        return {
            'timestamps': [datetime.fromtimestamp(t/1000).strftime('%H:%M') 
                          for t in df['openTime'].tolist()],
            'prices': df['close'].tolist()
        }
    except sqlite3.Error as e:
        logger.error(f"Erro de banco de dados ao buscar histórico: {e}", exc_info=True)
        return {'timestamps': [], 'prices': []}
    except Exception as e:
        logger.error(f"Erro inesperado ao buscar histórico: {e}", exc_info=True)
        return {'timestamps': [], 'prices': []}

# 
# ROTAS DO FLASK
# 
@app.route('/')
def index() -> str:
    """Rota principal que renderiza o dashboard HTML."""
    return render_template('dashboard.html')

@app.route('/api/prediction')
def api_prediction() -> jsonify:
    """API para dados de predição em tempo real."""
    prediction = get_prediction()
    if prediction:
        return jsonify(prediction)
    logger.warning("Falha ao gerar predição para a API.")
    return jsonify({'error': 'Erro ao gerar predição'}), 500

@app.route('/api/history')
def api_history() -> jsonify:
    """API para histórico de preços para o gráfico."""
    history = get_historical_prices()
    return jsonify(history)

@app.route('/api/log')
def api_log() -> jsonify:
    """API para as últimas entradas do log do atualizador em tempo real."""
    log_entries = get_latest_log_entries(num_lines=15) # Ajuste o número de linhas conforme necessário
    return jsonify(log_entries)

# 
# CRIAR TEMPLATE HTML (dashboard.html)
# 
def create_template() -> None:
    """Cria o arquivo HTML do dashboard na pasta 'templates'."""
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maria Helena</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --color-background: #f5f5f7;
            --color-card-background: #ffffff;
            --color-text-primary: #1d1d1f;
            --color-text-secondary: #86868b;
            --color-accent-blue: #007aff;
            --color-green: #34c759;
            --color-red: #ff3b30;
            --color-neutral: #f5f5f7;
            --color-log-background: #2c2c2e;
            --color-log-text: #f5f5f7;
            --font-family-primary: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            --shadow-light: 0 4px 20px rgba(0,0,0,0.04);
            --shadow-medium: 0 6px 30px rgba(0,0,0,0.08);
            --border-radius-card: 18px;
            --transition-ease-apple: all 0.2s cubic-bezier(0.25, 0.1, 0.25, 1);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --color-background: #1c1c1e;
                --color-card-background: #2c2c2e;
                --color-text-primary: #f5f5f7;
                --color-text-secondary: #98989d;
                --color-neutral: #3a3a3c;
                --color-log-background: #1c1c1e;
                --color-log-text: #e0e0e0;
            }
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: var(--font-family-primary);
            background: var(--color-background);
            color: var(--color-text-primary);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            transition: background var(--transition-ease-apple), color var(--transition-ease-apple);
        }
        .container { max-width: 980px; margin: 0 auto; padding: 30px 20px; }
        header { text-align: center; margin-bottom: 40px; position: relative; }
        h1 { font-size: 42px; font-weight: 600; letter-spacing: -0.5px; margin-bottom: 10px; }
        .subtitle { font-size: 18px; color: var(--color-text-secondary); font-weight: 400; }
        .update-time { font-size: 12px; color: var(--color-text-secondary); margin-top: 6px; }

        /* Hero Section */
        .hero-section {
            background: var(--color-card-background);
            border-radius: var(--border-radius-card);
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: var(--shadow-light);
            transition: background var(--transition-ease-apple), box-shadow var(--transition-ease-apple);
        }
        .signal-main {
            font-size: 64px;
            font-weight: 700;
            letter-spacing: -2px;
            margin: 15px 0;
            display: block;
            animation: pulse 1.5s infinite alternate;
        }
        .signal-buy { color: var(--color-green); }
        .signal-sell { color: var(--color-red); }
        .signal-hold { color: var(--color-text-secondary); animation: none; }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            100% { transform: scale(1.02); opacity: 0.9; }
        }

        .confidence-badge {
            display: inline-block;
            font-size: 16px;
            font-weight: 500;
            padding: 6px 12px;
            border-radius: 980px;
            background: var(--color-neutral);
            color: var(--color-text-primary);
            margin-top: 10px;
            transition: background var(--transition-ease-apple), color var(--transition-ease-apple);
        }

        .insight-message {
            font-size: 20px;
            font-weight: 500;
            color: var(--color-text-primary);
            margin-top: 20px;
            line-height: 1.4;
        }
        .insight-reasons {
            font-size: 14px;
            color: var(--color-text-secondary);
            margin-top: 10px;
            list-style: none;
            padding: 0;
        }
        .insight-reasons li {
            margin-bottom: 4px;
        }

        .price-info {
            display: flex;
            justify-content: center;
            align-items: baseline;
            margin-top: 20px;
            gap: 20px;
        }
        .price-item {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .price-label {
            font-size: 14px;
            color: var(--color-text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
            margin-bottom: 4px;
        }
        .price-value {
            font-size: 28px;
            font-weight: 600;
            color: var(--color-text-primary);
        }
        .price-change {
            font-size: 18px;
            font-weight: 500;
            margin-top: 5px;
        }
        .change-positive { color: var(--color-green); }
        .change-negative { color: var(--color-red); }
        .change-neutral { color: var(--color-text-secondary); }

        .cta-button {
            background: var(--color-accent-blue);
            color: white;
            padding: 12px 24px;
            border-radius: 980px;
            text-decoration: none;
            font-weight: 600;
            font-size: 16px;
            margin-top: 30px;
            display: inline-block;
            transition: background var(--transition-ease-apple), transform 0.1s ease-out;
            border: none;
            cursor: pointer;
        }
        .cta-button:hover {
            background: #005bb5;
            transform: translateY(-1px);
        }
        .cta-button:active {
            transform: translateY(0);
        }

        /* Details Section (Initially Hidden) */
        .details-section {
            display: none; /* Controlled by JS */
            opacity: 0;
            transform: translateY(20px);
            transition: opacity 0.5s ease-out, transform 0.5s ease-out;
        }
        .details-section.active {
            display: block;
            opacity: 1;
            transform: translateY(0);
        }
        .details-section .chart-title {
             margin-top: 40px;
        }

        .indicators-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 30px;
        }
        .indicator-card {
            background: var(--color-card-background);
            border-radius: var(--border-radius-card);
            padding: 25px 20px;
            text-align: center;
            box-shadow: var(--shadow-light);
            transition: background var(--transition-ease-apple), box-shadow var(--transition-ease-apple);
        }
        .indicator-label { font-size: 14px; color: var(--color-text-secondary); margin-bottom: 10px; font-weight: 500; }
        .indicator-value { font-size: 38px; font-weight: 600; letter-spacing: -1px; color: var(--color-text-primary); }
        .indicator-status { font-size: 13px; color: var(--color-text-secondary); margin-top: 6px; }

        /* Chart Section */
        .chart-section {
            background: var(--color-card-background);
            border-radius: var(--border-radius-card);
            padding: 30px;
            box-shadow: var(--shadow-light);
            margin-bottom: 30px;
            transition: background var(--transition-ease-apple), box-shadow var(--transition-ease-apple);
        }
        .chart-title { font-size: 20px; font-weight: 600; margin-bottom: 20px; color: var(--color-text-primary); }

        /* Log Container Styles */
        .log-container {
            background: var(--color-log-background);
            color: var(--color-log-text);
            border-radius: var(--border-radius-card);
            padding: 15px;
            margin-top: 30px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
            font-size: 11px;
            max-height: 200px;
            overflow-y: auto;
            box-shadow: var(--shadow-medium);
            transition: background var(--transition-ease-apple), color var(--transition-ease-apple), box-shadow var(--transition-ease-apple);
        }
        .log-entry {
            white-space: pre-wrap;
            word-break: break-all;
            margin-bottom: 3px;
            line-height: 1.3;
        }

        /* Google Translate Widget Styles */
        #google_translate_element {
            position: absolute;
            top: 15px;
            right: 15px;
            z-index: 1000;
            font-size: 13px;
        }
        #google_translate_element .goog-te-gadget {
            font-family: var(--font-family-primary);
            color: var(--color-text-primary);
            white-space: nowrap;
        }
        #google_translate_element .goog-te-gadget-simple {
            background-color: var(--color-card-background);
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 6px 10px;
            line-height: 1.2;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: var(--transition-ease-apple);
        }
        #google_translate_element .goog-te-gadget-simple:hover {
            background-color: var(--color-neutral);
            border-color: #d0d0d0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        }
        #google_translate_element .goog-te-gadget-simple .goog-te-menu-value {
            color: var(--color-text-primary);
            font-weight: 500;
        }
        #google_translate_element .goog-te-gadget-simple .goog-te-menu-value span {
            color: var(--color-text-primary);
        }
        #google_translate_element .goog-te-gadget-simple .goog-te-menu-value img {
            display: none;
        }
        .goog-te-banner-frame.skiptranslate {
            display: none !important;
        }
        body {
            top: 0px !important;
        }

        /* Footer and External Insights */
        footer { text-align: center; padding: 30px 0; color: var(--color-text-secondary); font-size: 13px; }
        .external-insights {
            margin-top: 15px;
            font-size: 12px;
        }
        .external-insights a {
            color: var(--color-accent-blue);
            text-decoration: none;
            font-weight: 500;
            transition: color var(--transition-ease-apple);
        }
        .external-insights a:hover {
            color: #005bb5;
            text-decoration: underline;
        }

        /* Media Queries for smaller screens */
        @media (max-width: 768px) {
            .container { padding: 20px 15px; }
            h1 { font-size: 32px; }
            .subtitle { font-size: 16px; }
            .signal-main { font-size: 48px; }
            .insight-message { font-size: 16px; }
            .price-info { flex-direction: column; gap: 10px; }
            .price-value { font-size: 22px; }
            .price-change { font-size: 16px; }
            .cta-button { padding: 10px 20px; font-size: 14px; }
            .indicators-grid { grid-template-columns: 1fr; gap: 10px; }
            .indicator-card { padding: 20px 15px; }
            .indicator-value { font-size: 32px; }
            .chart-section { padding: 20px; }
            .log-container { max-height: 150px; font-size: 10px; padding: 10px; }
            #google_translate_element { top: 10px; right: 10px; font-size: 12px; }
            #google_translate_element .goog-te-gadget-simple { padding: 5px 8px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Maria Helena</h1>
            <p class="subtitle">Bitcoin Price Prediction</p>
            <p class="update-time">Last update: <span id="lastUpdate">--</span></p>
            <div id="google_translate_element"></div>
        </header>

        <section class="hero-section">
            <span id="signalMain" class="signal-main">--</span>
            <span id="confidenceBadge" class="confidence-badge">Confidence: --%</span>
            <p id="insightMessage" class="insight-message">--</p>
            <ul id="insightReasons" class="insight-reasons"></ul>

            <div class="price-info">
                <div class="price-item">
                    <span class="price-label">Current Price</span>
                    <span id="currentPrice" class="price-value">$--</span>
                </div>
                <div class="price-item">
                    <span class="price-label">Predicted Price</span>
                    <span id="predictedPrice" class="price-value">$--</span>
                    <span id="priceChange" class="price-change">--</span>
                </div>
            </div>
            <button id="ctaButton" class="cta-button">View Full Analysis</button>
        </section>

        <section id="detailsSection" class="details-section">
            <div class="chart-section">
                <div class="chart-title">Price History</div>
                <canvas id="priceChart"></canvas>
            </div>

            <div class="indicators-grid">
                <div class="indicator-card">
                    <div class="indicator-label">RSI</div>
                    <div class="indicator-value" id="rsiValue">--</div>
                    <div class="indicator-status" id="rsiStatus">--</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-label">MACD</div>
                    <div class="indicator-value" id="macdValue">--</div>
                    <div class="indicator-status" id="macdStatus">--</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-label">EMA 200</div>
                    <div class="indicator-value" id="ema200Value">--</div>
                    <div class="indicator-status" id="ema200Status">--</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-label">ATR 14</div>
                    <div class="indicator-value" id="atrValue">--</div>
                    <div class="indicator-status">Average True Range</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-label">Volume</div>
                    <div class="indicator-value" id="volumeValue">--</div>
                    <div class="indicator-status">Trading Activity</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-label">Bollinger Bands</div>
                    <div class="indicator-value" id="bbValue">--</div>
                    <div class="indicator-status" id="bbStatus">--</div>
                </div>
            </div>
        </section>

        <section class="log-container">
            <div class="chart-title">Realtime Updater Log</div>
            <div id="realtimeLog">
                <div class="log-entry">Loading log...</div>
            </div>
        </section>

        <footer>
            <p>Developed by Marcos Sea (WSS13Framework)</p>
            <p>wss13.framework@gmail.com</p>
            <div class="external-insights">
                <a href="https://trends.google.com/trends/explore?q=Bitcoin" target="_blank" rel="noopener noreferrer">Google Trends: Bitcoin Search Interest</a>
            </div>
        </footer>
    </div>

    <script>
        let priceChart = null;
        let detailsSectionVisible = false;

        function initChart() {
            const ctx = document.getElementById('priceChart').getContext('2d');
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'BTC/USDT',
                        data: [],
                        borderColor: 'var(--color-accent-blue)',
                        backgroundColor: 'rgba(0, 122, 255, 0.05)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            padding: 12,
                            titleFont: { size: 14, weight: '600' },
                            bodyFont: { size: 13 },
                            cornerRadius: 8
                        }
                    },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: 'var(--color-text-secondary)', font: { size: 12 } } },
                        y: { grid: { color: 'rgba(0, 0, 0, 0.05)', drawBorder: false }, ticks: { color: 'var(--color-text-secondary)', font: { size: 12 } } }
                    }
                }
            });
        }

        async function updateData() {
            try {
                const predResponse = await fetch('/api/prediction');
                const data = await predResponse.json();

                if (data.error) {
                    console.error('API Prediction Error:', data.error);
                    // Update main signal to indicate error
                    document.getElementById('signalMain').textContent = 'ERROR';
                    document.getElementById('signalMain').className = 'signal-main signal-hold';
                    document.getElementById('confidenceBadge').textContent = 'Confidence: --%';
                    document.getElementById('insightMessage').textContent = 'Could not fetch prediction data.';
                    document.getElementById('insightReasons').innerHTML = '';
                    return;
                }

                // Update Header
                document.getElementById('lastUpdate').textContent = data.timestamp;

                // Update Hero Section
                const signalMainEl = document.getElementById('signalMain');
                signalMainEl.textContent = data.signal;
                signalMainEl.className = 'signal-main'; // Reset class
                if (data.signal === 'BUY') {
                    signalMainEl.classList.add('signal-buy');
                } else if (data.signal === 'SELL') {
                    signalMainEl.classList.add('signal-sell');
                } else {
                    signalMainEl.classList.add('signal-hold');
                }

                document.getElementById('confidenceBadge').textContent = `Confidence: ${data.confidence_score}%`;
                document.getElementById('insightMessage').textContent = data.insight_message;
                const insightReasonsEl = document.getElementById('insightReasons');
                insightReasonsEl.innerHTML = '';
                data.insight_reasons.forEach(reason => {
                    const li = document.createElement('li');
                    li.textContent = reason;
                    insightReasonsEl.appendChild(li);
                });

                document.getElementById('currentPrice').textContent = '$' + data.current_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                document.getElementById('predictedPrice').textContent = '$' + data.predicted_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});

                const changeEl = document.getElementById('priceChange');
                const changeClass = data.change > 0 ? 'change-positive' : data.change < 0 ? 'change-negative' : 'change-neutral';
                changeEl.className = 'price-change ' + changeClass;
                changeEl.textContent = (data.change > 0 ? '↗ +' : data.change < 0 ? '↘ ' : '') + '$' + Math.abs(data.change).toFixed(2) + ' (' + (data.change_pct > 0 ? '+' : '') + data.change_pct.toFixed(2) + '%)';

                // Update Details Section Indicators
                document.getElementById('rsiValue').textContent = data.last_rsi.toFixed(0);
                let rsiStatus = '';
                if (data.last_rsi > 70) rsiStatus = 'Overbought';
                else if (data.last_rsi < 30) rsiStatus = 'Oversold';
                else rsiStatus = 'Normal';
                document.getElementById('rsiStatus').textContent = rsiStatus;

                document.getElementById('macdValue').textContent = data.last_macd.toFixed(2);
                let macdStatus = '';
                if (data.last_macd > data.last_macd_signal) macdStatus = 'Bullish Cross';
                else if (data.last_macd < data.last_macd_signal) macdStatus = 'Bearish Cross';
                else macdStatus = 'Neutral';
                document.getElementById('macdStatus').textContent = macdStatus;


                document.getElementById('ema200Value').textContent = '$' + data.last_ema_200.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
                document.getElementById('ema200Status').textContent = data.trend === 'Up' ? 'Above Price' : 'Below Price';

                document.getElementById('atrValue').textContent = data.last_atr_14.toFixed(2);
                document.getElementById('volumeValue').textContent = data.last_volume.toLocaleString('en-US', {maximumFractionDigits: 0});

                // Bollinger Bands Status
                let bbStatus = 'Neutral';
                if (data.current_price > data.last_bb_upper) bbStatus = 'Overbought';
                else if (data.current_price < data.last_bb_lower) bbStatus = 'Oversold';
                document.getElementById('bbValue').textContent = `Upper: $${data.last_bb_upper.toFixed(0)} | Lower: $${data.last_bb_lower.toFixed(0)}`;
                document.getElementById('bbStatus').textContent = bbStatus;


                // Update Chart
                const histResponse = await fetch('/api/history');
                const histData = await histResponse.json();
                if (priceChart && histData.timestamps && histData.prices) {
                    priceChart.data.labels = histData.timestamps;
                    priceChart.data.datasets[0].data = histData.prices;
                    priceChart.update('none');
                }

                // Vibrate on signal change (if supported)
                if (navigator.vibrate && signalMainEl.dataset.lastSignal && data.signal !== signalMainEl.dataset.lastSignal) {
                    navigator.vibrate(50); // Short vibration
                }
                signalMainEl.dataset.lastSignal = data.signal; // Store last signal

            } catch (error) {
                console.error('Error fetching prediction or history data:', error);
                document.getElementById('signalMain').textContent = 'OFFLINE';
                document.getElementById('signalMain').className = 'signal-main signal-hold';
                document.getElementById('confidenceBadge').textContent = 'Confidence: --%';
                document.getElementById('insightMessage').textContent = 'Failed to load data. Check console for errors.';
                document.getElementById('insightReasons').innerHTML = '';
            }
        }

        async function updateLog() {
            try {
                const logResponse = await fetch('/api/log');
                const logData = await logResponse.json();
                const logContainer = document.getElementById('realtimeLog');
                logContainer.innerHTML = '';
                logData.forEach(line => {
                    const p = document.createElement('div');
                    p.className = 'log-entry';
                    p.textContent = line;
                    logContainer.appendChild(p);
                });
                logContainer.scrollTop = logContainer.scrollHeight;
            } catch (error) {
                console.error('Error fetching log:', error);
                document.getElementById('realtimeLog').innerHTML = '<div class="log-entry" style="color: var(--color-red);">Error loading log.</div>';
            }
        }

        function googleTranslateElementInit() {
            new google.translate.TranslateElement({
                pageLanguage: 'en',
                includedLanguages: 'en,pt,es,fr,de',
                layout: google.translate.TranslateElement.InlineLayout.SIMPLE,
                autoDisplay: false
            }, 'google_translate_element');
        }

        function toggleDetails() {
            const detailsSection = document.getElementById('detailsSection');
            const ctaButton = document.getElementById('ctaButton');

            if (detailsSectionVisible) {
                detailsSection.classList.remove('active');
                // Use setTimeout to allow transition to complete before setting display: none
                setTimeout(() => { detailsSection.style.display = 'none'; }, 500); // Match transition duration
                ctaButton.textContent = 'View Full Analysis';
            } else {
                detailsSection.style.display = 'block'; // Show before transition
                setTimeout(() => detailsSection.classList.add('active'), 10); // Trigger transition
                ctaButton.textContent = 'Hide Details';
                // Scroll to details section
                detailsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            detailsSectionVisible = !detailsSectionVisible;
        }

        document.addEventListener('DOMContentLoaded', function() {
            initChart();
            updateData();
            updateLog();
            setInterval(updateData, 30000);
            setInterval(updateLog, 5000);

            const script = document.createElement('script');
            script.type = 'text/javascript';
            script.src = '//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
            document.body.appendChild(script);

            document.getElementById('ctaButton').addEventListener('click', toggleDetails);
        });
    </script>
</body>
</html>'''
    
    with open(os.path.join(template_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info("✅ Template HTML 'dashboard.html' criado na pasta 'templates'!")

# 
# MAIN
# 
if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("🚀 MARIA HELENA TRADING BOT - DASHBOARD WEB")
    logger.info("=" * 70)
    
    create_template()
    
    logger.info("\n🌐 Iniciando servidor web...")
    logger.info("📍 Acesse: http://localhost:5000")
    logger.info("⏹️  Pressione Ctrl+C para parar")
    logger.info("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)