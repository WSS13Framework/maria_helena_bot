#!/usr/bin/env python3
import sqlite3
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

class HealthCheck:
    def __init__(self, db_path="/root/.n8n/database.sqlite"):
        self.db_path = db_path
    
    def check_database(self):
        """Verifica se banco está OK"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM maria_helena_candles")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT MAX(timestamp) FROM maria_helena_candles")
            last_update = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "status": "✅ OK",
                "total_candles": total,
                "last_update": last_update
            }
        
        except Exception as e:
            return {
                "status": "❌ ERRO",
                "error": str(e)
            }
    
    def check_n8n(self):
        """Verifica se N8N está rodando"""
        try:
            import requests
            response = requests.get("http://localhost:5678/api/v1/executions", timeout=5)
            return {
                "status": "✅ N8N Online",
                "port": 5678
            }
        except:
            return {
                "status": "❌ N8N Offline",
                "port": 5678
            }
    
    def run(self):
        """Executa todos os health checks"""
        print("=" * 50)
        print(f"🏥 HEALTH CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        db_check = self.check_database()
        print(f"\n📊 Database: {db_check['status']}")
        if "total_candles" in db_check:
            print(f"   Total de candles: {db_check['total_candles']}")
            print(f"   Última atualização: {db_check['last_update']}")
        
        n8n_check = self.check_n8n()
        print(f"\n🤖 N8N: {n8n_check['status']}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    hc = HealthCheck()
    hc.run()
