"""
Konfigürasyon Yönetimi
Tüm bot ayarları ve environment değişkenleri burada yönetilir.
"""

import os
from dotenv import load_dotenv
from typing import Optional

# .env dosyasını yükle
load_dotenv()

class Config:
    """Bot konfigürasyon sınıfı"""
    
    def __init__(self):
        """Konfigürasyon değerlerini yükle"""
        
        # Telegram Bot Ayarları
        self.TELEGRAM_TOKEN: str = self._get_env_var("TELEGRAM_TOKEN")
        self.TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
        
        # Binance API Ayarları
        self.BINANCE_API_KEY: str = self._get_env_var("BINANCE_API_KEY")
        self.BINANCE_SECRET_KEY: str = self._get_env_var("BINANCE_SECRET_KEY")
        self.BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "True").lower() == "true"
        
        # Trading Ayarları
        self.DEFAULT_SYMBOL: str = os.getenv("DEFAULT_SYMBOL", "BTCUSDT")
        self.MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "100.0"))
        self.STOP_LOSS_PERCENTAGE: float = float(os.getenv("STOP_LOSS_PERCENTAGE", "2.0"))
        self.TAKE_PROFIT_PERCENTAGE: float = float(os.getenv("TAKE_PROFIT_PERCENTAGE", "5.0"))
        
        # Risk Yönetimi
        self.MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "50.0"))
        self.MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
        self.RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "1.0"))
        
        # ML Model Ayarları
        self.MODEL_UPDATE_INTERVAL: int = int(os.getenv("MODEL_UPDATE_INTERVAL", "24"))  # saat
        self.PREDICTION_CONFIDENCE_THRESHOLD: float = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.7"))
        
        # Database Ayarları
        self.DATABASE_PATH: str = os.getenv("DATABASE_PATH", "trading_bot.db")
        
        # Logging Ayarları
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE: str = os.getenv("LOG_FILE", "logs/trading_bot.log")
        
        # Güvenlik Ayarları
        self.ENCRYPTION_KEY: Optional[str] = os.getenv("ENCRYPTION_KEY")
        self.API_RATE_LIMIT: int = int(os.getenv("API_RATE_LIMIT", "1200"))  # dakika başına
        
        # Monitoring Ayarları
        self.HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "300"))  # saniye
        self.ALERT_THRESHOLD: float = float(os.getenv("ALERT_THRESHOLD", "5.0"))  # % kayıp
        
    def _get_env_var(self, var_name: str) -> str:
        """Zorunlu environment değişkenini al"""
        value = os.getenv(var_name)
        if not value:
            raise ValueError(f"Zorunlu environment değişkeni eksik: {var_name}")
        return value
    
    def validate_config(self) -> bool:
        """Konfigürasyon doğrulaması"""
        try:
            # Temel doğrulamalar
            assert self.TELEGRAM_TOKEN, "Telegram token eksik"
            assert self.BINANCE_API_KEY, "Binance API key eksik"
            assert self.BINANCE_SECRET_KEY, "Binance secret key eksik"
            
            # Sayısal değer doğrulamaları
            assert 0 < self.MAX_POSITION_SIZE <= 10000, "Geçersiz pozisyon boyutu"
            assert 0 < self.STOP_LOSS_PERCENTAGE <= 50, "Geçersiz stop loss yüzdesi"
            assert 0 < self.TAKE_PROFIT_PERCENTAGE <= 100, "Geçersiz take profit yüzdesi"
            assert 0 < self.RISK_PER_TRADE <= 10, "Geçersiz trade riski"
            
            return True
            
        except AssertionError as e:
            raise ValueError(f"Konfigürasyon doğrulama hatası: {e}")
    
    def get_binance_config(self) -> dict:
        """Binance konfigürasyonunu döndür"""
        return {
            "api_key": self.BINANCE_API_KEY,
            "secret_key": self.BINANCE_SECRET_KEY,
            "testnet": self.BINANCE_TESTNET
        }
    
    def get_trading_config(self) -> dict:
        """Trading konfigürasyonunu döndür"""
        return {
            "default_symbol": self.DEFAULT_SYMBOL,
            "max_position_size": self.MAX_POSITION_SIZE,
            "stop_loss_percentage": self.STOP_LOSS_PERCENTAGE,
            "take_profit_percentage": self.TAKE_PROFIT_PERCENTAGE,
            "max_daily_loss": self.MAX_DAILY_LOSS,
            "max_open_positions": self.MAX_OPEN_POSITIONS,
            "risk_per_trade": self.RISK_PER_TRADE
        }
    
    def __str__(self) -> str:
        """Konfigürasyon özeti (güvenli)"""
        return f"""
Trading Bot Konfigürasyonu:
- Default Symbol: {self.DEFAULT_SYMBOL}
- Max Position Size: {self.MAX_POSITION_SIZE}
- Stop Loss: {self.STOP_LOSS_PERCENTAGE}%
- Take Profit: {self.TAKE_PROFIT_PERCENTAGE}%
- Risk Per Trade: {self.RISK_PER_TRADE}%
- Testnet: {self.BINANCE_TESTNET}
        """

# Global config instance
config = Config()
