"""
Cryptocurrency Trading Bot - Ana Çalışma Dosyası
Bu dosya botun ana giriş noktasıdır ve tüm bileşenleri koordine eder.
"""

import asyncio
import logging
from config import Config
from bot.telegram_handler import TelegramBot
from exchange.binance_client import BinanceClient
from utils.database import DatabaseManager
from utils.logger import setup_logger
from risk.risk_manager import RiskManager

class TradingBot:
    def __init__(self):
        """Trading bot ana sınıfı"""
        self.config = Config()
        self.logger = setup_logger()
        self.db = DatabaseManager()
        self.telegram_bot = TelegramBot(self.config.TELEGRAM_TOKEN)
        self.binance_client = BinanceClient(
            self.config.BINANCE_API_KEY,
            self.config.BINANCE_SECRET_KEY
        )
        self.risk_manager = RiskManager()
        
    async def initialize(self):
        """Bot bileşenlerini başlat"""
        try:
            self.logger.info("Trading bot başlatılıyor...")
            
            # Database bağlantısını başlat
            await self.db.initialize()
            
            # Binance bağlantısını test et
            await self.binance_client.test_connection()
            
            # Telegram bot'u başlat
            await self.telegram_bot.start()
            
            self.logger.info("Trading bot başarıyla başlatıldı!")
            
        except Exception as e:
            self.logger.error(f"Bot başlatma hatası: {e}")
            raise
    
    async def run(self):
        """Ana bot döngüsü"""
        try:
            await self.initialize()
            
            # Ana trading döngüsü
            while True:
                try:
                    # Market verilerini al
                    # ML tahminlerini çalıştır
                    # Risk kontrolü yap
                    # Trade sinyalleri üret
                    
                    await asyncio.sleep(60)  # 1 dakika bekle
                    
                except Exception as e:
                    self.logger.error(f"Trading döngüsü hatası: {e}")
                    await asyncio.sleep(30)
                    
        except KeyboardInterrupt:
            self.logger.info("Bot kapatılıyor...")
        except Exception as e:
            self.logger.error(f"Kritik hata: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Temizlik işlemleri"""
        try:
            await self.telegram_bot.stop()
            await self.db.close()
            self.logger.info("Bot temizlik işlemleri tamamlandı")
        except Exception as e:
            self.logger.error(f"Temizlik hatası: {e}")

async def main():
    """Ana fonksiyon"""
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
