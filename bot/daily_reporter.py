"""
Daily Report Automation
Otomatik günlük rapor gönderimi ve scheduling
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Optional
import schedule
import threading

from ..utils.database import AdvancedDatabaseManager
from ..utils.monitoring import MonitoringSystem
from .telegram_handler import TelegramHandler

class DailyReporter:
    """Günlük rapor otomasyonu"""
    
    def __init__(self, db_manager: AdvancedDatabaseManager, telegram_handler: TelegramHandler):
        """
        Daily reporter başlatıcı
        
        Args:
            db_manager: Database manager
            telegram_handler: Telegram handler
        """
        self.db = db_manager
        self.telegram = telegram_handler
        self.monitoring = MonitoringSystem(db_manager)
        self.logger = logging.getLogger(__name__)
        
        # Scheduler
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # Report settings
        self.report_time = time(9, 0)  # 09:00 varsayılan
        self.enabled = True
        
    def start_scheduler(self):
        """Scheduler'ı başlat"""
        try:
            if self.scheduler_running:
                self.logger.warning("Scheduler already running")
                return
            
            # Günlük rapor schedule'ı
            schedule.every().day.at("09:00").do(self._send_daily_report_job)
            
            # Scheduler thread'i başlat
            self.scheduler_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            self.logger.info("Daily report scheduler started")
            
        except Exception as e:
            self.logger.error(f"Scheduler start error: {e}")
    
    def stop_scheduler(self):
        """Scheduler'ı durdur"""
        try:
            self.scheduler_running = False
            schedule.clear()
            
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=5)
            
            self.logger.info("Daily report scheduler stopped")
            
        except Exception as e:
            self.logger.error(f"Scheduler stop error: {e}")
    
    def _run_scheduler(self):
        """Scheduler loop"""
        while self.scheduler_running:
            try:
                schedule.run_pending()
                asyncio.sleep(60)  # Her dakika kontrol et
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                asyncio.sleep(60)
    
    def _send_daily_report_job(self):
        """Günlük rapor gönderme job'ı"""
        try:
            # Async fonksiyonu sync olarak çalıştır
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_daily_report())
            loop.close()
            
        except Exception as e:
            self.logger.error(f"Daily report job error: {e}")
    
    async def send_daily_report(self, force: bool = False) -> bool:
        """
        Günlük rapor gönder
        
        Args:
            force: Zorla gönder (tarih kontrolü yapma)
            
        Returns:
            bool: Başarılı mı
        """
        try:
            # Enabled kontrolü
            if not self.enabled and not force:
                self.logger.info("Daily reports disabled")
                return False
            
            # Rapor gönderilmeli mi kontrol et
            if not force and not await self.monitoring.should_send_daily_report():
                self.logger.info("Daily report already sent today")
                return False
            
            # Günlük rapor oluştur
            self.logger.info("Generating daily report...")
            report = self.monitoring.generate_daily_report()
            
            # Telegram formatına çevir
            telegram_message = self.monitoring.format_report_for_telegram(report)
            
            # Telegram'a gönder
            success = await self.telegram.send_message(telegram_message)
            
            if success:
                self.logger.info("Daily report sent successfully")
                
                # Son gönderim tarihini güncelle
                self.db.set_setting('last_report_date', report.date)
                
                return True
            else:
                self.logger.error("Failed to send daily report")
                return False
                
        except Exception as e:
            self.logger.error(f"Send daily report error: {e}")
            return False
    
    async def send_performance_summary(self, days: int = 7) -> bool:
        """
        Performans özeti gönder
        
        Args:
            days: Kaç günlük özet
            
        Returns:
            bool: Başarılı mı
        """
        try:
            # Performance summary al
            performance = self.db.get_performance_summary(days=days)
            
            if not performance:
                await self.telegram.send_message("❌ Performans verisi bulunamadı.")
                return False
            
            # Formatla
            message = self._format_performance_summary(performance, days)
            
            # Gönder
            return await self.telegram.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Performance summary error: {e}")
            return False
    
    def _format_performance_summary(self, performance: dict, days: int) -> str:
        """Performans özetini formatla"""
        try:
            lines = []
            lines.append(f"📊 **{days} Günlük Performans Özeti**")
            lines.append("")
            
            # Ana metrikler
            lines.append("💰 **Finansal Durum:**")
            lines.append(f"• Mevcut Sermaye: ${performance.get('current_capital', 0):.2f}")
            lines.append(f"• Toplam Return: %{performance.get('total_return', 0)*100:.2f}")
            lines.append(f"• Unrealized P&L: ${performance.get('unrealized_pnl', 0):.2f}")
            lines.append("")
            
            # Trading metrikleri
            lines.append("📈 **Trading Metrikleri:**")
            lines.append(f"• Toplam Trade: {performance.get('total_trades', 0)}")
            lines.append(f"• Kazanan Trade: {performance.get('winning_trades', 0)}")
            lines.append(f"• Kaybeden Trade: {performance.get('losing_trades', 0)}")
            lines.append(f"• Win Rate: %{performance.get('win_rate', 0)*100:.1f}")
            lines.append("")
            
            # Risk metrikleri
            lines.append("⚠️ **Risk Metrikleri:**")
            lines.append(f"• Max Drawdown: %{performance.get('max_drawdown', 0)*100:.2f}")
            lines.append(f"• Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
            lines.append(f"• Profit Factor: {performance.get('profit_factor', 0):.2f}")
            lines.append("")
            
            # Portfolio
            lines.append("💼 **Portfolio:**")
            lines.append(f"• Aktif Pozisyon: {performance.get('active_positions', 0)}")
            lines.append("")
            
            lines.append(f"📅 Son Güncelleme: {performance.get('last_updated', 'N/A')}")
            lines.append("🤖 *Trading Bot Performans Raporu*")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Performance format error: {e}")
            return "❌ Performans raporu formatlanırken hata oluştu."
    
    async def send_real_time_status(self) -> bool:
        """
        Real-time durum gönder
        
        Returns:
            bool: Başarılı mı
        """
        try:
            # Real-time metrics al
            metrics = self.monitoring.get_real_time_metrics()
            
            if not metrics:
                await self.telegram.send_message("❌ Real-time veriler alınamadı.")
                return False
            
            # Formatla
            message = self._format_real_time_status(metrics)
            
            # Gönder
            return await self.telegram.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Real-time status error: {e}")
            return False
    
    def _format_real_time_status(self, metrics: dict) -> str:
        """Real-time durumu formatla"""
        try:
            lines = []
            lines.append("⚡ **Real-Time Durum**")
            lines.append(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
            lines.append("")
            
            # Ana durum
            lines.append("💰 **Anlık Durum:**")
            lines.append(f"• Sermaye: ${metrics.get('current_capital', 0):.2f}")
            lines.append(f"• Günlük P&L: ${metrics.get('daily_pnl', 0):.2f}")
            lines.append(f"• Unrealized: ${metrics.get('unrealized_pnl', 0):.2f}")
            lines.append(f"• Aktif Pozisyon: {metrics.get('active_positions', 0)}")
            lines.append("")
            
            # Son aktiviteler
            last_trade = metrics.get('last_trade')
            if last_trade:
                lines.append("🔄 **Son Trade:**")
                lines.append(f"• {last_trade['symbol']} {last_trade['side']}")
                lines.append(f"• P&L: ${last_trade['pnl']:.2f}")
                lines.append(f"• Zaman: {last_trade['timestamp'][:16]}")
                lines.append("")
            
            last_signal = metrics.get('last_signal')
            if last_signal:
                lines.append("🎯 **Son Sinyal:**")
                lines.append(f"• {last_signal['model']}: {last_signal['symbol']}")
                lines.append(f"• Sinyal: {last_signal['signal']}")
                lines.append(f"• Güven: %{last_signal['confidence']*100:.1f}")
                lines.append("")
            
            # Performans
            lines.append("📊 **Performans:**")
            lines.append(f"• Win Rate: %{metrics.get('win_rate', 0)*100:.1f}")
            lines.append(f"• Toplam Trade: {metrics.get('total_trades', 0)}")
            lines.append("")
            
            lines.append("🤖 *Real-time Trading Bot Status*")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Real-time format error: {e}")
            return "❌ Real-time durum formatlanırken hata oluştu."
    
    def set_report_time(self, hour: int, minute: int = 0):
        """
        Rapor gönderim saatini ayarla
        
        Args:
            hour: Saat (0-23)
            minute: Dakika (0-59)
        """
        try:
            self.report_time = time(hour, minute)
            
            # Schedule'ı güncelle
            schedule.clear()
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(self._send_daily_report_job)
            
            # Database'e kaydet
            self.db.set_setting('daily_report_time', f"{hour:02d}:{minute:02d}")
            
            self.logger.info(f"Report time set to {hour:02d}:{minute:02d}")
            
        except Exception as e:
            self.logger.error(f"Set report time error: {e}")
    
    def enable_reports(self, enabled: bool = True):
        """
        Raporları etkinleştir/devre dışı bırak
        
        Args:
            enabled: Etkin mi
        """
        try:
            self.enabled = enabled
            
            # Database'e kaydet
            self.db.set_setting('telegram_reports_enabled', enabled)
            
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"Daily reports {status}")
            
        except Exception as e:
            self.logger.error(f"Enable reports error: {e}")
    
    async def send_test_report(self) -> bool:
        """
        Test raporu gönder
        
        Returns:
            bool: Başarılı mı
        """
        try:
            message = """
🧪 **Test Raporu**

Bu bir test mesajıdır. Günlük raporlar şu bilgileri içerir:

📊 **Trade Özeti**
📈 **Performans Metrikleri** 
💰 **Portföy Durumu**
🎯 **Sinyal Özeti**
⚠️ **Risk Olayları**
💡 **Öneriler**

✅ Telegram entegrasyonu çalışıyor!

🤖 *Trading Bot Test*
            """
            
            return await self.telegram.send_message(message.strip())
            
        except Exception as e:
            self.logger.error(f"Test report error: {e}")
            return False
    
    def get_status(self) -> dict:
        """
        Reporter durumunu al
        
        Returns:
            dict: Durum bilgileri
        """
        try:
            return {
                'enabled': self.enabled,
                'scheduler_running': self.scheduler_running,
                'report_time': self.report_time.strftime('%H:%M'),
                'last_report_date': self.db.get_setting('last_report_date'),
                'telegram_connected': self.telegram.is_connected() if hasattr(self.telegram, 'is_connected') else True
            }
            
        except Exception as e:
            self.logger.error(f"Get status error: {e}")
            return {}
