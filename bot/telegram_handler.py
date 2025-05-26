"""
Advanced Telegram Bot Handler
Gelişmiş Telegram bot işlevselliği ve mesaj yönetimi
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Set
from telegram import Update, Bot, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import TelegramError, BadRequest, Forbidden
from datetime import datetime, timedelta
import json

class RateLimiter:
    """Rate limiting sınıfı"""
    
    def __init__(self, max_requests: int = 30, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    def is_allowed(self, user_id: int) -> bool:
        """Kullanıcının rate limit'e takılıp takılmadığını kontrol et"""
        now = time.time()
        
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Eski istekleri temizle
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if now - req_time < self.time_window
        ]
        
        # Limit kontrolü
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Yeni isteği ekle
        self.requests[user_id].append(now)
        return True

class TelegramBot:
    """Gelişmiş Telegram bot ana sınıfı"""
    
    def __init__(self, token: str, authorized_users: List[int] = None, 
                 admin_users: List[int] = None):
        """
        Telegram bot başlatıcı
        
        Args:
            token: Telegram bot token
            authorized_users: Yetkili kullanıcı ID'leri
            admin_users: Admin kullanıcı ID'leri
        """
        self.token = token
        self.authorized_users = set(authorized_users or [])
        self.admin_users = set(admin_users or [])
        self.application: Optional[Application] = None
        self.bot: Optional[Bot] = None
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        
        # Rate limiting
        self.rate_limiter = RateLimiter(max_requests=30, time_window=60)
        
        # Bot durumu
        self.bot_status = {
            'trading_enabled': True,
            'maintenance_mode': False,
            'last_heartbeat': datetime.now(),
            'total_commands': 0,
            'active_users': set()
        }
        
        # Keyboard menüleri
        self.main_keyboard = self._create_main_keyboard()
        self.trading_keyboard = self._create_trading_keyboard()
        self.settings_keyboard = self._create_settings_keyboard()
        
    def _create_main_keyboard(self) -> ReplyKeyboardMarkup:
        """Ana menü keyboard'u oluştur"""
        keyboard = [
            [KeyboardButton("📊 Status"), KeyboardButton("💰 Balance")],
            [KeyboardButton("📈 Positions"), KeyboardButton("📉 P&L")],
            [KeyboardButton("🤖 AI Models"), KeyboardButton("⚙️ Settings")],
            [KeyboardButton("🆘 Emergency Stop"), KeyboardButton("ℹ️ Help")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    
    def _create_trading_keyboard(self) -> ReplyKeyboardMarkup:
        """Trading menü keyboard'u oluştur"""
        keyboard = [
            [KeyboardButton("🟢 Buy BTC"), KeyboardButton("🔴 Sell BTC")],
            [KeyboardButton("🟢 Buy ETH"), KeyboardButton("🔴 Sell ETH")],
            [KeyboardButton("📋 Quick Orders"), KeyboardButton("🔙 Back to Main")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    
    def _create_settings_keyboard(self) -> ReplyKeyboardMarkup:
        """Ayarlar menü keyboard'u oluştur"""
        keyboard = [
            [KeyboardButton("🛡️ Risk Settings"), KeyboardButton("🔔 Notifications")],
            [KeyboardButton("🤖 AI Settings"), KeyboardButton("📊 Display Settings")],
            [KeyboardButton("🔙 Back to Main")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    
    async def start(self):
        """Bot'u başlat"""
        try:
            # Application oluştur
            self.application = Application.builder().token(self.token).build()
            self.bot = self.application.bot
            
            # Komutları kaydet
            await self._register_handlers()
            
            # Bot'u başlat
            await self.application.initialize()
            await self.application.start()
            
            self.is_running = True
            self.bot_status['last_heartbeat'] = datetime.now()
            
            self.logger.info("Telegram bot başarıyla başlatıldı")
            
            # Bot bilgilerini al
            bot_info = await self.bot.get_me()
            self.logger.info(f"Bot aktif: @{bot_info.username}")
            
            # Başlangıç mesajı gönder (admin'lere)
            await self._send_startup_notification()
            
        except Exception as e:
            self.logger.error(f"Telegram bot başlatma hatası: {e}")
            raise
    
    async def stop(self):
        """Bot'u durdur"""
        try:
            if self.application and self.is_running:
                # Kapatma mesajı gönder
                await self._send_shutdown_notification()
                
                await self.application.stop()
                await self.application.shutdown()
                self.is_running = False
                self.logger.info("Telegram bot durduruldu")
        except Exception as e:
            self.logger.error(f"Bot durdurma hatası: {e}")
    
    async def _register_handlers(self):
        """Komut ve mesaj handler'larını kaydet"""
        from .commands import CommandHandler as CmdHandler
        
        cmd_handler = CmdHandler(self)
        
        # Komut handler'ları
        self.application.add_handler(CommandHandler("start", cmd_handler.start_command))
        self.application.add_handler(CommandHandler("help", cmd_handler.help_command))
        self.application.add_handler(CommandHandler("status", cmd_handler.status_command))
        self.application.add_handler(CommandHandler("balance", cmd_handler.balance_command))
        self.application.add_handler(CommandHandler("buy", cmd_handler.buy_command))
        self.application.add_handler(CommandHandler("sell", cmd_handler.sell_command))
        self.application.add_handler(CommandHandler("positions", cmd_handler.positions_command))
        self.application.add_handler(CommandHandler("pnl", cmd_handler.pnl_command))
        self.application.add_handler(CommandHandler("settings", cmd_handler.settings_command))
        self.application.add_handler(CommandHandler("models", cmd_handler.models_command))
        self.application.add_handler(CommandHandler("emergency", cmd_handler.emergency_command))
        self.application.add_handler(CommandHandler("logs", cmd_handler.logs_command))
        
        # Admin komutları
        self.application.add_handler(CommandHandler("admin", cmd_handler.admin_command))
        self.application.add_handler(CommandHandler("users", cmd_handler.users_command))
        self.application.add_handler(CommandHandler("broadcast", cmd_handler.broadcast_command))
        
        # Callback query handler (inline keyboard'lar için)
        self.application.add_handler(CallbackQueryHandler(cmd_handler.handle_callback))
        
        # Mesaj handler'ı (keyboard butonları için)
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, cmd_handler.handle_message)
        )
        
        # Hata handler'ı
        self.application.add_error_handler(self._error_handler)
    
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gelişmiş hata yönetimi"""
        error = context.error
        self.logger.error(f"Telegram bot hatası: {error}")
        
        # Hata tipine göre farklı mesajlar
        error_message = "❌ Bir hata oluştu."
        
        if isinstance(error, BadRequest):
            error_message = "❌ Geçersiz istek. Lütfen komutu kontrol edin."
        elif isinstance(error, Forbidden):
            error_message = "❌ Bot'a mesaj gönderme izni yok."
        elif isinstance(error, TelegramError):
            error_message = "❌ Telegram bağlantı hatası. Lütfen tekrar deneyin."
        
        if update and update.effective_chat:
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=error_message,
                    reply_markup=self.main_keyboard
                )
            except Exception as e:
                self.logger.error(f"Hata mesajı gönderme hatası: {e}")
    
    def is_authorized_user(self, user_id: int) -> bool:
        """
        Kullanıcı yetkilendirme kontrolü
        
        Args:
            user_id: Telegram kullanıcı ID
            
        Returns:
            bool: Yetki durumu
        """
        # Eğer authorized_users boşsa, herkese izin ver (geliştirme için)
        if not self.authorized_users:
            return True
        
        return user_id in self.authorized_users or user_id in self.admin_users
    
    def is_admin_user(self, user_id: int) -> bool:
        """
        Admin kullanıcı kontrolü
        
        Args:
            user_id: Telegram kullanıcı ID
            
        Returns:
            bool: Admin durumu
        """
        return user_id in self.admin_users
    
    def check_rate_limit(self, user_id: int) -> bool:
        """
        Rate limit kontrolü
        
        Args:
            user_id: Telegram kullanıcı ID
            
        Returns:
            bool: Rate limit durumu
        """
        return self.rate_limiter.is_allowed(user_id)
    
    async def send_message(self, chat_id: int, text: str, 
                          parse_mode: str = "HTML",
                          reply_markup=None,
                          disable_web_page_preview: bool = True) -> bool:
        """
        Güvenli mesaj gönderme
        
        Args:
            chat_id: Hedef chat ID
            text: Gönderilecek mesaj
            parse_mode: Mesaj formatı
            reply_markup: Keyboard markup
            disable_web_page_preview: Web önizleme devre dışı
            
        Returns:
            bool: Başarı durumu
        """
        try:
            if not self.bot:
                self.logger.error("Bot başlatılmamış")
                return False
            
            # Mesaj uzunluğu kontrolü (Telegram limiti: 4096 karakter)
            if len(text) > 4000:
                # Uzun mesajları böl
                chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                for i, chunk in enumerate(chunks):
                    markup = reply_markup if i == len(chunks) - 1 else None
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        parse_mode=parse_mode,
                        reply_markup=markup,
                        disable_web_page_preview=disable_web_page_preview
                    )
                return True
            else:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    disable_web_page_preview=disable_web_page_preview
                )
                return True
            
        except TelegramError as e:
            self.logger.error(f"Telegram mesaj gönderme hatası: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Mesaj gönderme hatası: {e}")
            return False
    
    async def send_alert(self, message: str, level: str = "INFO", 
                        chat_ids: List[int] = None) -> bool:
        """
        Alert mesajı gönder
        
        Args:
            message: Alert mesajı
            level: Alert seviyesi (INFO, WARNING, ERROR, SUCCESS)
            chat_ids: Hedef chat ID'leri (None ise tüm admin'ler)
            
        Returns:
            bool: Başarı durumu
        """
        emoji_map = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✅",
            "CRITICAL": "🚨"
        }
        
        emoji = emoji_map.get(level, "📢")
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"{emoji} <b>[{level}]</b> {timestamp}\n{message}"
        
        # Hedef kullanıcıları belirle
        targets = chat_ids or list(self.admin_users)
        
        success_count = 0
        for chat_id in targets:
            if await self.send_message(chat_id, formatted_message):
                success_count += 1
        
        return success_count > 0
    
    async def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """
        Trade bildirimi gönder
        
        Args:
            trade_data: Trade verileri
            
        Returns:
            bool: Başarı durumu
        """
        try:
            symbol = trade_data.get('symbol', 'N/A')
            side = trade_data.get('side', 'N/A')
            quantity = trade_data.get('quantity', 'N/A')
            price = trade_data.get('price', 'N/A')
            order_type = trade_data.get('order_type', 'MARKET')
            
            side_emoji = "🟢" if side.upper() == "BUY" else "🔴"
            
            message = f"""
{side_emoji} <b>Trade Executed</b>

📊 <b>Symbol:</b> {symbol}
📈 <b>Side:</b> {side.upper()}
💰 <b>Quantity:</b> {quantity}
💵 <b>Price:</b> ${price}
📋 <b>Type:</b> {order_type}
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
            """
            
            return await self.send_alert(message.strip(), "SUCCESS")
            
        except Exception as e:
            self.logger.error(f"Trade bildirimi hatası: {e}")
            return False
    
    async def send_performance_report(self, performance_data: Dict[str, Any]) -> bool:
        """
        Performans raporu gönder
        
        Args:
            performance_data: Performans verileri
            
        Returns:
            bool: Başarı durumu
        """
        try:
            total_pnl = performance_data.get('total_pnl', 0)
            daily_pnl = performance_data.get('daily_pnl', 0)
            win_rate = performance_data.get('win_rate', 0)
            total_trades = performance_data.get('total_trades', 0)
            
            pnl_emoji = "📈" if total_pnl >= 0 else "📉"
            daily_emoji = "📈" if daily_pnl >= 0 else "📉"
            
            message = f"""
📊 <b>Performance Report</b>

{pnl_emoji} <b>Total P&L:</b> ${total_pnl:.2f}
{daily_emoji} <b>Daily P&L:</b> ${daily_pnl:.2f}
🎯 <b>Win Rate:</b> {win_rate:.1f}%
📈 <b>Total Trades:</b> {total_trades}
⏰ <b>Report Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return await self.send_alert(message.strip(), "INFO")
            
        except Exception as e:
            self.logger.error(f"Performans raporu hatası: {e}")
            return False
    
    async def _send_startup_notification(self):
        """Bot başlangıç bildirimi"""
        message = f"""
🤖 <b>Trading Bot Started</b>

✅ Bot successfully initialized
⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
👥 Authorized users: {len(self.authorized_users)}
🔧 Admin users: {len(self.admin_users)}

Bot is ready for trading operations.
        """
        await self.send_alert(message.strip(), "SUCCESS")
    
    async def _send_shutdown_notification(self):
        """Bot kapatma bildirimi"""
        message = f"""
🤖 <b>Trading Bot Shutdown</b>

⏹️ Bot is shutting down
⏰ Shutdown time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Total commands processed: {self.bot_status['total_commands']}

Bot has been safely stopped.
        """
        await self.send_alert(message.strip(), "WARNING")
    
    def update_bot_status(self, **kwargs):
        """Bot durumunu güncelle"""
        self.bot_status.update(kwargs)
        self.bot_status['last_heartbeat'] = datetime.now()
    
    def add_active_user(self, user_id: int):
        """Aktif kullanıcı ekle"""
        self.bot_status['active_users'].add(user_id)
    
    def increment_command_count(self):
        """Komut sayacını artır"""
        self.bot_status['total_commands'] += 1
    
    def get_bot_stats(self) -> Dict[str, Any]:
        """Bot istatistiklerini al"""
        uptime = datetime.now() - self.bot_status['last_heartbeat']
        
        return {
            'is_running': self.is_running,
            'trading_enabled': self.bot_status['trading_enabled'],
            'maintenance_mode': self.bot_status['maintenance_mode'],
            'uptime_seconds': uptime.total_seconds(),
            'total_commands': self.bot_status['total_commands'],
            'active_users_count': len(self.bot_status['active_users']),
            'authorized_users_count': len(self.authorized_users),
            'admin_users_count': len(self.admin_users),
            'last_heartbeat': self.bot_status['last_heartbeat'].isoformat()
        }
    
    async def broadcast_message(self, message: str, user_type: str = "all") -> Dict[str, int]:
        """
        Toplu mesaj gönder
        
        Args:
            message: Gönderilecek mesaj
            user_type: Hedef kullanıcı tipi (all, authorized, admin)
            
        Returns:
            Dict: Gönderim istatistikleri
        """
        targets = set()
        
        if user_type == "all":
            targets.update(self.authorized_users)
            targets.update(self.admin_users)
        elif user_type == "authorized":
            targets.update(self.authorized_users)
        elif user_type == "admin":
            targets.update(self.admin_users)
        
        success_count = 0
        failed_count = 0
        
        for user_id in targets:
            try:
                if await self.send_message(user_id, message):
                    success_count += 1
                else:
                    failed_count += 1
                    
                # Rate limiting için kısa bekleme
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Broadcast hatası (user {user_id}): {e}")
                failed_count += 1
        
        return {
            'success': success_count,
            'failed': failed_count,
            'total': len(targets)
        }
