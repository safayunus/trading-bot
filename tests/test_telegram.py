"""
Telegram Bot Tests
Telegram bot fonksiyonlarının test edilmesi
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

from tests import TestUtils, PerformanceTimer, get_memory_usage


class TestTelegramHandler:
    """TelegramBot sınıfı testleri"""
    
    @pytest.fixture
    def telegram_handler(self, mock_telegram_bot):
        """TelegramBot instance oluştur"""
        with patch('bot.telegram_handler.Bot', return_value=mock_telegram_bot):
            from bot.telegram_handler import TelegramBot
            
            # Mock config
            config = Mock()
            config.TELEGRAM_BOT_TOKEN = 'test_token'
            config.TELEGRAM_CHAT_ID = '123456789'
            config.TELEGRAM_ADMIN_IDS = ['123456789']
            
            handler = TelegramBot(config)
            return handler
    
    def test_telegram_handler_initialization(self, telegram_handler):
        """TelegramBot başlatma testi"""
        assert telegram_handler is not None
        assert telegram_handler.authorized_users == ['123456789']
        assert telegram_handler.admin_users == ['123456789']
        assert telegram_handler.bot_status['is_running'] == False
    
    def test_user_authorization(self, telegram_handler):
        """Kullanıcı yetkilendirme testi"""
        # Yetkili kullanıcı
        assert telegram_handler.is_authorized_user('123456789') == True
        
        # Yetkisiz kullanıcı
        assert telegram_handler.is_authorized_user('987654321') == False
    
    def test_admin_authorization(self, telegram_handler):
        """Admin yetkilendirme testi"""
        # Admin kullanıcı
        assert telegram_handler.is_admin_user('123456789') == True
        
        # Normal kullanıcı
        assert telegram_handler.is_admin_user('987654321') == False
    
    def test_rate_limiting(self, telegram_handler):
        """Rate limiting testi"""
        user_id = '123456789'
        
        # İlk istekler geçmeli
        for i in range(10):
            assert telegram_handler.check_rate_limit(user_id) == True
        
        # 11. istek reddedilmeli
        assert telegram_handler.check_rate_limit(user_id) == False
    
    def test_bot_status_management(self, telegram_handler):
        """Bot durum yönetimi testi"""
        # Başlangıç durumu
        assert telegram_handler.bot_status['is_running'] == False
        assert telegram_handler.bot_status['trading_enabled'] == True
        
        # Durum güncelleme
        telegram_handler.update_bot_status(is_running=True, trading_enabled=False)
        assert telegram_handler.bot_status['is_running'] == True
        assert telegram_handler.bot_status['trading_enabled'] == False
    
    def test_active_user_management(self, telegram_handler):
        """Aktif kullanıcı yönetimi testi"""
        user_id = '123456789'
        
        # Kullanıcı ekleme
        telegram_handler.add_active_user(user_id)
        assert user_id in telegram_handler.bot_status['active_users']
        
        # Kullanıcı sayısı
        stats = telegram_handler.get_bot_stats()
        assert stats['active_users_count'] == 1
    
    def test_command_counting(self, telegram_handler):
        """Komut sayma testi"""
        initial_count = telegram_handler.bot_status['total_commands']
        
        # Komut sayısını artır
        telegram_handler.increment_command_count()
        telegram_handler.increment_command_count()
        
        assert telegram_handler.bot_status['total_commands'] == initial_count + 2
    
    @pytest.mark.asyncio
    async def test_send_message(self, telegram_handler, mock_telegram_bot):
        """Mesaj gönderme testi"""
        # Mock async send_message
        mock_telegram_bot.send_message = AsyncMock(return_value=Mock(message_id=123))
        
        # Mesaj gönder
        result = await telegram_handler.send_message("Test message")
        
        # Doğrulama
        assert result == True
        mock_telegram_bot.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_alert(self, telegram_handler, mock_telegram_bot):
        """Alert gönderme testi"""
        # Mock async send_message
        mock_telegram_bot.send_message = AsyncMock(return_value=Mock(message_id=123))
        
        # Alert gönder
        result = await telegram_handler.send_alert("Test alert", "HIGH")
        
        # Doğrulama
        assert result == True
        mock_telegram_bot.send_message.assert_called_once()
        
        # Alert mesajının formatını kontrol et
        call_args = mock_telegram_bot.send_message.call_args
        message_text = call_args[1]['text']
        assert "🚨" in message_text
        assert "Test alert" in message_text
    
    @pytest.mark.asyncio
    async def test_send_trade_notification(self, telegram_handler, mock_telegram_bot):
        """Trade bildirimi testi"""
        # Mock async send_message
        mock_telegram_bot.send_message = AsyncMock(return_value=Mock(message_id=123))
        
        # Trade verisi
        trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.01,
            'price': 45000.0,
            'order_type': 'MARKET'
        }
        
        # Trade bildirimi gönder
        result = await telegram_handler.send_trade_notification(trade_data)
        
        # Doğrulama
        assert result == True
        mock_telegram_bot.send_message.assert_called_once()
        
        # Mesaj formatını kontrol et
        call_args = mock_telegram_bot.send_message.call_args
        message_text = call_args[1]['text']
        assert "BTCUSDT" in message_text
        assert "BUY" in message_text
    
    def test_keyboard_generation(self, telegram_handler):
        """Keyboard oluşturma testi"""
        # Ana keyboard
        main_keyboard = telegram_handler.main_keyboard
        assert main_keyboard is not None
        assert len(main_keyboard.keyboard) > 0
        
        # Settings keyboard
        settings_keyboard = telegram_handler.settings_keyboard
        assert settings_keyboard is not None
        assert len(settings_keyboard.keyboard) > 0
    
    def test_performance_send_message(self, telegram_handler, mock_telegram_bot):
        """Mesaj gönderme performans testi"""
        # Mock async send_message
        mock_telegram_bot.send_message = AsyncMock(return_value=Mock(message_id=123))
        
        async def test_performance():
            with PerformanceTimer() as timer:
                # 10 mesaj gönder
                tasks = []
                for i in range(10):
                    task = telegram_handler.send_message(f"Test message {i}")
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
            
            # Performance kontrolü (10 mesaj 1 saniyeden az sürmeli)
            assert timer.elapsed < 1.0
        
        # Test çalıştır
        asyncio.run(test_performance())
    
    def test_memory_usage(self, telegram_handler):
        """Memory kullanım testi"""
        initial_memory = get_memory_usage()
        
        # Çok sayıda kullanıcı ekle
        for i in range(1000):
            telegram_handler.add_active_user(str(i))
        
        current_memory = get_memory_usage()
        memory_increase = current_memory - initial_memory
        
        # Memory artışı 50MB'dan az olmalı
        assert memory_increase < 50


class TestCommandHandler:
    """CommandHandler sınıfı testleri"""
    
    @pytest.fixture
    def command_handler(self, telegram_handler):
        """CommandHandler instance oluştur"""
        from bot.commands import CommandHandler
        return CommandHandler(telegram_handler)
    
    @pytest.fixture
    def mock_update(self):
        """Mock Telegram Update"""
        update = Mock()
        update.effective_user.id = 123456789
        update.effective_user.first_name = "Test User"
        update.message.reply_text = AsyncMock()
        return update
    
    @pytest.fixture
    def mock_context(self):
        """Mock Telegram Context"""
        context = Mock()
        context.args = []
        return context
    
    @pytest.mark.asyncio
    async def test_start_command(self, command_handler, mock_update, mock_context):
        """Start komutu testi"""
        await command_handler.start_command(mock_update, mock_context)
        
        # Reply_text çağrıldığını kontrol et
        mock_update.message.reply_text.assert_called_once()
        
        # Mesaj içeriğini kontrol et
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        assert "Welcome" in message_text
        assert "Test User" in message_text
    
    @pytest.mark.asyncio
    async def test_help_command(self, command_handler, mock_update, mock_context):
        """Help komutu testi"""
        await command_handler.help_command(mock_update, mock_context)
        
        # Reply_text çağrıldığını kontrol et
        mock_update.message.reply_text.assert_called_once()
        
        # Mesaj içeriğini kontrol et
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        assert "Command Reference" in message_text
    
    @pytest.mark.asyncio
    async def test_status_command(self, command_handler, mock_update, mock_context):
        """Status komutu testi"""
        await command_handler.status_command(mock_update, mock_context)
        
        # Reply_text çağrıldığını kontrol et
        mock_update.message.reply_text.assert_called_once()
        
        # Mesaj içeriğini kontrol et
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        assert "Bot Status Dashboard" in message_text
    
    @pytest.mark.asyncio
    async def test_balance_command(self, command_handler, mock_update, mock_context):
        """Balance komutu testi"""
        await command_handler.balance_command(mock_update, mock_context)
        
        # Reply_text çağrıldığını kontrol et
        mock_update.message.reply_text.assert_called_once()
        
        # Mesaj içeriğini kontrol et
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        assert "Portfolio Balance" in message_text
    
    @pytest.mark.asyncio
    async def test_buy_command_valid_input(self, command_handler, mock_update, mock_context):
        """Buy komutu geçerli input testi"""
        mock_context.args = ['BTCUSDT', '0.01']
        
        await command_handler.buy_command(mock_update, mock_context)
        
        # Reply_text çağrıldığını kontrol et
        mock_update.message.reply_text.assert_called_once()
        
        # Mesaj içeriğini kontrol et
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        assert "Buy Order Confirmation" in message_text
        assert "BTCUSDT" in message_text
    
    @pytest.mark.asyncio
    async def test_buy_command_invalid_input(self, command_handler, mock_update, mock_context):
        """Buy komutu geçersiz input testi"""
        mock_context.args = ['INVALID']
        
        await command_handler.buy_command(mock_update, mock_context)
        
        # Help mesajı gönderilmeli
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        assert "Buy Order Help" in message_text
    
    @pytest.mark.asyncio
    async def test_sell_command_valid_input(self, command_handler, mock_update, mock_context):
        """Sell komutu geçerli input testi"""
        mock_context.args = ['BTCUSDT', '0.01']
        
        await command_handler.sell_command(mock_update, mock_context)
        
        # Reply_text çağrıldığını kontrol et
        mock_update.message.reply_text.assert_called_once()
        
        # Mesaj içeriğini kontrol et
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        assert "Sell Order Confirmation" in message_text
        assert "BTCUSDT" in message_text
    
    @pytest.mark.asyncio
    async def test_positions_command(self, command_handler, mock_update, mock_context):
        """Positions komutu testi"""
        await command_handler.positions_command(mock_update, mock_context)
        
        # Reply_text çağrıldığını kontrol et
        mock_update.message.reply_text.assert_called_once()
        
        # Mesaj içeriğini kontrol et
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        # Mock data olduğu için "No Open Positions" veya "Open Positions" olabilir
        assert ("No Open Positions" in message_text) or ("Open Positions" in message_text)
    
    @pytest.mark.asyncio
    async def test_pnl_command(self, command_handler, mock_update, mock_context):
        """PnL komutu testi"""
        await command_handler.pnl_command(mock_update, mock_context)
        
        # Reply_text çağrıldığını kontrol et
        mock_update.message.reply_text.assert_called_once()
        
        # Mesaj içeriğini kontrol et
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        assert "Profit & Loss Report" in message_text
    
    @pytest.mark.asyncio
    async def test_emergency_command(self, command_handler, mock_update, mock_context):
        """Emergency komutu testi"""
        await command_handler.emergency_command(mock_update, mock_context)
        
        # Reply_text çağrıldığını kontrol et
        mock_update.message.reply_text.assert_called_once()
        
        # Mesaj içeriğini kontrol et
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        assert "EMERGENCY STOP" in message_text
        assert "WARNING" in message_text
    
    @pytest.mark.asyncio
    async def test_unauthorized_user(self, command_handler, mock_update, mock_context):
        """Yetkisiz kullanıcı testi"""
        # Yetkisiz kullanıcı ID
        mock_update.effective_user.id = 999999999
        
        await command_handler.start_command(mock_update, mock_context)
        
        # Unauthorized mesajı gönderilmeli
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        message_text = call_args[0][0]
        assert "Unauthorized Access" in message_text
    
    @pytest.mark.asyncio
    async def test_callback_query_handling(self, command_handler, mock_update, mock_context):
        """Callback query işleme testi"""
        # Mock callback query
        mock_update.callback_query = Mock()
        mock_update.callback_query.answer = AsyncMock()
        mock_update.callback_query.edit_message_text = AsyncMock()
        mock_update.callback_query.from_user.id = 123456789
        mock_update.callback_query.data = "refresh_status"
        
        await command_handler.handle_callback(mock_update, mock_context)
        
        # Callback answer çağrıldığını kontrol et
        mock_update.callback_query.answer.assert_called_once()
    
    def test_command_handler_performance(self, command_handler, mock_update, mock_context):
        """Command handler performans testi"""
        async def test_multiple_commands():
            with PerformanceTimer() as timer:
                # 50 komut çalıştır
                tasks = []
                for i in range(50):
                    task = command_handler.status_command(mock_update, mock_context)
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
            
            # 50 komut 5 saniyeden az sürmeli
            assert timer.elapsed < 5.0
        
        # Test çalıştır
        asyncio.run(test_multiple_commands())


class TestTelegramIntegration:
    """Telegram entegrasyon testleri"""
    
    @pytest.mark.asyncio
    async def test_full_message_flow(self, mock_telegram_bot):
        """Tam mesaj akışı testi"""
        with patch('bot.telegram_handler.Bot', return_value=mock_telegram_bot):
            from bot.telegram_handler import TelegramBot
            from bot.commands import CommandHandler
            
            # Mock config
            config = Mock()
            config.TELEGRAM_BOT_TOKEN = 'test_token'
            config.TELEGRAM_CHAT_ID = '123456789'
            config.TELEGRAM_ADMIN_IDS = ['123456789']
            
            # Bot ve handler oluştur
            telegram_bot = TelegramBot(config)
            command_handler = CommandHandler(telegram_bot)
            
            # Mock update ve context
            update = Mock()
            update.effective_user.id = 123456789
            update.effective_user.first_name = "Test User"
            update.message.reply_text = AsyncMock()
            
            context = Mock()
            context.args = []
            
            # Komut çalıştır
            await command_handler.start_command(update, context)
            
            # Doğrulama
            update.message.reply_text.assert_called_once()
    
    def test_telegram_error_handling(self, telegram_handler, mock_telegram_bot):
        """Telegram hata yönetimi testi"""
        # Mock exception
        mock_telegram_bot.send_message = AsyncMock(side_effect=Exception("Network error"))
        
        async def test_error():
            result = await telegram_handler.send_message("Test message")
            # Hata durumunda False dönmeli
            assert result == False
        
        # Test çalıştır
        asyncio.run(test_error())
    
    def test_telegram_rate_limit_integration(self, telegram_handler):
        """Telegram rate limit entegrasyon testi"""
        user_id = '123456789'
        
        # Rate limit'e kadar istek gönder
        for i in range(15):  # Limit 10
            result = telegram_handler.check_rate_limit(user_id)
            if i < 10:
                assert result == True
            else:
                assert result == False
    
    @pytest.mark.asyncio
    async def test_telegram_message_formatting(self, telegram_handler, mock_telegram_bot):
        """Telegram mesaj formatlama testi"""
        # Mock async send_message
        mock_telegram_bot.send_message = AsyncMock(return_value=Mock(message_id=123))
        
        # HTML formatında mesaj gönder
        html_message = "<b>Bold</b> and <i>italic</i> text"
        result = await telegram_handler.send_message(html_message, parse_mode='HTML')
        
        # Doğrulama
        assert result == True
        call_args = mock_telegram_bot.send_message.call_args
        assert call_args[1]['parse_mode'] == 'HTML'
        assert call_args[1]['text'] == html_message


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
