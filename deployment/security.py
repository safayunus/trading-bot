"""
Basic Security Module
Basit güvenlik kontrolleri ve input validation
"""

import re
import os
import logging
from typing import List, Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)

class SecurityValidator:
    """Basit güvenlik doğrulama sınıfı"""
    
    @staticmethod
    def validate_telegram_chat_id(chat_id: str) -> bool:
        """
        Telegram chat ID doğrulama
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            bool: Geçerli mi
        """
        try:
            # Chat ID sayısal olmalı ve makul uzunlukta
            chat_id_int = int(chat_id)
            return -999999999999 <= chat_id_int <= 999999999999
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        API key format doğrulama
        
        Args:
            api_key: API anahtarı
            
        Returns:
            bool: Geçerli mi
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Binance API key formatı: 64 karakter, alphanumeric
        if len(api_key) == 64 and re.match(r'^[A-Za-z0-9]+$', api_key):
            return True
        
        return False
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Trading symbol doğrulama
        
        Args:
            symbol: Trading pair (örn: BTCUSDT)
            
        Returns:
            bool: Geçerli mi
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Symbol formatı: 3-12 karakter, büyük harf, alphanumeric
        symbol = symbol.upper()
        if 3 <= len(symbol) <= 12 and re.match(r'^[A-Z0-9]+$', symbol):
            return True
        
        return False
    
    @staticmethod
    def validate_amount(amount: Any) -> bool:
        """
        Trade miktarı doğrulama
        
        Args:
            amount: Trade miktarı
            
        Returns:
            bool: Geçerli mi
        """
        try:
            amount_float = float(amount)
            # Pozitif ve makul aralıkta olmalı
            return 0.000001 <= amount_float <= 1000000
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_price(price: Any) -> bool:
        """
        Fiyat doğrulama
        
        Args:
            price: Fiyat
            
        Returns:
            bool: Geçerli mi
        """
        try:
            price_float = float(price)
            # Pozitif ve makul aralıkta olmalı
            return 0.000001 <= price_float <= 10000000
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def sanitize_input(input_str: str, max_length: int = 100) -> str:
        """
        Input temizleme
        
        Args:
            input_str: Temizlenecek string
            max_length: Maksimum uzunluk
            
        Returns:
            str: Temizlenmiş string
        """
        if not isinstance(input_str, str):
            return ""
        
        # Tehlikeli karakterleri kaldır
        sanitized = re.sub(r'[<>"\';\\]', '', input_str)
        
        # Uzunluk sınırla
        sanitized = sanitized[:max_length]
        
        # Boşlukları temizle
        sanitized = sanitized.strip()
        
        return sanitized
    
    @staticmethod
    def validate_env_vars() -> List[str]:
        """
        Gerekli environment variable'ları kontrol et
        
        Returns:
            List[str]: Eksik değişkenler listesi
        """
        required_vars = [
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID',
            'BINANCE_API_KEY',
            'BINANCE_SECRET_KEY'
        ]
        
        missing_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value or value.strip() == '' or 'your_' in value.lower():
                missing_vars.append(var)
        
        return missing_vars

class RateLimiter:
    """Basit rate limiting"""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Rate limiter başlatıcı
        
        Args:
            max_requests: Maksimum istek sayısı
            window_seconds: Zaman penceresi (saniye)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, user_id: str) -> bool:
        """
        İstek izni kontrol et
        
        Args:
            user_id: Kullanıcı ID
            
        Returns:
            bool: İzin var mı
        """
        import time
        
        current_time = time.time()
        
        # Kullanıcının isteklerini al
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        user_requests = self.requests[user_id]
        
        # Eski istekleri temizle
        user_requests[:] = [req_time for req_time in user_requests 
                           if current_time - req_time < self.window_seconds]
        
        # Limit kontrolü
        if len(user_requests) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False
        
        # Yeni isteği ekle
        user_requests.append(current_time)
        return True

def require_auth(authorized_users: List[str]):
    """
    Authentication decorator
    
    Args:
        authorized_users: Yetkili kullanıcı listesi
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(update, context, *args, **kwargs):
            user_id = str(update.effective_user.id)
            
            if user_id not in authorized_users:
                logger.warning(f"Unauthorized access attempt from user {user_id}")
                await update.message.reply_text(
                    "❌ Unauthorized access. Contact administrator."
                )
                return
            
            return await func(update, context, *args, **kwargs)
        return wrapper
    return decorator

def validate_trading_input(func):
    """
    Trading input validation decorator
    """
    @wraps(func)
    async def wrapper(update, context, *args, **kwargs):
        # Temel input doğrulama
        if context.args:
            for arg in context.args:
                # Tehlikeli karakterleri kontrol et
                if re.search(r'[<>"\';\\]', str(arg)):
                    await update.message.reply_text(
                        "❌ Invalid characters in input."
                    )
                    return
                
                # Uzunluk kontrolü
                if len(str(arg)) > 50:
                    await update.message.reply_text(
                        "❌ Input too long."
                    )
                    return
        
        return await func(update, context, *args, **kwargs)
    return wrapper

class SecurityConfig:
    """Güvenlik konfigürasyonu"""
    
    def __init__(self):
        """Güvenlik ayarlarını yükle"""
        self.rate_limit_enabled = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
        self.rate_limit_requests = int(os.getenv('RATE_LIMIT_REQUESTS', '10'))
        self.rate_limit_window = int(os.getenv('RATE_LIMIT_WINDOW', '60'))
        self.input_validation = os.getenv('INPUT_VALIDATION', 'true').lower() == 'true'
        
        # Authorized users
        self.authorized_users = self._load_authorized_users()
        self.admin_users = self._load_admin_users()
    
    def _load_authorized_users(self) -> List[str]:
        """Yetkili kullanıcıları yükle"""
        chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        admin_ids = os.getenv('TELEGRAM_ADMIN_IDS', '')
        
        users = []
        
        # Ana chat ID
        if chat_id and SecurityValidator.validate_telegram_chat_id(chat_id):
            users.append(chat_id)
        
        # Admin ID'ler
        if admin_ids:
            for admin_id in admin_ids.split(','):
                admin_id = admin_id.strip()
                if SecurityValidator.validate_telegram_chat_id(admin_id):
                    users.append(admin_id)
        
        return list(set(users))  # Duplicate'leri kaldır
    
    def _load_admin_users(self) -> List[str]:
        """Admin kullanıcıları yükle"""
        admin_ids = os.getenv('TELEGRAM_ADMIN_IDS', '')
        
        admins = []
        if admin_ids:
            for admin_id in admin_ids.split(','):
                admin_id = admin_id.strip()
                if SecurityValidator.validate_telegram_chat_id(admin_id):
                    admins.append(admin_id)
        
        return admins
    
    def is_authorized_user(self, user_id: str) -> bool:
        """Kullanıcı yetkili mi kontrol et"""
        return str(user_id) in self.authorized_users
    
    def is_admin_user(self, user_id: str) -> bool:
        """Kullanıcı admin mi kontrol et"""
        return str(user_id) in self.admin_users
    
    def validate_environment(self) -> bool:
        """Environment değişkenlerini doğrula"""
        missing_vars = SecurityValidator.validate_env_vars()
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return False
        
        # API key formatlarını kontrol et
        api_key = os.getenv('BINANCE_API_KEY', '')
        if not SecurityValidator.validate_api_key(api_key):
            logger.error("Invalid Binance API key format")
            return False
        
        return True

# Global security instance
security_config = SecurityConfig()
rate_limiter = RateLimiter(
    max_requests=security_config.rate_limit_requests,
    window_seconds=security_config.rate_limit_window
)

def check_security_setup() -> bool:
    """
    Güvenlik kurulumunu kontrol et
    
    Returns:
        bool: Güvenlik kurulumu tamam mı
    """
    try:
        # Environment validation
        if not security_config.validate_environment():
            return False
        
        # Authorized users kontrolü
        if not security_config.authorized_users:
            logger.error("No authorized users configured")
            return False
        
        logger.info(f"Security setup OK. Authorized users: {len(security_config.authorized_users)}")
        return True
        
    except Exception as e:
        logger.error(f"Security setup error: {e}")
        return False
