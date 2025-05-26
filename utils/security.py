"""
Security Manager
Güvenlik yönetimi ve şifreleme işlemleri
"""

import os
import hashlib
import hmac
import base64
import secrets
import logging
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import json

class SecurityManager:
    """Güvenlik yönetici sınıfı"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Security manager başlatıcı
        
        Args:
            encryption_key: Şifreleme anahtarı (opsiyonel)
        """
        self.logger = logging.getLogger(__name__)
        self.encryption_key = encryption_key
        self.fernet = None
        
        if encryption_key:
            self._initialize_encryption(encryption_key)
    
    def _initialize_encryption(self, key: str):
        """
        Şifreleme sistemini başlat
        
        Args:
            key: Şifreleme anahtarı
        """
        try:
            # Key'i Fernet formatına dönüştür
            if len(key) != 44:  # Fernet key 44 karakter olmalı
                # Password'dan key türet
                password = key.encode()
                salt = b'trading_bot_salt'  # Gerçek uygulamada rastgele salt kullanın
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key_bytes = kdf.derive(password)
                fernet_key = base64.urlsafe_b64encode(key_bytes)
            else:
                fernet_key = key.encode()
            
            self.fernet = Fernet(fernet_key)
            self.logger.info("Şifreleme sistemi başlatıldı")
            
        except Exception as e:
            self.logger.error(f"Şifreleme başlatma hatası: {e}")
            self.fernet = None
    
    def generate_key(self) -> str:
        """
        Yeni şifreleme anahtarı oluştur
        
        Returns:
            str: Base64 encoded şifreleme anahtarı
        """
        try:
            key = Fernet.generate_key()
            return key.decode()
        except Exception as e:
            self.logger.error(f"Anahtar oluşturma hatası: {e}")
            return ""
    
    def encrypt_data(self, data: str) -> Optional[str]:
        """
        Veriyi şifrele
        
        Args:
            data: Şifrelenecek veri
            
        Returns:
            str: Şifrelenmiş veri (base64 encoded)
        """
        try:
            if not self.fernet:
                self.logger.warning("Şifreleme sistemi başlatılmamış")
                return None
            
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
            
        except Exception as e:
            self.logger.error(f"Şifreleme hatası: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: str) -> Optional[str]:
        """
        Veriyi çöz
        
        Args:
            encrypted_data: Şifrelenmiş veri (base64 encoded)
            
        Returns:
            str: Çözülmüş veri
        """
        try:
            if not self.fernet:
                self.logger.warning("Şifreleme sistemi başlatılmamış")
                return None
            
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error(f"Çözme hatası: {e}")
            return None
    
    def encrypt_config(self, config_data: Dict[str, Any]) -> Optional[str]:
        """
        Konfigürasyon verilerini şifrele
        
        Args:
            config_data: Konfigürasyon verileri
            
        Returns:
            str: Şifrelenmiş konfigürasyon
        """
        try:
            json_data = json.dumps(config_data)
            return self.encrypt_data(json_data)
        except Exception as e:
            self.logger.error(f"Konfigürasyon şifreleme hatası: {e}")
            return None
    
    def decrypt_config(self, encrypted_config: str) -> Optional[Dict[str, Any]]:
        """
        Şifrelenmiş konfigürasyonu çöz
        
        Args:
            encrypted_config: Şifrelenmiş konfigürasyon
            
        Returns:
            Dict: Konfigürasyon verileri
        """
        try:
            decrypted_data = self.decrypt_data(encrypted_config)
            if decrypted_data:
                return json.loads(decrypted_data)
            return None
        except Exception as e:
            self.logger.error(f"Konfigürasyon çözme hatası: {e}")
            return None
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """
        Şifreyi hash'le
        
        Args:
            password: Şifre
            salt: Salt (opsiyonel)
            
        Returns:
            Dict: Hash ve salt
        """
        try:
            if not salt:
                salt = secrets.token_hex(32)
            
            # PBKDF2 ile hash
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000  # iterations
            )
            
            return {
                'hash': password_hash.hex(),
                'salt': salt
            }
            
        except Exception as e:
            self.logger.error(f"Şifre hash hatası: {e}")
            return {'hash': '', 'salt': ''}
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """
        Şifreyi doğrula
        
        Args:
            password: Kontrol edilecek şifre
            stored_hash: Saklanan hash
            salt: Salt
            
        Returns:
            bool: Doğrulama sonucu
        """
        try:
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            
            return hmac.compare_digest(password_hash.hex(), stored_hash)
            
        except Exception as e:
            self.logger.error(f"Şifre doğrulama hatası: {e}")
            return False
    
    def generate_api_signature(self, secret_key: str, message: str) -> str:
        """
        API imzası oluştur
        
        Args:
            secret_key: Gizli anahtar
            message: İmzalanacak mesaj
            
        Returns:
            str: HMAC imzası
        """
        try:
            signature = hmac.new(
                secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            self.logger.error(f"API imza hatası: {e}")
            return ""
    
    def verify_api_signature(self, secret_key: str, message: str, signature: str) -> bool:
        """
        API imzasını doğrula
        
        Args:
            secret_key: Gizli anahtar
            message: Orijinal mesaj
            signature: Doğrulanacak imza
            
        Returns:
            bool: Doğrulama sonucu
        """
        try:
            expected_signature = self.generate_api_signature(secret_key, message)
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"API imza doğrulama hatası: {e}")
            return False
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Güvenli token oluştur
        
        Args:
            length: Token uzunluğu
            
        Returns:
            str: Güvenli token
        """
        try:
            return secrets.token_urlsafe(length)
        except Exception as e:
            self.logger.error(f"Token oluşturma hatası: {e}")
            return ""
    
    def mask_sensitive_data(self, data: str, visible_chars: int = 4) -> str:
        """
        Hassas veriyi maskele
        
        Args:
            data: Maskelenecek veri
            visible_chars: Görünür karakter sayısı
            
        Returns:
            str: Maskelenmiş veri
        """
        try:
            if len(data) <= visible_chars * 2:
                return "*" * len(data)
            
            start = data[:visible_chars]
            end = data[-visible_chars:]
            middle = "*" * (len(data) - visible_chars * 2)
            
            return f"{start}{middle}{end}"
            
        except Exception as e:
            self.logger.error(f"Maskeleme hatası: {e}")
            return "***"
    
    def sanitize_input(self, input_data: str) -> str:
        """
        Girdi verilerini temizle
        
        Args:
            input_data: Temizlenecek veri
            
        Returns:
            str: Temizlenmiş veri
        """
        try:
            # Tehlikeli karakterleri kaldır
            dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\n', '\r', '\t']
            
            sanitized = input_data
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            
            # Uzunluk sınırı
            max_length = 1000
            if len(sanitized) > max_length:
                sanitized = sanitized[:max_length]
            
            return sanitized.strip()
            
        except Exception as e:
            self.logger.error(f"Girdi temizleme hatası: {e}")
            return ""
    
    def validate_api_key_format(self, api_key: str) -> bool:
        """
        API key formatını doğrula
        
        Args:
            api_key: API anahtarı
            
        Returns:
            bool: Format doğruluğu
        """
        try:
            # Temel format kontrolleri
            if not api_key or len(api_key) < 16:
                return False
            
            # Sadece alfanumerik ve belirli özel karakterler
            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
            if not all(c in allowed_chars for c in api_key):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"API key doğrulama hatası: {e}")
            return False
    
    def check_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Şifre gücünü kontrol et
        
        Args:
            password: Kontrol edilecek şifre
            
        Returns:
            Dict: Şifre gücü analizi
        """
        try:
            score = 0
            feedback = []
            
            # Uzunluk kontrolü
            if len(password) >= 8:
                score += 1
            else:
                feedback.append("En az 8 karakter olmalı")
            
            if len(password) >= 12:
                score += 1
            
            # Karakter çeşitliliği
            has_lower = any(c.islower() for c in password)
            has_upper = any(c.isupper() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
            
            if has_lower:
                score += 1
            else:
                feedback.append("Küçük harf içermeli")
            
            if has_upper:
                score += 1
            else:
                feedback.append("Büyük harf içermeli")
            
            if has_digit:
                score += 1
            else:
                feedback.append("Rakam içermeli")
            
            if has_special:
                score += 1
            else:
                feedback.append("Özel karakter içermeli")
            
            # Güç seviyesi
            if score >= 6:
                strength = "Güçlü"
            elif score >= 4:
                strength = "Orta"
            else:
                strength = "Zayıf"
            
            return {
                'score': score,
                'max_score': 6,
                'strength': strength,
                'feedback': feedback
            }
            
        except Exception as e:
            self.logger.error(f"Şifre güç kontrolü hatası: {e}")
            return {
                'score': 0,
                'max_score': 6,
                'strength': "Bilinmiyor",
                'feedback': ["Kontrol edilemedi"]
            }
    
    def secure_delete_file(self, file_path: str) -> bool:
        """
        Dosyayı güvenli şekilde sil
        
        Args:
            file_path: Silinecek dosya yolu
            
        Returns:
            bool: Silme başarısı
        """
        try:
            if not os.path.exists(file_path):
                return True
            
            # Dosyayı rastgele verilerle üzerine yaz
            file_size = os.path.getsize(file_path)
            
            with open(file_path, 'r+b') as file:
                for _ in range(3):  # 3 kez üzerine yaz
                    file.seek(0)
                    file.write(os.urandom(file_size))
                    file.flush()
                    os.fsync(file.fileno())
            
            # Dosyayı sil
            os.remove(file_path)
            
            self.logger.info(f"Dosya güvenli şekilde silindi: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Güvenli dosya silme hatası: {e}")
            return False
    
    def get_security_headers(self) -> Dict[str, str]:
        """
        Güvenlik başlıklarını al
        
        Returns:
            Dict: HTTP güvenlik başlıkları
        """
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """
        Güvenlik olayını logla
        
        Args:
            event_type: Olay tipi
            details: Olay detayları
        """
        try:
            # Hassas bilgileri maskele
            safe_details = {}
            for key, value in details.items():
                if key.lower() in ['password', 'token', 'key', 'secret']:
                    safe_details[key] = self.mask_sensitive_data(str(value))
                else:
                    safe_details[key] = value
            
            self.logger.warning(f"SECURITY_EVENT: {event_type} - {safe_details}")
            
        except Exception as e:
            self.logger.error(f"Güvenlik olay loglama hatası: {e}")

# Global security manager instance
_security_manager = None

def get_security_manager(encryption_key: Optional[str] = None) -> SecurityManager:
    """Global security manager instance'ını al"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager(encryption_key)
    return _security_manager

def encrypt_sensitive_config(config_dict: Dict[str, Any], 
                           encryption_key: str) -> Dict[str, Any]:
    """
    Hassas konfigürasyon verilerini şifrele
    
    Args:
        config_dict: Konfigürasyon sözlüğü
        encryption_key: Şifreleme anahtarı
        
    Returns:
        Dict: Şifrelenmiş konfigürasyon
    """
    security_manager = SecurityManager(encryption_key)
    
    sensitive_keys = [
        'TELEGRAM_TOKEN', 'BINANCE_API_KEY', 'BINANCE_SECRET_KEY',
        'ENCRYPTION_KEY', 'EMAIL_PASSWORD', 'WEBHOOK_SECRET'
    ]
    
    encrypted_config = config_dict.copy()
    
    for key in sensitive_keys:
        if key in encrypted_config:
            encrypted_value = security_manager.encrypt_data(str(encrypted_config[key]))
            if encrypted_value:
                encrypted_config[key] = encrypted_value
    
    return encrypted_config

def decrypt_sensitive_config(encrypted_config: Dict[str, Any], 
                           encryption_key: str) -> Dict[str, Any]:
    """
    Şifrelenmiş konfigürasyon verilerini çöz
    
    Args:
        encrypted_config: Şifrelenmiş konfigürasyon
        encryption_key: Şifreleme anahtarı
        
    Returns:
        Dict: Çözülmüş konfigürasyon
    """
    security_manager = SecurityManager(encryption_key)
    
    sensitive_keys = [
        'TELEGRAM_TOKEN', 'BINANCE_API_KEY', 'BINANCE_SECRET_KEY',
        'ENCRYPTION_KEY', 'EMAIL_PASSWORD', 'WEBHOOK_SECRET'
    ]
    
    decrypted_config = encrypted_config.copy()
    
    for key in sensitive_keys:
        if key in decrypted_config:
            decrypted_value = security_manager.decrypt_data(str(decrypted_config[key]))
            if decrypted_value:
                decrypted_config[key] = decrypted_value
    
    return decrypted_config
