"""
Logger Setup
Logging konfigürasyonu ve yönetimi
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

def setup_logger(name: str = "trading_bot", 
                log_file: str = "logs/trading_bot.log",
                log_level: str = "INFO",
                max_bytes: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5) -> logging.Logger:
    """
    Logger'ı yapılandır
    
    Args:
        name: Logger adı
        log_file: Log dosya yolu
        log_level: Log seviyesi
        max_bytes: Maksimum dosya boyutu
        backup_count: Yedek dosya sayısı
        
    Returns:
        logging.Logger: Yapılandırılmış logger
    """
    
    # Log dizinini oluştur
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Logger oluştur
    logger = logging.getLogger(name)
    
    # Eğer logger zaten yapılandırılmışsa, mevcut handler'ları temizle
    if logger.handlers:
        logger.handlers.clear()
    
    # Log seviyesini ayarla
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logger.setLevel(log_level_map.get(log_level.upper(), logging.INFO))
    
    # Formatter oluştur
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (rotating)
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Dosya handler oluşturulamadı: {e}")
    
    # Error handler (ayrı dosya)
    try:
        error_file = log_file.replace('.log', '_error.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    except Exception as e:
        logger.warning(f"Error handler oluşturulamadı: {e}")
    
    # İlk log mesajı
    logger.info(f"Logger başlatıldı - Seviye: {log_level}")
    
    return logger

class TradingLogger:
    """Trading bot için özelleştirilmiş logger sınıfı"""
    
    def __init__(self, base_logger: Optional[logging.Logger] = None):
        """
        Trading logger başlatıcı
        
        Args:
            base_logger: Temel logger (opsiyonel)
        """
        self.logger = base_logger or setup_logger()
        
        # Trading-specific log dosyaları
        self.trade_logger = self._setup_trade_logger()
        self.signal_logger = self._setup_signal_logger()
        self.risk_logger = self._setup_risk_logger()
        self.performance_logger = self._setup_performance_logger()
    
    def _setup_trade_logger(self) -> logging.Logger:
        """Trade işlemleri için özel logger"""
        trade_logger = logging.getLogger("trading_bot.trades")
        
        if not trade_logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - TRADE - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            try:
                handler = logging.handlers.RotatingFileHandler(
                    'logs/trades.log',
                    maxBytes=5 * 1024 * 1024,  # 5MB
                    backupCount=10,
                    encoding='utf-8'
                )
                handler.setFormatter(formatter)
                trade_logger.addHandler(handler)
                trade_logger.setLevel(logging.INFO)
            except Exception as e:
                self.logger.warning(f"Trade logger oluşturulamadı: {e}")
        
        return trade_logger
    
    def _setup_signal_logger(self) -> logging.Logger:
        """Sinyal işlemleri için özel logger"""
        signal_logger = logging.getLogger("trading_bot.signals")
        
        if not signal_logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - SIGNAL - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            try:
                handler = logging.handlers.RotatingFileHandler(
                    'logs/signals.log',
                    maxBytes=5 * 1024 * 1024,  # 5MB
                    backupCount=5,
                    encoding='utf-8'
                )
                handler.setFormatter(formatter)
                signal_logger.addHandler(handler)
                signal_logger.setLevel(logging.INFO)
            except Exception as e:
                self.logger.warning(f"Signal logger oluşturulamadı: {e}")
        
        return signal_logger
    
    def _setup_risk_logger(self) -> logging.Logger:
        """Risk yönetimi için özel logger"""
        risk_logger = logging.getLogger("trading_bot.risk")
        
        if not risk_logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - RISK - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            try:
                handler = logging.handlers.RotatingFileHandler(
                    'logs/risk.log',
                    maxBytes=5 * 1024 * 1024,  # 5MB
                    backupCount=10,
                    encoding='utf-8'
                )
                handler.setFormatter(formatter)
                risk_logger.addHandler(handler)
                risk_logger.setLevel(logging.WARNING)
            except Exception as e:
                self.logger.warning(f"Risk logger oluşturulamadı: {e}")
        
        return risk_logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Performans metrikleri için özel logger"""
        perf_logger = logging.getLogger("trading_bot.performance")
        
        if not perf_logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - PERFORMANCE - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            try:
                handler = logging.handlers.RotatingFileHandler(
                    'logs/performance.log',
                    maxBytes=5 * 1024 * 1024,  # 5MB
                    backupCount=5,
                    encoding='utf-8'
                )
                handler.setFormatter(formatter)
                perf_logger.addHandler(handler)
                perf_logger.setLevel(logging.INFO)
            except Exception as e:
                self.logger.warning(f"Performance logger oluşturulamadı: {e}")
        
        return perf_logger
    
    def log_trade(self, symbol: str, side: str, quantity: float, 
                  price: float, order_type: str, status: str, **kwargs):
        """
        Trade işlemini logla
        
        Args:
            symbol: Trading pair
            side: BUY/SELL
            quantity: Miktar
            price: Fiyat
            order_type: Emir tipi
            status: Durum
            **kwargs: Ek bilgiler
        """
        try:
            extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            message = f"{symbol} | {side} | {quantity} | {price} | {order_type} | {status}"
            if extra_info:
                message += f" | {extra_info}"
            
            self.trade_logger.info(message)
            
        except Exception as e:
            self.logger.error(f"Trade log hatası: {e}")
    
    def log_signal(self, symbol: str, signal_type: str, strength: float, 
                   model: str, confidence: float = None, **kwargs):
        """
        Sinyal işlemini logla
        
        Args:
            symbol: Trading pair
            signal_type: Sinyal tipi (BUY/SELL/HOLD)
            strength: Sinyal gücü
            model: Model adı
            confidence: Güven seviyesi
            **kwargs: Ek bilgiler
        """
        try:
            message = f"{symbol} | {signal_type} | {strength:.3f} | {model}"
            if confidence is not None:
                message += f" | confidence={confidence:.3f}"
            
            extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            if extra_info:
                message += f" | {extra_info}"
            
            self.signal_logger.info(message)
            
        except Exception as e:
            self.logger.error(f"Signal log hatası: {e}")
    
    def log_risk_event(self, event_type: str, level: str, message: str, 
                       symbol: str = None, value: float = None):
        """
        Risk olayını logla
        
        Args:
            event_type: Olay tipi
            level: Risk seviyesi
            message: Mesaj
            symbol: Trading pair (opsiyonel)
            value: Değer (opsiyonel)
        """
        try:
            log_message = f"{event_type} | {level} | {message}"
            if symbol:
                log_message += f" | symbol={symbol}"
            if value is not None:
                log_message += f" | value={value}"
            
            # Risk seviyesine göre log level belirle
            if level in ['CRITICAL', 'HIGH']:
                self.risk_logger.error(log_message)
            elif level == 'MEDIUM':
                self.risk_logger.warning(log_message)
            else:
                self.risk_logger.info(log_message)
                
        except Exception as e:
            self.logger.error(f"Risk log hatası: {e}")
    
    def log_performance(self, period: str, pnl: float, trades: int, 
                       win_rate: float, **kwargs):
        """
        Performans metriklerini logla
        
        Args:
            period: Dönem (daily, weekly, monthly)
            pnl: Kar/zarar
            trades: Trade sayısı
            win_rate: Kazanma oranı
            **kwargs: Ek metrikler
        """
        try:
            message = f"{period} | PnL={pnl:.2f} | Trades={trades} | WinRate={win_rate:.1f}%"
            
            extra_metrics = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            if extra_metrics:
                message += f" | {extra_metrics}"
            
            self.performance_logger.info(message)
            
        except Exception as e:
            self.logger.error(f"Performance log hatası: {e}")
    
    def log_model_performance(self, model_name: str, symbol: str, 
                             mse: float, mae: float, direction_accuracy: float):
        """
        Model performansını logla
        
        Args:
            model_name: Model adı
            symbol: Trading pair
            mse: Mean Squared Error
            mae: Mean Absolute Error
            direction_accuracy: Yön doğruluğu
        """
        try:
            message = (f"MODEL_EVAL | {model_name} | {symbol} | "
                      f"MSE={mse:.6f} | MAE={mae:.6f} | "
                      f"DirectionAcc={direction_accuracy:.3f}")
            
            self.performance_logger.info(message)
            
        except Exception as e:
            self.logger.error(f"Model performance log hatası: {e}")
    
    def log_system_health(self, component: str, status: str, **metrics):
        """
        Sistem sağlığını logla
        
        Args:
            component: Bileşen adı
            status: Durum (OK, WARNING, ERROR)
            **metrics: Sistem metrikleri
        """
        try:
            message = f"HEALTH | {component} | {status}"
            
            if metrics:
                metric_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
                message += f" | {metric_str}"
            
            if status == "ERROR":
                self.logger.error(message)
            elif status == "WARNING":
                self.logger.warning(message)
            else:
                self.logger.info(message)
                
        except Exception as e:
            self.logger.error(f"System health log hatası: {e}")
    
    def get_recent_logs(self, log_type: str = "main", lines: int = 50) -> list:
        """
        Son log kayıtlarını al
        
        Args:
            log_type: Log tipi (main, trades, signals, risk, performance)
            lines: Satır sayısı
            
        Returns:
            list: Log satırları
        """
        try:
            log_files = {
                "main": "logs/trading_bot.log",
                "trades": "logs/trades.log",
                "signals": "logs/signals.log",
                "risk": "logs/risk.log",
                "performance": "logs/performance.log"
            }
            
            log_file = log_files.get(log_type, "logs/trading_bot.log")
            
            if not os.path.exists(log_file):
                return []
            
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
                
        except Exception as e:
            self.logger.error(f"Log okuma hatası: {e}")
            return []
    
    def cleanup_old_logs(self, days: int = 30):
        """
        Eski log dosyalarını temizle
        
        Args:
            days: Tutulacak gün sayısı
        """
        try:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                return
            
            cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            for filename in os.listdir(log_dir):
                if filename.endswith('.log'):
                    file_path = os.path.join(log_dir, filename)
                    if os.path.getmtime(file_path) < cutoff_time:
                        try:
                            os.remove(file_path)
                            self.logger.info(f"Eski log dosyası silindi: {filename}")
                        except Exception as e:
                            self.logger.warning(f"Log dosyası silinemedi {filename}: {e}")
                            
        except Exception as e:
            self.logger.error(f"Log temizleme hatası: {e}")

# Global trading logger instance
_trading_logger = None

def get_trading_logger() -> TradingLogger:
    """Global trading logger instance'ını al"""
    global _trading_logger
    if _trading_logger is None:
        _trading_logger = TradingLogger()
    return _trading_logger

def log_trade(*args, **kwargs):
    """Kısayol: Trade logla"""
    get_trading_logger().log_trade(*args, **kwargs)

def log_signal(*args, **kwargs):
    """Kısayol: Sinyal logla"""
    get_trading_logger().log_signal(*args, **kwargs)

def log_risk_event(*args, **kwargs):
    """Kısayol: Risk olayı logla"""
    get_trading_logger().log_risk_event(*args, **kwargs)

def log_performance(*args, **kwargs):
    """Kısayol: Performans logla"""
    get_trading_logger().log_performance(*args, **kwargs)
