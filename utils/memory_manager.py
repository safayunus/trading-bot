"""
Memory Manager
Memory yönetimi ve optimizasyon araçları
"""

import gc
import psutil
import logging
import threading
import time
import weakref
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import sys
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MemoryStats:
    """Memory istatistikleri"""
    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    process_mb: float
    timestamp: datetime


class MemoryManager:
    """Memory yönetim sistemi"""
    
    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0):
        """
        Memory manager başlatıcı
        
        Args:
            warning_threshold: Uyarı eşiği (%)
            critical_threshold: Kritik eşik (%)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.memory_history: List[MemoryStats] = []
        self.max_history = 1000
        
        # Object tracking
        self.tracked_objects: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)
        self.object_counts: Dict[str, int] = defaultdict(int)
        
        # Cleanup callbacks
        self.cleanup_callbacks: List[Callable] = []
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 30  # seconds
    
    def get_memory_stats(self) -> MemoryStats:
        """Mevcut memory istatistiklerini al"""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        stats = MemoryStats(
            total_mb=system_memory.total / 1024 / 1024,
            available_mb=system_memory.available / 1024 / 1024,
            used_mb=system_memory.used / 1024 / 1024,
            percent=system_memory.percent,
            process_mb=process_memory,
            timestamp=datetime.now()
        )
        
        # History'ye ekle
        self.memory_history.append(stats)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)
        
        return stats
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Memory kullanımını kontrol et"""
        stats = self.get_memory_stats()
        
        status = {
            'stats': stats,
            'warning': False,
            'critical': False,
            'action_taken': False,
            'recommendations': []
        }
        
        # Threshold kontrolü
        if stats.percent >= self.critical_threshold:
            status['critical'] = True
            status['recommendations'].append('Immediate memory cleanup required')
            
            # Otomatik cleanup
            self.emergency_cleanup()
            status['action_taken'] = True
            
        elif stats.percent >= self.warning_threshold:
            status['warning'] = True
            status['recommendations'].append('Consider running memory cleanup')
            
            # Soft cleanup
            self.soft_cleanup()
            status['action_taken'] = True
        
        # Log
        if status['critical']:
            self.logger.critical(f"Critical memory usage: {stats.percent:.1f}%")
        elif status['warning']:
            self.logger.warning(f"High memory usage: {stats.percent:.1f}%")
        
        return status
    
    def soft_cleanup(self) -> Dict[str, Any]:
        """Yumuşak memory temizliği"""
        self.logger.info("Starting soft memory cleanup...")
        
        initial_stats = self.get_memory_stats()
        
        # Garbage collection
        collected = gc.collect()
        
        # Cache temizliği (eğer varsa)
        for callback in self.cleanup_callbacks:
            try:
                callback('soft')
            except Exception as e:
                self.logger.error(f"Cleanup callback failed: {e}")
        
        final_stats = self.get_memory_stats()
        
        result = {
            'type': 'soft',
            'initial_usage': initial_stats.percent,
            'final_usage': final_stats.percent,
            'memory_freed_mb': initial_stats.process_mb - final_stats.process_mb,
            'objects_collected': collected,
            'timestamp': datetime.now()
        }
        
        self.logger.info(
            f"Soft cleanup completed. "
            f"Memory usage: {initial_stats.percent:.1f}% -> {final_stats.percent:.1f}%"
        )
        
        return result
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """Acil durum memory temizliği"""
        self.logger.warning("Starting emergency memory cleanup...")
        
        initial_stats = self.get_memory_stats()
        
        # Aggressive garbage collection
        for _ in range(3):
            collected = gc.collect()
        
        # Force cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback('emergency')
            except Exception as e:
                self.logger.error(f"Emergency cleanup callback failed: {e}")
        
        # Clear tracked objects
        self.clear_tracked_objects()
        
        # Force garbage collection again
        gc.collect()
        
        final_stats = self.get_memory_stats()
        
        result = {
            'type': 'emergency',
            'initial_usage': initial_stats.percent,
            'final_usage': final_stats.percent,
            'memory_freed_mb': initial_stats.process_mb - final_stats.process_mb,
            'timestamp': datetime.now()
        }
        
        self.logger.warning(
            f"Emergency cleanup completed. "
            f"Memory usage: {initial_stats.percent:.1f}% -> {final_stats.percent:.1f}%"
        )
        
        return result
    
    def track_object(self, obj: Any, category: str = 'general') -> None:
        """Object'i takip et"""
        try:
            self.tracked_objects[category].add(obj)
            self.object_counts[category] += 1
        except TypeError:
            # Weak reference oluşturulamayan objeler için
            pass
    
    def get_tracked_objects_count(self) -> Dict[str, int]:
        """Takip edilen object sayılarını al"""
        current_counts = {}
        for category, weak_set in self.tracked_objects.items():
            current_counts[category] = len(weak_set)
        return current_counts
    
    def clear_tracked_objects(self) -> int:
        """Takip edilen objeleri temizle"""
        total_cleared = 0
        for category in self.tracked_objects:
            count = len(self.tracked_objects[category])
            self.tracked_objects[category].clear()
            total_cleared += count
        
        self.object_counts.clear()
        return total_cleared
    
    def register_cleanup_callback(self, callback: Callable) -> None:
        """Cleanup callback kaydet"""
        self.cleanup_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Memory monitoring başlat"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Memory monitoring durdur"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Memory monitoring döngüsü"""
        while self.monitoring:
            try:
                self.check_memory_usage()
                time.sleep(self.monitor_interval)
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.monitor_interval)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Kapsamlı memory raporu"""
        current_stats = self.get_memory_stats()
        
        # Son 1 saatlik history
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_history = [
            stats for stats in self.memory_history 
            if stats.timestamp >= one_hour_ago
        ]
        
        # Trend analizi
        if len(recent_history) >= 2:
            trend = recent_history[-1].percent - recent_history[0].percent
        else:
            trend = 0
        
        # Peak usage
        peak_usage = max(stats.percent for stats in recent_history) if recent_history else current_stats.percent
        
        return {
            'current_stats': current_stats,
            'trend_1h': trend,
            'peak_usage_1h': peak_usage,
            'tracked_objects': self.get_tracked_objects_count(),
            'history_count': len(self.memory_history),
            'monitoring_active': self.monitoring,
            'thresholds': {
                'warning': self.warning_threshold,
                'critical': self.critical_threshold
            }
        }


class ErrorRecoveryManager:
    """Hata kurtarma yöneticisi"""
    
    def __init__(self):
        """Error recovery manager başlatıcı"""
        self.logger = logging.getLogger(__name__)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # Recovery statistics
        self.recovery_stats = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'last_error': None
        }
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable) -> None:
        """
        Kurtarma stratejisi kaydet
        
        Args:
            error_type: Hata türü (exception class name)
            strategy: Kurtarma fonksiyonu
        """
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Recovery strategy registered for {error_type}")
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """
        Hatayı işle ve kurtarma dene
        
        Args:
            error: Exception objesi
            context: Hata context bilgisi
            
        Returns:
            bool: Kurtarma başarılı mı
        """
        error_type = type(error).__name__
        error_info = {
            'error_type': error_type,
            'error_message': str(error),
            'context': context or {},
            'timestamp': datetime.now(),
            'recovery_attempted': False,
            'recovery_successful': False
        }
        
        self.recovery_stats['total_errors'] += 1
        self.recovery_stats['last_error'] = error_info
        
        self.logger.error(f"Handling error: {error_type} - {error}")
        
        # Kurtarma stratejisi var mı kontrol et
        if error_type in self.recovery_strategies:
            try:
                error_info['recovery_attempted'] = True
                strategy = self.recovery_strategies[error_type]
                
                # Kurtarma stratejisini çalıştır
                recovery_result = strategy(error, context)
                
                if recovery_result:
                    error_info['recovery_successful'] = True
                    self.recovery_stats['successful_recoveries'] += 1
                    self.logger.info(f"Successfully recovered from {error_type}")
                else:
                    self.recovery_stats['failed_recoveries'] += 1
                    self.logger.warning(f"Recovery failed for {error_type}")
                
                return recovery_result
                
            except Exception as recovery_error:
                self.recovery_stats['failed_recoveries'] += 1
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
                error_info['recovery_error'] = str(recovery_error)
        
        # History'ye ekle
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Hata istatistiklerini al"""
        # Son 24 saatlik hatalar
        last_24h = datetime.now() - timedelta(hours=24)
        recent_errors = [
            error for error in self.error_history 
            if error['timestamp'] >= last_24h
        ]
        
        # Hata türü dağılımı
        error_types = defaultdict(int)
        for error in recent_errors:
            error_types[error['error_type']] += 1
        
        return {
            'total_stats': self.recovery_stats,
            'recent_24h': {
                'total_errors': len(recent_errors),
                'error_types': dict(error_types),
                'recovery_rate': (
                    sum(1 for e in recent_errors if e['recovery_successful']) / 
                    len(recent_errors) if recent_errors else 0
                )
            },
            'registered_strategies': list(self.recovery_strategies.keys())
        }


# Global instances
memory_manager = MemoryManager()
error_recovery = ErrorRecoveryManager()


# Decorator'lar
def memory_monitored(category: str = 'general'):
    """Memory monitoring decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Başlangıç memory
            initial_stats = memory_manager.get_memory_stats()
            
            try:
                result = func(*args, **kwargs)
                
                # Sonuç objesini takip et
                if result is not None:
                    memory_manager.track_object(result, category)
                
                return result
                
            finally:
                # Bitiş memory
                final_stats = memory_manager.get_memory_stats()
                memory_increase = final_stats.process_mb - initial_stats.process_mb
                
                if memory_increase > 10:  # 10MB'dan fazla artış
                    memory_manager.logger.warning(
                        f"Function {func.__name__} increased memory by {memory_increase:.2f}MB"
                    )
        
        return wrapper
    return decorator


def error_recovery_enabled(recovery_strategy: Optional[Callable] = None):
    """Error recovery decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                # Özel recovery strategy varsa kaydet
                if recovery_strategy:
                    error_type = type(e).__name__
                    error_recovery.register_recovery_strategy(error_type, recovery_strategy)
                
                # Hata kurtarma dene
                recovered = error_recovery.handle_error(e, context)
                
                if not recovered:
                    raise  # Kurtarılamazsa hatayı yeniden fırlat
                
                # Kurtarma başarılıysa None döndür veya default değer
                return None
        
        return wrapper
    return decorator


# Varsayılan recovery stratejileri
def connection_error_recovery(error: Exception, context: Dict[str, Any]) -> bool:
    """Bağlantı hatası kurtarma stratejisi"""
    import time
    
    # Kısa bekleme
    time.sleep(1)
    
    # Bağlantı yeniden kurma deneme
    # Bu gerçek uygulamada connection pool'u yeniden başlatma olabilir
    return True


def memory_error_recovery(error: Exception, context: Dict[str, Any]) -> bool:
    """Memory hatası kurtarma stratejisi"""
    # Acil memory temizliği
    result = memory_manager.emergency_cleanup()
    
    # Memory temizliği başarılıysa kurtarma başarılı
    return result['memory_freed_mb'] > 0


def database_error_recovery(error: Exception, context: Dict[str, Any]) -> bool:
    """Database hatası kurtarma stratejisi"""
    # Database bağlantısını yeniden kurma
    # Bu gerçek uygulamada connection pool'u yeniden başlatma olabilir
    return True


# Varsayılan stratejileri kaydet
error_recovery.register_recovery_strategy('ConnectionError', connection_error_recovery)
error_recovery.register_recovery_strategy('MemoryError', memory_error_recovery)
error_recovery.register_recovery_strategy('DatabaseError', database_error_recovery)


# Cleanup callbacks
def cache_cleanup_callback(cleanup_type: str):
    """Cache temizlik callback'i"""
    if cleanup_type == 'emergency':
        # Tüm cache'i temizle
        pass
    elif cleanup_type == 'soft':
        # Sadece eski cache'leri temizle
        pass


# Callback'i kaydet
memory_manager.register_cleanup_callback(cache_cleanup_callback)


# Örnek kullanım
if __name__ == "__main__":
    # Memory monitoring başlat
    memory_manager.start_monitoring()
    
    @memory_monitored(category='test')
    @error_recovery_enabled()
    def test_function():
        """Test fonksiyonu"""
        # Büyük liste oluştur
        data = [i for i in range(100000)]
        return data
    
    try:
        # Test
        result = test_function()
        print(f"Function completed, result length: {len(result)}")
        
        # Memory raporu
        report = memory_manager.get_memory_report()
        print(f"Memory usage: {report['current_stats'].percent:.1f}%")
        
        # Error statistics
        error_stats = error_recovery.get_error_statistics()
        print(f"Total errors: {error_stats['total_stats']['total_errors']}")
        
    finally:
        # Monitoring durdur
        memory_manager.stop_monitoring()
