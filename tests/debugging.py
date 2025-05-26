"""
Debugging Tools
Hata ayıklama ve performans izleme araçları
"""

import time
import psutil
import logging
import traceback
import functools
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import json
import os
import sys

from tests import PerformanceTimer, get_memory_usage


@dataclass
class PerformanceMetric:
    """Performans metriği"""
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    args_info: str = ""
    result_info: str = ""


@dataclass
class ErrorInfo:
    """Hata bilgisi"""
    error_type: str
    error_message: str
    traceback_info: str
    function_name: str
    timestamp: datetime
    context: Dict[str, Any]


class PerformanceProfiler:
    """Performans profiler"""
    
    def __init__(self):
        """Profiler başlatıcı"""
        self.metrics: List[PerformanceMetric] = []
        self.enabled = True
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def profile(self, include_args: bool = False, include_result: bool = False):
        """
        Performans profiling decorator
        
        Args:
            include_args: Argümanları kaydet
            include_result: Sonucu kaydet
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Başlangıç metrikleri
                start_time = time.time()
                start_memory = get_memory_usage()
                process = psutil.Process()
                start_cpu = process.cpu_percent()
                
                try:
                    # Fonksiyonu çalıştır
                    result = func(*args, **kwargs)
                    
                    # Bitiş metrikleri
                    end_time = time.time()
                    end_memory = get_memory_usage()
                    end_cpu = process.cpu_percent()
                    
                    # Metrik oluştur
                    metric = PerformanceMetric(
                        function_name=f"{func.__module__}.{func.__name__}",
                        execution_time=end_time - start_time,
                        memory_usage=end_memory - start_memory,
                        cpu_usage=end_cpu - start_cpu,
                        timestamp=datetime.now(),
                        args_info=str(args[:2]) if include_args and args else "",
                        result_info=str(result)[:100] if include_result and result else ""
                    )
                    
                    # Thread-safe kayıt
                    with self.lock:
                        self.metrics.append(metric)
                    
                    return result
                    
                except Exception as e:
                    # Hata durumunda da metrik kaydet
                    end_time = time.time()
                    metric = PerformanceMetric(
                        function_name=f"{func.__module__}.{func.__name__}",
                        execution_time=end_time - start_time,
                        memory_usage=0,
                        cpu_usage=0,
                        timestamp=datetime.now(),
                        args_info=f"ERROR: {str(e)}"
                    )
                    
                    with self.lock:
                        self.metrics.append(metric)
                    
                    raise
            
            return wrapper
        return decorator
    
    def get_metrics(self, function_name: Optional[str] = None) -> List[PerformanceMetric]:
        """
        Metrikleri al
        
        Args:
            function_name: Belirli fonksiyon için filtrele
            
        Returns:
            Metrik listesi
        """
        with self.lock:
            if function_name:
                return [m for m in self.metrics if function_name in m.function_name]
            return self.metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Performans özeti"""
        with self.lock:
            if not self.metrics:
                return {}
            
            # Fonksiyon bazında grupla
            function_stats = {}
            
            for metric in self.metrics:
                func_name = metric.function_name
                if func_name not in function_stats:
                    function_stats[func_name] = {
                        'call_count': 0,
                        'total_time': 0,
                        'avg_time': 0,
                        'max_time': 0,
                        'min_time': float('inf'),
                        'total_memory': 0,
                        'avg_memory': 0
                    }
                
                stats = function_stats[func_name]
                stats['call_count'] += 1
                stats['total_time'] += metric.execution_time
                stats['max_time'] = max(stats['max_time'], metric.execution_time)
                stats['min_time'] = min(stats['min_time'], metric.execution_time)
                stats['total_memory'] += metric.memory_usage
            
            # Ortalama hesapla
            for stats in function_stats.values():
                if stats['call_count'] > 0:
                    stats['avg_time'] = stats['total_time'] / stats['call_count']
                    stats['avg_memory'] = stats['total_memory'] / stats['call_count']
                    if stats['min_time'] == float('inf'):
                        stats['min_time'] = 0
            
            return function_stats
    
    def clear_metrics(self):
        """Metrikleri temizle"""
        with self.lock:
            self.metrics.clear()
    
    def save_metrics(self, filepath: str):
        """Metrikleri dosyaya kaydet"""
        with self.lock:
            data = []
            for metric in self.metrics:
                data.append({
                    'function_name': metric.function_name,
                    'execution_time': metric.execution_time,
                    'memory_usage': metric.memory_usage,
                    'cpu_usage': metric.cpu_usage,
                    'timestamp': metric.timestamp.isoformat(),
                    'args_info': metric.args_info,
                    'result_info': metric.result_info
                })
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)


class ErrorTracker:
    """Hata izleyici"""
    
    def __init__(self):
        """Error tracker başlatıcı"""
        self.errors: List[ErrorInfo] = []
        self.enabled = True
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def track_errors(self, include_context: bool = True):
        """
        Hata izleme decorator
        
        Args:
            include_context: Context bilgisini dahil et
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if self.enabled:
                        # Hata bilgisini topla
                        error_info = ErrorInfo(
                            error_type=type(e).__name__,
                            error_message=str(e),
                            traceback_info=traceback.format_exc(),
                            function_name=f"{func.__module__}.{func.__name__}",
                            timestamp=datetime.now(),
                            context={
                                'args_count': len(args),
                                'kwargs_keys': list(kwargs.keys()),
                                'thread_name': threading.current_thread().name
                            } if include_context else {}
                        )
                        
                        # Thread-safe kayıt
                        with self.lock:
                            self.errors.append(error_info)
                        
                        self.logger.error(f"Error tracked in {func.__name__}: {e}")
                    
                    # Hatayı yeniden fırlat
                    raise
            
            return wrapper
        return decorator
    
    def get_errors(self, error_type: Optional[str] = None) -> List[ErrorInfo]:
        """
        Hataları al
        
        Args:
            error_type: Belirli hata türü için filtrele
            
        Returns:
            Hata listesi
        """
        with self.lock:
            if error_type:
                return [e for e in self.errors if e.error_type == error_type]
            return self.errors.copy()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Hata özeti"""
        with self.lock:
            if not self.errors:
                return {}
            
            # Hata türü bazında grupla
            error_stats = {}
            function_errors = {}
            
            for error in self.errors:
                # Hata türü istatistikleri
                error_type = error.error_type
                if error_type not in error_stats:
                    error_stats[error_type] = {
                        'count': 0,
                        'functions': set(),
                        'latest': None
                    }
                
                error_stats[error_type]['count'] += 1
                error_stats[error_type]['functions'].add(error.function_name)
                error_stats[error_type]['latest'] = error.timestamp
                
                # Fonksiyon bazında hata sayısı
                func_name = error.function_name
                if func_name not in function_errors:
                    function_errors[func_name] = 0
                function_errors[func_name] += 1
            
            # Set'leri list'e çevir (JSON serializable)
            for stats in error_stats.values():
                stats['functions'] = list(stats['functions'])
                if stats['latest']:
                    stats['latest'] = stats['latest'].isoformat()
            
            return {
                'total_errors': len(self.errors),
                'error_types': error_stats,
                'function_errors': function_errors
            }
    
    def clear_errors(self):
        """Hataları temizle"""
        with self.lock:
            self.errors.clear()


class MemoryMonitor:
    """Memory izleyici"""
    
    def __init__(self, threshold_mb: float = 100.0):
        """
        Memory monitor başlatıcı
        
        Args:
            threshold_mb: Uyarı eşiği (MB)
        """
        self.threshold_mb = threshold_mb
        self.enabled = True
        self.logger = logging.getLogger(__name__)
        self.baseline_memory = get_memory_usage()
    
    def monitor_memory(self, warn_threshold: Optional[float] = None):
        """
        Memory monitoring decorator
        
        Args:
            warn_threshold: Özel uyarı eşiği
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Başlangıç memory
                start_memory = get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Bitiş memory
                    end_memory = get_memory_usage()
                    memory_increase = end_memory - start_memory
                    
                    # Eşik kontrolü
                    threshold = warn_threshold or self.threshold_mb
                    if memory_increase > threshold:
                        self.logger.warning(
                            f"High memory usage in {func.__name__}: "
                            f"{memory_increase:.2f}MB increase "
                            f"(threshold: {threshold}MB)"
                        )
                    
                    return result
                    
                except Exception as e:
                    # Hata durumunda da memory kontrol et
                    end_memory = get_memory_usage()
                    memory_increase = end_memory - start_memory
                    
                    if memory_increase > (warn_threshold or self.threshold_mb):
                        self.logger.warning(
                            f"Memory leak suspected in {func.__name__}: "
                            f"{memory_increase:.2f}MB increase during error"
                        )
                    
                    raise
            
            return wrapper
        return decorator
    
    def get_current_usage(self) -> Dict[str, float]:
        """Mevcut memory kullanımı"""
        current = get_memory_usage()
        return {
            'current_mb': current,
            'baseline_mb': self.baseline_memory,
            'increase_mb': current - self.baseline_memory,
            'system_total_mb': psutil.virtual_memory().total / 1024 / 1024,
            'system_available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'system_percent': psutil.virtual_memory().percent
        }


class HealthChecker:
    """Sistem sağlık kontrolü"""
    
    def __init__(self):
        """Health checker başlatıcı"""
        self.checks: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_check(self, name: str, check_func: Callable):
        """
        Sağlık kontrolü kaydet
        
        Args:
            name: Kontrol adı
            check_func: Kontrol fonksiyonu (bool döndürmeli)
        """
        self.checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Tüm sağlık kontrollerini çalıştır"""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                check_time = time.time() - start_time
                
                results[name] = {
                    'healthy': bool(is_healthy),
                    'check_time': check_time,
                    'error': None
                }
                
                if not is_healthy:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    'healthy': False,
                    'check_time': 0,
                    'error': str(e)
                }
                overall_healthy = False
                self.logger.error(f"Health check '{name}' failed: {e}")
        
        results['overall_healthy'] = overall_healthy
        results['timestamp'] = datetime.now().isoformat()
        
        return results


class DebugLogger:
    """Debug logger"""
    
    def __init__(self, log_file: str = "debug.log"):
        """
        Debug logger başlatıcı
        
        Args:
            log_file: Log dosyası
        """
        self.log_file = log_file
        self.logger = logging.getLogger("debug")
        
        # File handler ekle
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
    
    def debug_function(self, include_args: bool = True, include_result: bool = True):
        """
        Debug logging decorator
        
        Args:
            include_args: Argümanları logla
            include_result: Sonucu logla
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = f"{func.__module__}.{func.__name__}"
                
                # Fonksiyon başlangıcı
                if include_args:
                    self.logger.debug(f"ENTER {func_name} with args={args}, kwargs={kwargs}")
                else:
                    self.logger.debug(f"ENTER {func_name}")
                
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Fonksiyon bitişi
                    if include_result:
                        self.logger.debug(
                            f"EXIT {func_name} in {execution_time:.4f}s with result={result}"
                        )
                    else:
                        self.logger.debug(f"EXIT {func_name} in {execution_time:.4f}s")
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.debug(
                        f"ERROR {func_name} in {execution_time:.4f}s: {e}"
                    )
                    raise
            
            return wrapper
        return decorator


# Global instances
profiler = PerformanceProfiler()
error_tracker = ErrorTracker()
memory_monitor = MemoryMonitor()
health_checker = HealthChecker()
debug_logger = DebugLogger()


def setup_debugging(enable_profiling: bool = True, enable_error_tracking: bool = True,
                   enable_memory_monitoring: bool = True, log_level: str = "DEBUG"):
    """
    Debugging araçlarını kur
    
    Args:
        enable_profiling: Profiling etkinleştir
        enable_error_tracking: Hata izleme etkinleştir
        enable_memory_monitoring: Memory monitoring etkinleştir
        log_level: Log seviyesi
    """
    # Global ayarlar
    profiler.enabled = enable_profiling
    error_tracker.enabled = enable_error_tracking
    memory_monitor.enabled = enable_memory_monitoring
    
    # Logging seviyesi
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    print(f"Debugging setup complete:")
    print(f"  - Profiling: {'ON' if enable_profiling else 'OFF'}")
    print(f"  - Error Tracking: {'ON' if enable_error_tracking else 'OFF'}")
    print(f"  - Memory Monitoring: {'ON' if enable_memory_monitoring else 'OFF'}")
    print(f"  - Log Level: {log_level}")


def get_debug_report() -> Dict[str, Any]:
    """Kapsamlı debug raporu"""
    return {
        'timestamp': datetime.now().isoformat(),
        'performance_summary': profiler.get_summary(),
        'error_summary': error_tracker.get_error_summary(),
        'memory_usage': memory_monitor.get_current_usage(),
        'health_status': health_checker.run_health_checks(),
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    }


def save_debug_report(filepath: str = None):
    """Debug raporunu kaydet"""
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"debug_report_{timestamp}.json"
    
    report = get_debug_report()
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Debug report saved to: {filepath}")
    return filepath


# Örnek sağlık kontrolleri
def check_memory_usage() -> bool:
    """Memory kullanım kontrolü"""
    usage = psutil.virtual_memory().percent
    return usage < 90  # %90'dan az olmalı


def check_disk_space() -> bool:
    """Disk alanı kontrolü"""
    usage = psutil.disk_usage('/').percent
    return usage < 95  # %95'den az olmalı


def check_cpu_usage() -> bool:
    """CPU kullanım kontrolü"""
    usage = psutil.cpu_percent(interval=1)
    return usage < 80  # %80'den az olmalı


# Varsayılan sağlık kontrollerini kaydet
health_checker.register_check('memory_usage', check_memory_usage)
health_checker.register_check('disk_space', check_disk_space)
health_checker.register_check('cpu_usage', check_cpu_usage)


# Örnek kullanım
if __name__ == "__main__":
    # Debugging'i etkinleştir
    setup_debugging()
    
    # Örnek fonksiyon
    @profiler.profile(include_args=True)
    @error_tracker.track_errors()
    @memory_monitor.monitor_memory()
    @debug_logger.debug_function()
    def example_function(x: int, y: int) -> int:
        """Örnek fonksiyon"""
        time.sleep(0.1)  # Simüle edilmiş işlem
        if x < 0:
            raise ValueError("x cannot be negative")
        return x + y
    
    # Test çalıştır
    try:
        for i in range(5):
            result = example_function(i, i * 2)
            print(f"Result: {result}")
        
        # Hata durumu test et
        example_function(-1, 5)
        
    except ValueError as e:
        print(f"Expected error: {e}")
    
    # Rapor oluştur
    report_file = save_debug_report()
    print(f"\nDebug report generated: {report_file}")
    
    # Özet yazdır
    print("\n=== Performance Summary ===")
    perf_summary = profiler.get_summary()
    for func_name, stats in perf_summary.items():
        print(f"{func_name}: {stats['call_count']} calls, avg {stats['avg_time']:.4f}s")
    
    print("\n=== Error Summary ===")
    error_summary = error_tracker.get_error_summary()
    print(f"Total errors: {error_summary.get('total_errors', 0)}")
    
    print("\n=== Health Check ===")
    health_status = health_checker.run_health_checks()
    print(f"Overall healthy: {health_status['overall_healthy']}")
