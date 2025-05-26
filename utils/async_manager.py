"""
Async Manager
Asenkron işlemler için optimizasyon ve yönetim
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import time
from contextlib import asynccontextmanager
import weakref


class AsyncConnectionPool:
    """Asenkron bağlantı havuzu yöneticisi"""
    
    def __init__(self, max_connections: int = 100, timeout: int = 30):
        """
        Bağlantı havuzu başlatıcı
        
        Args:
            max_connections: Maksimum bağlantı sayısı
            timeout: Bağlantı timeout süresi (saniye)
        """
        self.max_connections = max_connections
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Connection pool ayarları
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
    
    async def get_session(self) -> aiohttp.ClientSession:
        """HTTP session al"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'TradingBot/1.0',
                    'Accept': 'application/json',
                    'Connection': 'keep-alive'
                }
            )
        return self.session
    
    async def close(self):
        """Bağlantıları kapat"""
        if self.session and not self.session.closed:
            await self.session.close()
        if self.connector:
            await self.connector.close()


class AsyncTaskManager:
    """Asenkron görev yöneticisi"""
    
    def __init__(self, max_concurrent_tasks: int = 50):
        """
        Görev yöneticisi başlatıcı
        
        Args:
            max_concurrent_tasks: Maksimum eşzamanlı görev sayısı
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    async def run_task(self, task_id: str, coro: Callable, *args, **kwargs) -> Any:
        """
        Görev çalıştır
        
        Args:
            task_id: Görev ID'si
            coro: Coroutine fonksiyonu
            *args: Pozisyonel argümanlar
            **kwargs: Anahtar kelime argümanları
            
        Returns:
            Görev sonucu
        """
        async with self.semaphore:
            try:
                self.logger.debug(f"Starting task: {task_id}")
                start_time = time.time()
                
                # Görevi çalıştır
                if asyncio.iscoroutinefunction(coro):
                    result = await coro(*args, **kwargs)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, coro, *args, **kwargs
                    )
                
                execution_time = time.time() - start_time
                self.logger.debug(f"Task {task_id} completed in {execution_time:.2f}s")
                
                # Sonucu kaydet
                self.task_results[task_id] = {
                    'result': result,
                    'execution_time': execution_time,
                    'timestamp': datetime.now(),
                    'success': True
                }
                
                return result
                
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}")
                self.task_results[task_id] = {
                    'error': str(e),
                    'timestamp': datetime.now(),
                    'success': False
                }
                raise
            finally:
                # Görev listesinden kaldır
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
    
    async def run_concurrent_tasks(self, tasks: Dict[str, tuple]) -> Dict[str, Any]:
        """
        Eşzamanlı görevler çalıştır
        
        Args:
            tasks: {task_id: (coro, args, kwargs)} formatında görevler
            
        Returns:
            Görev sonuçları
        """
        # Görevleri oluştur
        task_coroutines = []
        for task_id, (coro, args, kwargs) in tasks.items():
            task_coro = self.run_task(task_id, coro, *args, **kwargs)
            task_coroutines.append(task_coro)
            
        # Tüm görevleri çalıştır
        try:
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Sonuçları organize et
            organized_results = {}
            for i, (task_id, _) in enumerate(tasks.items()):
                if isinstance(results[i], Exception):
                    organized_results[task_id] = {'error': str(results[i]), 'success': False}
                else:
                    organized_results[task_id] = {'result': results[i], 'success': True}
            
            return organized_results
            
        except Exception as e:
            self.logger.error(f"Concurrent tasks failed: {e}")
            raise
    
    def get_task_status(self) -> Dict[str, Any]:
        """Görev durumu al"""
        return {
            'running_tasks': len(self.running_tasks),
            'max_concurrent': self.max_concurrent_tasks,
            'available_slots': self.semaphore._value,
            'completed_tasks': len(self.task_results)
        }


class AsyncRateLimiter:
    """Asenkron rate limiter"""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Rate limiter başlatıcı
        
        Args:
            max_requests: Maksimum istek sayısı
            time_window: Zaman penceresi (saniye)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Rate limit kontrolü yap"""
        async with self.lock:
            current_time = time.time()
            
            # Eski istekleri temizle
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < self.time_window]
            
            # Limit kontrolü
            if len(self.requests) >= self.max_requests:
                # Bekleme süresi hesapla
                oldest_request = min(self.requests)
                wait_time = self.time_window - (current_time - oldest_request)
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()  # Tekrar dene
            
            # Yeni isteği kaydet
            self.requests.append(current_time)
            return True


class AsyncCache:
    """Asenkron cache sistemi"""
    
    def __init__(self, default_ttl: int = 300):
        """
        Cache başlatıcı
        
        Args:
            default_ttl: Varsayılan TTL (saniye)
        """
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Cache'den veri al"""
        async with self.lock:
            if key in self.cache:
                item = self.cache[key]
                
                # TTL kontrolü
                if time.time() - item['timestamp'] < item['ttl']:
                    return item['value']
                else:
                    # Süresi dolmuş, sil
                    del self.cache[key]
            
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Cache'e veri kaydet"""
        async with self.lock:
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl or self.default_ttl
            }
    
    async def delete(self, key: str) -> bool:
        """Cache'den veri sil"""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Cache'i temizle"""
        async with self.lock:
            self.cache.clear()
    
    async def cleanup_expired(self) -> int:
        """Süresi dolmuş verileri temizle"""
        async with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, item in self.cache.items():
                if current_time - item['timestamp'] >= item['ttl']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)


class AsyncRetryManager:
    """Asenkron retry yöneticisi"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 backoff_factor: float = 2.0):
        """
        Retry manager başlatıcı
        
        Args:
            max_retries: Maksimum deneme sayısı
            base_delay: Temel bekleme süresi
            backoff_factor: Backoff çarpanı
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
    
    async def retry_async(self, coro: Callable, *args, **kwargs) -> Any:
        """
        Asenkron fonksiyonu retry ile çalıştır
        
        Args:
            coro: Coroutine fonksiyonu
            *args: Pozisyonel argümanlar
            **kwargs: Anahtar kelime argümanları
            
        Returns:
            Fonksiyon sonucu
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(coro):
                    return await coro(*args, **kwargs)
                else:
                    return await asyncio.get_event_loop().run_in_executor(
                        None, coro, *args, **kwargs
                    )
                    
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (self.backoff_factor ** attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception


# Global instances
connection_pool = AsyncConnectionPool()
task_manager = AsyncTaskManager()
rate_limiter = AsyncRateLimiter()
cache = AsyncCache()
retry_manager = AsyncRetryManager()


@asynccontextmanager
async def async_context():
    """Async context manager"""
    try:
        yield {
            'connection_pool': connection_pool,
            'task_manager': task_manager,
            'rate_limiter': rate_limiter,
            'cache': cache,
            'retry_manager': retry_manager
        }
    finally:
        await connection_pool.close()


async def optimize_async_performance():
    """Async performans optimizasyonu"""
    # Event loop policy ayarla (Windows için)
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Task factory ayarla
    loop = asyncio.get_event_loop()
    
    # Debug mode'u kapat (production için)
    loop.set_debug(False)
    
    return loop


# Decorator'lar
def async_cached(ttl: int = 300):
    """Async cache decorator"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Cache key oluştur
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Cache'den kontrol et
            cached_result = await cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Fonksiyonu çalıştır ve cache'le
            result = await func(*args, **kwargs)
            await cache.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator


def async_rate_limited(max_requests: int = 10, time_window: int = 60):
    """Async rate limit decorator"""
    limiter = AsyncRateLimiter(max_requests, time_window)
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            await limiter.acquire()
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def async_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Async retry decorator"""
    retry_mgr = AsyncRetryManager(max_retries, base_delay)
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await retry_mgr.retry_async(func, *args, **kwargs)
        return wrapper
    return decorator


# Örnek kullanım
if __name__ == "__main__":
    async def example_usage():
        """Örnek kullanım"""
        
        # Cache kullanımı
        @async_cached(ttl=60)
        async def get_price_data(symbol: str):
            # Simulated API call
            await asyncio.sleep(0.1)
            return f"Price data for {symbol}"
        
        # Rate limiting kullanımı
        @async_rate_limited(max_requests=5, time_window=10)
        async def api_call(endpoint: str):
            # Simulated API call
            await asyncio.sleep(0.1)
            return f"Response from {endpoint}"
        
        # Retry kullanımı
        @async_retry(max_retries=3)
        async def unreliable_function():
            import random
            if random.random() < 0.7:  # %70 başarısızlık
                raise Exception("Random failure")
            return "Success!"
        
        # Test
        print("Testing async optimizations...")
        
        # Cache test
        result1 = await get_price_data("BTCUSDT")
        result2 = await get_price_data("BTCUSDT")  # Cache'den gelecek
        print(f"Cache test: {result1 == result2}")
        
        # Rate limiting test
        tasks = [api_call(f"endpoint_{i}") for i in range(3)]
        results = await asyncio.gather(*tasks)
        print(f"Rate limiting test: {len(results)} results")
        
        # Retry test
        try:
            result = await unreliable_function()
            print(f"Retry test: {result}")
        except Exception as e:
            print(f"Retry test failed: {e}")
    
    # Çalıştır
    asyncio.run(example_usage())
