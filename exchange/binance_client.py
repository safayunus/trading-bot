"""
Advanced Binance API Client
Async Binance exchange entegrasyonu, websocket desteği ve gelişmiş özellikler
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
from urllib.parse import urlencode
import websockets
from dataclasses import dataclass

@dataclass
class OrderResult:
    """Order sonucu veri sınıfı"""
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    status: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict] = None

class RateLimiter:
    """Gelişmiş rate limiter"""
    
    def __init__(self, requests_per_minute: int = 1200):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Rate limit kontrolü"""
        async with self.lock:
            now = time.time()
            # Son 1 dakikadaki istekleri filtrele
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.requests_per_minute:
                # Rate limit aşıldı, bekle
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()
            
            self.requests.append(now)

class BinanceWebSocket:
    """Binance WebSocket client"""
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        self.base_url = "wss://testnet.binance.vision/ws/" if testnet else "wss://stream.binance.com:9443/ws/"
        self.connections = {}
        self.callbacks = {}
        self.logger = logging.getLogger(__name__)
    
    async def subscribe_ticker(self, symbol: str, callback: Callable):
        """Ticker verilerine abone ol"""
        stream = f"{symbol.lower()}@ticker"
        await self._subscribe(stream, callback)
    
    async def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """Kline verilerine abone ol"""
        stream = f"{symbol.lower()}@kline_{interval}"
        await self._subscribe(stream, callback)
    
    async def subscribe_depth(self, symbol: str, callback: Callable):
        """Order book verilerine abone ol"""
        stream = f"{symbol.lower()}@depth"
        await self._subscribe(stream, callback)
    
    async def _subscribe(self, stream: str, callback: Callable):
        """WebSocket stream'e abone ol"""
        try:
            url = f"{self.base_url}{stream}"
            self.callbacks[stream] = callback
            
            async def handle_connection():
                try:
                    async with websockets.connect(url) as websocket:
                        self.connections[stream] = websocket
                        self.logger.info(f"WebSocket connected: {stream}")
                        
                        async for message in websocket:
                            data = json.loads(message)
                            await callback(data)
                            
                except Exception as e:
                    self.logger.error(f"WebSocket error ({stream}): {e}")
                    # Reconnection logic
                    await asyncio.sleep(5)
                    await self._subscribe(stream, callback)
            
            asyncio.create_task(handle_connection())
            
        except Exception as e:
            self.logger.error(f"WebSocket subscription error: {e}")
    
    async def unsubscribe(self, stream: str):
        """Stream aboneliğini iptal et"""
        if stream in self.connections:
            await self.connections[stream].close()
            del self.connections[stream]
            del self.callbacks[stream]

class BinanceClient:
    """Gelişmiş async Binance API client"""
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        """
        Binance client başlatıcı
        
        Args:
            api_key: Binance API key
            secret_key: Binance secret key
            testnet: Testnet kullanımı
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.logger = logging.getLogger(__name__)
        
        # API endpoints
        if testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"
        
        # Rate limiter
        self.rate_limiter = RateLimiter(requests_per_minute=1200)
        
        # Session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket
        self.websocket = BinanceWebSocket(testnet)
        
        # Cache
        self.symbol_info_cache = {}
        self.price_cache = {}
        self.cache_ttl = 60  # 60 saniye
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Client'ı başlat"""
        try:
            # HTTP session oluştur
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Bağlantıyı test et
            await self.test_connection()
            
            self.logger.info(f"Binance client başlatıldı (Testnet: {self.testnet})")
            
        except Exception as e:
            self.logger.error(f"Binance client başlatma hatası: {e}")
            raise
    
    async def close(self):
        """Client'ı kapat"""
        try:
            if self.session:
                await self.session.close()
            
            # WebSocket bağlantılarını kapat
            for stream in list(self.websocket.connections.keys()):
                await self.websocket.unsubscribe(stream)
            
            self.logger.info("Binance client kapatıldı")
            
        except Exception as e:
            self.logger.error(f"Client kapatma hatası: {e}")
    
    async def test_connection(self) -> bool:
        """
        Bağlantıyı test et
        
        Returns:
            bool: Bağlantı durumu
        """
        try:
            # Server time kontrolü
            server_time = await self.get_server_time()
            if not server_time:
                return False
            
            self.logger.info(f"Binance server time: {server_time}")
            
            # Account bilgisi kontrolü (API key gerekli)
            if self.api_key and self.secret_key:
                account_info = await self.get_account_info()
                if account_info:
                    self.logger.info("Binance hesap bilgisi alındı")
                    return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Bağlantı test hatası: {e}")
            return False
    
    async def _make_request(self, method: str, endpoint: str, 
                           params: Optional[Dict] = None, 
                           signed: bool = False) -> Optional[Dict]:
        """
        HTTP request yap
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parametreleri
            signed: İmzalı request mi
            
        Returns:
            Dict: Response data
        """
        if not self.session:
            await self.initialize()
        
        await self.rate_limiter.acquire()
        
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        
        if params is None:
            params = {}
        
        # İmzalı request için
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = urlencode(params)
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
        
        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                if method.upper() == 'GET':
                    async with self.session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            self.logger.error(f"API error {response.status}: {error_text}")
                            
                elif method.upper() == 'POST':
                    async with self.session.post(url, data=params, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            self.logger.error(f"API error {response.status}: {error_text}")
                
                elif method.upper() == 'DELETE':
                    async with self.session.delete(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            self.logger.error(f"API error {response.status}: {error_text}")
                
                # Rate limit hatası için özel bekleme
                if response.status == 429:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                break
                
            except Exception as e:
                self.logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    return None
        
        return None
    
    async def get_server_time(self) -> Optional[int]:
        """Server zamanını al"""
        result = await self._make_request('GET', '/api/v3/time')
        return result.get('serverTime') if result else None
    
    async def get_exchange_info(self) -> Optional[Dict]:
        """Exchange bilgilerini al"""
        return await self._make_request('GET', '/api/v3/exchangeInfo')
    
    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Hesap bilgilerini al
        
        Returns:
            Dict: Hesap bilgileri
        """
        try:
            result = await self._make_request('GET', '/api/v3/account', signed=True)
            
            if not result:
                return None
            
            # Bakiye bilgilerini işle
            balances = {}
            for balance in result['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                
                if free > 0 or locked > 0:
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            
            return {
                'balances': balances,
                'can_trade': result['canTrade'],
                'can_withdraw': result['canWithdraw'],
                'can_deposit': result['canDeposit'],
                'update_time': result['updateTime'],
                'account_type': result['accountType'],
                'maker_commission': result['makerCommission'],
                'taker_commission': result['takerCommission']
            }
            
        except Exception as e:
            self.logger.error(f"Hesap bilgisi alma hatası: {e}")
            return None
    
    async def get_symbol_price(self, symbol: str, use_cache: bool = True) -> Optional[float]:
        """
        Sembol fiyatını al
        
        Args:
            symbol: Trading pair (örn: BTCUSDT)
            use_cache: Cache kullanımı
            
        Returns:
            float: Güncel fiyat
        """
        try:
            # Cache kontrolü
            if use_cache and symbol in self.price_cache:
                cache_time, price = self.price_cache[symbol]
                if time.time() - cache_time < self.cache_ttl:
                    return price
            
            result = await self._make_request('GET', '/api/v3/ticker/price', {'symbol': symbol})
            
            if result:
                price = float(result['price'])
                # Cache'e kaydet
                self.price_cache[symbol] = (time.time(), price)
                return price
            
            return None
            
        except Exception as e:
            self.logger.error(f"Fiyat alma hatası ({symbol}): {e}")
            return None
    
    async def get_all_prices(self) -> Optional[Dict[str, float]]:
        """Tüm sembollerin fiyatlarını al"""
        try:
            result = await self._make_request('GET', '/api/v3/ticker/price')
            
            if result:
                prices = {}
                for ticker in result:
                    prices[ticker['symbol']] = float(ticker['price'])
                return prices
            
            return None
            
        except Exception as e:
            self.logger.error(f"Tüm fiyatlar alma hatası: {e}")
            return None
    
    async def get_klines(self, symbol: str, interval: str = '1h', 
                        limit: int = 100, start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Kline verilerini al
        
        Args:
            symbol: Trading pair
            interval: Zaman aralığı (1m, 5m, 1h, 1d, vb.)
            limit: Veri sayısı
            start_time: Başlangıç zamanı (timestamp)
            end_time: Bitiş zamanı (timestamp)
            
        Returns:
            DataFrame: OHLCV verileri
        """
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            result = await self._make_request('GET', '/api/v3/klines', params)
            
            if not result:
                return None
            
            # DataFrame'e dönüştür
            df = pd.DataFrame(result, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'ignore'
            ])
            
            # Veri tiplerini düzenle
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Sadece gerekli kolonları tut
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Kline verisi alma hatası ({symbol}): {e}")
            return None
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Order book verilerini al
        
        Args:
            symbol: Trading pair
            limit: Derinlik seviyesi (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Dict: Order book verileri
        """
        try:
            params = {'symbol': symbol, 'limit': limit}
            result = await self._make_request('GET', '/api/v3/depth', params)
            
            if result:
                return {
                    'symbol': symbol,
                    'bids': [[float(price), float(qty)] for price, qty in result['bids']],
                    'asks': [[float(price), float(qty)] for price, qty in result['asks']],
                    'last_update_id': result['lastUpdateId']
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Order book alma hatası ({symbol}): {e}")
            return None
    
    async def get_24hr_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        24 saatlik ticker verilerini al
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict: Ticker verileri
        """
        try:
            params = {'symbol': symbol}
            result = await self._make_request('GET', '/api/v3/ticker/24hr', params)
            
            if result:
                return {
                    'symbol': result['symbol'],
                    'price_change': float(result['priceChange']),
                    'price_change_percent': float(result['priceChangePercent']),
                    'weighted_avg_price': float(result['weightedAvgPrice']),
                    'prev_close_price': float(result['prevClosePrice']),
                    'last_price': float(result['lastPrice']),
                    'bid_price': float(result['bidPrice']),
                    'ask_price': float(result['askPrice']),
                    'open_price': float(result['openPrice']),
                    'high_price': float(result['highPrice']),
                    'low_price': float(result['lowPrice']),
                    'volume': float(result['volume']),
                    'quote_volume': float(result['quoteVolume']),
                    'open_time': result['openTime'],
                    'close_time': result['closeTime'],
                    'count': result['count']
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"24hr ticker alma hatası ({symbol}): {e}")
            return None
    
    async def place_market_order(self, symbol: str, side: str, 
                                quantity: float) -> OrderResult:
        """
        Market order ver
        
        Args:
            symbol: Trading pair
            side: BUY veya SELL
            quantity: Miktar
            
        Returns:
            OrderResult: Order sonucu
        """
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'MARKET',
            'quantity': quantity
        }
        
        return await self._place_order(params)
    
    async def place_limit_order(self, symbol: str, side: str, 
                               quantity: float, price: float) -> OrderResult:
        """
        Limit order ver
        
        Args:
            symbol: Trading pair
            side: BUY veya SELL
            quantity: Miktar
            price: Limit fiyatı
            
        Returns:
            OrderResult: Order sonucu
        """
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': quantity,
            'price': price
        }
        
        return await self._place_order(params)
    
    async def place_stop_loss_order(self, symbol: str, side: str, 
                                   quantity: float, stop_price: float) -> OrderResult:
        """
        Stop loss order ver
        
        Args:
            symbol: Trading pair
            side: BUY veya SELL
            quantity: Miktar
            stop_price: Stop fiyatı
            
        Returns:
            OrderResult: Order sonucu
        """
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'STOP_LOSS_LIMIT',
            'timeInForce': 'GTC',
            'quantity': quantity,
            'price': stop_price,  # Limit price = stop price
            'stopPrice': stop_price
        }
        
        return await self._place_order(params)
    
    async def _place_order(self, params: Dict) -> OrderResult:
        """
        Order ver (internal method)
        
        Args:
            params: Order parametreleri
            
        Returns:
            OrderResult: Order sonucu
        """
        try:
            # Test mode için
            if self.testnet:
                result = await self._make_request('POST', '/api/v3/order/test', params, signed=True)
                if result is not None:  # Test order başarılı (boş response döner)
                    self.logger.info(f"Test order created: {params}")
                    return OrderResult(
                        success=True,
                        client_order_id=f"test_{int(time.time())}",
                        symbol=params['symbol'],
                        side=params['side'],
                        quantity=params['quantity'],
                        status='TEST',
                        raw_response={'test_order': True, 'params': params}
                    )
                else:
                    return OrderResult(
                        success=False,
                        error_message="Test order failed"
                    )
            else:
                result = await self._make_request('POST', '/api/v3/order', params, signed=True)
                
                if result:
                    self.logger.info(f"Order placed: {result.get('orderId')}")
                    return OrderResult(
                        success=True,
                        order_id=str(result['orderId']),
                        client_order_id=result['clientOrderId'],
                        symbol=result['symbol'],
                        side=result['side'],
                        quantity=float(result['origQty']),
                        price=float(result.get('price', 0)),
                        status=result['status'],
                        raw_response=result
                    )
                else:
                    return OrderResult(
                        success=False,
                        error_message="Order placement failed"
                    )
            
        except Exception as e:
            self.logger.error(f"Order placement error: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Emri iptal et
        
        Args:
            symbol: Trading pair
            order_id: Emir ID
            
        Returns:
            bool: İptal durumu
        """
        try:
            params = {'symbol': symbol, 'orderId': order_id}
            result = await self._make_request('DELETE', '/api/v3/order', params, signed=True)
            
            if result:
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Order cancel error: {e}")
            return False
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Açık emirleri al
        
        Args:
            symbol: Trading pair (opsiyonel)
            
        Returns:
            List: Açık emirler
        """
        try:
            params = {'symbol': symbol} if symbol else {}
            result = await self._make_request('GET', '/api/v3/openOrders', params, signed=True)
            
            return result if result else []
            
        except Exception as e:
            self.logger.error(f"Open orders error: {e}")
            return []
    
    async def get_order_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Order geçmişini al
        
        Args:
            symbol: Trading pair
            limit: Kayıt sayısı
            
        Returns:
            List: Order geçmişi
        """
        try:
            params = {'symbol': symbol, 'limit': limit}
            result = await self._make_request('GET', '/api/v3/allOrders', params, signed=True)
            
            return result if result else []
            
        except Exception as e:
            self.logger.error(f"Order history error: {e}")
            return []
    
    async def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Trade geçmişini al
        
        Args:
            symbol: Trading pair
            limit: Kayıt sayısı
            
        Returns:
            List: Trade geçmişi
        """
        try:
            params = {'symbol': symbol, 'limit': limit}
            result = await self._make_request('GET', '/api/v3/myTrades', params, signed=True)
            
            return result if result else []
            
        except Exception as e:
            self.logger.error(f"Trade history error: {e}")
            return []
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Sembol bilgilerini al
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict: Sembol bilgileri
        """
        try:
            # Cache kontrolü
            if symbol in self.symbol_info_cache:
                return self.symbol_info_cache[symbol]
            
            exchange_info = await self.get_exchange_info()
            
            if exchange_info:
                for symbol_info in exchange_info['symbols']:
                    if symbol_info['symbol'] == symbol:
                        # Cache'e kaydet
                        self.symbol_info_cache[symbol] = symbol_info
                        return symbol_info
            
            return None
            
        except Exception as e:
            self.logger.error(f"Symbol info error ({symbol}): {e}")
            return None
    
    async def calculate_portfolio_value(self, base_currency: str = 'USDT') -> Optional[float]:
        """
        Portfolio değerini hesapla
        
        Args:
            base_currency: Temel para birimi
            
        Returns:
            float: Portfolio değeri
        """
        try:
            account_info = await self.get_account_info()
            if not account_info:
                return None
            
            total_value = 0.0
            balances = account_info['balances']
            
            # Tüm fiyatları al
            all_prices = await self.get_all_prices()
            if not all_prices:
                return None
            
            for asset, balance in balances.items():
                if balance['total'] <= 0:
                    continue
                
                if asset == base_currency:
                    total_value += balance['total']
                else:
                    # Fiyat çiftini bul
                    symbol = f"{asset}{base_currency}"
                    if symbol in all_prices:
                        total_value += balance['total'] * all_prices[symbol]
                    else:
                        # Ters çifti dene
                        reverse_symbol = f"{base_currency}{asset}"
                        if reverse_symbol in all_prices:
                            total_value += balance['total'] / all_prices[reverse_symbol]
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Portfolio value calculation error: {e}")
            return None
    
    # WebSocket methods
    async def start_price_stream(self, symbol: str, callback: Callable):
        """Fiyat stream'ini başlat"""
        await self.websocket.subscribe_ticker(symbol, callback)
    
    async def start_kline_stream(self, symbol: str, interval: str, callback: Callable):
        """Kline stream'ini başlat"""
        await self.websocket.subscribe_kline(symbol, interval, callback)
    
    async def start_depth_stream(self, symbol: str, callback: Callable):
        """Order book stream'ini başlat"""
        await self.websocket.subscribe_depth(symbol, callback)
    
    async def stop_stream(self, symbol: str, stream_type: str):
        """Stream'i durdur"""
        if stream_type == 'ticker':
            stream = f"{symbol.lower()}@ticker"
        elif stream_type == 'kline':
            stream = f"{symbol.lower()}@kline_1m"  # Default interval
        elif stream_type == 'depth':
            stream = f"{symbol.lower()}@depth"
        else:
            return
        
        await self.websocket.unsubscribe(stream)
