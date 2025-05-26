"""
Binance API Tests
Binance API fonksiyonlarının test edilmesi
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from tests import TestUtils, PerformanceTimer, get_memory_usage


class TestBinanceClient:
    """BinanceClient sınıfı testleri"""
    
    @pytest.fixture
    def binance_client(self, mock_binance_client):
        """BinanceClient instance oluştur"""
        with patch('binance.client.Client', return_value=mock_binance_client):
            from exchange.binance_client import BinanceClient
            
            # Mock config
            config = Mock()
            config.BINANCE_API_KEY = 'test_api_key'
            config.BINANCE_SECRET_KEY = 'test_secret_key'
            config.BINANCE_TESTNET = True
            
            client = BinanceClient(config)
            return client
    
    def test_binance_client_initialization(self, binance_client):
        """BinanceClient başlatma testi"""
        assert binance_client is not None
        assert binance_client.testnet == True
        assert binance_client.client is not None
    
    def test_connection_test(self, binance_client, mock_binance_client):
        """Bağlantı testi"""
        # Mock ping response
        mock_binance_client.ping.return_value = {}
        
        # Test connection
        result = binance_client.test_connection()
        
        # Doğrulama
        assert result == True
        mock_binance_client.ping.assert_called_once()
    
    def test_connection_test_failure(self, binance_client, mock_binance_client):
        """Bağlantı hatası testi"""
        # Mock ping exception
        mock_binance_client.ping.side_effect = Exception("Connection failed")
        
        # Test connection
        result = binance_client.test_connection()
        
        # Doğrulama
        assert result == False
    
    def test_get_account_info(self, binance_client, mock_binance_client):
        """Hesap bilgisi alma testi"""
        # Mock account response
        mock_account = {
            'balances': [
                {'asset': 'USDT', 'free': '1000.0', 'locked': '0.0'},
                {'asset': 'BTC', 'free': '0.1', 'locked': '0.0'},
                {'asset': 'ETH', 'free': '2.5', 'locked': '0.5'}
            ],
            'canTrade': True,
            'canWithdraw': True,
            'canDeposit': True
        }
        mock_binance_client.get_account.return_value = mock_account
        
        # Test get account
        result = binance_client.get_account_info()
        
        # Doğrulama
        assert result is not None
        assert 'balances' in result
        assert len(result['balances']) == 3
        assert result['canTrade'] == True
        mock_binance_client.get_account.assert_called_once()
    
    def test_get_balance(self, binance_client, mock_binance_client):
        """Bakiye alma testi"""
        # Mock account response
        mock_account = {
            'balances': [
                {'asset': 'USDT', 'free': '1000.0', 'locked': '50.0'},
                {'asset': 'BTC', 'free': '0.1', 'locked': '0.0'}
            ]
        }
        mock_binance_client.get_account.return_value = mock_account
        
        # Test get balance
        usdt_balance = binance_client.get_balance('USDT')
        btc_balance = binance_client.get_balance('BTC')
        nonexistent_balance = binance_client.get_balance('XRP')
        
        # Doğrulama
        assert usdt_balance == {'free': 1000.0, 'locked': 50.0, 'total': 1050.0}
        assert btc_balance == {'free': 0.1, 'locked': 0.0, 'total': 0.1}
        assert nonexistent_balance == {'free': 0.0, 'locked': 0.0, 'total': 0.0}
    
    def test_get_symbol_price(self, binance_client, mock_binance_client):
        """Sembol fiyatı alma testi"""
        # Mock price response
        mock_binance_client.get_symbol_ticker.return_value = {'price': '45000.50'}
        
        # Test get price
        price = binance_client.get_symbol_price('BTCUSDT')
        
        # Doğrulama
        assert price == 45000.50
        mock_binance_client.get_symbol_ticker.assert_called_once_with(symbol='BTCUSDT')
    
    def test_get_symbol_price_failure(self, binance_client, mock_binance_client):
        """Fiyat alma hatası testi"""
        # Mock price exception
        mock_binance_client.get_symbol_ticker.side_effect = Exception("Symbol not found")
        
        # Test get price
        price = binance_client.get_symbol_price('INVALID')
        
        # Doğrulama
        assert price is None
    
    def test_get_klines(self, binance_client, mock_binance_client):
        """Kline verisi alma testi"""
        # Mock klines response
        mock_klines = [
            [1640995200000, '45000.0', '45100.0', '44900.0', '45050.0', '100.5', 1640995259999, '4505000.0', 1000, '50.25', '2252500.0', '0'],
            [1640995260000, '45050.0', '45200.0', '45000.0', '45150.0', '120.3', 1640995319999, '5430450.0', 1200, '60.15', '2715225.0', '0']
        ]
        mock_binance_client.get_klines.return_value = mock_klines
        
        # Test get klines
        klines = binance_client.get_klines('BTCUSDT', '1h', limit=2)
        
        # Doğrulama
        assert len(klines) == 2
        assert klines[0]['open'] == 45000.0
        assert klines[0]['close'] == 45050.0
        assert klines[0]['volume'] == 100.5
        mock_binance_client.get_klines.assert_called_once()
    
    def test_get_24hr_ticker(self, binance_client, mock_binance_client):
        """24 saat ticker testi"""
        # Mock ticker response
        mock_ticker = {
            'symbol': 'BTCUSDT',
            'priceChange': '500.0',
            'priceChangePercent': '1.12',
            'weightedAvgPrice': '45250.0',
            'prevClosePrice': '44500.0',
            'lastPrice': '45000.0',
            'bidPrice': '44995.0',
            'askPrice': '45005.0',
            'openPrice': '44500.0',
            'highPrice': '45500.0',
            'lowPrice': '44200.0',
            'volume': '12345.67',
            'count': 98765
        }
        mock_binance_client.get_ticker.return_value = mock_ticker
        
        # Test get ticker
        ticker = binance_client.get_24hr_ticker('BTCUSDT')
        
        # Doğrulama
        assert ticker['symbol'] == 'BTCUSDT'
        assert ticker['priceChange'] == 500.0
        assert ticker['priceChangePercent'] == 1.12
        assert ticker['volume'] == 12345.67
        mock_binance_client.get_ticker.assert_called_once_with(symbol='BTCUSDT')
    
    def test_get_order_book(self, binance_client, mock_binance_client):
        """Order book alma testi"""
        # Mock order book response
        mock_order_book = {
            'lastUpdateId': 123456789,
            'bids': [
                ['44995.0', '1.5'],
                ['44990.0', '2.3'],
                ['44985.0', '0.8']
            ],
            'asks': [
                ['45005.0', '1.2'],
                ['45010.0', '2.1'],
                ['45015.0', '0.9']
            ]
        }
        mock_binance_client.get_order_book.return_value = mock_order_book
        
        # Test get order book
        order_book = binance_client.get_order_book('BTCUSDT', limit=5)
        
        # Doğrulama
        assert 'bids' in order_book
        assert 'asks' in order_book
        assert len(order_book['bids']) == 3
        assert len(order_book['asks']) == 3
        assert order_book['bids'][0] == [44995.0, 1.5]
        assert order_book['asks'][0] == [45005.0, 1.2]
        mock_binance_client.get_order_book.assert_called_once_with(symbol='BTCUSDT', limit=5)
    
    def test_place_market_order(self, binance_client, mock_binance_client):
        """Market order testi"""
        # Mock order response
        mock_order = {
            'symbol': 'BTCUSDT',
            'orderId': 123456,
            'clientOrderId': 'test_order_123',
            'transactTime': 1640995200000,
            'price': '0.0',
            'origQty': '0.01',
            'executedQty': '0.01',
            'cummulativeQuoteQty': '450.0',
            'status': 'FILLED',
            'timeInForce': 'GTC',
            'type': 'MARKET',
            'side': 'BUY'
        }
        mock_binance_client.order_market_buy.return_value = mock_order
        
        # Test place order
        result = binance_client.place_market_order('BTCUSDT', 'BUY', 0.01)
        
        # Doğrulama
        assert result is not None
        assert result['symbol'] == 'BTCUSDT'
        assert result['side'] == 'BUY'
        assert result['status'] == 'FILLED'
        assert result['executedQty'] == 0.01
        mock_binance_client.order_market_buy.assert_called_once()
    
    def test_place_limit_order(self, binance_client, mock_binance_client):
        """Limit order testi"""
        # Mock order response
        mock_order = {
            'symbol': 'BTCUSDT',
            'orderId': 123457,
            'clientOrderId': 'test_limit_order_123',
            'transactTime': 1640995200000,
            'price': '44000.0',
            'origQty': '0.01',
            'executedQty': '0.0',
            'cummulativeQuoteQty': '0.0',
            'status': 'NEW',
            'timeInForce': 'GTC',
            'type': 'LIMIT',
            'side': 'BUY'
        }
        mock_binance_client.order_limit_buy.return_value = mock_order
        
        # Test place limit order
        result = binance_client.place_limit_order('BTCUSDT', 'BUY', 0.01, 44000.0)
        
        # Doğrulama
        assert result is not None
        assert result['symbol'] == 'BTCUSDT'
        assert result['side'] == 'BUY'
        assert result['type'] == 'LIMIT'
        assert result['price'] == 44000.0
        assert result['status'] == 'NEW'
        mock_binance_client.order_limit_buy.assert_called_once()
    
    def test_cancel_order(self, binance_client, mock_binance_client):
        """Order iptal testi"""
        # Mock cancel response
        mock_cancel = {
            'symbol': 'BTCUSDT',
            'orderId': 123456,
            'clientOrderId': 'test_order_123',
            'status': 'CANCELED'
        }
        mock_binance_client.cancel_order.return_value = mock_cancel
        
        # Test cancel order
        result = binance_client.cancel_order('BTCUSDT', 123456)
        
        # Doğrulama
        assert result is not None
        assert result['status'] == 'CANCELED'
        assert result['orderId'] == 123456
        mock_binance_client.cancel_order.assert_called_once_with(symbol='BTCUSDT', orderId=123456)
    
    def test_get_open_orders(self, binance_client, mock_binance_client):
        """Açık orderlar testi"""
        # Mock open orders response
        mock_orders = [
            {
                'symbol': 'BTCUSDT',
                'orderId': 123456,
                'price': '44000.0',
                'origQty': '0.01',
                'executedQty': '0.0',
                'status': 'NEW',
                'type': 'LIMIT',
                'side': 'BUY'
            },
            {
                'symbol': 'ETHUSDT',
                'orderId': 123457,
                'price': '3000.0',
                'origQty': '0.5',
                'executedQty': '0.0',
                'status': 'NEW',
                'type': 'LIMIT',
                'side': 'SELL'
            }
        ]
        mock_binance_client.get_open_orders.return_value = mock_orders
        
        # Test get open orders
        orders = binance_client.get_open_orders('BTCUSDT')
        
        # Doğrulama
        assert len(orders) == 2
        assert orders[0]['symbol'] == 'BTCUSDT'
        assert orders[0]['side'] == 'BUY'
        assert orders[1]['symbol'] == 'ETHUSDT'
        assert orders[1]['side'] == 'SELL'
        mock_binance_client.get_open_orders.assert_called_once_with(symbol='BTCUSDT')
    
    def test_get_order_history(self, binance_client, mock_binance_client):
        """Order geçmişi testi"""
        # Mock order history response
        mock_history = [
            {
                'symbol': 'BTCUSDT',
                'orderId': 123454,
                'price': '45000.0',
                'origQty': '0.01',
                'executedQty': '0.01',
                'status': 'FILLED',
                'type': 'MARKET',
                'side': 'BUY',
                'time': 1640995200000
            },
            {
                'symbol': 'BTCUSDT',
                'orderId': 123455,
                'price': '46000.0',
                'origQty': '0.01',
                'executedQty': '0.01',
                'status': 'FILLED',
                'type': 'MARKET',
                'side': 'SELL',
                'time': 1640995800000
            }
        ]
        mock_binance_client.get_all_orders.return_value = mock_history
        
        # Test get order history
        history = binance_client.get_order_history('BTCUSDT', limit=10)
        
        # Doğrulama
        assert len(history) == 2
        assert history[0]['status'] == 'FILLED'
        assert history[0]['side'] == 'BUY'
        assert history[1]['side'] == 'SELL'
        mock_binance_client.get_all_orders.assert_called_once()
    
    def test_performance_multiple_requests(self, binance_client, mock_binance_client):
        """Çoklu istek performans testi"""
        # Mock responses
        mock_binance_client.get_symbol_ticker.return_value = {'price': '45000.0'}
        
        with PerformanceTimer() as timer:
            # 100 fiyat isteği
            for i in range(100):
                binance_client.get_symbol_price('BTCUSDT')
        
        # 100 istek 2 saniyeden az sürmeli
        assert timer.elapsed < 2.0
    
    def test_memory_usage_large_data(self, binance_client, mock_binance_client):
        """Büyük veri memory kullanım testi"""
        # Mock large klines response (1000 entries)
        mock_klines = []
        for i in range(1000):
            mock_klines.append([
                1640995200000 + i * 60000,  # timestamp
                '45000.0', '45100.0', '44900.0', '45050.0',  # OHLC
                '100.5', 1640995259999, '4505000.0', 1000,  # volume data
                '50.25', '2252500.0', '0'
            ])
        mock_binance_client.get_klines.return_value = mock_klines
        
        initial_memory = get_memory_usage()
        
        # Get large dataset
        klines = binance_client.get_klines('BTCUSDT', '1m', limit=1000)
        
        current_memory = get_memory_usage()
        memory_increase = current_memory - initial_memory
        
        # Memory artışı 100MB'dan az olmalı
        assert memory_increase < 100
        assert len(klines) == 1000


class TestOrderManager:
    """OrderManager sınıfı testleri"""
    
    @pytest.fixture
    def order_manager(self, binance_client):
        """OrderManager instance oluştur"""
        from exchange.order_manager import OrderManager
        
        # Mock risk manager
        risk_manager = Mock()
        risk_manager.validate_trade.return_value = Mock(is_valid=True, message="OK")
        
        # Mock database
        database = Mock()
        
        manager = OrderManager(binance_client, risk_manager, database)
        return manager
    
    def test_order_manager_initialization(self, order_manager):
        """OrderManager başlatma testi"""
        assert order_manager is not None
        assert order_manager.binance_client is not None
        assert order_manager.risk_manager is not None
        assert order_manager.database is not None
    
    def test_execute_market_buy_order(self, order_manager, mock_binance_client):
        """Market buy order execution testi"""
        # Mock order response
        mock_order = {
            'symbol': 'BTCUSDT',
            'orderId': 123456,
            'side': 'BUY',
            'executedQty': '0.01',
            'cummulativeQuoteQty': '450.0',
            'status': 'FILLED'
        }
        mock_binance_client.order_market_buy.return_value = mock_order
        
        # Test execute order
        result = order_manager.execute_market_order('BTCUSDT', 'BUY', 0.01)
        
        # Doğrulama
        assert result is not None
        assert result['success'] == True
        assert result['order']['symbol'] == 'BTCUSDT'
        assert result['order']['side'] == 'BUY'
    
    def test_execute_market_sell_order(self, order_manager, mock_binance_client):
        """Market sell order execution testi"""
        # Mock order response
        mock_order = {
            'symbol': 'BTCUSDT',
            'orderId': 123457,
            'side': 'SELL',
            'executedQty': '0.01',
            'cummulativeQuoteQty': '450.0',
            'status': 'FILLED'
        }
        mock_binance_client.order_market_sell.return_value = mock_order
        
        # Test execute order
        result = order_manager.execute_market_order('BTCUSDT', 'SELL', 0.01)
        
        # Doğrulama
        assert result is not None
        assert result['success'] == True
        assert result['order']['symbol'] == 'BTCUSDT'
        assert result['order']['side'] == 'SELL'
    
    def test_execute_limit_order(self, order_manager, mock_binance_client):
        """Limit order execution testi"""
        # Mock order response
        mock_order = {
            'symbol': 'BTCUSDT',
            'orderId': 123458,
            'side': 'BUY',
            'price': '44000.0',
            'origQty': '0.01',
            'status': 'NEW'
        }
        mock_binance_client.order_limit_buy.return_value = mock_order
        
        # Test execute limit order
        result = order_manager.execute_limit_order('BTCUSDT', 'BUY', 0.01, 44000.0)
        
        # Doğrulama
        assert result is not None
        assert result['success'] == True
        assert result['order']['type'] == 'LIMIT'
        assert result['order']['price'] == 44000.0
    
    def test_risk_validation_failure(self, order_manager):
        """Risk doğrulama hatası testi"""
        # Mock risk validation failure
        order_manager.risk_manager.validate_trade.return_value = Mock(
            is_valid=False, 
            message="Risk limit exceeded"
        )
        
        # Test execute order with risk failure
        result = order_manager.execute_market_order('BTCUSDT', 'BUY', 10.0)  # Large amount
        
        # Doğrulama
        assert result is not None
        assert result['success'] == False
        assert "risk" in result['error'].lower()
    
    def test_order_execution_failure(self, order_manager, mock_binance_client):
        """Order execution hatası testi"""
        # Mock order exception
        mock_binance_client.order_market_buy.side_effect = Exception("Insufficient balance")
        
        # Test execute order with failure
        result = order_manager.execute_market_order('BTCUSDT', 'BUY', 0.01)
        
        # Doğrulama
        assert result is not None
        assert result['success'] == False
        assert "error" in result
    
    def test_cancel_order(self, order_manager, mock_binance_client):
        """Order iptal testi"""
        # Mock cancel response
        mock_cancel = {
            'symbol': 'BTCUSDT',
            'orderId': 123456,
            'status': 'CANCELED'
        }
        mock_binance_client.cancel_order.return_value = mock_cancel
        
        # Test cancel order
        result = order_manager.cancel_order('BTCUSDT', 123456)
        
        # Doğrulama
        assert result is not None
        assert result['success'] == True
        assert result['order']['status'] == 'CANCELED'
    
    def test_get_order_status(self, order_manager, mock_binance_client):
        """Order durum kontrolü testi"""
        # Mock order status response
        mock_status = {
            'symbol': 'BTCUSDT',
            'orderId': 123456,
            'status': 'PARTIALLY_FILLED',
            'executedQty': '0.005',
            'origQty': '0.01'
        }
        mock_binance_client.get_order.return_value = mock_status
        
        # Test get order status
        result = order_manager.get_order_status('BTCUSDT', 123456)
        
        # Doğrulama
        assert result is not None
        assert result['status'] == 'PARTIALLY_FILLED'
        assert result['executedQty'] == 0.005
        assert result['origQty'] == 0.01


class TestBinanceIntegration:
    """Binance entegrasyon testleri"""
    
    def test_full_trading_flow(self, mock_binance_client):
        """Tam trading akışı testi"""
        with patch('binance.client.Client', return_value=mock_binance_client):
            from exchange.binance_client import BinanceClient
            from exchange.order_manager import OrderManager
            
            # Mock responses
            mock_binance_client.ping.return_value = {}
            mock_binance_client.get_account.return_value = {
                'balances': [{'asset': 'USDT', 'free': '1000.0', 'locked': '0.0'}]
            }
            mock_binance_client.get_symbol_ticker.return_value = {'price': '45000.0'}
            mock_binance_client.order_market_buy.return_value = {
                'symbol': 'BTCUSDT',
                'orderId': 123456,
                'status': 'FILLED',
                'executedQty': '0.01'
            }
            
            # Setup
            config = Mock()
            config.BINANCE_API_KEY = 'test_key'
            config.BINANCE_SECRET_KEY = 'test_secret'
            config.BINANCE_TESTNET = True
            
            binance_client = BinanceClient(config)
            
            # Mock dependencies
            risk_manager = Mock()
            risk_manager.validate_trade.return_value = Mock(is_valid=True)
            database = Mock()
            
            order_manager = OrderManager(binance_client, risk_manager, database)
            
            # Test full flow
            # 1. Test connection
            assert binance_client.test_connection() == True
            
            # 2. Get account info
            account = binance_client.get_account_info()
            assert account is not None
            
            # 3. Get price
            price = binance_client.get_symbol_price('BTCUSDT')
            assert price == 45000.0
            
            # 4. Execute order
            result = order_manager.execute_market_order('BTCUSDT', 'BUY', 0.01)
            assert result['success'] == True
    
    def test_error_handling_integration(self, mock_binance_client):
        """Hata yönetimi entegrasyon testi"""
        with patch('binance.client.Client', return_value=mock_binance_client):
            from exchange.binance_client import BinanceClient
            
            # Mock network error
            mock_binance_client.ping.side_effect = Exception("Network error")
            
            config = Mock()
            config.BINANCE_API_KEY = 'test_key'
            config.BINANCE_SECRET_KEY = 'test_secret'
            config.BINANCE_TESTNET = True
            
            binance_client = BinanceClient(config)
            
            # Test error handling
            assert binance_client.test_connection() == False
    
    def test_data_validation(self, binance_client):
        """Veri doğrulama testi"""
        # Test symbol validation
        with pytest.raises(ValueError):
            binance_client.get_symbol_price('')  # Empty symbol
        
        with pytest.raises(ValueError):
            binance_client.get_symbol_price('INVALID_SYMBOL_TOO_LONG')  # Too long
        
        # Test quantity validation
        with pytest.raises(ValueError):
            binance_client.place_market_order('BTCUSDT', 'BUY', 0)  # Zero quantity
        
        with pytest.raises(ValueError):
            binance_client.place_market_order('BTCUSDT', 'BUY', -1)  # Negative quantity


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
