"""
Test Suite for Trading Bot
Trading bot için test paketi
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Test ortamı için path ayarları
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test konfigürasyonu
TEST_CONFIG = {
    'TELEGRAM_BOT_TOKEN': 'test_token_123456789:ABCdefGHIjklMNOpqrsTUVwxyz',
    'TELEGRAM_CHAT_ID': '123456789',
    'BINANCE_API_KEY': 'test_api_key_' + 'A' * 50,
    'BINANCE_SECRET_KEY': 'test_secret_key_' + 'B' * 46,
    'BINANCE_TESTNET': 'true',
    'DATABASE_URL': 'sqlite:///:memory:',
    'LOG_LEVEL': 'DEBUG',
    'TRADING_ENABLED': 'false',
    'MOCK_TRADING': 'true'
}

# Test environment setup
def setup_test_environment():
    """Test ortamını hazırla"""
    for key, value in TEST_CONFIG.items():
        os.environ[key] = value

# Mock data generators
def generate_mock_price_data(symbol: str = 'BTCUSDT', days: int = 30) -> List[Dict[str, Any]]:
    """Mock fiyat verisi oluştur"""
    import random
    from datetime import datetime, timedelta
    
    data = []
    base_price = 45000.0 if 'BTC' in symbol else 3000.0
    current_time = datetime.now()
    
    for i in range(days * 24):  # Saatlik veri
        timestamp = current_time - timedelta(hours=i)
        
        # Random price movement
        change = random.uniform(-0.05, 0.05)
        price = base_price * (1 + change)
        
        data.append({
            'timestamp': timestamp,
            'open': price * 0.999,
            'high': price * 1.002,
            'low': price * 0.998,
            'close': price,
            'volume': random.uniform(100, 1000)
        })
        
        base_price = price
    
    return list(reversed(data))

def generate_mock_trade_data(count: int = 50) -> List[Dict[str, Any]]:
    """Mock trade verisi oluştur"""
    import random
    from datetime import datetime, timedelta
    
    trades = []
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    for i in range(count):
        symbol = random.choice(symbols)
        side = random.choice(['BUY', 'SELL'])
        quantity = random.uniform(0.001, 1.0)
        price = random.uniform(30000, 50000) if 'BTC' in symbol else random.uniform(2000, 4000)
        pnl = random.uniform(-100, 200)
        
        trades.append({
            'id': i + 1,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'pnl': pnl,
            'timestamp': datetime.now() - timedelta(hours=i),
            'strategy': random.choice(['lstm', 'technical', 'ensemble'])
        })
    
    return trades

# Test fixtures
@pytest.fixture
def mock_telegram_bot():
    """Mock Telegram bot"""
    with patch('telegram.Bot') as mock_bot:
        mock_bot.return_value.get_me.return_value = Mock(username='test_bot')
        mock_bot.return_value.send_message.return_value = Mock(message_id=123)
        yield mock_bot.return_value

@pytest.fixture
def mock_binance_client():
    """Mock Binance client"""
    with patch('binance.client.Client') as mock_client:
        mock_client.return_value.ping.return_value = {}
        mock_client.return_value.get_account.return_value = {
            'balances': [
                {'asset': 'USDT', 'free': '1000.0', 'locked': '0.0'},
                {'asset': 'BTC', 'free': '0.1', 'locked': '0.0'}
            ]
        }
        mock_client.return_value.get_symbol_ticker.return_value = {'price': '45000.0'}
        yield mock_client.return_value

@pytest.fixture
def mock_database():
    """Mock database"""
    from utils.database import AdvancedDatabaseManager
    
    with patch.object(AdvancedDatabaseManager, '__init__', return_value=None):
        db = AdvancedDatabaseManager()
        db.engine = Mock()
        db.Session = Mock()
        yield db

@pytest.fixture
def sample_price_data():
    """Sample price data for testing"""
    return generate_mock_price_data()

@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing"""
    return generate_mock_trade_data()

# Test utilities
class TestUtils:
    """Test yardımcı fonksiyonları"""
    
    @staticmethod
    def assert_valid_signal(signal: Dict[str, Any]):
        """Sinyal formatını doğrula"""
        required_fields = ['symbol', 'signal', 'confidence', 'timestamp']
        for field in required_fields:
            assert field in signal, f"Missing field: {field}"
        
        assert signal['signal'] in ['BUY', 'SELL', 'HOLD'], f"Invalid signal: {signal['signal']}"
        assert 0 <= signal['confidence'] <= 1, f"Invalid confidence: {signal['confidence']}"
    
    @staticmethod
    def assert_valid_trade(trade: Dict[str, Any]):
        """Trade formatını doğrula"""
        required_fields = ['symbol', 'side', 'quantity', 'price']
        for field in required_fields:
            assert field in trade, f"Missing field: {field}"
        
        assert trade['side'] in ['BUY', 'SELL'], f"Invalid side: {trade['side']}"
        assert trade['quantity'] > 0, f"Invalid quantity: {trade['quantity']}"
        assert trade['price'] > 0, f"Invalid price: {trade['price']}"
    
    @staticmethod
    def assert_valid_risk_metrics(metrics: Dict[str, Any]):
        """Risk metriklerini doğrula"""
        required_fields = ['max_drawdown', 'current_drawdown', 'var_95']
        for field in required_fields:
            assert field in metrics, f"Missing field: {field}"
        
        assert 0 <= metrics['current_drawdown'] <= 1, f"Invalid drawdown: {metrics['current_drawdown']}"

# Performance testing utilities
class PerformanceTimer:
    """Performance ölçüm aracı"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.end_time = time.time()
    
    @property
    def elapsed(self):
        """Geçen süre (saniye)"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

# Memory monitoring
def get_memory_usage():
    """Memory kullanımını al"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Test data cleanup
def cleanup_test_data():
    """Test verilerini temizle"""
    import tempfile
    import shutil
    
    # Temporary files cleanup
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith('test_trading_bot'):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except:
                pass

# Setup test environment on import
setup_test_environment()
