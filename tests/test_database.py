"""
Database Tests
Veritabanı fonksiyonlarının test edilmesi
"""

import pytest
import sqlite3
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List
import tempfile
import os

from tests import TestUtils, PerformanceTimer, get_memory_usage, generate_mock_trade_data


class TestAdvancedDatabaseManager:
    """AdvancedDatabaseManager sınıfı testleri"""
    
    @pytest.fixture
    def temp_db(self):
        """Geçici test veritabanı oluştur"""
        # Temporary file for testing
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass
    
    @pytest.fixture
    def database_manager(self, temp_db):
        """DatabaseManager instance oluştur"""
        from utils.database import AdvancedDatabaseManager
        
        # Mock config to use temp database
        with patch('utils.database.os.getenv') as mock_getenv:
            mock_getenv.return_value = f'sqlite:///{temp_db}'
            
            db_manager = AdvancedDatabaseManager()
            db_manager.initialize()
            
            return db_manager
    
    def test_database_initialization(self, database_manager):
        """Veritabanı başlatma testi"""
        assert database_manager is not None
        assert database_manager.engine is not None
        assert database_manager.Session is not None
    
    def test_create_tables(self, database_manager):
        """Tablo oluşturma testi"""
        # Tables should be created during initialization
        session = database_manager.Session()
        
        try:
            # Check if tables exist by querying them
            result = session.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in result.fetchall()]
            
            # Expected tables
            expected_tables = [
                'trades', 'signals', 'portfolio', 'settings', 
                'logs', 'performance_metrics', 'model_performance'
            ]
            
            for table in expected_tables:
                assert table in tables, f"Table {table} not found"
                
        finally:
            session.close()
    
    def test_add_trade(self, database_manager):
        """Trade ekleme testi"""
        # Test trade data
        trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.01,
            'price': 45000.0,
            'pnl': 150.0,
            'strategy': 'ensemble',
            'timestamp': datetime.now()
        }
        
        # Add trade
        trade_id = database_manager.add_trade(trade_data)
        
        # Doğrulama
        assert trade_id is not None
        assert isinstance(trade_id, int)
        assert trade_id > 0
    
    def test_get_trades(self, database_manager):
        """Trade alma testi"""
        # Add some test trades
        for i in range(5):
            trade_data = {
                'symbol': 'BTCUSDT',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'quantity': 0.01 + (i * 0.001),
                'price': 45000.0 + (i * 100),
                'pnl': 50.0 + (i * 10),
                'strategy': 'ensemble',
                'timestamp': datetime.now() - timedelta(hours=i)
            }
            database_manager.add_trade(trade_data)
        
        # Get trades
        trades = database_manager.get_trades(limit=3)
        
        # Doğrulama
        assert len(trades) == 3
        assert all('symbol' in trade for trade in trades)
        assert all('pnl' in trade for trade in trades)
        
        # Test with symbol filter
        btc_trades = database_manager.get_trades(symbol='BTCUSDT')
        assert len(btc_trades) == 5
    
    def test_add_signal(self, database_manager):
        """Sinyal ekleme testi"""
        # Test signal data
        signal_data = {
            'model': 'LSTM',
            'symbol': 'BTCUSDT',
            'signal': 'BUY',
            'confidence': 0.85,
            'price': 45000.0,
            'details': {'rsi': 65, 'macd': 'bullish'},
            'timestamp': datetime.now()
        }
        
        # Add signal
        signal_id = database_manager.add_signal(signal_data)
        
        # Doğrulama
        assert signal_id is not None
        assert isinstance(signal_id, int)
        assert signal_id > 0
    
    def test_get_signals(self, database_manager):
        """Sinyal alma testi"""
        # Add some test signals
        models = ['LSTM', 'Technical', 'Sentiment']
        signals = ['BUY', 'SELL', 'HOLD']
        
        for i in range(6):
            signal_data = {
                'model': models[i % 3],
                'symbol': 'BTCUSDT',
                'signal': signals[i % 3],
                'confidence': 0.7 + (i * 0.05),
                'price': 45000.0 + (i * 50),
                'timestamp': datetime.now() - timedelta(minutes=i * 10)
            }
            database_manager.add_signal(signal_data)
        
        # Get signals
        all_signals = database_manager.get_signals(limit=4)
        assert len(all_signals) == 4
        
        # Test with model filter
        lstm_signals = database_manager.get_signals(model='LSTM')
        assert len(lstm_signals) == 2
        assert all(signal['model'] == 'LSTM' for signal in lstm_signals)
    
    def test_update_portfolio(self, database_manager):
        """Portföy güncelleme testi"""
        # Test portfolio data
        portfolio_data = {
            'symbol': 'BTCUSDT',
            'quantity': 0.05,
            'avg_price': 44500.0,
            'current_price': 45000.0,
            'unrealized_pnl': 25.0
        }
        
        # Update portfolio
        result = database_manager.update_portfolio(portfolio_data)
        
        # Doğrulama
        assert result == True
        
        # Get portfolio to verify
        portfolio = database_manager.get_portfolio()
        assert len(portfolio) == 1
        assert portfolio[0]['symbol'] == 'BTCUSDT'
        assert portfolio[0]['quantity'] == 0.05
    
    def test_get_portfolio(self, database_manager):
        """Portföy alma testi"""
        # Add multiple portfolio positions
        positions = [
            {
                'symbol': 'BTCUSDT',
                'quantity': 0.1,
                'avg_price': 45000.0,
                'current_price': 46000.0,
                'unrealized_pnl': 100.0
            },
            {
                'symbol': 'ETHUSDT',
                'quantity': 2.0,
                'avg_price': 3000.0,
                'current_price': 3100.0,
                'unrealized_pnl': 200.0
            }
        ]
        
        for position in positions:
            database_manager.update_portfolio(position)
        
        # Get portfolio
        portfolio = database_manager.get_portfolio()
        
        # Doğrulama
        assert len(portfolio) == 2
        symbols = [pos['symbol'] for pos in portfolio]
        assert 'BTCUSDT' in symbols
        assert 'ETHUSDT' in symbols
    
    def test_add_performance_metric(self, database_manager):
        """Performans metriği ekleme testi"""
        # Test performance data
        performance_data = {
            'date': datetime.now().date(),
            'total_return': 0.15,
            'daily_return': 0.02,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.08,
            'win_rate': 0.65,
            'total_trades': 25
        }
        
        # Add performance metric
        result = database_manager.add_performance_metric(performance_data)
        
        # Doğrulama
        assert result == True
    
    def test_get_performance_metrics(self, database_manager):
        """Performans metriği alma testi"""
        # Add some performance data
        for i in range(7):
            performance_data = {
                'date': (datetime.now() - timedelta(days=i)).date(),
                'total_return': 0.10 + (i * 0.01),
                'daily_return': 0.01 + (i * 0.002),
                'sharpe_ratio': 1.2 + (i * 0.1),
                'max_drawdown': 0.05 + (i * 0.005),
                'win_rate': 0.60 + (i * 0.01),
                'total_trades': 20 + i
            }
            database_manager.add_performance_metric(performance_data)
        
        # Get performance metrics
        metrics = database_manager.get_performance_metrics(days=5)
        
        # Doğrulama
        assert len(metrics) == 5
        assert all('total_return' in metric for metric in metrics)
        assert all('sharpe_ratio' in metric for metric in metrics)
    
    def test_add_log(self, database_manager):
        """Log ekleme testi"""
        # Test log data
        log_data = {
            'level': 'INFO',
            'message': 'Test log message',
            'module': 'test_module',
            'timestamp': datetime.now()
        }
        
        # Add log
        result = database_manager.add_log(log_data)
        
        # Doğrulama
        assert result == True
    
    def test_get_logs(self, database_manager):
        """Log alma testi"""
        # Add some test logs
        log_levels = ['INFO', 'WARNING', 'ERROR']
        
        for i in range(6):
            log_data = {
                'level': log_levels[i % 3],
                'message': f'Test log message {i}',
                'module': 'test_module',
                'timestamp': datetime.now() - timedelta(minutes=i)
            }
            database_manager.add_log(log_data)
        
        # Get logs
        all_logs = database_manager.get_logs(limit=4)
        assert len(all_logs) == 4
        
        # Test with level filter
        error_logs = database_manager.get_logs(level='ERROR')
        assert len(error_logs) == 2
        assert all(log['level'] == 'ERROR' for log in error_logs)
    
    def test_set_get_setting(self, database_manager):
        """Ayar kaydetme/alma testi"""
        # Set setting
        result = database_manager.set_setting('test_key', 'test_value')
        assert result == True
        
        # Get setting
        value = database_manager.get_setting('test_key')
        assert value == 'test_value'
        
        # Get non-existent setting
        non_existent = database_manager.get_setting('non_existent_key', 'default')
        assert non_existent == 'default'
    
    def test_backup_database(self, database_manager, temp_db):
        """Veritabanı yedekleme testi"""
        # Add some data first
        trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.01,
            'price': 45000.0,
            'pnl': 150.0,
            'strategy': 'ensemble',
            'timestamp': datetime.now()
        }
        database_manager.add_trade(trade_data)
        
        # Create backup
        backup_path = temp_db + '.backup'
        result = database_manager.backup_database(backup_path)
        
        # Doğrulama
        assert result == True
        assert os.path.exists(backup_path)
        
        # Cleanup
        try:
            os.unlink(backup_path)
        except:
            pass
    
    def test_get_trade_statistics(self, database_manager):
        """Trade istatistikleri testi"""
        # Add some test trades with varying P&L
        pnl_values = [100, -50, 75, -25, 150, -30, 200]
        
        for i, pnl in enumerate(pnl_values):
            trade_data = {
                'symbol': 'BTCUSDT',
                'side': 'BUY' if pnl > 0 else 'SELL',
                'quantity': 0.01,
                'price': 45000.0,
                'pnl': pnl,
                'strategy': 'ensemble',
                'timestamp': datetime.now() - timedelta(hours=i)
            }
            database_manager.add_trade(trade_data)
        
        # Get statistics
        stats = database_manager.get_trade_statistics()
        
        # Doğrulama
        assert 'total_trades' in stats
        assert 'winning_trades' in stats
        assert 'losing_trades' in stats
        assert 'total_pnl' in stats
        assert 'win_rate' in stats
        assert 'avg_win' in stats
        assert 'avg_loss' in stats
        
        assert stats['total_trades'] == 7
        assert stats['winning_trades'] == 4  # Positive P&L
        assert stats['losing_trades'] == 3   # Negative P&L
        assert stats['total_pnl'] == sum(pnl_values)
    
    def test_performance_large_dataset(self, database_manager):
        """Büyük veri seti performans testi"""
        with PerformanceTimer() as timer:
            # Add 1000 trades
            for i in range(1000):
                trade_data = {
                    'symbol': 'BTCUSDT',
                    'side': 'BUY' if i % 2 == 0 else 'SELL',
                    'quantity': 0.01,
                    'price': 45000.0 + i,
                    'pnl': (-50 + (i % 100)),
                    'strategy': 'ensemble',
                    'timestamp': datetime.now() - timedelta(seconds=i)
                }
                database_manager.add_trade(trade_data)
        
        # 1000 trade ekleme 5 saniyeden az sürmeli
        assert timer.elapsed < 5.0
        
        # Query performance test
        with PerformanceTimer() as query_timer:
            trades = database_manager.get_trades(limit=100)
            stats = database_manager.get_trade_statistics()
        
        # Query 1 saniyeden az sürmeli
        assert query_timer.elapsed < 1.0
        assert len(trades) == 100
    
    def test_memory_usage_database_operations(self, database_manager):
        """Veritabanı işlemleri memory kullanım testi"""
        initial_memory = get_memory_usage()
        
        # Perform many database operations
        for i in range(500):
            # Add trade
            trade_data = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.01,
                'price': 45000.0 + i,
                'pnl': 50.0,
                'strategy': 'ensemble',
                'timestamp': datetime.now()
            }
            database_manager.add_trade(trade_data)
            
            # Add signal
            signal_data = {
                'model': 'LSTM',
                'symbol': 'BTCUSDT',
                'signal': 'BUY',
                'confidence': 0.8,
                'price': 45000.0 + i,
                'timestamp': datetime.now()
            }
            database_manager.add_signal(signal_data)
        
        current_memory = get_memory_usage()
        memory_increase = current_memory - initial_memory
        
        # Memory artışı 100MB'dan az olmalı
        assert memory_increase < 100
    
    def test_transaction_rollback(self, database_manager):
        """Transaction rollback testi"""
        # Get initial trade count
        initial_trades = database_manager.get_trades()
        initial_count = len(initial_trades)
        
        # Test transaction rollback
        session = database_manager.Session()
        try:
            # Add a trade
            from utils.database import Trade
            trade = Trade(
                symbol='BTCUSDT',
                side='BUY',
                quantity=0.01,
                price=45000.0,
                pnl=100.0,
                strategy='test'
            )
            session.add(trade)
            session.flush()  # This should assign an ID
            
            # Simulate an error
            raise Exception("Simulated error")
            
        except Exception:
            session.rollback()
        finally:
            session.close()
        
        # Verify rollback worked
        final_trades = database_manager.get_trades()
        final_count = len(final_trades)
        
        assert final_count == initial_count  # No new trades should be added
    
    def test_concurrent_access(self, database_manager):
        """Eşzamanlı erişim testi"""
        import threading
        import time
        
        results = []
        
        def add_trades(thread_id):
            """Thread function to add trades"""
            try:
                for i in range(10):
                    trade_data = {
                        'symbol': f'THREAD{thread_id}',
                        'side': 'BUY',
                        'quantity': 0.01,
                        'price': 45000.0 + i,
                        'pnl': 50.0,
                        'strategy': 'concurrent_test',
                        'timestamp': datetime.now()
                    }
                    trade_id = database_manager.add_trade(trade_data)
                    results.append(trade_id)
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                results.append(f"Error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_trades, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        successful_results = [r for r in results if isinstance(r, int)]
        assert len(successful_results) == 30  # 3 threads * 10 trades each
        
        # Verify all trades were added
        all_trades = database_manager.get_trades()
        concurrent_trades = [t for t in all_trades if t['strategy'] == 'concurrent_test']
        assert len(concurrent_trades) == 30


class TestDatabaseIntegration:
    """Veritabanı entegrasyon testleri"""
    
    def test_full_trading_workflow(self, temp_db):
        """Tam trading workflow veritabanı testi"""
        from utils.database import AdvancedDatabaseManager
        
        # Mock config
        with patch('utils.database.os.getenv') as mock_getenv:
            mock_getenv.return_value = f'sqlite:///{temp_db}'
            
            db_manager = AdvancedDatabaseManager()
            db_manager.initialize()
            
            # 1. Add initial settings
            db_manager.set_setting('initial_capital', '10000.0')
            db_manager.set_setting('max_risk', '0.02')
            
            # 2. Add AI model signal
            signal_data = {
                'model': 'LSTM',
                'symbol': 'BTCUSDT',
                'signal': 'BUY',
                'confidence': 0.85,
                'price': 45000.0,
                'timestamp': datetime.now()
            }
            signal_id = db_manager.add_signal(signal_data)
            assert signal_id is not None
            
            # 3. Execute trade based on signal
            trade_data = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.01,
                'price': 45000.0,
                'pnl': 0.0,  # Initial P&L
                'strategy': 'lstm_signal',
                'timestamp': datetime.now()
            }
            trade_id = db_manager.add_trade(trade_data)
            assert trade_id is not None
            
            # 4. Update portfolio
            portfolio_data = {
                'symbol': 'BTCUSDT',
                'quantity': 0.01,
                'avg_price': 45000.0,
                'current_price': 45000.0,
                'unrealized_pnl': 0.0
            }
            db_manager.update_portfolio(portfolio_data)
            
            # 5. Simulate price change and update
            portfolio_data['current_price'] = 46000.0
            portfolio_data['unrealized_pnl'] = 10.0  # $1000 profit
            db_manager.update_portfolio(portfolio_data)
            
            # 6. Close position (sell)
            close_trade_data = {
                'symbol': 'BTCUSDT',
                'side': 'SELL',
                'quantity': 0.01,
                'price': 46000.0,
                'pnl': 10.0,
                'strategy': 'take_profit',
                'timestamp': datetime.now()
            }
            close_trade_id = db_manager.add_trade(close_trade_data)
            assert close_trade_id is not None
            
            # 7. Add performance metrics
            performance_data = {
                'date': datetime.now().date(),
                'total_return': 0.001,  # 0.1% return
                'daily_return': 0.001,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.0,
                'win_rate': 1.0,  # 100% win rate (1 winning trade)
                'total_trades': 2
            }
            db_manager.add_performance_metric(performance_data)
            
            # 8. Verify complete workflow
            trades = db_manager.get_trades()
            assert len(trades) == 2
            
            signals = db_manager.get_signals()
            assert len(signals) == 1
            
            portfolio = db_manager.get_portfolio()
            assert len(portfolio) == 1
            
            stats = db_manager.get_trade_statistics()
            assert stats['total_trades'] == 2
            assert stats['total_pnl'] == 10.0
            
            settings = db_manager.get_setting('initial_capital')
            assert settings == '10000.0'
    
    def test_database_migration_simulation(self, temp_db):
        """Veritabanı migration simülasyonu testi"""
        from utils.database import AdvancedDatabaseManager
        
        # Create database with old schema (simulate)
        with patch('utils.database.os.getenv') as mock_getenv:
            mock_getenv.return_value = f'sqlite:///{temp_db}'
            
            # First version of database
            db_manager_v1 = AdvancedDatabaseManager()
            db_manager_v1.initialize()
            
            # Add some data
            trade_data = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.01,
                'price': 45000.0,
                'pnl': 100.0,
                'strategy': 'v1_strategy',
                'timestamp': datetime.now()
            }
            trade_id = db_manager_v1.add_trade(trade_data)
            assert trade_id is not None
            
            # Simulate upgrade (reinitialize)
            db_manager_v2 = AdvancedDatabaseManager()
            db_manager_v2.initialize()
            
            # Verify data is still accessible
            trades = db_manager_v2.get_trades()
            assert len(trades) >= 1
            assert any(trade['strategy'] == 'v1_strategy' for trade in trades)
    
    def test_database_error_recovery(self, temp_db):
        """Veritabanı hata kurtarma testi"""
        from utils.database import AdvancedDatabaseManager
        
        with patch('utils.database.os.getenv') as mock_getenv:
            mock_getenv.return_value = f'sqlite:///{temp_db}'
            
            db_manager = AdvancedDatabaseManager()
            db_manager.initialize()
            
            # Add some data
            trade_data = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.01,
                'price': 45000.0,
                'pnl': 100.0,
                'strategy': 'test',
                'timestamp': datetime.now()
            }
            db_manager.add_trade(trade_data)
            
            # Simulate database corruption by closing connection
            db_manager.engine.dispose()
            
            # Try to reconnect and verify data recovery
            db_manager_recovered = AdvancedDatabaseManager()
            db_manager_recovered.initialize()
            
            trades = db_manager_recovered.get_trades()
            assert len(trades) >= 1


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
