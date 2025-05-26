"""
Risk Management Tests
Risk yönetimi fonksiyonlarının test edilmesi
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List

from tests import TestUtils, PerformanceTimer, get_memory_usage, generate_mock_trade_data


class TestRiskManager:
    """RiskManager sınıfı testleri"""
    
    @pytest.fixture
    def risk_manager(self):
        """RiskManager instance oluştur"""
        from risk.risk_manager_complete import AdvancedRiskManager
        
        # Mock config
        config = Mock()
        config.INITIAL_CAPITAL = 10000.0
        config.MAX_RISK_PER_TRADE = 0.02
        config.MAX_DAILY_LOSS = 0.05
        config.MAX_DRAWDOWN = 0.15
        config.POSITION_SIZE_METHOD = 'volatility_based'
        config.EMERGENCY_STOP_ENABLED = True
        config.CIRCUIT_BREAKER_THRESHOLD = 0.10
        
        # Mock database
        database = Mock()
        database.get_trades.return_value = generate_mock_trade_data(50)
        database.get_portfolio.return_value = []
        
        risk_manager = AdvancedRiskManager(config, database)
        return risk_manager
    
    def test_risk_manager_initialization(self, risk_manager):
        """Risk manager başlatma testi"""
        assert risk_manager is not None
        assert risk_manager.initial_capital == 10000.0
        assert risk_manager.max_risk_per_trade == 0.02
        assert risk_manager.max_daily_loss == 0.05
        assert risk_manager.max_drawdown == 0.15
        assert risk_manager.emergency_stop_enabled == True
    
    def test_validate_trade_valid(self, risk_manager):
        """Geçerli trade doğrulama testi"""
        # Valid trade data
        trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.01,
            'price': 45000.0,
            'strategy': 'ensemble'
        }
        
        # Validate trade
        result = risk_manager.validate_trade(trade_data)
        
        # Doğrulama
        assert result.is_valid == True
        assert result.message == "Trade approved"
        assert result.risk_score < 1.0
    
    def test_validate_trade_excessive_risk(self, risk_manager):
        """Aşırı risk trade doğrulama testi"""
        # High risk trade data
        trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 5.0,  # Very large quantity
            'price': 45000.0,
            'strategy': 'ensemble'
        }
        
        # Validate trade
        result = risk_manager.validate_trade(trade_data)
        
        # Doğrulama
        assert result.is_valid == False
        assert "risk" in result.message.lower()
        assert result.risk_score > 0.5
    
    def test_calculate_position_size_fixed(self, risk_manager):
        """Sabit pozisyon boyutu hesaplama testi"""
        risk_manager.position_size_method = 'fixed'
        
        # Calculate position size
        position_size = risk_manager.calculate_position_size(
            symbol='BTCUSDT',
            price=45000.0,
            risk_amount=100.0
        )
        
        # Doğrulama
        assert position_size > 0
        assert isinstance(position_size, float)
    
    def test_calculate_position_size_volatility_based(self, risk_manager):
        """Volatilite bazlı pozisyon boyutu hesaplama testi"""
        risk_manager.position_size_method = 'volatility_based'
        
        # Mock volatility calculation
        with patch.object(risk_manager, 'calculate_volatility', return_value=0.05):
            position_size = risk_manager.calculate_position_size(
                symbol='BTCUSDT',
                price=45000.0,
                risk_amount=100.0
            )
            
            # Doğrulama
            assert position_size > 0
            assert isinstance(position_size, float)
    
    def test_calculate_daily_pnl(self, risk_manager):
        """Günlük P&L hesaplama testi"""
        # Mock today's trades
        today_trades = [
            {'pnl': 50.0, 'timestamp': datetime.now()},
            {'pnl': -20.0, 'timestamp': datetime.now()},
            {'pnl': 30.0, 'timestamp': datetime.now()}
        ]
        
        risk_manager.database.get_trades.return_value = today_trades
        
        # Calculate daily P&L
        daily_pnl = risk_manager.calculate_daily_pnl()
        
        # Doğrulama
        assert daily_pnl == 60.0  # 50 - 20 + 30
    
    def test_calculate_drawdown(self, risk_manager):
        """Drawdown hesaplama testi"""
        # Mock equity curve
        equity_curve = [10000, 10500, 10200, 9800, 10100, 10300]
        
        # Calculate drawdown
        current_dd, max_dd = risk_manager.calculate_drawdown(equity_curve)
        
        # Doğrulama
        assert current_dd >= 0
        assert max_dd >= current_dd
        assert max_dd <= 1.0  # Should be percentage
    
    def test_check_daily_loss_limit(self, risk_manager):
        """Günlük kayıp limiti kontrolü testi"""
        # Test within limit
        risk_manager.current_daily_pnl = -300.0  # 3% loss
        within_limit = risk_manager.check_daily_loss_limit()
        assert within_limit == True
        
        # Test exceeding limit
        risk_manager.current_daily_pnl = -600.0  # 6% loss (exceeds 5% limit)
        exceeding_limit = risk_manager.check_daily_loss_limit()
        assert exceeding_limit == False
    
    def test_check_drawdown_limit(self, risk_manager):
        """Drawdown limiti kontrolü testi"""
        # Test within limit
        risk_manager.current_drawdown = 0.10  # 10% drawdown
        within_limit = risk_manager.check_drawdown_limit()
        assert within_limit == True
        
        # Test exceeding limit
        risk_manager.current_drawdown = 0.20  # 20% drawdown (exceeds 15% limit)
        exceeding_limit = risk_manager.check_drawdown_limit()
        assert exceeding_limit == False
    
    def test_calculate_portfolio_risk(self, risk_manager):
        """Portföy riski hesaplama testi"""
        # Mock portfolio positions
        portfolio = [
            {
                'symbol': 'BTCUSDT',
                'quantity': 0.1,
                'current_price': 45000.0,
                'unrealized_pnl': 500.0
            },
            {
                'symbol': 'ETHUSDT',
                'quantity': 2.0,
                'current_price': 3000.0,
                'unrealized_pnl': -200.0
            }
        ]
        
        risk_manager.database.get_portfolio.return_value = portfolio
        
        # Calculate portfolio risk
        portfolio_risk = risk_manager.calculate_portfolio_risk()
        
        # Doğrulama
        assert portfolio_risk >= 0
        assert portfolio_risk <= 1.0
    
    def test_calculate_var(self, risk_manager):
        """Value at Risk hesaplama testi"""
        # Mock returns data
        returns = [-0.02, 0.01, -0.01, 0.03, -0.015, 0.02, -0.005, 0.01]
        
        # Calculate VaR
        var_95 = risk_manager.calculate_var(returns, confidence_level=0.95)
        var_99 = risk_manager.calculate_var(returns, confidence_level=0.99)
        
        # Doğrulama
        assert var_95 < 0  # VaR should be negative (loss)
        assert var_99 < var_95  # 99% VaR should be more extreme
    
    def test_calculate_sharpe_ratio(self, risk_manager):
        """Sharpe ratio hesaplama testi"""
        # Mock returns data
        returns = [0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.02, 0.01]
        risk_free_rate = 0.02  # 2% annual
        
        # Calculate Sharpe ratio
        sharpe = risk_manager.calculate_sharpe_ratio(returns, risk_free_rate)
        
        # Doğrulama
        assert isinstance(sharpe, float)
        # Sharpe ratio can be positive or negative
    
    def test_emergency_stop_trigger(self, risk_manager):
        """Emergency stop tetikleme testi"""
        # Set conditions for emergency stop
        risk_manager.current_drawdown = 0.12  # 12% drawdown (exceeds 10% circuit breaker)
        
        # Check emergency stop
        should_stop = risk_manager.should_trigger_emergency_stop()
        
        # Doğrulama
        assert should_stop == True
    
    def test_emergency_stop_no_trigger(self, risk_manager):
        """Emergency stop tetiklenmeme testi"""
        # Set normal conditions
        risk_manager.current_drawdown = 0.05  # 5% drawdown (within limits)
        risk_manager.current_daily_pnl = -200.0  # 2% daily loss (within limits)
        
        # Check emergency stop
        should_stop = risk_manager.should_trigger_emergency_stop()
        
        # Doğrulama
        assert should_stop == False
    
    def test_risk_score_calculation(self, risk_manager):
        """Risk skoru hesaplama testi"""
        # Test low risk trade
        low_risk_trade = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.001,  # Small quantity
            'price': 45000.0
        }
        
        low_risk_score = risk_manager.calculate_risk_score(low_risk_trade)
        assert 0 <= low_risk_score <= 0.5
        
        # Test high risk trade
        high_risk_trade = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 1.0,  # Large quantity
            'price': 45000.0
        }
        
        high_risk_score = risk_manager.calculate_risk_score(high_risk_trade)
        assert high_risk_score > low_risk_score
    
    def test_correlation_analysis(self, risk_manager):
        """Korelasyon analizi testi"""
        # Mock price data for correlation
        btc_prices = [45000, 45500, 44800, 46000, 45200]
        eth_prices = [3000, 3100, 2950, 3150, 3050]
        
        # Calculate correlation
        correlation = risk_manager.calculate_correlation(btc_prices, eth_prices)
        
        # Doğrulama
        assert -1 <= correlation <= 1
        assert isinstance(correlation, float)
    
    def test_volatility_calculation(self, risk_manager):
        """Volatilite hesaplama testi"""
        # Mock price data
        prices = [45000, 45500, 44800, 46000, 45200, 46500, 45800]
        
        # Calculate volatility
        volatility = risk_manager.calculate_volatility(prices)
        
        # Doğrulama
        assert volatility > 0
        assert isinstance(volatility, float)
    
    def test_risk_metrics_update(self, risk_manager):
        """Risk metriklerini güncelleme testi"""
        # Update risk metrics
        risk_manager.update_risk_metrics()
        
        # Doğrulama
        assert hasattr(risk_manager, 'current_daily_pnl')
        assert hasattr(risk_manager, 'current_drawdown')
        assert hasattr(risk_manager, 'portfolio_risk')
    
    def test_performance_risk_calculation(self, risk_manager):
        """Risk hesaplama performans testi"""
        # Mock large trade data
        large_trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'price': 45000.0,
            'strategy': 'ensemble'
        }
        
        with PerformanceTimer() as timer:
            # 1000 risk validation
            for i in range(1000):
                risk_manager.validate_trade(large_trade_data)
        
        # 1000 validation 1 saniyeden az sürmeli
        assert timer.elapsed < 1.0
    
    def test_memory_usage_risk_calculations(self, risk_manager):
        """Risk hesaplama memory kullanım testi"""
        initial_memory = get_memory_usage()
        
        # Perform many risk calculations
        for i in range(1000):
            trade_data = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.01,
                'price': 45000.0 + i,  # Varying price
                'strategy': 'ensemble'
            }
            risk_manager.validate_trade(trade_data)
        
        current_memory = get_memory_usage()
        memory_increase = current_memory - initial_memory
        
        # Memory artışı 50MB'dan az olmalı
        assert memory_increase < 50


class TestRiskValidation:
    """Risk doğrulama testleri"""
    
    @pytest.fixture
    def risk_validator(self):
        """Risk validator oluştur"""
        from risk.risk_manager_complete import RiskValidator
        
        config = Mock()
        config.MAX_RISK_PER_TRADE = 0.02
        config.MAX_POSITION_SIZE = 1000.0
        config.MIN_POSITION_SIZE = 10.0
        
        validator = RiskValidator(config)
        return validator
    
    def test_validate_trade_amount(self, risk_validator):
        """Trade miktarı doğrulama testi"""
        # Valid amount
        valid_result = risk_validator.validate_trade_amount(500.0)
        assert valid_result.is_valid == True
        
        # Too small amount
        small_result = risk_validator.validate_trade_amount(5.0)
        assert small_result.is_valid == False
        assert "minimum" in small_result.message.lower()
        
        # Too large amount
        large_result = risk_validator.validate_trade_amount(2000.0)
        assert large_result.is_valid == False
        assert "maximum" in large_result.message.lower()
    
    def test_validate_symbol(self, risk_validator):
        """Sembol doğrulama testi"""
        # Valid symbols
        assert risk_validator.validate_symbol('BTCUSDT') == True
        assert risk_validator.validate_symbol('ETHUSDT') == True
        
        # Invalid symbols
        assert risk_validator.validate_symbol('INVALID') == False
        assert risk_validator.validate_symbol('') == False
        assert risk_validator.validate_symbol('BTC') == False  # Too short
    
    def test_validate_side(self, risk_validator):
        """Trade yönü doğrulama testi"""
        # Valid sides
        assert risk_validator.validate_side('BUY') == True
        assert risk_validator.validate_side('SELL') == True
        
        # Invalid sides
        assert risk_validator.validate_side('HOLD') == False
        assert risk_validator.validate_side('buy') == False  # Case sensitive
        assert risk_validator.validate_side('') == False
    
    def test_validate_price(self, risk_validator):
        """Fiyat doğrulama testi"""
        # Valid prices
        assert risk_validator.validate_price(45000.0) == True
        assert risk_validator.validate_price(0.001) == True
        
        # Invalid prices
        assert risk_validator.validate_price(0) == False
        assert risk_validator.validate_price(-100) == False
        assert risk_validator.validate_price(float('inf')) == False
    
    def test_validate_quantity(self, risk_validator):
        """Miktar doğrulama testi"""
        # Valid quantities
        assert risk_validator.validate_quantity(0.01) == True
        assert risk_validator.validate_quantity(1.0) == True
        
        # Invalid quantities
        assert risk_validator.validate_quantity(0) == False
        assert risk_validator.validate_quantity(-0.5) == False
        assert risk_validator.validate_quantity(float('nan')) == False


class TestRiskMetrics:
    """Risk metrikleri testleri"""
    
    @pytest.fixture
    def risk_metrics(self):
        """Risk metrics calculator oluştur"""
        from risk.risk_manager_complete import RiskMetrics
        
        metrics = RiskMetrics()
        return metrics
    
    def test_calculate_beta(self, risk_metrics):
        """Beta hesaplama testi"""
        # Mock market and asset returns
        market_returns = [0.01, 0.02, -0.01, 0.015, -0.005]
        asset_returns = [0.015, 0.025, -0.015, 0.02, -0.008]
        
        # Calculate beta
        beta = risk_metrics.calculate_beta(asset_returns, market_returns)
        
        # Doğrulama
        assert isinstance(beta, float)
        # Beta can be any value, but typically between 0 and 2 for most assets
    
    def test_calculate_alpha(self, risk_metrics):
        """Alpha hesaplama testi"""
        # Mock returns
        asset_returns = [0.015, 0.025, -0.015, 0.02, -0.008]
        market_returns = [0.01, 0.02, -0.01, 0.015, -0.005]
        risk_free_rate = 0.02
        
        # Calculate alpha
        alpha = risk_metrics.calculate_alpha(asset_returns, market_returns, risk_free_rate)
        
        # Doğrulama
        assert isinstance(alpha, float)
        # Alpha can be positive or negative
    
    def test_calculate_information_ratio(self, risk_metrics):
        """Information ratio hesaplama testi"""
        # Mock returns
        portfolio_returns = [0.015, 0.025, -0.015, 0.02, -0.008]
        benchmark_returns = [0.01, 0.02, -0.01, 0.015, -0.005]
        
        # Calculate information ratio
        info_ratio = risk_metrics.calculate_information_ratio(portfolio_returns, benchmark_returns)
        
        # Doğrulama
        assert isinstance(info_ratio, float)
    
    def test_calculate_sortino_ratio(self, risk_metrics):
        """Sortino ratio hesaplama testi"""
        # Mock returns
        returns = [0.01, 0.02, -0.01, 0.015, -0.005, 0.02, -0.015, 0.01]
        risk_free_rate = 0.02
        
        # Calculate Sortino ratio
        sortino = risk_metrics.calculate_sortino_ratio(returns, risk_free_rate)
        
        # Doğrulama
        assert isinstance(sortino, float)
    
    def test_calculate_calmar_ratio(self, risk_metrics):
        """Calmar ratio hesaplama testi"""
        # Mock returns and max drawdown
        annual_return = 0.15  # 15% annual return
        max_drawdown = 0.08   # 8% max drawdown
        
        # Calculate Calmar ratio
        calmar = risk_metrics.calculate_calmar_ratio(annual_return, max_drawdown)
        
        # Doğrulama
        assert isinstance(calmar, float)
        assert calmar > 0  # Should be positive for positive returns
    
    def test_calculate_maximum_adverse_excursion(self, risk_metrics):
        """Maximum Adverse Excursion hesaplama testi"""
        # Mock trade data
        trades = [
            {'entry_price': 45000, 'exit_price': 46000, 'lowest_price': 44500},
            {'entry_price': 46000, 'exit_price': 45500, 'lowest_price': 45000},
            {'entry_price': 45500, 'exit_price': 47000, 'lowest_price': 45200}
        ]
        
        # Calculate MAE
        mae = risk_metrics.calculate_mae(trades)
        
        # Doğrulama
        assert isinstance(mae, float)
        assert mae >= 0  # MAE should be non-negative
    
    def test_calculate_maximum_favorable_excursion(self, risk_metrics):
        """Maximum Favorable Excursion hesaplama testi"""
        # Mock trade data
        trades = [
            {'entry_price': 45000, 'exit_price': 46000, 'highest_price': 46500},
            {'entry_price': 46000, 'exit_price': 45500, 'highest_price': 46200},
            {'entry_price': 45500, 'exit_price': 47000, 'highest_price': 47500}
        ]
        
        # Calculate MFE
        mfe = risk_metrics.calculate_mfe(trades)
        
        # Doğrulama
        assert isinstance(mfe, float)
        assert mfe >= 0  # MFE should be non-negative


class TestRiskIntegration:
    """Risk yönetimi entegrasyon testleri"""
    
    def test_full_risk_pipeline(self):
        """Tam risk yönetimi pipeline testi"""
        from risk.risk_manager_complete import AdvancedRiskManager
        
        # Mock config
        config = Mock()
        config.INITIAL_CAPITAL = 10000.0
        config.MAX_RISK_PER_TRADE = 0.02
        config.MAX_DAILY_LOSS = 0.05
        config.MAX_DRAWDOWN = 0.15
        config.POSITION_SIZE_METHOD = 'volatility_based'
        config.EMERGENCY_STOP_ENABLED = True
        
        # Mock database
        database = Mock()
        database.get_trades.return_value = generate_mock_trade_data(20)
        database.get_portfolio.return_value = []
        
        # Create risk manager
        risk_manager = AdvancedRiskManager(config, database)
        
        # Test full pipeline
        trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.01,
            'price': 45000.0,
            'strategy': 'ensemble'
        }
        
        # 1. Validate trade
        validation_result = risk_manager.validate_trade(trade_data)
        assert validation_result.is_valid == True
        
        # 2. Calculate position size
        position_size = risk_manager.calculate_position_size(
            symbol='BTCUSDT',
            price=45000.0,
            risk_amount=100.0
        )
        assert position_size > 0
        
        # 3. Update risk metrics
        risk_manager.update_risk_metrics()
        
        # 4. Check emergency stop
        emergency_stop = risk_manager.should_trigger_emergency_stop()
        assert isinstance(emergency_stop, bool)
    
    def test_risk_manager_with_real_data_simulation(self):
        """Gerçek veri simülasyonu ile risk manager testi"""
        from risk.risk_manager_complete import AdvancedRiskManager
        
        # Mock config
        config = Mock()
        config.INITIAL_CAPITAL = 10000.0
        config.MAX_RISK_PER_TRADE = 0.02
        config.MAX_DAILY_LOSS = 0.05
        config.MAX_DRAWDOWN = 0.15
        config.POSITION_SIZE_METHOD = 'fixed'
        
        # Mock database with realistic trade data
        realistic_trades = []
        for i in range(100):
            trade = {
                'symbol': 'BTCUSDT',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'quantity': 0.01,
                'price': 45000 + (i * 10),  # Varying prices
                'pnl': (-50 + (i % 100)) if i % 3 == 0 else (20 + (i % 50)),  # Mixed P&L
                'timestamp': datetime.now() - timedelta(hours=i)
            }
            realistic_trades.append(trade)
        
        database = Mock()
        database.get_trades.return_value = realistic_trades
        database.get_portfolio.return_value = []
        
        # Create risk manager
        risk_manager = AdvancedRiskManager(config, database)
        
        # Test with multiple trades
        for i in range(10):
            trade_data = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.01 + (i * 0.001),  # Varying quantities
                'price': 45000.0 + (i * 100),
                'strategy': 'ensemble'
            }
            
            result = risk_manager.validate_trade(trade_data)
            # Most trades should be valid with reasonable parameters
            assert isinstance(result.is_valid, bool)
    
    def test_risk_manager_error_handling(self):
        """Risk manager hata yönetimi testi"""
        from risk.risk_manager_complete import AdvancedRiskManager
        
        config = Mock()
        config.INITIAL_CAPITAL = 10000.0
        config.MAX_RISK_PER_TRADE = 0.02
        
        # Mock database that raises exception
        database = Mock()
        database.get_trades.side_effect = Exception("Database error")
        
        risk_manager = AdvancedRiskManager(config, database)
        
        # Test error handling
        with pytest.raises(Exception):
            risk_manager.calculate_daily_pnl()


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
