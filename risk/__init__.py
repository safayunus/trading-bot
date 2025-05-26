"""
Risk Management Package
Gelişmiş risk yönetimi ve portföy koruması sistemi
"""

from .risk_manager_complete import (
    AdvancedRiskManager,
    RiskLevel,
    PositionSizingMethod,
    MarketRegime,
    RiskMetrics,
    PositionRisk,
    RiskAlert,
    TradeRiskAssessment
)

__all__ = [
    'AdvancedRiskManager',
    'RiskLevel',
    'PositionSizingMethod', 
    'MarketRegime',
    'RiskMetrics',
    'PositionRisk',
    'RiskAlert',
    'TradeRiskAssessment'
]

# Risk management versiyonu
RISK_MANAGER_VERSION = "1.0.0"

# Varsayılan risk parametreleri
DEFAULT_RISK_CONFIG = {
    'max_risk_per_trade': 0.02,      # %2 maksimum risk per trade
    'max_daily_loss': 0.05,          # %5 maksimum günlük kayıp
    'max_drawdown': 0.15,            # %15 maksimum drawdown
    'max_portfolio_risk': 0.10,      # %10 maksimum portföy riski
    'max_correlation_exposure': 0.30, # %30 maksimum korelasyon riski
    'circuit_breaker_threshold': 0.10, # %10 acil durum eşiği
    'flash_crash_threshold': 0.05,   # %5 hızlı düşüş eşiği
    'position_sizing_method': 'Volatility Based',
    'trailing_stop_enabled': True,
    'default_stop_loss_atr': 2.0,
    'default_take_profit_atr': 4.0
}

# Risk seviyeleri ve açıklamaları
RISK_LEVEL_DESCRIPTIONS = {
    RiskLevel.VERY_LOW: "Minimal risk - Safe to proceed",
    RiskLevel.LOW: "Low risk - Proceed with normal caution",
    RiskLevel.MEDIUM: "Medium risk - Increased caution advised",
    RiskLevel.HIGH: "High risk - Reduce position size",
    RiskLevel.VERY_HIGH: "Very high risk - Consider avoiding",
    RiskLevel.EXTREME: "Extreme risk - Do not trade"
}

# Position sizing açıklamaları
POSITION_SIZING_DESCRIPTIONS = {
    PositionSizingMethod.FIXED_AMOUNT: "Fixed dollar amount per trade",
    PositionSizingMethod.FIXED_PERCENTAGE: "Fixed percentage of capital",
    PositionSizingMethod.KELLY_CRITERION: "Optimal sizing based on win rate and avg win/loss",
    PositionSizingMethod.VOLATILITY_BASED: "Size adjusted for market volatility",
    PositionSizingMethod.ATR_BASED: "Size based on Average True Range"
}

def create_risk_manager(initial_capital: float = 10000.0, 
                       config: dict = None) -> AdvancedRiskManager:
    """
    Risk manager oluştur
    
    Args:
        initial_capital: Başlangıç sermayesi
        config: Risk konfigürasyonu
        
    Returns:
        AdvancedRiskManager: Risk manager instance
    """
    risk_manager = AdvancedRiskManager(initial_capital)
    
    if config:
        # Konfigürasyonu uygula
        for key, value in config.items():
            if hasattr(risk_manager, key):
                setattr(risk_manager, key, value)
    
    return risk_manager

def get_risk_info():
    """Risk management bilgilerini al"""
    return {
        'version': RISK_MANAGER_VERSION,
        'default_config': DEFAULT_RISK_CONFIG,
        'risk_levels': RISK_LEVEL_DESCRIPTIONS,
        'position_sizing_methods': POSITION_SIZING_DESCRIPTIONS,
        'features': [
            'Position Sizing (Kelly, Volatility, ATR)',
            'Stop Loss/Take Profit Management',
            'Portfolio Risk Monitoring',
            'Daily Loss Limits',
            'Drawdown Protection',
            'Emergency Circuit Breaker',
            'Correlation Risk Analysis',
            'Market Regime Detection',
            'Real-time Risk Alerts',
            'Performance Tracking'
        ]
    }
