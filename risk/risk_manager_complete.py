"""
Advanced Risk Management System - Complete Version
Gelişmiş risk yönetimi, pozisyon boyutlandırma ve portföy koruması
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math

class RiskLevel(Enum):
    """Risk seviyeleri"""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"
    EXTREME = "Extreme"

class PositionSizingMethod(Enum):
    """Pozisyon boyutlandırma yöntemleri"""
    FIXED_AMOUNT = "Fixed Amount"
    FIXED_PERCENTAGE = "Fixed Percentage"
    KELLY_CRITERION = "Kelly Criterion"
    VOLATILITY_BASED = "Volatility Based"
    ATR_BASED = "ATR Based"

class MarketRegime(Enum):
    """Market rejimleri"""
    LOW_VOLATILITY = "Low Volatility"
    NORMAL = "Normal"
    HIGH_VOLATILITY = "High Volatility"
    TRENDING = "Trending"
    RANGING = "Ranging"
    CRISIS = "Crisis"

@dataclass
class RiskMetrics:
    """Risk metrikleri"""
    daily_pnl: float
    daily_pnl_percent: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    var_95: float  # Value at Risk %95
    portfolio_beta: float
    correlation_risk: float
    timestamp: datetime

@dataclass
class PositionRisk:
    """Pozisyon riski"""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    risk_amount: float
    risk_percent: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    atr_risk: float
    volatility: float
    correlation_exposure: float
    time_in_position: timedelta
    timestamp: datetime

@dataclass
class RiskAlert:
    """Risk uyarısı"""
    alert_type: str
    severity: RiskLevel
    message: str
    symbol: Optional[str]
    current_value: float
    threshold: float
    action_required: bool
    timestamp: datetime

@dataclass
class TradeRiskAssessment:
    """Trade risk değerlendirmesi"""
    is_valid: bool
    risk_level: RiskLevel
    position_size: float
    stop_loss_price: float
    take_profit_price: float
    risk_amount: float
    risk_percent: float
    reward_risk_ratio: float
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]

class AdvancedRiskManager:
    """Gelişmiş risk yönetimi sistemi"""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Risk manager başlatıcı
        
        Args:
            initial_capital: Başlangıç sermayesi
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # Risk parametreleri
        self.max_risk_per_trade = 0.02  # %2 maksimum risk per trade
        self.max_daily_loss = 0.05      # %5 maksimum günlük kayıp
        self.max_drawdown = 0.15        # %15 maksimum drawdown
        self.max_portfolio_risk = 0.10  # %10 maksimum portföy riski
        self.max_correlation_exposure = 0.30  # %30 maksimum korelasyon riski
        
        # Position sizing
        self.default_sizing_method = PositionSizingMethod.VOLATILITY_BASED
        self.kelly_lookback_days = 30
        self.volatility_lookback_days = 20
        self.atr_multiplier = 2.0
        
        # Stop loss/Take profit
        self.default_stop_loss_atr = 2.0
        self.default_take_profit_atr = 4.0
        self.trailing_stop_enabled = True
        self.trailing_stop_atr = 1.5
        self.breakeven_threshold = 1.5  # ATR
        
        # Portfolio tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.daily_pnl_history: List[float] = []
        self.equity_curve: List[float] = [initial_capital]
        self.peak_equity = initial_capital
        
        # Market regime
        self.current_market_regime = MarketRegime.NORMAL
        self.volatility_threshold_low = 0.01   # %1
        self.volatility_threshold_high = 0.05  # %5
        self.trend_strength_threshold = 0.7
        
        # Emergency conditions
        self.emergency_stop_triggered = False
        self.circuit_breaker_threshold = 0.10  # %10 günlük kayıp
        self.flash_crash_threshold = 0.05      # %5 hızlı düşüş
        
        # Alerts
        self.active_alerts: List[RiskAlert] = []
        self.alert_cooldown = 300  # 5 dakika
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
    async def validate_trade(self, symbol: str, side: str, quantity: float, 
                           price: float, account_balance: float) -> TradeRiskAssessment:
        """
        Trade risk değerlendirmesi
        
        Args:
            symbol: Trading pair
            side: BUY/SELL
            quantity: Miktar
            price: Fiyat
            account_balance: Hesap bakiyesi
            
        Returns:
            TradeRiskAssessment: Risk değerlendirmesi
        """
        try:
            errors = []
            warnings = []
            recommendations = []
            
            # Emergency stop kontrolü
            if self.emergency_stop_triggered:
                return TradeRiskAssessment(
                    is_valid=False,
                    risk_level=RiskLevel.EXTREME,
                    position_size=0,
                    stop_loss_price=0,
                    take_profit_price=0,
                    risk_amount=0,
                    risk_percent=0,
                    reward_risk_ratio=0,
                    errors=["Emergency stop is active - no new trades allowed"],
                    warnings=[],
                    recommendations=["Wait for emergency conditions to clear"]
                )
            
            # Güncel sermaye güncelle
            self.current_capital = account_balance
            
            # Risk kontrolleri
            daily_loss_check = await self._check_daily_loss_limit()
            if not daily_loss_check['valid']:
                errors.append(daily_loss_check['message'])
            
            drawdown_check = await self._check_drawdown_limit()
            if not drawdown_check['valid']:
                errors.append(drawdown_check['message'])
            
            portfolio_risk_check = await self._check_portfolio_risk(symbol, quantity, price)
            if not portfolio_risk_check['valid']:
                errors.append(portfolio_risk_check['message'])
            
            correlation_check = await self._check_correlation_risk(symbol)
            if not correlation_check['valid']:
                warnings.append(correlation_check['message'])
            
            # Pozisyon boyutu hesapla
            position_size = await self._calculate_position_size(symbol, price, account_balance)
            
            # Stop loss/take profit hesapla
            stop_loss_price, take_profit_price = await self._calculate_stop_take_profit(symbol, side, price)
            
            # Risk miktarı hesapla
            if side.upper() == 'BUY':
                risk_amount = (price - stop_loss_price) * position_size
            else:
                risk_amount = (stop_loss_price - price) * position_size
            
            risk_percent = risk_amount / account_balance
            
            # Reward/Risk oranı
            if side.upper() == 'BUY':
                reward_amount = (take_profit_price - price) * position_size
            else:
                reward_amount = (price - take_profit_price) * position_size
            
            reward_risk_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Risk seviyesi belirle
            risk_level = self._determine_risk_level(risk_percent, reward_risk_ratio)
            
            # Risk kontrolü
            if risk_percent > self.max_risk_per_trade:
                errors.append(f"Risk too high: {risk_percent:.2%} > {self.max_risk_per_trade:.2%}")
            
            if reward_risk_ratio < 1.0:
                warnings.append(f"Poor risk/reward ratio: {reward_risk_ratio:.2f}")
                recommendations.append("Consider better entry point or adjust targets")
            
            # Final validation
            is_valid = len(errors) == 0
            
            return TradeRiskAssessment(
                is_valid=is_valid,
                risk_level=risk_level,
                position_size=position_size,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                risk_amount=risk_amount,
                risk_percent=risk_percent,
                reward_risk_ratio=reward_risk_ratio,
                errors=errors,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Trade validation error: {e}")
            return TradeRiskAssessment(
                is_valid=False,
                risk_level=RiskLevel.EXTREME,
                position_size=0,
                stop_loss_price=0,
                take_profit_price=0,
                risk_amount=0,
                risk_percent=0,
                reward_risk_ratio=0,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                recommendations=["Contact support"]
            )
    
    async def monitor_risk(self) -> List[RiskAlert]:
        """Risk monitoring ve alert üretimi"""
        try:
            alerts = []
            
            # Günlük kayıp kontrolü
            if self.daily_pnl_history:
                daily_pnl = self.daily_pnl_history[-1]
                daily_loss_percent = abs(daily_pnl) / self.current_capital
                
                if daily_pnl < 0 and daily_loss_percent > self.max_daily_loss * 0.8:
                    alerts.append(RiskAlert(
                        alert_type="Daily Loss Warning",
                        severity=RiskLevel.HIGH,
                        message=f"Daily loss approaching limit: {daily_loss_percent:.2%}",
                        symbol=None,
                        current_value=daily_loss_percent,
                        threshold=self.max_daily_loss,
                        action_required=daily_loss_percent > self.max_daily_loss,
                        timestamp=datetime.now()
                    ))
            
            # Drawdown kontrolü
            current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
            if current_drawdown > self.max_drawdown * 0.8:
                alerts.append(RiskAlert(
                    alert_type="Drawdown Warning",
                    severity=RiskLevel.HIGH if current_drawdown > self.max_drawdown else RiskLevel.MEDIUM,
                    message=f"Drawdown approaching limit: {current_drawdown:.2%}",
                    symbol=None,
                    current_value=current_drawdown,
                    threshold=self.max_drawdown,
                    action_required=current_drawdown > self.max_drawdown,
                    timestamp=datetime.now()
                ))
            
            # Circuit breaker kontrolü
            if self.daily_pnl_history:
                daily_pnl = self.daily_pnl_history[-1]
                if daily_pnl < 0 and abs(daily_pnl) / self.current_capital > self.circuit_breaker_threshold:
                    await self._trigger_emergency_stop("Circuit breaker triggered")
                    alerts.append(RiskAlert(
                        alert_type="Emergency Stop",
                        severity=RiskLevel.EXTREME,
                        message="Circuit breaker activated - all trading stopped",
                        symbol=None,
                        current_value=abs(daily_pnl) / self.current_capital,
                        threshold=self.circuit_breaker_threshold,
                        action_required=True,
                        timestamp=datetime.now()
                    ))
            
            self.active_alerts = alerts
            return alerts
            
        except Exception as e:
            self.logger.error(f"Risk monitoring error: {e}")
            return []
    
    async def _trigger_emergency_stop(self, reason: str):
        """Acil durum durdurma"""
        try:
            self.emergency_stop_triggered = True
            self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")
    
    async def reset_emergency_stop(self):
        """Acil durum durumunu sıfırla"""
        try:
            self.emergency_stop_triggered = False
            self.logger.info("Emergency stop reset")
            
        except Exception as e:
            self.logger.error(f"Emergency stop reset error: {e}")
    
    async def update_risk_config(self, config: Dict[str, Any]):
        """Risk konfigürasyonunu güncelle"""
        try:
            if 'max_risk_per_trade' in config:
                self.max_risk_per_trade = config['max_risk_per_trade']
            
            if 'max_daily_loss' in config:
                self.max_daily_loss = config['max_daily_loss']
            
            if 'max_drawdown' in config:
                self.max_drawdown = config['max_drawdown']
            
            if 'position_sizing_method' in config:
                method_name = config['position_sizing_method']
                for method in PositionSizingMethod:
                    if method.value == method_name:
                        self.default_sizing_method = method
                        break
            
            if 'trailing_stop_enabled' in config:
                self.trailing_stop_enabled = config['trailing_stop_enabled']
            
            self.logger.info(f"Risk configuration updated: {config}")
            
        except Exception as e:
            self.logger.error(f"Risk config update error: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Risk özeti"""
        try:
            current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
            
            # Performance metrikleri
            total_trades = self.win_count + self.loss_count
            win_rate = self.win_count / total_trades if total_trades > 0 else 0
            profit_factor = abs(self.total_profit / self.total_loss) if self.total_loss != 0 else 0
            
            # Günlük P&L
            daily_pnl = self.daily_pnl_history[-1] if self.daily_pnl_history else 0
            daily_pnl_percent = daily_pnl / self.current_capital if self.current_capital > 0 else 0
            
            return {
                'capital': {
                    'initial': self.initial_capital,
                    'current': self.current_capital,
                    'peak': self.peak_equity,
                    'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
                },
                'risk_metrics': {
                    'current_drawdown': current_drawdown,
                    'max_drawdown_limit': self.max_drawdown,
                    'daily_pnl': daily_pnl,
                    'daily_pnl_percent': daily_pnl_percent,
                    'daily_loss_limit': self.max_daily_loss
                },
                'performance': {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'winning_trades': self.win_count,
                    'losing_trades': self.loss_count
                },
                'positions': {
                    'count': len(self.positions),
                    'total_risk': sum(pos.risk_percent for pos in self.positions.values()),
                    'portfolio_risk_limit': self.max_portfolio_risk
                },
                'alerts': {
                    'active_count': len(self.active_alerts),
                    'emergency_stop': self.emergency_stop_triggered,
                    'market_regime': self.current_market_regime.value
                },
                'configuration': {
                    'max_risk_per_trade': self.max_risk_per_trade,
                    'position_sizing_method': self.default_sizing_method.value,
                    'trailing_stop_enabled': self.trailing_stop_enabled
                }
            }
            
        except Exception as e:
            self.logger.error(f"Risk summary error: {e}")
            return {}
    
    # Helper methods (simplified implementations)
    async def _calculate_position_size(self, symbol: str, price: float, account_balance: float) -> float:
        """Pozisyon boyutu hesapla"""
        risk_amount = account_balance * self.max_risk_per_trade
        return risk_amount / price
    
    async def _calculate_stop_take_profit(self, symbol: str, side: str, price: float) -> Tuple[float, float]:
        """Stop loss ve take profit hesapla"""
        atr = price * 0.02  # Varsayılan %2
        
        if side.upper() == 'BUY':
            stop_loss = price - (atr * self.default_stop_loss_atr)
            take_profit = price + (atr * self.default_take_profit_atr)
        else:
            stop_loss = price + (atr * self.default_stop_loss_atr)
            take_profit = price - (atr * self.default_take_profit_atr)
        
        return stop_loss, take_profit
    
    async def _check_daily_loss_limit(self) -> Dict[str, Any]:
        """Günlük kayıp limiti kontrolü"""
        if not self.daily_pnl_history:
            return {'valid': True, 'message': ''}
        
        daily_pnl = self.daily_pnl_history[-1]
        daily_loss_percent = abs(daily_pnl) / self.current_capital
        
        if daily_pnl < 0 and daily_loss_percent > self.max_daily_loss:
            return {
                'valid': False,
                'message': f"Daily loss limit exceeded: {daily_loss_percent:.2%} > {self.max_daily_loss:.2%}"
            }
        
        return {'valid': True, 'message': ''}
    
    async def _check_drawdown_limit(self) -> Dict[str, Any]:
        """Drawdown limiti kontrolü"""
        current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
        
        if current_drawdown > self.max_drawdown:
            return {
                'valid': False,
                'message': f"Drawdown limit exceeded: {current_drawdown:.2%} > {self.max_drawdown:.2%}"
            }
        
        return {'valid': True, 'message': ''}
    
    async def _check_portfolio_risk(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        """Portföy risk kontrolü"""
        current_portfolio_risk = sum(pos.risk_percent for pos in self.positions.values())
        new_trade_value = quantity * price
        new_trade_risk = new_trade_value / self.current_capital
        total_risk = current_portfolio_risk + new_trade_risk
        
        if total_risk > self.max_portfolio_risk:
            return {
                'valid': False,
                'message': f"Portfolio risk limit exceeded: {total_risk:.2%} > {self.max_portfolio_risk:.2%}"
            }
        
        return {'valid': True, 'message': ''}
    
    async def _check_correlation_risk(self, symbol: str) -> Dict[str, Any]:
        """Korelasyon riski kontrolü"""
        return {'valid': True, 'message': ''}  # Basitleştirilmiş
    
    def _determine_risk_level(self, risk_percent: float, reward_risk_ratio: float) -> RiskLevel:
        """Risk seviyesi belirle"""
        if risk_percent > 0.05:
            return RiskLevel.EXTREME
        elif risk_percent > 0.03:
            return RiskLevel.VERY_HIGH
        elif risk_percent > 0.02:
            return RiskLevel.HIGH
        elif risk_percent > 0.01:
            return RiskLevel.MEDIUM
        elif risk_percent > 0.005:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    # Position management methods
    async def add_position(self, symbol: str, side: str, quantity: float, 
                          entry_price: float, stop_loss: float = 0, take_profit: float = 0):
        """Pozisyon ekle"""
        try:
            if side.upper() == 'BUY':
                risk_amount = (entry_price - stop_loss) * quantity if stop_loss > 0 else 0
            else:
                risk_amount = (stop_loss - entry_price) * quantity if stop_loss > 0 else 0
            
            risk_percent = risk_amount / self.current_capital if self.current_capital > 0 else 0
            
            position_risk = PositionRisk(
                symbol=symbol,
                position_size=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                unrealized_pnl=0.0,
                risk_amount=risk_amount,
                risk_percent=risk_percent,
                stop_loss=stop_loss if stop_loss > 0 else None,
                take_profit=take_profit if take_profit > 0 else None,
                atr_risk=entry_price * 0.02,
                volatility=0.03,
                correlation_exposure=0.0,
                time_in_position=timedelta(0),
                timestamp=datetime.now()
            )
            
            self.positions[symbol] = position_risk
            self.logger.info(f"Position added: {symbol} {side} {quantity} @ {entry_price}")
            
        except Exception as e:
            self.logger.error(f"Add position error: {e}")
    
    async def remove_position(self, symbol: str, exit_price: float):
        """Pozisyon kaldır"""
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                pnl = (exit_price - position.entry_price) * position.position_size
                
                # Performance tracking
                if pnl > 0:
                    self.win_count += 1
                    self.total_profit += pnl
                else:
                    self.loss_count += 1
                    self.total_loss += abs(pnl)
                
                # P&L güncelle
                if self.daily_pnl_history:
                    self.daily_pnl_history[-1] += pnl
                else:
                    self.daily_pnl_history.append(pnl)
                
                self.current_capital += pnl
                self.equity_curve.append(self.current_capital)
                
                if self.current_capital > self.peak_equity:
                    self.peak_equity = self.current_capital
                
                del self.positions[symbol]
                self.logger.info(f"Position removed: {symbol} @ {exit_price}, P&L: {pnl:.2f}")
                
        except Exception as e:
            self.logger.error(f"Remove position error: {e}")
    
    async def update_position_price(self, symbol: str, current_price: float):
        """Pozisyon fiyatını güncelle"""
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.position_size
                
        except Exception as e:
            self.logger.error(f"Update position price error: {e}")
