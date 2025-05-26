"""
Advanced Risk Management System
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

class RiskManager:
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
        
        # Configuration
        self.risk_config = {
            'position_sizing': {
                'method': self.default_sizing_method.value,
                'max_risk_per_trade': self.max_risk_per_trade,
                'kelly_lookback': self.kelly_lookback_days,
                'volatility_lookback': self.volatility_lookback_days
            },
            'stop_loss': {
                'atr_multiplier': self.default_stop_loss_atr,
                'trailing_enabled': self.trailing_stop_enabled,
                'trailing_atr': self.trailing_stop_atr,
                'breakeven_threshold': self.breakeven_threshold
            },
            'portfolio': {
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_correlation_exposure': self.max_correlation_exposure
            },
            'emergency': {
                'circuit_breaker': self.circuit_breaker_threshold,
                'flash_crash': self.flash_crash_threshold
            }
        }
        
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
            
            # Günlük kayıp kontrolü
            daily_loss_check = await self._check_daily_loss_limit()
            if not daily_loss_check['valid']:
                errors.append(daily_loss_check['message'])
            
            # Drawdown kontrolü
            drawdown_check = await self._check_drawdown_limit()
            if not drawdown_check['valid']:
                errors.append(drawdown_check['message'])
            
            # Portföy risk kontrolü
            portfolio_risk_check = await self._check_portfolio_risk(symbol, quantity, price)
            if not portfolio_risk_check['valid']:
                errors.append(portfolio_risk_check['message'])
            
            # Korelasyon riski kontrolü
            correlation_check = await self._check_correlation_risk(symbol)
            if not correlation_check['valid']:
                warnings.append(correlation_check['message'])
            
            # Market rejimi kontrolü
            market_regime_check = await self._check_market_regime()
            if market_regime_check['risk_level'] == RiskLevel.HIGH:
                warnings.append(f"High risk market regime: {self.current_market_regime.value}")
                recommendations.append("Consider reducing position size")
            
            # Pozisyon boyutu hesapla
            position_size = await self._calculate_position_size(
                symbol, price, account_balance
            )
            
            # Stop loss/take profit hesapla
            stop_loss_price, take_profit_price = await self._calculate_stop_take_profit(
                symbol, side, price
            )
            
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
            
            # Pozisyon boyutu ayarla
            if quantity > position_size:
                warnings.append(f"Requested quantity ({quantity}) exceeds recommended size ({position_size:.6f})")
                recommendations.append(f"Consider reducing to {position_size:.6f}")
            
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
    
    async def _calculate_position_size(self, symbol: str, price: float, 
                                     account_balance: float) -> float:
        """Pozisyon boyutu hesapla"""
        try:
            if self.default_sizing_method == PositionSizingMethod.FIXED_PERCENTAGE:
                return await self._fixed_percentage_sizing(price, account_balance)
            
            elif self.default_sizing_method == PositionSizingMethod.KELLY_CRITERION:
                return await self._kelly_criterion_sizing(symbol, price, account_balance)
            
            elif self.default_sizing_method == PositionSizingMethod.VOLATILITY_BASED:
                return await self._volatility_based_sizing(symbol, price, account_balance)
            
            elif self.default_sizing_method == PositionSizingMethod.ATR_BASED:
                return await self._atr_based_sizing(symbol, price, account_balance)
            
            else:
                return await self._fixed_percentage_sizing(price, account_balance)
                
        except Exception as e:
            self.logger.error(f"Position sizing error: {e}")
            return 0.0
    
    async def _fixed_percentage_sizing(self, price: float, account_balance: float) -> float:
        """Sabit yüzde pozisyon boyutlandırma"""
        risk_amount = account_balance * self.max_risk_per_trade
        return risk_amount / price
    
    async def _kelly_criterion_sizing(self, symbol: str, price: float, 
                                    account_balance: float) -> float:
        """Kelly Criterion pozisyon boyutlandırma"""
        try:
            # Basit Kelly implementasyonu
            # Gerçek uygulamada geçmiş trade verilerinden hesaplanmalı
            
            # Varsayılan değerler (örnek)
            win_rate = 0.55  # %55 kazanma oranı
            avg_win = 0.03   # %3 ortalama kazanç
            avg_loss = 0.02  # %2 ortalama kayıp
            
            # Kelly formülü: f = (bp - q) / b
            # b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # %25 ile sınırla
            
            position_value = account_balance * kelly_fraction
            return position_value / price
            
        except Exception as e:
            self.logger.error(f"Kelly criterion sizing error: {e}")
            return await self._fixed_percentage_sizing(price, account_balance)
    
    async def _volatility_based_sizing(self, symbol: str, price: float, 
                                     account_balance: float) -> float:
        """Volatilite bazlı pozisyon boyutlandırma"""
        try:
            # Basit volatilite hesaplaması
            # Gerçek uygulamada market data'dan hesaplanmalı
            
            # Varsayılan volatilite (günlük)
            daily_volatility = 0.03  # %3
            
            # Hedef volatilite
            target_volatility = 0.02  # %2
            
            # Volatilite ayarlaması
            volatility_adjustment = target_volatility / daily_volatility
            volatility_adjustment = max(0.1, min(volatility_adjustment, 2.0))
            
            base_position_value = account_balance * self.max_risk_per_trade
            adjusted_position_value = base_position_value * volatility_adjustment
            
            return adjusted_position_value / price
            
        except Exception as e:
            self.logger.error(f"Volatility-based sizing error: {e}")
            return await self._fixed_percentage_sizing(price, account_balance)
    
    async def _atr_based_sizing(self, symbol: str, price: float, 
                              account_balance: float) -> float:
        """ATR bazlı pozisyon boyutlandırma"""
        try:
            # Basit ATR hesaplaması
            # Gerçek uygulamada market data'dan hesaplanmalı
            
            # Varsayılan ATR (fiyatın %2'si)
            atr = price * 0.02
            
            # Risk miktarı
            risk_amount = account_balance * self.max_risk_per_trade
            
            # Stop loss distance (ATR multiplier)
            stop_distance = atr * self.default_stop_loss_atr
            
            # Position size
            position_size = risk_amount / stop_distance
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"ATR-based sizing error: {e}")
            return await self._fixed_percentage_sizing(price, account_balance)
    
    async def _calculate_stop_take_profit(self, symbol: str, side: str, 
                                        price: float) -> Tuple[float, float]:
        """Stop loss ve take profit hesapla"""
        try:
            # Basit ATR hesaplaması
            atr = price * 0.02  # Varsayılan %2
            
            if side.upper() == 'BUY':
                stop_loss = price - (atr * self.default_stop_loss_atr)
                take_profit = price + (atr * self.default_take_profit_atr)
            else:  # SELL
                stop_loss = price + (atr * self.default_stop_loss_atr)
                take_profit = price - (atr * self.default_take_profit_atr)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Stop/TP calculation error: {e}")
            return price * 0.98, price * 1.04  # Varsayılan %2 stop, %4 profit
    
    async def _check_daily_loss_limit(self) -> Dict[str, Any]:
        """Günlük kayıp limiti kontrolü"""
        try:
            if not self.daily_pnl_history:
                return {'valid': True, 'message': ''}
            
            today_pnl = self.daily_pnl_history[-1] if self.daily_pnl_history else 0
            daily_loss_percent = abs(today_pnl) / self.current_capital
            
            if today_pnl < 0 and daily_loss_percent > self.max_daily_loss:
                return {
                    'valid': False,
                    'message': f"Daily loss limit exceeded: {daily_loss_percent:.2%} > {self.max_daily_loss:.2%}"
                }
            
            return {'valid': True, 'message': ''}
            
        except Exception as e:
            self.logger.error(f"Daily loss check error: {e}")
            return {'valid': False, 'message': 'Error checking daily loss limit'}
    
    async def _check_drawdown_limit(self) -> Dict[str, Any]:
        """Drawdown limiti kontrolü"""
        try:
            current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
            
            if current_drawdown > self.max_drawdown:
                return {
                    'valid': False,
                    'message': f"Drawdown limit exceeded: {current_drawdown:.2%} > {self.max_drawdown:.2%}"
                }
            
            return {'valid': True, 'message': ''}
            
        except Exception as e:
            self.logger.error(f"Drawdown check error: {e}")
            return {'valid': False, 'message': 'Error checking drawdown limit'}
    
    async def _check_portfolio_risk(self, symbol: str, quantity: float, 
                                  price: float) -> Dict[str, Any]:
        """Portföy risk kontrolü"""
        try:
            # Mevcut portföy riski
            current_portfolio_risk = sum(
                pos.risk_percent for pos in self.positions.values()
            )
            
            # Yeni trade riski
            new_trade_value = quantity * price
            new_trade_risk = new_trade_value / self.current_capital
            
            total_risk = current_portfolio_risk + new_trade_risk
            
            if total_risk > self.max_portfolio_risk:
                return {
                    'valid': False,
                    'message': f"Portfolio risk limit exceeded: {total_risk:.2%} > {self.max_portfolio_risk:.2%}"
                }
            
            return {'valid': True, 'message': ''}
            
        except Exception as e:
            self.logger.error(f"Portfolio risk check error: {e}")
            return {'valid': False, 'message': 'Error checking portfolio risk'}
    
    async def _check_correlation_risk(self, symbol: str) -> Dict[str, Any]:
        """Korelasyon riski kontrolü"""
        try:
            # Basit korelasyon kontrolü
            # Gerçek uygulamada asset korelasyonları hesaplanmalı
            
            # Aynı base asset kontrolü
            base_asset = symbol.replace('USDT', '').replace('BTC', '')
            
            correlated_exposure = 0.0
            for pos_symbol in self.positions.keys():
                pos_base = pos_symbol.replace('USDT', '').replace('BTC', '')
                if pos_base == base_asset:
                    correlated_exposure += self.positions[pos_symbol].risk_percent
            
            if correlated_exposure > self.max_correlation_exposure:
                return {
                    'valid': False,
                    'message': f"Correlation risk too high for {base_asset}: {correlated_exposure:.2%}"
                }
            
            return {'valid': True, 'message': ''}
            
        except Exception as e:
            self.logger.error(f"Correlation risk check error: {e}")
            return {'valid': True, 'message': ''}
    
    async def _check_market_regime(self) -> Dict[str, Any]:
        """Market rejimi kontrolü"""
        try:
            # Basit market rejimi tespiti
            # Gerçek uygulamada volatilite ve trend analizi yapılmalı
            
            risk_level = RiskLevel.LOW
            
            if self.current_market_regime == MarketRegime.CRISIS:
                risk_level = RiskLevel.EXTREME
            elif self.current_market_regime == MarketRegime.HIGH_VOLATILITY:
                risk_level = RiskLevel.HIGH
            elif self.current_market_regime == MarketRegime.LOW_VOLATILITY:
                risk_level = RiskLevel.LOW
            
            return {
                'valid': True,
                'risk_level': risk_level,
                'regime': self.current_market_regime.value
            }
            
        except Exception as e:
            self.logger.error(f"Market regime check error: {e}")
            return {'valid': True, 'risk_level': RiskLevel.MEDIUM, 'regime': 'Unknown'}
    
    def _determine_risk_level(self, risk_percent: float, reward_risk_ratio: float) -> RiskLevel:
        """Risk seviyesi belirle"""
        try:
            # Risk yüzdesine göre
            if risk_percent > 0.05:  # %5'ten fazla
                return RiskLevel.EXTREME
            elif risk_percent > 0.03:  # %3'ten fazla
                return RiskLevel.VERY_HIGH
            elif risk_percent > 0.02:  # %2'den fazla
                return RiskLevel.HIGH
            elif risk_percent > 0.01:  # %1'den fazla
                return RiskLevel.MEDIUM
            elif risk_percent > 0.005:  # %0.5'ten fazla
                return RiskLevel.LOW
            else:
                return RiskLevel.VERY_LOW
                
        except Exception as e:
            self.logger.error(f"Risk level determination error: {e}")
            return RiskLevel.HIGH
    
    async def add_position(self, symbol: str, side: str, quantity: float, 
                          entry_price: float, stop_loss: float = 0, 
                          take_profit: float = 0):
        """Pozisyon ekle"""
        try:
            # Risk hesapla
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
                atr_risk=entry_price * 0.02,  # Varsayılan
                volatility=0.03,  # Varsayılan
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
                
                # P&L hesapla
                pnl = (exit_price - position.entry_price) * position.position_size
                
                # Günlük P&L güncelle
                if self.daily_pnl_history:
                    self.daily_pnl_history[-1] += pnl
                else:
                    self.daily_pnl_history.append(pnl)
                
                # Sermaye güncelle
                self.current_capital += pnl
                self.equity_curve.append(self.current_capital)
                
                # Peak equity güncelle
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
                
                # Unrealized P&L hesapla
                position.unrealized_pnl = (current_price - position.entry_price) * position.position_size
                
                # Trailing stop kontrolü
                if self.trailing_stop_enabled and position.stop_loss:
                    await self._update_trailing_stop(symbol, current_price)
                
                # Breakeven kontrolü
                await self._check_breakeven_move(symbol, current_price)
                
        except Exception as e:
            self.logger.error(f"Update position price error: {e}")
    
    async def _update_trailing_stop(self, symbol: str, current_price: float):
        """Trailing stop güncelle"""
        try:
            position = self.positions[symbol]
            if not position.stop_loss:
                return
            
            atr = position.atr_risk
            trailing_distance = atr * self.trailing_stop_atr
            
            # BUY pozisyonu için
            if position.position_size > 0:
                new_stop = current_price - trailing_distance
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    self.logger.info(f"Trailing stop updated for {symbol}: {new_stop:.6f}")
            
            # SELL pozisyonu için
            else:
                new_stop = current_price + trailing_distance
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    self.logger.info(f"Trailing stop updated for {symbol}: {new_stop:.6f}")
                    
        except Exception as e:
            self.logger.error(f"Trailing stop update error: {e}")
    
    async def _check_breakeven_move(self, symbol: str, current_price: float):
        """Breakeven hareketi kontrolü"""
        try:
            position = self.positions[symbol]
            if not position.stop_loss:
                return
            
            atr = position.atr_risk
            breakeven_distance = atr * self.breakeven_threshold
            
            # BUY pozisyonu için
            if position.position_size > 0:
                if current_price >= position.entry_price + breakeven_distance:
                    if position.stop_loss < position.entry_price:
                        position.stop_loss = position.entry_price
                        self.logger.info(f"Breakeven move for {symbol}: Stop moved to entry")
            
            # SELL pozisyonu için
            else:
                if current_price <= position.entry_price - breakeven_distance:
                    if position.stop_loss > position.entry_price:
                        position.stop_loss = position.entry_price
                        self.logger.info(f"Breakeven move for {symbol}: Stop moved to entry")
                        
        except Exception as e:
            self.logger.error(f"Breakeven check error: {e}")
    
    async def monitor_risk(self) -> List[RiskAlert]:
        """Risk monitoring"""
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
                        action_required=daily
