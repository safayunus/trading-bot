"""
Advanced Database Management System
SQLAlchemy ile gelişmiş veritabanı yönetimi ve monitoring
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from sqlalchemy.pool import StaticPool
import json
import os
import numpy as np

Base = declarative_base()

class Trade(Base):
    """Trade tablosu"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY/SELL
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    pnl = Column(Float, default=0.0)
    strategy = Column(String(50), default='ensemble')
    order_id = Column(String(100))
    status = Column(String(20), default='FILLED')
    commission = Column(Float, default=0.0)
    entry_price = Column(Float)
    exit_price = Column(Float)
    duration_minutes = Column(Integer)

class Signal(Base):
    """Signal tablosu"""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal = Column(String(10), nullable=False)  # BUY/SELL/HOLD
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    strength = Column(Float, default=0.0)
    price = Column(Float)
    details = Column(Text)  # JSON string

class Portfolio(Base):
    """Portfolio tablosu"""
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    quantity = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    current_price = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
    entry_date = Column(DateTime, default=datetime.utcnow)

class Settings(Base):
    """Settings tablosu"""
    __tablename__ = 'settings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)
    description = Column(Text)

class Logs(Base):
    """Logs tablosu"""
    __tablename__ = 'logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    module = Column(String(50))
    function = Column(String(50))
    line_number = Column(Integer)

class PerformanceMetrics(Base):
    """Performance metrics tablosu"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    total_return = Column(Float, default=0.0)
    daily_return = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_volume = Column(Float, default=0.0)
    volatility = Column(Float, default=0.0)

class ModelPerformance(Base):
    """Model performance tablosu"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    accuracy = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    signals_generated = Column(Integer, default=0)
    successful_signals = Column(Integer, default=0)
    avg_signal_return = Column(Float, default=0.0)

class AdvancedDatabaseManager:
    """Gelişmiş veritabanı yönetici sınıfı"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        """
        Database manager başlatıcı
        
        Args:
            db_path: Veritabanı dosya yolu
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # SQLAlchemy engine ve session
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            poolclass=StaticPool,
            connect_args={
                'check_same_thread': False,
                'timeout': 20
            },
            echo=False
        )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Performance tracking
        self.daily_returns = []
        self.equity_curve = []
        self.initial_capital = 10000.0
        self.current_capital = 10000.0
        
    def initialize(self):
        """Veritabanını başlat ve tabloları oluştur"""
        try:
            # Dizini oluştur
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
            
            # Tabloları oluştur
            Base.metadata.create_all(bind=self.engine)
            
            # Varsayılan ayarları ekle
            self._initialize_default_settings()
            
            self.logger.info(f"Veritabanı başlatıldı: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Veritabanı başlatma hatası: {e}")
            raise
    
    def _initialize_default_settings(self):
        """Varsayılan ayarları ekle"""
        try:
            with self.SessionLocal() as session:
                default_settings = [
                    {'key': 'initial_capital', 'value': '10000.0', 'description': 'Initial trading capital'},
                    {'key': 'max_risk_per_trade', 'value': '0.02', 'description': 'Maximum risk per trade (2%)'},
                    {'key': 'max_daily_loss', 'value': '0.05', 'description': 'Maximum daily loss (5%)'},
                    {'key': 'trading_enabled', 'value': 'true', 'description': 'Trading enabled/disabled'},
                    {'key': 'last_report_date', 'value': str(datetime.now().date()), 'description': 'Last daily report date'},
                    {'key': 'telegram_reports_enabled', 'value': 'true', 'description': 'Telegram daily reports enabled'}
                ]
                
                for setting in default_settings:
                    existing = session.query(Settings).filter(Settings.key == setting['key']).first()
                    if not existing:
                        new_setting = Settings(**setting)
                        session.add(new_setting)
                
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Varsayılan ayar ekleme hatası: {e}")
    
    # Trade işlemleri
    def add_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Yeni trade kaydı ekle
        
        Args:
            trade_data: Trade verileri
            
        Returns:
            int: Eklenen kaydın ID'si
        """
        try:
            with self.SessionLocal() as session:
                trade = Trade(
                    symbol=trade_data.get('symbol'),
                    side=trade_data.get('side'),
                    quantity=trade_data.get('quantity'),
                    price=trade_data.get('price'),
                    pnl=trade_data.get('pnl', 0.0),
                    strategy=trade_data.get('strategy', 'ensemble'),
                    order_id=trade_data.get('order_id'),
                    status=trade_data.get('status', 'FILLED'),
                    commission=trade_data.get('commission', 0.0),
                    entry_price=trade_data.get('entry_price'),
                    exit_price=trade_data.get('exit_price'),
                    duration_minutes=trade_data.get('duration_minutes')
                )
                
                session.add(trade)
                session.commit()
                session.refresh(trade)
                
                # Performance güncelle
                self._update_performance_after_trade(trade_data)
                
                return trade.id
                
        except Exception as e:
            self.logger.error(f"Trade ekleme hatası: {e}")
            raise
    
    def get_trades(self, symbol: Optional[str] = None, days: int = 30, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Trade kayıtlarını al
        
        Args:
            symbol: Sembol filtresi
            days: Son X gün
            limit: Maksimum kayıt sayısı
            
        Returns:
            List: Trade kayıtları
        """
        try:
            with self.SessionLocal() as session:
                query = session.query(Trade)
                
                if symbol:
                    query = query.filter(Trade.symbol == symbol)
                
                if days:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    query = query.filter(Trade.timestamp >= cutoff_date)
                
                trades = query.order_by(Trade.timestamp.desc()).limit(limit).all()
                
                return [
                    {
                        'id': trade.id,
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'timestamp': trade.timestamp,
                        'pnl': trade.pnl,
                        'strategy': trade.strategy,
                        'order_id': trade.order_id,
                        'status': trade.status,
                        'commission': trade.commission
                    }
                    for trade in trades
                ]
                
        except Exception as e:
            self.logger.error(f"Trade alma hatası: {e}")
            return []
    
    # Signal işlemleri
    def add_signal(self, signal_data: Dict[str, Any]) -> int:
        """
        Yeni sinyal kaydı ekle
        
        Args:
            signal_data: Sinyal verileri
            
        Returns:
            int: Eklenen kaydın ID'si
        """
        try:
            with self.SessionLocal() as session:
                signal = Signal(
                    model=signal_data.get('model'),
                    symbol=signal_data.get('symbol'),
                    signal=signal_data.get('signal'),
                    confidence=signal_data.get('confidence'),
                    strength=signal_data.get('strength', 0.0),
                    price=signal_data.get('price'),
                    details=json.dumps(signal_data.get('details', {}))
                )
                
                session.add(signal)
                session.commit()
                session.refresh(signal)
                
                return signal.id
                
        except Exception as e:
            self.logger.error(f"Sinyal ekleme hatası: {e}")
            raise
    
    def get_signals(self, model: Optional[str] = None, symbol: Optional[str] = None, 
                   days: int = 7, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Sinyal kayıtlarını al
        
        Args:
            model: Model filtresi
            symbol: Sembol filtresi
            days: Son X gün
            limit: Maksimum kayıt sayısı
            
        Returns:
            List: Sinyal kayıtları
        """
        try:
            with self.SessionLocal() as session:
                query = session.query(Signal)
                
                if model:
                    query = query.filter(Signal.model == model)
                
                if symbol:
                    query = query.filter(Signal.symbol == symbol)
                
                if days:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    query = query.filter(Signal.timestamp >= cutoff_date)
                
                signals = query.order_by(Signal.timestamp.desc()).limit(limit).all()
                
                return [
                    {
                        'id': signal.id,
                        'model': signal.model,
                        'symbol': signal.symbol,
                        'signal': signal.signal,
                        'confidence': signal.confidence,
                        'timestamp': signal.timestamp,
                        'strength': signal.strength,
                        'price': signal.price,
                        'details': json.loads(signal.details) if signal.details else {}
                    }
                    for signal in signals
                ]
                
        except Exception as e:
            self.logger.error(f"Sinyal alma hatası: {e}")
            return []
    
    # Portfolio işlemleri
    def update_portfolio(self, portfolio_data: Dict[str, Any]):
        """
        Portföy güncelle
        
        Args:
            portfolio_data: Portföy verileri
        """
        try:
            with self.SessionLocal() as session:
                symbol = portfolio_data.get('symbol')
                
                portfolio = session.query(Portfolio).filter(Portfolio.symbol == symbol).first()
                
                if portfolio:
                    # Mevcut pozisyonu güncelle
                    portfolio.quantity = portfolio_data.get('quantity')
                    portfolio.avg_price = portfolio_data.get('avg_price')
                    portfolio.unrealized_pnl = portfolio_data.get('unrealized_pnl', 0.0)
                    portfolio.current_price = portfolio_data.get('current_price')
                    portfolio.last_updated = datetime.now()
                else:
                    # Yeni pozisyon ekle
                    portfolio = Portfolio(
                        symbol=symbol,
                        quantity=portfolio_data.get('quantity'),
                        avg_price=portfolio_data.get('avg_price'),
                        unrealized_pnl=portfolio_data.get('unrealized_pnl', 0.0),
                        current_price=portfolio_data.get('current_price')
                    )
                    session.add(portfolio)
                
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Portföy güncelleme hatası: {e}")
            raise
    
    def remove_from_portfolio(self, symbol: str):
        """
        Portföyden pozisyonu kaldır
        
        Args:
            symbol: Trading pair
        """
        try:
            with self.SessionLocal() as session:
                portfolio = session.query(Portfolio).filter(Portfolio.symbol == symbol).first()
                if portfolio:
                    session.delete(portfolio)
                    session.commit()
                    
        except Exception as e:
            self.logger.error(f"Portföy silme hatası: {e}")
            raise
    
    def get_portfolio(self) -> List[Dict[str, Any]]:
        """
        Aktif portföyü al
        
        Returns:
            List: Portföy pozisyonları
        """
        try:
            with self.SessionLocal() as session:
                positions = session.query(Portfolio).all()
                
                return [
                    {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'avg_price': pos.avg_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'current_price': pos.current_price,
                        'last_updated': pos.last_updated,
                        'entry_date': pos.entry_date
                    }
                    for pos in positions
                ]
                
        except Exception as e:
            self.logger.error(f"Portföy alma hatası: {e}")
            return []
    
    # Settings işlemleri
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Ayar değeri al
        
        Args:
            key: Ayar anahtarı
            default: Varsayılan değer
            
        Returns:
            Any: Ayar değeri
        """
        try:
            with self.SessionLocal() as session:
                setting = session.query(Settings).filter(Settings.key == key).first()
                
                if setting:
                    # Type conversion
                    value = setting.value
                    if value.lower() in ['true', 'false']:
                        return value.lower() == 'true'
                    try:
                        return float(value)
                    except ValueError:
                        return value
                
                return default
                
        except Exception as e:
            self.logger.error(f"Ayar alma hatası: {e}")
            return default
    
    def set_setting(self, key: str, value: Any, description: str = None):
        """
        Ayar değeri belirle
        
        Args:
            key: Ayar anahtarı
            value: Ayar değeri
            description: Açıklama
        """
        try:
            with self.SessionLocal() as session:
                setting = session.query(Settings).filter(Settings.key == key).first()
                
                if setting:
                    setting.value = str(value)
                    setting.updated_at = datetime.now()
                    if description:
                        setting.description = description
                else:
                    setting = Settings(
                        key=key,
                        value=str(value),
                        description=description
                    )
                    session.add(setting)
                
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Ayar belirleme hatası: {e}")
            raise
    
    # Logging işlemleri
    def add_log(self, level: str, message: str, module: str = None, function: str = None, line_number: int = None):
        """
        Log kaydı ekle
        
        Args:
            level: Log seviyesi
            message: Log mesajı
            module: Modül adı
            function: Fonksiyon adı
            line_number: Satır numarası
        """
        try:
            with self.SessionLocal() as session:
                log = Logs(
                    level=level,
                    message=message,
                    module=module,
                    function=function,
                    line_number=line_number
                )
                
                session.add(log)
                session.commit()
                
        except Exception as e:
            # Log ekleme hatası için print kullan (sonsuz döngü önleme)
            print(f"Log ekleme hatası: {e}")
    
    def get_logs(self, level: Optional[str] = None, days: int = 7, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Log kayıtlarını al
        
        Args:
            level: Log seviyesi filtresi
            days: Son X gün
            limit: Maksimum kayıt sayısı
            
        Returns:
            List: Log kayıtları
        """
        try:
            with self.SessionLocal() as session:
                query = session.query(Logs)
                
                if level:
                    query = query.filter(Logs.level == level)
                
                if days:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    query = query.filter(Logs.timestamp >= cutoff_date)
                
                logs = query.order_by(Logs.timestamp.desc()).limit(limit).all()
                
                return [
                    {
                        'id': log.id,
                        'level': log.level,
                        'message': log.message,
                        'timestamp': log.timestamp,
                        'module': log.module,
                        'function': log.function,
                        'line_number': log.line_number
                    }
                    for log in logs
                ]
                
        except Exception as e:
            self.logger.error(f"Log alma hatası: {e}")
            return []
    
    # Performance tracking
    def _update_performance_after_trade(self, trade_data: Dict[str, Any]):
        """Trade sonrası performans güncelle"""
        try:
            pnl = trade_data.get('pnl', 0.0)
            self.current_capital += pnl
            
            # Günlük performans güncelle
            self._update_daily_performance()
            
        except Exception as e:
            self.logger.error(f"Performans güncelleme hatası: {e}")
    
    def _update_daily_performance(self):
        """Günlük performans metriklerini güncelle"""
        try:
            today = datetime.now().date()
            
            with self.SessionLocal() as session:
                # Bugünkü trades
                today_trades = session.query(Trade).filter(
                    func.date(Trade.timestamp) == today
                ).all()
                
                if not today_trades:
                    return
                
                # Metrikleri hesapla
                total_pnl = sum(trade.pnl for trade in today_trades)
                total_trades = len(today_trades)
                winning_trades = len([t for t in today_trades if t.pnl > 0])
                losing_trades = len([t for t in today_trades if t.pnl < 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                avg_win = np.mean([t.pnl for t in today_trades if t.pnl > 0]) if winning_trades > 0 else 0
                avg_loss = abs(np.mean([t.pnl for t in today_trades if t.pnl < 0])) if losing_trades > 0 else 0
                profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
                
                # Total return hesapla
                total_return = (self.current_capital - self.initial_capital) / self.initial_capital
                daily_return = total_pnl / self.current_capital if self.current_capital > 0 else 0
                
                # Max drawdown hesapla
                max_drawdown = self._calculate_max_drawdown()
                
                # Sharpe ratio hesapla (basitleştirilmiş)
                sharpe_ratio = self._calculate_sharpe_ratio()
                
                # Performance kaydını güncelle
                performance = session.query(PerformanceMetrics).filter(
                    PerformanceMetrics.date == today
                ).first()
                
                if performance:
                    performance.total_return = total_return
                    performance.daily_return = daily_return
                    performance.win_rate = win_rate
                    performance.avg_win = avg_win
                    performance.avg_loss = avg_loss
                    performance.profit_factor = profit_factor
                    performance.total_trades = total_trades
                    performance.winning_trades = winning_trades
                    performance.losing_trades = losing_trades
                    performance.max_drawdown = max_drawdown
                    performance.sharpe_ratio = sharpe_ratio
                else:
                    performance = PerformanceMetrics(
                        date=today,
                        total_return=total_return,
                        daily_return=daily_return,
                        win_rate=win_rate,
                        avg_win=avg_win,
                        avg_loss=avg_loss,
                        profit_factor=profit_factor,
                        total_trades=total_trades,
                        winning_trades=winning_trades,
                        losing_trades=losing_trades,
                        max_drawdown=max_drawdown,
                        sharpe_ratio=sharpe_ratio
                    )
                    session.add(performance)
                
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Günlük performans güncelleme hatası: {e}")
    
    def _calculate_max_drawdown(self) -> float:
        """Maximum drawdown hesapla"""
        try:
            with self.SessionLocal() as session:
                trades = session.query(Trade).order_by(Trade.timestamp).all()
                
                if not trades:
                    return 0.0
                
                capital = self.initial_capital
                peak = capital
                max_dd = 0.0
                
                for trade in trades:
                    capital += trade.pnl
                    if capital > peak:
                        peak = capital
                    
                    drawdown = (peak - capital) / peak
                    if drawdown > max_dd:
                        max_dd = drawdown
                
                return max_dd
                
        except Exception as e:
            self.logger.error(f"Max drawdown hesaplama hatası: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, days: int = 30) -> float:
        """Sharpe ratio hesapla"""
        try:
            with self.SessionLocal() as session:
                # Son X günün performansı
                cutoff_date = datetime.now().date() - timedelta(days=days)
                
                performance_data = session.query(PerformanceMetrics).filter(
                    PerformanceMetrics.date >= cutoff_date
                ).all()
                
                if len(performance_data) < 2:
                    return 0.0
                
                daily_returns = [p.daily_return for p in performance_data]
                
                if not daily_returns:
                    return 0.0
                
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                
                if std_return == 0:
                    return 0.0
                
                # Risk-free rate varsayılan 0
                sharpe = mean_return / std_return * np.sqrt(365)  # Annualized
                
                return sharpe
                
        except Exception as e:
            self.logger.error(f"Sharpe ratio hesaplama hatası: {e}")
            return 0.0
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Performans özeti al
        
        Args:
            days: Son X gün
            
        Returns:
            Dict: Performans özeti
        """
        try:
            with self.SessionLocal() as session:
                cutoff_date = datetime.now().date() - timedelta(days=days)
                
                # Performance metrics
                performance_data = session.query(PerformanceMetrics).filter(
                    PerformanceMetrics.date >= cutoff_date
                ).order_by(PerformanceMetrics.date.desc()).all()
                
                if not performance_data:
                    return {}
                
                latest = performance_data[0]
                
                # Trades
                trades = session.query(Trade).filter(
                    Trade.timestamp >= datetime.combine(cutoff_date, datetime.min.time())
                ).all()
                
                # Portfolio
                portfolio = self.get_portfolio()
                total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in portfolio)
                
                return {
                    'period_days': days,
                    'total_return': latest.total_return,
                    'total_trades': len(trades),
                    'winning_trades': len([t for t in trades if t.pnl > 0]),
                    'losing_trades': len([t for t in trades if t.pnl < 0]),
                    'win_rate': latest.win_rate,
                    'avg_win': latest.avg_win,
                    'avg_loss': latest.avg_loss,
                    'profit_factor': latest.profit_factor,
                    'max_drawdown': latest.max_drawdown,
                    'sharpe_ratio': latest.sharpe_ratio,
                    'current_capital': self.current_capital,
                    'unrealized_pnl': total_unrealized_pnl,
                    'total_pnl': sum(trade.pnl for trade in trades),
                    'active_positions': len(portfolio),
                    'last_updated': latest.date if performance_data else None
                }
                
        except Exception as e:
            self.logger.error(f"Performans özeti hatası: {e}")
            return {}
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """
        Günlük rapor oluştur
        
        Returns:
            Dict: Günlük rapor
        """
        try:
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            with self.SessionLocal() as session:
                # Bugünkü trades
                today_trades = session.query(Trade).filter(
                    func.date(Trade.timestamp) == today
                ).all()
                
                # Bugünkü performance
                today_performance = session.query(PerformanceMetrics).filter(
                    PerformanceMetrics.date == today
                ).first()
                
                # Dünkü performance
                yesterday_performance = session.query(PerformanceMetrics).filter(
                    PerformanceMetrics.date == yesterday
                ).first()
                
                # Portfolio
                portfolio = self.get_portfolio()
                
                # Signals
                today_signals = session.query(Signal).filter(
                    func.date(Signal.timestamp) == today
                ).all()
                
                report = {
                    'date': today.isoformat(),
                    'trades': {
                        'total': len(today_trades),
