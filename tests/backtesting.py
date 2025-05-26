"""
Backtesting Framework
Geçmiş verilerle strateji test etme sistemi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from tests import generate_mock_price_data, PerformanceTimer


class OrderType(Enum):
    """Order türleri"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order yönleri"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class BacktestOrder:
    """Backtest order sınıfı"""
    id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float] = None
    timestamp: datetime = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    status: str = "NEW"  # NEW, FILLED, CANCELLED, REJECTED


@dataclass
class BacktestPosition:
    """Backtest pozisyon sınıfı"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class BacktestTrade:
    """Backtest trade sınıfı"""
    id: int
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    entry_time: datetime
    exit_time: datetime
    strategy: str
    commission: float = 0.0


class BacktestEngine:
    """Backtesting motoru"""
    
    def __init__(self, initial_capital: float = 10000.0, commission_rate: float = 0.001):
        """
        Backtest engine başlatıcı
        
        Args:
            initial_capital: Başlangıç sermayesi
            commission_rate: Komisyon oranı (0.1% = 0.001)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        
        # Trading state
        self.positions: Dict[str, BacktestPosition] = {}
        self.orders: List[BacktestOrder] = []
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []
        
        # Counters
        self.order_id_counter = 1
        self.trade_id_counter = 1
        
        # Current market data
        self.current_prices: Dict[str, float] = {}
        self.current_timestamp: datetime = None
        
        self.logger = logging.getLogger(__name__)
    
    def add_market_data(self, symbol: str, price_data: pd.DataFrame):
        """
        Market verisi ekle
        
        Args:
            symbol: Trading pair
            price_data: OHLCV verisi
        """
        if not hasattr(self, 'market_data'):
            self.market_data = {}
        
        self.market_data[symbol] = price_data.copy()
        self.logger.info(f"Market data added for {symbol}: {len(price_data)} bars")
    
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                   quantity: float, price: Optional[float] = None, 
                   stop_price: Optional[float] = None) -> int:
        """
        Order ver
        
        Args:
            symbol: Trading pair
            side: BUY/SELL
            order_type: Order türü
            quantity: Miktar
            price: Limit fiyatı
            stop_price: Stop fiyatı
            
        Returns:
            Order ID
        """
        order = BacktestOrder(
            id=self.order_id_counter,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            timestamp=self.current_timestamp
        )
        
        self.orders.append(order)
        self.order_id_counter += 1
        
        # Market order ise hemen doldur
        if order_type == OrderType.MARKET:
            self._fill_market_order(order)
        
        return order.id
    
    def _fill_market_order(self, order: BacktestOrder):
        """Market order'ı doldur"""
        current_price = self.current_prices.get(order.symbol)
        if current_price is None:
            order.status = "REJECTED"
            self.logger.warning(f"No price data for {order.symbol}")
            return
        
        # Slippage simülasyonu (0.01%)
        slippage = 0.0001
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + slippage)
        else:
            fill_price = current_price * (1 - slippage)
        
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.status = "FILLED"
        
        # Pozisyonu güncelle
        self._update_position(order)
        
        self.logger.debug(f"Order filled: {order.symbol} {order.side.value} {order.quantity} @ {fill_price}")
    
    def _update_position(self, order: BacktestOrder):
        """Pozisyonu güncelle"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = BacktestPosition(
                symbol=symbol,
                quantity=0.0,
                avg_price=0.0,
                current_price=order.filled_price,
                unrealized_pnl=0.0
            )
        
        position = self.positions[symbol]
        commission = order.filled_quantity * order.filled_price * self.commission_rate
        
        if order.side == OrderSide.BUY:
            # Long pozisyon aç/artır
            if position.quantity >= 0:
                # Mevcut long pozisyonu artır
                total_cost = (position.quantity * position.avg_price) + (order.filled_quantity * order.filled_price)
                position.quantity += order.filled_quantity
                position.avg_price = total_cost / position.quantity
            else:
                # Short pozisyonu kapat/azalt
                if abs(position.quantity) >= order.filled_quantity:
                    # Kısmi/tam kapama
                    realized_pnl = order.filled_quantity * (position.avg_price - order.filled_price)
                    position.realized_pnl += realized_pnl
                    position.quantity += order.filled_quantity
                    
                    # Trade kaydı oluştur
                    self._create_trade_record(order, realized_pnl - commission)
                else:
                    # Pozisyon tersine döner
                    close_quantity = abs(position.quantity)
                    realized_pnl = close_quantity * (position.avg_price - order.filled_price)
                    position.realized_pnl += realized_pnl
                    
                    # Trade kaydı oluştur
                    self._create_trade_record(order, realized_pnl - commission)
                    
                    # Yeni long pozisyon
                    remaining_quantity = order.filled_quantity - close_quantity
                    position.quantity = remaining_quantity
                    position.avg_price = order.filled_price
        
        else:  # SELL
            # Short pozisyon aç/artır veya long pozisyonu kapat
            if position.quantity <= 0:
                # Mevcut short pozisyonu artır
                total_cost = (abs(position.quantity) * position.avg_price) + (order.filled_quantity * order.filled_price)
                position.quantity -= order.filled_quantity
                position.avg_price = total_cost / abs(position.quantity)
            else:
                # Long pozisyonu kapat/azalt
                if position.quantity >= order.filled_quantity:
                    # Kısmi/tam kapama
                    realized_pnl = order.filled_quantity * (order.filled_price - position.avg_price)
                    position.realized_pnl += realized_pnl
                    position.quantity -= order.filled_quantity
                    
                    # Trade kaydı oluştur
                    self._create_trade_record(order, realized_pnl - commission)
                else:
                    # Pozisyon tersine döner
                    close_quantity = position.quantity
                    realized_pnl = close_quantity * (order.filled_price - position.avg_price)
                    position.realized_pnl += realized_pnl
                    
                    # Trade kaydı oluştur
                    self._create_trade_record(order, realized_pnl - commission)
                    
                    # Yeni short pozisyon
                    remaining_quantity = order.filled_quantity - close_quantity
                    position.quantity = -remaining_quantity
                    position.avg_price = order.filled_price
        
        # Komisyonu sermayeden düş
        self.current_capital -= commission
    
    def _create_trade_record(self, order: BacktestOrder, pnl: float):
        """Trade kaydı oluştur"""
        trade = BacktestTrade(
            id=self.trade_id_counter,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            entry_price=self.positions[order.symbol].avg_price,
            exit_price=order.filled_price,
            pnl=pnl,
            entry_time=self.current_timestamp,  # Simplified
            exit_time=self.current_timestamp,
            strategy="backtest",
            commission=order.filled_quantity * order.filled_price * self.commission_rate
        )
        
        self.trades.append(trade)
        self.trade_id_counter += 1
    
    def update_market_prices(self, timestamp: datetime, prices: Dict[str, float]):
        """Market fiyatlarını güncelle"""
        self.current_timestamp = timestamp
        self.current_prices.update(prices)
        
        # Pozisyonların unrealized P&L'ini güncelle
        for symbol, position in self.positions.items():
            if symbol in prices and position.quantity != 0:
                position.current_price = prices[symbol]
                if position.quantity > 0:
                    # Long pozisyon
                    position.unrealized_pnl = position.quantity * (position.current_price - position.avg_price)
                else:
                    # Short pozisyon
                    position.unrealized_pnl = abs(position.quantity) * (position.avg_price - position.current_price)
        
        # Equity curve güncelle
        total_equity = self.current_capital + sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)
    
    def get_portfolio_value(self) -> float:
        """Toplam portföy değeri"""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        return self.current_capital + unrealized_pnl + realized_pnl
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Performans metriklerini hesapla"""
        if len(self.equity_curve) < 2:
            return {}
        
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Temel metrikler
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Drawdown hesaplama
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade istatistikleri
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Risk metrikleri
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': self.equity_curve[-1],
            'total_pnl': sum(t.pnl for t in self.trades)
        }


class StrategyBacktester:
    """Strateji backtesting sınıfı"""
    
    def __init__(self, strategy_func, initial_capital: float = 10000.0):
        """
        Strategy backtester başlatıcı
        
        Args:
            strategy_func: Test edilecek strateji fonksiyonu
            initial_capital: Başlangıç sermayesi
        """
        self.strategy_func = strategy_func
        self.engine = BacktestEngine(initial_capital)
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(self, symbol: str, start_date: datetime, end_date: datetime, 
                    timeframe: str = '1h') -> Dict[str, Any]:
        """
        Backtest çalıştır
        
        Args:
            symbol: Trading pair
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            timeframe: Zaman dilimi
            
        Returns:
            Backtest sonuçları
        """
        self.logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        with PerformanceTimer() as timer:
            # Mock data oluştur (gerçek uygulamada historical data API'den gelir)
            days = (end_date - start_date).days
            price_data = generate_mock_price_data(symbol, days)
            df = pd.DataFrame(price_data)
            
            # Market data ekle
            self.engine.add_market_data(symbol, df)
            
            # Backtest çalıştır
            for i, row in df.iterrows():
                # Market fiyatlarını güncelle
                prices = {symbol: row['close']}
                self.engine.update_market_prices(row['timestamp'], prices)
                
                # Strateji sinyallerini al
                # Strateji fonksiyonuna geçmiş veri ver
                historical_data = df.iloc[:i+1] if i > 0 else df.iloc[:1]
                
                try:
                    signals = self.strategy_func(historical_data, symbol)
                    
                    # Sinyalleri işle
                    if signals:
                        for signal in signals:
                            if signal['action'] == 'BUY':
                                self.engine.place_order(
                                    symbol=symbol,
                                    side=OrderSide.BUY,
                                    order_type=OrderType.MARKET,
                                    quantity=signal.get('quantity', 0.01)
                                )
                            elif signal['action'] == 'SELL':
                                self.engine.place_order(
                                    symbol=symbol,
                                    side=OrderSide.SELL,
                                    order_type=OrderType.MARKET,
                                    quantity=signal.get('quantity', 0.01)
                                )
                
                except Exception as e:
                    self.logger.error(f"Strategy error at {row['timestamp']}: {e}")
                    continue
        
        # Sonuçları hesapla
        performance = self.engine.get_performance_metrics()
        performance['backtest_duration'] = timer.elapsed
        performance['data_points'] = len(df)
        
        self.logger.info(f"Backtest completed in {timer.elapsed:.2f} seconds")
        self.logger.info(f"Total return: {performance.get('total_return', 0):.2%}")
        self.logger.info(f"Total trades: {performance.get('total_trades', 0)}")
        
        return {
            'performance': performance,
            'trades': self.engine.trades,
            'equity_curve': self.engine.equity_curve,
            'timestamps': self.engine.timestamps,
            'positions': self.engine.positions
        }


def simple_ma_strategy(data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
    """
    Basit moving average stratejisi
    
    Args:
        data: Fiyat verisi
        symbol: Trading pair
        
    Returns:
        Sinyal listesi
    """
    if len(data) < 20:
        return []
    
    # Moving averages hesapla
    data = data.copy()
    data['ma_short'] = data['close'].rolling(window=5).mean()
    data['ma_long'] = data['close'].rolling(window=20).mean()
    
    signals = []
    
    # Son iki bar'ı kontrol et
    if len(data) >= 2:
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Golden cross (short MA yukarı keserse long MA'yı)
        if (previous['ma_short'] <= previous['ma_long'] and 
            current['ma_short'] > current['ma_long']):
            signals.append({
                'action': 'BUY',
                'quantity': 0.01,
                'reason': 'Golden cross'
            })
        
        # Death cross (short MA aşağı keserse long MA'yı)
        elif (previous['ma_short'] >= previous['ma_long'] and 
              current['ma_short'] < current['ma_long']):
            signals.append({
                'action': 'SELL',
                'quantity': 0.01,
                'reason': 'Death cross'
            })
    
    return signals


def rsi_strategy(data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
    """
    RSI tabanlı strateji
    
    Args:
        data: Fiyat verisi
        symbol: Trading pair
        
    Returns:
        Sinyal listesi
    """
    if len(data) < 15:
        return []
    
    # RSI hesapla
    data = data.copy()
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    signals = []
    current_rsi = data['rsi'].iloc[-1]
    
    # RSI sinyalleri
    if current_rsi < 30:  # Oversold
        signals.append({
            'action': 'BUY',
            'quantity': 0.01,
            'reason': f'RSI oversold: {current_rsi:.2f}'
        })
    elif current_rsi > 70:  # Overbought
        signals.append({
            'action': 'SELL',
            'quantity': 0.01,
            'reason': f'RSI overbought: {current_rsi:.2f}'
        })
    
    return signals


def run_strategy_comparison(strategies: Dict[str, callable], symbol: str = 'BTCUSDT', 
                          days: int = 30) -> Dict[str, Any]:
    """
    Birden fazla stratejiyi karşılaştır
    
    Args:
        strategies: Strateji fonksiyonları dict'i
        symbol: Trading pair
        days: Test günü sayısı
        
    Returns:
        Karşılaştırma sonuçları
    """
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()
    
    results = {}
    
    for strategy_name, strategy_func in strategies.items():
        print(f"Testing strategy: {strategy_name}")
        
        backtester = StrategyBacktester(strategy_func)
        result = backtester.run_backtest(symbol, start_date, end_date)
        
        results[strategy_name] = result
    
    # Karşılaştırma tablosu oluştur
    comparison = {}
    for strategy_name, result in results.items():
        perf = result['performance']
        comparison[strategy_name] = {
            'Total Return': f"{perf.get('total_return', 0):.2%}",
            'Total Trades': perf.get('total_trades', 0),
            'Win Rate': f"{perf.get('win_rate', 0):.2%}",
            'Profit Factor': f"{perf.get('profit_factor', 0):.2f}",
            'Max Drawdown': f"{perf.get('max_drawdown', 0):.2%}",
            'Sharpe Ratio': f"{perf.get('sharpe_ratio', 0):.2f}",
            'Final Capital': f"${perf.get('final_capital', 0):.2f}"
        }
    
    return {
        'detailed_results': results,
        'comparison': comparison
    }


# Örnek kullanım
if __name__ == "__main__":
    # Stratejileri tanımla
    strategies = {
        'Simple MA': simple_ma_strategy,
        'RSI': rsi_strategy
    }
    
    # Karşılaştırma çalıştır
    results = run_strategy_comparison(strategies, days=30)
    
    print("\n=== Strategy Comparison ===")
    import pandas as pd
    comparison_df = pd.DataFrame(results['comparison']).T
    print(comparison_df)
