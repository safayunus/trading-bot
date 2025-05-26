"""
Advanced Order Manager
Gelişmiş emir yönetimi, pozisyon takibi ve risk kontrolü
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from .binance_client import BinanceClient, OrderResult
from ..risk.risk_manager import RiskManager
from ..utils.database import DatabaseManager
from ..utils.logger import get_trading_logger

class OrderStatus(Enum):
    """Order durumları"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderType(Enum):
    """Order tipleri"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"

@dataclass
class Position:
    """Pozisyon veri sınıfı"""
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    def update_current_price(self, price: float):
        """Güncel fiyatı güncelle ve P&L hesapla"""
        self.current_price = price
        self.last_update = datetime.now()
        
        if self.side.upper() == 'BUY':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # SELL
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e dönüştür"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat(),
            'last_update': self.last_update.isoformat()
        }

@dataclass
class OrderRequest:
    """Order isteği veri sınıfı"""
    symbol: str
    side: str
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'GTC'
    client_order_id: Optional[str] = None
    
    def validate(self) -> bool:
        """Order isteğini doğrula"""
        if not self.symbol or not self.side or not self.quantity:
            return False
        
        if self.quantity <= 0:
            return False
        
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
            if not self.price or self.price <= 0:
                return False
        
        if self.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT]:
            if not self.stop_price or self.stop_price <= 0:
                return False
        
        return True

class OrderManager:
    """Gelişmiş order yönetici sınıfı"""
    
    def __init__(self, binance_client: BinanceClient, risk_manager: RiskManager,
                 database: DatabaseManager):
        """
        Order manager başlatıcı
        
        Args:
            binance_client: Binance client instance
            risk_manager: Risk manager instance
            database: Database manager instance
        """
        self.binance_client = binance_client
        self.risk_manager = risk_manager
        self.database = database
        self.logger = logging.getLogger(__name__)
        self.trading_logger = get_trading_logger()
        
        # Pozisyon takibi
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Dict] = {}
        
        # Price tracking
        self.price_callbacks: Dict[str, List[Callable]] = {}
        self.price_streams_active = False
        
        # Order validation
        self.min_order_sizes = {}  # Symbol -> minimum order size
        self.tick_sizes = {}       # Symbol -> tick size
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        # Auto-management settings
        self.auto_stop_loss = True
        self.auto_take_profit = True
        self.trailing_stop_enabled = False
        
    async def initialize(self):
        """Order manager'ı başlat"""
        try:
            # Symbol bilgilerini yükle
            await self._load_symbol_info()
            
            # Mevcut pozisyonları yükle
            await self._load_positions()
            
            # Açık emirleri yükle
            await self._load_open_orders()
            
            # Price tracking başlat
            await self._start_price_tracking()
            
            self.logger.info("Order manager başlatıldı")
            
        except Exception as e:
            self.logger.error(f"Order manager başlatma hatası: {e}")
            raise
    
    async def close(self):
        """Order manager'ı kapat"""
        try:
            # Price streams'i durdur
            await self._stop_price_tracking()
            
            # Pozisyonları kaydet
            await self._save_positions()
            
            self.logger.info("Order manager kapatıldı")
            
        except Exception as e:
            self.logger.error(f"Order manager kapatma hatası: {e}")
    
    async def place_market_order(self, symbol: str, side: str, quantity: float,
                                validate_risk: bool = True) -> OrderResult:
        """
        Market order ver
        
        Args:
            symbol: Trading pair
            side: BUY veya SELL
            quantity: Miktar
            validate_risk: Risk doğrulaması yapılsın mı
            
        Returns:
            OrderResult: Order sonucu
        """
        try:
            # Order isteği oluştur
            order_request = OrderRequest(
                symbol=symbol,
                side=side.upper(),
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            
            # Doğrulama
            if not order_request.validate():
                return OrderResult(
                    success=False,
                    error_message="Invalid order request"
                )
            
            # Risk doğrulaması
            if validate_risk:
                risk_check = await self._validate_risk(order_request)
                if not risk_check['valid']:
                    return OrderResult(
                        success=False,
                        error_message=f"Risk validation failed: {risk_check['reason']}"
                    )
            
            # Order ver
            result = await self.binance_client.place_market_order(symbol, side, quantity)
            
            if result.success:
                # Order'ı kaydet
                await self._record_order(result)
                
                # Pozisyonu güncelle
                await self._update_position_from_order(result)
                
                # Trading log
                self.trading_logger.log_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=result.price or 0,
                    order_type='MARKET',
                    status='FILLED'
                )
                
                self.logger.info(f"Market order executed: {symbol} {side} {quantity}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Market order error: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def place_limit_order(self, symbol: str, side: str, quantity: float,
                               price: float, validate_risk: bool = True) -> OrderResult:
        """
        Limit order ver
        
        Args:
            symbol: Trading pair
            side: BUY veya SELL
            quantity: Miktar
            price: Limit fiyatı
            validate_risk: Risk doğrulaması yapılsın mı
            
        Returns:
            OrderResult: Order sonucu
        """
        try:
            # Order isteği oluştur
            order_request = OrderRequest(
                symbol=symbol,
                side=side.upper(),
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price
            )
            
            # Doğrulama
            if not order_request.validate():
                return OrderResult(
                    success=False,
                    error_message="Invalid order request"
                )
            
            # Risk doğrulaması
            if validate_risk:
                risk_check = await self._validate_risk(order_request)
                if not risk_check['valid']:
                    return OrderResult(
                        success=False,
                        error_message=f"Risk validation failed: {risk_check['reason']}"
                    )
            
            # Order ver
            result = await self.binance_client.place_limit_order(symbol, side, quantity, price)
            
            if result.success:
                # Pending order olarak kaydet
                if result.order_id:
                    self.pending_orders[result.order_id] = {
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'price': price,
                        'type': 'LIMIT',
                        'timestamp': datetime.now()
                    }
                
                # Order'ı kaydet
                await self._record_order(result)
                
                self.logger.info(f"Limit order placed: {symbol} {side} {quantity} @ {price}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Limit order error: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def place_stop_loss_order(self, symbol: str, side: str, quantity: float,
                                   stop_price: float) -> OrderResult:
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
        try:
            result = await self.binance_client.place_stop_loss_order(symbol, side, quantity, stop_price)
            
            if result.success:
                # Pending order olarak kaydet
                if result.order_id:
                    self.pending_orders[result.order_id] = {
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'stop_price': stop_price,
                        'type': 'STOP_LOSS',
                        'timestamp': datetime.now()
                    }
                
                # Order'ı kaydet
                await self._record_order(result)
                
                self.logger.info(f"Stop loss order placed: {symbol} {side} {quantity} @ {stop_price}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Stop loss order error: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Order'ı iptal et
        
        Args:
            symbol: Trading pair
            order_id: Order ID
            
        Returns:
            bool: İptal durumu
        """
        try:
            success = await self.binance_client.cancel_order(symbol, order_id)
            
            if success:
                # Pending orders'dan kaldır
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                
                # Database'de güncelle
                await self.database.update_trade(
                    int(order_id),
                    {'status': 'CANCELLED'}
                )
                
                self.logger.info(f"Order cancelled: {order_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cancel order error: {e}")
            return False
    
    async def close_position(self, symbol: str, percentage: float = 100.0) -> OrderResult:
        """
        Pozisyonu kapat
        
        Args:
            symbol: Trading pair
            percentage: Kapatılacak yüzde (100 = tamamı)
            
        Returns:
            OrderResult: Order sonucu
        """
        try:
            if symbol not in self.positions:
                return OrderResult(
                    success=False,
                    error_message=f"No position found for {symbol}"
                )
            
            position = self.positions[symbol]
            close_quantity = position.quantity * (percentage / 100.0)
            
            # Ters yönde order ver
            close_side = 'SELL' if position.side.upper() == 'BUY' else 'BUY'
            
            result = await self.place_market_order(
                symbol=symbol,
                side=close_side,
                quantity=close_quantity,
                validate_risk=False  # Pozisyon kapatma için risk kontrolü yapma
            )
            
            if result.success:
                # Pozisyonu güncelle veya kaldır
                if percentage >= 100.0:
                    # Pozisyonu tamamen kapat
                    realized_pnl = position.unrealized_pnl
                    self.total_pnl += realized_pnl
                    
                    if realized_pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    
                    self.total_trades += 1
                    
                    # Risk manager'dan kaldır
                    self.risk_manager.remove_position(symbol, position.current_price)
                    
                    # Pozisyonu kaldır
                    del self.positions[symbol]
                    
                    # Database'den sil
                    await self.database.delete_position(symbol)
                    
                    self.logger.info(f"Position closed: {symbol}, P&L: {realized_pnl:.2f}")
                else:
                    # Kısmi kapatma
                    position.quantity -= close_quantity
                    await self._save_position(position)
                
                # Trading log
                self.trading_logger.log_trade(
                    symbol=symbol,
                    side=close_side,
                    quantity=close_quantity,
                    price=result.price or 0,
                    order_type='MARKET',
                    status='POSITION_CLOSE'
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Close position error: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def close_all_positions(self) -> Dict[str, OrderResult]:
        """
        Tüm pozisyonları kapat
        
        Returns:
            Dict: Symbol -> OrderResult mapping
        """
        results = {}
        
        for symbol in list(self.positions.keys()):
            result = await self.close_position(symbol)
            results[symbol] = result
            
            # Kısa bekleme (rate limiting için)
            await asyncio.sleep(0.1)
        
        return results
    
    async def set_stop_loss(self, symbol: str, stop_price: float) -> bool:
        """
        Pozisyon için stop loss ayarla
        
        Args:
            symbol: Trading pair
            stop_price: Stop loss fiyatı
            
        Returns:
            bool: Başarı durumu
        """
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            position.stop_loss = stop_price
            
            # Database'de güncelle
            await self._save_position(position)
            
            # Risk manager'ı güncelle
            self.risk_manager.add_position(
                symbol=symbol,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                stop_loss=stop_price,
                take_profit=position.take_profit or 0
            )
            
            self.logger.info(f"Stop loss set for {symbol}: {stop_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Set stop loss error: {e}")
            return False
    
    async def set_take_profit(self, symbol: str, take_profit_price: float) -> bool:
        """
        Pozisyon için take profit ayarla
        
        Args:
            symbol: Trading pair
            take_profit_price: Take profit fiyatı
            
        Returns:
            bool: Başarı durumu
        """
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            position.take_profit = take_profit_price
            
            # Database'de güncelle
            await self._save_position(position)
            
            # Risk manager'ı güncelle
            self.risk_manager.add_position(
                symbol=symbol,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                stop_loss=position.stop_loss or 0,
                take_profit=take_profit_price
            )
            
            self.logger.info(f"Take profit set for {symbol}: {take_profit_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Set take profit error: {e}")
            return False
    
    async def update_positions(self):
        """Pozisyonları güncelle (fiyat ve P&L)"""
        try:
            if not self.positions:
                return
            
            # Tüm sembollerin fiyatlarını al
            symbols = list(self.positions.keys())
            
            for symbol in symbols:
                current_price = await self.binance_client.get_symbol_price(symbol)
                
                if current_price and symbol in self.positions:
                    position = self.positions[symbol]
                    position.update_current_price(current_price)
                    
                    # Stop loss / take profit kontrolü
                    if self.auto_stop_loss or self.auto_take_profit:
                        await self._check_stop_loss_take_profit(symbol, current_price)
                    
                    # Database'de güncelle
                    await self._save_position(position)
            
        except Exception as e:
            self.logger.error(f"Update positions error: {e}")
    
    async def get_positions(self) -> Dict[str, Position]:
        """Aktif pozisyonları al"""
        return self.positions.copy()
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Belirli bir pozisyonu al"""
        return self.positions.get(symbol)
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Portfolio özetini al"""
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_positions = len(self.positions)
            
            # Account bilgisi
            account_info = await self.binance_client.get_account_info()
            portfolio_value = await self.binance_client.calculate_portfolio_value()
            
            return {
                'total_positions': total_positions,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': self.total_pnl,
                'portfolio_value': portfolio_value,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
                'account_info': account_info,
                'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio summary error: {e}")
            return {}
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Açık emirleri al"""
        try:
            return await self.binance_client.get_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Get open orders error: {e}")
            return []
    
    async def get_order_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Order geçmişini al"""
        try:
            return await self.binance_client.get_order_history(symbol, limit)
        except Exception as e:
            self.logger.error(f"Get order history error: {e}")
            return []
    
    async def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Trade geçmişini al"""
        try:
            return await self.binance_client.get_trade_history(symbol, limit)
        except Exception as e:
            self.logger.error(f"Get trade history error: {e}")
            return []
    
    # Private methods
    async def _validate_risk(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Risk doğrulaması yap"""
        try:
            # Hesap bilgisi al
            account_info = await self.binance_client.get_account_info()
            if not account_info:
                return {'valid': False, 'reason': 'Cannot get account info'}
            
            # Güncel fiyat al
            current_price = await self.binance_client.get_symbol_price(order_request.symbol)
            if not current_price:
                return {'valid': False, 'reason': 'Cannot get current price'}
            
            # Order değerini hesapla
            order_price = order_request.price or current_price
            order_value = order_request.quantity * order_price
            
            # Risk manager ile doğrula
            validation = self.risk_manager.validate_trade(
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                price=order_price,
                account_balance=account_info['balances'].get('USDT', {}).get('free', 0)
            )
            
            return {
                'valid': validation['is_valid'],
                'reason': '; '.join(validation['errors']) if validation['errors'] else None,
                'warnings': validation['warnings'],
                'risk_level': validation['risk_level'].value
            }
            
        except Exception as e:
            self.logger.error(f"Risk validation error: {e}")
            return {'valid': False, 'reason': str(e)}
    
    async def _record_order(self, order_result: OrderResult):
        """Order'ı database'e kaydet"""
        try:
            trade_data = {
                'symbol': order_result.symbol,
                'side': order_result.side,
                'quantity': order_result.quantity,
                'price': order_result.price or 0,
                'order_type': 'MARKET',  # Simplified
                'status': order_result.status or 'FILLED',
                'order_id': order_result.order_id,
                'filled_quantity': order_result.quantity,
                'avg_fill_price': order_result.price or 0
            }
            
            await self.database.insert_trade(trade_data)
            
        except Exception as e:
            self.logger.error(f"Record order error: {e}")
    
    async def _update_position_from_order(self, order_result: OrderResult):
        """Order'dan pozisyonu güncelle"""
        try:
            symbol = order_result.symbol
            side = order_result.side
            quantity = order_result.quantity
            price = order_result.price or 0
            
            if symbol in self.positions:
                # Mevcut pozisyon var
                position = self.positions[symbol]
                
                if position.side.upper() == side.upper():
                    # Aynı yönde - pozisyonu artır
                    total_quantity = position.quantity + quantity
                    weighted_price = ((position.entry_price * position.quantity) + 
                                    (price * quantity)) / total_quantity
                    
                    position.quantity = total_quantity
                    position.entry_price = weighted_price
                else:
                    # Ters yönde - pozisyonu azalt veya kapat
                    if quantity >= position.quantity:
                        # Pozisyonu kapat
                        realized_pnl = position.unrealized_pnl
                        self.total_pnl += realized_pnl
                        
                        if realized_pnl > 0:
                            self.winning_trades += 1
                        else:
                            self.losing_trades += 1
                        
                        self.total_trades += 1
                        
                        del self.positions[symbol]
                        await self.database.delete_position(symbol)
                        
                        # Kalan miktar varsa yeni pozisyon oluştur
                        remaining_quantity = quantity - position.quantity
                        if remaining_quantity > 0:
                            new_position = Position(
                                symbol=symbol,
                                side=side,
                                quantity=remaining_quantity,
                                entry_price=price,
                                current_price=price
                            )
                            self.positions[symbol] = new_position
                            await self._save_position(new_position)
                    else:
                        # Pozisyonu azalt
                        position.quantity -= quantity
                        await self._save_position(position)
            else:
                # Yeni pozisyon oluştur
                new_position = Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price
                )
                self.positions[symbol] = new_position
                await self._save_position(new_position)
                
                # Risk manager'a ekle
                self.risk_manager.add_position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=price,
                    stop_loss=0,
                    take_profit=0
                )
            
        except Exception as e:
            self.logger.error(f"Update position error: {e}")
    
    async def _check_stop_loss_take_profit(self, symbol: str, current_price: float):
        """Stop loss ve take profit kontrolü"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            # Stop loss kontrolü
            if position.stop_loss:
                should_trigger = False
                
                if position.side.upper() == 'BUY' and current_price <= position.stop_loss:
                    should_trigger = True
                elif position.side.upper() == 'SELL' and current_price >= position.stop_loss:
                    should_trigger = True
                
                if should_trigger:
                    self.logger.warning(f"Stop loss triggered for {symbol} at {current_price}")
                    await self.close_position(symbol)
                    return
            
            # Take profit kontrolü
            if position.take_profit:
                should_trigger = False
                
                if position.side.upper() == 'BUY' and current_price >= position.take_profit:
                    should_trigger = True
                elif position.side.upper() == 'SELL' and current_price <= position.take_profit:
                    should_trigger = True
                
                if should_trigger:
                    self.logger.info(f"Take profit triggered for {symbol} at {current_price}")
                    await self.close_position(symbol)
                    return
            
        except Exception as e:
            self.logger.error(f"Stop loss/take profit check error: {e}")
    
    async def _load_symbol_info(self):
        """Symbol bilgilerini yükle"""
        try:
            # Temel semboller için bilgi al
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            
            for symbol in symbols:
                symbol_info = await self.binance_client.get_symbol_info(symbol)
                if symbol_info:
                    # Minimum order size ve tick size bilgilerini çıkar
                    for filter_info in symbol_info.get('filters', []):
                        if filter_info['filterType'] == 'LOT_SIZE':
                            self.min_order_sizes[symbol] = float(filter_info['minQty'])
                        elif filter_info['filterType'] == 'PRICE_FILTER':
                            self.tick_sizes[symbol] = float(filter_info['tickSize'])
            
        except Exception as e:
            self.logger.error(f"Load symbol info error: {e}")
    
    async def _load_positions(self):
        """Database'den pozisyonları yükle"""
        try:
            positions_data = await self.database.get_positions()
            
            for pos_data in positions_data:
                position = Position(
                    symbol=pos_data['symbol'],
                    side=pos_data['side'],
                    quantity=pos_data['quantity'],
                    entry_price=pos_data['entry_price'],
                    current_price=pos_data['current_price'],
                    unrealized_pnl=pos_data['unrealized_pnl'],
                    stop_loss=pos_data['stop_loss'],
                    take_profit=pos_data['take_profit']
                )
                
                self.positions[pos_data['symbol']] = position
            
            self.logger.info(f"Loaded {len(self.positions)} positions from database")
            
        except Exception as e:
            self.logger.error(f"Load positions error: {e}")
    
    async def _load_open_orders(self):
        """Açık emirleri yükle"""
        try:
            open_orders = await self.binance_client.get_open_orders()
            
            for order in open_orders:
                order_id = str(order['orderId'])
                self.pending_orders[order_id] = {
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'quantity': float(order['origQty']),
                    'price': float(order['price']) if order['price'] != '0.00000000' else None,
                    'type': order['type'],
                    'timestamp': datetime.fromtimestamp(order['time'] / 1000)
                }
            
            self.logger.info(f"Loaded {len(self.pending_orders)} open orders")
            
        except Exception as e:
            self.logger.error(f"Load open orders error: {e}")
    
    async def _start_price_tracking(self):
        """Price tracking başlat"""
        try:
            if not self.positions:
                return
            
            self.price_streams_active = True
            
            # Her pozisyon için price stream başlat
            for symbol in self.positions.keys():
                await self.binance_client.start_price_stream(
                    symbol, 
                    self._price_update_callback
                )
            
            self.logger.info("Price tracking started")
            
        except Exception as e:
            self.logger.error(f"Start price tracking error: {e}")
    
    async def _stop_price_tracking(self):
        """Price tracking durdur"""
        try:
            self.price_streams_active = False
            
            # Tüm streams'i durdur
            for symbol in self.positions.keys():
                await self.binance_client.stop_stream(symbol, 'ticker')
            
            self.logger.info("Price tracking stopped")
            
        except Exception as e:
            self.logger.error(f"Stop price tracking error: {e}")
    
    async def _price_update_callback(self, data: Dict[str, Any]):
        """Price update callback"""
        try:
            if not self.price_streams_active:
                return
            
            symbol = data.get('s')  # Symbol
            price = float(data.get('c', 0))  # Current price
            
            if symbol in self.positions and price > 0:
                position = self.positions[symbol]
                position.update_current_price(price)
                
                # Stop loss / take profit kontrolü
                if self.auto_stop_loss or self.auto_take_profit:
                    await self._check_stop_loss_take_profit(symbol, price)
            
        except Exception as e:
            self.logger.error(f"Price update callback error: {e}")
    
    async def _save_position(self, position: Position):
        """Pozisyonu database'e kaydet"""
        try:
            position_data = {
                'symbol': position.symbol,
                'side': position.side,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'entry_time': position.entry_time,
                'last_update': position.last_update
            }
            
            await self.database.upsert_position(position_data)
            
        except Exception as e:
            self.logger.error(f"Save position error: {e}")
    
    async def _save_positions(self):
        """Tüm pozisyonları kaydet"""
        try:
            for position in self.positions.values():
                await self._save_position(position)
            
            self.logger.info(f"Saved {len(self.positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Save positions error: {e}")
    
    # Performance tracking methods
    def get_performance_stats(self) -> Dict[str, Any]:
        """Performans istatistiklerini al"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'total_pnl': self.total_pnl,
            'avg_win': self.total_pnl / self.winning_trades if self.winning_trades > 0 else 0,
            'avg_loss': abs(self.total_pnl) / self.losing_trades if self.losing_trades > 0 else 0,
            'profit_factor': abs(self.total_pnl / self.losing_trades) / (self.total_pnl / self.winning_trades) if self.winning_trades > 0 and self.losing_trades > 0 else 0
        }
    
    def reset_performance_stats(self):
        """Performans istatistiklerini sıfırla"""
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        self.logger.info("Performance stats reset")
    
    # Configuration methods
    def set_auto_stop_loss(self, enabled: bool):
        """Auto stop loss ayarını değiştir"""
        self.auto_stop_loss = enabled
        self.logger.info(f"Auto stop loss: {'enabled' if enabled else 'disabled'}")
    
    def set_auto_take_profit(self, enabled: bool):
        """Auto take profit ayarını değiştir"""
        self.auto_take_profit = enabled
        self.logger.info(f"Auto take profit: {'enabled' if enabled else 'disabled'}")
    
    def set_trailing_stop(self, enabled: bool):
        """Trailing stop ayarını değiştir"""
        self.trailing_stop_enabled = enabled
        self.logger.info(f"Trailing stop: {'enabled' if enabled else 'disabled'}")
    
    # Utility methods
    async def validate_symbol(self, symbol: str) -> bool:
        """Symbol'ün geçerli olup olmadığını kontrol et"""
        try:
            symbol_info = await self.binance_client.get_symbol_info(symbol)
            return symbol_info is not None
        except Exception:
            return False
    
    def get_min_order_size(self, symbol: str) -> float:
        """Symbol için minimum order size'ı al"""
        return self.min_order_sizes.get(symbol, 0.001)
    
    def get_tick_size(self, symbol: str) -> float:
        """Symbol için tick size'ı al"""
        return self.tick_sizes.get(symbol, 0.01)
    
    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Quantity'yi symbol'e uygun şekilde yuvarla"""
        min_qty = self.get_min_order_size(symbol)
        
        # Minimum quantity'nin decimal places'ini bul
        decimal_places = len(str(min_qty).split('.')[-1]) if '.' in str(min_qty) else 0
        
        return round(quantity, decimal_places)
    
    def round_price(self, symbol: str, price: float) -> float:
        """Price'ı symbol'e uygun şekilde yuvarla"""
        tick_size = self.get_tick_size(symbol)
        
        # Tick size'ın decimal places'ini bul
        decimal_places = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
        
        return round(price, decimal_places)
    
    # Emergency methods
    async def emergency_stop(self) -> Dict[str, Any]:
        """Acil durum - tüm pozisyonları kapat ve emirleri iptal et"""
        try:
            results = {
                'closed_positions': {},
                'cancelled_orders': {},
                'errors': []
            }
            
            # Tüm açık emirleri iptal et
            for order_id, order_info in list(self.pending_orders.items()):
                try:
                    success = await self.cancel_order(order_info['symbol'], order_id)
                    results['cancelled_orders'][order_id] = success
                except Exception as e:
                    results['errors'].append(f"Cancel order {order_id}: {str(e)}")
            
            # Tüm pozisyonları kapat
            close_results = await self.close_all_positions()
            results['closed_positions'] = {
                symbol: result.success for symbol, result in close_results.items()
            }
            
            # Performans istatistiklerini kaydet
            stats = self.get_performance_stats()
            
            self.logger.critical(f"Emergency stop executed: {results}")
            
            return {
                'success': True,
                'results': results,
                'final_stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # Health check methods
    async def health_check(self) -> Dict[str, Any]:
        """Order manager sağlık kontrolü"""
        try:
            # Binance bağlantısını test et
            connection_ok = await self.binance_client.test_connection()
            
            # Pozisyon sayısı
            position_count = len(self.positions)
            
            # Pending order sayısı
            pending_order_count = len(self.pending_orders)
            
            # Price tracking durumu
            price_tracking_active = self.price_streams_active
            
            # Son güncelleme zamanları
            last_position_update = None
            if self.positions:
                last_position_update = max(pos.last_update for pos in self.positions.values())
            
            return {
                'status': 'healthy' if connection_ok else 'unhealthy',
                'binance_connection': connection_ok,
                'position_count': position_count,
                'pending_order_count': pending_order_count,
                'price_tracking_active': price_tracking_active,
                'last_position_update': last_position_update.isoformat() if last_position_update else None,
                'performance_stats': self.get_performance_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
