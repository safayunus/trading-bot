"""
Technical Analysis Model
Gelişmiş teknik analiz indikatörleri ve sinyal üretimi
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import aiohttp
from dataclasses import dataclass
from enum import Enum

class SignalStrength(Enum):
    """Sinyal gücü seviyeleri"""
    VERY_WEAK = 0.2
    WEAK = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0

@dataclass
class TechnicalSignal:
    """Teknik analiz sinyali"""
    indicator: str
    signal: str  # BUY, SELL, HOLD
    strength: float  # 0-1 arası
    value: float
    description: str
    timestamp: datetime

@dataclass
class TechnicalAnalysisResult:
    """Teknik analiz sonucu"""
    symbol: str
    current_price: float
    signals: List[TechnicalSignal]
    combined_signal: str  # BUY, SELL, HOLD
    combined_strength: float
    confidence: float
    support_levels: List[float]
    resistance_levels: List[float]
    trend_direction: str  # BULLISH, BEARISH, SIDEWAYS
    volatility: float
    volume_analysis: Dict[str, Any]
    timestamp: datetime

class TechnicalAnalysisModel:
    """Gelişmiş teknik analiz modeli"""
    
    def __init__(self, symbol: str = 'BTCUSDT'):
        """
        Technical Analysis model başlatıcı
        
        Args:
            symbol: Trading pair
        """
        self.symbol = symbol
        self.logger = logging.getLogger(__name__)
        
        # Indikatör parametreleri
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        self.bb_period = 20
        self.bb_std = 2
        
        self.sma_short = 20
        self.sma_medium = 50
        self.sma_long = 200
        
        # Volume analizi
        self.volume_sma_period = 20
        self.volume_spike_threshold = 2.0
        
        # Support/Resistance
        self.sr_lookback = 50
        self.sr_min_touches = 2
        
        # Trend analizi
        self.trend_period = 20
        
        # Signal weights (toplam 1.0 olmalı)
        self.signal_weights = {
            'rsi': 0.15,
            'macd': 0.20,
            'bollinger_bands': 0.15,
            'moving_averages': 0.25,
            'volume': 0.10,
            'support_resistance': 0.15
        }
        
    async def analyze(self, data: pd.DataFrame) -> Optional[TechnicalAnalysisResult]:
        """
        Teknik analiz yap
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            TechnicalAnalysisResult: Analiz sonucu
        """
        try:
            if len(data) < max(self.sma_long, self.sr_lookback):
                self.logger.warning(f"Insufficient data for technical analysis. Need at least {max(self.sma_long, self.sr_lookback)} records.")
                return None
            
            # Teknik indikatörleri hesapla
            data_with_indicators = await self._calculate_indicators(data)
            
            # Sinyalleri üret
            signals = []
            
            # RSI analizi
            rsi_signal = await self._analyze_rsi(data_with_indicators)
            if rsi_signal:
                signals.append(rsi_signal)
            
            # MACD analizi
            macd_signal = await self._analyze_macd(data_with_indicators)
            if macd_signal:
                signals.append(macd_signal)
            
            # Bollinger Bands analizi
            bb_signal = await self._analyze_bollinger_bands(data_with_indicators)
            if bb_signal:
                signals.append(bb_signal)
            
            # Moving Averages analizi
            ma_signal = await self._analyze_moving_averages(data_with_indicators)
            if ma_signal:
                signals.append(ma_signal)
            
            # Volume analizi
            volume_signal = await self._analyze_volume(data_with_indicators)
            if volume_signal:
                signals.append(volume_signal)
            
            # Support/Resistance analizi
            sr_signal = await self._analyze_support_resistance(data_with_indicators)
            if sr_signal:
                signals.append(sr_signal)
            
            # Combined signal hesapla
            combined_signal, combined_strength = self._calculate_combined_signal(signals)
            
            # Support/Resistance seviyeleri
            support_levels, resistance_levels = await self._find_support_resistance_levels(data_with_indicators)
            
            # Trend yönü
            trend_direction = await self._determine_trend(data_with_indicators)
            
            # Volatilite
            volatility = await self._calculate_volatility(data_with_indicators)
            
            # Volume analizi detayları
            volume_analysis = await self._detailed_volume_analysis(data_with_indicators)
            
            # Confidence hesapla
            confidence = self._calculate_confidence(signals, data_with_indicators)
            
            result = TechnicalAnalysisResult(
                symbol=self.symbol,
                current_price=float(data['close'].iloc[-1]),
                signals=signals,
                combined_signal=combined_signal,
                combined_strength=combined_strength,
                confidence=confidence,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend_direction=trend_direction,
                volatility=volatility,
                volume_analysis=volume_analysis,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Technical analysis completed: {combined_signal} ({combined_strength:.2f}) - "
                           f"Trend: {trend_direction}, Confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Technical analysis error: {e}")
            return None
    
    async def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Tüm teknik indikatörleri hesapla"""
        try:
            df = data.copy()
            
            # RSI
            df = self._calculate_rsi(df)
            
            # MACD
            df = self._calculate_macd(df)
            
            # Bollinger Bands
            df = self._calculate_bollinger_bands(df)
            
            # Moving Averages
            df = self._calculate_moving_averages(df)
            
            # Volume indicators
            df = self._calculate_volume_indicators(df)
            
            # Additional indicators
            df = self._calculate_additional_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Indicator calculation error: {e}")
            return data
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI hesapla"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            return df
        except Exception as e:
            self.logger.error(f"RSI calculation error: {e}")
            return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD hesapla"""
        try:
            ema_fast = df['close'].ewm(span=self.macd_fast).mean()
            ema_slow = df['close'].ewm(span=self.macd_slow).mean()
            
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=self.macd_signal).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            return df
        except Exception as e:
            self.logger.error(f"MACD calculation error: {e}")
            return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands hesapla"""
        try:
            df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
            bb_std = df['close'].rolling(window=self.bb_period).std()
            
            df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std)
            
            # BB position (0-1 arası, 0=lower band, 1=upper band)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # BB width (volatilite göstergesi)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            return df
        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation error: {e}")
            return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving averages hesapla"""
        try:
            df['sma_20'] = df['close'].rolling(window=self.sma_short).mean()
            df['sma_50'] = df['close'].rolling(window=self.sma_medium).mean()
            df['sma_200'] = df['close'].rolling(window=self.sma_long).mean()
            
            # EMA'lar
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            return df
        except Exception as e:
            self.logger.error(f"Moving averages calculation error: {e}")
            return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume indikatörleri hesapla"""
        try:
            df['volume_sma'] = df['volume'].rolling(window=self.volume_sma_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # On Balance Volume (OBV)
            df['obv'] = 0
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
                else:
                    df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1]
            
            # Volume Price Trend (VPT)
            df['vpt'] = 0
            for i in range(1, len(df)):
                price_change = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
                df.loc[df.index[i], 'vpt'] = df['vpt'].iloc[i-1] + (df['volume'].iloc[i] * price_change)
            
            return df
        except Exception as e:
            self.logger.error(f"Volume indicators calculation error: {e}")
            return df
    
    def _calculate_additional_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ek indikatörler hesapla"""
        try:
            # Stochastic Oscillator
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
            
            # Commodity Channel Index (CCI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr'] = true_range.rolling(window=14).mean()
            
            return df
        except Exception as e:
            self.logger.error(f"Additional indicators calculation error: {e}")
            return df
    
    async def _analyze_rsi(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """RSI analizi"""
        try:
            if 'rsi' not in df.columns or df['rsi'].isna().iloc[-1]:
                return None
            
            current_rsi = df['rsi'].iloc[-1]
            
            if current_rsi <= self.rsi_oversold:
                signal = "BUY"
                strength = min((self.rsi_oversold - current_rsi) / self.rsi_oversold, 1.0)
                description = f"RSI oversold at {current_rsi:.1f}"
            elif current_rsi >= self.rsi_overbought:
                signal = "SELL"
                strength = min((current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought), 1.0)
                description = f"RSI overbought at {current_rsi:.1f}"
            else:
                signal = "HOLD"
                # Neutral zone'da strength hesapla
                mid_point = (self.rsi_oversold + self.rsi_overbought) / 2
                distance_from_mid = abs(current_rsi - mid_point)
                max_distance = (self.rsi_overbought - self.rsi_oversold) / 2
                strength = 1.0 - (distance_from_mid / max_distance)
                description = f"RSI neutral at {current_rsi:.1f}"
            
            return TechnicalSignal(
                indicator="RSI",
                signal=signal,
                strength=strength,
                value=current_rsi,
                description=description,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"RSI analysis error: {e}")
            return None
    
    async def _analyze_macd(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """MACD analizi"""
        try:
            if 'macd' not in df.columns or df['macd'].isna().iloc[-1]:
                return None
            
            current_macd = df['macd'].iloc[-1]
            current_signal = df['macd_signal'].iloc[-1]
            current_histogram = df['macd_histogram'].iloc[-1]
            
            # MACD crossover kontrolü
            prev_macd = df['macd'].iloc[-2] if len(df) > 1 else current_macd
            prev_signal = df['macd_signal'].iloc[-2] if len(df) > 1 else current_signal
            
            # Bullish crossover (MACD > Signal)
            if current_macd > current_signal and prev_macd <= prev_signal:
                signal = "BUY"
                strength = min(abs(current_histogram) / abs(current_macd), 1.0) if current_macd != 0 else 0.5
                description = "MACD bullish crossover"
            # Bearish crossover (MACD < Signal)
            elif current_macd < current_signal and prev_macd >= prev_signal:
                signal = "SELL"
                strength = min(abs(current_histogram) / abs(current_macd), 1.0) if current_macd != 0 else 0.5
                description = "MACD bearish crossover"
            # Histogram momentum
            elif current_histogram > 0:
                signal = "BUY"
                strength = min(abs(current_histogram) / abs(current_macd), 1.0) if current_macd != 0 else 0.3
                description = f"MACD bullish momentum"
            elif current_histogram < 0:
                signal = "SELL"
                strength = min(abs(current_histogram) / abs(current_macd), 1.0) if current_macd != 0 else 0.3
                description = f"MACD bearish momentum"
            else:
                signal = "HOLD"
                strength = 0.1
                description = "MACD neutral"
            
            return TechnicalSignal(
                indicator="MACD",
                signal=signal,
                strength=strength,
                value=current_macd,
                description=description,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"MACD analysis error: {e}")
            return None
    
    async def _analyze_bollinger_bands(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Bollinger Bands analizi"""
        try:
            if 'bb_position' not in df.columns or df['bb_position'].isna().iloc[-1]:
                return None
            
            current_price = df['close'].iloc[-1]
            bb_position = df['bb_position'].iloc[-1]
            bb_width = df['bb_width'].iloc[-1]
            
            # BB position analizi
            if bb_position <= 0.1:  # Alt banda yakın
                signal = "BUY"
                strength = (0.1 - bb_position) / 0.1
                description = f"Price near lower Bollinger Band"
            elif bb_position >= 0.9:  # Üst banda yakın
                signal = "SELL"
                strength = (bb_position - 0.9) / 0.1
                description = f"Price near upper Bollinger Band"
            elif bb_width < 0.02:  # Dar bantlar (squeeze)
                signal = "HOLD"
                strength = 0.7  # Yüksek strength çünkü breakout bekleniyor
                description = f"Bollinger Band squeeze detected"
            else:
                signal = "HOLD"
                # Orta bölgede strength hesapla
                distance_from_middle = abs(bb_position - 0.5)
                strength = 1.0 - (distance_from_middle * 2)
                description = f"Price in middle of Bollinger Bands"
            
            return TechnicalSignal(
                indicator="Bollinger Bands",
                signal=signal,
                strength=max(0.1, min(strength, 1.0)),
                value=bb_position,
                description=description,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Bollinger Bands analysis error: {e}")
            return None
    
    async def _analyze_moving_averages(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Moving averages analizi"""
        try:
            if 'sma_20' not in df.columns or df['sma_20'].isna().iloc[-1]:
                return None
            
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else sma_20
            sma_200 = df['sma_200'].iloc[-1] if 'sma_200' in df.columns else sma_50
            
            signals = []
            
            # Price vs MA'lar
            if current_price > sma_20:
                signals.append(1)
            else:
                signals.append(-1)
            
            if current_price > sma_50:
                signals.append(1)
            else:
                signals.append(-1)
            
            if current_price > sma_200:
                signals.append(1)
            else:
                signals.append(-1)
            
            # MA crossovers
            if sma_20 > sma_50:
                signals.append(1)
            else:
                signals.append(-1)
            
            if sma_50 > sma_200:
                signals.append(1)
            else:
                signals.append(-1)
            
            # Golden/Death cross kontrolü
            prev_sma_20 = df['sma_20'].iloc[-2] if len(df) > 1 else sma_20
            prev_sma_50 = df['sma_50'].iloc[-2] if len(df) > 1 and 'sma_50' in df.columns else sma_50
            
            golden_cross = sma_20 > sma_50 and prev_sma_20 <= prev_sma_50
            death_cross = sma_20 < sma_50 and prev_sma_20 >= prev_sma_50
            
            if golden_cross:
                signals.append(2)  # Güçlü sinyal
            elif death_cross:
                signals.append(-2)  # Güçlü sinyal
            
            # Toplam sinyal
            total_signal = sum(signals)
            max_possible = len(signals) + 2  # +2 golden cross için
            
            if total_signal > 0:
                signal = "BUY"
                strength = total_signal / max_possible
                description = f"Moving averages bullish ({total_signal}/{max_possible})"
            elif total_signal < 0:
                signal = "SELL"
                strength = abs(total_signal) / max_possible
                description = f"Moving averages bearish ({total_signal}/{max_possible})"
            else:
                signal = "HOLD"
                strength = 0.1
                description = "Moving averages neutral"
            
            return TechnicalSignal(
                indicator="Moving Averages",
                signal=signal,
                strength=min(strength, 1.0),
                value=total_signal,
                description=description,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Moving averages analysis error: {e}")
            return None
    
    async def _analyze_volume(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Volume analizi"""
        try:
            if 'volume_ratio' not in df.columns or df['volume_ratio'].isna().iloc[-1]:
                return None
            
            current_volume_ratio = df['volume_ratio'].iloc[-1]
            price_change = df['close'].pct_change().iloc[-1]
            
            # Volume spike kontrolü
            if current_volume_ratio >= self.volume_spike_threshold:
                if price_change > 0:
                    signal = "BUY"
                    strength = min(current_volume_ratio / self.volume_spike_threshold, 1.0)
                    description = f"High volume bullish breakout"
                elif price_change < 0:
                    signal = "SELL"
                    strength = min(current_volume_ratio / self.volume_spike_threshold, 1.0)
                    description = f"High volume bearish breakdown"
                else:
                    signal = "HOLD"
                    strength = 0.5
                    description = f"High volume, price unchanged"
            elif current_volume_ratio < 0.5:  # Düşük volume
                signal = "HOLD"
                strength = 0.2
                description = f"Low volume, weak signal"
            else:
                # Normal volume
                if price_change > 0.01:  # %1'den fazla artış
                    signal = "BUY"
                    strength = 0.4
                    description = f"Normal volume, price rising"
                elif price_change < -0.01:  # %1'den fazla düşüş
                    signal = "SELL"
                    strength = 0.4
                    description = f"Normal volume, price falling"
                else:
                    signal = "HOLD"
                    strength = 0.1
                    description = f"Normal volume, price stable"
            
            return TechnicalSignal(
                indicator="Volume",
                signal=signal,
                strength=strength,
                value=current_volume_ratio,
                description=description,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Volume analysis error: {e}")
            return None
    
    async def _analyze_support_resistance(self, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Support/Resistance analizi"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Support/Resistance seviyelerini bul
            support_levels, resistance_levels = await self._find_support_resistance_levels(df)
            
            if not support_levels and not resistance_levels:
                return None
            
            # En yakın support ve resistance
            nearest_support = max([s for s in support_levels if s < current_price], default=0)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=float('inf'))
            
            # Distance to support/resistance
            support_distance = (current_price - nearest_support) / current_price if nearest_support > 0 else 1
            resistance_distance = (nearest_resistance - current_price) / current_price if nearest_resistance != float('inf') else 1
            
            # Signal generation
            if support_distance < 0.02:  # %2'den yakın support'a
                signal = "BUY"
                strength = (0.02 - support_distance) / 0.02
                description = f"Price near support level ${nearest_support:.2f}"
            elif resistance_distance < 0.02:  # %2'den yakın resistance'a
                signal = "SELL"
                strength = (0.02 - resistance_distance) / 0.02
                description = f"Price near resistance level ${nearest_resistance:.2f}"
            elif support_distance < resistance_distance:
                signal = "BUY"
                strength = 0.3
                description = f"Price closer to support than resistance"
            else:
                signal = "SELL"
                strength = 0.3
                description = f"Price closer to resistance than support"
            
            return TechnicalSignal(
                indicator="Support/Resistance",
                signal=signal,
                strength=min(strength, 1.0),
                value=min(support_distance, resistance_distance),
                description=description,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Support/Resistance analysis error: {e}")
            return None
    
    def _calculate_combined_signal(self, signals: List[TechnicalSignal]) -> Tuple[str, float]:
        """Birleşik sinyal hesapla"""
        try:
            if not signals:
                return "HOLD", 0.0
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for signal in signals:
                weight = self.signal_weights.get(signal.indicator.lower().replace(' ', '_').replace('/', '_'), 0.1)
                
                if signal.signal == "BUY":
                    weighted_score += weight * signal.strength
                elif signal.signal == "SELL":
                    weighted_score -= weight * signal.strength
                # HOLD signals don't affect the score
                
                total_weight += weight
            
            if total_weight > 0:
                normalized_score = weighted_score / total_weight
            else:
                normalized_score = 0.0
            
            # Signal determination
            if normalized_score > 0.3:
                return "BUY", min(normalized_score, 1.0)
            elif normalized_score < -0.3:
                return "SELL", min(abs(normalized_score), 1.0)
            else:
                return "HOLD", 1.0 - abs(normalized_score)
            
        except Exception as e:
            self.logger.error(f"Combined signal calculation error: {e}")
            return "HOLD", 0.0
    
    async def _find_support_resistance_levels(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Support ve resistance seviyelerini bul"""
        try:
            if len(df) < self.sr_lookback:
                return [], []
            
            recent_data = df.tail(self.sr_lookback)
            
            # Local minima (support) ve maxima (resistance) bul
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            support_levels = []
            resistance_levels = []
            
            # Pivot points bul
            for i in range(2, len(recent_data) - 2):
                # Local minimum (support)
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    support_levels.append(lows[i])
                
                # Local maximum (resistance)
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    resistance_levels.append(highs[i])
            
            # Benzer seviyeleri grupla
            support_levels = self._group_similar_levels(support_levels)
            resistance_levels = self._group_similar_levels(resistance_levels)
            
            return support_levels,
