"""
Technical Analysis Model
Gelişmiş teknik analiz indikatörleri ve sinyal üretimi (pandas-ta kullanarak)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
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
    """Gelişmiş teknik analiz modeli (pandas-ta kullanarak)"""
    
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
        """Tüm teknik indikatörleri hesapla (pandas-ta kullanarak)"""
        try:
            df = data.copy()
            
            # pandas-ta ile indikatörleri hesapla
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
            
            # MACD
            macd_data = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            if macd_data is not None:
                df = pd.concat([df, macd_data], axis=1)
            
            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std)
            if bb_data is not None:
                df = pd.concat([df, bb_data], axis=1)
                # BB position hesapla
                if f'BBL_{self.bb_period}_{self.bb_std}' in df.columns and f'BBU_{self.bb_period}_{self.bb_std}' in df.columns:
                    bb_lower = df[f'BBL_{self.bb_period}_{self.bb_std}']
                    bb_upper = df[f'BBU_{self.bb_period}_{self.bb_std}']
                    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
                    df['bb_width'] = (bb_upper - bb_lower) / df[f'BBM_{self.bb_period}_{self.bb_std}']
            
            # Moving Averages
            df['sma_20'] = ta.sma(df['close'], length=self.sma_short)
            df['sma_50'] = ta.sma(df['close'], length=self.sma_medium)
            df['sma_200'] = ta.sma(df['close'], length=self.sma_long)
            df['ema_12'] = ta.ema(df['close'], length=12)
            df['ema_26'] = ta.ema(df['close'], length=26)
            
            # Volume indicators
            df['volume_sma'] = ta.sma(df['volume'], length=self.volume_sma_period)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # OBV
            df['obv'] = ta.obv(df['close'], df['volume'])
            
            # Additional indicators
            df['stoch_k'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHk_14_3_3']
            df['stoch_d'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHd_14_3_3']
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
            df['cci'] = ta.cci(df['high'], df['low'], df['close'])
            df['atr'] = ta.atr(df['high'], df['low'], df['close'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Indicator calculation error: {e}")
            return data
    
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
            macd_col = f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
            signal_col = f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
            hist_col = f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
            
            if macd_col not in df.columns or df[macd_col].isna().iloc[-1]:
                return None
            
            current_macd = df[macd_col].iloc[-1]
            current_signal = df[signal_col].iloc[-1]
            current_histogram = df[hist_col].iloc[-1]
            
            # MACD crossover kontrolü
            prev_macd = df[macd_col].iloc[-2] if len(df) > 1 else current_macd
            prev_signal = df[signal_col].iloc[-2] if len(df) > 1 else current_signal
            
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
            bb_width = df['bb_width'].iloc[-1] if 'bb_width' in df.columns else 0.05
            
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
            
            return support_levels, resistance_levels
            
        except Exception as e:
            self.logger.error(f"Support/Resistance levels calculation error: {e}")
            return [], []
    
    def _group_similar_levels(self, levels: List[float], threshold: float = 0.01) -> List[float]:
        """Benzer seviyeleri grupla"""
        if not levels:
            return []
        
        levels = sorted(levels)
        grouped = []
        current_group = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_group[-1]) / current_group[-1] <= threshold:
                current_group.append(level)
            else:
                # Grup ortalamasını al
                grouped.append(sum(current_group) / len(current_group))
                current_group = [level]
        
        # Son grubu ekle
        grouped.append(sum(current_group) / len(current_group))
        
        return grouped
    
    async def _determine_trend(self, df: pd.DataFrame) -> str:
        """Trend yönünü belirle"""
        try:
            if len(df) < self.trend_period:
                return "SIDEWAYS"
            
            recent_data = df.tail(self.trend_period)
            
            # Price trend
            price_slope = np.polyfit(range(len(recent_data)), recent_data['close'], 1)[0]
            
            # Moving average trend
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                sma_20_current = df['sma_20'].iloc[-1]
                sma_50_current = df['sma_50'].iloc[-1]
                
                if sma_20_current > sma_50_current and price_slope > 0:
                    return "BULLISH"
                elif sma_20_current < sma_50_current and price_slope < 0:
                    return "BEARISH"
            
            # Sadece price slope'a göre
            if price_slope > 0.001:
                return "BULLISH"
            elif price_slope < -0.001:
                return "BEARISH"
            else:
                return "SIDEWAYS"
                
        except Exception as e:
            self.logger.error(f"Trend determination error: {e}")
            return "SIDEWAYS"
    
    async def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Volatilite hesapla"""
        try:
            if 'atr' in df.columns and not df['atr'].isna().iloc[-1]:
                current_price = df['close'].iloc[-1]
                atr = df['atr'].iloc[-1]
                return atr / current_price
            else:
                # Fallback: price volatility
                returns = df['close'].pct_change().dropna()
                return returns.std() if len(returns) > 0 else 0.0
                
        except Exception as e:
            self.logger.error(f"Volatility calculation error: {e}")
            return 0.0
    
    async def _detailed_volume_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detaylı volume analizi"""
        try:
            volume_analysis = {
                'current_volume': float(df['volume'].iloc[-1]),
                'average_volume': float(df['volume'].mean()),
                'volume_trend': 'neutral',
                'volume_spike': False,
                'obv_trend': 'neutral'
            }
            
            if 'volume_ratio' in df.columns:
                current_ratio = df['volume_ratio'].iloc[-1]
                volume_analysis['volume_ratio'] = float(current_ratio)
                volume_analysis['volume_spike'] = current_ratio >= self.volume_spike_threshold
                
                # Volume trend
                if len(df) >= 5:
                    recent_volume = df['volume_ratio'].tail(5).mean()
                    if recent_volume > 1.2:
                        volume_analysis['volume_trend'] = 'increasing'
                    elif recent_volume < 0.8:
                        volume_analysis['volume_trend'] = 'decreasing'
            
            # OBV trend
            if 'obv' in df.columns and len(df) >= 10:
                obv_slope = np.polyfit(range(10), df['obv'].tail(10), 1)[0]
                if obv_slope > 0:
                    volume_analysis['obv_trend'] = 'bullish'
                elif obv_slope < 0:
                    volume_analysis['obv_trend'] = 'bearish'
            
            return volume_analysis
            
        except Exception as e:
            self.logger.error(f"Volume analysis error: {e}")
            return {
                'current_volume': 0,
                'average_volume': 0,
                'volume_trend': 'neutral',
                'volume_spike': False,
                'obv_trend': 'neutral'
            }
    
    def _calculate_confidence(self, signals: List[TechnicalSignal], df: pd.DataFrame) -> float:
        """Confidence score hesapla"""
        try:
            if not signals:
                return 0.0
            
            # Signal agreement
            buy_signals = sum(1 for s in signals if s.signal == "BUY")
            sell_signals = sum(1 for s in signals if s.signal == "SELL")
            total_signals = len(signals)
            
            # Agreement ratio
            max_agreement = max(buy_signals, sell_signals)
            agreement_ratio = max_agreement / total_signals
            
            # Average strength
            avg_strength = sum(s.strength for s in signals) / total_signals
            
            # Data quality factor
            data_quality = min(len(df) / 200, 1.0)  # 200+ bars için full confidence
            
            # Volume confirmation
            volume_factor = 1.0
            if 'volume_ratio' in df.columns:
                current_volume_ratio = df['volume_ratio'].iloc[-1]
                if current_volume_ratio >= 1.5:  # High volume
                    volume_factor = 1.2
                elif current_volume_ratio < 0.5:  # Low volume
                    volume_factor = 0.8
            
            # Combined confidence
            confidence = (agreement_ratio * 0.4 + avg_strength * 0.4 + data_quality * 0.2) * volume_factor
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.5


# Örnek kullanım
if __name__ == "__main__":
    async def test_technical_analysis():
        """Test fonksiyonu"""
        # Sample data oluştur
        dates = pd.date_range(start='2023-01-01', periods=300, freq='1H')
        np.random.seed(42)
        
        # Realistic price data
        price = 50000
        prices = [price]
        volumes = []
        
        for i in range(299):
            change = np.random.normal(0, 0.02)  # %2 volatilite
            price = price * (1 + change)
            prices.append(price)
            volumes.append(np.random.uniform(100, 1000))
        
        # DataFrame oluştur
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Technical analysis
        ta_model = TechnicalAnalysisModel('BTCUSDT')
        result = await ta_model.analyze(data)
        
        if result:
            print(f"Symbol: {result.symbol}")
            print(f"Current Price: ${result.current_price:.2f}")
            print(f"Combined Signal: {result.combined_signal} ({result.combined_strength:.2f})")
            print(f"Trend: {result.trend_direction}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Volatility: {result.volatility:.4f}")
            
            print("\nIndividual Signals:")
            for signal in result.signals:
                print(f"  {signal.indicator}: {signal.signal} ({signal.strength:.2f}) - {signal.description}")
            
            print(f"\nSupport Levels: {[f'${s:.2f}' for s in result.support_levels[:3]]}")
            print(f"Resistance Levels: {[f'${r:.2f}' for r in result.resistance_levels[:3]]}")
            
            print(f"\nVolume Analysis: {result.volume_analysis}")
        else:
            print("Technical analysis failed")
    
    # Test çalıştır
    import asyncio
    asyncio.run(test_technical_analysis())
