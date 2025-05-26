"""
Ensemble Model
Tüm AI/ML modellerinin sonuçlarını birleştiren ana model
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .lstm_model import LSTMModel, LSTMPrediction
from .technical_analysis import TechnicalAnalysisModel, TechnicalAnalysisResult
from .sentiment_analysis import SentimentAnalysisModel, SentimentAnalysisResult

class SignalConfidence(Enum):
    """Sinyal güven seviyeleri"""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"

@dataclass
class ModelResult:
    """Model sonucu"""
    model_name: str
    signal: str  # BUY, SELL, HOLD
    strength: float  # 0-1 arası
    confidence: float  # 0-1 arası
    weight: float  # Model ağırlığı
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class EnsembleResult:
    """Ensemble model sonucu"""
    symbol: str
    final_signal: str  # BUY, SELL, HOLD
    signal_strength: float  # 0-1 arası
    confidence_level: SignalConfidence
    overall_confidence: float  # 0-1 arası
    
    # Model sonuçları
    lstm_result: Optional[ModelResult]
    technical_result: Optional[ModelResult]
    sentiment_result: Optional[ModelResult]
    
    # Detaylı analiz
    signal_consensus: float  # Model uyumu (-1 to 1)
    risk_assessment: str
    market_conditions: str
    recommendation: str
    
    # Ek bilgiler
    price_target: Optional[float]
    stop_loss_suggestion: Optional[float]
    take_profit_suggestion: Optional[float]
    
    timestamp: datetime

class EnsembleModel:
    """Ensemble AI/ML model sistemi"""
    
    def __init__(self, symbol: str = 'BTCUSDT'):
        """
        Ensemble model başlatıcı
        
        Args:
            symbol: Trading pair
        """
        self.symbol = symbol
        self.logger = logging.getLogger(__name__)
        
        # Alt modeller
        self.lstm_model = LSTMModel(symbol)
        self.technical_model = TechnicalAnalysisModel(symbol)
        self.sentiment_model = SentimentAnalysisModel(symbol)
        
        # Model ağırlıkları (toplam 1.0)
        self.model_weights = {
            'lstm': 0.35,        # LSTM fiyat tahmini
            'technical': 0.45,   # Teknik analiz
            'sentiment': 0.20    # Sentiment analizi
        }
        
        # Sinyal eşikleri
        self.strong_signal_threshold = 0.7
        self.weak_signal_threshold = 0.3
        
        # Consensus eşikleri
        self.high_consensus_threshold = 0.8
        self.medium_consensus_threshold = 0.5
        
        # Risk parametreleri
        self.max_risk_per_trade = 0.02  # %2
        self.risk_reward_ratio = 2.0    # 1:2
        
        # Model durumları
        self.models_initialized = False
        
    async def initialize(self):
        """Tüm modelleri başlat"""
        try:
            # LSTM model'i başlat
            await self.lstm_model.initialize()
            
            # Diğer modeller başlatma gerektirmiyor
            
            self.models_initialized = True
            self.logger.info(f"Ensemble model initialized for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"Ensemble model initialization error: {e}")
            raise
    
    async def analyze(self, data: pd.DataFrame) -> Optional[EnsembleResult]:
        """
        Ensemble analiz yap
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            EnsembleResult: Birleşik analiz sonucu
        """
        try:
            if not self.models_initialized:
                await self.initialize()
            
            self.logger.info(f"Starting ensemble analysis for {self.symbol}")
            
            # Paralel model analizleri
            model_results = await self._run_parallel_analysis(data)
            
            # Sonuçları birleştir
            ensemble_result = await self._combine_results(model_results, data)
            
            self.logger.info(f"Ensemble analysis completed: {ensemble_result.final_signal} "
                           f"(Strength: {ensemble_result.signal_strength:.2f}, "
                           f"Confidence: {ensemble_result.confidence_level.value})")
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Ensemble analysis error: {e}")
            return None
    
    async def _run_parallel_analysis(self, data: pd.DataFrame) -> Dict[str, Optional[ModelResult]]:
        """Paralel model analizleri"""
        try:
            # Async tasks oluştur
            tasks = []
            
            # LSTM analizi
            tasks.append(self._run_lstm_analysis(data))
            
            # Technical analizi
            tasks.append(self._run_technical_analysis(data))
            
            # Sentiment analizi
            tasks.append(self._run_sentiment_analysis())
            
            # Paralel çalıştır
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                'lstm': results[0] if not isinstance(results[0], Exception) else None,
                'technical': results[1] if not isinstance(results[1], Exception) else None,
                'sentiment': results[2] if not isinstance(results[2], Exception) else None
            }
            
        except Exception as e:
            self.logger.error(f"Parallel analysis error: {e}")
            return {'lstm': None, 'technical': None, 'sentiment': None}
    
    async def _run_lstm_analysis(self, data: pd.DataFrame) -> Optional[ModelResult]:
        """LSTM model analizi"""
        try:
            prediction = await self.lstm_model.predict(data)
            
            if not prediction:
                return None
            
            return ModelResult(
                model_name="LSTM",
                signal=prediction.signal,
                strength=prediction.signal_strength,
                confidence=prediction.confidence,
                weight=self.model_weights['lstm'],
                details={
                    'current_price': prediction.current_price,
                    'predicted_price': prediction.predicted_price,
                    'price_change_percent': prediction.price_change_percent,
                    'model_accuracy': prediction.model_accuracy,
                    'prediction_horizon': prediction.prediction_horizon
                },
                timestamp=prediction.timestamp
            )
            
        except Exception as e:
            self.logger.error(f"LSTM analysis error: {e}")
            return None
    
    async def _run_technical_analysis(self, data: pd.DataFrame) -> Optional[ModelResult]:
        """Technical analysis model analizi"""
        try:
            analysis = await self.technical_model.analyze(data)
            
            if not analysis:
                return None
            
            return ModelResult(
                model_name="Technical Analysis",
                signal=analysis.combined_signal,
                strength=analysis.combined_strength,
                confidence=analysis.confidence,
                weight=self.model_weights['technical'],
                details={
                    'current_price': analysis.current_price,
                    'trend_direction': analysis.trend_direction,
                    'volatility': analysis.volatility,
                    'support_levels': analysis.support_levels,
                    'resistance_levels': analysis.resistance_levels,
                    'signals': [
                        {
                            'indicator': signal.indicator,
                            'signal': signal.signal,
                            'strength': signal.strength,
                            'description': signal.description
                        }
                        for signal in analysis.signals
                    ]
                },
                timestamp=analysis.timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Technical analysis error: {e}")
            return None
    
    async def _run_sentiment_analysis(self) -> Optional[ModelResult]:
        """Sentiment analysis model analizi"""
        try:
            analysis = await self.sentiment_model.analyze()
            
            if not analysis:
                return None
            
            return ModelResult(
                model_name="Sentiment Analysis",
                signal=analysis.combined_sentiment.signal,
                strength=analysis.combined_sentiment.confidence,  # Sentiment'te strength yerine confidence kullan
                confidence=analysis.combined_sentiment.confidence,
                weight=self.model_weights['sentiment'],
                details={
                    'sentiment_score': analysis.combined_sentiment.sentiment_score,
                    'sentiment_level': analysis.combined_sentiment.sentiment_level.value,
                    'market_mood': analysis.market_mood,
                    'fear_greed_score': analysis.fear_greed_index.sentiment_score if analysis.fear_greed_index else None,
                    'trending_coins_count': len(analysis.trending_coins),
                    'contrarian_signal': analysis.contrarian_signal
                },
                timestamp=analysis.timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return None
    
    async def _combine_results(self, model_results: Dict[str, Optional[ModelResult]], 
                              data: pd.DataFrame) -> EnsembleResult:
        """Model sonuçlarını birleştir"""
        try:
            # Geçerli sonuçları filtrele
            valid_results = {k: v for k, v in model_results.items() if v is not None}
            
            if not valid_results:
                return self._create_default_result()
            
            # Weighted signal hesapla
            final_signal, signal_strength = self._calculate_weighted_signal(valid_results)
            
            # Consensus hesapla
            signal_consensus = self._calculate_consensus(valid_results)
            
            # Confidence level belirle
            overall_confidence = self._calculate_overall_confidence(valid_results, signal_consensus)
            confidence_level = self._confidence_to_level(overall_confidence)
            
            # Risk assessment
            risk_assessment = self._assess_risk(valid_results, signal_consensus)
            
            # Market conditions
            market_conditions = self._assess_market_conditions(valid_results)
            
            # Recommendation
            recommendation = self._generate_recommendation(
                final_signal, signal_strength, confidence_level, risk_assessment
            )
            
            # Price targets
            price_target, stop_loss, take_profit = self._calculate_price_targets(
                valid_results, data, final_signal
            )
            
            return EnsembleResult(
                symbol=self.symbol,
                final_signal=final_signal,
                signal_strength=signal_strength,
                confidence_level=confidence_level,
                overall_confidence=overall_confidence,
                lstm_result=model_results.get('lstm'),
                technical_result=model_results.get('technical'),
                sentiment_result=model_results.get('sentiment'),
                signal_consensus=signal_consensus,
                risk_assessment=risk_assessment,
                market_conditions=market_conditions,
                recommendation=recommendation,
                price_target=price_target,
                stop_loss_suggestion=stop_loss,
                take_profit_suggestion=take_profit,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Results combination error: {e}")
            return self._create_default_result()
    
    def _calculate_weighted_signal(self, results: Dict[str, ModelResult]) -> Tuple[str, float]:
        """Ağırlıklı sinyal hesapla"""
        try:
            weighted_score = 0.0
            total_weight = 0.0
            
            for result in results.values():
                # Signal'i score'a çevir
                if result.signal == "BUY":
                    signal_score = result.strength
                elif result.signal == "SELL":
                    signal_score = -result.strength
                else:  # HOLD
                    signal_score = 0.0
                
                # Confidence ile ağırlıklandır
                effective_weight = result.weight * result.confidence
                weighted_score += signal_score * effective_weight
                total_weight += effective_weight
            
            if total_weight > 0:
                normalized_score = weighted_score / total_weight
            else:
                normalized_score = 0.0
            
            # Final signal belirle
            if normalized_score > self.weak_signal_threshold:
                final_signal = "BUY"
                signal_strength = min(abs(normalized_score), 1.0)
            elif normalized_score < -self.weak_signal_threshold:
                final_signal = "SELL"
                signal_strength = min(abs(normalized_score), 1.0)
            else:
                final_signal = "HOLD"
                signal_strength = 1.0 - abs(normalized_score)
            
            return final_signal, signal_strength
            
        except Exception as e:
            self.logger.error(f"Weighted signal calculation error: {e}")
            return "HOLD", 0.0
    
    def _calculate_consensus(self, results: Dict[str, ModelResult]) -> float:
        """Model consensus hesapla"""
        try:
            if len(results) < 2:
                return 0.0
            
            signals = [result.signal for result in results.values()]
            
            # En yaygın sinyal
            signal_counts = {
                'BUY': signals.count('BUY'),
                'SELL': signals.count('SELL'),
                'HOLD': signals.count('HOLD')
            }
            
            max_count = max(signal_counts.values())
            total_count = len(signals)
            
            # Consensus oranı
            consensus = max_count / total_count
            
            # Strength'lerin uyumu
            strengths = [result.strength for result in results.values()]
            strength_std = np.std(strengths) if len(strengths) > 1 else 0
            strength_consensus = 1.0 - min(strength_std, 1.0)
            
            # Birleşik consensus
            overall_consensus = (consensus + strength_consensus) / 2
            
            return overall_consensus
            
        except Exception as e:
            self.logger.error(f"Consensus calculation error: {e}")
            return 0.0
    
    def _calculate_overall_confidence(self, results: Dict[str, ModelResult], 
                                    consensus: float) -> float:
        """Genel güven seviyesi hesapla"""
        try:
            # Model confidence'larının ortalaması
            confidences = [result.confidence for result in results.values()]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Consensus ile ağırlıklandır
            overall_confidence = (avg_confidence * 0.7) + (consensus * 0.3)
            
            return min(overall_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Overall confidence calculation error: {e}")
            return 0.0
    
    def _confidence_to_level(self, confidence: float) -> SignalConfidence:
        """Confidence'ı level'a çevir"""
        if confidence >= 0.8:
            return SignalConfidence.VERY_HIGH
        elif confidence >= 0.6:
            return SignalConfidence.HIGH
        elif confidence >= 0.4:
            return SignalConfidence.MEDIUM
        elif confidence >= 0.2:
            return SignalConfidence.LOW
        else:
            return SignalConfidence.VERY_LOW
    
    def _assess_risk(self, results: Dict[str, ModelResult], consensus: float) -> str:
        """Risk değerlendirmesi"""
        try:
            # Consensus'a göre risk
            if consensus >= self.high_consensus_threshold:
                risk_level = "Low"
            elif consensus >= self.medium_consensus_threshold:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            # Volatilite kontrolü (technical analysis'den)
            if 'technical' in results:
                tech_details = results['technical'].details
                volatility = tech_details.get('volatility', 0)
                
                if volatility > 0.05:  # %5'ten fazla volatilite
                    if risk_level == "Low":
                        risk_level = "Medium"
                    elif risk_level == "Medium":
                        risk_level = "High"
            
            return risk_level
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            return "High"
    
    def _assess_market_conditions(self, results: Dict[str, ModelResult]) -> str:
        """Market koşulları değerlendirmesi"""
        try:
            conditions = []
            
            # Technical analysis'den trend
            if 'technical' in results:
                tech_details = results['technical'].details
                trend = tech_details.get('trend_direction', 'SIDEWAYS')
                conditions.append(f"Trend: {trend}")
            
            # Sentiment'den market mood
            if 'sentiment' in results:
                sent_details = results['sentiment'].details
                mood = sent_details.get('market_mood', 'Neutral')
                conditions.append(f"Sentiment: {mood}")
            
            # LSTM'den price prediction
            if 'lstm' in results:
                lstm_details = results['lstm'].details
                price_change = lstm_details.get('price_change_percent', 0)
                if abs(price_change) > 5:
                    conditions.append(f"Expected: {price_change:+.1f}%")
            
            return " | ".join(conditions) if conditions else "Normal"
            
        except Exception as e:
            self.logger.error(f"Market conditions assessment error: {e}")
            return "Unknown"
    
    def _generate_recommendation(self, signal: str, strength: float, 
                               confidence: SignalConfidence, risk: str) -> str:
        """Trading önerisi oluştur"""
        try:
            recommendations = []
            
            # Ana sinyal
            if signal == "BUY":
                if strength > self.strong_signal_threshold:
                    recommendations.append("Strong BUY signal detected")
                else:
                    recommendations.append("Weak BUY signal detected")
            elif signal == "SELL":
                if strength > self.strong_signal_threshold:
                    recommendations.append("Strong SELL signal detected")
                else:
                    recommendations.append("Weak SELL signal detected")
            else:
                recommendations.append("HOLD - No clear direction")
            
            # Confidence uyarısı
            if confidence in [SignalConfidence.VERY_LOW, SignalConfidence.LOW]:
                recommendations.append("⚠️ Low confidence - proceed with caution")
            elif confidence == SignalConfidence.VERY_HIGH:
                recommendations.append("✅ High confidence signal")
            
            # Risk uyarısı
            if risk == "High":
                recommendations.append("⚠️ High risk conditions - reduce position size")
            elif risk == "Low":
                recommendations.append("✅ Favorable risk conditions")
            
            return " | ".join(recommendations)
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return "Unable to generate recommendation"
    
    def _calculate_price_targets(self, results: Dict[str, ModelResult], 
                               data: pd.DataFrame, signal: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Price target'ları hesapla"""
        try:
            current_price = float(data['close'].iloc[-1])
            
            # LSTM'den price target
            price_target = None
            if 'lstm' in results:
                lstm_details = results['lstm'].details
                price_target = lstm_details.get('predicted_price')
            
            # Technical analysis'den support/resistance
            support_levels = []
            resistance_levels = []
            
            if 'technical' in results:
                tech_details = results['technical'].details
                support_levels = tech_details.get('support_levels', [])
                resistance_levels = tech_details.get('resistance_levels', [])
            
            # Stop loss ve take profit hesapla
            stop_loss = None
            take_profit = None
            
            if signal == "BUY":
                # Stop loss: En yakın support veya %2 aşağı
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < current_price], default=None)
                    if nearest_support:
                        stop_loss = nearest_support
                
                if not stop_loss:
                    stop_loss = current_price * (1 - self.max_risk_per_trade)
                
                # Take profit: Risk/reward oranına göre
                risk_amount = current_price - stop_loss
                take_profit = current_price + (risk_amount * self.risk_reward_ratio)
                
                # Resistance kontrolü
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
                    if nearest_resistance and nearest_resistance < take_profit:
                        take_profit = nearest_resistance * 0.99  # %1 altında
            
            elif signal == "SELL":
                # Stop loss: En yakın resistance veya %2 yukarı
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
                    if nearest_resistance:
                        stop_loss = nearest_resistance
                
                if not stop_loss:
                    stop_loss = current_price * (1 + self.max_risk_per_trade)
                
                # Take profit: Risk/reward oranına göre
                risk_amount = stop_loss - current_price
                take_profit = current_price - (risk_amount * self.risk_reward_ratio)
                
                # Support kontrolü
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < current_price], default=None)
                    if nearest_support and nearest_support > take_profit:
                        take_profit = nearest_support * 1.01  # %1 üstünde
            
            return price_target, stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Price targets calculation error: {e}")
            return None, None, None
    
    def _create_default_result(self) -> EnsembleResult:
        """Varsayılan sonuç oluştur"""
        return EnsembleResult(
            symbol=self.symbol,
            final_signal="HOLD",
            signal_strength=0.0,
            confidence_level=SignalConfidence.VERY_LOW,
            overall_confidence=0.0,
            lstm_result=None,
            technical_result=None,
            sentiment_result=None,
            signal_consensus=0.0,
            risk_assessment="High",
            market_conditions="Unknown",
            recommendation="No analysis available - HOLD recommended",
            price_target=None,
            stop_loss_suggestion=None,
            take_profit_suggestion=None,
            timestamp=datetime.now()
        )
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Model durumlarını al"""
        try:
            return {
                'ensemble_initialized': self.models_initialized,
                'lstm_status': {
                    'initialized': self.lstm_model.is_trained,
                    'accuracy': self.lstm_model.last_accuracy,
                    'last_training': self.lstm_model.last_training_time.isoformat() if self.lstm_model.last_training_time else None
                },
                'technical_status': {
                    'initialized': True,
                    'symbol': self.technical_model.symbol
                },
                'sentiment_status': {
                    'initialized': True,
                    'symbol': self.sentiment_model.symbol,
                    'cache_size': len(self.sentiment_model.cache)
                },
                'model_weights': self.model_weights,
                'thresholds': {
                    'strong_signal': self.strong_signal_threshold,
                    'weak_signal': self.weak_signal_threshold,
                    'high_consensus': self.high_consensus_threshold,
                    'medium_consensus': self.medium_consensus_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"Model status error: {e}")
            return {}
    
    async def update_model_weights(self, weights: Dict[str, float]):
        """Model ağırlıklarını güncelle"""
        try:
            # Ağırlıkların toplamının 1.0 olduğunu kontrol et
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
            
            self.model_weights.update(weights)
            self.logger.info(f"Model weights updated: {self.model_weights}")
            
        except Exception as e:
            self.logger.error(f"Model weights update error: {e}")
            raise
    
    async def cleanup(self):
        """Kaynakları temizle"""
        try:
            # Alt modelleri temizle
            await self.lstm_model.cleanup()
            await self.sentiment_model.cleanup()
            
            self.models_initialized = False
            self.logger.info("Ensemble model cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Ensemble cleanup error: {e}")
    
    def get_analysis_summary(self, result: EnsembleResult) -> str:
        """Analiz özetini al"""
        try:
            summary_parts = []
            
            # Ana sinyal
            summary_parts.append(f"Signal: {result.final_signal}")
            summary_parts.append(f"Strength: {result.signal_strength:.2f}")
            summary_parts.append(f"Confidence: {result.confidence_level.value}")
            
            # Model uyumu
            summary_parts.append(f"Consensus: {result.signal_consensus:.2f}")
            
            # Risk
            summary_parts.append(f"Risk: {result.risk_assessment}")
            
            # Price targets
            if result.price_target:
                summary_parts.append(f"Target: ${result.price_target:.2f}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Analysis summary error: {e}")
            return "Error generating summary"
