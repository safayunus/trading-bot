"""
Sentiment Analysis Model
Market sentiment analizi ve Fear & Greed Index entegrasyonu
"""

import asyncio
import logging
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class SentimentLevel(Enum):
    """Sentiment seviyeleri"""
    EXTREME_FEAR = "Extreme Fear"
    FEAR = "Fear"
    NEUTRAL = "Neutral"
    GREED = "Greed"
    EXTREME_GREED = "Extreme Greed"

@dataclass
class SentimentData:
    """Sentiment verisi"""
    source: str
    sentiment_score: float  # 0-100 arası
    sentiment_level: SentimentLevel
    signal: str  # BUY, SELL, HOLD
    confidence: float
    description: str
    timestamp: datetime

@dataclass
class SentimentAnalysisResult:
    """Sentiment analiz sonucu"""
    symbol: str
    fear_greed_index: Optional[SentimentData]
    trending_coins: List[Dict[str, Any]]
    social_sentiment: Optional[SentimentData]
    combined_sentiment: SentimentData
    market_mood: str
    contrarian_signal: str  # Contrarian trading için
    timestamp: datetime

class SentimentAnalysisModel:
    """Market sentiment analiz modeli"""
    
    def __init__(self, symbol: str = 'BTCUSDT'):
        """
        Sentiment Analysis model başlatıcı
        
        Args:
            symbol: Trading pair
        """
        self.symbol = symbol
        self.logger = logging.getLogger(__name__)
        
        # API endpoints
        self.fear_greed_api = "https://api.alternative.me/fng/"
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        
        # Sentiment thresholds
        self.extreme_fear_threshold = 25
        self.fear_threshold = 45
        self.greed_threshold = 55
        self.extreme_greed_threshold = 75
        
        # Signal weights
        self.fear_greed_weight = 0.4
        self.trending_weight = 0.3
        self.social_weight = 0.3
        
        # Cache
        self.cache = {}
        self.cache_ttl = 3600  # 1 saat
        
    async def analyze(self) -> Optional[SentimentAnalysisResult]:
        """
        Sentiment analizi yap
        
        Returns:
            SentimentAnalysisResult: Analiz sonucu
        """
        try:
            # Fear & Greed Index
            fear_greed_data = await self._get_fear_greed_index()
            
            # Trending coins
            trending_coins = await self._get_trending_coins()
            
            # Social sentiment (basit implementasyon)
            social_sentiment = await self._get_social_sentiment()
            
            # Combined sentiment hesapla
            combined_sentiment = self._calculate_combined_sentiment(
                fear_greed_data, trending_coins, social_sentiment
            )
            
            # Market mood
            market_mood = self._determine_market_mood(combined_sentiment)
            
            # Contrarian signal
            contrarian_signal = self._generate_contrarian_signal(combined_sentiment)
            
            result = SentimentAnalysisResult(
                symbol=self.symbol,
                fear_greed_index=fear_greed_data,
                trending_coins=trending_coins,
                social_sentiment=social_sentiment,
                combined_sentiment=combined_sentiment,
                market_mood=market_mood,
                contrarian_signal=contrarian_signal,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Sentiment analysis completed: {combined_sentiment.sentiment_level.value} "
                           f"({combined_sentiment.sentiment_score:.1f}) - Signal: {combined_sentiment.signal}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return None
    
    async def _get_fear_greed_index(self) -> Optional[SentimentData]:
        """Fear & Greed Index verilerini al"""
        try:
            cache_key = "fear_greed_index"
            
            # Cache kontrolü
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.fear_greed_api) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'data' in data and len(data['data']) > 0:
                            fng_data = data['data'][0]
                            score = int(fng_data['value'])
                            
                            # Sentiment level belirle
                            sentiment_level = self._score_to_sentiment_level(score)
                            
                            # Signal üret
                            signal = self._sentiment_to_signal(score, contrarian=False)
                            
                            # Confidence hesapla
                            confidence = self._calculate_fng_confidence(score)
                            
                            sentiment_data = SentimentData(
                                source="Fear & Greed Index",
                                sentiment_score=float(score),
                                sentiment_level=sentiment_level,
                                signal=signal,
                                confidence=confidence,
                                description=f"Fear & Greed Index: {score} ({sentiment_level.value})",
                                timestamp=datetime.now()
                            )
                            
                            # Cache'e kaydet
                            self._cache_data(cache_key, sentiment_data)
                            
                            return sentiment_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Fear & Greed Index error: {e}")
            return None
    
    async def _get_trending_coins(self) -> List[Dict[str, Any]]:
        """Trending coins verilerini al"""
        try:
            cache_key = "trending_coins"
            
            # Cache kontrolü
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            url = f"{self.coingecko_api}/search/trending"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        trending_coins = []
                        
                        if 'coins' in data:
                            for coin_data in data['coins'][:10]:  # İlk 10 coin
                                coin = coin_data['item']
                                
                                trending_coins.append({
                                    'id': coin['id'],
                                    'name': coin['name'],
                                    'symbol': coin['symbol'],
                                    'market_cap_rank': coin.get('market_cap_rank'),
                                    'price_btc': coin.get('price_btc'),
                                    'score': coin.get('score', 0)
                                })
                        
                        # Cache'e kaydet
                        self._cache_data(cache_key, trending_coins)
                        
                        return trending_coins
            
            return []
            
        except Exception as e:
            self.logger.error(f"Trending coins error: {e}")
            return []
    
    async def _get_social_sentiment(self) -> Optional[SentimentData]:
        """Social sentiment analizi (basit implementasyon)"""
        try:
            # Bu basit bir implementasyon
            # Gerçek uygulamada Twitter API, Reddit API vb. kullanılabilir
            
            # Trending coins'e göre basit sentiment hesapla
            trending_coins = await self._get_trending_coins()
            
            if not trending_coins:
                return None
            
            # Symbol'ümüz trending'de mi kontrol et
            symbol_base = self.symbol.replace('USDT', '').replace('BTC', '').lower()
            
            is_trending = any(
                coin['symbol'].lower() == symbol_base or 
                coin['id'].lower() == symbol_base
                for coin in trending_coins
            )
            
            if is_trending:
                # Trending ise pozitif sentiment
                score = 65.0
                signal = "BUY"
                description = f"{symbol_base.upper()} is trending - positive social sentiment"
            else:
                # Trending değilse nötr
                score = 50.0
                signal = "HOLD"
                description = f"{symbol_base.upper()} not trending - neutral social sentiment"
            
            sentiment_level = self._score_to_sentiment_level(score)
            confidence = 0.6  # Basit implementasyon için düşük confidence
            
            return SentimentData(
                source="Social Media",
                sentiment_score=score,
                sentiment_level=sentiment_level,
                signal=signal,
                confidence=confidence,
                description=description,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Social sentiment error: {e}")
            return None
    
    def _calculate_combined_sentiment(self, fear_greed: Optional[SentimentData],
                                    trending_coins: List[Dict[str, Any]],
                                    social: Optional[SentimentData]) -> SentimentData:
        """Birleşik sentiment hesapla"""
        try:
            weighted_score = 0.0
            total_weight = 0.0
            signals = []
            
            # Fear & Greed Index
            if fear_greed:
                weighted_score += fear_greed.sentiment_score * self.fear_greed_weight
                total_weight += self.fear_greed_weight
                signals.append(fear_greed.signal)
            
            # Trending coins etkisi
            if trending_coins:
                trending_score = self._calculate_trending_sentiment(trending_coins)
                weighted_score += trending_score * self.trending_weight
                total_weight += self.trending_weight
                
                trending_signal = self._sentiment_to_signal(trending_score)
                signals.append(trending_signal)
            
            # Social sentiment
            if social:
                weighted_score += social.sentiment_score * self.social_weight
                total_weight += self.social_weight
                signals.append(social.signal)
            
            # Final score
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 50.0  # Neutral
            
            # Combined signal
            buy_signals = signals.count("BUY")
            sell_signals = signals.count("SELL")
            
            if buy_signals > sell_signals:
                combined_signal = "BUY"
            elif sell_signals > buy_signals:
                combined_signal = "SELL"
            else:
                combined_signal = "HOLD"
            
            # Sentiment level
            sentiment_level = self._score_to_sentiment_level(final_score)
            
            # Confidence
            confidence = min(total_weight, 1.0)
            
            return SentimentData(
                source="Combined Sentiment",
                sentiment_score=final_score,
                sentiment_level=sentiment_level,
                signal=combined_signal,
                confidence=confidence,
                description=f"Combined market sentiment: {sentiment_level.value}",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Combined sentiment calculation error: {e}")
            return SentimentData(
                source="Combined Sentiment",
                sentiment_score=50.0,
                sentiment_level=SentimentLevel.NEUTRAL,
                signal="HOLD",
                confidence=0.0,
                description="Error calculating sentiment",
                timestamp=datetime.now()
            )
    
    def _calculate_trending_sentiment(self, trending_coins: List[Dict[str, Any]]) -> float:
        """Trending coins'den sentiment hesapla"""
        try:
            if not trending_coins:
                return 50.0
            
            # Coin sayısına göre sentiment
            coin_count = len(trending_coins)
            
            # Daha fazla trending coin = daha pozitif sentiment
            if coin_count >= 8:
                return 70.0  # Yüksek aktivite
            elif coin_count >= 5:
                return 60.0  # Orta aktivite
            elif coin_count >= 3:
                return 55.0  # Düşük aktivite
            else:
                return 45.0  # Çok düşük aktivite
            
        except Exception as e:
            self.logger.error(f"Trending sentiment calculation error: {e}")
            return 50.0
    
    def _score_to_sentiment_level(self, score: float) -> SentimentLevel:
        """Score'u sentiment level'a çevir"""
        if score <= self.extreme_fear_threshold:
            return SentimentLevel.EXTREME_FEAR
        elif score <= self.fear_threshold:
            return SentimentLevel.FEAR
        elif score <= self.greed_threshold:
            return SentimentLevel.NEUTRAL
        elif score <= self.extreme_greed_threshold:
            return SentimentLevel.GREED
        else:
            return SentimentLevel.EXTREME_GREED
    
    def _sentiment_to_signal(self, score: float, contrarian: bool = False) -> str:
        """Sentiment'den trading signal üret"""
        if contrarian:
            # Contrarian yaklaşım (korku = al, açgözlülük = sat)
            if score <= self.extreme_fear_threshold:
                return "BUY"  # Extreme fear = buy opportunity
            elif score <= self.fear_threshold:
                return "BUY"  # Fear = buy
            elif score >= self.extreme_greed_threshold:
                return "SELL"  # Extreme greed = sell
            elif score >= self.greed_threshold:
                return "SELL"  # Greed = sell
            else:
                return "HOLD"
        else:
            # Normal yaklaşım (trend takibi)
            if score >= self.greed_threshold:
                return "BUY"  # Greed = momentum
            elif score <= self.fear_threshold:
                return "SELL"  # Fear = weakness
            else:
                return "HOLD"
    
    def _calculate_fng_confidence(self, score: float) -> float:
        """Fear & Greed Index confidence hesapla"""
        # Extreme değerlerde daha yüksek confidence
        if score <= 20 or score >= 80:
            return 0.9
        elif score <= 30 or score >= 70:
            return 0.7
        elif score <= 40 or score >= 60:
            return 0.5
        else:
            return 0.3
    
    def _determine_market_mood(self, sentiment: SentimentData) -> str:
        """Market mood belirle"""
        score = sentiment.sentiment_score
        
        if score <= 25:
            return "Panic Selling"
        elif score <= 35:
            return "High Fear"
        elif score <= 45:
            return "Cautious"
        elif score <= 55:
            return "Neutral"
        elif score <= 65:
            return "Optimistic"
        elif score <= 75:
            return "Greedy"
        else:
            return "Euphoric"
    
    def _generate_contrarian_signal(self, sentiment: SentimentData) -> str:
        """Contrarian trading signal üret"""
        return self._sentiment_to_signal(sentiment.sentiment_score, contrarian=True)
    
    def _is_cached(self, key: str) -> bool:
        """Cache kontrolü"""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]['timestamp']
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl
    
    def _cache_data(self, key: str, data: Any):
        """Veriyi cache'e kaydet"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Market genel durumu"""
        try:
            analysis = await self.analyze()
            
            if not analysis:
                return {}
            
            return {
                'fear_greed_score': analysis.fear_greed_index.sentiment_score if analysis.fear_greed_index else None,
                'fear_greed_level': analysis.fear_greed_index.sentiment_level.value if analysis.fear_greed_index else None,
                'market_mood': analysis.market_mood,
                'trending_coins_count': len(analysis.trending_coins),
                'combined_sentiment': analysis.combined_sentiment.sentiment_level.value,
                'recommended_signal': analysis.combined_sentiment.signal,
                'contrarian_signal': analysis.contrarian_signal,
                'confidence': analysis.combined_sentiment.confidence
            }
            
        except Exception as e:
            self.logger.error(f"Market overview error: {e}")
            return {}
    
    def get_sentiment_summary(self, analysis: SentimentAnalysisResult) -> str:
        """Sentiment özeti"""
        try:
            summary_parts = []
            
            # Fear & Greed
            if analysis.fear_greed_index:
                fng = analysis.fear_greed_index
                summary_parts.append(f"Fear & Greed: {fng.sentiment_score:.0f} ({fng.sentiment_level.value})")
            
            # Market mood
            summary_parts.append(f"Market Mood: {analysis.market_mood}")
            
            # Trending
            if analysis.trending_coins:
                summary_parts.append(f"Trending Coins: {len(analysis.trending_coins)}")
            
            # Combined signal
            combined = analysis.combined_sentiment
            summary_parts.append(f"Signal: {combined.signal} (Confidence: {combined.confidence:.1f})")
            
            # Contrarian
            summary_parts.append(f"Contrarian: {analysis.contrarian_signal}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Sentiment summary error: {e}")
            return "Error generating sentiment summary"
    
    async def cleanup(self):
        """Kaynakları temizle"""
        try:
            self.cache.clear()
            self.logger.info("Sentiment analysis cleanup completed")
        except Exception as e:
            self.logger.error(f"Sentiment cleanup error: {e}")
