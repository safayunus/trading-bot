"""
AI/ML Models Package
Gelişmiş trading modelleri ve ensemble sistemi
"""

from .lstm_model import LSTMModel, LSTMPrediction
from .technical_analysis import TechnicalAnalysisModel, TechnicalAnalysisResult, TechnicalSignal
from .sentiment_analysis import SentimentAnalysisModel, SentimentAnalysisResult, SentimentData
from .ensemble import EnsembleModel, EnsembleResult, ModelResult, SignalConfidence

__all__ = [
    # LSTM Model
    'LSTMModel',
    'LSTMPrediction',
    
    # Technical Analysis
    'TechnicalAnalysisModel',
    'TechnicalAnalysisResult',
    'TechnicalSignal',
    
    # Sentiment Analysis
    'SentimentAnalysisModel',
    'SentimentAnalysisResult',
    'SentimentData',
    
    # Ensemble Model
    'EnsembleModel',
    'EnsembleResult',
    'ModelResult',
    'SignalConfidence'
]

# Model versiyonları
MODEL_VERSIONS = {
    'lstm': '1.0.0',
    'technical_analysis': '1.0.0',
    'sentiment_analysis': '1.0.0',
    'ensemble': '1.0.0'
}

# Model konfigürasyonları
DEFAULT_MODEL_CONFIG = {
    'lstm': {
        'sequence_length': 60,
        'prediction_horizon': 24,
        'lstm_units': [128, 64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    },
    'technical_analysis': {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'sma_periods': [20, 50, 200]
    },
    'sentiment_analysis': {
        'fear_greed_weight': 0.4,
        'trending_weight': 0.3,
        'social_weight': 0.3,
        'cache_ttl': 3600
    },
    'ensemble': {
        'model_weights': {
            'lstm': 0.35,
            'technical': 0.45,
            'sentiment': 0.20
        },
        'strong_signal_threshold': 0.7,
        'weak_signal_threshold': 0.3,
        'high_consensus_threshold': 0.8,
        'medium_consensus_threshold': 0.5
    }
}

def get_model_info():
    """Model bilgilerini al"""
    return {
        'versions': MODEL_VERSIONS,
        'config': DEFAULT_MODEL_CONFIG,
        'available_models': list(MODEL_VERSIONS.keys()),
        'description': 'Advanced AI/ML trading models with ensemble system'
    }
