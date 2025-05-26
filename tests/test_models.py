"""
AI Models Tests
AI model fonksiyonlarının test edilmesi
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from tests import TestUtils, PerformanceTimer, get_memory_usage, generate_mock_price_data


class TestLSTMModel:
    """LSTM Model testleri"""
    
    @pytest.fixture
    def lstm_model(self, sample_price_data):
        """LSTM Model instance oluştur"""
        from models.lstm_model import LSTMModel
        
        # Mock config
        config = Mock()
        config.LSTM_SEQUENCE_LENGTH = 60
        config.LSTM_FEATURES = ['close', 'volume', 'high', 'low']
        config.LSTM_EPOCHS = 10
        config.LSTM_BATCH_SIZE = 32
        
        model = LSTMModel(config)
        return model
    
    def test_lstm_model_initialization(self, lstm_model):
        """LSTM model başlatma testi"""
        assert lstm_model is not None
        assert lstm_model.sequence_length == 60
        assert lstm_model.model is None  # Model henüz eğitilmemiş
        assert lstm_model.scaler is not None
    
    def test_data_preprocessing(self, lstm_model, sample_price_data):
        """Veri ön işleme testi"""
        # DataFrame'e çevir
        df = pd.DataFrame(sample_price_data)
        
        # Preprocess data
        processed_data = lstm_model.preprocess_data(df)
        
        # Doğrulama
        assert processed_data is not None
        assert isinstance(processed_data, np.ndarray)
        assert processed_data.shape[0] > 0
        assert processed_data.shape[1] == len(lstm_model.features)
    
    def test_create_sequences(self, lstm_model):
        """Sequence oluşturma testi"""
        # Mock data
        data = np.random.random((100, 4))  # 100 samples, 4 features
        
        # Create sequences
        X, y = lstm_model.create_sequences(data)
        
        # Doğrulama
        assert X.shape[0] == y.shape[0]  # Same number of samples
        assert X.shape[1] == lstm_model.sequence_length  # Correct sequence length
        assert X.shape[2] == 4  # Correct number of features
        assert len(y.shape) == 1  # y should be 1D
    
    @patch('tensorflow.keras.models.Sequential')
    def test_build_model(self, mock_sequential, lstm_model):
        """Model oluşturma testi"""
        # Mock model
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        
        # Build model
        lstm_model.build_model(input_shape=(60, 4))
        
        # Doğrulama
        assert lstm_model.model is not None
        mock_sequential.assert_called_once()
        mock_model.add.assert_called()  # Layers should be added
        mock_model.compile.assert_called_once()  # Model should be compiled
    
    @patch('tensorflow.keras.models.Sequential')
    def test_train_model(self, mock_sequential, lstm_model, sample_price_data):
        """Model eğitimi testi"""
        # Mock model
        mock_model = Mock()
        mock_model.fit.return_value = Mock(history={'loss': [0.1, 0.05, 0.02]})
        mock_sequential.return_value = mock_model
        
        # DataFrame'e çevir
        df = pd.DataFrame(sample_price_data)
        
        # Train model
        history = lstm_model.train(df)
        
        # Doğrulama
        assert history is not None
        assert 'loss' in history.history
        mock_model.fit.assert_called_once()
    
    @patch('tensorflow.keras.models.Sequential')
    def test_predict(self, mock_sequential, lstm_model, sample_price_data):
        """Tahmin testi"""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[45500.0]])
        mock_sequential.return_value = mock_model
        lstm_model.model = mock_model
        lstm_model.is_trained = True
        
        # DataFrame'e çevir
        df = pd.DataFrame(sample_price_data)
        
        # Make prediction
        prediction = lstm_model.predict(df)
        
        # Doğrulama
        assert prediction is not None
        assert isinstance(prediction, float)
        assert prediction > 0
        mock_model.predict.assert_called_once()
    
    def test_generate_signal(self, lstm_model, sample_price_data):
        """Sinyal üretimi testi"""
        # Mock prediction
        with patch.object(lstm_model, 'predict', return_value=46000.0):
            df = pd.DataFrame(sample_price_data)
            current_price = df['close'].iloc[-1]
            
            # Generate signal
            signal = lstm_model.generate_signal(df, 'BTCUSDT')
            
            # Doğrulama
            TestUtils.assert_valid_signal(signal)
            assert signal['symbol'] == 'BTCUSDT'
            assert signal['model'] == 'LSTM'
            assert 0 <= signal['confidence'] <= 1
    
    def test_model_performance_metrics(self, lstm_model):
        """Model performans metrikleri testi"""
        # Mock predictions and actual values
        y_true = np.array([45000, 45500, 46000, 45800, 46200])
        y_pred = np.array([45100, 45400, 46100, 45900, 46000])
        
        # Calculate metrics
        metrics = lstm_model.calculate_metrics(y_true, y_pred)
        
        # Doğrulama
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert all(metric >= 0 for metric in metrics.values())
    
    def test_model_save_load(self, lstm_model, tmp_path):
        """Model kaydetme/yükleme testi"""
        # Mock model
        mock_model = Mock()
        lstm_model.model = mock_model
        lstm_model.is_trained = True
        
        model_path = tmp_path / "test_lstm_model.h5"
        
        # Save model
        with patch.object(mock_model, 'save'):
            lstm_model.save_model(str(model_path))
            mock_model.save.assert_called_once()
        
        # Load model
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            lstm_model.load_model(str(model_path))
            assert lstm_model.model is not None
            assert lstm_model.is_trained == True
    
    def test_performance_prediction_speed(self, lstm_model, sample_price_data):
        """Tahmin hızı performans testi"""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[45500.0]])
        lstm_model.model = mock_model
        lstm_model.is_trained = True
        
        df = pd.DataFrame(sample_price_data)
        
        with PerformanceTimer() as timer:
            # 100 tahmin yap
            for i in range(100):
                lstm_model.predict(df)
        
        # 100 tahmin 1 saniyeden az sürmeli
        assert timer.elapsed < 1.0


class TestTechnicalAnalysis:
    """Technical Analysis testleri"""
    
    @pytest.fixture
    def technical_analysis(self):
        """TechnicalAnalysis instance oluştur"""
        from models.technical_analysis import TechnicalAnalysis
        
        config = Mock()
        config.TA_RSI_PERIOD = 14
        config.TA_MACD_FAST = 12
        config.TA_MACD_SLOW = 26
        config.TA_MACD_SIGNAL = 9
        config.TA_BB_PERIOD = 20
        config.TA_BB_STD = 2
        
        ta = TechnicalAnalysis(config)
        return ta
    
    def test_technical_analysis_initialization(self, technical_analysis):
        """Technical Analysis başlatma testi"""
        assert technical_analysis is not None
        assert technical_analysis.rsi_period == 14
        assert technical_analysis.macd_fast == 12
        assert technical_analysis.macd_slow == 26
    
    def test_calculate_rsi(self, technical_analysis, sample_price_data):
        """RSI hesaplama testi"""
        df = pd.DataFrame(sample_price_data)
        
        # Calculate RSI
        rsi = technical_analysis.calculate_rsi(df['close'])
        
        # Doğrulama
        assert rsi is not None
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(df)
        assert all(0 <= val <= 100 for val in rsi.dropna())
    
    def test_calculate_macd(self, technical_analysis, sample_price_data):
        """MACD hesaplama testi"""
        df = pd.DataFrame(sample_price_data)
        
        # Calculate MACD
        macd_line, signal_line, histogram = technical_analysis.calculate_macd(df['close'])
        
        # Doğrulama
        assert macd_line is not None
        assert signal_line is not None
        assert histogram is not None
        assert len(macd_line) == len(df)
        assert len(signal_line) == len(df)
        assert len(histogram) == len(df)
    
    def test_calculate_bollinger_bands(self, technical_analysis, sample_price_data):
        """Bollinger Bands hesaplama testi"""
        df = pd.DataFrame(sample_price_data)
        
        # Calculate Bollinger Bands
        upper, middle, lower = technical_analysis.calculate_bollinger_bands(df['close'])
        
        # Doğrulama
        assert upper is not None
        assert middle is not None
        assert lower is not None
        assert len(upper) == len(df)
        assert len(middle) == len(df)
        assert len(lower) == len(df)
        
        # Upper band should be above lower band
        valid_data = ~(upper.isna() | lower.isna())
        assert all(upper[valid_data] >= lower[valid_data])
    
    def test_calculate_moving_averages(self, technical_analysis, sample_price_data):
        """Moving averages hesaplama testi"""
        df = pd.DataFrame(sample_price_data)
        
        # Calculate moving averages
        sma_20 = technical_analysis.calculate_sma(df['close'], 20)
        ema_20 = technical_analysis.calculate_ema(df['close'], 20)
        
        # Doğrulama
        assert sma_20 is not None
        assert ema_20 is not None
        assert len(sma_20) == len(df)
        assert len(ema_20) == len(df)
        assert all(val > 0 for val in sma_20.dropna())
        assert all(val > 0 for val in ema_20.dropna())
    
    def test_calculate_stochastic(self, technical_analysis, sample_price_data):
        """Stochastic oscillator hesaplama testi"""
        df = pd.DataFrame(sample_price_data)
        
        # Calculate Stochastic
        k_percent, d_percent = technical_analysis.calculate_stochastic(
            df['high'], df['low'], df['close']
        )
        
        # Doğrulama
        assert k_percent is not None
        assert d_percent is not None
        assert len(k_percent) == len(df)
        assert len(d_percent) == len(df)
        assert all(0 <= val <= 100 for val in k_percent.dropna())
        assert all(0 <= val <= 100 for val in d_percent.dropna())
    
    def test_generate_signals(self, technical_analysis, sample_price_data):
        """Sinyal üretimi testi"""
        df = pd.DataFrame(sample_price_data)
        
        # Generate signals
        signals = technical_analysis.analyze(df, 'BTCUSDT')
        
        # Doğrulama
        assert signals is not None
        assert isinstance(signals, dict)
        assert 'rsi_signal' in signals
        assert 'macd_signal' in signals
        assert 'bb_signal' in signals
        assert 'overall_signal' in signals
        
        # Validate overall signal
        TestUtils.assert_valid_signal(signals['overall_signal'])
    
    def test_rsi_signal_generation(self, technical_analysis, sample_price_data):
        """RSI sinyal üretimi testi"""
        df = pd.DataFrame(sample_price_data)
        
        # Calculate RSI
        rsi = technical_analysis.calculate_rsi(df['close'])
        
        # Test different RSI values
        # Oversold condition (RSI < 30)
        rsi_oversold = pd.Series([25.0])
        signal_oversold = technical_analysis.generate_rsi_signal(rsi_oversold.iloc[-1])
        assert signal_oversold == 'BUY'
        
        # Overbought condition (RSI > 70)
        rsi_overbought = pd.Series([75.0])
        signal_overbought = technical_analysis.generate_rsi_signal(rsi_overbought.iloc[-1])
        assert signal_overbought == 'SELL'
        
        # Neutral condition
        rsi_neutral = pd.Series([50.0])
        signal_neutral = technical_analysis.generate_rsi_signal(rsi_neutral.iloc[-1])
        assert signal_neutral == 'HOLD'
    
    def test_macd_signal_generation(self, technical_analysis, sample_price_data):
        """MACD sinyal üretimi testi"""
        df = pd.DataFrame(sample_price_data)
        
        # Calculate MACD
        macd_line, signal_line, histogram = technical_analysis.calculate_macd(df['close'])
        
        # Generate MACD signal
        signal = technical_analysis.generate_macd_signal(
            macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
        )
        
        # Doğrulama
        assert signal in ['BUY', 'SELL', 'HOLD']
    
    def test_performance_multiple_indicators(self, technical_analysis, sample_price_data):
        """Çoklu indikatör performans testi"""
        df = pd.DataFrame(sample_price_data)
        
        with PerformanceTimer() as timer:
            # 50 analiz yap
            for i in range(50):
                technical_analysis.analyze(df, 'BTCUSDT')
        
        # 50 analiz 2 saniyeden az sürmeli
        assert timer.elapsed < 2.0


class TestSentimentAnalysis:
    """Sentiment Analysis testleri"""
    
    @pytest.fixture
    def sentiment_analysis(self):
        """SentimentAnalysis instance oluştur"""
        from models.sentiment_analysis import SentimentAnalysis
        
        config = Mock()
        config.SENTIMENT_SOURCES = ['twitter', 'reddit', 'news']
        config.SENTIMENT_UPDATE_INTERVAL = 3600
        
        sa = SentimentAnalysis(config)
        return sa
    
    def test_sentiment_analysis_initialization(self, sentiment_analysis):
        """Sentiment Analysis başlatma testi"""
        assert sentiment_analysis is not None
        assert sentiment_analysis.sources == ['twitter', 'reddit', 'news']
        assert sentiment_analysis.update_interval == 3600
    
    @patch('requests.get')
    def test_fetch_news_sentiment(self, mock_get, sentiment_analysis):
        """News sentiment alma testi"""
        # Mock news API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'articles': [
                {'title': 'Bitcoin reaches new highs', 'sentiment': 0.8},
                {'title': 'Crypto market shows positive trends', 'sentiment': 0.6},
                {'title': 'Bitcoin price drops', 'sentiment': -0.3}
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Fetch news sentiment
        sentiment = sentiment_analysis.fetch_news_sentiment('BTC')
        
        # Doğrulama
        assert sentiment is not None
        assert isinstance(sentiment, float)
        assert -1 <= sentiment <= 1
    
    def test_analyze_text_sentiment(self, sentiment_analysis):
        """Metin sentiment analizi testi"""
        # Test positive sentiment
        positive_text = "Bitcoin is going to the moon! Great investment opportunity!"
        positive_sentiment = sentiment_analysis.analyze_text_sentiment(positive_text)
        assert positive_sentiment > 0
        
        # Test negative sentiment
        negative_text = "Bitcoin is crashing! Sell everything now!"
        negative_sentiment = sentiment_analysis.analyze_text_sentiment(negative_text)
        assert negative_sentiment < 0
        
        # Test neutral sentiment
        neutral_text = "Bitcoin price is stable today."
        neutral_sentiment = sentiment_analysis.analyze_text_sentiment(neutral_text)
        assert abs(neutral_sentiment) < 0.5
    
    def test_aggregate_sentiment(self, sentiment_analysis):
        """Sentiment toplama testi"""
        # Mock sentiment data
        sentiment_data = {
            'twitter': 0.6,
            'reddit': 0.4,
            'news': 0.2
        }
        
        # Aggregate sentiment
        overall_sentiment = sentiment_analysis.aggregate_sentiment(sentiment_data)
        
        # Doğrulama
        assert overall_sentiment is not None
        assert isinstance(overall_sentiment, float)
        assert -1 <= overall_sentiment <= 1
    
    def test_generate_sentiment_signal(self, sentiment_analysis):
        """Sentiment sinyal üretimi testi"""
        # Test bullish sentiment
        bullish_sentiment = 0.7
        bullish_signal = sentiment_analysis.generate_signal(bullish_sentiment, 'BTCUSDT')
        TestUtils.assert_valid_signal(bullish_signal)
        assert bullish_signal['signal'] == 'BUY'
        
        # Test bearish sentiment
        bearish_sentiment = -0.7
        bearish_signal = sentiment_analysis.generate_signal(bearish_sentiment, 'BTCUSDT')
        TestUtils.assert_valid_signal(bearish_signal)
        assert bearish_signal['signal'] == 'SELL'
        
        # Test neutral sentiment
        neutral_sentiment = 0.1
        neutral_signal = sentiment_analysis.generate_signal(neutral_sentiment, 'BTCUSDT')
        TestUtils.assert_valid_signal(neutral_signal)
        assert neutral_signal['signal'] == 'HOLD'


class TestEnsembleModel:
    """Ensemble Model testleri"""
    
    @pytest.fixture
    def ensemble_model(self):
        """EnsembleModel instance oluştur"""
        from models.ensemble import EnsembleModel
        
        # Mock sub-models
        lstm_model = Mock()
        technical_analysis = Mock()
        sentiment_analysis = Mock()
        
        config = Mock()
        config.ENSEMBLE_WEIGHTS = {'lstm': 0.4, 'technical': 0.4, 'sentiment': 0.2}
        config.ENSEMBLE_MIN_CONFIDENCE = 0.6
        
        ensemble = EnsembleModel(
            lstm_model, technical_analysis, sentiment_analysis, config
        )
        return ensemble
    
    def test_ensemble_model_initialization(self, ensemble_model):
        """Ensemble model başlatma testi"""
        assert ensemble_model is not None
        assert ensemble_model.weights == {'lstm': 0.4, 'technical': 0.4, 'sentiment': 0.2}
        assert ensemble_model.min_confidence == 0.6
        assert sum(ensemble_model.weights.values()) == 1.0
    
    def test_combine_signals(self, ensemble_model):
        """Sinyal birleştirme testi"""
        # Mock signals from different models
        signals = {
            'lstm': {'signal': 'BUY', 'confidence': 0.8, 'price_target': 46000},
            'technical': {'signal': 'BUY', 'confidence': 0.7, 'price_target': 45800},
            'sentiment': {'signal': 'HOLD', 'confidence': 0.5, 'price_target': None}
        }
        
        # Combine signals
        combined_signal = ensemble_model.combine_signals(signals, 'BTCUSDT')
        
        # Doğrulama
        TestUtils.assert_valid_signal(combined_signal)
        assert combined_signal['symbol'] == 'BTCUSDT'
        assert combined_signal['model'] == 'ensemble'
        assert 0 <= combined_signal['confidence'] <= 1
    
    def test_weighted_voting(self, ensemble_model):
        """Ağırlıklı oylama testi"""
        # Test case 1: Majority BUY
        signals_buy = {
            'lstm': {'signal': 'BUY', 'confidence': 0.8},
            'technical': {'signal': 'BUY', 'confidence': 0.7},
            'sentiment': {'signal': 'SELL', 'confidence': 0.6}
        }
        
        result_buy = ensemble_model.weighted_voting(signals_buy)
        assert result_buy['signal'] == 'BUY'
        
        # Test case 2: Majority SELL
        signals_sell = {
            'lstm': {'signal': 'SELL', 'confidence': 0.9},
            'technical': {'signal': 'SELL', 'confidence': 0.8},
            'sentiment': {'signal': 'BUY', 'confidence': 0.5}
        }
        
        result_sell = ensemble_model.weighted_voting(signals_sell)
        assert result_sell['signal'] == 'SELL'
        
        # Test case 3: Mixed signals with low confidence
        signals_mixed = {
            'lstm': {'signal': 'BUY', 'confidence': 0.5},
            'technical': {'signal': 'SELL', 'confidence': 0.5},
            'sentiment': {'signal': 'HOLD', 'confidence': 0.4}
        }
        
        result_mixed = ensemble_model.weighted_voting(signals_mixed)
        assert result_mixed['signal'] == 'HOLD'
    
    def test_confidence_calculation(self, ensemble_model):
        """Güven seviyesi hesaplama testi"""
        # High confidence signals
        high_conf_signals = {
            'lstm': {'signal': 'BUY', 'confidence': 0.9},
            'technical': {'signal': 'BUY', 'confidence': 0.8},
            'sentiment': {'signal': 'BUY', 'confidence': 0.7}
        }
        
        high_confidence = ensemble_model.calculate_ensemble_confidence(high_conf_signals)
        assert high_confidence > 0.7
        
        # Low confidence signals
        low_conf_signals = {
            'lstm': {'signal': 'BUY', 'confidence': 0.4},
            'technical': {'signal': 'SELL', 'confidence': 0.3},
            'sentiment': {'signal': 'HOLD', 'confidence': 0.2}
        }
        
        low_confidence = ensemble_model.calculate_ensemble_confidence(low_conf_signals)
        assert low_confidence < 0.5
    
    def test_analyze_full_pipeline(self, ensemble_model, sample_price_data):
        """Tam analiz pipeline testi"""
        df = pd.DataFrame(sample_price_data)
        
        # Mock sub-model responses
        ensemble_model.lstm_model.generate_signal.return_value = {
            'signal': 'BUY', 'confidence': 0.8, 'model': 'LSTM',
            'symbol': 'BTCUSDT', 'timestamp': datetime.now()
        }
        
        ensemble_model.technical_analysis.analyze.return_value = {
            'overall_signal': {
                'signal': 'BUY', 'confidence': 0.7, 'model': 'technical',
                'symbol': 'BTCUSDT', 'timestamp': datetime.now()
            }
        }
        
        ensemble_model.sentiment_analysis.get_current_sentiment.return_value = 0.6
        ensemble_model.sentiment_analysis.generate_signal.return_value = {
            'signal': 'BUY', 'confidence': 0.6, 'model': 'sentiment',
            'symbol': 'BTCUSDT', 'timestamp': datetime.now()
        }
        
        # Analyze
        result = ensemble_model.analyze(df, 'BTCUSDT')
        
        # Doğrulama
        assert result is not None
        TestUtils.assert_valid_signal(result['final_signal'])
        assert 'individual_signals' in result
        assert 'confidence_breakdown' in result
    
    def test_performance_ensemble_analysis(self, ensemble_model, sample_price_data):
        """Ensemble analiz performans testi"""
        df = pd.DataFrame(sample_price_data)
        
        # Mock quick responses
        ensemble_model.lstm_model.generate_signal.return_value = {
            'signal': 'BUY', 'confidence': 0.8, 'model': 'LSTM',
            'symbol': 'BTCUSDT', 'timestamp': datetime.now()
        }
        
        ensemble_model.technical_analysis.analyze.return_value = {
            'overall_signal': {
                'signal': 'BUY', 'confidence': 0.7, 'model': 'technical',
                'symbol': 'BTCUSDT', 'timestamp': datetime.now()
            }
        }
        
        ensemble_model.sentiment_analysis.get_current_sentiment.return_value = 0.6
        ensemble_model.sentiment_analysis.generate_signal.return_value = {
            'signal': 'BUY', 'confidence': 0.6, 'model': 'sentiment',
            'symbol': 'BTCUSDT', 'timestamp': datetime.now()
        }
        
        with PerformanceTimer() as timer:
            # 20 ensemble analiz
            for i in range(20):
                ensemble_model.analyze(df, 'BTCUSDT')
        
        # 20 analiz 3 saniyeden az sürmeli
        assert timer.elapsed < 3.0


class TestModelIntegration:
    """Model entegrasyon testleri"""
    
    def test_model_pipeline_integration(self, sample_price_data):
        """Model pipeline entegrasyon testi"""
        from models.lstm_model import LSTMModel
        from models.technical_analysis import TechnicalAnalysis
        from models.ensemble import EnsembleModel
        
        # Mock configs
        lstm_config = Mock()
        lstm_config.LSTM_SEQUENCE_LENGTH = 60
        lstm_config.LSTM_FEATURES = ['close', 'volume', 'high', 'low']
        
        ta_config = Mock()
        ta_config.TA_RSI_PERIOD = 14
        ta_config.TA_MACD_FAST = 12
        ta_config.TA_MACD_SLOW = 26
        ta_config.TA_MACD_SIGNAL = 9
        
        ensemble_config = Mock()
        ensemble_config.ENSEMBLE_WEIGHTS = {'lstm': 0.5, 'technical': 0.5}
        ensemble_config.ENSEMBLE_MIN_CONFIDENCE = 0.6
        
        # Create models
        lstm_model = LSTMModel(lstm_config)
        technical_analysis = TechnicalAnalysis(ta_config)
        
        # Mock LSTM prediction
        with patch.object(lstm_model, 'predict', return_value=46000.0):
            with patch.object(lstm_model, 'is_trained', True):
                # Create ensemble
                ensemble_model = EnsembleModel(
                    lstm_model, technical_analysis, None, ensemble_config
                )
                
                df = pd.DataFrame(sample_price_data)
                
                # Test full pipeline
                # 1. Technical analysis
                ta_result = technical_analysis.analyze(df, 'BTCUSDT')
                assert ta_result is not None
                
                # 2. LSTM signal
                lstm_signal = lstm_model.generate_signal(df, 'BTCUSDT')
                TestUtils.assert_valid_signal(lstm_signal)
                
                # 3. Ensemble analysis (without sentiment)
                ensemble_result = ensemble_model.combine_signals({
                    'lstm': lstm_signal,
                    'technical': ta_result['overall_signal']
                }, 'BTCUSDT')
                
                TestUtils.assert_valid_signal(ensemble_result)
    
    def test_model_error_handling(self):
        """Model hata yönetimi testi"""
        from models.lstm_model import LSTMModel
        
        config = Mock()
        config.LSTM_SEQUENCE_LENGTH = 60
        config.LSTM_FEATURES = ['close', 'volume', 'high', 'low']
        
        lstm_model = LSTMModel(config)
        
        # Test with empty data
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            lstm_model.preprocess_data(empty_df)
        
        # Test prediction without training
        sample_df = pd.DataFrame(generate_mock_price_data())
        
        with pytest.raises(ValueError):
            lstm_model.predict(sample_df)  # Model not trained
    
    def test_model_memory_usage(self, sample_price_data):
        """Model memory kullanım testi"""
        from models.technical_analysis import TechnicalAnalysis
        
        config = Mock()
        config.TA_RSI_PERIOD = 14
        config.TA_MACD_FAST = 12
        config.TA_MACD_SLOW = 26
        config.TA_MACD_SIGNAL = 9
        
        ta = TechnicalAnalysis(config)
        
        initial_memory = get_memory_usage()
        
        # Process large dataset
        large_data = generate_mock_price_data(days=365)  # 1 year of data
        df = pd.DataFrame(large_data)
        
        # Run analysis multiple times
        for i in range(10):
            ta.analyze(df, 'BTCUSDT')
        
        current_memory = get_memory_usage()
        memory_increase = current_memory - initial_memory
        
        # Memory artışı 200MB'dan az ol
