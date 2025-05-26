"""
LSTM Price Prediction Model
Gelişmiş LSTM neural network ile fiyat tahmini
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from dataclasses import dataclass

@dataclass
class LSTMPrediction:
    """LSTM tahmin sonucu"""
    symbol: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_percent: float
    confidence: float
    signal: str  # BUY, SELL, HOLD
    signal_strength: float  # 0-1 arası
    prediction_horizon: int  # saat
    model_accuracy: float
    timestamp: datetime

class LSTMModel:
    """Gelişmiş LSTM fiyat tahmin modeli"""
    
    def __init__(self, symbol: str = 'BTCUSDT', sequence_length: int = 60,
                 prediction_horizon: int = 24):
        """
        LSTM model başlatıcı
        
        Args:
            symbol: Trading pair
            sequence_length: Girdi sekans uzunluğu (60 gün)
            prediction_horizon: Tahmin ufku (24 saat)
        """
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.logger = logging.getLogger(__name__)
        
        # Model ve scaler
        self.model: Optional[Sequential] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Model parametreleri
        self.lstm_units = [128, 64, 32]
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
        
        # Model performansı
        self.last_accuracy = 0.0
        self.last_mse = 0.0
        self.last_mae = 0.0
        
        # Model dosya yolları
        self.model_dir = f"models/saved/{symbol}"
        self.model_path = f"{self.model_dir}/lstm_model.h5"
        self.scaler_path = f"{self.model_dir}/scaler.pkl"
        self.feature_scaler_path = f"{self.model_dir}/feature_scaler.pkl"
        
        # Özellik kolonları
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']
        self.target_column = 'close'
        
        # Signal thresholds
        self.buy_threshold = 0.02  # %2 artış beklentisi
        self.sell_threshold = -0.02  # %2 düşüş beklentisi
        
        # Model durumu
        self.is_trained = False
        self.last_training_time = None
        self.training_data_size = 0
        
    async def initialize(self):
        """Model'i başlat"""
        try:
            # Model dizinini oluştur
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Mevcut model'i yükle
            await self._load_model()
            
            self.logger.info(f"LSTM model initialized for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"LSTM model initialization error: {e}")
            raise
    
    async def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Veriyi model için hazırla
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            # Veri kontrolü
            if len(data) < self.sequence_length + self.prediction_horizon:
                raise ValueError(f"Insufficient data. Need at least {self.sequence_length + self.prediction_horizon} records")
            
            # Teknik indikatörler ekle
            data = await self._add_technical_indicators(data)
            
            # Missing values'ları doldur
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Özellikleri normalize et
            feature_data = data[self.feature_columns].values
            feature_data_scaled = self.feature_scaler.fit_transform(feature_data)
            
            # Target'ı normalize et
            target_data = data[self.target_column].values.reshape(-1, 1)
            target_data_scaled = self.scaler.fit_transform(target_data)
            
            # Sekansları oluştur
            X, y = [], []
            
            for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
                # Girdi sekansı (son 60 gün)
                X.append(feature_data_scaled[i-self.sequence_length:i])
                
                # Hedef (24 saat sonraki fiyat)
                future_idx = min(i + self.prediction_horizon - 1, len(target_data_scaled) - 1)
                y.append(target_data_scaled[future_idx])
            
            X, y = np.array(X), np.array(y)
            
            # Train/test split (80/20)
            split_idx = int(len(X) * 0.8)
            
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            self.logger.info(f"Data prepared: Train={len(X_train)}, Test={len(X_test)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Data preparation error: {e}")
            raise
    
    async def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Teknik indikatörler ekle"""
        try:
            df = data.copy()
            
            # Moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volatility
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Güncellenen feature columns
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'volume_sma', 'volume_ratio',
                'price_change', 'high_low_ratio', 'close_open_ratio',
                'volatility'
            ]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Technical indicators error: {e}")
            return data
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """LSTM model'ini oluştur"""
        try:
            model = Sequential()
            
            # İlk LSTM katmanı
            model.add(LSTM(
                units=self.lstm_units[0],
                return_sequences=True,
                input_shape=input_shape,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
            model.add(BatchNormalization())
            
            # İkinci LSTM katmanı
            model.add(LSTM(
                units=self.lstm_units[1],
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
            model.add(BatchNormalization())
            
            # Üçüncü LSTM katmanı
            model.add(LSTM(
                units=self.lstm_units[2],
                return_sequences=False,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
            model.add(BatchNormalization())
            
            # Dense katmanları
            model.add(Dense(units=50, activation='relu'))
            model.add(Dropout(self.dropout_rate))
            
            model.add(Dense(units=25, activation='relu'))
            model.add(Dropout(self.dropout_rate))
            
            # Çıktı katmanı
            model.add(Dense(units=1, activation='linear'))
            
            # Model'i compile et
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            self.logger.info(f"LSTM model built with {model.count_params()} parameters")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Model building error: {e}")
            raise
    
    async def train(self, data: pd.DataFrame, retrain: bool = False) -> Dict[str, Any]:
        """
        Model'i eğit
        
        Args:
            data: Eğitim verisi
            retrain: Yeniden eğitim yapılsın mı
            
        Returns:
            Dict: Eğitim sonuçları
        """
        try:
            # Mevcut model varsa ve retrain False ise eğitim yapma
            if self.is_trained and not retrain:
                self.logger.info("Model already trained. Use retrain=True to force retraining.")
                return {'status': 'skipped', 'reason': 'already_trained'}
            
            self.logger.info(f"Starting LSTM training for {self.symbol}")
            
            # Veriyi hazırla
            X_train, X_test, y_train, y_test = await self.prepare_data(data)
            
            # Model'i oluştur
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self._build_model(input_shape)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Model'i eğit
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            # Model performansını değerlendir
            train_predictions = self.model.predict(X_train)
            test_predictions = self.model.predict(X_test)
            
            # Denormalize et
            train_predictions = self.scaler.inverse_transform(train_predictions)
            test_predictions = self.scaler.inverse_transform(test_predictions)
            y_train_actual = self.scaler.inverse_transform(y_train)
            y_test_actual = self.scaler.inverse_transform(y_test)
            
            # Metrikleri hesapla
            train_mse = mean_squared_error(y_train_actual, train_predictions)
            test_mse = mean_squared_error(y_test_actual, test_predictions)
            train_mae = mean_absolute_error(y_train_actual, train_predictions)
            test_mae = mean_absolute_error(y_test_actual, test_predictions)
            
            # Accuracy hesapla (direction accuracy)
            train_direction_accuracy = self._calculate_direction_accuracy(
                y_train_actual, train_predictions
            )
            test_direction_accuracy = self._calculate_direction_accuracy(
                y_test_actual, test_predictions
            )
            
            # Model durumunu güncelle
            self.last_accuracy = test_direction_accuracy
            self.last_mse = test_mse
            self.last_mae = test_mae
            self.is_trained = True
            self.last_training_time = datetime.now()
            self.training_data_size = len(data)
            
            # Model'i kaydet
            await self._save_model()
            
            results = {
                'status': 'success',
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'train_mse': float(train_mse),
                'test_mse': float(test_mse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'train_direction_accuracy': float(train_direction_accuracy),
                'test_direction_accuracy': float(test_direction_accuracy),
                'epochs_trained': len(history.history['loss']),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
            
            self.logger.info(f"LSTM training completed: {results}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"LSTM training error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_direction_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Yön tahmini doğruluğunu hesapla"""
        try:
            if len(actual) < 2 or len(predicted) < 2:
                return 0.0
            
            # Gerçek yön değişimleri
            actual_directions = np.diff(actual.flatten()) > 0
            
            # Tahmin edilen yön değişimleri
            predicted_directions = np.diff(predicted.flatten()) > 0
            
            # Doğruluk oranı
            accuracy = np.mean(actual_directions == predicted_directions)
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Direction accuracy calculation error: {e}")
            return 0.0
    
    async def predict(self, data: pd.DataFrame) -> Optional[LSTMPrediction]:
        """
        Fiyat tahmini yap
        
        Args:
            data: Son OHLCV verileri
            
        Returns:
            LSTMPrediction: Tahmin sonucu
        """
        try:
            if not self.is_trained or self.model is None:
                self.logger.warning("Model not trained. Cannot make predictions.")
                return None
            
            # Veri kontrolü
            if len(data) < self.sequence_length:
                self.logger.warning(f"Insufficient data for prediction. Need {self.sequence_length} records.")
                return None
            
            # Teknik indikatörler ekle
            data_with_indicators = await self._add_technical_indicators(data)
            
            # Son sequence'ı al
            latest_data = data_with_indicators.tail(self.sequence_length)
            
            # Missing values'ları doldur
            latest_data = latest_data.fillna(method='ffill').fillna(method='bfill')
            
            # Normalize et
            feature_data = latest_data[self.feature_columns].values
            feature_data_scaled = self.feature_scaler.transform(feature_data)
            
            # Tahmin için reshape et
            X_pred = feature_data_scaled.reshape(1, self.sequence_length, len(self.feature_columns))
            
            # Tahmin yap
            prediction_scaled = self.model.predict(X_pred, verbose=0)
            
            # Denormalize et
            predicted_price = self.scaler.inverse_transform(prediction_scaled)[0][0]
            
            # Güncel fiyat
            current_price = float(data['close'].iloc[-1])
            
            # Fiyat değişimi
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Signal üret
            signal, signal_strength = self._generate_signal(price_change_percent)
            
            # Confidence hesapla
            confidence = self._calculate_confidence(data_with_indicators)
            
            prediction = LSTMPrediction(
                symbol=self.symbol,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                confidence=confidence,
                signal=signal,
                signal_strength=signal_strength,
                prediction_horizon=self.prediction_horizon,
                model_accuracy=self.last_accuracy,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"LSTM prediction: {signal} ({signal_strength:.2f}) - "
                           f"Price: {current_price:.2f} -> {predicted_price:.2f} "
                           f"({price_change_percent:+.2f}%)")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"LSTM prediction error: {e}")
            return None
    
    def _generate_signal(self, price_change_percent: float) -> Tuple[str, float]:
        """Trading signal üret"""
        try:
            if price_change_percent >= self.buy_threshold * 100:
                signal = "BUY"
                strength = min(abs(price_change_percent) / (self.buy_threshold * 100), 1.0)
            elif price_change_percent <= self.sell_threshold * 100:
                signal = "SELL"
                strength = min(abs(price_change_percent) / (abs(self.sell_threshold) * 100), 1.0)
            else:
                signal = "HOLD"
                strength = 1.0 - (abs(price_change_percent) / (self.buy_threshold * 100))
                strength = max(0.0, min(strength, 1.0))
            
            return signal, strength
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return "HOLD", 0.0
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        """Tahmin güvenilirliğini hesapla"""
        try:
            confidence_factors = []
            
            # Model accuracy
            confidence_factors.append(self.last_accuracy)
            
            # Veri kalitesi
            data_quality = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            confidence_factors.append(data_quality)
            
            # Volatilite (düşük volatilite = yüksek güven)
            if 'volatility' in data.columns:
                volatility = data['volatility'].iloc[-1]
                volatility_confidence = max(0.0, 1.0 - (volatility * 10))  # Normalize
                confidence_factors.append(volatility_confidence)
            
            # Trend consistency
            if len(data) >= 5:
                recent_prices = data['close'].tail(5).values
                trend_consistency = self._calculate_trend_consistency(recent_prices)
                confidence_factors.append(trend_consistency)
            
            # Ortalama güven
            overall_confidence = np.mean(confidence_factors)
            
            return float(np.clip(overall_confidence, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def _calculate_trend_consistency(self, prices: np.ndarray) -> float:
        """Trend tutarlılığını hesapla"""
        try:
            if len(prices) < 3:
                return 0.5
            
            # Fiyat değişimlerinin yönü
            changes = np.diff(prices)
            directions = changes > 0
            
            # Tutarlılık oranı
            consistency = np.mean(directions == directions[0])
            
            return float(consistency)
            
        except Exception as e:
            self.logger.error(f"Trend consistency calculation error: {e}")
            return 0.5
    
    async def _save_model(self):
        """Model'i kaydet"""
        try:
            if self.model is not None:
                self.model.save(self.model_path)
                
                # Scaler'ları kaydet
                joblib.dump(self.scaler, self.scaler_path)
                joblib.dump(self.feature_scaler, self.feature_scaler_path)
                
                self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Model saving error: {e}")
    
    async def _load_model(self):
        """Model'i yükle"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                
                # Scaler'ları yükle
                if os.path.exists(self.scaler_path):
                    self.scaler = joblib.load(self.scaler_path)
                
                if os.path.exists(self.feature_scaler_path):
                    self.feature_scaler = joblib.load(self.feature_scaler_path)
                
                self.is_trained = True
                self.logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Model bilgilerini al"""
        return {
            'symbol': self.symbol,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'is_trained': self.is_trained,
            'last_accuracy': self.last_accuracy,
            'last_mse': self.last_mse,
            'last_mae': self.last_mae,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_data_size': self.training_data_size,
            'model_parameters': self.model.count_params() if self.model else 0,
            'feature_count': len(self.feature_columns)
        }
    
    async def update_thresholds(self, buy_threshold: float, sell_threshold: float):
        """Signal threshold'larını güncelle"""
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.logger.info(f"Thresholds updated: BUY={buy_threshold}, SELL={sell_threshold}")
    
    async def cleanup(self):
        """Kaynakları temizle"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            # TensorFlow session'ı temizle
            tf.keras.backend.clear_session()
            
            self.logger.info("LSTM model cleanup completed")
            
        except Exception as e:
            self.logger.error(f"LSTM cleanup error: {e}")
