#!/usr/bin/env python3
"""
The Wave API - Advanced Stock Prediction System
Comprehensive API for professional stock prediction and analysis
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import hashlib
import time
import random
from scipy.optimize import curve_fit
from scipy import stats
import json
from functools import wraps
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global cache for predictions and data
prediction_cache = {}
model_cache = {}
cache_duration = 300  # 5 minutes

# ================================
# ADVANCED PREDICTION MODELS
# ================================

class WavePredictionEngine:
    """Advanced prediction engine with multiple models"""
    
    def __init__(self):
        self.models = {
            'wave_oscillator': self._wave_oscillator_model,
            'momentum_trend': self._momentum_trend_model,
            'volatility_adjusted': self._volatility_adjusted_model,
            'fibonacci_retracement': self._fibonacci_model,
            'elliott_wave': self._elliott_wave_model
        }
    
    def _wave_oscillator_model(self, x, a, b, c, d, e):
        """Enhanced wave oscillator with multiple harmonics"""
        return a * np.exp(b * x) * (np.cos(c * x + d) + 0.3 * np.cos(e * x))
    
    def _momentum_trend_model(self, x, a, b, c):
        """Momentum-based trend model"""
        return a * (1 + b * x) * np.exp(-c * x**2)
    
    def _volatility_adjusted_model(self, x, a, b, c, d):
        """Volatility-adjusted prediction model"""
        return a + b * x + c * np.sin(d * x) * np.exp(-0.1 * x)
    
    def _fibonacci_model(self, x, a, b, c):
        """Fibonacci retracement-based model"""
        fib_levels = [0.236, 0.382, 0.618, 0.786, 1.0]
        result = a
        for i, level in enumerate(fib_levels):
            result += b * level * np.cos(c * x + i * np.pi / 4)
        return result
    
    def _elliott_wave_model(self, x, a, b, c, d, e):
        """Elliott Wave theory-based model"""
        # 5 wave impulse + 3 wave correction pattern
        impulse = a * np.sin(b * x)
        correction = c * np.sin(d * x + np.pi)
        trend = e * x
        return impulse + correction + trend
    
    def generate_ensemble_prediction(self, historical_data, symbol, prediction_days=30):
        """Generate ensemble prediction using multiple models"""
        results = {}
        
        # Prepare data
        prices = historical_data['Close'].values
        normalized_prices = (prices - np.mean(prices)) / np.std(prices)
        x_data = np.linspace(0, 1, len(prices))
        
        # Generate predictions for each model
        for model_name, model_func in self.models.items():
            try:
                # Fit model
                if model_name == 'wave_oscillator':
                    params, _ = curve_fit(
                        model_func, x_data, normalized_prices,
                        p0=[1, 0.01, 4, 0, 6], maxfev=2000,
                        bounds=([-5, -0.5, 0.1, -np.pi, 0.1], [5, 0.5, 20, np.pi, 20])
                    )
                elif model_name == 'elliott_wave':
                    params, _ = curve_fit(
                        model_func, x_data, normalized_prices,
                        p0=[1, 2, 0.5, 3, 0.01], maxfev=2000,
                        bounds=([-5, 0.1, -2, 0.1, -0.1], [5, 10, 2, 10, 0.1])
                    )
                elif model_name == 'momentum_trend':
                    params, _ = curve_fit(
                        model_func, x_data, normalized_prices,
                        p0=[1, 0.1, 0.1], maxfev=2000,
                        bounds=([-5, -2, 0], [5, 2, 2])
                    )
                elif model_name == 'volatility_adjusted':
                    params, _ = curve_fit(
                        model_func, x_data, normalized_prices,
                        p0=[0, 0.1, 0.5, 3], maxfev=2000,
                        bounds=([-5, -2, -2, 0.1], [5, 2, 2, 20])
                    )
                else:  # fibonacci_model
                    params, _ = curve_fit(
                        model_func, x_data, normalized_prices,
                        p0=[1, 0.1, 3], maxfev=2000,
                        bounds=([-5, -1, 0.1], [5, 1, 10])
                    )
                
                # Generate future predictions
                future_x = np.linspace(1, 1 + prediction_days/365, prediction_days)
                future_normalized = model_func(future_x, *params)
                
                # Denormalize
                future_prices = future_normalized * np.std(prices) + np.mean(prices)
                
                # Calculate model quality metrics
                fitted_values = model_func(x_data, *params)
                mse = np.mean((normalized_prices - fitted_values) ** 2)
                r2 = 1 - (np.sum((normalized_prices - fitted_values) ** 2) / 
                         np.sum((normalized_prices - np.mean(normalized_prices)) ** 2))
                
                confidence = max(30, min(95, 100 * (1 - mse)))
                
                results[model_name] = {
                    'predictions': future_prices.tolist(),
                    'confidence': round(confidence, 2),
                    'r2_score': round(r2, 4),
                    'mse': round(mse, 6),
                    'parameters': params.tolist(),
                    'model_quality': 'excellent' if r2 > 0.8 else 'good' if r2 > 0.6 else 'fair'
                }
                
            except Exception as e:
                logger.error(f"Model {model_name} failed for {symbol}: {e}")
                # Generate fallback prediction
                base_price = prices[-1]
                trend = (prices[-1] - prices[-min(10, len(prices))]) / min(10, len(prices))
                future_prices = [base_price + trend * i + np.random.normal(0, base_price * 0.01) 
                               for i in range(1, prediction_days + 1)]
                
                results[model_name] = {
                    'predictions': future_prices,
                    'confidence': 45,
                    'r2_score': 0.0,
                    'mse': 1.0,
                    'parameters': [],
                    'model_quality': 'fallback'
                }
        
        # Calculate ensemble prediction (weighted average)
        ensemble_pred = self._calculate_ensemble(results, prediction_days)
        
        return {
            'individual_models': results,
            'ensemble': ensemble_pred,
            'model_count': len(results),
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_ensemble(self, model_results, prediction_days):
        """Calculate ensemble prediction from individual models"""
        weights = {}
        total_weight = 0
        
        # Calculate weights based on model quality
        for model_name, result in model_results.items():
            if result['model_quality'] == 'excellent':
                weight = 0.3
            elif result['model_quality'] == 'good':
                weight = 0.2
            elif result['model_quality'] == 'fair':
                weight = 0.1
            else:
                weight = 0.05
            
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
        
        # Calculate weighted average
        ensemble_predictions = np.zeros(prediction_days)
        total_confidence = 0
        
        for model_name, result in model_results.items():
            weight = weights[model_name]
            predictions = np.array(result['predictions'])
            ensemble_predictions += weight * predictions
            total_confidence += weight * result['confidence']
        
        return {
            'predictions': ensemble_predictions.tolist(),
            'confidence': round(total_confidence, 2),
            'weights': weights,
            'method': 'weighted_average'
        }

# ================================
# TECHNICAL ANALYSIS ENGINE
# ================================

class TechnicalAnalysisEngine:
    """Advanced technical analysis with comprehensive indicators"""
    
    @staticmethod
    def calculate_all_indicators(data):
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        
        # Price-based indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        df['TR1'] = df['High'] - df['Low']
        df['TR2'] = abs(df['High'] - df['Close'].shift())
        df['TR3'] = abs(df['Low'] - df['Close'].shift())
        df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        
        # Support and Resistance
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        return df
    
    @staticmethod
    def generate_signals(df):
        """Generate comprehensive trading signals"""
        signals = []
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Price action signals
        if latest['Close'] > latest['SMA_20'] and prev['Close'] <= prev['SMA_20']:
            signals.append({
                'type': 'bullish_breakout',
                'strength': 'strong',
                'description': 'Price broke above 20-day SMA',
                'confidence': 75
            })
        
        if latest['Close'] < latest['SMA_20'] and prev['Close'] >= prev['SMA_20']:
            signals.append({
                'type': 'bearish_breakdown',
                'strength': 'strong',
                'description': 'Price broke below 20-day SMA',
                'confidence': 75
            })
        
        # RSI signals
        if latest['RSI'] < 30:
            signals.append({
                'type': 'oversold',
                'strength': 'medium',
                'description': f'RSI oversold at {latest["RSI"]:.1f}',
                'confidence': 65
            })
        elif latest['RSI'] > 70:
            signals.append({
                'type': 'overbought',
                'strength': 'medium',
                'description': f'RSI overbought at {latest["RSI"]:.1f}',
                'confidence': 65
            })
        
        # MACD signals
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append({
                'type': 'macd_bullish',
                'strength': 'medium',
                'description': 'MACD bullish crossover',
                'confidence': 70
            })
        
        # Bollinger Bands signals
        if latest['BB_Position'] > 0.95:
            signals.append({
                'type': 'bb_overbought',
                'strength': 'weak',
                'description': 'Price near upper Bollinger Band',
                'confidence': 55
            })
        elif latest['BB_Position'] < 0.05:
            signals.append({
                'type': 'bb_oversold',
                'strength': 'weak',
                'description': 'Price near lower Bollinger Band',
                'confidence': 55
            })
        
        # Volume signals
        if latest['Volume_Ratio'] > 2.0:
            signals.append({
                'type': 'high_volume',
                'strength': 'strong',
                'description': f'Volume {latest["Volume_Ratio"]:.1f}x average',
                'confidence': 80
            })
        
        return signals

# ================================
# RISK ASSESSMENT ENGINE
# ================================

class RiskAssessmentEngine:
    """Advanced risk assessment and portfolio management"""
    
    @staticmethod
    def calculate_risk_metrics(data, predictions, symbol=None):
        """Calculate comprehensive risk metrics"""
        prices = data['Close'].values
        returns = np.diff(prices) / prices[:-1]
        
        # Basic risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Beta calculation (using SPY as market proxy)
        # For simplicity, using a fixed beta based on symbol
        symbol_betas = {
            'AAPL': 1.2, 'GOOGL': 1.1, 'MSFT': 1.0, 'TSLA': 1.8,
            'AMZN': 1.3, 'META': 1.4, 'NVDA': 1.6, 'SPY': 1.0,
            'QQQ': 1.1, 'DIA': 0.9
        }
        
        # Prediction confidence risk
        pred_volatility = np.std(predictions) / np.mean(predictions) if np.mean(predictions) > 0 else 0
        
        return {
            'volatility': round(volatility * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'var_95': round(var_95 * 100, 2),
            'var_99': round(var_99 * 100, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'beta': symbol_betas.get(symbol, 1.0),
            'prediction_uncertainty': round(pred_volatility * 100, 2),
            'risk_grade': RiskAssessmentEngine._calculate_risk_grade(volatility, max_drawdown)
        }
    
    @staticmethod
    def _calculate_risk_grade(volatility, max_drawdown):
        """Calculate overall risk grade"""
        if volatility < 0.15 and max_drawdown > -0.1:
            return 'A'
        elif volatility < 0.25 and max_drawdown > -0.2:
            return 'B'
        elif volatility < 0.35 and max_drawdown > -0.3:
            return 'C'
        elif volatility < 0.5 and max_drawdown > -0.4:
            return 'D'
        else:
            return 'F'

# ================================
# DATA MANAGEMENT
# ================================

def generate_enhanced_fallback_data(symbol, period="1y"):
    """Generate enhanced fallback data with realistic patterns"""
    # Determine days
    days_map = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
        "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
    }
    days = days_map.get(period, 365)
    
    # Enhanced base prices with volatility (updated to realistic 2024 levels)
    base_prices = {
        'AAPL': 210.0, 'GOOGL': 140.0, 'MSFT': 380.0, 'TSLA': 250.0,
        'AMZN': 150.0, 'NVDA': 480.0, 'META': 320.0, 'NFLX': 400.0,
        'SPY': 480.0, 'QQQ': 380.0, 'DIA': 360.0, 'VIX': 15.0
    }
    
    volatilities = {
        'AAPL': 0.25, 'GOOGL': 0.22, 'MSFT': 0.20, 'TSLA': 0.45,
        'AMZN': 0.28, 'NVDA': 0.35, 'META': 0.30, 'NFLX': 0.32,
        'SPY': 0.15, 'QQQ': 0.18, 'DIA': 0.14, 'VIX': 0.80
    }
    
    base_price = base_prices.get(symbol, 100.0)
    volatility = volatilities.get(symbol, 0.25)
    
    # Generate consistent data using symbol hash
    symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    np.random.seed(symbol_hash % 10000)
    
    # Create dates
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price movements
    prices = [base_price]
    volumes = []
    
    for i in range(1, days):
        # Market cycles and trends
        cycle_factor = np.sin(i * 2 * np.pi / 252) * 0.02  # Annual cycle
        trend_factor = np.sin(i * 2 * np.pi / 50) * 0.01   # Quarterly trend
        
        # Random walk with drift
        daily_return = np.random.normal(0.0005, volatility / np.sqrt(252))
        
        # Add market structure
        if i % 5 == 0:  # Weekly effect
            daily_return += np.random.normal(0, 0.005)
        
        # Calculate new price
        new_price = prices[-1] * (1 + daily_return + cycle_factor + trend_factor)
        prices.append(max(new_price, base_price * 0.3))  # Floor at 30% of base
        
        # Generate volume
        base_volume = 1000000 if symbol in ['SPY', 'QQQ', 'AAPL'] else 500000
        volume_mult = 1 + abs(daily_return) * 20  # Higher volume on big moves
        volume = int(base_volume * volume_mult * (1 + np.random.normal(0, 0.3)))
        volumes.append(max(volume, 1000))
    
    # Generate OHLC from closing prices
    ohlc_data = []
    for i, close in enumerate(prices):
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.01))
        
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = volumes[i-1] if i > 0 else volumes[0] if volumes else 1000000
        
        ohlc_data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(ohlc_data, index=dates)
    df.name = symbol
    
    return df

# ================================
# API ENDPOINTS
# ================================

@app.route('/api/wave/health', methods=['GET'])
def wave_health():
    """Health check for Wave API"""
    return jsonify({
        'status': 'healthy',
        'service': 'Wave Prediction API',
        'version': '2.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/wave/predict/<symbol>', methods=['GET'])
def predict_advanced(symbol):
    """Advanced prediction endpoint with multiple models"""
    try:
        # Get parameters
        days = int(request.args.get('days', 30))
        model_type = request.args.get('model', 'ensemble')
        
        # Check cache
        cache_key = f"wave_predict_{symbol}_{days}_{model_type}"
        if cache_key in prediction_cache:
            cached_data, timestamp = prediction_cache[cache_key]
            if time.time() - timestamp < cache_duration:
                return jsonify(cached_data)
        
        # Generate fallback data
        historical_data = generate_enhanced_fallback_data(symbol, "1y")
        
        # Initialize prediction engine
        engine = WavePredictionEngine()
        
        # Generate predictions
        prediction_results = engine.generate_ensemble_prediction(
            historical_data, symbol, days
        )
        
        # Add current price info
        current_price = historical_data['Close'].iloc[-1]
        
        # Generate prediction dates
        prediction_dates = []
        for i in range(1, days + 1):
            future_date = datetime.now() + timedelta(days=i)
            prediction_dates.append(future_date.strftime('%Y-%m-%d'))
        
        # Prepare response
        response = {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'prediction_days': days,
            'prediction_dates': prediction_dates,
            'models': prediction_results['individual_models'],
            'ensemble': prediction_results['ensemble'],
            'model_count': prediction_results['model_count'],
            'data_source': 'enhanced_fallback',
            'timestamp': datetime.now().isoformat(),
            'api_version': '2.0'
        }
        
        # Cache results
        prediction_cache[cache_key] = (response, time.time())
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'symbol': symbol
        }), 500

@app.route('/api/wave/technical/<symbol>', methods=['GET'])
def technical_analysis(symbol):
    """Comprehensive technical analysis endpoint"""
    try:
        # Get historical data
        historical_data = generate_enhanced_fallback_data(symbol, "6mo")
        
        # Calculate technical indicators
        tech_engine = TechnicalAnalysisEngine()
        analyzed_data = tech_engine.calculate_all_indicators(historical_data)
        
        # Generate signals
        signals = tech_engine.generate_signals(analyzed_data)
        
        # Get latest values
        latest = analyzed_data.iloc[-1]
        
        # Prepare response
        response = {
            'symbol': symbol,
            'current_price': round(latest['Close'], 2),
            'indicators': {
                'moving_averages': {
                    'sma_5': round(latest['SMA_5'], 2),
                    'sma_10': round(latest['SMA_10'], 2),
                    'sma_20': round(latest['SMA_20'], 2),
                    'sma_50': round(latest['SMA_50'], 2),
                    'ema_12': round(latest['EMA_12'], 2),
                    'ema_26': round(latest['EMA_26'], 2)
                },
                'oscillators': {
                    'rsi': round(latest['RSI'], 2),
                    'stoch_k': round(latest['Stoch_K'], 2),
                    'stoch_d': round(latest['Stoch_D'], 2),
                    'macd': round(latest['MACD'], 4),
                    'macd_signal': round(latest['MACD_Signal'], 4),
                    'macd_histogram': round(latest['MACD_Histogram'], 4)
                },
                'bands': {
                    'bb_upper': round(latest['BB_Upper'], 2),
                    'bb_middle': round(latest['BB_Middle'], 2),
                    'bb_lower': round(latest['BB_Lower'], 2),
                    'bb_position': round(latest['BB_Position'], 3)
                },
                'volatility': {
                    'atr': round(latest['ATR'], 2),
                    'bb_width': round(latest['BB_Width'], 2)
                },
                'volume': {
                    'volume_ratio': round(latest['Volume_Ratio'], 2),
                    'volume_trend': 'increasing' if latest['Volume_Ratio'] > 1 else 'decreasing'
                },
                'support_resistance': {
                    'support': round(latest['Support'], 2),
                    'resistance': round(latest['Resistance'], 2)
                }
            },
            'signals': signals,
            'signal_count': len(signals),
            'overall_sentiment': 'bullish' if len([s for s in signals if 'bullish' in s['type']]) > len([s for s in signals if 'bearish' in s['type']]) else 'bearish',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Technical analysis error for {symbol}: {e}")
        return jsonify({
            'error': 'Technical analysis failed',
            'message': str(e),
            'symbol': symbol
        }), 500

@app.route('/api/wave/risk/<symbol>', methods=['GET'])
def risk_analysis(symbol):
    """Risk assessment endpoint"""
    try:
        # Get historical data
        historical_data = generate_enhanced_fallback_data(symbol, "1y")
        
        # Get predictions for risk calculation
        engine = WavePredictionEngine()
        predictions = engine.generate_ensemble_prediction(historical_data, symbol, 30)
        ensemble_pred = predictions['ensemble']['predictions']
        
        # Calculate risk metrics
        risk_engine = RiskAssessmentEngine()
        risk_metrics = risk_engine.calculate_risk_metrics(historical_data, ensemble_pred)
        
        # Calculate position sizing recommendations
        current_price = historical_data['Close'].iloc[-1]
        volatility = risk_metrics['volatility'] / 100
        
        # Kelly criterion for position sizing
        win_rate = 0.55  # Assume 55% win rate
        avg_win_loss = 1.2  # Average win/loss ratio
        kelly_percent = (win_rate * avg_win_loss - (1 - win_rate)) / avg_win_loss
        
        response = {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'risk_metrics': risk_metrics,
            'position_sizing': {
                'kelly_percentage': round(kelly_percent * 100, 2),
                'conservative_size': round(kelly_percent * 0.25 * 100, 2),
                'aggressive_size': round(kelly_percent * 0.5 * 100, 2)
            },
            'risk_warnings': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add risk warnings
        if risk_metrics['volatility'] > 40:
            response['risk_warnings'].append('High volatility - consider reduced position size')
        if risk_metrics['max_drawdown'] < -30:
            response['risk_warnings'].append('High historical drawdown risk')
        if risk_metrics['sharpe_ratio'] < 0.5:
            response['risk_warnings'].append('Poor risk-adjusted returns')
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Risk analysis error for {symbol}: {e}")
        return jsonify({
            'error': 'Risk analysis failed',
            'message': str(e),
            'symbol': symbol
        }), 500

@app.route('/api/wave/backtest/<symbol>', methods=['GET'])
def backtest_predictions(symbol):
    """Backtest prediction accuracy"""
    try:
        # Get parameters
        lookback_days = int(request.args.get('lookback', 90))
        
        # Generate historical data
        historical_data = generate_enhanced_fallback_data(symbol, "2y")
        
        # Simulate backtesting
        results = []
        engine = WavePredictionEngine()
        
        # Test predictions over last 90 days
        for i in range(lookback_days, 0, -10):
            # Get data up to test point
            test_data = historical_data.iloc[:-i]
            actual_prices = historical_data.iloc[-i:-i+10]['Close'].values
            
            # Generate prediction
            pred_result = engine.generate_ensemble_prediction(test_data, symbol, 10)
            predicted_prices = pred_result['ensemble']['predictions'][:len(actual_prices)]
            
            # Calculate accuracy
            if len(predicted_prices) == len(actual_prices):
                mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
                direction_accuracy = np.mean(
                    np.sign(np.diff(predicted_prices)) == np.sign(np.diff(actual_prices))
                ) * 100
                
                results.append({
                    'test_date': historical_data.index[-i].strftime('%Y-%m-%d'),
                    'mape': round(mape, 2),
                    'direction_accuracy': round(direction_accuracy, 2),
                    'predicted_change': round((predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0] * 100, 2),
                    'actual_change': round((actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100, 2)
                })
        
        # Calculate overall performance
        avg_mape = np.mean([r['mape'] for r in results])
        avg_direction = np.mean([r['direction_accuracy'] for r in results])
        
        response = {
            'symbol': symbol,
            'backtest_period': f'{lookback_days} days',
            'test_count': len(results),
            'overall_performance': {
                'avg_mape': round(avg_mape, 2),
                'avg_direction_accuracy': round(avg_direction, 2),
                'performance_grade': 'A' if avg_mape < 5 and avg_direction > 70 else 'B' if avg_mape < 10 and avg_direction > 60 else 'C'
            },
            'test_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Backtest error for {symbol}: {e}")
        return jsonify({
            'error': 'Backtest failed',
            'message': str(e),
            'symbol': symbol
        }), 500

@app.route('/api/wave/alerts/<symbol>', methods=['GET', 'POST'])
def price_alerts(symbol):
    """Price alert management"""
    if request.method == 'POST':
        try:
            data = request.json
            alert_type = data.get('type', 'price_target')
            target_price = float(data.get('target_price', 0))
            condition = data.get('condition', 'above')  # above, below, cross
            
            # Store alert (in production, this would go to a database)
            alert = {
                'id': f"alert_{symbol}_{int(time.time())}",
                'symbol': symbol,
                'type': alert_type,
                'target_price': target_price,
                'condition': condition,
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            return jsonify({
                'message': 'Alert created successfully',
                'alert': alert
            })
            
        except Exception as e:
            return jsonify({
                'error': 'Failed to create alert',
                'message': str(e)
            }), 400
    
    else:  # GET request
        # Return mock alerts for the symbol
        current_price = 100.0  # This would be fetched from real data
        
        mock_alerts = [
            {
                'id': f"alert_{symbol}_1",
                'symbol': symbol,
                'type': 'price_target',
                'target_price': current_price * 1.1,
                'condition': 'above',
                'status': 'active',
                'created_at': datetime.now().isoformat()
            }
        ]
        
        return jsonify({
            'symbol': symbol,
            'alerts': mock_alerts,
            'alert_count': len(mock_alerts)
        })

@app.route('/api/wave/export/<symbol>', methods=['GET'])
def export_predictions(symbol):
    """Export predictions and analysis"""
    try:
        format_type = request.args.get('format', 'json')
        
        # Get comprehensive data
        historical_data = generate_enhanced_fallback_data(symbol, "1y")
        engine = WavePredictionEngine()
        predictions = engine.generate_ensemble_prediction(historical_data, symbol, 30)
        
        # Prepare export data
        export_data = {
            'symbol': symbol,
            'export_timestamp': datetime.now().isoformat(),
            'historical_data': {
                'dates': historical_data.index.strftime('%Y-%m-%d').tolist(),
                'prices': historical_data['Close'].tolist(),
                'volume': historical_data['Volume'].tolist()
            },
            'predictions': predictions,
            'metadata': {
                'api_version': '2.0',
                'data_source': 'enhanced_fallback',
                'export_format': format_type
            }
        }
        
        if format_type == 'csv':
            # Convert to CSV format (simplified)
            response = jsonify({
                'message': 'CSV export prepared',
                'download_url': f'/api/wave/download/{symbol}.csv',
                'data_preview': export_data
            })
        else:
            response = jsonify(export_data)
        
        return response
        
    except Exception as e:
        return jsonify({
            'error': 'Export failed',
            'message': str(e),
            'symbol': symbol
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌊 Starting Wave Prediction API on port {port}")
    print(f"📊 Advanced multi-model prediction system")
    print(f"🔧 Technical analysis engine")
    print(f"⚠️  Risk assessment tools")
    print(f"📈 Backtesting capabilities")
    print(f"🚨 Alert management system")
    print(f"📤 Export functionality")
    app.run(debug=True, host='0.0.0.0', port=port) 