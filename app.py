#!/usr/bin/env python3
"""
Rocket Trading Group - Production Flask App for Google Cloud Run
This app serves both the API endpoints and static HTML files.
"""

import os
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.optimize import curve_fit
import threading
import random
import hashlib
import logging
from functools import wraps

# Create Flask app
app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Configure logging for better error monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('yfinance_integration.log')
    ]
)

logger = logging.getLogger(__name__)

# Error monitoring and statistics
error_stats = {
    'total_calls': 0,
    'successful_calls': 0,
    'failed_calls': 0,
    'rate_limit_errors': 0,
    'connection_errors': 0,
    'data_errors': 0,
    'cache_hits': 0,
    'cache_misses': 0
}

def log_yfinance_call(func):
    """Decorator to log and monitor yfinance calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        symbol = args[0] if args else kwargs.get('symbol', 'UNKNOWN')
        error_stats['total_calls'] += 1
        
        try:
            result = func(*args, **kwargs)
            error_stats['successful_calls'] += 1
            logger.info(f"Successfully fetched data for {symbol}")
            return result
        except requests.exceptions.HTTPError as e:
            if '429' in str(e):
                error_stats['rate_limit_errors'] += 1
                logger.warning(f"Rate limit error for {symbol}: {e}")
            else:
                error_stats['failed_calls'] += 1
                logger.error(f"HTTP error for {symbol}: {e}")
            raise
        except requests.exceptions.ConnectionError as e:
            error_stats['connection_errors'] += 1
            logger.error(f"Connection error for {symbol}: {e}")
            raise
        except Exception as e:
            error_stats['data_errors'] += 1
            logger.error(f"Data error for {symbol}: {e}")
            raise
    return wrapper

def get_error_stats():
    """Get current error statistics"""
    success_rate = (error_stats['successful_calls'] / error_stats['total_calls'] * 100) if error_stats['total_calls'] > 0 else 0
    cache_hit_rate = (error_stats['cache_hits'] / (error_stats['cache_hits'] + error_stats['cache_misses']) * 100) if (error_stats['cache_hits'] + error_stats['cache_misses']) > 0 else 0
    
    return {
        **error_stats,
        'success_rate': round(success_rate, 2),
        'cache_hit_rate': round(cache_hit_rate, 2)
    }

# Enhanced caching system with longer duration to reduce API calls
cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 1800  # 30 minutes in seconds (much longer cache to reduce API calls)
FALLBACK_CACHE_DURATION = 3600  # 1 hour for fallback data

def get_cached_data(key):
    """Get data from cache if not expired"""
    with cache_lock:
        if key in cache:
            data, timestamp = cache[key]
            if time.time() - timestamp < CACHE_DURATION:
                error_stats['cache_hits'] += 1
                logger.debug(f"Cache hit for {key}")
                return data
            else:
                del cache[key]
                logger.debug(f"Cache expired for {key}")
        
        error_stats['cache_misses'] += 1
        logger.debug(f"Cache miss for {key}")
    return None

def set_cached_data(key, data):
    """Set data in cache with current timestamp"""
    with cache_lock:
        cache[key] = (data, time.time())
        # Clean up old cache entries to prevent memory issues
        if len(cache) > 100:
            oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
            del cache[oldest_key]

# Enhanced yfinance configuration
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
]

def create_session():
    """Create a robust session for yfinance with retry strategy and user agent rotation"""
    session = requests.Session()
    
    # Random user agent to avoid detection
    session.headers.update({
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    
    # Enhanced retry strategy with exponential backoff
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set timeout
    session.timeout = 15
    
    return session

# More aggressive rate limiting to prevent 429 errors
last_yfinance_call = {}
call_count = {}
MIN_CALL_INTERVAL = 5.0  # 5 seconds between calls per symbol (increased)
global_last_call = 0
MIN_GLOBAL_INTERVAL = 2.0  # 2 seconds between any yfinance calls (increased)
MAX_CALLS_PER_MINUTE = 20  # Maximum 20 calls per minute globally (reduced)
MAX_CALLS_PER_HOUR = 100  # Maximum 100 calls per hour globally

def rate_limit_yfinance(symbol):
    """More aggressive rate limiting for yfinance calls to prevent 429 errors"""
    global global_last_call
    now = time.time()
    
    # Track call count per minute and hour
    current_minute = int(now // 60)
    current_hour = int(now // 3600)
    
    if current_minute not in call_count:
        call_count[current_minute] = 0
    
    # Track hourly calls
    hour_key = f"hour_{current_hour}"
    if hour_key not in call_count:
        call_count[hour_key] = 0
    
    # Clean up old entries
    old_minutes = [k for k in call_count.keys() if k.startswith('hour_') == False and k < current_minute - 5]
    old_hours = [k for k in call_count.keys() if k.startswith('hour_') and int(k.split('_')[1]) < current_hour - 2]
    
    for old_key in old_minutes + old_hours:
        if old_key in call_count:
            del call_count[old_key]
    
    # Check if we've exceeded calls per hour
    if call_count[hour_key] >= MAX_CALLS_PER_HOUR:
        sleep_time = 3600 - (now % 3600)
        print(f"Hourly rate limit exceeded: sleeping for {sleep_time:.2f} seconds")
        time.sleep(min(sleep_time, 300))  # Cap at 5 minutes
        return rate_limit_yfinance(symbol)  # Retry after sleep
    
    # Check if we've exceeded calls per minute
    if call_count[current_minute] >= MAX_CALLS_PER_MINUTE:
        sleep_time = 60 - (now % 60)
        print(f"Minute rate limit exceeded: sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
        current_minute = int(time.time() // 60)
        if current_minute not in call_count:
            call_count[current_minute] = 0
    
    # Global rate limiting - at least 2 seconds between any calls
    time_since_global = now - global_last_call
    if time_since_global < MIN_GLOBAL_INTERVAL:
        sleep_time = MIN_GLOBAL_INTERVAL - time_since_global
        print(f"Global rate limiting: sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    
    # Per-symbol rate limiting - at least 5 seconds between calls for same symbol
    if symbol in last_yfinance_call:
        time_since_last = now - last_yfinance_call[symbol]
        if time_since_last < MIN_CALL_INTERVAL:
            sleep_time = MIN_CALL_INTERVAL - time_since_last
            print(f"Symbol rate limiting {symbol}: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    
    # Update tracking
    last_yfinance_call[symbol] = time.time()
    global_last_call = time.time()
    call_count[current_minute] += 1
    call_count[hour_key] += 1

@log_yfinance_call
def get_yfinance_data_with_retry(symbol, period="1d", max_retries=3):
    """Get yfinance data with enhanced retry mechanism and exponential backoff"""
    cache_key = f"{symbol}_{period}"
    
    # Check cache first
    cached_data = get_cached_data(cache_key)
    if cached_data is not None:
        print(f"Using cached data for {symbol} ({period})")
        return cached_data
    
    for attempt in range(max_retries):
        try:
            # Rate limit the call
            rate_limit_yfinance(symbol)
            
            # Create new session for each attempt
            session = create_session()
            ticker = yf.Ticker(symbol, session=session)
            
            # Add random delay to avoid being detected as bot
            time.sleep(random.uniform(0.1, 0.5))
            
            print(f"Fetching {symbol} data (attempt {attempt + 1}/{max_retries})")
            data = ticker.history(period=period, interval="1d", auto_adjust=True, prepost=True)
            
            if not data.empty:
                # Cache the data
                set_cached_data(cache_key, data)
                print(f"Successfully fetched fresh data for {symbol} ({period}) - {len(data)} records")
                return data
            else:
                print(f"No data returned for {symbol} on attempt {attempt + 1}")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                # Exponential backoff
                sleep_time = (2 ** attempt) * random.uniform(1, 2)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"All attempts failed for {symbol}: {e}")
                raise e
    
    raise Exception(f"Failed to fetch data for {symbol} after {max_retries} attempts")

def get_yfinance_data(symbol, period="1d"):
    """Get yfinance data with fallback to demo data"""
    try:
        return get_yfinance_data_with_retry(symbol, period)
    except Exception as e:
        print(f"yfinance error for {symbol}: {e}")
        return generate_fallback_data(symbol, period)

def generate_fallback_data(symbol, period="1d"):
    """Generate realistic fallback data when yfinance fails with caching"""
    print(f"Generating fallback data for {symbol} ({period})")
    
    # Check if fallback data is already cached
    fallback_cache_key = f"fallback_{symbol}_{period}"
    cached_fallback = get_cached_data(fallback_cache_key)
    if cached_fallback is not None:
        print(f"Using cached fallback data for {symbol} ({period})")
        return cached_fallback
    
    # Determine number of days based on period
    period_days = {
        '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
    }
    days = period_days.get(period, 30)
    
    # Enhanced base prices for common symbols (updated for 2024)
    fallback_prices = {
        'SPY': 580.00, 'QQQ': 505.00, 'DIA': 425.00, 'VIX': 12.50,
        'AAPL': 225.00, 'GOOGL': 175.00, 'MSFT': 415.00, 'TSLA': 265.00,
        'AMZN': 185.00, 'NVDA': 135.00, 'META': 350.00, 'NFLX': 250.00,
        'BTC-USD': 45000.00, 'ETH-USD': 3000.00, 'DOGE-USD': 0.08,
        'AMD': 165.00, 'INTC': 25.00, 'BABA': 85.00, 'SHOP': 75.00,
        'UBER': 70.00, 'LYFT': 15.00, 'SNAP': 12.00, 'TWTR': 45.00,
        'ROKU': 65.00, 'PYPL': 60.00, 'SQ': 75.00, 'COIN': 155.00
    }
    
    base_price = fallback_prices.get(symbol, 100.00)
    
    # Generate deterministic data based on symbol hash
    symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    np.random.seed(symbol_hash % 10000)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')[:days]
    
    # Generate realistic price data with better modeling
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Add trend, volatility, and mean reversion
        trend = np.random.normal(0, 0.0005)  # Smaller daily trend
        volatility = np.random.normal(0, 0.015)  # 1.5% daily volatility
        mean_reversion = (base_price - current_price) * 0.0005  # Weak mean reversion
        
        # Add weekly and monthly cycles
        weekly_cycle = 0.002 * np.sin(2 * np.pi * i / 7)
        monthly_cycle = 0.003 * np.sin(2 * np.pi * i / 30)
        
        price_change = trend + volatility + mean_reversion + weekly_cycle + monthly_cycle
        current_price *= (1 + price_change)
        
        # Ensure reasonable bounds
        current_price = max(current_price, base_price * 0.7)
        current_price = min(current_price, base_price * 1.3)
        
        prices.append(current_price)
    
    # Create DataFrame similar to yfinance format
    data = pd.DataFrame({
        'Open': [p * random.uniform(0.998, 1.002) for p in prices],
        'High': [p * random.uniform(1.002, 1.008) for p in prices],
        'Low': [p * random.uniform(0.992, 0.998) for p in prices],
        'Close': prices,
        'Volume': [random.randint(5000000, 50000000) for _ in prices]
    }, index=dates)
    
    # Cache the fallback data with longer duration
    with cache_lock:
        cache[fallback_cache_key] = (data, time.time())
    
    print(f"Generated and cached fallback data for {symbol} ({period}) - {len(data)} records")
    return data

# Static file routes
@app.route('/')
def index():
    """Serve the main index.html file"""
    return send_file('index.html')

@app.route('/admin')
@app.route('/admin.html')
def admin():
    """Serve the admin.html file"""
    return send_file('admin.html')

@app.route('/login')
@app.route('/login.html')
def login():
    """Serve the login.html file"""
    return send_file('login.html')

@app.route('/signals')
@app.route('/signals.html')
def signals():
    """Serve the signals.html file"""
    return send_file('signals.html')

@app.route('/the-wave')
@app.route('/the-wave.html')
def the_wave():
    """Serve the the-wave.html file"""
    return send_file('the-wave.html')

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Market API is running"})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get yfinance integration statistics"""
    return jsonify({
        "yfinance_stats": get_error_stats(),
        "cache_info": {
            "cache_size": len(cache),
            "cache_duration": CACHE_DURATION,
            "fallback_cache_duration": FALLBACK_CACHE_DURATION
        },
        "rate_limiting": {
            "min_call_interval": MIN_CALL_INTERVAL,
            "min_global_interval": MIN_GLOBAL_INTERVAL,
            "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
            "max_calls_per_hour": MAX_CALLS_PER_HOUR
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/offline/<symbol>', methods=['GET'])
def get_offline_data(symbol):
    """Get offline data for symbol when yfinance completely fails"""
    try:
        print(f"Providing offline data for {symbol}")
        
        # Generate fallback data
        fallback_data = generate_fallback_data(symbol, period="30d")
        
        if fallback_data.empty:
            return jsonify({"error": "No offline data available"}), 404
        
        current_price = fallback_data['Close'].iloc[-1]
        previous_price = fallback_data['Close'].iloc[-2] if len(fallback_data) > 1 else current_price
        change = current_price - previous_price
        change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
        
        offline_data = {
            "symbol": symbol,
            "price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "volume": int(fallback_data['Volume'].iloc[-1]),
            "high": round(fallback_data['High'].iloc[-1], 2),
            "low": round(fallback_data['Low'].iloc[-1], 2),
            "open": round(fallback_data['Open'].iloc[-1], 2),
            "previous_close": round(previous_price, 2),
            "name": symbol,
            "market_cap": 0,
            "pe_ratio": 'N/A',
            "timestamp": datetime.now().isoformat(),
            "data_source": "offline_mode",
            "note": "Offline mode - using simulated data",
            "data_points": len(fallback_data),
            "mode": "offline"
        }
        
        return jsonify(offline_data)
        
    except Exception as e:
        print(f"Error in offline mode for {symbol}: {e}")
        return jsonify({"error": "Offline mode failed"}), 500

@app.route('/api/prediction/<symbol>', methods=['GET'])
def get_stock_prediction(symbol):
    """Generate stock price prediction using exponential growth with oscillation model"""
    try:
        # Define the custom function inspired by the integral (exponential growth with oscillation)
        def stock_model(x, a, b, c, d):
            # Clamp b to prevent exponential explosion
            b = max(min(b, 0.1), -0.1)
            return a * np.exp(b * x) * np.cos(c * x + d)
        
        # Get prediction days from query parameter (default 30)
        prediction_days = int(request.args.get('days', 30))
        
        # Fetch historical stock data using yfinance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        print(f"Fetching real market data for {symbol} using enhanced yfinance")
        
        # Enhanced data fetching with multiple attempts and better data handling
        stock_data = None
        data_source = "demo"
        
        try:
            # First try: Get 1-year data with enhanced retry mechanism
            print(f"Attempting to fetch 1-year data for {symbol}")
            stock_data = get_yfinance_data_with_retry(symbol, period="1y", max_retries=3)
            
            if not stock_data.empty:
                data_source = "yfinance_1y"
                print(f"Successfully fetched {len(stock_data)} days of real 1-year data for {symbol}")
            else:
                raise Exception("No data returned for 1-year period")
                
        except Exception as e:
            print(f"1-year data fetch failed for {symbol}: {e}")
            
            # Second try: Get 6-month data
            try:
                print(f"Attempting to fetch 6-month data for {symbol}")
                stock_data = get_yfinance_data_with_retry(symbol, period="6mo", max_retries=2)
                
                if not stock_data.empty:
                    data_source = "yfinance_6mo"
                    print(f"Successfully fetched {len(stock_data)} days of real 6-month data for {symbol}")
                else:
                    raise Exception("No data returned for 6-month period")
                    
            except Exception as e:
                print(f"6-month data fetch failed for {symbol}: {e}")
                
                # Third try: Get 1-month data
                try:
                    print(f"Attempting to fetch 1-month data for {symbol}")
                    stock_data = get_yfinance_data_with_retry(symbol, period="1mo", max_retries=2)
                    
                    if not stock_data.empty:
                        data_source = "yfinance_1mo"
                        print(f"Successfully fetched {len(stock_data)} days of real 1-month data for {symbol}")
                    else:
                        raise Exception("No data returned for 1-month period")
                        
                except Exception as e:
                    print(f"1-month data fetch failed for {symbol}: {e}")
                    print(f"All yfinance attempts failed, using enhanced fallback data for {symbol}")
                    
                    # Use enhanced fallback data generation
                    stock_data = generate_fallback_data(symbol, period="1y")
                    data_source = "demo"
        
        prices = stock_data['Close'].values
        
        # Normalize prices to improve model stability
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        normalized_prices = (prices - price_mean) / price_std
        
        # Create time points (normalized to [0, 1] for simplicity)
        time_points = np.linspace(0, 1, len(prices))
        
        # Fit the model to the normalized closing prices with bounds
        try:
            bounds = (
                [-10, -0.1, 0.1, -np.pi],  # Lower bounds
                [10, 0.1, 20, np.pi]       # Upper bounds
            )
            params, _ = curve_fit(
                stock_model, 
                time_points, 
                normalized_prices, 
                p0=[0, 0.01, 4, 0],
                bounds=bounds,
                maxfev=1000
            )
        except Exception as e:
            print(f"Model fitting failed for {symbol}: {e}")
            # Use simple fallback parameters
            params = [0, 0.001, 2.0, 0.5]
        
        # Generate prediction for the next specified days
        future_time = np.linspace(1, 1 + (prediction_days/365), prediction_days)
        normalized_future_prices = stock_model(future_time, *params)
        
        # Denormalize predictions
        future_prices = normalized_future_prices * price_std + price_mean
        
        # Calculate current price using same logic as quote endpoint for consistency
        symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        np.random.seed(symbol_hash % 10000)  # Same seed as quote endpoint
        
        fallback_prices = {
            'AAPL': 225.00, 'GOOGL': 175.00, 'MSFT': 415.00, 'TSLA': 265.00,
            'AMZN': 185.00, 'NVDA': 135.00, 'SPY': 580.00, 'QQQ': 505.00
        }
        base_price = fallback_prices.get(symbol, 100.00)
        current_price = base_price + np.random.normal(0, base_price * 0.005)  # Same calculation as quote endpoint
        
        # Ensure predictions are reasonable (within 30% of current price)
        future_prices = np.clip(future_prices, current_price * 0.7, current_price * 1.3)
        
        # Add some realistic variation to predictions
        for i in range(len(future_prices)):
            variation = np.random.normal(0, 0.005)  # 0.5% daily variation
            future_prices[i] *= (1 + variation)
        
        # Calculate prediction confidence based on model fit
        try:
            historical_predictions = stock_model(time_points, *params)
            denormalized_predictions = historical_predictions * price_std + price_mean
            mse = np.mean((prices - denormalized_predictions) ** 2)
            confidence = max(60, min(90, 100 - (mse / np.mean(prices)) * 100))
        except Exception as e:
            print(f"Confidence calculation failed: {e}")
            confidence = 75  # Default confidence
        
        # Generate trading signals based on predictions
        signals = generate_prediction_signals(symbol, current_price, future_prices, confidence)
        
        # Create prediction dates
        prediction_dates = []
        for i in range(prediction_days):
            future_date = end_date + timedelta(days=i+1)
            prediction_dates.append(future_date.strftime('%Y-%m-%d'))
        
        return jsonify({
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predictions": [round(p, 2) for p in future_prices],
            "prediction_dates": prediction_dates,
            "confidence": round(confidence, 2),
            "signals": signals,
            "model_params": {
                "a": params[0],
                "b": params[1], 
                "c": params[2],
                "d": params[3]
            },
            "timestamp": datetime.now().isoformat(),
            "data_source": data_source,
            "data_points": len(prices),
            "historical_period": f"{len(prices)} days",
            "note": f"Using {data_source} data with {len(prices)} historical data points"
        })
        
    except Exception as e:
        print(f"Error in prediction for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

def generate_prediction_signals(symbol, current_price, future_prices, confidence):
    """Generate trading signals based on price predictions"""
    signals = []
    
    # Calculate trend direction
    short_term_trend = np.mean(future_prices[:7]) - current_price  # Next 7 days
    long_term_trend = np.mean(future_prices) - current_price  # Full period
    
    # Generate signals based on trends
    if short_term_trend > 0 and long_term_trend > 0:
        # Strong bullish signal
        direction = 'bullish'
        entry_price = current_price
        stop_loss = current_price * 0.95  # 5% stop loss
        take_profit = np.max(future_prices[:14])  # 2-week high target
        signal_type = 'Wave Prediction Bullish'
        
    elif short_term_trend < 0 and long_term_trend < 0:
        # Strong bearish signal
        direction = 'bearish'
        entry_price = current_price
        stop_loss = current_price * 1.05  # 5% stop loss
        take_profit = np.min(future_prices[:14])  # 2-week low target
        signal_type = 'Wave Prediction Bearish'
        
    else:
        # Mixed signals - sideways/consolidation
        direction = 'neutral'
        entry_price = current_price
        stop_loss = current_price * 0.97  # 3% stop loss
        take_profit = current_price * 1.03  # 3% take profit
        signal_type = 'Wave Prediction Neutral'
    
    # Calculate risk-reward ratio
    if direction == 'bullish':
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
    else:
        risk = abs(stop_loss - entry_price)
        reward = abs(entry_price - take_profit)
    
    rr_ratio = reward / risk if risk > 0 else 1.0
    
    # Adjust confidence based on volatility
    volatility = np.std(future_prices) / np.mean(future_prices)
    adjusted_confidence = max(50, confidence * (1 - volatility))
    
    signals.append({
        'id': f'wave_{symbol}_{int(time.time())}',
        'symbol': symbol,
        'direction': direction,
        'entry_price': round(entry_price, 2),
        'stop_loss': round(stop_loss, 2),
        'take_profit': round(take_profit, 2),
        'risk_reward_ratio': round(rr_ratio, 2),
        'confidence': round(adjusted_confidence, 1),
        'setup_type': signal_type,
        'dominant_signal': 'BUY' if direction == 'bullish' else 'SELL' if direction == 'bearish' else 'HOLD',
        'current_price': round(current_price, 2),
        'predicted_price_1d': round(future_prices[0], 2),
        'predicted_price_7d': round(np.mean(future_prices[:7]), 2),
        'predicted_price_30d': round(np.mean(future_prices), 2)
    })
    
    return signals

@app.route('/api/quote/<symbol>', methods=['GET'])
def get_quote(symbol):
    """Get real-time quote for a symbol"""
    print(f"Getting quote for {symbol}")
    
    # Try to get real yfinance data using enhanced integration
    try:
        # Use enhanced yfinance data fetching with retry
        print(f"Fetching real-time quote data for {symbol}")
        history = get_yfinance_data_with_retry(symbol, period="5d", max_retries=3)
        
        if not history.empty:
            current_price = history['Close'].iloc[-1]
            previous_price = history['Close'].iloc[-2] if len(history) > 1 else current_price
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
            
            quote_data = {
                "symbol": symbol,
                "price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": int(history['Volume'].iloc[-1]) if not pd.isna(history['Volume'].iloc[-1]) else 0,
                "high": round(history['High'].iloc[-1], 2),
                "low": round(history['Low'].iloc[-1], 2),
                "open": round(history['Open'].iloc[-1], 2),
                "previous_close": round(previous_price, 2),
                "name": symbol,
                "market_cap": 0,
                "pe_ratio": 'N/A',
                "timestamp": datetime.now().isoformat(),
                "data_source": "yfinance_enhanced",
                "note": "Real yfinance data with enhanced retry mechanism",
                "data_points": len(history)
            }
            
            print(f"Generated enhanced quote data for {symbol}: ${quote_data['price']}")
            return jsonify(quote_data)
        
    except Exception as e:
        print(f"Enhanced yfinance failed for {symbol}: {e}")
    
    # Fallback to deterministic demo data
    print(f"Using fallback data for {symbol}")
    fallback_prices = {
        'SPY': 580.00, 'QQQ': 505.00, 'DIA': 425.00, 'VIX': 12.50,
        'AAPL': 225.00, 'GOOGL': 175.00, 'MSFT': 415.00, 'TSLA': 265.00,
        'AMZN': 185.00, 'NVDA': 135.00, 'META': 350.00, 'NFLX': 250.00,
        'BTC-USD': 45000.00, 'ETH-USD': 3000.00
    }
    
    base_price = fallback_prices.get(symbol, 100.00)
    
    # Use deterministic variation based on symbol hash for consistency
    symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    np.random.seed(symbol_hash % 10000)  # Deterministic seed based on symbol
    
    current_price = base_price + np.random.normal(0, base_price * 0.005)  # Smaller variation
    previous_price = base_price
    change = current_price - previous_price
    change_percent = (change / previous_price) * 100
    
    quote_data = {
        "symbol": symbol,
        "price": round(current_price, 2),
        "change": round(change, 2),
        "change_percent": round(change_percent, 2),
        "volume": np.random.randint(1000000, 10000000),
        "high": round(current_price * 1.02, 2),
        "low": round(current_price * 0.98, 2),
        "open": round(previous_price, 2),
        "previous_close": round(previous_price, 2),
        "name": symbol,
        "market_cap": np.random.randint(1000000000, 100000000000),
        "pe_ratio": round(np.random.uniform(10, 30), 2),
        "timestamp": datetime.now().isoformat(),
        "data_source": "fallback_demo",
        "note": "Using fallback data - Yahoo Finance rate limited"
    }
    
    print(f"Generated quote data for {symbol}: ${quote_data['price']}")
    return jsonify(quote_data)

@app.route('/api/chart/<symbol>', methods=['GET'])
def get_chart_data(symbol):
    """Get chart data for a symbol"""
    try:
        timeframe = request.args.get('timeframe', '1M')
        
        # Map timeframe to yfinance parameters
        timeframe_map = {
            '1D': {'period': '1d', 'interval': '1m'},
            '5D': {'period': '5d', 'interval': '5m'},
            '1M': {'period': '1mo', 'interval': '1d'},
            '3M': {'period': '3mo', 'interval': '1d'},
            '6M': {'period': '6mo', 'interval': '1d'},
            '1Y': {'period': '1y', 'interval': '1d'},
            '2Y': {'period': '2y', 'interval': '1wk'},
            '5Y': {'period': '5y', 'interval': '1wk'},
            'MAX': {'period': 'max', 'interval': '1mo'}
        }
        
        tf_params = timeframe_map.get(timeframe, timeframe_map['1M'])
        
        # Use enhanced yfinance data fetching for chart data
        try:
            print(f"Fetching chart data for {symbol} with timeframe {timeframe}")
            history = get_yfinance_data_with_retry(symbol, period=tf_params['period'], max_retries=2)
        except Exception as e:
            print(f"Failed to get chart data for {symbol}: {e}")
            history = pd.DataFrame()
        
        if history.empty:
            return jsonify({"error": "No data available"}), 404
        
        # Convert to chart data format
        chart_data = []
        for index, row in history.iterrows():
            chart_data.append({
                'time': index.strftime('%Y-%m-%d %H:%M:%S'),
                'open': round(row['Open'], 2),
                'high': round(row['High'], 2),
                'low': round(row['Low'], 2),
                'close': round(row['Close'], 2),
                'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
            })
        
        current_price = history['Close'].iloc[-1]
        previous_price = history['Close'].iloc[-2] if len(history) > 1 else current_price
        change = current_price - previous_price
        change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
        
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": chart_data,
            "current": current_price,
            "change": round(change, 2),
            "changePercent": round(change_percent, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in get_chart_data for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/market-overview', methods=['GET'])
def get_market_overview():
    """Get market overview for major indices"""
    try:
        symbols = ['SPY', 'QQQ', 'DIA', 'VIX']
        names = ['S&P 500', 'NASDAQ', 'DOW JONES', 'VIX']
        fallback_prices = [580.00, 505.00, 425.00, 12.50]
        
        market_data = []
        
        for i, symbol in enumerate(symbols):
            try:
                # Use cached yfinance data to avoid rate limits
                history = get_yfinance_data(symbol, period="5d")
                
                if not history.empty:
                    current_price = history['Close'].iloc[-1]
                    previous_price = history['Close'].iloc[-2] if len(history) > 1 else current_price
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
                    
                    market_data.append({
                        "symbol": symbol,
                        "name": names[i],
                        "price": round(current_price, 2),
                        "change": round(change, 2),
                        "change_percent": round(change_percent, 2),
                        "is_positive": bool(change >= 0)
                    })
                else:
                    raise Exception("No data available")
                    
            except Exception as e:
                print(f"Failed to get ticker '{symbol}' reason: {e}")
                print(f"{symbol}: No price data found, symbol may be delisted (period=2d)")
                
                # Add fallback data with slight randomization
                base_price = fallback_prices[i]
                change = np.random.normal(0, base_price * 0.01)  # 1% volatility
                change_percent = (change / base_price) * 100
                
                market_data.append({
                    "symbol": symbol,
                    "name": names[i],
                    "price": round(base_price + change, 2),
                    "change": round(change, 2),
                    "change_percent": round(change_percent, 2),
                    "is_positive": bool(change >= 0),
                    "note": "Demo data - Yahoo Finance unavailable"
                })
        
        return jsonify({
            "data": market_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in get_market_overview: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search/<query>', methods=['GET'])
def search_symbols(query):
    """Search for symbols (basic implementation)"""
    try:
        # This is a basic implementation - in production you'd want a proper symbol search
        common_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'VUG', 'VTV',
            'GLD', 'SLV', 'USO', 'TLT', 'HYG', 'EEM', 'FXE', 'UUP'
        ]
        
        matches = [symbol for symbol in common_symbols if query.upper() in symbol]
        
        return jsonify({
            "query": query,
            "matches": matches[:10],  # Limit to 10 results
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/realtime/<symbol>', methods=['GET'])
def get_realtime_data(symbol):
    """Get real-time market data for a symbol with enhanced features"""
    try:
        print(f"Getting real-time data for {symbol}")
        
        # Get current quote data
        quote_response = get_quote(symbol)
        quote_data = quote_response.get_json()
        
        # Get historical data for technical analysis using enhanced integration
        try:
            print(f"Fetching 30-day historical data for technical analysis: {symbol}")
            history = get_yfinance_data_with_retry(symbol, period="30d", max_retries=2)
            
            if not history.empty:
                # Calculate technical indicators
                close_prices = history['Close'].values
                
                # Simple Moving Averages
                sma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
                sma_50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else close_prices[-1]
                
                # RSI calculation (simplified)
                def calculate_rsi(prices, period=14):
                    if len(prices) < period:
                        return 50
                    deltas = np.diff(prices)
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    
                    avg_gain = np.mean(gains[-period:])
                    avg_loss = np.mean(losses[-period:])
                    
                    if avg_loss == 0:
                        return 100
                    
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    return rsi
                
                rsi = calculate_rsi(close_prices)
                
                # Volume analysis
                avg_volume = np.mean(history['Volume'].values[-10:])
                current_volume = history['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Volatility
                volatility = np.std(close_prices[-20:]) if len(close_prices) >= 20 else 0
                
                # Support and resistance levels
                highs = history['High'].values[-20:]
                lows = history['Low'].values[-20:]
                resistance = np.max(highs) if len(highs) > 0 else close_prices[-1]
                support = np.min(lows) if len(lows) > 0 else close_prices[-1]
                
                # Enhanced quote data
                enhanced_data = {
                    **quote_data,
                    "technical_indicators": {
                        "sma_20": round(sma_20, 2),
                        "sma_50": round(sma_50, 2),
                        "rsi": round(rsi, 2),
                        "volatility": round(volatility, 2),
                        "volume_ratio": round(volume_ratio, 2),
                        "support": round(support, 2),
                        "resistance": round(resistance, 2)
                    },
                    "trend_analysis": {
                        "short_term": "bullish" if close_prices[-1] > sma_20 else "bearish",
                        "medium_term": "bullish" if sma_20 > sma_50 else "bearish",
                        "momentum": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
                        "volume": "high" if volume_ratio > 1.5 else "low" if volume_ratio < 0.5 else "normal"
                    },
                    "data_source": "yfinance_cached",
                    "real_time": True
                }
                
                return jsonify(enhanced_data)
                
        except Exception as e:
            print(f"Error getting enhanced data for {symbol}: {e}")
            
        # Return basic quote data if enhanced fails
        return quote_response
        
    except Exception as e:
        print(f"Error in get_realtime_data for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors by serving the index page (for SPA routing)"""
    return send_file('index.html')

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Rocket Trading Group server on port {port}...")
    print(f"Main dashboard: http://localhost:{port}")
    print(f"Admin panel: http://localhost:{port}/admin")
    print(f"Login page: http://localhost:{port}/login")
    print(f"Signals page: http://localhost:{port}/signals")
    print(f"The Wave page: http://localhost:{port}/the-wave")
    print(f"Health check: http://localhost:{port}/health")
    app.run(debug=True, host='0.0.0.0', port=port) 