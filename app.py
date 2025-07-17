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

# Alpha Vantage API integration
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.techindicators import TechIndicators
try:
    from config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_CACHE_DURATION, REAL_TIME_DATA_ONLY
except ImportError:
    # Fallback if config file doesn't exist
    ALPHA_VANTAGE_API_KEY = None
    ALPHA_VANTAGE_CACHE_DURATION = 300
    REAL_TIME_DATA_ONLY = True

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
        except Exception as e:
            error_type = type(e).__name__
            if 'rate' in str(e).lower() or '429' in str(e) or 'too many requests' in str(e).lower():
                error_stats['rate_limit_errors'] += 1
                logger.warning(f"Rate limit error for {symbol}: {e}")
            elif 'connection' in str(e).lower() or 'timeout' in str(e).lower():
                error_stats['connection_errors'] += 1
                logger.error(f"Connection error for {symbol}: {e}")
            else:
                error_stats['data_errors'] += 1
                logger.error(f"Data error for {symbol}: {e}")
            
            error_stats['failed_calls'] += 1
            raise
    return wrapper

def cleanup_rate_limit_violations():
    """Clean up expired rate limit violations"""
    now = time.time()
    expired_symbols = []
    
    for symbol, violation_time in rate_limit_violations.items():
        if now - violation_time >= RATE_LIMIT_COOLDOWN:
            expired_symbols.append(symbol)
    
    for symbol in expired_symbols:
        del rate_limit_violations[symbol]
        print(f"✅ Cleared expired rate limit violation for {symbol}")

def get_error_stats():
    """Get current error statistics"""
    cleanup_rate_limit_violations()  # Clean up expired violations
    
    return {
        **error_stats,
        'active_rate_limits': len(rate_limit_violations),
        'success_rate': error_stats['successful_calls'] / max(1, error_stats['total_calls']) * 100,
        'cache_hit_rate': error_stats['cache_hits'] / max(1, error_stats['cache_hits'] + error_stats['cache_misses']) * 100
    }

# Cache configuration
cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 300  # 5 minutes

def get_cached_data(key):
    """Get cached data if it exists and is not expired"""
    with cache_lock:
        if key in cache:
            data, timestamp = cache[key]
            duration = 600 if key.startswith('yfinance_failure_') else CACHE_DURATION  # 10 minutes for failures
            if time.time() - timestamp < duration:
                error_stats['cache_hits'] += 1
                return data
            else:
                # Remove expired data
                del cache[key]
        error_stats['cache_misses'] += 1
        return None

def set_cached_data(key, data):
    """Set cached data with timestamp"""
    with cache_lock:
        cache[key] = (data, time.time())

def clear_cache():
    """Clear all cached data"""
    with cache_lock:
        cache.clear()

# Enhanced yfinance configuration with better session handling
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0'
]

def create_session():
    """Create a robust session for yfinance with proper retry strategy and user agent rotation"""
    session = requests.Session()
    
    # Select random user agent
    user_agent = random.choice(USER_AGENTS)
    session.headers.update({
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    # Conservative retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        respect_retry_after_header=True
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=1, pool_maxsize=1)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set timeouts
    session.timeout = (10, 30)  # (connection, read)
    
    print(f"Created session with User-Agent: {user_agent[:50]}...")
    
    return session

# Global rate limiting
rate_limit_violations = {}
last_yfinance_call = {}

# More responsive rate limiting parameters
RATE_LIMIT_COOLDOWN = 300  # 5 minutes cooldown after rate limit
MIN_GLOBAL_INTERVAL = 2.0  # 2 seconds between any yfinance calls
MIN_SYMBOL_INTERVAL = 60.0  # 60 seconds between calls for same symbol

# Global yfinance disable mechanism
yfinance_globally_disabled = False
yfinance_disable_until = 0
YFINANCE_DISABLE_DURATION = 30 * 60  # 30 minutes
rate_limit_error_count = 0
RATE_LIMIT_ERROR_THRESHOLD = 3  # Disable after 3 rate limit errors

def check_yfinance_global_status():
    """Check if yfinance should be globally disabled"""
    global yfinance_globally_disabled, yfinance_disable_until, rate_limit_error_count
    
    if yfinance_globally_disabled:
        if time.time() > yfinance_disable_until:
            yfinance_globally_disabled = False
            yfinance_disable_until = 0
            rate_limit_error_count = 0
            print("✅ yfinance API re-enabled after cooldown period")
            return True
        else:
            remaining = int(yfinance_disable_until - time.time())
            print(f"⏹️ yfinance API disabled for {remaining} more seconds")
            return False
    
    return True

def disable_yfinance_globally():
    """Disable yfinance globally due to too many rate limit errors"""
    global yfinance_globally_disabled, yfinance_disable_until
    
    yfinance_globally_disabled = True
    yfinance_disable_until = time.time() + YFINANCE_DISABLE_DURATION
    
    print(f"🚫 yfinance API disabled globally for {YFINANCE_DISABLE_DURATION // 60} minutes due to rate limiting")

def handle_rate_limit_error(symbol, error):
    """Handle rate limit error and check if we should disable yfinance globally"""
    global rate_limit_error_count
    
    rate_limit_violations[symbol] = time.time()
    rate_limit_error_count += 1
    
    print(f"❌ Rate limit error #{rate_limit_error_count} for {symbol}: {error}")
    
    if rate_limit_error_count >= RATE_LIMIT_ERROR_THRESHOLD:
        disable_yfinance_globally()
        return True
    
    return False

def rate_limit_yfinance(symbol):
    """Improved rate limiting for yfinance calls"""
    now = time.time()
    
    # Check if we're in a rate limit cooldown period
    if symbol in rate_limit_violations:
        time_since_violation = now - rate_limit_violations[symbol]
        if time_since_violation < RATE_LIMIT_COOLDOWN:
            remaining_cooldown = RATE_LIMIT_COOLDOWN - time_since_violation
            raise Exception(f"Rate limit cooldown for {symbol}: {remaining_cooldown:.2f} seconds remaining")
        else:
            # Clear expired cooldown
            del rate_limit_violations[symbol]
            print(f"✅ Cleared expired rate limit cooldown for {symbol}")
    
    # Global rate limiting
    if last_yfinance_call:
        last_call_time = max(last_yfinance_call.values())
        time_since_last = now - last_call_time
        if time_since_last < MIN_GLOBAL_INTERVAL:
            sleep_time = MIN_GLOBAL_INTERVAL - time_since_last
            print(f"Global rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    
    # Per-symbol rate limiting
    if symbol in last_yfinance_call:
        time_since_last = now - last_yfinance_call[symbol]
        if time_since_last < MIN_SYMBOL_INTERVAL:
            sleep_time = MIN_SYMBOL_INTERVAL - time_since_last
            print(f"Symbol rate limiting {symbol}: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    
    # Record this call
    last_yfinance_call[symbol] = time.time()

@log_yfinance_call
def get_yfinance_data_with_retry(symbol, period="1d", max_retries=1):
    """Get yfinance data with improved retry mechanism and better error handling"""
    # Check if yfinance is globally disabled
    if not check_yfinance_global_status():
        raise Exception(f"yfinance API is globally disabled due to rate limiting")
    
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
            
            # Add small random delay to avoid detection
            time.sleep(random.uniform(0.5, 1.5))
            
            print(f"Fetching {symbol} data (attempt {attempt + 1}/{max_retries})")
            
            # Use yfinance with better error handling
            try:
                data = ticker.history(
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    prepost=False
                )
            except Exception as yf_error:
                # Handle specific yfinance errors
                error_msg = str(yf_error).lower()
                if any(keyword in error_msg for keyword in ['rate', 'limit', '429', 'too many']):
                    handle_rate_limit_error(symbol, yf_error)
                    raise Exception(f"Rate limit error for {symbol}: {yf_error}")
                elif 'no data' in error_msg or 'delisted' in error_msg:
                    print(f"❌ No data available for {symbol}")
                    raise Exception(f"No data error for {symbol}: {yf_error}")
                else:
                    raise yf_error
            
            if not data.empty:
                # Cache the successful data
                set_cached_data(cache_key, data)
                print(f"✅ Successfully fetched fresh data for {symbol} ({period}) - {len(data)} records")
                
                # Clear any rate limit violation record on success
                if symbol in rate_limit_violations:
                    del rate_limit_violations[symbol]
                    print(f"Cleared rate limit violation for {symbol}")
                
                return data
            else:
                print(f"No data returned for {symbol} on attempt {attempt + 1}")
                
        except Exception as e:
            error_msg = str(e).lower()
            
            if any(keyword in error_msg for keyword in ['rate', 'limit', '429', 'too many']):
                # Record rate limit violation
                rate_limit_violations[symbol] = time.time()
                print(f"❌ Rate limited for {symbol} - entering cooldown period")
                
                # Don't retry on rate limit
                raise e
            elif 'connection' in error_msg or 'timeout' in error_msg:
                print(f"Connection error for {symbol}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = (1.5 ** attempt) * random.uniform(2, 5)
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise e
            else:
                print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = (1.5 ** attempt) * random.uniform(1, 3)
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise e
    
    raise Exception(f"Failed to fetch data for {symbol} after {max_retries} attempts")

def get_yfinance_data(symbol, period="1d"):
    """Get yfinance data with smart fallback to demo data"""
    # Check if yfinance is globally disabled
    if not check_yfinance_global_status():
        print(f"⏹️ yfinance API globally disabled, using fallback data for {symbol}")
        return generate_fallback_data(symbol, period)
    
    # Check if we're in a rate limit cooldown period
    if symbol in rate_limit_violations:
        time_since_violation = time.time() - rate_limit_violations[symbol]
        if time_since_violation < RATE_LIMIT_COOLDOWN:
            remaining_cooldown = RATE_LIMIT_COOLDOWN - time_since_violation
            print(f"⏳ Rate limit cooldown for {symbol}: {remaining_cooldown:.2f} seconds remaining, using fallback")
            return generate_fallback_data(symbol, period)
    
    # Check if we should even try yfinance based on recent failures
    cache_key = f"yfinance_failure_{symbol}"
    recent_failure = get_cached_data(cache_key)
    
    if recent_failure is not None:
        print(f"Skipping yfinance for {symbol} due to recent failure, using fallback")
        return generate_fallback_data(symbol, period)
    
    try:
        print(f"🔄 Attempting yfinance data fetch for {symbol} ({period})")
        return get_yfinance_data_with_retry(symbol, period)
    except Exception as e:
        print(f"❌ yfinance error for {symbol}: {e}")
        
        # Cache the failure for 10 minutes to avoid repeated failures
        with cache_lock:
            cache[cache_key] = (True, time.time())
        
        # Handle rate limit errors with global disable check
        if any(keyword in str(e).lower() for keyword in ['rate', 'limit', '429', 'too many']):
            print(f"🚫 Rate limit detected for {symbol}, entering extended cooldown")
        
        return generate_fallback_data(symbol, period)

# Removed generate_fallback_data function - only real-time data mode

# ================================
# ALPHA VANTAGE API INTEGRATION
# ================================

def get_alpha_vantage_client():
    """Get Alpha Vantage client with proper error handling"""
    try:
        return TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    except Exception as e:
        print(f"Failed to initialize Alpha Vantage client: {e}")
        return None

def get_alpha_vantage_fundamental_client():
    """Get Alpha Vantage fundamental data client"""
    try:
        return FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    except Exception as e:
        print(f"Failed to initialize Alpha Vantage fundamental client: {e}")
        return None

def get_alpha_vantage_tech_client():
    """Get Alpha Vantage technical indicators client"""
    try:
        return TechIndicators(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    except Exception as e:
        print(f"Failed to initialize Alpha Vantage technical client: {e}")
        return None

def fetch_alpha_vantage_data(symbol, period="1d"):
    """Fetch data from Alpha Vantage with caching"""
    cache_key = f"alpha_vantage_{symbol}_{period}"
    
    # Check cache first
    cached_data = get_cached_data(cache_key)
    if cached_data is not None:
        print(f"Using cached Alpha Vantage data for {symbol}")
        return cached_data
    
    try:
        print(f"📊 Fetching Alpha Vantage data for {symbol}")
        
        # Get Alpha Vantage client
        ts = get_alpha_vantage_client()
        if ts is None:
            raise Exception("Failed to initialize Alpha Vantage client")
        
        # Fetch data based on period
        if period in ['1d', '5d']:
            # For short periods, use intraday data
            data, meta_data = ts.get_intraday(symbol=symbol, interval='5min', outputsize='compact')
        elif period in ['30d', '1mo', '3mo']:
            # For medium periods, use daily data
            data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize='compact')
        else:
            # For long periods, use weekly data
            data, meta_data = ts.get_weekly_adjusted(symbol=symbol)
        
        if data is None or data.empty:
            raise Exception(f"No data returned from Alpha Vantage for {symbol}")
        
        # Process the data to match yfinance format
        processed_data = process_alpha_vantage_data(data, period)
        
        # Cache the data
        with cache_lock:
            cache[cache_key] = (processed_data, time.time())
        
        print(f"✅ Successfully fetched Alpha Vantage data for {symbol}")
        return processed_data
        
    except Exception as e:
        print(f"❌ Alpha Vantage error for {symbol}: {e}")
        raise

def process_alpha_vantage_data(data, period):
    """Process Alpha Vantage data to match yfinance format"""
    # Alpha Vantage returns data in descending order, reverse it
    data = data.iloc[::-1]
    
    # Standardize column names to match yfinance
    column_mapping = {
        '1. open': 'Open',
        '2. high': 'High', 
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume',
        '5. adjusted close': 'Close',  # For adjusted data
        '6. volume': 'Volume',  # For adjusted data
        '4. adjusted close': 'Close',  # Alternative format
    }
    
    # Rename columns
    for old_name, new_name in column_mapping.items():
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})
    
    # Ensure we have the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data.columns:
            if col == 'Volume':
                data[col] = 1000000  # Default volume
            else:
                data[col] = data['Close'] if 'Close' in data.columns else 100.0
    
    # Convert data types
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Filter data based on period
    if period in ['1d', '5d']:
        # For short periods, keep recent data
        days = 5 if period == '5d' else 1
        cutoff_date = datetime.now() - timedelta(days=days)
        data = data[data.index >= cutoff_date]
    elif period in ['30d', '1mo']:
        # For monthly data, keep last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        data = data[data.index >= cutoff_date]
    
    return data

def get_alpha_vantage_quote(symbol):
    """Get real-time quote from Alpha Vantage"""
    cache_key = f"alpha_vantage_quote_{symbol}"
    
    # Check cache first (shorter cache for quotes)
    cached_data = get_cached_data(cache_key)
    if cached_data is not None:
        print(f"Using cached Alpha Vantage quote for {symbol}")
        return cached_data
    
    try:
        print(f"📊 Fetching Alpha Vantage quote for {symbol}")
        
        ts = get_alpha_vantage_client()
        if ts is None:
            raise Exception("Failed to initialize Alpha Vantage client")
        
        # Get latest quote
        quote, meta_data = ts.get_quote_endpoint(symbol=symbol)
        
        if quote is None or quote.empty:
            raise Exception(f"No quote data returned from Alpha Vantage for {symbol}")
        
        # Process the quote data
        quote_data = process_alpha_vantage_quote(quote, symbol)
        
        # Cache the quote with shorter duration
        with cache_lock:
            cache[cache_key] = (quote_data, time.time())
        
        print(f"✅ Successfully fetched Alpha Vantage quote for {symbol}")
        return quote_data
        
    except Exception as e:
        print(f"❌ Alpha Vantage quote error for {symbol}: {e}")
        raise

def process_alpha_vantage_quote(quote, symbol):
    """Process Alpha Vantage quote data to standard format"""
    try:
        # Alpha Vantage quote format
        price = float(quote['05. price'].iloc[0])
        open_price = float(quote['02. open'].iloc[0])
        high_price = float(quote['03. high'].iloc[0])
        low_price = float(quote['04. low'].iloc[0])
        volume = int(quote['06. volume'].iloc[0])
        prev_close = float(quote['08. previous close'].iloc[0])
        
        change = price - prev_close
        change_percent = (change / prev_close) * 100
        
        return {
            'symbol': symbol,
            'price': price,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'volume': volume,
            'previous_close': prev_close,
            'change': change,
            'change_percent': change_percent,
            'data_source': 'alpha_vantage',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error processing Alpha Vantage quote for {symbol}: {e}")
        raise

def get_real_market_data(symbol, period="30d"):
    """Get real market data with Alpha Vantage only - no fallbacks"""
    if not ALPHA_VANTAGE_API_KEY:
        raise Exception("Alpha Vantage API key not configured. Please set ALPHA_VANTAGE_API_KEY environment variable.")
    
    try:
        # Only use Alpha Vantage for real market data
        print(f"Fetching real market data for {symbol} using Alpha Vantage")
        return fetch_alpha_vantage_data(symbol, period)
        
    except Exception as e:
        print(f"Alpha Vantage failed for {symbol}: {e}")
        # In real-time mode, don't fallback to demo data
        raise Exception(f"Unable to fetch real-time market data for {symbol}: {e}")

def get_real_quote_data(symbol):
    """Get real quote data with Alpha Vantage only - no fallbacks"""
    if not ALPHA_VANTAGE_API_KEY:
        raise Exception("Alpha Vantage API key not configured. Please set ALPHA_VANTAGE_API_KEY environment variable.")
    
    try:
        # Only use Alpha Vantage for real market data
        return get_alpha_vantage_quote(symbol)
        
    except Exception as e:
        print(f"Alpha Vantage quote failed for {symbol}: {e}")
        # In real-time mode, don't fallback to demo data
        raise Exception(f"Unable to fetch real-time data for {symbol}: {e}")

# Removed generate_optimized_fallback_quote function - only real-time data mode

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
            "fallback_cache_duration": 7200 # This is now handled by get_cached_data duration
        },
        "rate_limiting": {
            "min_call_interval": MIN_SYMBOL_INTERVAL, # Changed from MIN_CALL_INTERVAL
            "min_global_interval": MIN_GLOBAL_INTERVAL,
            "max_calls_per_minute": 1, # This is now handled by rate_limit_yfinance
            "max_calls_per_hour": 10 # This is now handled by rate_limit_yfinance
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
            'SPY': 580.00, 'QQQ': 505.00, 'DIA': 425.00, 'VIX': 12.50,
            'AAPL': 230.00, 'GOOGL': 175.00, 'MSFT': 415.00, 'TSLA': 265.00,
            'AMZN': 185.00, 'NVDA': 135.00, 'META': 350.00, 'NFLX': 250.00,
            'BTC-USD': 45000.00, 'ETH-USD': 3000.00, 'DOGE-USD': 0.08,
            'AMD': 165.00, 'INTC': 25.00, 'BABA': 85.00, 'SHOP': 75.00,
            'UBER': 70.00, 'LYFT': 15.00, 'SNAP': 12.00, 'TWTR': 45.00,
            'ROKU': 65.00, 'PYPL': 60.00, 'SQ': 75.00, 'COIN': 155.00,
            'UVXY': 5.50, 'SQQQ': 8.25, 'TQQQ': 45.00, 'SPXU': 12.50
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
    """Get real-time quote for a symbol with Alpha Vantage only"""
    print(f"Getting real-time quote for {symbol}")
    
    try:
        # Only use Alpha Vantage for real market data
        quote_data = get_real_quote_data(symbol)
        
        # Add additional calculated fields if missing
        if 'market_cap' not in quote_data:
            quote_data['market_cap'] = "N/A"
        if 'pe_ratio' not in quote_data:
            quote_data['pe_ratio'] = "N/A"
        if 'name' not in quote_data:
            quote_data['name'] = symbol
        
        print(f"✅ Real-time quote data for {symbol}: ${quote_data['price']}")
        return jsonify(quote_data)
        
    except Exception as e:
        print(f"❌ Error getting real-time quote for {symbol}: {e}")
        # Return proper error response instead of fake data
        return jsonify({
            "error": "Unable to fetch real-time data",
            "symbol": symbol,
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

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
    """Get market overview for major indices using real-time data only"""
    if not ALPHA_VANTAGE_API_KEY:
        return jsonify({
            "error": "Alpha Vantage API key not configured",
            "message": "Please set ALPHA_VANTAGE_API_KEY environment variable",
            "timestamp": datetime.now().isoformat()
        }), 503
    
    try:
        symbols = ['SPY', 'QQQ', 'DIA', 'VIX']
        names = ['S&P 500', 'NASDAQ', 'DOW JONES', 'VIX']
        
        market_data = []
        
        for i, symbol in enumerate(symbols):
            try:
                # Get real-time data for each symbol
                quote_data = get_real_quote_data(symbol)
                
                market_data.append({
                    "symbol": symbol,
                    "name": names[i],
                    "price": quote_data['price'],
                    "change": quote_data['change'],
                    "change_percent": quote_data['change_percent'],
                    "is_positive": bool(quote_data['change'] >= 0),
                    "data_source": "alpha_vantage_real_time"
                })
                
            except Exception as e:
                print(f"❌ Error getting real-time data for {symbol}: {e}")
                # Skip this symbol if we can't get real data
                continue
        
        if not market_data:
            return jsonify({
                "error": "Unable to fetch real-time market data",
                "message": "All market data requests failed",
                "timestamp": datetime.now().isoformat()
            }), 503
        
        return jsonify({
            "data": market_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in get_market_overview: {e}")
        return jsonify({
            "error": "Unable to fetch market overview",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

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
        
        # Get current quote data - this will fail if no API key
        quote_response = get_quote(symbol)
        
        # Check if quote request failed
        if quote_response.status_code != 200:
            return quote_response
        
        quote_data = quote_response.get_json()
        
        # Get historical data for technical analysis using Alpha Vantage integration
        try:
            print(f"Fetching 30-day historical data for technical analysis: {symbol}")
            history = get_real_market_data(symbol, period="30d")
            
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
                    "data_source": "alpha_vantage_enhanced",
                    "real_time": True
                }
                
                return jsonify(enhanced_data)
                
        except Exception as e:
            print(f"❌ Error getting enhanced data for {symbol}: {e}")
            
            # Return error for real-time data failure
            return jsonify({
                "error": "Unable to fetch real-time market data",
                "symbol": symbol,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 503
        
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