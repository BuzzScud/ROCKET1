from flask import Flask, jsonify, request
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
import random
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
rate_limit_violations = {}
last_yfinance_call = {}
RATE_LIMIT_COOLDOWN = 600  # 10 minutes cooldown after rate limit
MIN_GLOBAL_INTERVAL = 1.0  # 1 second between any yfinance calls
MIN_SYMBOL_INTERVAL = 30.0  # 30 seconds between calls for same symbol

# Enhanced user agents for better compatibility
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0'
]

def create_session():
    """Create a robust session for yfinance with proper retry strategy"""
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
    
    return session

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
    
    # Global rate limiting
    if last_yfinance_call:
        last_call_time = max(last_yfinance_call.values())
        time_since_last = now - last_call_time
        if time_since_last < MIN_GLOBAL_INTERVAL:
            sleep_time = MIN_GLOBAL_INTERVAL - time_since_last
            time.sleep(sleep_time)
    
    # Per-symbol rate limiting
    if symbol in last_yfinance_call:
        time_since_last = now - last_yfinance_call[symbol]
        if time_since_last < MIN_SYMBOL_INTERVAL:
            sleep_time = MIN_SYMBOL_INTERVAL - time_since_last
            time.sleep(sleep_time)
    
    # Record this call
    last_yfinance_call[symbol] = time.time()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Market API is running"})

@app.route('/api/quote/<symbol>', methods=['GET'])
def get_quote(symbol):
    """Get real-time quote for a symbol with improved error handling"""
    try:
        # Apply rate limiting
        rate_limit_yfinance(symbol)
        
        # Create session
        session = create_session()
        ticker = yf.Ticker(symbol, session=session)
        
        # Add small random delay
        time.sleep(random.uniform(0.5, 1.5))
        
        # Try to get current data with better error handling
        try:
            info = ticker.info
            history = ticker.history(
                period="5d",
                interval="1d",
                auto_adjust=True,
                prepost=False
            )
        except Exception as yf_error:
            error_msg = str(yf_error).lower()
            if any(keyword in error_msg for keyword in ['rate', 'limit', '429', 'too many']):
                rate_limit_violations[symbol] = time.time()
                logger.error(f"Rate limit error for {symbol}: {yf_error}")
                return jsonify({
                    "symbol": symbol,
                    "error": "Rate limit exceeded",
                    "message": "Please try again later",
                    "timestamp": datetime.now().isoformat()
                })
            elif 'no data' in error_msg or 'delisted' in error_msg:
                logger.error(f"No data available for {symbol}: {yf_error}")
                return jsonify({
                    "symbol": symbol,
                    "error": "No data available",
                    "message": "Symbol may be delisted or invalid",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                # Fallback to basic data
                logger.warning(f"Error getting full data for {symbol}, trying basic history: {yf_error}")
                history = ticker.history(period="1d")
                info = {}
        
        if history.empty:
            logger.warning(f"{symbol}: No price data found")
            return jsonify({
                "symbol": symbol,
                "price": 0.00,
                "change": 0.00,
                "change_percent": 0.00,
                "volume": 0,
                "high": 0.00,
                "low": 0.00,
                "open": 0.00,
                "previous_close": 0.00,
                "name": symbol,
                "market_cap": 0,
                "pe_ratio": 'N/A',
                "timestamp": datetime.now().isoformat(),
                "error": "No data available"
            })
        
        # Calculate price data
        current_price = history['Close'].iloc[-1]
        previous_price = history['Close'].iloc[-2] if len(history) > 1 else current_price
        change = current_price - previous_price
        change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
        
        # Build quote data with safe access to info
        quote_data = {
            "symbol": symbol,
            "price": round(float(current_price), 2),
            "change": round(float(change), 2),
            "change_percent": round(float(change_percent), 2),
            "volume": int(history['Volume'].iloc[-1]) if not pd.isna(history['Volume'].iloc[-1]) else 0,
            "high": round(float(history['High'].iloc[-1]), 2),
            "low": round(float(history['Low'].iloc[-1]), 2),
            "open": round(float(history['Open'].iloc[-1]), 2),
            "previous_close": round(float(previous_price), 2),
            "name": info.get('longName', symbol) if info else symbol,
            "market_cap": info.get('marketCap', 0) if info else 0,
            "pe_ratio": info.get('trailingPE', 'N/A') if info else 'N/A',
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(quote_data)
        
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['rate', 'limit', '429', 'too many']):
            rate_limit_violations[symbol] = time.time()
            logger.error(f"Rate limit error in get_quote for {symbol}: {e}")
            return jsonify({
                "symbol": symbol,
                "error": "Rate limit exceeded",
                "message": "Please try again later",
                "timestamp": datetime.now().isoformat()
            }), 429
        else:
            logger.error(f"Error in get_quote for {symbol}: {e}")
            return jsonify({
                "symbol": symbol,
                "error": "Internal error",
                "message": "Unable to fetch quote data",
                "timestamp": datetime.now().isoformat()
            }), 500

@app.route('/api/history/<symbol>', methods=['GET'])
def get_history(symbol):
    """Get historical data for a symbol with improved error handling"""
    try:
        # Get parameters
        period = request.args.get('period', '1y')
        interval = request.args.get('interval', '1d')
        
        # Apply rate limiting
        rate_limit_yfinance(symbol)
        
        # Create session
        session = create_session()
        ticker = yf.Ticker(symbol, session=session)
        
        # Add small random delay
        time.sleep(random.uniform(0.5, 1.5))
        
        # Get historical data with error handling
        try:
            history = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
        except Exception as yf_error:
            error_msg = str(yf_error).lower()
            if any(keyword in error_msg for keyword in ['rate', 'limit', '429', 'too many']):
                rate_limit_violations[symbol] = time.time()
                logger.error(f"Rate limit error for {symbol}: {yf_error}")
                return jsonify({
                    "symbol": symbol,
                    "error": "Rate limit exceeded",
                    "message": "Please try again later",
                    "timestamp": datetime.now().isoformat()
                }), 429
            elif 'no data' in error_msg or 'delisted' in error_msg:
                logger.error(f"No data available for {symbol}: {yf_error}")
                return jsonify({
                    "symbol": symbol,
                    "error": "No data available",
                    "message": "Symbol may be delisted or invalid",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                raise yf_error
        
        if history.empty:
            logger.warning(f"No historical data found for {symbol}")
            return jsonify({
                "symbol": symbol,
                "data": [],
                "error": "No data available",
                "timestamp": datetime.now().isoformat()
            })
        
        # Convert to JSON-serializable format
        history_data = []
        for date, row in history.iterrows():
            history_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "volume": int(row['Volume']) if not pd.isna(row['Volume']) else 0
            })
        
        return jsonify({
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data": history_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['rate', 'limit', '429', 'too many']):
            rate_limit_violations[symbol] = time.time()
            logger.error(f"Rate limit error in get_history for {symbol}: {e}")
            return jsonify({
                "symbol": symbol,
                "error": "Rate limit exceeded",
                "message": "Please try again later",
                "timestamp": datetime.now().isoformat()
            }), 429
        else:
            logger.error(f"Error in get_history for {symbol}: {e}")
            return jsonify({
                "symbol": symbol,
                "error": "Internal error",
                "message": "Unable to fetch historical data",
                "timestamp": datetime.now().isoformat()
            }), 500

@app.route('/api/info/<symbol>', methods=['GET'])
def get_info(symbol):
    """Get company information for a symbol with improved error handling"""
    try:
        # Apply rate limiting
        rate_limit_yfinance(symbol)
        
        # Create session
        session = create_session()
        ticker = yf.Ticker(symbol, session=session)
        
        # Add small random delay
        time.sleep(random.uniform(0.5, 1.5))
        
        # Get company info with error handling
        try:
            info = ticker.info
        except Exception as yf_error:
            error_msg = str(yf_error).lower()
            if any(keyword in error_msg for keyword in ['rate', 'limit', '429', 'too many']):
                rate_limit_violations[symbol] = time.time()
                logger.error(f"Rate limit error for {symbol}: {yf_error}")
                return jsonify({
                    "symbol": symbol,
                    "error": "Rate limit exceeded",
                    "message": "Please try again later",
                    "timestamp": datetime.now().isoformat()
                }), 429
            elif 'no data' in error_msg or 'delisted' in error_msg:
                logger.error(f"No data available for {symbol}: {yf_error}")
                return jsonify({
                    "symbol": symbol,
                    "error": "No data available",
                    "message": "Symbol may be delisted or invalid",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                raise yf_error
        
        if not info:
            logger.warning(f"No info found for {symbol}")
            return jsonify({
                "symbol": symbol,
                "error": "No data available",
                "timestamp": datetime.now().isoformat()
            })
        
        # Filter and format relevant info
        company_info = {
            "symbol": symbol,
            "longName": info.get('longName', symbol),
            "shortName": info.get('shortName', symbol),
            "marketCap": info.get('marketCap', 0),
            "trailingPE": info.get('trailingPE', 'N/A'),
            "forwardPE": info.get('forwardPE', 'N/A'),
            "dividendYield": info.get('dividendYield', 'N/A'),
            "beta": info.get('beta', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "country": info.get('country', 'N/A'),
            "website": info.get('website', 'N/A'),
            "longBusinessSummary": info.get('longBusinessSummary', 'N/A'),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(company_info)
        
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['rate', 'limit', '429', 'too many']):
            rate_limit_violations[symbol] = time.time()
            logger.error(f"Rate limit error in get_info for {symbol}: {e}")
            return jsonify({
                "symbol": symbol,
                "error": "Rate limit exceeded",
                "message": "Please try again later",
                "timestamp": datetime.now().isoformat()
            }), 429
        else:
            logger.error(f"Error in get_info for {symbol}: {e}")
            return jsonify({
                "symbol": symbol,
                "error": "Internal error",
                "message": "Unable to fetch company info",
                "timestamp": datetime.now().isoformat()
            }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status and rate limiting information"""
    now = time.time()
    
    # Calculate active rate limit violations
    active_violations = {}
    for symbol, violation_time in rate_limit_violations.items():
        remaining_time = RATE_LIMIT_COOLDOWN - (now - violation_time)
        if remaining_time > 0:
            active_violations[symbol] = round(remaining_time, 2)
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rate_limiting": {
            "active_violations": active_violations,
            "total_violations": len(rate_limit_violations),
            "cooldown_period": RATE_LIMIT_COOLDOWN,
            "min_global_interval": MIN_GLOBAL_INTERVAL,
            "min_symbol_interval": MIN_SYMBOL_INTERVAL
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 