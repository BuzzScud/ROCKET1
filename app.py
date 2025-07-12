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

# Create Flask app
app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Configure better session for yfinance
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

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

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Market API is running"})

@app.route('/api/quote/<symbol>', methods=['GET'])
def get_quote(symbol):
    """Get real-time quote for a symbol"""
    try:
        session = create_session()
        ticker = yf.Ticker(symbol, session=session)
        
        # Try to get current data first
        try:
            info = ticker.info
            history = ticker.history(period="5d", interval="1d")
        except Exception as e:
            print(f"Failed to get ticker '{symbol}' reason: {e}")
            # Fallback to basic data
            history = ticker.history(period="1d")
        
        if history.empty:
            print(f"{symbol}: No price data found, using fallback data")
            # Return fallback data based on symbol
            fallback_prices = {
                'SPY': 580.00,
                'QQQ': 505.00,
                'DIA': 425.00,
                'VIX': 12.50,
                'AAPL': 225.00,
                'GOOGL': 175.00,
                'MSFT': 415.00,
                'TSLA': 265.00,
                'AMZN': 185.00,
                'NVDA': 135.00
            }
            
            base_price = fallback_prices.get(symbol, 100.00)
            # Add some random variation
            current_price = base_price + np.random.normal(0, base_price * 0.01)
            change = np.random.normal(0, base_price * 0.005)
            change_percent = (change / base_price) * 100
            
            return jsonify({
                "symbol": symbol,
                "price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": int(np.random.normal(1000000, 200000)),
                "high": round(current_price + abs(change), 2),
                "low": round(current_price - abs(change), 2),
                "open": round(current_price - change, 2),
                "previous_close": round(current_price - change, 2),
                "name": symbol,
                "market_cap": int(np.random.normal(500000000000, 100000000000)),
                "pe_ratio": round(np.random.normal(20, 5), 2),
                "timestamp": datetime.now().isoformat(),
                "note": "Demo data - Yahoo Finance unavailable"
            })
        
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
            "name": info.get('longName', symbol) if 'info' in locals() else symbol,
            "market_cap": info.get('marketCap', 0) if 'info' in locals() else 0,
            "pe_ratio": info.get('trailingPE', 'N/A') if 'info' in locals() else 'N/A',
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(quote_data)
        
    except Exception as e:
        print(f"Error in get_quote for {symbol}: {e}")
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
            "error": str(e)
        })

@app.route('/api/chart/<symbol>', methods=['GET'])
def get_chart_data(symbol):
    """Get chart data for a symbol"""
    try:
        timeframe = request.args.get('timeframe', '1M')
        
        # Map timeframe to yfinance periods
        period_map = {
            '1D': '1d',
            '5D': '5d',
            '1M': '1mo',
            '3M': '3mo',
            '6M': '6mo',
            '1Y': '1y',
            '5Y': '5y'
        }
        
        interval_map = {
            '1D': '1h',
            '5D': '1d',
            '1M': '1d',
            '3M': '1d',
            '6M': '1d',
            '1Y': '1d',
            '5Y': '1wk'
        }
        
        period = period_map.get(timeframe, '1mo')
        interval = interval_map.get(timeframe, '1d')
        
        session = create_session()
        ticker = yf.Ticker(symbol, session=session)
        
        try:
            history = ticker.history(period=period, interval=interval)
        except Exception as e:
            print(f"Failed to get chart data for {symbol}: {e}")
            history = pd.DataFrame()
        
        if history.empty:
            print(f"No chart data for {symbol}, generating fallback data")
            # Generate sample data for demonstration
            days = {'1D': 1, '5D': 5, '1M': 30, '3M': 90, '6M': 180, '1Y': 365, '5Y': 1825}
            num_days = days.get(timeframe, 30)
            dates = pd.date_range(start=datetime.now() - timedelta(days=num_days), end=datetime.now(), freq='D')
            
            base_price = 100
            chart_data = []
            for i, date in enumerate(dates):
                price = base_price + (i * 0.5) + np.random.normal(0, 2)
                chart_data.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "price": round(price, 2),
                    "high": round(price + 2, 2),
                    "low": round(price - 2, 2),
                    "open": round(price - 1, 2),
                    "volume": int(np.random.normal(1000000, 200000))
                })
            
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": chart_data,
                "current": chart_data[-1]['price'],
                "change": 0.50,
                "changePercent": 0.50,
                "timestamp": datetime.now().isoformat(),
                "note": "Demo data - Yahoo Finance unavailable"
            }
            
            return jsonify(result)
        
        # Convert to the format expected by the frontend
        chart_data = []
        for index, row in history.iterrows():
            chart_data.append({
                "date": index.strftime('%Y-%m-%d'),
                "price": round(row['Close'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "open": round(row['Open'], 2),
                "volume": int(row['Volume']) if not pd.isna(row['Volume']) else 0
            })
        
        current_price = chart_data[-1]['price']
        first_price = chart_data[0]['price']
        change = current_price - first_price
        change_percent = (change / first_price) * 100 if first_price != 0 else 0
        
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
        return jsonify({"error": str(e), "symbol": symbol})

@app.route('/api/market-overview', methods=['GET'])
def get_market_overview():
    """Get market overview data"""
    try:
        symbols = ['SPY', 'QQQ', 'DIA', 'VIX']
        overview_data = []
        
        for symbol in symbols:
            try:
                session = create_session()
                ticker = yf.Ticker(symbol, session=session)
                history = ticker.history(period="5d", interval="1d")
                
                if history.empty:
                    # Fallback data
                    fallback_prices = {'SPY': 580.00, 'QQQ': 505.00, 'DIA': 425.00, 'VIX': 12.50}
                    base_price = fallback_prices.get(symbol, 100.00)
                    current_price = base_price + np.random.normal(0, base_price * 0.01)
                    change = np.random.normal(0, base_price * 0.005)
                    change_percent = (change / base_price) * 100
                    
                    overview_data.append({
                        "symbol": symbol,
                        "price": round(current_price, 2),
                        "change": round(change, 2),
                        "change_percent": round(change_percent, 2),
                        "volume": int(np.random.normal(50000000, 10000000)),
                        "note": "Demo data"
                    })
                else:
                    current_price = history['Close'].iloc[-1]
                    previous_price = history['Close'].iloc[-2] if len(history) > 1 else current_price
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
                    
                    overview_data.append({
                        "symbol": symbol,
                        "price": round(current_price, 2),
                        "change": round(change, 2),
                        "change_percent": round(change_percent, 2),
                        "volume": int(history['Volume'].iloc[-1]) if not pd.isna(history['Volume'].iloc[-1]) else 0
                    })
            except Exception as e:
                print(f"Error getting data for {symbol}: {e}")
                continue
        
        return jsonify({
            "data": overview_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in get_market_overview: {e}")
        return jsonify({"error": str(e)})

@app.route('/api/search/<query>', methods=['GET'])
def search_symbols(query):
    """Search for symbols - simplified for demo"""
    try:
        # Simple symbol search - in production, you'd use a proper API
        common_symbols = [
            {"symbol": "AAPL", "name": "Apple Inc.", "type": "stock"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "type": "stock"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "type": "stock"},
            {"symbol": "TSLA", "name": "Tesla, Inc.", "type": "stock"},
            {"symbol": "AMZN", "name": "Amazon.com, Inc.", "type": "stock"},
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "type": "stock"},
            {"symbol": "META", "name": "Meta Platforms, Inc.", "type": "stock"},
            {"symbol": "NFLX", "name": "Netflix, Inc.", "type": "stock"},
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "type": "etf"},
            {"symbol": "QQQ", "name": "Invesco QQQ Trust", "type": "etf"},
        ]
        
        # Filter results based on query
        results = [
            symbol for symbol in common_symbols
            if query.upper() in symbol['symbol'].upper() or query.upper() in symbol['name'].upper()
        ]
        
        return jsonify({
            "results": results[:10],  # Limit to 10 results
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in search_symbols: {e}")
        return jsonify({"error": str(e), "results": []})

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
    # For local development
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port) 