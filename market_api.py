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

app = Flask(__name__)
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
            print(f"{symbol}: No price data found, symbol may be delisted")
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
            # Generate fallback data
            history = pd.DataFrame()
        
        if history.empty:
            print(f"No chart data for {symbol}, generating fallback data")
            # Generate sample data for demonstration
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
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
                session = create_session()
                ticker = yf.Ticker(symbol, session=session)
                history = ticker.history(period="5d", interval="1d")
                
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
                        "is_positive": change >= 0
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
                    "is_positive": change >= 0,
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

if __name__ == '__main__':
    print("Starting Market API server...")
    print("Access the API at: http://localhost:5001")
    print("Health check: http://localhost:5001/health")
    print("Example quote: http://localhost:5001/api/quote/AAPL")
    print("Example chart: http://localhost:5001/api/chart/QQQ?timeframe=1M")
    app.run(debug=True, host='0.0.0.0', port=5001) 