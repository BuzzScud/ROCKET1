# Yahoo Finance API Integration Setup

## 🚀 Overview
Your Rocket Trading Group application now uses Yahoo Finance (yfinance) to fetch real-time market data instead of Alpha Vantage. This provides more reliable data with no API key limits.

## 📋 Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

## 🔧 Quick Setup

### Option 1: Automatic Setup (Recommended)
```bash
python3 start_api.py
```

This will automatically:
- Install all required Python dependencies
- Start the Yahoo Finance API server
- Display connection information

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python3 market_api.py
```

## 🌐 API Endpoints

The Python backend provides these endpoints (running on port 5001):

- **Health Check**: `GET /health`
- **Stock Quote**: `GET /api/quote/{symbol}`
- **Chart Data**: `GET /api/chart/{symbol}?timeframe=1M`
- **Market Overview**: `GET /api/market-overview`
- **Symbol Search**: `GET /api/search/{query}`

## 📊 Supported Symbols

### Major Indices (Updated Buttons)
- **QQQ** - NASDAQ-100 ETF
- **SPY** - S&P 500 ETF
- **DIA** - Dow Jones ETF
- **VIX** - Volatility Index

### Individual Stocks
- All major US stocks (AAPL, MSFT, GOOGL, etc.)
- ETFs and mutual funds
- Cryptocurrency symbols (BTC-USD, ETH-USD)

## 🔄 How It Works

1. **Frontend**: JavaScript makes requests to `http://localhost:5001`
2. **Backend**: Python Flask server processes requests
3. **Yahoo Finance**: `yfinance` library fetches real-time data
4. **Response**: Clean JSON data returned to frontend

## 🛠️ Usage Instructions

### 1. Start the API Server
```bash
python3 start_api.py
```

Wait for the message: "🚀 Market API Server Starting..."

### 2. Open the Application
- Open `index.html` in your browser
- Navigate to the Charts section
- Click on QQQ, SPY, or DIA buttons

### 3. Test the Integration
- Charts should load with real Yahoo Finance data
- Market overview tiles should update with live prices
- Stock search should work with any valid symbol

## 🐛 Troubleshooting

### API Server Not Starting
```bash
# Check Python version
python3 --version

# Install dependencies manually
pip install flask yfinance flask-cors pandas numpy

# Run the server directly
python3 market_api.py
```

### Python 3.13 Compatibility Issues
If you're using Python 3.13, you may encounter pandas compilation errors. Try:
```bash
# Use compatible pandas version
pip install "pandas>=2.2.0" "numpy>=1.25.0"

# Or downgrade Python to 3.11/3.12
pyenv install 3.11.9
pyenv local 3.11.9
```

### Port 5000 Conflict (macOS)
The server now uses port 5001 to avoid conflicts with macOS AirPlay Receiver. If you still have issues:
- Disable AirPlay Receiver in System Preferences → General → AirDrop & Handoff
- Or use a different port by editing `market_api.py` and `index.html`

### Charts Not Loading
1. Check browser console for errors
2. Verify API server is running on port 5001
3. Test API endpoint directly: `http://localhost:5001/health`

### No Data for Symbol
- Some symbols may not be available on Yahoo Finance
- Try alternative symbols (e.g., use ETFs instead of futures)
- Check the console for specific error messages

## 📈 Benefits of Yahoo Finance

✅ **No API Key Required** - No rate limits or registration needed
✅ **Real-Time Data** - Live market prices and historical data
✅ **Comprehensive Coverage** - Stocks, ETFs, crypto, forex
✅ **Reliable Service** - Stable and fast data provider
✅ **Free to Use** - No subscription fees or costs

## 🔧 Technical Details

### Backend Architecture
- **Flask** - Web framework for API endpoints
- **yfinance** - Yahoo Finance data library
- **pandas** - Data processing and analysis
- **Flask-CORS** - Cross-origin resource sharing

### Frontend Integration
- Charts now call local API instead of Alpha Vantage
- Fallback data system for offline usage
- Real-time updates every 5 minutes

## 🚀 Next Steps

1. **Start the API server** using the instructions above
2. **Test the charts** to ensure data is loading
3. **Explore different symbols** using the search functionality
4. **Monitor the console** for any errors or issues

## 🆘 Support

If you encounter any issues:
1. Check the browser console for JavaScript errors
2. Check the Python terminal for API server errors
3. Verify all dependencies are installed correctly
4. Test the API endpoints directly using curl or browser

---

**Happy Trading! 📊💹** 