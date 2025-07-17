# Real-Time Market Data Setup Guide

## 🚀 Overview

Your Flask trading app has been configured to **only use real-time market data** from Alpha Vantage API. No more demo data or fallbacks - only authentic market information!

## ⚡ Quick Setup

### 1. Get Your Free API Key
```bash
# Visit and get your free API key
https://www.alphavantage.co/support/#api-key
```

### 2. Set Your API Key
```bash
# Set environment variable
export ALPHA_VANTAGE_API_KEY=your_actual_api_key_here

# Or run our setup script
python setup_api_key.py
```

### 3. Start the App
```bash
python app.py
```

## 🔧 What Changed

### ✅ **Real-Time Data Only**
- ❌ Removed all demo/fallback data generation
- ✅ Only Alpha Vantage API for real market data
- ✅ Proper error handling when APIs are unavailable

### ✅ **Enhanced Error Handling**
- Clear error messages when API key is missing
- Proper HTTP status codes (503 for service unavailable)
- Detailed error responses instead of fake data

### ✅ **Updated Endpoints**
- `/api/quote/{symbol}` - Real-time quotes only
- `/api/realtime/{symbol}` - Real-time with technical analysis
- `/api/market-overview` - Real-time market indices

## 📊 API Limits

- **Free Tier**: 25 requests per day
- **Rate Limit**: 5 requests per minute
- **Coverage**: Stocks, ETFs, indices, crypto

## 🧪 Testing

### Without API Key
```bash
curl http://localhost:8080/api/quote/AAPL
# Returns: {"error": "Alpha Vantage API key not configured"}
```

### With API Key
```bash
curl http://localhost:8080/api/quote/AAPL
# Returns: Real-time AAPL quote data
```

## 🛠️ Troubleshooting

### "API key not configured" Error
1. Check your API key is set: `echo $ALPHA_VANTAGE_API_KEY`
2. Restart the Flask app after setting the key
3. Verify your API key is valid at alphavantage.co

### "Unable to fetch real-time data" Error
1. Check your daily API quota (25 requests/day)
2. Verify the stock symbol is valid
3. Check your internet connection

### App Shows Warning on Startup
```
⚠️  WARNING: No Alpha Vantage API key found!
```
This means you need to set your API key before starting the app.

## 🎯 Benefits

### **Real Market Data**
- ✅ Authentic stock prices
- ✅ Real-time quotes
- ✅ Accurate technical indicators
- ✅ Live market movements

### **No More Fake Data**
- ❌ No demo data generation
- ❌ No fallback simulations
- ❌ No random price movements
- ✅ Only real market information

### **Professional Grade**
- ✅ NASDAQ-licensed data provider
- ✅ Reliable API infrastructure
- ✅ Comprehensive market coverage
- ✅ Technical analysis ready

## 🔐 Security

- API key is stored as environment variable
- No hardcoded credentials in source code
- Safe for production deployment
- Easy to rotate API keys

## 🚀 Next Steps

1. **Get your API key** from Alpha Vantage
2. **Set the environment variable**
3. **Restart your Flask app**
4. **Start trading with real market data!**

---

**Need help?** Check the `ALPHA_VANTAGE_SETUP.md` file for detailed API setup instructions. 