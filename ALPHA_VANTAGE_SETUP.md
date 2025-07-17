# Alpha Vantage API Setup

## Getting Your Free API Key

1. **Visit Alpha Vantage**: Go to [alphavantage.co](https://www.alphavantage.co/)
2. **Sign up**: Click "Get your free API key today"
3. **Fill in details**:
   - Select "Software Developer" or "Investor"
   - Enter your email address
   - Provide organization name (optional)
4. **Get your key**: You'll receive your API key immediately

## Setting Up Your API Key

### Option 1: Environment Variable (Recommended)
```bash
export ALPHA_VANTAGE_API_KEY=your_api_key_here
```

### Option 2: Update config.py
Edit the `config.py` file and replace `'demo'` with your actual API key:
```python
ALPHA_VANTAGE_API_KEY = 'your_actual_api_key_here'
```

## API Limits

- **Free Tier**: 25 requests per day
- **Rate Limit**: 5 requests per minute
- **Data Coverage**: Real-time and historical data for stocks, ETFs, crypto

## Features Available

✅ **Real-time stock quotes**
✅ **Historical price data**
✅ **Technical indicators**
✅ **Fundamental data**
✅ **Multiple asset classes**

## Testing Your Setup

1. Restart your Flask application
2. Test the quote endpoint: `curl http://localhost:8080/api/quote/AAPL`
3. Test the realtime endpoint: `curl http://localhost:8080/api/realtime/TSLA`

## Troubleshooting

If you see "demo" in the logs, your API key is not set up correctly. Make sure to:
1. Check your API key is valid
2. Restart the Flask app after setting the key
3. Verify the environment variable or config.py is updated

## Fallback System

The app includes a robust fallback system:
1. **Primary**: Alpha Vantage API
2. **Secondary**: yfinance (if Alpha Vantage fails)
3. **Tertiary**: Generated fallback data (if both fail)

This ensures your trading app always has data available! 