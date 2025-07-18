# Configuration file for external APIs
import os

# Alpha Vantage API Configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY') or 'YTHM5DS3LGQ9NE5C'

# REQUIRED: You must get your free API key from: https://www.alphavantage.co/support/#api-key
# Set your API key using: export ALPHA_VANTAGE_API_KEY=your_actual_api_key_here

if not ALPHA_VANTAGE_API_KEY:
    print("⚠️  WARNING: No Alpha Vantage API key found!")
    print("📋 To get real market data, please:")
    print("   1. Visit: https://www.alphavantage.co/support/#api-key")
    print("   2. Get your free API key")
    print("   3. Set it: export ALPHA_VANTAGE_API_KEY=your_key")
    print("   4. Restart the app")
    print("🚫 App will only return API errors without a valid key")

# API Rate Limiting
ALPHA_VANTAGE_REQUESTS_PER_MINUTE = 5  # Free tier allows 25 requests per day
ALPHA_VANTAGE_DAILY_LIMIT = 25

# Cache settings
ALPHA_VANTAGE_CACHE_DURATION = 300  # 5 minutes cache for Alpha Vantage data

# Real-time data only mode
REAL_TIME_DATA_ONLY = True  # No fallback to demo data 