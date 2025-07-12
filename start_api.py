#!/usr/bin/env python3
"""
Market API Startup Script
This script sets up the environment and starts the Yahoo Finance API server.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def start_api_server():
    """Start the API server"""
    print("Starting Yahoo Finance API server...")
    try:
        # Import and run the Flask app
        from market_api import app
        print("🚀 Market API Server Starting...")
        print("📊 Yahoo Finance data provider ready!")
        print("🌐 Server will be available at: http://localhost:5001")
        print("💡 Open your browser and navigate to the charts page to test")
        print("⚡ Press Ctrl+C to stop the server")
        app.run(debug=True, host='0.0.0.0', port=5001)
    except ImportError as e:
        print(f"❌ Error importing market_api: {e}")
        print("Make sure market_api.py is in the same directory")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main function"""
    print("=" * 50)
    print("🏢 ROCKET TRADING GROUP - MARKET API SETUP")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        print("Make sure you're in the correct directory")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    print("\n" + "=" * 50)
    print("🚀 STARTING MARKET API SERVER")
    print("=" * 50)
    
    # Start the API server
    start_api_server()

if __name__ == "__main__":
    main() 