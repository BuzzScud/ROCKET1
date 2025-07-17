#!/usr/bin/env python3
"""
Setup script for Alpha Vantage API key
This script helps you configure your Alpha Vantage API key for real-time market data.
"""

import os
import sys

def main():
    print("🚀 Alpha Vantage API Key Setup")
    print("=" * 50)
    
    # Check if API key is already set
    current_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if current_key:
        print(f"✅ API key already set: {current_key[:8]}...")
        overwrite = input("Do you want to update it? (y/n): ").lower()
        if overwrite != 'y':
            print("Setup cancelled.")
            return
    
    print("\n📋 Instructions:")
    print("1. Visit: https://www.alphavantage.co/support/#api-key")
    print("2. Fill in your details and get your free API key")
    print("3. Copy the API key and paste it below")
    print()
    
    # Get API key from user
    api_key = input("Enter your Alpha Vantage API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return
    
    if len(api_key) < 10:
        print("❌ API key seems too short. Please check and try again.")
        return
    
    # Set the environment variable
    os.environ['ALPHA_VANTAGE_API_KEY'] = api_key
    
    print(f"\n✅ API key set successfully!")
    print(f"Key: {api_key[:8]}...")
    print()
    print("🔧 To make this permanent, add this to your shell profile:")
    print(f"export ALPHA_VANTAGE_API_KEY={api_key}")
    print()
    print("Or add it to your .bashrc/.zshrc file:")
    print(f"echo 'export ALPHA_VANTAGE_API_KEY={api_key}' >> ~/.bashrc")
    print()
    print("🎯 Now restart your Flask app to start getting real market data!")
    print("python app.py")

if __name__ == "__main__":
    main() 