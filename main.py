#!/usr/bin/env python3
"""
Main entry point for Google Cloud Run - redirects to Flask app
This file exists to satisfy buildpack requirements but we use Docker.
"""

if __name__ == '__main__':
    # Import and run the actual Flask app
    from app import app
    import os
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 