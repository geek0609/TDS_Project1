#!/usr/bin/env python3
"""
Vercel serverless function entry point for TDS Virtual TA
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the Python path so we can import virtual_ta_serverless
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import the Flask app from virtual_ta_serverless
from api.virtual_ta_serverless import app

# Vercel expects the app to be available as 'app'
# This is the WSGI application that Vercel will use
if __name__ == "__main__":
    app.run(debug=False) 