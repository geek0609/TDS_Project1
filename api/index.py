#!/usr/bin/env python3
"""
Vercel serverless function entry point for TDS Virtual TA
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_simple import app

# Vercel expects the Flask app to be available as 'app'
# This is the entry point for Vercel serverless functions
if __name__ == "__main__":
    app.run() 