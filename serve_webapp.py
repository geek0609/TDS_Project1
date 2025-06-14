#!/usr/bin/env python3
"""
Simple HTTP server to serve the TDS Virtual TA web app
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

# Configuration
PORT = 8080
HOST = 'localhost'

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for API requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()

def serve_webapp():
    """Start the web server and open the browser"""
    
    # Change to the directory containing index.html
    web_dir = Path(__file__).parent
    os.chdir(web_dir)
    
    # Create server
    with socketserver.TCPServer((HOST, PORT), CustomHTTPRequestHandler) as httpd:
        print("🚀 Starting TDS Virtual TA Web App Server...")
        print(f"📍 Server running at: http://{HOST}:{PORT}")
        print(f"🌐 Web App URL: http://{HOST}:{PORT}/index.html")
        print("📱 The web app will open in your default browser")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Open browser
        webbrowser.open(f'http://{HOST}:{PORT}/index.html')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
            print("👋 Thanks for using TDS Virtual TA!")

if __name__ == "__main__":
    serve_webapp() 