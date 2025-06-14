#!/bin/bash

# Start the TDS Virtual TA Flask application
echo "🚀 Starting TDS Virtual TA on Render..."
echo "📍 Current directory: $(pwd)"
echo "📁 Files in directory: $(ls -la)"

# Check if knowledge base files exist
if [ -d "scripts/processed" ]; then
    echo "✅ scripts/processed directory found"
    echo "📄 Files: $(ls scripts/processed/)"
else
    echo "❌ scripts/processed directory NOT found"
fi

if [ -d "scripts/processed_course" ]; then
    echo "✅ scripts/processed_course directory found"
    echo "📄 Files: $(ls scripts/processed_course/)"
else
    echo "❌ scripts/processed_course directory NOT found"
fi

# Start the application with gunicorn
echo "🔥 Starting gunicorn server..."
exec gunicorn app_gemini:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 