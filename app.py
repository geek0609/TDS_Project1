# Import the main Flask app from app_gemini.py
# This ensures compatibility with default deployment commands like 'gunicorn app:app'
from app_gemini import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 