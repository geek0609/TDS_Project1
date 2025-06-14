#!/usr/bin/env python3
"""
Test script to compare local vs deployed API responses
"""

import requests
import json
import time

# API endpoints
LOCAL_API = "http://127.0.0.1:5000/api/"
DEPLOYED_API = "https://tds-project1-5e44.onrender.com/api/"

# Test questions
test_questions = [
    "What is the deadline for project submission P1?",
    "How do I submit my project?",
    "What are the grading criteria for the project?",
    "Can I use external libraries in my project?",
    "What is the format for the project report?"
]

def test_api(api_url, name):
    print(f"\n🔍 Testing {name} API: {api_url}")
    print("=" * 60)
    
    # Test health endpoint first
    try:
        health_response = requests.get(api_url.replace('/api/', '/api/health'), timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Health check passed")
            print(f"   📊 Topics: {health_data.get('discourse_topics', 'N/A')}")
            print(f"   📝 Q&A pairs: {health_data.get('discourse_qa_pairs', 'N/A')}")
            print(f"   📚 Course topics: {health_data.get('course_topics', 'N/A')}")
            print(f"   💻 Code examples: {health_data.get('course_code_examples', 'N/A')}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test first question
    question = test_questions[0]
    print(f"\n📝 Testing question: '{question}'")
    
    try:
        start_time = time.time()
        response = requests.post(api_url, 
                               json={'question': question}, 
                               timeout=30)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', 'No answer provided')
            links = data.get('links', [])
            
            print(f"✅ Response received ({response_time:.2f}s)")
            print(f"📄 Answer preview: {answer[:150]}...")
            print(f"🔗 Links provided: {len(links)}")
            
            # Check if it's a generic response
            if "Based on an accepted answer" in answer and len(answer) < 300:
                print("⚠️  WARNING: This looks like a generic fallback response!")
            else:
                print("✅ Response appears to be specific and detailed")
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Request error: {e}")

def main():
    print("🚀 TDS Virtual TA API Comparison Test")
    print("=" * 60)
    
    # Test deployed API
    test_api(DEPLOYED_API, "Deployed")
    
    print("\n" + "=" * 60)
    print("💡 To test local API, start the server with: python app_gemini.py")
    print("   Then run this script again to compare both APIs")

if __name__ == "__main__":
    main() 