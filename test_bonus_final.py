#!/usr/bin/env python3
"""
Test script for the GA4 bonus question with the updated less conservative system.
"""

import requests
import json
import time

def test_bonus_question():
    """Test the specific GA4 bonus question"""
    
    # Wait for server to be ready
    print("🔄 Waiting for server to initialize...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:5000/api/init-status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("initialization_complete"):
                    print("✅ Server is ready!")
                    break
                else:
                    print(f"⏳ Server still initializing... ({i+1}/{max_retries})")
            time.sleep(2)
        except:
            print(f"⏳ Waiting for server... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("❌ Server not ready after waiting")
        return
    
    # Test the bonus question
    question = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"
    
    print(f"\n🤔 Testing question: {question}")
    print("-" * 80)
    
    try:
        # First test the search endpoint to see what context is found
        print("🔍 Testing search endpoint...")
        search_response = requests.post(
            "http://localhost:5000/api/search",
            json={"query": question, "top_k": 10},
            timeout=30
        )
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            print(f"📊 Search results:")
            for category, results in search_data["results"].items():
                if results:
                    print(f"  {category}: {len(results)} results")
                    for i, result in enumerate(results[:3]):
                        similarity = result.get("similarity", "N/A")
                        title = result.get("title", result.get("question", "No title"))[:100]
                        print(f"    {i+1}. Similarity: {similarity:.3f} - {title}")
        
        print("\n💬 Testing answer endpoint...")
        # Test the main answer endpoint
        response = requests.post(
            "http://localhost:5000/api/",
            json={"question": question},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer")
            links = data.get("links", [])
            search_counts = data.get("search_results_count", {})
            
            print(f"✅ Response received!")
            print(f"📝 Answer: {answer}")
            print(f"\n📊 Search results count: {search_counts}")
            print(f"🔗 Links provided: {len(links)}")
            for link in links:
                print(f"  - {link.get('text', 'No title')}: {link.get('url', 'No URL')}")
            
            # Check if the answer mentions key terms
            answer_lower = answer.lower()
            key_terms = ["110", "11/10", "bonus", "dashboard", "ga4"]
            found_terms = [term for term in key_terms if term in answer_lower]
            
            print(f"\n🔍 Key terms found in answer: {found_terms}")
            
            if "110" in answer_lower or "11/10" in answer_lower:
                print("🎉 SUCCESS: Answer mentions the correct dashboard display!")
            elif "bonus" in answer_lower and "ga4" in answer_lower:
                print("⚠️  PARTIAL: Answer mentions GA4 bonus but not the specific display")
            else:
                print("❌ ISSUE: Answer doesn't seem to address the bonus question properly")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing: {e}")

def test_simple_ga4_question():
    """Test a simple GA4 question to ensure basic functionality"""
    question = "What is GA4?"
    
    print(f"\n🤔 Testing simple question: {question}")
    print("-" * 80)
    
    try:
        response = requests.post(
            "http://localhost:5000/api/",
            json={"question": question},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer")
            print(f"✅ Simple GA4 answer: {answer[:200]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 TESTING GA4 BONUS QUESTION WITH LESS CONSERVATIVE SYSTEM")
    print("=" * 80)
    
    # Test simple question first
    test_simple_ga4_question()
    
    # Test the bonus question
    test_bonus_question()
    
    print("\n" + "=" * 80)
    print("🏁 Testing complete!") 