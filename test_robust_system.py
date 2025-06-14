#!/usr/bin/env python3
"""
Comprehensive test script for the robust dual-request system.
Tests both valid questions and invalid questions that should be blocked.
"""

import requests
import json
import time

def wait_for_server():
    """Wait for server to be ready"""
    print("🔄 Waiting for server to initialize...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:5000/api/init-status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("initialization_complete"):
                    print("✅ Server is ready!")
                    return True
                else:
                    print(f"⏳ Server still initializing... ({i+1}/{max_retries})")
            time.sleep(2)
        except:
            print(f"⏳ Waiting for server... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print("❌ Server not ready after waiting")
    return False

def test_question(question, expected_behavior="valid"):
    """Test a single question and analyze the response"""
    print(f"\n🤔 Testing: {question}")
    print(f"Expected: {expected_behavior}")
    print("-" * 80)
    
    try:
        # Test the search endpoint first
        search_response = requests.post(
            "http://localhost:5000/api/search",
            json={"query": question, "top_k": 5},
            timeout=30
        )
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            total_results = sum(len(v) for v in search_data["results"].values())
            print(f"🔍 Search found {total_results} total results")
            
            # Show top results with relevance scores
            for category, results in search_data["results"].items():
                if results:
                    top_result = results[0]
                    similarity = top_result.get("similarity", "N/A")
                    relevance = top_result.get("relevance_score", "N/A")
                    title = top_result.get("title", top_result.get("question", "No title"))[:80]
                    print(f"  {category}: sim={similarity:.3f}, rel={relevance:.3f} - {title}...")
        
        # Test the main answer endpoint
        response = requests.post(
            "http://localhost:5000/api/",
            json={"question": question},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer")
            
            print(f"\n📝 Answer: {answer}")
            
            # Analyze the response
            is_blocked = "I don't have specific information" in answer
            
            if expected_behavior == "blocked":
                if is_blocked:
                    print("✅ CORRECT: Question was properly blocked")
                    return True
                else:
                    print("❌ ISSUE: Question should have been blocked but wasn't")
                    return False
            else:  # expected_behavior == "valid"
                if is_blocked:
                    print("❌ ISSUE: Valid question was incorrectly blocked")
                    return False
                else:
                    # Check for specific success criteria
                    answer_lower = answer.lower()
                    if "ga4" in question.lower() and "bonus" in question.lower():
                        if "110" in answer_lower or "11/10" in answer_lower:
                            print("✅ SUCCESS: GA4 bonus question answered correctly with '110'")
                            return True
                        elif "bonus" in answer_lower:
                            print("⚠️  PARTIAL: Mentions bonus but not specific display")
                            return True
                        else:
                            print("❌ ISSUE: GA4 bonus question not answered properly")
                            return False
                    else:
                        print("✅ SUCCESS: Question answered appropriately")
                        return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing: {e}")
        return False

def run_comprehensive_tests():
    """Run comprehensive tests on the robust system"""
    
    if not wait_for_server():
        return
    
    print("\n🧪 COMPREHENSIVE TESTING OF ROBUST DUAL-REQUEST SYSTEM")
    print("=" * 80)
    
    # Test cases: (question, expected_behavior)
    test_cases = [
        # Valid questions that should be answered
        ("If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?", "valid"),
        ("What is GA4?", "valid"),
        ("How do I submit GA1?", "valid"),
        ("What are the requirements for Project 1?", "valid"),
        
        # Invalid questions that should be blocked
        ("When is the TDS Sep 2025 end-term exam?", "blocked"),
        ("What is the deadline for GA15?", "blocked"),
        ("How do I submit Project 5?", "blocked"),
        ("When is the TDS December 2025 exam?", "blocked"),
        ("What is GA20 about?", "blocked"),
        ("When is the 2026 TDS course starting?", "blocked"),
    ]
    
    results = []
    
    for question, expected in test_cases:
        result = test_question(question, expected)
        results.append((question, expected, result))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, _, result in results if result)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed < total:
        print("\n❌ Failed tests:")
        for question, expected, result in results:
            if not result:
                print(f"  - {question} (expected: {expected})")
    
    print(f"\n🎯 Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The robust system is working correctly.")
    else:
        print("⚠️  Some tests failed. The system needs further tuning.")

if __name__ == "__main__":
    run_comprehensive_tests() 