#!/usr/bin/env python3
"""
Specific test for GA4 bonus question with different phrasings
to see if we can get the "110" answer.
"""

import requests
import time

def test_ga4_bonus_variations():
    """Test different variations of the GA4 bonus question"""
    
    questions = [
        "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?",
        "What does 10/10 + bonus look like on the GA4 dashboard?",
        "How is GA4 bonus score displayed on dashboard?",
        "If someone gets full marks plus bonus on GA4, what shows on dashboard?",
        "GA4 dashboard display for 10/10 with bonus marks?",
        "What number appears on dashboard for GA4 10/10 + bonus?",
        "How does GA4 bonus calculation show up - like 110 or 11/10?",
        "GA4 scoring: if base score is 10/10 and bonus added, dashboard shows what?",
    ]
    
    print("🧪 TESTING GA4 BONUS QUESTION VARIATIONS")
    print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n🤔 Test {i}: {question}")
        print("-" * 60)
        
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
                
                # Show top results
                for category, results in search_data["results"].items():
                    if results:
                        top_result = results[0]
                        similarity = top_result.get("similarity", "N/A")
                        relevance = top_result.get("relevance_score", "N/A")
                        title = top_result.get("title", top_result.get("question", "No title"))[:60]
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
                
                # Check for specific success indicators
                answer_lower = answer.lower()
                success_indicators = ["110", "11/10", "102.5", "dashboard", "bonus"]
                found_indicators = [ind for ind in success_indicators if ind in answer_lower]
                
                if "110" in answer_lower:
                    print("🎯 SUCCESS: Found '110' in answer!")
                elif "11/10" in answer_lower:
                    print("🎯 SUCCESS: Found '11/10' in answer!")
                elif "102.5" in answer_lower:
                    print("🎯 PARTIAL: Found calculation '102.5' in answer!")
                elif found_indicators:
                    print(f"⚠️  PARTIAL: Found indicators: {found_indicators}")
                else:
                    print("❌ MISS: No specific indicators found")
                    
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing: {e}")
        
        time.sleep(2)  # Brief pause between tests

if __name__ == "__main__":
    test_ga4_bonus_variations() 