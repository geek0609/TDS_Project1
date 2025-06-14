#!/usr/bin/env python3
"""
Test script for the optimized TDS Virtual TA system
Tests all promptfoo cases including image support
"""

import requests
import json
import time
import base64
from pathlib import Path

def test_virtual_ta():
    """Test the Virtual TA with all promptfoo test cases"""
    
    base_url = "http://localhost:5000/api/"
    
    print("🧪 Testing Optimized TDS Virtual TA System")
    print("=" * 50)
    
    # Test cases from promptfoo.yaml
    test_cases = [
        {
            "name": "GA5 Model Clarification (with image)",
            "question": "The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?",
            "image": "file://project-tds-virtual-ta-q1.webp",
            "expected_link": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939",
            "expected_content": ["gpt-3.5-turbo-0125", "gpt-4o-mini"],
            "should_not_contain": []
        },
        {
            "name": "GA4 Bonus Score Display",
            "question": "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?",
            "image": None,
            "expected_link": "https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959",
            "expected_content": ["110", "dashboard"],
            "should_not_contain": []
        },
        {
            "name": "Docker vs Podman",
            "question": "I know Docker but have not used Podman before. Should I use Docker for this course?",
            "image": None,
            "expected_link": "https://tds.s-anand.net/#/docker",
            "expected_content": ["Podman", "Docker"],
            "should_not_contain": []
        },
        {
            "name": "Future Exam Date (Anti-hallucination)",
            "question": "When is the TDS Sep 2025 end-term exam?",
            "image": None,
            "expected_link": None,
            "expected_content": ["don't know", "don't have information"],
            "should_not_contain": ["September", "exam date", "2025"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print("-" * 40)
        
        # Prepare request data
        request_data = {
            "question": test_case["question"]
        }
        
        # Add image if specified
        if test_case["image"]:
            request_data["image"] = test_case["image"]
        
        try:
            # Make request
            start_time = time.time()
            response = requests.post(base_url, json=request_data, timeout=60)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "")
                links = result.get("links", [])
                search_count = result.get("search_results_count", 0)
                intent_analysis = result.get("intent_analysis", {})
                
                print(f"✅ Response received ({response_time:.2f}s)")
                print(f"📊 Search results: {search_count}")
                print(f"🔗 Links found: {len(links)}")
                print(f"🧠 Intent: {intent_analysis.get('type', 'unknown')}")
                
                # Check answer content
                answer_lower = answer.lower()
                content_score = 0
                content_details = []
                
                for expected in test_case["expected_content"]:
                    if expected.lower() in answer_lower:
                        content_score += 1
                        content_details.append(f"✅ Found: '{expected}'")
                    else:
                        content_details.append(f"❌ Missing: '{expected}'")
                
                # Check for content that should not be present
                unwanted_score = 0
                for unwanted in test_case["should_not_contain"]:
                    if unwanted.lower() in answer_lower:
                        unwanted_score += 1
                        content_details.append(f"⚠️ Unwanted: '{unwanted}'")
                
                # Check links
                link_score = 0
                link_details = []
                if test_case["expected_link"]:
                    link_urls = [link.get("url", "") for link in links]
                    if any(test_case["expected_link"] in url for url in link_urls):
                        link_score = 1
                        link_details.append(f"✅ Found expected link")
                    else:
                        link_details.append(f"❌ Missing expected link: {test_case['expected_link']}")
                        link_details.append(f"   Found links: {link_urls}")
                else:
                    link_score = 1  # No specific link expected
                    link_details.append("✅ No specific link required")
                
                # Calculate overall score
                total_expected = len(test_case["expected_content"])
                content_percentage = (content_score / total_expected * 100) if total_expected > 0 else 100
                unwanted_penalty = unwanted_score * 20  # 20% penalty per unwanted content
                final_score = max(0, (content_percentage + link_score * 50) / 1.5 - unwanted_penalty)
                
                # Store result
                test_result = {
                    "name": test_case["name"],
                    "score": final_score,
                    "content_score": content_score,
                    "total_expected": total_expected,
                    "link_score": link_score,
                    "unwanted_count": unwanted_score,
                    "response_time": response_time,
                    "answer_length": len(answer),
                    "links_count": len(links),
                    "search_results": search_count,
                    "intent": intent_analysis.get('type', 'unknown')
                }
                results.append(test_result)
                
                # Print detailed results
                print(f"📝 Answer length: {len(answer)} characters")
                print(f"🎯 Content score: {content_score}/{total_expected}")
                for detail in content_details:
                    print(f"   {detail}")
                print(f"🔗 Link score: {link_score}")
                for detail in link_details:
                    print(f"   {detail}")
                if unwanted_score > 0:
                    print(f"⚠️ Unwanted content penalty: -{unwanted_penalty}%")
                print(f"🏆 Final score: {final_score:.1f}%")
                
                # Show first 200 chars of answer
                print(f"💬 Answer preview: {answer[:200]}...")
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Response: {response.text}")
                results.append({
                    "name": test_case["name"],
                    "score": 0,
                    "error": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                "name": test_case["name"],
                "score": 0,
                "error": str(e)
            })
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 EVALUATION SUMMARY")
    print("=" * 50)
    
    total_score = 0
    successful_tests = 0
    
    for result in results:
        if "error" not in result:
            print(f"{result['name']}: {result['score']:.1f}%")
            total_score += result['score']
            successful_tests += 1
        else:
            print(f"{result['name']}: FAILED ({result['error']})")
    
    if successful_tests > 0:
        average_score = total_score / successful_tests
        print(f"\n🎯 Overall Score: {average_score:.1f}%")
        print(f"✅ Successful tests: {successful_tests}/{len(test_cases)}")
        
        # Performance metrics
        response_times = [r.get('response_time', 0) for r in results if 'response_time' in r]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            print(f"⚡ Average response time: {avg_response_time:.2f}s")
        
        # Detailed breakdown
        print(f"\n📈 DETAILED BREAKDOWN:")
        for result in results:
            if "error" not in result:
                print(f"  {result['name']}:")
                print(f"    Score: {result['score']:.1f}%")
                print(f"    Content: {result['content_score']}/{result['total_expected']}")
                print(f"    Links: {result['link_score']}")
                print(f"    Response time: {result['response_time']:.2f}s")
                print(f"    Intent detected: {result['intent']}")
    else:
        print("❌ No tests completed successfully")
    
    return results

if __name__ == "__main__":
    results = test_virtual_ta() 