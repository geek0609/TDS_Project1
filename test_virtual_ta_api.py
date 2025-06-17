#!/usr/bin/env python3
"""
Test script for TDS Virtual TA API based on promptfoo YAML test cases
"""

import requests
import json
import time
from pathlib import Path

def test_api_endpoint(question, image=None, expected_link=None):
    """Test a single API endpoint with a question"""
    url = "http://localhost:5000/api/"
    
    payload = {"question": question}
    if image:
        payload["image"] = image
    
    try:
        print(f"\n{'='*80}")
        print(f"Testing question: {question[:100]}...")
        if image:
            print(f"With image: {image}")
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ API Response successful")
            print(f"Answer length: {len(result.get('answer', ''))}")
            print(f"Number of links: {len(result.get('links', []))}")
            
            print(f"\n📝 Answer:")
            print(result.get('answer', 'No answer'))
            
            print(f"\n🔗 Links:")
            for i, link in enumerate(result.get('links', []), 1):
                print(f"  {i}. {link.get('text', 'No title')}")
                print(f"     {link.get('url', 'No URL')}")
            
            # Check if expected link is present
            if expected_link:
                links_str = json.dumps(result.get('links', []))
                if expected_link in links_str:
                    print(f"\n✅ Expected link found: {expected_link}")
                else:
                    print(f"\n❌ Expected link NOT found: {expected_link}")
                    print(f"Available links: {[link.get('url') for link in result.get('links', [])]}")
            
            return result
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the server running?")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🚀 Testing TDS Virtual TA API")
    print("Based on promptfoo YAML test cases")
    
    # Wait until the server is ready
    print("Waiting for server to start and finish embeddings...")
    health_url = "http://localhost:5000/api/health"
    for _ in range(180):  # Up to ~180 seconds
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print("Server is ready")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("Server did not become ready in time")
        return
    
    # Test 1: GPT model question with image
    print(f"\n{'='*80}")
    print("TEST 1: GPT Model Question (with image)")
    result1 = test_api_endpoint(
        question="The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?",
        image="file://project-tds-virtual-ta-q1.webp",
        expected_link="https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939"
    )
    
    # Test 2: GA4 scoring question
    print(f"\n{'='*80}")
    print("TEST 2: GA4 Scoring Question")
    result2 = test_api_endpoint(
        question="If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?",
        expected_link="https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959"
    )
    
    # Test 3: Docker vs Podman question
    print(f"\n{'='*80}")
    print("TEST 3: Docker vs Podman Question")
    result3 = test_api_endpoint(
        question="I know Docker but have not used Podman before. Should I use Docker for this course?",
        expected_link="https://tds.s-anand.net/#/docker"
    )
    
    # Test 4: Future exam date question
    print(f"\n{'='*80}")
    print("TEST 4: Future Exam Date Question")
    result4 = test_api_endpoint(
        question="When is the TDS Sep 2025 end-term exam?"
    )
    
    # Test 5: GPT model question WITHOUT image
    print(f"\n{'='*80}")
    print("TEST 5: GPT Model Question (without image)")
    result5 = test_api_endpoint(
        question="The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?"
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"Test 1 (GPT Model): {'✅ Success' if result1 else '❌ Failed'}")
    print(f"Test 2 (GA4 Scoring): {'✅ Success' if result2 else '❌ Failed'}")
    print(f"Test 3 (Docker/Podman): {'✅ Success' if result3 else '❌ Failed'}")
    print(f"Test 4 (Future Exam): {'✅ Success' if result4 else '❌ Failed'}")
    print(f"Test 5 (GPT Model w/o image): {'✅ Success' if result5 else '❌ Failed'}")
    
    # Analyze responses for expected content
    print(f"\n📋 CONTENT ANALYSIS")
    
    if result1:
        answer1 = result1.get('answer', '').lower()
        if 'gpt-3.5-turbo' in answer1 or 'gpt-4o-mini' in answer1:
            print("✅ Test 1: Mentions expected models")
        else:
            print("❌ Test 1: Doesn't mention expected models")
    
    if result2:
        answer2 = result2.get('answer', '').lower()
        if '110' in answer2 or 'dashboard' in answer2:
            print("✅ Test 2: Mentions dashboard scoring")
        else:
            print("❌ Test 2: Doesn't mention dashboard scoring")
    
    if result3:
        answer3 = result3.get('answer', '').lower()
        if 'podman' in answer3:
            print("✅ Test 3: Mentions Podman")
        else:
            print("❌ Test 3: Doesn't mention Podman")
    
    if result4:
        answer4 = result4.get('answer', '').lower()
        if "don't know" in answer4 or "not available" in answer4 or "don't have" in answer4:
            print("✅ Test 4: Correctly says it doesn't know future information")
        else:
            print("❌ Test 4: Should say it doesn't know future information")

    if result5:
        answer5 = result5.get('answer', '').lower()
        if 'gpt-3.5-turbo' in answer5 or 'gpt-4o-mini' in answer5:
            print("✅ Test 5: Mentions expected models")
        else:
            print("❌ Test 5: Doesn't mention expected models")

if __name__ == "__main__":
    main() 