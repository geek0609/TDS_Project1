#!/usr/bin/env python3
"""
Test script for TDS Virtual TA API based on promptfoo YAML test cases
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:5000/api"
TEST_CASES = [
    {
        "name": "GPT Model Question (with image)",
        "payload": {
            "question": "The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?",
            "image": "project-tds-virtual-ta-q1.webp"
        },
        "expectations": [
            "Answer clarifies use of gpt-3.5-turbo-0125 not gpt-4o-mini",
            "Links contain: https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939"
        ]
    },
    {
        "name": "GA4 Scoring Question",
        "payload": {
            "question": "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"
        },
        "expectations": [
            "Answer mentions the dashboard showing '110'",
            "Links contain: https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959"
        ]
    },
    {
        "name": "Docker vs Podman Question",
        "payload": {
            "question": "I know Docker but have not used Podman before. Should I use Docker for this course?"
        },
        "expectations": [
            "Answer recommends Podman for the course",
            "Answer mentions that Docker is acceptable",
            "Links contain: https://tds.s-anand.net/#/docker"
        ]
    },
    {
        "name": "Future Exam Date Question",
        "payload": {
            "question": "When is the TDS Sep 2025 end-term exam?"
        },
        "expectations": [
            "Answer says it doesn't know (since this information is not available yet)"
        ]
    },
    {
        "name": "GPT Model Question (without image)",
        "payload": {
            "question": "The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?"
        },
        "expectations": [
            "Answer clarifies use of gpt-3.5-turbo-0125 not gpt-4o-mini"
        ]
    }
]

def wait_for_server():
    """Wait for the Flask server to be ready."""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("Server is ready")
                return True
        except requests.ConnectionError:
            time.sleep(1)
    print("Server is not ready after multiple retries.")
    return False

def run_test(test_case):
    """Run a single test case and print detailed results."""
    print("="*80)
    print(f"TEST: {test_case['name']}")
    print(f"QUESTION: {test_case['payload']['question']}")
    print("\nEXPECTATIONS:")
    for expectation in test_case['expectations']:
        print(f"- {expectation}")
    print("="*80)
    
    try:
        response = requests.post(f"{BASE_URL}/", json=test_case["payload"], timeout=120)
        response.raise_for_status()
        data = response.json()

        print("\n--- RESPONSE ---")
        print(f"ANSWER:\n{data.get('answer', 'N/A')}")
        print("\nLINKS:")
        if data.get("links"):
            for link in data["links"]:
                print(f"- {link['text']}: {link['url']}")
        else:
            print("N/A")
        print("--- END RESPONSE ---\n")

        # Basic assertion checks for logging
        for expectation in test_case['expectations']:
            if "Links contain: " in expectation:
                link_to_find = expectation.replace("Links contain: ", "")
                found = any(link_to_find in link['url'] for link in data.get('links', []))
                status = "✅" if found else "❌"
                print(f"{status} Link check: {link_to_find}")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API request failed: {e}")
    
    print("\n\n")

def main():
    print("🚀 Testing TDS Virtual TA API")
    print("Based on promptfoo YAML test cases")
    print("Waiting for server to start and finish embeddings...")
    if not wait_for_server():
        return

    for test_case in TEST_CASES:
        run_test(test_case)

if __name__ == "__main__":
    main() 