#!/usr/bin/env python3
"""
Python-based test script to replace PromptFoo evaluation.
Tests the TDS Virtual TA API with the same questions from project-tds-virtual-ta-promptfoo.yaml
"""

import requests
import json
import base64
import os
from pathlib import Path
import time
from typing import Dict, List, Any, Optional

class VirtualTATestRunner:
    def __init__(self, api_url: str = "http://localhost:5000/api/"):
        self.api_url = api_url
        self.results = []
        
    def load_image_as_base64(self, image_path: str) -> Optional[str]:
        """Load image file and convert to base64"""
        try:
            if image_path.startswith("file://"):
                image_path = image_path[7:]  # Remove file:// prefix
            
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            else:
                print(f"⚠️  Image file not found: {image_path}")
                return None
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def make_api_request(self, question: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Make API request to the Virtual TA"""
        payload = {"question": question}
        
        if image_path:
            image_b64 = self.load_image_as_base64(image_path)
            if image_b64:
                payload["image"] = image_b64
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "answer": "",
                    "links": []
                }
        except Exception as e:
            return {
                "error": f"Request failed: {str(e)}",
                "answer": "",
                "links": []
            }
    
    def check_json_structure(self, response: Dict[str, Any]) -> bool:
        """Check if response has required JSON structure"""
        required_fields = ["answer", "links"]
        
        for field in required_fields:
            if field not in response:
                return False
        
        if not isinstance(response["answer"], str):
            return False
        
        if not isinstance(response["links"], list):
            return False
        
        for link in response["links"]:
            if not isinstance(link, dict):
                return False
            if "url" not in link or "text" not in link:
                return False
            if not isinstance(link["url"], str) or not isinstance(link["text"], str):
                return False
        
        return True
    
    def check_contains(self, text: str, search_term: str) -> bool:
        """Check if text contains search term (case insensitive)"""
        return search_term.lower() in text.lower()
    
    def check_llm_rubric(self, answer: str, criteria: str) -> bool:
        """
        Simple heuristic check for LLM rubric criteria.
        In a real implementation, this could use another LLM to evaluate.
        """
        answer_lower = answer.lower()
        criteria_lower = criteria.lower()
        
        # Extract key terms from criteria
        if "gpt-3.5-turbo-0125" in criteria_lower and "not gpt-4o-mini" in criteria_lower:
            return "gpt-3.5-turbo" in answer_lower and "gpt-4o-mini" not in answer_lower
        
        if "dashboard showing" in criteria_lower and "110" in criteria_lower:
            return "110" in answer_lower or "11/10" in answer_lower or "bonus" in answer_lower
        
        if "recommends podman" in criteria_lower:
            return "podman" in answer_lower and ("recommend" in answer_lower or "use" in answer_lower)
        
        if "docker is acceptable" in criteria_lower:
            return "docker" in answer_lower and ("acceptable" in answer_lower or "fine" in answer_lower or "ok" in answer_lower)
        
        if "doesn't know" in criteria_lower:
            return any(phrase in answer_lower for phrase in [
                "don't know", "doesn't know", "not available", "not sure", 
                "cannot find", "no information", "not provided"
            ])
        
        # Fallback: check if any key words from criteria appear in answer
        criteria_words = criteria_lower.split()
        return any(word in answer_lower for word in criteria_words if len(word) > 3)
    
    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        question = test_case["question"]
        image_path = test_case.get("image")
        expected_link = test_case.get("link")
        assertions = test_case.get("assertions", [])
        
        print(f"\n🧪 Testing: {question[:80]}...")
        
        start_time = time.time()
        response = self.make_api_request(question, image_path)
        response_time = time.time() - start_time
        
        result = {
            "question": question,
            "response": response,
            "response_time": response_time,
            "assertions": [],
            "passed": True
        }
        
        # Check JSON structure
        json_valid = self.check_json_structure(response)
        result["assertions"].append({
            "type": "json_structure",
            "passed": json_valid,
            "description": "Response has valid JSON structure"
        })
        
        if not json_valid:
            result["passed"] = False
            print(f"❌ JSON structure validation failed")
            return result
        
        # Run custom assertions
        for assertion in assertions:
            assertion_result = self.run_assertion(response, assertion)
            result["assertions"].append(assertion_result)
            if not assertion_result["passed"]:
                result["passed"] = False
        
        # Check for expected link if provided
        if expected_link:
            links_text = json.dumps([link["url"] for link in response.get("links", [])])
            link_found = self.check_contains(links_text, expected_link)
            result["assertions"].append({
                "type": "contains_link",
                "passed": link_found,
                "description": f"Response contains expected link: {expected_link}",
                "expected": expected_link,
                "actual": links_text
            })
            if not link_found:
                result["passed"] = False
        
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"{status} ({response_time:.2f}s)")
        
        return result
    
    def run_assertion(self, response: Dict[str, Any], assertion: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single assertion"""
        assertion_type = assertion["type"]
        
        if assertion_type == "llm_rubric":
            criteria = assertion["criteria"]
            answer = response.get("answer", "")
            passed = self.check_llm_rubric(answer, criteria)
            
            return {
                "type": "llm_rubric",
                "passed": passed,
                "description": f"LLM rubric: {criteria}",
                "criteria": criteria,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer
            }
        
        elif assertion_type == "contains":
            search_term = assertion["value"]
            search_text = assertion.get("transform", "answer")
            
            if search_text == "links":
                text_to_search = json.dumps(response.get("links", []))
            else:
                text_to_search = response.get("answer", "")
            
            passed = self.check_contains(text_to_search, search_term)
            
            return {
                "type": "contains",
                "passed": passed,
                "description": f"Contains '{search_term}' in {search_text}",
                "expected": search_term,
                "actual": text_to_search[:200] + "..." if len(text_to_search) > 200 else text_to_search
            }
        
        else:
            return {
                "type": assertion_type,
                "passed": False,
                "description": f"Unknown assertion type: {assertion_type}"
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases from the PromptFoo YAML equivalent"""
        
        test_cases = [
            {
                "question": "The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?",
                "image": "project-tds-virtual-ta-q1.webp",
                "link": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939",
                "assertions": [
                    {
                        "type": "llm_rubric",
                        "criteria": "Clarifies use of gpt-3.5-turbo-0125 not gpt-4o-mini"
                    }
                ]
            },
            {
                "question": "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?",
                "link": "https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959/388",
                "assertions": [
                    {
                        "type": "llm_rubric",
                        "criteria": "Mentions the dashboard showing \"110\""
                    }
                ]
            },
            {
                "question": "I know Docker but have not used Podman before. Should I use Docker for this course?",
                "assertions": [
                    {
                        "type": "llm_rubric",
                        "criteria": "Recommends Podman for the course"
                    },
                    {
                        "type": "llm_rubric",
                        "criteria": "Mentions that Docker is acceptable"
                    },
                    {
                        "type": "contains",
                        "value": "https://tds.s-anand.net/#/docker",
                        "transform": "links"
                    }
                ]
            },
            {
                "question": "When is the TDS Sep 2025 end-term exam?",
                "assertions": [
                    {
                        "type": "llm_rubric",
                        "criteria": "Says it doesn't know (since this information is not available yet)"
                    }
                ]
            }
        ]
        
        print("🚀 Starting TDS Virtual TA Test Suite")
        print(f"📡 API URL: {self.api_url}")
        print(f"🧪 Running {len(test_cases)} test cases...\n")
        
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}]", end=" ")
            result = self.run_test_case(test_case)
            self.results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate summary
        passed_tests = sum(1 for r in self.results if r["passed"])
        total_tests = len(self.results)
        total_assertions = sum(len(r["assertions"]) for r in self.results)
        passed_assertions = sum(sum(1 for a in r["assertions"] if a["passed"]) for r in self.results)
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "total_assertions": total_assertions,
            "passed_assertions": passed_assertions,
            "failed_assertions": total_assertions - passed_assertions,
            "total_time": total_time,
            "average_response_time": sum(r["response_time"] for r in self.results) / total_tests,
            "results": self.results
        }
        
        self.print_summary(summary)
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        print(f"🧪 Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        print(f"✅ Assertions: {summary['passed_assertions']}/{summary['total_assertions']} passed")
        print(f"⏱️  Total time: {summary['total_time']:.2f}s")
        print(f"📈 Average response time: {summary['average_response_time']:.2f}s")
        
        success_rate = (summary['passed_tests'] / summary['total_tests']) * 100
        print(f"🎯 Success rate: {success_rate:.1f}%")
        
        if summary['failed_tests'] > 0:
            print(f"\n❌ FAILED TESTS ({summary['failed_tests']}):")
            for i, result in enumerate(summary['results'], 1):
                if not result['passed']:
                    print(f"\n[{i}] {result['question'][:80]}...")
                    for assertion in result['assertions']:
                        if not assertion['passed']:
                            print(f"   ❌ {assertion['description']}")
        
        print("\n" + "="*60)

def main():
    """Main function to run the tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TDS Virtual TA API")
    parser.add_argument(
        "--url", 
        default="http://localhost:5000/api/",
        help="API URL to test (default: http://localhost:5000/api/)"
    )
    parser.add_argument(
        "--output",
        help="Save detailed results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Check if API is running
    try:
        health_response = requests.get(args.url.replace("/api/", "/api/health"), timeout=10)
        if health_response.status_code != 200:
            print(f"⚠️  Warning: Health check failed (HTTP {health_response.status_code})")
    except Exception as e:
        print(f"❌ Error: Cannot connect to API at {args.url}")
        print(f"   Make sure the server is running: python app_gemini.py")
        return 1
    
    # Run tests
    runner = VirtualTATestRunner(args.url)
    summary = runner.run_all_tests()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Detailed results saved to: {args.output}")
    
    # Return exit code based on success
    return 0 if summary['failed_tests'] == 0 else 1

if __name__ == "__main__":
    exit(main()) 