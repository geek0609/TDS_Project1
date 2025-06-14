#!/usr/bin/env python3
"""
Comprehensive Test Runner for TDS Virtual TA
Runs 1000 test questions and evaluates API performance including hallucination detection
"""

import json
import requests
import time
import re
from pathlib import Path
from typing import Dict, List, Any
import statistics

class ComprehensiveTestRunner:
    def __init__(self, api_url: str = "http://localhost:5000/api/"):
        self.api_url = api_url
        self.results = []
        self.stats = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "hallucination_detected": 0,
            "hallucination_prevented": 0,
            "response_times": [],
            "category_stats": {},
            "difficulty_stats": {}
        }
    
    def load_test_suite(self, filename: str = "comprehensive_test_suite.json") -> List[Dict]:
        """Load the comprehensive test suite"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("questions", [])
        except Exception as e:
            print(f"❌ Error loading test suite: {e}")
            return []
    
    def make_api_request(self, question: str) -> Dict[str, Any]:
        """Make API request to the Virtual TA"""
        try:
            response = requests.post(
                self.api_url,
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=30
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
    
    def check_hallucination(self, question: str, answer: str, should_find_answer: bool) -> bool:
        """Check if the API is hallucinating"""
        answer_lower = answer.lower()
        
        uncertainty_phrases = [
            "don't know", "doesn't know", "not available", "not sure", 
            "cannot find", "no information", "not provided", "unclear",
            "i don't have", "i cannot", "i'm not sure", "i don't see",
            "not in my knowledge", "not in the data", "cannot determine",
            "insufficient information", "no specific information",
            "i apologize", "sorry", "unable to find"
        ]
        
        admits_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)
        
        if not should_find_answer:
            return not admits_uncertainty
        else:
            return admits_uncertainty and len(answer) < 50
    
    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        question = test_case["question"]
        should_find_answer = test_case.get("should_find_answer", True)
        category = test_case.get("category", "unknown")
        difficulty = test_case.get("difficulty", "medium")
        
        start_time = time.time()
        response = self.make_api_request(question)
        response_time = time.time() - start_time
        
        answer = response.get("answer", "")
        links = response.get("links", [])
        
        is_hallucinating = self.check_hallucination(question, answer, should_find_answer)
        
        # Simple scoring
        if should_find_answer:
            passed = not is_hallucinating and len(answer) > 20
        else:
            passed = not is_hallucinating
        
        result = {
            "test_id": test_case.get("id", 0),
            "question": question,
            "category": category,
            "difficulty": difficulty,
            "should_find_answer": should_find_answer,
            "response": response,
            "response_time": response_time,
            "is_hallucinating": is_hallucinating,
            "passed": passed
        }
        
        return result
    
    def run_all_tests(self, test_questions: List[Dict], max_tests: int = None) -> Dict[str, Any]:
        """Run all test cases"""
        if max_tests:
            test_questions = test_questions[:max_tests]
        
        print(f"🚀 Starting comprehensive test run with {len(test_questions)} questions...")
        
        for i, test_case in enumerate(test_questions, 1):
            print(f"\r[{i}/{len(test_questions)}] Testing... ", end="", flush=True)
            
            result = self.run_single_test(test_case)
            self.results.append(result)
            
            self.update_stats(result)
            time.sleep(0.1)
        
        print(f"\n✅ Completed {len(test_questions)} tests")
        return {"results": self.results, "statistics": self.generate_final_stats()}
    
    def update_stats(self, result: Dict[str, Any]):
        """Update running statistics"""
        self.stats["total_tests"] += 1
        
        if result["passed"]:
            self.stats["passed_tests"] += 1
        else:
            self.stats["failed_tests"] += 1
        
        if result["is_hallucinating"]:
            self.stats["hallucination_detected"] += 1
        else:
            self.stats["hallucination_prevented"] += 1
        
        self.stats["response_times"].append(result["response_time"])
    
    def generate_final_stats(self) -> Dict[str, Any]:
        """Generate final statistics"""
        response_times = self.stats["response_times"]
        
        return {
            "total_tests": self.stats["total_tests"],
            "passed_tests": self.stats["passed_tests"],
            "failed_tests": self.stats["failed_tests"],
            "success_rate": (self.stats["passed_tests"] / self.stats["total_tests"]) * 100,
            "hallucination_detected": self.stats["hallucination_detected"],
            "hallucination_prevented": self.stats["hallucination_prevented"],
            "hallucination_rate": (self.stats["hallucination_detected"] / self.stats["total_tests"]) * 100,
            "average_response_time": statistics.mean(response_times) if response_times else 0
        }
    
    def print_report(self, stats: Dict[str, Any]):
        """Print test report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*60)
        print(f"🧪 Total Tests: {stats['total_tests']}")
        print(f"✅ Passed: {stats['passed_tests']}")
        print(f"❌ Failed: {stats['failed_tests']}")
        print(f"🎯 Success Rate: {stats['success_rate']:.1f}%")
        print(f"🧠 Hallucination Rate: {stats['hallucination_rate']:.1f}%")
        print(f"⏱️  Avg Response Time: {stats['average_response_time']:.2f}s")
        print("="*60)
    
    def save_results_markdown(self, results: Dict[str, Any], timestamp: str):
        """Save test results to markdown file for manual review"""
        filename = f"test_results_{timestamp}.md"
        
        stats = results["statistics"]
        test_results = results["results"]
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# TDS Virtual TA Test Results\n\n")
            f.write(f"**Test Run:** {timestamp}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Tests:** {stats['total_tests']}\n")
            f.write(f"- **Passed:** {stats['passed_tests']}\n")
            f.write(f"- **Failed:** {stats['failed_tests']}\n")
            f.write(f"- **Success Rate:** {stats['success_rate']:.1f}%\n")
            f.write(f"- **Hallucination Rate:** {stats['hallucination_rate']:.1f}%\n")
            f.write(f"- **Average Response Time:** {stats['average_response_time']:.2f}s\n\n")
            
            # Group results by category
            categories = {}
            for result in test_results:
                category = result.get('category', 'unknown')
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)
            
            f.write(f"## Test Results by Category\n\n")
            
            for category, tests in categories.items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                
                passed_count = sum(1 for t in tests if t['passed'])
                total_count = len(tests)
                
                f.write(f"**Results:** {passed_count}/{total_count} passed ({(passed_count/total_count)*100:.1f}%)\n\n")
                
                # Show sample questions and answers
                for i, test in enumerate(tests[:5], 1):  # Show first 5 tests per category
                    status = "✅ PASS" if test['passed'] else "❌ FAIL"
                    hallucination = " 🧠 HALLUCINATION" if test['is_hallucinating'] else ""
                    
                    f.write(f"#### Test {test['test_id']}: {status}{hallucination}\n\n")
                    f.write(f"**Question:** {test['question']}\n\n")
                    
                    answer = test['response'].get('answer', 'No answer')
                    f.write(f"**Answer:** {answer}\n\n")
                    
                    links = test['response'].get('links', [])
                    if links:
                        f.write(f"**Links:**\n")
                        for link in links:
                            f.write(f"- [{link.get('text', 'Link')}]({link.get('url', '#')})\n")
                        f.write(f"\n")
                    
                    f.write(f"**Response Time:** {test['response_time']:.2f}s\n\n")
                    f.write(f"**Should Find Answer:** {test['should_find_answer']}\n\n")
                    f.write(f"---\n\n")
                
                if len(tests) > 5:
                    f.write(f"*... and {len(tests) - 5} more tests in this category*\n\n")
            
            # Failed tests section
            failed_tests = [r for r in test_results if not r['passed']]
            if failed_tests:
                f.write(f"## Failed Tests Analysis\n\n")
                f.write(f"Total failed tests: {len(failed_tests)}\n\n")
                
                for test in failed_tests[:10]:  # Show first 10 failed tests
                    f.write(f"### Failed Test {test['test_id']}\n\n")
                    f.write(f"**Category:** {test['category']}\n\n")
                    f.write(f"**Question:** {test['question']}\n\n")
                    f.write(f"**Answer:** {test['response'].get('answer', 'No answer')}\n\n")
                    f.write(f"**Hallucination Detected:** {test['is_hallucinating']}\n\n")
                    f.write(f"**Should Find Answer:** {test['should_find_answer']}\n\n")
                    f.write(f"---\n\n")
            
            # Hallucination tests section
            hallucination_tests = [r for r in test_results if r['is_hallucinating']]
            if hallucination_tests:
                f.write(f"## Hallucination Detection\n\n")
                f.write(f"Total hallucinations detected: {len(hallucination_tests)}\n\n")
                
                for test in hallucination_tests[:10]:  # Show first 10 hallucinations
                    f.write(f"### Hallucination Test {test['test_id']}\n\n")
                    f.write(f"**Question:** {test['question']}\n\n")
                    f.write(f"**Answer:** {test['response'].get('answer', 'No answer')}\n\n")
                    f.write(f"---\n\n")
        
        print(f"📝 Markdown report saved to: {filename}")
        return filename

def main():
    """Main function"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Run comprehensive tests")
    parser.add_argument("--url", default="http://localhost:5000/api/", help="API URL")
    parser.add_argument("--tests", type=int, help="Max tests to run")
    
    args = parser.parse_args()
    
    # Generate timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    runner = ComprehensiveTestRunner(args.url)
    test_questions = runner.load_test_suite()
    
    if not test_questions:
        print("❌ No test questions found. Run generate_comprehensive_tests.py first.")
        return 1
    
    print(f"📅 Test run timestamp: {timestamp}")
    results = runner.run_all_tests(test_questions, args.tests)
    runner.print_report(results["statistics"])
    
    # Save detailed markdown report
    markdown_file = runner.save_results_markdown(results, timestamp)
    
    success_rate = results["statistics"]["success_rate"]
    hallucination_rate = results["statistics"]["hallucination_rate"]
    
    if success_rate >= 80 and hallucination_rate <= 10:
        print(f"🎉 EXCELLENT! Success: {success_rate:.1f}%, Hallucination: {hallucination_rate:.1f}%")
        print(f"📄 Detailed report: {markdown_file}")
        return 0
    else:
        print(f"⚠️  NEEDS IMPROVEMENT! Success: {success_rate:.1f}%, Hallucination: {hallucination_rate:.1f}%")
        print(f"📄 Detailed report: {markdown_file}")
        return 1

if __name__ == "__main__":
    exit(main()) 