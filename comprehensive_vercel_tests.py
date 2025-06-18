#!/usr/bin/env python3
"""
Comprehensive API Testing for TDS Virtual TA Vercel Deployment
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any

class VercelAPITester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.test_results = []
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'TDS-Virtual-TA-Tester/1.0'
        })

    def run_test(self, test_name: str, method: str, endpoint: str, 
                 payload: Dict = None, expected_status: int = 200,
                 expected_keys: List[str] = None, description: str = ""):
        """Run a single API test and record results"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=30)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=payload, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)  # ms
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = {"raw_text": response.text}
            
            # Check expected keys
            missing_keys = []
            if expected_keys:
                for key in expected_keys:
                    if key not in response_data:
                        missing_keys.append(key)
            
            # Determine test status
            status_ok = response.status_code == expected_status
            keys_ok = len(missing_keys) == 0 if expected_keys else True
            test_passed = status_ok and keys_ok
            
            test_result = {
                'test_name': test_name,
                'description': description,
                'method': method.upper(),
                'endpoint': endpoint,
                'url': url,
                'payload': payload,
                'expected_status': expected_status,
                'actual_status': response.status_code,
                'expected_keys': expected_keys or [],
                'missing_keys': missing_keys,
                'response_data': response_data,
                'response_time_ms': response_time,
                'response_size_bytes': len(response.content),
                'headers': dict(response.headers),
                'test_passed': test_passed,
                'timestamp': datetime.now().isoformat(),
                'error': None
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000, 2)
            
            test_result = {
                'test_name': test_name,
                'description': description,
                'method': method.upper(),
                'endpoint': endpoint,
                'url': f"{self.base_url}{endpoint}",
                'payload': payload,
                'expected_status': expected_status,
                'actual_status': 'ERROR',
                'expected_keys': expected_keys or [],
                'missing_keys': [],
                'response_data': {},
                'response_time_ms': response_time,
                'response_size_bytes': 0,
                'headers': {},
                'test_passed': False,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
        
        self.test_results.append(test_result)
        print(f"{'✅' if test_result['test_passed'] else '❌'} {test_name} - {response_time}ms")
        return test_result

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🧪 Starting comprehensive API tests...")
        
        # Test 1: Health Check
        self.run_test(
            test_name="Health Check",
            method="GET",
            endpoint="/api/health",
            expected_status=200,
            expected_keys=["status", "has_api", "discourse_topics", "course_files"],
            description="Basic health check to verify API is running and data is loaded"
        )
        
        # Test 2: Root Endpoint
        self.run_test(
            test_name="Root Endpoint Info",
            method="GET",
            endpoint="/",
            expected_status=200,
            expected_keys=["message", "status", "endpoints"],
            description="Check root endpoint provides API information"
        )
        
        # Test 3: Simple Question
        self.run_test(
            test_name="Simple Question - Docker",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "What is Docker?"},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test basic question answering about Docker"
        )
        
        # Test 4: Complex Question
        self.run_test(
            test_name="Complex Question - Git vs GitHub",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "What's the difference between Git and GitHub? How do I use them together?"},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test complex multi-part question about Git and GitHub"
        )
        
        # Test 5: Course-specific Question
        self.run_test(
            test_name="Course Question - Python Data Analysis",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "How do I use pandas for data analysis in Python?"},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test course-specific question about Python and pandas"
        )
        
        print("✅ All tests completed!")

    def generate_html_report(self, output_file: str = "vercel_api_test_report.html"):
        """Generate detailed HTML test report"""
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test['test_passed'])
        failed_tests = total_tests - passed_tests
        
        avg_response_time = sum(test['response_time_ms'] for test in self.test_results) / total_tests
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TDS Virtual TA - Vercel API Test Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .test-item {{
            background: white;
            border-radius: 10px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }}
        
        .test-header {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .test-header.passed {{ border-left: 5px solid #16a34a; }}
        .test-header.failed {{ border-left: 5px solid #dc2626; }}
        
        .response-content {{
            padding: 20px;
            background: #f8f9fa;
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 TDS Virtual TA API Test Report</h1>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p>API: {self.base_url}</p>
            <p>Tests: {passed_tests}/{total_tests} passed | Avg Response: {avg_response_time:.1f}ms</p>
        </div>
"""
        
        # Add individual test results
        for test in self.test_results:
            status_class = "passed" if test['test_passed'] else "failed"
            status_text = "✅ PASSED" if test['test_passed'] else "❌ FAILED"
            
            payload_str = json.dumps(test['payload'], indent=2) if test['payload'] else "None"
            response_str = json.dumps(test['response_data'], indent=2)
            
            html_content += f"""
        <div class="test-item">
            <div class="test-header {status_class}">
                <h3>{test['test_name']} - {status_text}</h3>
                <p><strong>Description:</strong> {test['description']}</p>
                <p><strong>Method:</strong> {test['method']} | <strong>Endpoint:</strong> {test['endpoint']}</p>
                <p><strong>Response Time:</strong> {test['response_time_ms']}ms | <strong>Size:</strong> {test['response_size_bytes']} bytes</p>
                <p><strong>Expected Status:</strong> {test['expected_status']} | <strong>Actual Status:</strong> {test['actual_status']}</p>
                {f"<p><strong>Error:</strong> {test['error']}</p>" if test['error'] else ""}
            </div>
            
            <div class="response-content">
<strong>REQUEST PAYLOAD:</strong>
{payload_str}

<strong>RESPONSE DATA:</strong>
{response_str}
            </div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Test report generated: {output_file}")
        return output_file

def main():
    base_url = "https://tds-project1-nh0zbupt1-geek0609s-projects.vercel.app"
    
    print("🚀 Starting TDS Virtual TA API Tests")
    print(f"🔗 Testing: {base_url}")
    print("-" * 50)
    
    tester = VercelAPITester(base_url)
    tester.run_all_tests()
    
    report_file = tester.generate_html_report()
    
    print(f"\n📊 Test Summary:")
    print(f"   Total Tests: {len(tester.test_results)}")
    print(f"   Passed: {sum(1 for test in tester.test_results if test['test_passed'])}")
    print(f"   Failed: {sum(1 for test in tester.test_results if not test['test_passed'])}")
    print(f"   Report: {report_file}")

if __name__ == "__main__":
    main() 