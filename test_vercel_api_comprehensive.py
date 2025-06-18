#!/usr/bin/env python3
"""
Comprehensive API Testing for TDS Virtual TA Vercel Deployment
Generates detailed HTML test report with expected vs actual results
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
        
        # Test 6: Assignment Question
        self.run_test(
            test_name="Assignment Question - GA4",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "Can you help me understand the GA4 assignment requirements?"},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test assignment-related question about GA4"
        )
        
        # Test 7: Search Endpoint
        self.run_test(
            test_name="Search Functionality",
            method="POST",
            endpoint="/api/search",
            payload={"query": "Docker container", "top_k": 5},
            expected_status=200,
            expected_keys=["query", "results"],
            description="Test search endpoint functionality"
        )
        
        # Test 8: Edge Case - Empty Question
        self.run_test(
            test_name="Edge Case - Empty Question",
            method="POST",
            endpoint="/api/ask",
            payload={"question": ""},
            expected_status=400,
            expected_keys=["error"],
            description="Test error handling for empty question"
        )
        
        # Test 9: Edge Case - Very Long Question
        long_question = "This is a very long question about Docker and containerization technology that goes on and on and includes many details about microservices, orchestration, Kubernetes, container registries, image building, dockerfile creation, multi-stage builds, security scanning, vulnerability management, and deployment strategies. " * 3
        self.run_test(
            test_name="Edge Case - Very Long Question",
            method="POST",
            endpoint="/api/ask",
            payload={"question": long_question},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test handling of very long questions"
        )
        
        # Test 10: Performance Test - Multiple Requests
        start_time = time.time()
        for i in range(3):
            self.run_test(
                test_name=f"Performance Test {i+1}/3",
                method="POST",
                endpoint="/api/ask",
                payload={"question": f"Tell me about data science tools - request {i+1}"},
                expected_status=200,
                expected_keys=["answer", "links", "search_results_count"],
                description=f"Performance test - batch request {i+1} of 3"
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
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
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
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .summary {{
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }}
        
        .summary-card h3 {{
            font-size: 2rem;
            margin-bottom: 5px;
        }}
        
        .summary-card.passed h3 {{
            color: #16a34a;
        }}
        
        .summary-card.failed h3 {{
            color: #dc2626;
        }}
        
        .summary-card.info h3 {{
            color: #2563eb;
        }}
        
        .api-info {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        
        .api-info h3 {{
            color: #374151;
            margin-bottom: 10px;
        }}
        
        .api-url {{
            background: #f1f5f9;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            word-break: break-all;
        }}
        
        .test-results {{
            padding: 30px;
        }}
        
        .test-results h2 {{
            color: #374151;
            margin-bottom: 20px;
            font-size: 1.8rem;
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
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        
        .test-header:hover {{
            background: #e9ecef;
        }}
        
        .test-header.passed {{
            border-left: 5px solid #16a34a;
        }}
        
        .test-header.failed {{
            border-left: 5px solid #dc2626;
        }}
        
        .test-header h3 {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }}
        
        .test-status {{
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        .test-status.passed {{
            background: #dcfce7;
            color: #16a34a;
        }}
        
        .test-status.failed {{
            background: #fee2e2;
            color: #dc2626;
        }}
        
        .test-details {{
            padding: 20px;
            display: none;
        }}
        
        .test-details.expanded {{
            display: block;
        }}
        
        .detail-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .detail-section {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }}
        
        .detail-section h4 {{
            color: #374151;
            margin-bottom: 10px;
            font-size: 1rem;
        }}
        
        .detail-content {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.9rem;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .response-section {{
            grid-column: 1 / -1;
        }}
        
        .response-content {{
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .metrics {{
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
        }}
        
        .metric {{
            background: #e0e7ff;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 0.9rem;
        }}
        
        .expand-btn {{
            background: none;
            border: none;
            color: #6b7280;
            cursor: pointer;
            font-size: 0.9rem;
        }}
        
        .expand-btn:hover {{
            color: #374151;
        }}
        
        @media (max-width: 768px) {{
            .detail-grid {{
                grid-template-columns: 1fr;
            }}
            
            .summary-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 TDS Virtual TA API Test Report</h1>
            <p>Comprehensive testing results for Vercel deployment</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="summary">
            <div class="summary-grid">
                <div class="summary-card passed">
                    <h3>{passed_tests}</h3>
                    <p>Tests Passed</p>
                </div>
                <div class="summary-card failed">
                    <h3>{failed_tests}</h3>
                    <p>Tests Failed</p>
                </div>
                <div class="summary-card info">
                    <h3>{total_tests}</h3>
                    <p>Total Tests</p>
                </div>
                <div class="summary-card info">
                    <h3>{avg_response_time:.1f}ms</h3>
                    <p>Avg Response Time</p>
                </div>
            </div>
            
            <div class="api-info">
                <h3>🔗 API Endpoint</h3>
                <div class="api-url">{self.base_url}</div>
            </div>
        </div>
        
        <div class="test-results">
            <h2>📋 Detailed Test Results</h2>
"""
        
        # Add individual test results
        for i, test in enumerate(self.test_results):
            status_class = "passed" if test['test_passed'] else "failed"
            status_text = "✅ PASSED" if test['test_passed'] else "❌ FAILED"
            
            # Format expected vs actual
            expected_info = f"Status: {test['expected_status']}"
            if test['expected_keys']:
                expected_info += f", Keys: {', '.join(test['expected_keys'])}"
            
            actual_info = f"Status: {test['actual_status']}"
            if test['missing_keys']:
                actual_info += f", Missing Keys: {', '.join(test['missing_keys'])}"
            
            # Format payload
            payload_str = json.dumps(test['payload'], indent=2) if test['payload'] else "None"
            
            # Format response
            response_str = json.dumps(test['response_data'], indent=2)
            
            # Format headers
            headers_str = json.dumps(test['headers'], indent=2)
            
            html_content += f"""
            <div class="test-item">
                <div class="test-header {status_class}" onclick="toggleTest({i})">
                    <h3>
                        <span>{test['test_name']}</span>
                        <span class="test-status {status_class}">{status_text}</span>
                    </h3>
                    <div class="metrics">
                        <div class="metric">⏱️ {test['response_time_ms']}ms</div>
                        <div class="metric">📦 {test['response_size_bytes']} bytes</div>
                        <div class="metric">🔗 {test['method']} {test['endpoint']}</div>
                    </div>
                    <p>{test['description']}</p>
                    <button class="expand-btn">Click to expand details ▼</button>
                </div>
                
                <div class="test-details" id="test-{i}">
                    <div class="detail-grid">
                        <div class="detail-section">
                            <h4>📤 Request Details</h4>
                            <div class="detail-content">URL: {test['url']}
Method: {test['method']}
Payload: {payload_str}</div>
                        </div>
                        
                        <div class="detail-section">
                            <h4>✅ Expected vs 📥 Actual</h4>
                            <div class="detail-content">Expected: {expected_info}
Actual: {actual_info}
                            
Error: {test['error'] or 'None'}</div>
                        </div>
                        
                        <div class="detail-section response-section">
                            <h4>📋 Response Data</h4>
                            <div class="detail-content response-content">{response_str}</div>
                        </div>
                        
                        <div class="detail-section response-section">
                            <h4>📑 Response Headers</h4>
                            <div class="detail-content response-content">{headers_str}</div>
                        </div>
                    </div>
                </div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
    
    <script>
        function toggleTest(index) {
            const testDetails = document.getElementById('test-' + index);
            const expandBtn = testDetails.previousElementSibling.querySelector('.expand-btn');
            
            if (testDetails.classList.contains('expanded')) {
                testDetails.classList.remove('expanded');
                expandBtn.textContent = 'Click to expand details ▼';
            } else {
                testDetails.classList.add('expanded');
                expandBtn.textContent = 'Click to collapse details ▲';
            }
        }
        
        // Auto-expand failed tests
        document.addEventListener('DOMContentLoaded', function() {
            const failedTests = document.querySelectorAll('.test-header.failed');
            failedTests.forEach((header, index) => {
                const testIndex = Array.from(document.querySelectorAll('.test-header')).indexOf(header);
                toggleTest(testIndex);
            });
        });
    </script>
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