#!/usr/bin/env python3
"""
Extended Comprehensive API Testing for TDS Virtual TA Vercel Deployment
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any

class ExtendedVercelAPITester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.test_results = []
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'TDS-Virtual-TA-Extended-Tester/1.0'
        })

    def run_test(self, test_name: str, method: str, endpoint: str, 
                 payload: Dict = None, expected_status: int = 200,
                 expected_keys: List[str] = None, description: str = "",
                 check_content: Dict = None):
        """Run a single API test and record results"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=60)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=payload, timeout=60)
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
            
            # Check content expectations
            content_checks = []
            if check_content:
                for field, expected_value in check_content.items():
                    actual_value = response_data.get(field)
                    if isinstance(expected_value, str):
                        check_passed = expected_value.lower() in str(actual_value).lower()
                    elif isinstance(expected_value, (int, float)):
                        check_passed = actual_value >= expected_value
                    else:
                        check_passed = actual_value == expected_value
                    
                    content_checks.append({
                        'field': field,
                        'expected': expected_value,
                        'actual': actual_value,
                        'passed': check_passed
                    })
            
            # Determine test status
            status_ok = response.status_code == expected_status
            keys_ok = len(missing_keys) == 0 if expected_keys else True
            content_ok = all(check['passed'] for check in content_checks) if content_checks else True
            test_passed = status_ok and keys_ok and content_ok
            
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
                'content_checks': content_checks,
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
                'content_checks': [],
                'response_data': {},
                'response_time_ms': response_time,
                'response_size_bytes': 0,
                'headers': {},
                'test_passed': False,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
        
        self.test_results.append(test_result)
        status_icon = '✅' if test_result['test_passed'] else '❌'
        print(f"{status_icon} {test_name} - {response_time}ms")
        return test_result

    def run_all_tests(self):
        """Run extended comprehensive test suite"""
        print("🧪 Starting EXTENDED comprehensive API tests...")
        
        # Test 1: Health Check
        self.run_test(
            test_name="Health Check",
            method="GET",
            endpoint="/api/health",
            expected_status=200,
            expected_keys=["status", "has_api", "discourse_topics", "course_files"],
            check_content={"status": "healthy", "discourse_topics": 100, "course_files": 100},
            description="Basic health check to verify API is running and data is loaded"
        )
        
        # Test 2: Root Endpoint
        self.run_test(
            test_name="Root Endpoint Info",
            method="GET",
            endpoint="/",
            expected_status=200,
            expected_keys=["message", "status", "endpoints"],
            check_content={"status": "running"},
            description="Check root endpoint provides API information"
        )
        
        # Test 3: Docker Question
        self.run_test(
            test_name="Question: What is Docker?",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "What is Docker?"},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test basic question about Docker containerization technology"
        )
        
        # Test 4: Git vs GitHub Question
        self.run_test(
            test_name="Question: Git vs GitHub",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "What's the difference between Git and GitHub? How do I use them together?"},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test complex multi-part question about version control"
        )
        
        # Test 5: Python Data Analysis
        self.run_test(
            test_name="Question: Python Data Analysis",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "How do I use pandas for data analysis in Python? Show me examples."},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test course-specific question about Python and data analysis"
        )
        
        # Test 6: Assignment Question
        self.run_test(
            test_name="Question: GA4 Assignment",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "Can you help me understand the GA4 assignment requirements and submission process?"},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test assignment-related question"
        )
        
        # Test 7: Kubernetes Question
        self.run_test(
            test_name="Question: Kubernetes",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "What is Kubernetes and how does it relate to Docker containers?"},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test advanced containerization question"
        )
        
        # Test 8: Search Endpoint - Docker
        self.run_test(
            test_name="Search: Docker containers",
            method="POST",
            endpoint="/api/search",
            payload={"query": "Docker container deployment", "top_k": 5},
            expected_status=200,
            expected_keys=["query", "results"],
            description="Test search functionality for Docker-related content"
        )
        
        # Test 9: Search Endpoint - Python
        self.run_test(
            test_name="Search: Python programming",
            method="POST",
            endpoint="/api/search",
            payload={"query": "Python data science pandas", "top_k": 10},
            expected_status=200,
            expected_keys=["query", "results"],
            description="Test search functionality for Python-related content"
        )
        
        # Test 10: Edge Case - Empty Question
        self.run_test(
            test_name="Edge Case: Empty Question",
            method="POST",
            endpoint="/api/ask",
            payload={"question": ""},
            expected_status=400,
            expected_keys=["error"],
            description="Test error handling for empty question"
        )
        
        # Test 11: Edge Case - Missing Question Field
        self.run_test(
            test_name="Edge Case: Missing Question Field",
            method="POST",
            endpoint="/api/ask",
            payload={},
            expected_status=400,
            expected_keys=["error"],
            description="Test error handling for missing question field"
        )
        
        # Test 12: Edge Case - Very Long Question
        long_question = "This is an extremely long question about Docker, Kubernetes, containerization, microservices architecture, CI/CD pipelines, DevOps practices, cloud computing, infrastructure as code, monitoring, logging, security best practices, and deployment strategies that goes on and on with many technical details about container orchestration, service mesh, load balancing, auto-scaling, and distributed systems. " * 5
        self.run_test(
            test_name="Edge Case: Very Long Question",
            method="POST",
            endpoint="/api/ask",
            payload={"question": long_question},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test handling of extremely long questions (2000+ characters)"
        )
        
        # Test 13: Special Characters Question
        self.run_test(
            test_name="Edge Case: Special Characters",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "What about Docker & containers? How do they work with 'quotation marks' and symbols like @#$%?"},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test handling of questions with special characters and symbols"
        )
        
        # Test 14: Non-English Question
        self.run_test(
            test_name="Edge Case: Non-English Question",
            method="POST",
            endpoint="/api/ask",
            payload={"question": "¿Qué es Docker? Explain in simple terms."},
            expected_status=200,
            expected_keys=["answer", "links", "search_results_count"],
            description="Test handling of mixed-language questions"
        )
        
        # Test 15: Performance Test - Rapid Fire Questions
        print("🚀 Running rapid-fire performance tests...")
        for i in range(3):
            self.run_test(
                test_name=f"Performance Test {i+1}/3",
                method="POST",
                endpoint="/api/ask",
                payload={"question": f"Tell me about data science tools and methodologies - rapid test {i+1}"},
                expected_status=200,
                expected_keys=["answer", "links", "search_results_count"],
                description=f"Performance test - rapid request {i+1} of 3 for load testing"
            )
        
        print("✅ All extended tests completed!")

    def generate_comprehensive_html_report(self, output_file: str = "extended_vercel_test_report.html"):
        """Generate comprehensive HTML test report with detailed analysis"""
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test['test_passed'])
        failed_tests = total_tests - passed_tests
        
        avg_response_time = sum(test['response_time_ms'] for test in self.test_results) / total_tests
        min_response_time = min(test['response_time_ms'] for test in self.test_results)
        max_response_time = max(test['response_time_ms'] for test in self.test_results)
        
        total_data_transferred = sum(test['response_size_bytes'] for test in self.test_results)
        
        # Categorize tests
        question_tests = [t for t in self.test_results if 'Question:' in t['test_name']]
        search_tests = [t for t in self.test_results if 'Search:' in t['test_name']]
        edge_case_tests = [t for t in self.test_results if 'Edge Case:' in t['test_name']]
        performance_tests = [t for t in self.test_results if 'Performance Test' in t['test_name']]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TDS Virtual TA - Extended API Test Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
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
        
        .summary {{
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            border: 1px solid #e9ecef;
        }}
        
        .summary-card h3 {{
            font-size: 2.2rem;
            margin-bottom: 8px;
            font-weight: bold;
        }}
        
        .summary-card.passed h3 {{ color: #16a34a; }}
        .summary-card.failed h3 {{ color: #dc2626; }}
        .summary-card.info h3 {{ color: #2563eb; }}
        .summary-card.warning h3 {{ color: #d97706; }}
        
        .category-section {{
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 12px;
            border: 1px solid #e9ecef;
        }}
        
        .category-section h2 {{
            color: #374151;
            margin-bottom: 15px;
            font-size: 1.5rem;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }}
        
        .test-item {{
            background: white;
            border-radius: 10px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border: 1px solid #e9ecef;
        }}
        
        .test-header {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .test-header:hover {{
            background: #e9ecef;
        }}
        
        .test-header.passed {{ border-left: 5px solid #16a34a; }}
        .test-header.failed {{ border-left: 5px solid #dc2626; }}
        
        .test-header h3 {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 1.2rem;
        }}
        
        .test-status {{
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
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
        
        .test-metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 10px 0;
        }}
        
        .metric {{
            background: #e0e7ff;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        .response-content {{
            padding: 20px;
            background: #f8f9fa;
            font-family: 'Consolas', 'Monaco', monospace;
            white-space: pre-wrap;
            font-size: 0.85rem;
            max-height: 500px;
            overflow-y: auto;
            border-top: 1px solid #e9ecef;
        }}
        
        .content-checks {{
            margin: 15px 0;
            padding: 15px;
            background: #f0f9ff;
            border-radius: 8px;
            border: 1px solid #bae6fd;
        }}
        
        .content-check {{
            margin: 5px 0;
            padding: 5px;
            font-size: 0.9rem;
        }}
        
        .content-check.passed {{ color: #16a34a; }}
        .content-check.failed {{ color: #dc2626; }}
        
        .api-info {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            border: 1px solid #e9ecef;
        }}
        
        .api-url {{
            background: #f1f5f9;
            padding: 12px;
            border-radius: 8px;
            font-family: monospace;
            word-break: break-all;
            font-size: 0.9rem;
        }}
        
        .expand-btn {{
            background: none;
            border: none;
            color: #6b7280;
            cursor: pointer;
            font-size: 0.9rem;
            margin-top: 10px;
        }}
        
        .expand-btn:hover {{ color: #374151; }}
        
        .test-details {{
            display: none;
        }}
        
        .test-details.expanded {{
            display: block;
        }}
        
        @media (max-width: 768px) {{
            .summary-grid {{ grid-template-columns: 1fr 1fr; }}
            .test-metrics {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 TDS Virtual TA - Extended API Test Report</h1>
            <p>Comprehensive testing with detailed analysis and performance metrics</p>
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
                    <p>Avg Response</p>
                </div>
                <div class="summary-card warning">
                    <h3>{min_response_time:.1f}ms</h3>
                    <p>Fastest Response</p>
                </div>
                <div class="summary-card warning">
                    <h3>{max_response_time:.1f}ms</h3>
                    <p>Slowest Response</p>
                </div>
                <div class="summary-card info">
                    <h3>{total_data_transferred:,}</h3>
                    <p>Total Bytes</p>
                </div>
                <div class="summary-card info">
                    <h3>{(passed_tests/total_tests*100):.1f}%</h3>
                    <p>Success Rate</p>
                </div>
            </div>
            
            <div class="api-info">
                <h3>🔗 API Endpoint Under Test</h3>
                <div class="api-url">{self.base_url}</div>
            </div>
        </div>
"""
        
        # Add test categories
        categories = [
            ("🏥 System Health & Info Tests", [t for t in self.test_results if t['test_name'] in ['Health Check', 'Root Endpoint Info']]),
            ("❓ Question & Answer Tests", question_tests),
            ("🔍 Search Functionality Tests", search_tests),
            ("⚠️ Edge Case & Error Handling Tests", edge_case_tests),
            ("🚀 Performance & Load Tests", performance_tests),
        ]
        
        for category_name, category_tests in categories:
            if not category_tests:
                continue
                
            category_passed = sum(1 for t in category_tests if t['test_passed'])
            category_total = len(category_tests)
            
            html_content += f"""
        <div class="category-section">
            <h2>{category_name} ({category_passed}/{category_total} passed)</h2>
"""
            
            for i, test in enumerate(category_tests):
                test_index = self.test_results.index(test)
                status_class = "passed" if test['test_passed'] else "failed"
                status_text = "✅ PASSED" if test['test_passed'] else "❌ FAILED"
                
                # Content checks display
                content_checks_html = ""
                if test['content_checks']:
                    content_checks_html = '<div class="content-checks"><strong>Content Validations:</strong><br>'
                    for check in test['content_checks']:
                        check_class = "passed" if check['passed'] else "failed"
                        icon = "✅" if check['passed'] else "❌"
                        content_checks_html += f'<div class="content-check {check_class}">{icon} {check["field"]}: Expected {check["expected"]}, Got {check["actual"]}</div>'
                    content_checks_html += '</div>'
                
                payload_str = json.dumps(test['payload'], indent=2) if test['payload'] else "None"
                response_str = json.dumps(test['response_data'], indent=2)
                
                html_content += f"""
            <div class="test-item">
                <div class="test-header {status_class}" onclick="toggleTest({test_index})">
                    <h3>
                        <span>{test['test_name']}</span>
                        <span class="test-status {status_class}">{status_text}</span>
                    </h3>
                    <div class="test-metrics">
                        <div class="metric">⏱️ {test['response_time_ms']}ms</div>
                        <div class="metric">📦 {test['response_size_bytes']:,} bytes</div>
                        <div class="metric">🔗 {test['method']} {test['endpoint']}</div>
                        <div class="metric">📊 Status: {test['actual_status']}</div>
                    </div>
                    <p><strong>Description:</strong> {test['description']}</p>
                    {content_checks_html}
                    {f'<p style="color: #dc2626;"><strong>Error:</strong> {test["error"]}</p>' if test['error'] else ''}
                    <button class="expand-btn">▼ Click to view detailed request/response data</button>
                </div>
                
                <div class="test-details" id="test-{test_index}">
                    <div class="response-content">
<strong>🔗 REQUEST URL:</strong>
{test['url']}

<strong>📤 REQUEST METHOD:</strong>
{test['method']}

<strong>📋 REQUEST PAYLOAD:</strong>
{payload_str}

<strong>📥 RESPONSE STATUS:</strong>
{test['actual_status']} (Expected: {test['expected_status']})

<strong>📊 RESPONSE HEADERS:</strong>
{json.dumps(test['headers'], indent=2)}

<strong>📄 RESPONSE DATA:</strong>
{response_str}
                    </div>
                </div>
            </div>
"""
            
            html_content += '</div>'
        
        html_content += """
    </div>
    
    <script>
        function toggleTest(index) {
            const testDetails = document.getElementById('test-' + index);
            const expandBtn = testDetails.previousElementSibling.querySelector('.expand-btn');
            
            if (testDetails.classList.contains('expanded')) {
                testDetails.classList.remove('expanded');
                expandBtn.textContent = '▼ Click to view detailed request/response data';
            } else {
                testDetails.classList.add('expanded');
                expandBtn.textContent = '▲ Click to hide detailed request/response data';
            }
        }
        
        // Auto-expand failed tests
        document.addEventListener('DOMContentLoaded', function() {
            const failedTests = document.querySelectorAll('.test-header.failed');
            failedTests.forEach(header => {
                const testIndex = Array.from(document.querySelectorAll('.test-header')).indexOf(header);
                toggleTest(testIndex);
            });
            
            // Add click handlers to all test headers
            document.querySelectorAll('.test-header').forEach((header, index) => {
                header.addEventListener('click', () => toggleTest(index));
            });
        });
    </script>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Extended test report generated: {output_file}")
        return output_file

def main():
    base_url = "https://tds-project1-nh0zbupt1-geek0609s-projects.vercel.app"
    
    print("🚀 Starting EXTENDED TDS Virtual TA API Tests")
    print(f"🔗 Testing: {base_url}")
    print("=" * 60)
    
    tester = ExtendedVercelAPITester(base_url)
    tester.run_all_tests()
    
    report_file = tester.generate_comprehensive_html_report()
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL TEST SUMMARY:")
    print(f"   📋 Total Tests: {len(tester.test_results)}")
    print(f"   ✅ Passed: {sum(1 for test in tester.test_results if test['test_passed'])}")
    print(f"   ❌ Failed: {sum(1 for test in tester.test_results if not test['test_passed'])}")
    print(f"   📄 Report: {report_file}")
    print(f"   🌐 API: {base_url}")

if __name__ == "__main__":
    main() 