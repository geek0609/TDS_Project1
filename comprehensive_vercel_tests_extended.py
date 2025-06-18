#!/usr/bin/env python3
"""
Comprehensive Vercel API Tests with Image Support for TDS Virtual TA
"""

import requests
import time
import json
import os
from datetime import datetime
from pathlib import Path

class VercelAPITester:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        self.results = []
        self.start_time = time.time()
        
    def log_test(self, name, description, method, endpoint, status_code, 
                 response_time, response_size, expected_status, actual_status,
                 passed, request_payload=None, response_data=None):
        """Log test results"""
        self.results.append({
            'name': name,
            'description': description,
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'response_time': response_time,
            'response_size': response_size,
            'expected_status': expected_status,
            'actual_status': actual_status,
            'passed': passed,
            'request_payload': request_payload,
            'response_data': response_data
        })
    
    def test_health_check(self):
        """Test health endpoint"""
        print("🏥 Testing health check...")
        
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            passed = response.status_code == 200
            response_data = response.json() if response.status_code == 200 else response.text
            
            self.log_test(
                "Health Check",
                "Basic health check to verify API is running and data is loaded",
                "GET", "/api/health",
                response.status_code, response_time, len(response.content),
                200, response.status_code, passed,
                None, response_data
            )
            
            if passed:
                print(f"✅ Health check passed ({response_time:.2f}ms)")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            self.log_test(
                "Health Check", "Basic health check to verify API is running",
                "GET", "/api/health", 0, 0, 0, 200, "Error", False,
                None, str(e)
            )
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        print("🏠 Testing root endpoint...")
        
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            passed = response.status_code == 200
            response_data = response.json() if response.status_code == 200 else response.text
            
            self.log_test(
                "Root Endpoint Info",
                "Check root endpoint provides API information",
                "GET", "/",
                response.status_code, response_time, len(response.content),
                200, response.status_code, passed,
                None, response_data
            )
            
            if passed:
                print(f"✅ Root endpoint passed ({response_time:.2f}ms)")
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            self.log_test(
                "Root Endpoint Info", "Check root endpoint provides API information",
                "GET", "/", 0, 0, 0, 200, "Error", False,
                None, str(e)
            )
    
    def test_simple_question(self):
        """Test simple question"""
        print("📝 Testing simple question...")
        
        payload = {"question": "What is Docker?"}
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/ask",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response_time = (time.time() - start_time) * 1000
            
            passed = response.status_code == 200
            response_data = response.json() if response.status_code == 200 else response.text
            
            self.log_test(
                "Simple Question - Docker",
                "Test basic question answering about Docker",
                "POST", "/api/ask",
                response.status_code, response_time, len(response.content),
                200, response.status_code, passed,
                payload, response_data
            )
            
            if passed:
                print(f"✅ Simple question passed ({response_time:.2f}ms)")
            else:
                print(f"❌ Simple question failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Simple question error: {e}")
            self.log_test(
                "Simple Question - Docker", "Test basic question answering",
                "POST", "/api/ask", 0, 0, 0, 200, "Error", False,
                payload, str(e)
            )
    
    def test_complex_question(self):
        """Test complex question"""
        print("🧠 Testing complex question...")
        
        payload = {"question": "What's the difference between Git and GitHub? How do I use them together?"}
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/ask",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response_time = (time.time() - start_time) * 1000
            
            passed = response.status_code == 200
            response_data = response.json() if response.status_code == 200 else response.text
            
            self.log_test(
                "Complex Question - Git vs GitHub",
                "Test complex multi-part question about Git and GitHub",
                "POST", "/api/ask",
                response.status_code, response_time, len(response.content),
                200, response.status_code, passed,
                payload, response_data
            )
            
            if passed:
                print(f"✅ Complex question passed ({response_time:.2f}ms)")
            else:
                print(f"❌ Complex question failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Complex question error: {e}")
            self.log_test(
                "Complex Question - Git vs GitHub", "Test complex multi-part question",
                "POST", "/api/ask", 0, 0, 0, 200, "Error", False,
                payload, str(e)
            )
    
    def test_course_question(self):
        """Test course-specific question"""
        print("📚 Testing course question...")
        
        payload = {"question": "How do I use pandas for data analysis in Python?"}
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/ask",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response_time = (time.time() - start_time) * 1000
            
            passed = response.status_code == 200
            response_data = response.json() if response.status_code == 200 else response.text
            
            self.log_test(
                "Course Question - Python Data Analysis",
                "Test course-specific question about Python and pandas",
                "POST", "/api/ask",
                response.status_code, response_time, len(response.content),
                200, response.status_code, passed,
                payload, response_data
            )
            
            if passed:
                print(f"✅ Course question passed ({response_time:.2f}ms)")
            else:
                print(f"❌ Course question failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Course question error: {e}")
            self.log_test(
                "Course Question - Python Data Analysis", "Test course-specific question",
                "POST", "/api/ask", 0, 0, 0, 200, "Error", False,
                payload, str(e)
            )
    
    def test_image_question(self):
        """Test question with image"""
        print("🖼️ Testing image question...")
        
        image_path = "project-tds-virtual-ta-q1.webp"
        if not Path(image_path).exists():
            print(f"❌ Image file not found: {image_path}")
            self.log_test(
                "Image Question - GPT Model Choice",
                "Test question with image support",
                "POST", "/api/ask", 0, 0, 0, 200, "Image Not Found", False,
                None, f"Image file not found: {image_path}"
            )
            return
        
        payload = {
            "question": "The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?",
            "image": f"file://{os.path.abspath(image_path)}"
        }
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/ask",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # Longer timeout for image processing
            )
            response_time = (time.time() - start_time) * 1000
            
            passed = response.status_code == 200
            response_data = response.json() if response.status_code == 200 else response.text
            
            self.log_test(
                "Image Question - GPT Model Choice",
                "Test question with image support from YAML test case",
                "POST", "/api/ask",
                response.status_code, response_time, len(response.content),
                200, response.status_code, passed,
                {"question": payload["question"], "image": "project-tds-virtual-ta-q1.webp"}, 
                response_data
            )
            
            if passed:
                print(f"✅ Image question passed ({response_time:.2f}ms)")
                # Verify the response contains image analysis
                if isinstance(response_data, dict) and 'answer' in response_data:
                    answer = response_data['answer']
                    if len(answer) > 100:  # Basic check for substantial response
                        print("✅ Image analysis appears successful")
                    else:
                        print("⚠️ Response seems short for image analysis")
            else:
                print(f"❌ Image question failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Image question error: {e}")
            self.log_test(
                "Image Question - GPT Model Choice", "Test question with image support",
                "POST", "/api/ask", 0, 0, 0, 200, "Error", False,
                payload, str(e)
            )
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting comprehensive Vercel API tests...")
        print(f"🌐 Testing: {self.base_url}")
        print("=" * 60)
        
        self.test_health_check()
        self.test_root_endpoint()
        self.test_simple_question()
        self.test_complex_question()
        self.test_course_question()
        self.test_image_question()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed!")
        
        # Calculate stats
        passed_tests = sum(1 for r in self.results if r['passed'])
        total_tests = len(self.results)
        avg_response_time = sum(r['response_time'] for r in self.results if r['response_time'] > 0) / max(1, sum(1 for r in self.results if r['response_time'] > 0))
        
        print(f"📊 Results: {passed_tests}/{total_tests} passed")
        print(f"⏱️ Average response time: {avg_response_time:.1f}ms")
        
        return self.results
    
    def generate_html_report(self, output_file="vercel_api_test_report.html"):
        """Generate HTML report"""
        passed_tests = sum(1 for r in self.results if r['passed'])
        total_tests = len(self.results)
        avg_response_time = sum(r['response_time'] for r in self.results if r['response_time'] > 0) / max(1, sum(1 for r in self.results if r['response_time'] > 0))
        
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TDS Virtual TA - Enhanced Vercel API Test Report</title>
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
        
        .image-indicator {{
            background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
            color: white;
            padding: 5px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            display: inline-block;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 TDS Virtual TA Enhanced API Test Report</h1>
            <p>Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
            <p>API: {self.base_url}</p>
            <p>Tests: {passed_tests}/{total_tests} passed | Avg Response: {avg_response_time:.1f}ms</p>
            <p style="margin-top: 10px; color: #fbbf24;">✨ Now with Image Processing Support ✨</p>
        </div>
'''
        
        for result in self.results:
            status_class = "passed" if result['passed'] else "failed"
            status_icon = "✅ PASSED" if result['passed'] else "❌ FAILED"
            
            # Add image indicator for image tests
            image_indicator = ""
            if "image" in str(result.get('request_payload', '')).lower():
                image_indicator = '<span class="image-indicator">🖼️ IMAGE TEST</span>'
            
            request_payload = json.dumps(result['request_payload'], indent=2) if result['request_payload'] else "None"
            response_data = json.dumps(result['response_data'], indent=2) if isinstance(result['response_data'], dict) else str(result['response_data'])
            
            html_content += f'''
        <div class="test-item">
            <div class="test-header {status_class}">
                <h3>{result['name']} - {status_icon}{image_indicator}</h3>
                <p><strong>Description:</strong> {result['description']}</p>
                <p><strong>Method:</strong> {result['method']} | <strong>Endpoint:</strong> {result['endpoint']}</p>
                <p><strong>Response Time:</strong> {result['response_time']:.2f}ms | <strong>Size:</strong> {result['response_size']} bytes</p>
                <p><strong>Expected Status:</strong> {result['expected_status']} | <strong>Actual Status:</strong> {result['actual_status']}</p>
                
            </div>
            
            <div class="response-content">
<strong>REQUEST PAYLOAD:</strong>
{request_payload}

<strong>RESPONSE DATA:</strong>
{response_data}
            </div>
        </div>
'''
        
        html_content += '''
    </div>
</body>
</html>
'''
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML report generated: {output_file}")

def main():
    """Main function"""
    # Use the deployed Vercel URL
    base_url = "https://tds-project1-89mdlg7gv-geek0609s-projects.vercel.app"
    
    tester = VercelAPITester(base_url)
    results = tester.run_all_tests()
    tester.generate_html_report()

if __name__ == "__main__":
    main() 