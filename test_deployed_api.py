#!/usr/bin/env python3
"""
Comprehensive API Testing Script for TDS Virtual TA
Tests 50 diverse questions across different topics
"""

import requests
import json
import time
from datetime import datetime
import sys

# API Configuration
API_BASE_URL = "https://tds-project1-5e44.onrender.com"
API_ENDPOINT = f"{API_BASE_URL}/api/"
HEALTH_ENDPOINT = f"{API_BASE_URL}/api/health"

# Test Questions covering various TDS topics
TEST_QUESTIONS = [
    # Data Science Fundamentals
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of overfitting in machine learning",
    "What are the main types of data visualization?",
    "How do you handle missing data in a dataset?",
    "What is the purpose of cross-validation?",
    
    # Python Programming
    "How do you create a pandas DataFrame?",
    "What is the difference between a list and a tuple in Python?",
    "How do you read a CSV file using pandas?",
    "Explain Python list comprehensions with an example",
    "What are lambda functions in Python?",
    
    # Statistics and Probability
    "What is the central limit theorem?",
    "Explain the difference between correlation and causation",
    "What is a p-value in hypothesis testing?",
    "How do you calculate standard deviation?",
    "What is the difference between mean, median, and mode?",
    
    # Data Visualization
    "How do you create a scatter plot using matplotlib?",
    "What is the best chart type for showing trends over time?",
    "How do you customize colors in a seaborn plot?",
    "What are the principles of good data visualization?",
    "How do you create subplots in matplotlib?",
    
    # Machine Learning Algorithms
    "How does linear regression work?",
    "What is the difference between classification and regression?",
    "Explain the k-means clustering algorithm",
    "What is a decision tree and how does it work?",
    "How do you evaluate a machine learning model?",
    
    # Data Preprocessing
    "How do you normalize data in pandas?",
    "What is feature scaling and why is it important?",
    "How do you encode categorical variables?",
    "What is data cleaning and why is it necessary?",
    "How do you handle outliers in a dataset?",
    
    # SQL and Databases
    "How do you write a basic SQL SELECT query?",
    "What is the difference between INNER JOIN and LEFT JOIN?",
    "How do you group data in SQL?",
    "What is database normalization?",
    "How do you create an index in SQL?",
    
    # Web Scraping and APIs
    "How do you scrape data from a website using Python?",
    "What is an API and how do you use it?",
    "How do you handle HTTP requests in Python?",
    "What are the ethical considerations in web scraping?",
    "How do you parse JSON data in Python?",
    
    # Advanced Topics
    "What is deep learning and how is it different from machine learning?",
    "Explain the concept of neural networks",
    "What is natural language processing?",
    "How do you work with time series data?",
    "What is big data and how do you handle it?",
    
    # Tools and Technologies
    "What is Jupyter Notebook and how do you use it?",
    "How do you install Python packages using pip?",
    "What is version control and why is Git important?",
    "How do you deploy a machine learning model?",
    "What is the difference between Python 2 and Python 3?",
    
    # Practical Applications
    "How do you build a recommendation system?",
    "What are the steps in a typical data science project?",
    "How do you perform A/B testing?",
    "What is customer segmentation and how do you do it?",
    "How do you analyze social media data?"
]

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=30)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Check Passed")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Discourse Topics: {health_data.get('discourse_topics', 0)}")
            print(f"   Q&A Pairs: {health_data.get('discourse_qa_pairs', 0)}")
            print(f"   Course Topics: {health_data.get('course_topics', 0)}")
            print(f"   Code Examples: {health_data.get('course_code_examples', 0)}")
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

def test_question(question, question_num, total_questions):
    """Test a single question"""
    print(f"\n📝 Question {question_num}/{total_questions}: {question[:60]}...")
    
    try:
        # Prepare request
        payload = {"question": question}
        headers = {"Content-Type": "application/json"}
        
        # Send request with timeout
        start_time = time.time()
        response = requests.post(API_ENDPOINT, 
                               json=payload, 
                               headers=headers, 
                               timeout=60)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer provided")
            links = data.get("links", [])
            search_results = data.get("search_results_count", {})
            
            print(f"✅ Success ({response_time:.2f}s)")
            print(f"   Answer Length: {len(answer)} characters")
            print(f"   Links Provided: {len(links)}")
            print(f"   Search Results: {search_results}")
            print(f"   Answer Preview: {answer[:100]}...")
            
            return {
                "success": True,
                "response_time": response_time,
                "answer_length": len(answer),
                "links_count": len(links),
                "search_results": search_results
            }
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return {
                "success": False,
                "status_code": response.status_code,
                "response_time": response_time
            }
            
    except requests.exceptions.Timeout:
        print(f"⏰ Timeout after 60 seconds")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

def run_comprehensive_test():
    """Run comprehensive API testing"""
    print("🚀 Starting Comprehensive TDS Virtual TA API Testing")
    print(f"📍 Testing API at: {API_BASE_URL}")
    print(f"📊 Total Questions: {len(TEST_QUESTIONS)}")
    print("=" * 60)
    
    # Test health endpoint first
    if not test_health_endpoint():
        print("❌ Health check failed. Aborting tests.")
        return
    
    print("\n" + "=" * 60)
    print("🧪 Starting Question Testing...")
    
    # Track results
    results = []
    successful_tests = 0
    total_response_time = 0
    
    # Test each question
    for i, question in enumerate(TEST_QUESTIONS, 1):
        result = test_question(question, i, len(TEST_QUESTIONS))
        results.append({
            "question": question,
            "result": result
        })
        
        if result.get("success"):
            successful_tests += 1
            total_response_time += result.get("response_time", 0)
        
        # Small delay between requests to be respectful
        time.sleep(1)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY REPORT")
    print("=" * 60)
    print(f"✅ Successful Tests: {successful_tests}/{len(TEST_QUESTIONS)}")
    print(f"❌ Failed Tests: {len(TEST_QUESTIONS) - successful_tests}")
    print(f"📈 Success Rate: {(successful_tests/len(TEST_QUESTIONS)*100):.1f}%")
    
    if successful_tests > 0:
        avg_response_time = total_response_time / successful_tests
        print(f"⏱️  Average Response Time: {avg_response_time:.2f} seconds")
    
    # Detailed failure analysis
    failed_tests = [r for r in results if not r["result"].get("success")]
    if failed_tests:
        print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
        for i, failed in enumerate(failed_tests[:5], 1):  # Show first 5 failures
            print(f"   {i}. {failed['question'][:50]}...")
            error = failed['result'].get('error', 'Unknown error')
            print(f"      Error: {error}")
    
    # Performance analysis
    successful_results = [r["result"] for r in results if r["result"].get("success")]
    if successful_results:
        response_times = [r["response_time"] for r in successful_results]
        answer_lengths = [r["answer_length"] for r in successful_results]
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Fastest Response: {min(response_times):.2f}s")
        print(f"   Slowest Response: {max(response_times):.2f}s")
        print(f"   Average Answer Length: {sum(answer_lengths)/len(answer_lengths):.0f} characters")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"api_test_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "test_summary": {
                "total_questions": len(TEST_QUESTIONS),
                "successful_tests": successful_tests,
                "failed_tests": len(TEST_QUESTIONS) - successful_tests,
                "success_rate": successful_tests/len(TEST_QUESTIONS)*100,
                "average_response_time": total_response_time / successful_tests if successful_tests > 0 else 0,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print("\n🎉 Testing Complete!")

if __name__ == "__main__":
    run_comprehensive_test() 