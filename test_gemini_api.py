#!/usr/bin/env python3
"""
Test script for the Enhanced TDS Virtual TA using Gemini 2.5 Flash with embeddings.
"""

import requests
import json
import time


API_URL = "http://localhost:5000/api/"
SEARCH_URL = "http://localhost:5000/api/search"
HEALTH_URL = "http://localhost:5000/api/health"
STATS_URL = "http://localhost:5000/api/stats"

def test_question(question: str, description: str = ""):
    """Test a question with the Gemini-powered Virtual TA API"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, json={"question": question})
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 GEMINI ANSWER ({end_time - start_time:.2f}s):")
            print(f"{data['answer']}")
            
            if data.get('links'):
                print(f"\n🔗 RELEVANT LINKS:")
                for i, link in enumerate(data['links'], 1):
                    print(f"  {i}. {link['text']}")
                    print(f"     {link['url']}")
            
            if data.get('search_results_count'):
                counts = data['search_results_count']
                print(f"\n📊 SEARCH RESULTS:")
                print(f"  • Discourse topics: {counts['discourse_topics']}")
                print(f"  • Q&A pairs: {counts['qa_pairs']}")
                print(f"  • Course topics: {counts['course_topics']}")
                print(f"  • Code examples: {counts['code_examples']}")
        else:
            print(f"❌ ERROR: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_semantic_search(query: str, description: str = ""):
    """Test semantic search endpoint"""
    print(f"\n{'='*80}")
    print(f"SEMANTIC SEARCH TEST: {description}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    try:
        response = requests.post(SEARCH_URL, json={"query": query, "top_k": 3})
        
        if response.status_code == 200:
            data = response.json()
            results = data['results']
            
            print(f"🔍 SEARCH RESULTS:")
            
            if results['qa_pairs']:
                print(f"\n💬 Q&A PAIRS:")
                for i, qa in enumerate(results['qa_pairs'], 1):
                    print(f"  {i}. Q: {qa['question'][:100]}...")
                    print(f"     A: {qa['answer'][:100]}...")
                    print(f"     Similarity: {qa.get('similarity', 0):.3f}")
            
            if results['course_topics']:
                print(f"\n📖 COURSE TOPICS:")
                for i, topic in enumerate(results['course_topics'], 1):
                    print(f"  {i}. {topic['title']}")
                    print(f"     Category: {topic.get('category', 'N/A')}")
                    print(f"     Similarity: {topic.get('similarity', 0):.3f}")
            
            if results['code_examples']:
                print(f"\n💻 CODE EXAMPLES:")
                for i, code in enumerate(results['code_examples'], 1):
                    print(f"  {i}. Language: {code.get('language', 'N/A')}")
                    print(f"     Code: {code.get('code', '')[:50]}...")
                    print(f"     Similarity: {code.get('similarity', 0):.3f}")
        else:
            print(f"❌ ERROR: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_health_and_stats():
    """Test health and stats endpoints"""
    print(f"\n{'='*80}")
    print("TESTING HEALTH AND STATS ENDPOINTS")
    print(f"{'='*80}")
    
    # Test health endpoint
    try:
        response = requests.get(HEALTH_URL)
        if response.status_code == 200:
            data = response.json()
            print("🏥 HEALTH CHECK:")
            print(f"  Status: {data['status']}")
            print(f"  Model: {data['model']}")
            print(f"  Discourse topics: {data['discourse_topics']}")
            print(f"  Discourse Q&A pairs: {data['discourse_qa_pairs']}")
            print(f"  Course topics: {data['course_topics']}")
            print(f"  Course code examples: {data['course_code_examples']}")
            
            embeddings = data['embeddings_ready']
            print(f"  Embeddings ready:")
            print(f"    • Discourse: {'✅' if embeddings['discourse'] else '❌'}")
            print(f"    • Q&A pairs: {'✅' if embeddings['qa_pairs'] else '❌'}")
            print(f"    • Course: {'✅' if embeddings['course'] else '❌'}")
            print(f"    • Code: {'✅' if embeddings['code'] else '❌'}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test stats endpoint
    try:
        response = requests.get(STATS_URL)
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 STATISTICS:")
            print(f"  AI Model: {data.get('ai_model', 'N/A')}")
            print(f"  Embedding Model: {data.get('embedding_model', 'N/A')}")
            print(f"  Combined total topics: {data.get('combined_total_topics', 'N/A')}")
            print(f"  Combined total content: {data.get('combined_total_content', 'N/A')}")
            
            discourse = data.get('discourse', {})
            print(f"  Discourse:")
            print(f"    • Topics: {discourse.get('total_topics', 'N/A')}")
            print(f"    • Q&A pairs: {discourse.get('total_qa_pairs', 'N/A')}")
            
            course = data.get('course_content', {})
            print(f"  Course content:")
            print(f"    • Topics: {course.get('total_topics', 'N/A')}")
            print(f"    • Code examples: {course.get('total_code_examples', 'N/A')}")
            print(f"    • Tools mentioned: {course.get('tools_mentioned', 'N/A')}")
        else:
            print(f"❌ Stats check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats check error: {e}")

def main():
    print("🚀 Enhanced TDS Virtual TA - Gemini API Testing")
    print("Testing semantic search, AI-powered answers, and various question types...")
    
    # Test health and stats first
    test_health_and_stats()
    
    # Test semantic search
    print(f"\n{'='*80}")
    print("SEMANTIC SEARCH TESTS")
    print(f"{'='*80}")
    
    search_tests = [
        ("prompt engineering", "LLM concept search"),
        ("vector database", "Database concept search"),
        ("python scraping", "Technical implementation search"),
        ("project submission", "Discourse discussion search"),
    ]
    
    for query, description in search_tests:
        test_semantic_search(query, description)
    
    # Test different types of questions with Gemini
    print(f"\n{'='*80}")
    print("GEMINI-POWERED QUESTION ANSWERING TESTS")
    print(f"{'='*80}")
    
    test_questions = [
        # Discourse-based questions
        ("How do I submit my TDS project?", "Project submission - Discourse"),
        ("What is the deadline for GA1?", "Assignment deadline - Discourse"),
        ("I'm getting a Docker error when running my code", "Technical issue - Discourse"),
        ("My assignment score is showing 0", "Score issue - Discourse"),
        
        # Course content questions
        ("What is prompt engineering and how do I use it?", "Course content - LLM topic"),
        ("Explain vector databases and their use cases", "Course content - Database topic"),
        ("How do I scrape data with Python? Show me examples", "Course content - Web scraping with code"),
        ("What tools are available for data visualization?", "Course content - Visualization tools"),
        ("How do I use SQLite for data analysis?", "Course content - Database analysis"),
        
        # Mixed/complex questions
        ("What's the difference between ChromaDB and LanceDB for vector storage?", "Complex technical comparison"),
        ("How do I deploy my TDS project using Docker and what are common issues?", "Mixed deployment question"),
        ("Show me how to use LLMs for text analysis with code examples", "Code-focused question"),
    ]
    
    for question, description in test_questions:
        test_question(question, description)
    
    print(f"\n{'='*80}")
    print("🎉 Testing completed!")
    print("The Enhanced Virtual TA with Gemini 2.5 Flash is ready to help students!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 