import requests
import json


API_URL = "http://localhost:5000/api/"

def test_question(question: str, description: str = ""):
    """Test a question with the Virtual TA API"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(API_URL, json={"question": question})
        
        if response.status_code == 200:
            data = response.json()
            print(f"ANSWER: {data['answer']}")
            
            if data.get('links'):
                print(f"\nRELEVANT LINKS:")
                for i, link in enumerate(data['links'], 1):
                    print(f"  {i}. {link['text']}")
                    print(f"     {link['url']}")
        else:
            print(f"ERROR: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"ERROR: {e}")


def test_health_and_stats():
    """Test health and stats endpoints"""
    print(f"\n{'='*60}")
    print("TESTING HEALTH AND STATS ENDPOINTS")
    print(f"{'='*60}")
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:5000/api/health")
        if response.status_code == 200:
            data = response.json()
            print("HEALTH CHECK:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"Health check failed: {response.status_code}")
    except Exception as e:
        print(f"Health check error: {e}")
    
    # Test stats endpoint
    try:
        response = requests.get("http://localhost:5000/api/stats")
        if response.status_code == 200:
            data = response.json()
            print("\nSTATISTICS:")
            print(f"  Discourse topics: {data.get('discourse', {}).get('total_topics', 'N/A')}")
            print(f"  Discourse Q&A pairs: {data.get('discourse', {}).get('total_qa_pairs', 'N/A')}")
            print(f"  Course topics: {data.get('course_content', {}).get('total_topics', 'N/A')}")
            print(f"  Course code examples: {data.get('course_content', {}).get('total_code_examples', 'N/A')}")
            print(f"  Combined total topics: {data.get('combined_total_topics', 'N/A')}")
        else:
            print(f"Stats check failed: {response.status_code}")
    except Exception as e:
        print(f"Stats check error: {e}")


def main():
    print("TDS Virtual TA - API Testing")
    print("Testing health, stats, and various types of student questions...")
    
    # Test health and stats first
    test_health_and_stats()
    
    # Test different types of questions
    test_questions = [
        ("How do I submit my TDS project?", "Project submission question"),
        ("What is the deadline for GA1?", "Assignment deadline question"),
        ("I'm getting an error in my Docker setup", "Technical issue question"),
        ("How are TDS assignments graded?", "Grading question"),
        ("What topics are covered in the TDS exam?", "Exam question"),
        ("Can I use gpt-4o-mini for my assignment?", "Specific tool question"),
        ("My assignment score is showing 0", "Score issue question"),
        ("How do I use prompt engineering?", "Course content - LLM topic"),
        ("What is a vector database?", "Course content - Database topic"),
        ("How do I scrape data with Python?", "Course content - Web scraping"),
        ("Show me an example of using LLMs", "Course content - Code example"),
        ("How do I use SQLite for data analysis?", "Course content - Database"),
        ("What tools are available for data visualization?", "Course content - Visualization"),
    ]
    
    for question, description in test_questions:
        test_question(question, description)
    
    print(f"\n{'='*60}")
    print("Testing completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 