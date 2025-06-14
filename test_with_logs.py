import requests
import json
import time

def test_with_server_logs():
    question = "if a student has 10/10 and his name is in the bonus list, how would that look like in the dashboard?"
    
    print("🧪 Testing bonus question - check server logs for context details...")
    print(f"❓ Question: {question}")
    print("=" * 80)
    print("📋 What to look for in server logs:")
    print("  - Context length (should be > 50 characters)")
    print("  - Context preview (should contain GA4 bonus info)")
    print("  - 'Has relevant context: True'")
    print("  - Keywords: bonus, ga4, marks, dashboard")
    print()
    print("🚀 Making API call now...")
    
    try:
        response = requests.post('http://localhost:5000/api/', 
                               json={'question': question}, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ API Response received")
            print(f"📊 Answer: {result['answer']}")
            
            # Check if we got the expected answer
            answer_lower = result['answer'].lower()
            if 'specific information' in answer_lower and 'knowledge base' in answer_lower:
                print("\n❌ ISSUE: Still getting generic response despite finding relevant content")
                print("💡 This suggests the context preparation or AI prompt needs adjustment")
            elif '11' in answer_lower or '110' in answer_lower:
                print("\n✅ SUCCESS: Got specific answer about 11/10 or 110%!")
            else:
                print(f"\n❓ UNCLEAR: Got a different response: {result['answer'][:100]}...")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_with_server_logs() 