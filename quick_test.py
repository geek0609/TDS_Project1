import requests
import json

def test_bonus_question():
    question = "if a student has 10/10 and his name is in the bonus list, how would that look like in the dashboard?"
    
    print("🧪 Testing the bonus question...")
    print(f"❓ Question: {question}")
    print("=" * 80)
    
    # First test a simpler question
    print("🔍 Testing simpler question first...")
    simple_response = requests.post('http://localhost:5000/api/', 
                                  json={'question': 'What is GA4?'}, 
                                  timeout=15)
    if simple_response.status_code == 200:
        simple_result = simple_response.json()
        print(f"Simple answer: {simple_result['answer'][:100]}...")
        if "don't have specific information" in simple_result['answer']:
            print("⚠️  Even simple questions get generic responses!")
        else:
            print("✅ Simple questions work - issue is specific to bonus question")
    print()
    
    try:
        # Test the main API endpoint
        response = requests.post('http://localhost:5000/api/', 
                               json={'question': question}, 
                               timeout=30)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n💬 Answer:")
            print(f"{result['answer']}")
            
            print(f"\n🔗 Links ({len(result.get('links', []))}):")
            for i, link in enumerate(result.get('links', []), 1):
                print(f"  {i}. {link['text']}")
                print(f"     {link['url']}")
            
            print(f"\n📈 Search Results:")
            counts = result.get('search_results_count', {})
            for category, count in counts.items():
                print(f"  - {category}: {count}")
            
            # Check if answer contains the expected information
            answer_lower = result['answer'].lower()
            if '11' in answer_lower and ('10' in answer_lower or 'dashboard' in answer_lower):
                print("\n✅ SUCCESS: Answer mentions 11/10 or 110%!")
            elif 'bonus' in answer_lower and 'ga4' in answer_lower:
                print("\n⚠️  PARTIAL: Mentions bonus and GA4 but may lack specifics")
            elif "don't have specific information" in result['answer']:
                print("\n❌ ISSUE: Generic 'no information' response")
            else:
                print("\n❓ UNCLEAR: Answer doesn't match expected patterns")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_bonus_question() 