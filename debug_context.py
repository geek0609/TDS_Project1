import requests
import json

def test_context_preparation():
    question = "if a student has 10/10 and his name is in the bonus list, how would that look like in the dashboard?"
    
    print("🔍 Testing context preparation...")
    print(f"❓ Question: {question}")
    print("=" * 80)
    
    try:
        # First get search results
        search_response = requests.post('http://localhost:5000/api/search', 
                                      json={'query': question}, 
                                      timeout=30)
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            search_results = search_data.get('results', {})
            
            print("📊 Raw Search Results:")
            for category, items in search_results.items():
                if isinstance(items, list) and items:
                    print(f"\n{category.upper()} ({len(items)} items):")
                    for i, item in enumerate(items[:2]):
                        print(f"  {i+1}. Similarity: {item.get('similarity', 'N/A')}")
                        print(f"     Title: {item.get('title', item.get('question', 'No title'))}")
                        
                        # Show content/answer
                        content = item.get('content', item.get('answer', ''))
                        if content:
                            print(f"     Content: {content[:200]}...")
                        print()
                elif isinstance(items, list):
                    print(f"\n{category.upper()}: 0 items (empty)")
                else:
                    print(f"\n{category.upper()}: {items}")
            
            # Test a simple GA4 question that we know works
            print("\n" + "="*50)
            print("🔍 Testing simple GA4 question for comparison...")
            
            simple_search = requests.post('http://localhost:5000/api/search', 
                                        json={'query': 'What is GA4?'}, 
                                        timeout=15)
            
            if simple_search.status_code == 200:
                simple_data = simple_search.json()
                simple_results = simple_data.get('results', {})
                print("📊 Simple GA4 Search Results:")
                for category, items in simple_results.items():
                    if isinstance(items, list) and items:
                        print(f"  {category}: {len(items)} items (top similarity: {items[0].get('similarity', 'N/A')})")
                    elif isinstance(items, list):
                        print(f"  {category}: 0 items (empty)")
                    else:
                        print(f"  {category}: {items}")
                        
        else:
            print(f"❌ Search Error: {search_response.status_code}")
            print(f"Response: {search_response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_context_preparation() 