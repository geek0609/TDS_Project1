import requests
import json

def debug_search():
    question = "if a student has 10/10 and his name is in the bonus list, how would that look like in the dashboard?"
    
    print("🔍 Debugging search results for bonus question...")
    print(f"❓ Question: {question}")
    print("=" * 80)
    
    try:
        # Test the search endpoint
        response = requests.post('http://localhost:5000/api/search', 
                               json={'query': question}, 
                               timeout=30)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📈 Search Results Summary:")
            for category, items in result.items():
                if isinstance(items, list):
                    print(f"  {category}: {len(items)} items")
                    
                    # Show top 2 results for each category
                    for i, item in enumerate(items[:2]):
                        similarity = item.get('similarity', 'N/A')
                        title = item.get('title', item.get('question', 'No title'))[:80]
                        print(f"    {i+1}. [{similarity:.3f}] {title}")
                        
                        # Show content preview
                        content = item.get('content', item.get('answer', ''))[:200]
                        if content:
                            print(f"       Content: {content}...")
                        print()
            
            # Check if any high-similarity results exist
            all_similarities = []
            for category, items in result.items():
                if isinstance(items, list):
                    for item in items:
                        if 'similarity' in item:
                            all_similarities.append(item['similarity'])
            
            if all_similarities:
                max_sim = max(all_similarities)
                print(f"🎯 Highest similarity: {max_sim:.3f}")
                if max_sim > 0.5:
                    print("✅ High similarity results found - should provide good context")
                elif max_sim > 0.3:
                    print("⚠️  Medium similarity - context may be relevant")
                else:
                    print("❌ Low similarity - may not have relevant context")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    debug_search() 