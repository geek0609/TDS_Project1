import requests
import json

def test_search_and_links():
    """Test the search functionality and link extraction"""
    
    # Test the search endpoint first
    print("=== TESTING SEARCH ENDPOINT ===")
    search_response = requests.post(
        "http://localhost:5000/api/search",
        json={"query": "GA4 bonus dashboard", "top_k": 5}
    )
    
    if search_response.status_code == 200:
        search_results = search_response.json()
        print("Search results found:")
        for category, items in search_results.get("results", {}).items():
            print(f"  {category}: {len(items)} items")
            for i, item in enumerate(items[:2]):  # Show first 2 items
                print(f"    {i+1}. {item.get('title', 'No title')} (similarity: {item.get('similarity', 0):.3f})")
                if 'url' in item:
                    print(f"       URL: {item['url']}")
    else:
        print(f"Search failed: {search_response.status_code}")
        return
    
    print("\n=== TESTING ANSWER ENDPOINT ===")
    # Test the answer endpoint
    answer_response = requests.post(
        "http://localhost:5000/api/",
        json={"question": "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"}
    )
    
    if answer_response.status_code == 200:
        result = answer_response.json()
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Links: {result.get('links', [])}")
        print(f"Number of links: {len(result.get('links', []))}")
    else:
        print(f"Answer failed: {answer_response.status_code}")

if __name__ == "__main__":
    test_search_and_links() 