#!/usr/bin/env python3
"""
Debug script to examine the exact context being found for the GA4 bonus question.
"""

import requests
import json

def debug_context():
    """Debug the context being found for the bonus question"""
    
    question = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"
    
    print(f"🔍 Debugging context for: {question}")
    print("=" * 80)
    
    try:
        # Get search results
        search_response = requests.post(
            "http://localhost:5000/api/search",
            json={"query": question, "top_k": 10},
            timeout=30
        )
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            
            # Examine discourse topics in detail
            discourse_topics = search_data["results"]["discourse_topics"]
            print(f"📖 Found {len(discourse_topics)} discourse topics:")
            
            for i, topic in enumerate(discourse_topics[:3]):
                print(f"\n--- Topic {i+1}: {topic['title']} (Similarity: {topic.get('similarity', 'N/A'):.3f}) ---")
                print(f"URL: {topic.get('url', 'No URL')}")
                
                # Check all_posts content
                if topic.get("all_posts"):
                    print(f"All posts ({len(topic['all_posts'])} posts):")
                    for j, post in enumerate(topic["all_posts"][:5]):
                        content = post.get("content", "")
                        if content.strip():
                            print(f"  Post {j+1}: {content[:300]}...")
                            # Look for key terms
                            if "110" in content or "11/10" in content:
                                print(f"    🎯 FOUND KEY TERM: {content}")
                
                # Check qa_pairs
                if topic.get("qa_pairs"):
                    print(f"Q&A pairs ({len(topic['qa_pairs'])} pairs):")
                    for j, qa in enumerate(topic["qa_pairs"][:3]):
                        print(f"  Q{j+1}: {qa.get('question', '')[:100]}...")
                        print(f"  A{j+1}: {qa.get('answer', '')[:200]}...")
                        # Look for key terms
                        answer = qa.get('answer', '')
                        if "110" in answer or "11/10" in answer:
                            print(f"    🎯 FOUND KEY TERM IN ANSWER: {answer}")
            
            # Examine Q&A pairs in detail
            qa_pairs = search_data["results"]["qa_pairs"]
            print(f"\n💬 Found {len(qa_pairs)} Q&A pairs:")
            
            for i, qa in enumerate(qa_pairs[:5]):
                print(f"\n--- Q&A {i+1} (Similarity: {qa.get('similarity', 'N/A'):.3f}) ---")
                print(f"Q: {qa.get('question', '')}")
                print(f"A: {qa.get('answer', '')[:400]}...")
                
                # Look for key terms
                answer = qa.get('answer', '')
                if "110" in answer or "11/10" in answer:
                    print(f"🎯 FOUND KEY TERM: {answer}")
                    
        else:
            print(f"❌ Search failed: {search_response.status_code}")
            print(search_response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_specific_search_terms():
    """Test searches for specific terms related to the bonus"""
    
    search_terms = [
        "GA4 bonus 110",
        "GA4 11/10",
        "GA4 dashboard 110",
        "bonus marks dashboard",
        "10/10 bonus dashboard"
    ]
    
    print("\n🔍 Testing specific search terms:")
    print("=" * 50)
    
    for term in search_terms:
        try:
            response = requests.post(
                "http://localhost:5000/api/search",
                json={"query": term, "top_k": 3},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                total_results = sum(len(v) for v in data["results"].values())
                print(f"\n'{term}': {total_results} total results")
                
                # Check for high similarity results
                for category, results in data["results"].items():
                    if results:
                        top_result = results[0]
                        similarity = top_result.get("similarity", 0)
                        if similarity > 0.5:
                            title = top_result.get("title", top_result.get("question", "No title"))
                            print(f"  {category}: {similarity:.3f} - {title[:80]}...")
                            
                            # Check content for key terms
                            if category == "qa_pairs":
                                answer = top_result.get("answer", "")
                                if "110" in answer or "11/10" in answer:
                                    print(f"    🎯 Contains key term: {answer[:200]}...")
                            
        except Exception as e:
            print(f"  Error with '{term}': {e}")

if __name__ == "__main__":
    print("🔍 DEBUGGING GA4 BONUS CONTEXT")
    print("=" * 80)
    
    debug_context()
    test_specific_search_terms()
    
    print("\n" + "=" * 80)
    print("🏁 Debug complete!") 