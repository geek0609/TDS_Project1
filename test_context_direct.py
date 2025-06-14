import sys
import os
sys.path.append('.')

# Import the VirtualTA class directly
from app_gemini import EnhancedVirtualTA

def test_context_preparation():
    print("🔍 Testing context preparation directly...")
    
    # Initialize the Virtual TA
    virtual_ta = EnhancedVirtualTA()
    
    question = "if a student has 10/10 and his name is in the bonus list, how would that look like in the dashboard?"
    
    print(f"❓ Question: {question}")
    print("=" * 80)
    
    # Perform semantic search
    search_results = virtual_ta.semantic_search(question)
    
    print("📊 Search Results Summary:")
    for category, items in search_results.items():
        print(f"  {category}: {len(items)} items")
        if items:
            print(f"    Top similarity: {items[0].get('similarity', 'N/A')}")
            if category == 'qa_pairs':
                print(f"    Top Q&A: {items[0].get('question', 'No question')[:100]}...")
    
    # Prepare context for Gemini
    context_text = virtual_ta.prepare_context_for_gemini(search_results)
    
    print(f"\n📝 Context Length: {len(context_text)} characters")
    print(f"📝 Context Preview (first 1000 chars):")
    print("-" * 50)
    print(context_text[:1000])
    print("-" * 50)
    
    # Check if context has relevant content
    has_relevant_context = len(context_text.strip()) > 50
    print(f"\n✅ Has relevant context: {has_relevant_context}")
    
    # Check for specific keywords
    context_lower = context_text.lower()
    keywords = ['bonus', 'ga4', '10/10', '11', 'dashboard', 'marks']
    found_keywords = [kw for kw in keywords if kw in context_lower]
    print(f"🔍 Found keywords: {found_keywords}")
    
    # Show the full context if it's not too long
    if len(context_text) < 2000:
        print(f"\n📄 Full Context:")
        print("=" * 80)
        print(context_text)
        print("=" * 80)

if __name__ == "__main__":
    test_context_preparation() 