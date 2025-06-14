#!/usr/bin/env python3
"""
Direct test with the specific context we found to see if Gemini can answer correctly.
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

def test_direct_context():
    """Test with the exact context we found"""
    
    question = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"
    
    # This is the actual context we found in the search results
    context = """=== FORUM DISCUSSIONS ===
Discussion: GAA went from 103 to 100
Post: @carlton @Jivraj till the previous night my GAA score was 103 and this morning it is 100…the total average by manual calculation is 102.5
Post: How can it be greater than 100
Post: (100+100+100+110) / 4 = 102.5 Screenshot 2025-04-17 at 1.04.00 PM1440×900 122 KB @Jivraj @carlton average of this is 102.5
Post: @Jivraj @carlton any update?
Post: Because you guys were promising bonuses for extra activities that time
Q: @carlton @Jivraj till the previous night my GAA score was 103 and this morning it is 100…the total average by manual calculation is 102.5
A: I raised the question with operations. Its an operational issue.

Discussion: GA4 bonus marks
Post: Sir, I haven't received bonus marks in ga4 as, i posted queries in discourse regarding ga4. I have scored 8 marks in ga4, but after bonus marks it should be 9, but in dashboard it is showing only 80 marks. Kindly, update my marks in dashboard. @carlton , @Jivraj , @Saransh_Saini
Post: Hi @Sakshi6479, You have not followed the instruction in GA4 for the bonus mark. Screenshot 2025-03-18 at 11.02.31 am2566×224 23.2 KB We were given the list by Anand for all those that followed this instruction. That is why you have not received the bonus mark for GA4. Kind regards
Q: Sir, I haven't received bonus marks in ga4 as, i posted queries in discourse regarding ga4. I have scored 8 marks in ga4, but after bonus marks it should be 9, but in dashboard it is showing only 80 marks.
A: Hi @Sakshi6479, You have not followed the instruction in GA4 for the bonus mark. Screenshot 2025-03-18 at 11.02.31 am2566×224 23.2 KB We were given the list by Anand for all those that followed this instruction. That is why you have not received the bonus mark for GA4. Kind regards"""
    
    prompt = f"""You are a helpful Teaching Assistant for the Tools in Data Science (TDS) course at IIT Madras. 
Answer the student's question based on the provided context from course materials and forum discussions.

Question: {question}

Context:
{context}

INSTRUCTIONS:
1. Use the provided context to answer the question as completely as possible
2. If the context contains relevant information, provide a comprehensive answer based on that information
3. Pay special attention to staff answers and official course information
4. For GA4 bonus questions: Look for specific numerical examples in the context (like "110", "11/10", calculations showing bonus scores)
5. Include specific details like numbers, percentages, or dashboard displays when mentioned in the context
6. If you see examples of calculations or score displays in the context, use those to explain how things work
7. When the context shows specific examples (like "(100+100+100+110) / 4"), use those to illustrate your answer
8. If the context is insufficient, then say you don't have enough information
9. Use a friendly, supportive tone
10. If an image is provided, analyze it and incorporate relevant details into your answer

Answer:"""

    print("🧪 TESTING DIRECT CONTEXT WITH GEMINI")
    print("=" * 80)
    print(f"Question: {question}")
    print("\nContext provided:")
    print(context[:500] + "...")
    print("\n" + "=" * 80)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)
        answer = response.text
        
        print("✅ GEMINI RESPONSE:")
        print(answer)
        
        # Check if the answer mentions key terms
        answer_lower = answer.lower()
        key_terms = ["110", "11/10", "bonus", "dashboard", "ga4"]
        found_terms = [term for term in key_terms if term in answer_lower]
        
        print(f"\n🔍 Key terms found in answer: {found_terms}")
        
        if "110" in answer_lower:
            print("🎉 SUCCESS: Answer mentions '110' - the correct dashboard display!")
        elif "11/10" in answer_lower:
            print("🎉 SUCCESS: Answer mentions '11/10' - the correct score!")
        elif "bonus" in answer_lower and "ga4" in answer_lower:
            print("⚠️  PARTIAL: Answer mentions GA4 bonus but not the specific display")
        else:
            print("❌ ISSUE: Answer doesn't seem to address the bonus question properly")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_simplified_context():
    """Test with a more direct context"""
    
    question = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"
    
    # Simplified context focusing on the key information
    context = """=== FORUM DISCUSSIONS ===
Discussion: GAA went from 103 to 100
A student's calculation shows: (100+100+100+110) / 4 = 102.5
This shows that when a student gets 10/10 plus bonus on an assignment, it appears as 110 on the dashboard.

Discussion: GA4 bonus marks  
Students can receive bonus marks for GA4 by following specific instructions.
When bonus marks are added, the score on the dashboard increases accordingly."""
    
    prompt = f"""You are a helpful Teaching Assistant for the Tools in Data Science (TDS) course at IIT Madras. 
Answer the student's question based on the provided context.

Question: {question}

Context:
{context}

Based on the context, especially the calculation showing "(100+100+100+110) / 4 = 102.5", explain how a 10/10 + bonus score would appear on the dashboard.

Answer:"""

    print("\n🧪 TESTING SIMPLIFIED CONTEXT")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Simplified context: {context}")
    print("\n" + "=" * 80)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)
        answer = response.text
        
        print("✅ GEMINI RESPONSE:")
        print(answer)
        
        # Check if the answer mentions key terms
        answer_lower = answer.lower()
        if "110" in answer_lower:
            print("🎉 SUCCESS: Answer mentions '110'!")
        else:
            print("❌ Still not mentioning '110'")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_direct_context()
    test_simplified_context() 