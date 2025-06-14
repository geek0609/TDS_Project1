import json
import re
import base64
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load knowledge base
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
PROCESSED_DIR = SCRIPTS_DIR / "processed"
COURSE_DIR = SCRIPTS_DIR / "processed_course"

# Load Discourse data
with open(PROCESSED_DIR / "knowledge_base.json", 'r', encoding='utf-8') as f:
    DISCOURSE_KB = json.load(f)

with open(PROCESSED_DIR / "search_index.json", 'r', encoding='utf-8') as f:
    DISCOURSE_INDEX = json.load(f)

with open(PROCESSED_DIR / "qa_pairs.json", 'r', encoding='utf-8') as f:
    DISCOURSE_QA = json.load(f)

# Load course content data
try:
    with open(COURSE_DIR / "course_topics.json", 'r', encoding='utf-8') as f:
        COURSE_TOPICS = json.load(f)
    
    with open(COURSE_DIR / "course_search_indices.json", 'r', encoding='utf-8') as f:
        COURSE_INDEX = json.load(f)
    
    with open(COURSE_DIR / "course_code_examples.json", 'r', encoding='utf-8') as f:
        COURSE_CODE = json.load(f)
    
    with open(COURSE_DIR / "course_project_info.json", 'r', encoding='utf-8') as f:
        COURSE_PROJECTS = json.load(f)
    
    print(f"Loaded course content: {len(COURSE_TOPICS)} topics, {len(COURSE_CODE)} code examples")
except FileNotFoundError as e:
    print(f"Course content not found: {e}")
    COURSE_TOPICS = []
    COURSE_INDEX = {"keywords": {}, "categories": {}, "tools": {}}
    COURSE_CODE = []
    COURSE_PROJECTS = []


def clean_text(text: str) -> str:
    """Clean and normalize text for searching"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text"""
    text = clean_text(text)
    words = text.split()
    # Filter out common stop words and short words
    stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'to', 'of', 'for', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'}
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    return keywords


def search_discourse_qa(question: str) -> List[Dict]:
    """Search for relevant Q&A pairs from Discourse"""
    question_keywords = extract_keywords(question)
    scored_pairs = []
    
    for qa in DISCOURSE_QA["qa_pairs"]:
        score = 0
        qa_text = clean_text(qa["question"] + " " + qa["answer"])
        
        # Keyword matching
        for keyword in question_keywords:
            if keyword in qa_text:
                score += 1
                if keyword in clean_text(qa["question"]):
                    score += 2  # Higher weight for question matches
        
        # Boost staff answers
        if qa.get("is_staff_answer"):
            score += 3
        elif qa.get("is_accepted"):
            score += 2
        
        if score > 0:
            scored_pairs.append((score, qa))
    
    # Sort by score and return top results
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    return [pair[1] for pair in scored_pairs[:5]]


def search_discourse_topics(question: str) -> List[Dict]:
    """Search Discourse topics by keywords"""
    question_keywords = extract_keywords(question)
    topic_scores = {}
    
    for keyword in question_keywords:
        if keyword in DISCOURSE_INDEX["by_keywords"]:
            for topic_id in DISCOURSE_INDEX["by_keywords"][keyword]:
                if topic_id not in topic_scores:
                    topic_scores[topic_id] = 0
                topic_scores[topic_id] += 1
    
    # Get top scoring topics
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for topic_id, score in sorted_topics[:3]:
        # Find the topic in knowledge base
        for topic in DISCOURSE_KB["topics"]:
            if topic["topic_id"] == topic_id:
                results.append(topic)
                break
    
    return results


def search_course_content(question: str) -> List[Dict]:
    """Search course content by keywords and categories"""
    question_keywords = extract_keywords(question)
    topic_scores = {}
    
    # Search by keywords
    for keyword in question_keywords:
        if keyword in COURSE_INDEX["keywords"]:
            for topic_id in COURSE_INDEX["keywords"][keyword]:
                if topic_id not in topic_scores:
                    topic_scores[topic_id] = 0
                topic_scores[topic_id] += 1
    
    # Search by tools mentioned
    for keyword in question_keywords:
        if keyword in COURSE_INDEX["tools"]:
            for topic_id in COURSE_INDEX["tools"][keyword]:
                if topic_id not in topic_scores:
                    topic_scores[topic_id] = 0
                topic_scores[topic_id] += 2  # Higher weight for tool matches
    
    # Get top scoring topics
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for topic_id, score in sorted_topics[:3]:
        # Find the topic in course content
        for topic in COURSE_TOPICS:
            if topic["id"] == topic_id:
                results.append(topic)
                break
    
    return results


def search_code_examples(question: str) -> List[Dict]:
    """Search for relevant code examples"""
    question_keywords = extract_keywords(question)
    relevant_code = []
    
    for code_example in COURSE_CODE:
        score = 0
        code_text = clean_text(code_example.get("context", "") + " " + code_example.get("code", ""))
        
        for keyword in question_keywords:
            if keyword in code_text:
                score += 1
            if keyword in code_example.get("language", "").lower():
                score += 2
        
        if score > 0:
            relevant_code.append((score, code_example))
    
    # Sort by score and return top results
    relevant_code.sort(key=lambda x: x[0], reverse=True)
    return [code[1] for code in relevant_code[:3]]


def categorize_question(question: str) -> str:
    """Categorize the question to help with search"""
    question_lower = clean_text(question)
    
    if any(word in question_lower for word in ["project", "assignment", "ga1", "ga2", "ga3", "ga4", "ga5", "ga6", "ga7"]):
        return "assignment"
    elif any(word in question_lower for word in ["error", "issue", "problem", "bug", "not working", "failed"]):
        return "technical_issue"
    elif any(word in question_lower for word in ["exam", "roe", "mock", "test"]):
        return "exam"
    elif any(word in question_lower for word in ["score", "marks", "grade", "evaluation"]):
        return "grading"
    elif any(word in question_lower for word in ["deadline", "submission", "submit"]):
        return "deadline"
    else:
        return "general"


def generate_answer(question: str, discourse_qa: List[Dict], discourse_topics: List[Dict], 
                   course_content: List[Dict], code_examples: List[Dict]) -> str:
    """Generate a comprehensive answer based on found information"""
    answer_parts = []
    
    # Priority 1: Staff answers from Discourse
    if discourse_qa:
        best_qa = discourse_qa[0]
        if best_qa.get("is_staff_answer"):
            answer_parts.append(f"Based on a staff response: {best_qa['answer']}")
        elif best_qa.get("is_accepted"):
            answer_parts.append(f"Based on an accepted answer: {best_qa['answer']}")
        else:
            answer_parts.append(f"Based on a community answer: {best_qa['answer']}")
    
    # Priority 2: Course content for conceptual questions
    elif course_content:
        topic = course_content[0]
        # Extract relevant section content
        if topic.get("sections"):
            relevant_section = topic["sections"][0]
            content_preview = relevant_section["content"][:300] + "..." if len(relevant_section["content"]) > 300 else relevant_section["content"]
            answer_parts.append(f"From the course material on '{topic['title']}': {content_preview}")
        else:
            answer_parts.append(f"This relates to the course topic '{topic['title']}'. Please check the course materials for detailed information.")
    
    # Priority 3: Discourse topics
    elif discourse_topics:
        topic = discourse_topics[0]
        if topic.get("qa_pairs"):
            qa = topic["qa_pairs"][0]
            answer_parts.append(f"From the discussion '{topic['title']}': {qa['answer']}")
        else:
            answer_parts.append(f"This relates to the topic '{topic['title']}'. Please check the full discussion for more details.")
    
    # Add code examples if relevant
    if code_examples and any(word in clean_text(question) for word in ["code", "example", "how to", "implement", "script"]):
        code_ex = code_examples[0]
        answer_parts.append(f"\n\nHere's a relevant code example in {code_ex['language']}:\n```{code_ex['language']}\n{code_ex['code'][:500]}{'...' if len(code_ex['code']) > 500 else ''}\n```")
    
    if not answer_parts:
        return "I couldn't find specific information about your question in the TDS course materials or discussions. Please check the course materials at https://tds.s-anand.net/ or ask in the Discourse forum."
    
    return " ".join(answer_parts)


def get_relevant_links(discourse_qa: List[Dict], discourse_topics: List[Dict], 
                      course_content: List[Dict]) -> List[Dict]:
    """Get relevant links from both Discourse and course content"""
    links = []
    
    # Add links from Discourse Q&A pairs (need to find the topic)
    for qa in discourse_qa[:2]:
        for topic in DISCOURSE_KB["topics"]:
            if any(qa_pair["question"] == qa["question"] for qa_pair in topic["qa_pairs"]):
                links.append({
                    "url": topic["url"],
                    "text": topic["title"]
                })
                break
    
    # Add links from relevant Discourse topics
    for topic in discourse_topics[:2]:
        if topic["url"] not in [link["url"] for link in links]:
            links.append({
                "url": topic["url"],
                "text": topic["title"]
            })
    
    # Add course content links
    for topic in course_content[:2]:
        course_url = f"https://tds.s-anand.net/#/{topic['relative_path']}"
        if course_url not in [link["url"] for link in links]:
            links.append({
                "url": course_url,
                "text": f"Course: {topic['title']}"
            })
    
    return links[:4]  # Limit to 4 links


@app.route('/api/', methods=['POST'])
def answer_question():
    """Main API endpoint for answering student questions"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "Question is required"}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        # Optional image processing (placeholder for now)
        image_data = data.get('image')
        if image_data:
            # In a real implementation, you might use OCR or image analysis
            # For now, we'll just acknowledge it was received
            pass
        
        # Search for relevant information from multiple sources
        discourse_qa = search_discourse_qa(question)
        discourse_topics = search_discourse_topics(question)
        course_content = search_course_content(question)
        code_examples = search_code_examples(question)
        
        # Generate comprehensive answer
        answer = generate_answer(question, discourse_qa, discourse_topics, course_content, code_examples)
        
        # Get relevant links
        links = get_relevant_links(discourse_qa, discourse_topics, course_content)
        
        response = {
            "answer": answer,
            "links": links
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "discourse_topics": len(DISCOURSE_KB["topics"]),
        "discourse_qa_pairs": len(DISCOURSE_QA["qa_pairs"]),
        "course_topics": len(COURSE_TOPICS),
        "course_code_examples": len(COURSE_CODE),
        "search_index_keywords": len(DISCOURSE_INDEX["by_keywords"]),
        "course_index_keywords": len(COURSE_INDEX["keywords"]),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the knowledge base"""
    discourse_stats = {
        "total_topics": DISCOURSE_KB["metadata"]["total_topics"],
        "total_qa_pairs": DISCOURSE_KB["metadata"]["total_qa_pairs"],
        "categories": DISCOURSE_KB["categories"],
        "date_range": DISCOURSE_KB["metadata"]["date_range"],
        "processed_at": DISCOURSE_KB["metadata"]["processed_at"]
    }
    
    course_stats = {
        "total_topics": len(COURSE_TOPICS),
        "total_code_examples": len(COURSE_CODE),
        "categories": COURSE_INDEX.get("categories", {}),
        "tools_mentioned": len(COURSE_INDEX.get("tools", {}))
    }
    
    return jsonify({
        "discourse": discourse_stats,
        "course_content": course_stats,
        "combined_total_topics": len(DISCOURSE_KB["topics"]) + len(COURSE_TOPICS),
        "combined_total_qa_pairs": len(DISCOURSE_QA["qa_pairs"]) + len(COURSE_CODE)
    })


if __name__ == '__main__':
    print("Starting TDS Virtual TA API...")
    print(f"Loaded {len(DISCOURSE_KB['topics'])} Discourse topics")
    print(f"Loaded {len(DISCOURSE_QA['qa_pairs'])} Discourse Q&A pairs")
    print(f"Loaded {len(COURSE_TOPICS)} course topics")
    print(f"Loaded {len(COURSE_CODE)} code examples")
    app.run(debug=True, host='0.0.0.0', port=5000) 