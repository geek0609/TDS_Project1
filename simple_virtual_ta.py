#!/usr/bin/env python3
"""
Simplified TDS Virtual TA for Vercel deployment
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    logger.warning("google-generativeai not available")

class SimpleTDSVirtualTA:
    def __init__(self):
        print("🚀 Initializing Simple TDS Virtual TA...")
        
        # Load environment
        load_dotenv()
        
        # Configure Gemini API if available
        self.has_api = False
        if HAS_GENAI:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.has_api = True
                    print("✅ Gemini API configured")
                except Exception as e:
                    print(f"⚠️ Warning: Could not configure Gemini API: {e}")
            else:
                print("⚠️ Warning: GEMINI_API_KEY not found")
        
        # Initialize data
        self.discourse_data = {"topics": [], "all_qa_pairs": []}
        self.course_data = []
        
        # Load data if available
        try:
            self.load_data()
        except Exception as e:
            print(f"⚠️ Warning: Could not load data: {e}")
        
        print("✅ Simple TDS Virtual TA ready!")
    
    def load_data(self):
        """Load discourse and course data"""
        print("📚 Loading data...")
        
        # Load discourse data
        discourse_file = Path("data/discourse_data.json")
        if discourse_file.exists():
            try:
                with open(discourse_file, 'r', encoding='utf-8') as f:
                    self.discourse_data = json.load(f)
                print(f"✅ Loaded {len(self.discourse_data['topics'])} discourse topics")
            except Exception as e:
                print(f"⚠️ Error loading discourse data: {e}")
        
        # Load course data from the tools-in-data-science-public directory
        course_dir = Path("tools-in-data-science-public")
        if course_dir.exists():
            try:
                # Find all markdown files
                md_files = list(course_dir.rglob("*.md"))
                for md_file in md_files:
                    try:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        self.course_data.append({
                            "title": md_file.stem,
                            "path": str(md_file.relative_to(course_dir)),
                            "url": f"https://tds.s-anand.net/#/{md_file.stem}",
                            "content": content,
                            "content_length": len(content)
                        })
                    except Exception as e:
                        logger.warning(f"Error reading {md_file}: {e}")
                
                print(f"✅ Loaded {len(self.course_data)} course files")
            except Exception as e:
                print(f"⚠️ Error loading course data: {e}")
    
    def simple_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Simple text-based search"""
        results = []
        query_terms = set(query.lower().split())
        
        # Search through discourse topics
        for topic in self.discourse_data.get("topics", []):
            try:
                title = topic.get("title", "").lower()
                content = topic.get("full_content", "").lower()
                combined_text = f"{title} {content}"
                
                # Calculate simple word overlap score
                text_words = set(combined_text.split())
                overlap = len(query_terms.intersection(text_words))
                if overlap > 0:
                    score = overlap / len(query_terms)
                    results.append({
                        "title": topic.get("title", ""),
                        "url": topic.get("url", ""),
                        "content": topic.get("full_content", "")[:500],
                        "score": score,
                        "type": "discourse_topic",
                        "data": topic
                    })
            except Exception as e:
                logger.warning(f"Error processing topic: {e}")
        
        # Search through Q&A pairs
        for qa in self.discourse_data.get("all_qa_pairs", []):
            try:
                question = qa.get("question", "").lower()
                answer = qa.get("answer", "").lower()
                combined_text = f"{question} {answer}"
                
                text_words = set(combined_text.split())
                overlap = len(query_terms.intersection(text_words))
                if overlap > 0:
                    score = overlap / len(query_terms)
                    results.append({
                        "title": f"Q: {qa.get('question', '')[:100]}...",
                        "url": qa.get("url", ""),
                        "content": f"Q: {qa.get('question', '')} A: {qa.get('answer', '')}",
                        "score": score,
                        "type": "qa_pair",
                        "data": qa
                    })
            except Exception as e:
                logger.warning(f"Error processing Q&A: {e}")
        
        # Search through course content
        for course in self.course_data:
            try:
                title = course.get("title", "").lower()
                content = course.get("content", "").lower()
                combined_text = f"{title} {content}"
                
                text_words = set(combined_text.split())
                overlap = len(query_terms.intersection(text_words))
                if overlap > 0:
                    score = overlap / len(query_terms)
                    results.append({
                        "title": course.get("title", ""),
                        "url": course.get("url", ""),
                        "content": course.get("content", "")[:500],
                        "score": score,
                        "type": "course_content",
                        "data": course
                    })
            except Exception as e:
                logger.warning(f"Error processing course: {e}")
        
        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def generate_answer(self, question: str, search_results: List[Dict]) -> Dict:
        """Generate answer based on search results"""
        
        # Build context from search results
        context_parts = []
        links = []
        
        for result in search_results[:10]:
            try:
                r_type = result.get("type")
                payload = result.get("data", result)
                
                if r_type in ("discourse_topic", "discourse"):
                    title = payload.get("title", "(untitled)")
                    content = payload.get("full_content") or payload.get("content") or ""
                    url = payload.get("url")
                    context_parts.append(f"Topic: {title}")
                    context_parts.append(f"Content: {content[:800]}...")
                    if url:
                        links.append({"url": url, "text": title})
                elif r_type == "qa_pair":
                    context_parts.append(f"Q: {payload.get('question','')}")
                    context_parts.append(f"A: {payload.get('answer','')}")
                    if result.get("url"):
                        links.append({"url": result.get("url"), "text": result.get("title")})
                elif r_type in ("course_content", "course"):
                    title = payload.get("title", "Course material")
                    content = payload.get("content", "")
                    url = payload.get("url")
                    context_parts.append(f"Course: {title}")
                    context_parts.append(f"Content: {content[:800]}...")
                    if url:
                        links.append({"url": url, "text": f"Course: {title}"})
                context_parts.append("")
            except Exception as e:
                logger.warning(f"Error processing result: {e}")
        
        context_text = "\n".join(context_parts)
        
        # Generate answer
        if self.has_api and HAS_GENAI:
            try:
                prompt_text = f"""You are a helpful teaching assistant. Answer the question based on the provided context.

Question: {question}

Context:
{context_text}

Answer:"""
                
                model = genai.GenerativeModel('gemini-2.0-flash')
                generation_config = {
                    'temperature': 0.3,
                    'max_output_tokens': 1000
                }
                
                response = model.generate_content(prompt_text, generation_config=generation_config)
                answer = response.text
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                answer = "I apologize, but I'm having trouble generating an answer right now. Please check the course materials directly."
        else:
            answer = "I apologize, but the AI service is not available. Please check the course materials directly."
        
        return {
            "answer": answer,
            "links": links[:20],
            "search_results_count": len(search_results),
        }

# Initialize the Virtual TA
virtual_ta = None

def get_virtual_ta():
    global virtual_ta
    if virtual_ta is None:
        virtual_ta = SimpleTDSVirtualTA()
    return virtual_ta

# Create Flask app
app = Flask(__name__)
CORS(app)

@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        search_results = get_virtual_ta().simple_search(question, top_k=20)
        result = get_virtual_ta().generate_answer(question, search_results)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in ask: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = get_virtual_ta().simple_search(query, top_k)
        
        return jsonify({
            'query': query,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'TDS Virtual TA API',
        'status': 'running',
        'endpoints': {
            'ask': '/api/ask (POST)',
            'search': '/api/search (POST)',
            'health': '/api/health (GET)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    try:
        ta = get_virtual_ta()
        return jsonify({
            'status': 'healthy',
            'has_api': ta.has_api,
            'discourse_topics': len(ta.discourse_data.get('topics', [])),
            'course_files': len(ta.course_data)
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 