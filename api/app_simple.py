#!/usr/bin/env python3
"""
Simplified TDS Virtual TA for Vercel deployment without heavy dependencies
"""

import json
import re
import base64
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")
genai.configure(api_key=GEMINI_API_KEY)

class SimpleVirtualTA:
    def __init__(self):
        """Initialize the Simple Virtual TA for Vercel"""
        self.discourse_kb = {}
        self.discourse_qa = {}
        self.course_topics = []
        self.course_code = []
        self.load_data()
    
    def load_data(self):
        """Load knowledge base data"""
        try:
            # Try to load processed data
            scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
            processed_dir = scripts_dir / "processed"
            course_dir = scripts_dir / "processed_course"
            
            # Load Discourse data from knowledge_base.json
            discourse_file = processed_dir / "knowledge_base.json"
            if discourse_file.exists():
                with open(discourse_file, 'r', encoding='utf-8') as f:
                    self.discourse_kb = json.load(f)
                logger.info(f"Loaded {len(self.discourse_kb.get('topics', []))} Discourse topics")
            
            # Load Q&A data from qa_pairs.json
            qa_file = processed_dir / "qa_pairs.json"
            if qa_file.exists():
                with open(qa_file, 'r', encoding='utf-8') as f:
                    self.discourse_qa = json.load(f)
                logger.info(f"Loaded {len(self.discourse_qa.get('qa_pairs', []))} Q&A pairs")
            
            # Load course topics
            course_topics_file = course_dir / "course_topics.json"
            if course_topics_file.exists():
                with open(course_topics_file, 'r', encoding='utf-8') as f:
                    self.course_topics = json.load(f)
                logger.info(f"Loaded {len(self.course_topics)} course topics")
            
            # Load code examples from course_code_examples.json
            code_file = course_dir / "course_code_examples.json"
            if code_file.exists():
                with open(code_file, 'r', encoding='utf-8') as f:
                    self.course_code = json.load(f)
                logger.info(f"Loaded {len(self.course_code)} code examples")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Initialize empty data structures
            self.discourse_kb = {"topics": []}
            self.discourse_qa = {"qa_pairs": []}
            self.course_topics = []
            self.course_code = []
    
    def simple_search(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """Simple keyword-based search without embeddings"""
        query_lower = query.lower()
        results = {
            "discourse_topics": [],
            "qa_pairs": [],
            "course_topics": [],
            "code_examples": []
        }
        
        # Search Discourse topics
        for topic in self.discourse_kb.get("topics", [])[:top_k]:
            if any(word in topic.get("content", "").lower() for word in query_lower.split()):
                results["discourse_topics"].append({
                    "content": topic.get("content", "")[:500],
                    "url": topic.get("url", ""),
                    "title": topic.get("title", ""),
                    "score": 0.8
                })
        
        # Search Q&A pairs
        for qa in self.discourse_qa.get("qa_pairs", [])[:top_k]:
            if any(word in qa.get("question", "").lower() or word in qa.get("answer", "").lower() 
                   for word in query_lower.split()):
                results["qa_pairs"].append({
                    "content": f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}"[:500],
                    "url": qa.get("url", ""),
                    "title": qa.get("question", ""),
                    "score": 0.9
                })
        
        # Search course topics
        for topic in self.course_topics[:top_k]:
            if any(word in topic.get("content", "").lower() for word in query_lower.split()):
                results["course_topics"].append({
                    "content": topic.get("content", "")[:500],
                    "url": topic.get("url", ""),
                    "title": topic.get("title", ""),
                    "score": 0.7
                })
        
        # Search code examples
        for code in self.course_code[:top_k]:
            if any(word in code.get("content", "").lower() for word in query_lower.split()):
                results["code_examples"].append({
                    "content": code.get("content", "")[:500],
                    "url": code.get("url", ""),
                    "title": code.get("title", ""),
                    "score": 0.6
                })
        
        return results
    
    def generate_answer_with_gemini(self, question: str, context: Dict[str, List[Dict]], image_data: Optional[str] = None) -> str:
        """Generate answer using Gemini"""
        try:
            # Prepare context
            context_text = self.prepare_context_for_gemini(context)
            
            # Create prompt
            prompt = f"""You are a helpful Teaching Assistant for the Tools in Data Science course at IIT Madras.

Context from course materials and discussions:
{context_text}

Student Question: {question}

Please provide a helpful, accurate answer based on the context provided. If the context doesn't contain enough information, say so clearly."""

            # Prepare content for Gemini
            content = [prompt]
            
            # Add image if provided
            if image_data:
                try:
                    from PIL import Image
                    import io
                    
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    content.append(image)
                    logger.info("Image successfully added to Gemini request")
                except Exception as e:
                    logger.warning(f"Error processing image: {e}")
            
            # Generate response
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(content)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {e}")
            return "I apologize, but I'm having trouble generating an answer right now. Please try again later."
    
    def prepare_context_for_gemini(self, context: Dict[str, List[Dict]]) -> str:
        """Prepare context text for Gemini"""
        context_parts = []
        
        # Add Q&A pairs first (highest priority)
        if context.get("qa_pairs"):
            context_parts.append("=== Q&A from Course Discussions ===")
            for item in context["qa_pairs"][:3]:
                context_parts.append(f"- {item['content'][:300]}...")
        
        # Add course topics
        if context.get("course_topics"):
            context_parts.append("\n=== Course Materials ===")
            for item in context["course_topics"][:3]:
                context_parts.append(f"- {item['content'][:300]}...")
        
        # Add discourse topics
        if context.get("discourse_topics"):
            context_parts.append("\n=== Student Discussions ===")
            for item in context["discourse_topics"][:2]:
                context_parts.append(f"- {item['content'][:300]}...")
        
        # Add code examples
        if context.get("code_examples"):
            context_parts.append("\n=== Code Examples ===")
            for item in context["code_examples"][:2]:
                context_parts.append(f"- {item['content'][:300]}...")
        
        return "\n".join(context_parts)
    
    def get_relevant_links(self, context: Dict[str, List[Dict]]) -> List[Dict]:
        """Extract relevant links from context"""
        links = []
        
        # Prioritize Q&A pairs and course content
        for category in ["qa_pairs", "course_topics", "discourse_topics", "code_examples"]:
            for item in context.get(category, [])[:2]:
                if item.get("url") and item.get("title"):
                    links.append({
                        "url": item["url"],
                        "text": item["title"][:100]
                    })
        
        return links[:4]  # Limit to 4 links

# Initialize the Virtual TA
virtual_ta = SimpleVirtualTA()

@app.route('/api/', methods=['POST'])
def answer_question():
    """Main endpoint for answering questions"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Question is required"}), 400
        
        question = data['question']
        image_data = data.get('image')
        
        # Simple search instead of semantic search
        context = virtual_ta.simple_search(question)
        
        # Generate answer
        answer = virtual_ta.generate_answer_with_gemini(question, context, image_data)
        
        # Get relevant links
        links = virtual_ta.get_relevant_links(context)
        
        # Count search results
        search_results_count = {
            category: len(results) for category, results in context.items()
        }
        
        return jsonify({
            "answer": answer,
            "links": links,
            "search_results_count": search_results_count
        })
        
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "Gemini 2.0 Flash",
        "search_type": "keyword_based",
        "discourse_topics": len(virtual_ta.discourse_kb.get("topics", [])),
        "discourse_qa_pairs": len(virtual_ta.discourse_qa.get("qa_pairs", [])),
        "course_topics": len(virtual_ta.course_topics),
        "course_code_examples": len(virtual_ta.course_code)
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the knowledge base"""
    return jsonify({
        "total_content": (
            len(virtual_ta.discourse_kb.get("topics", [])) +
            len(virtual_ta.discourse_qa.get("qa_pairs", [])) +
            len(virtual_ta.course_topics) +
            len(virtual_ta.course_code)
        ),
        "discourse_topics": len(virtual_ta.discourse_kb.get("topics", [])),
        "qa_pairs": len(virtual_ta.discourse_qa.get("qa_pairs", [])),
        "course_topics": len(virtual_ta.course_topics),
        "code_examples": len(virtual_ta.course_code),
        "ai_model": "Gemini 2.0 Flash",
        "search_type": "keyword_based"
    })

if __name__ == "__main__":
    app.run(debug=True) 