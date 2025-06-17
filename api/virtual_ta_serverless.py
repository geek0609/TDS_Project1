#!/usr/bin/env python3
"""
Serverless-optimized TDS Virtual TA using pre-computed embeddings
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from typing import List, Dict, Any, Tuple
import base64

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for cached data
embeddings_matrix = None
content_items = None
embedding_model = None

def load_cached_data():
    """Load pre-computed embeddings and metadata"""
    global embeddings_matrix, content_items, embedding_model
    
    if embeddings_matrix is not None:
        return True
    
    try:
        # Load embeddings
        embeddings_path = parent_dir / "embeddings_cache" / "embeddings.npy"
        if embeddings_path.exists():
            embeddings_matrix = np.load(embeddings_path)
        else:
            print("❌ No cached embeddings found")
            return False
        
        # Load metadata
        metadata_path = parent_dir / "embeddings_cache" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                content_items = metadata['items']
                embedding_model = metadata.get('model', 'text-embedding-004')
        else:
            print("❌ No cached metadata found")
            return False
        
        print(f"✅ Loaded {embeddings_matrix.shape[0]} cached embeddings")
        return True
        
    except Exception as e:
        print(f"❌ Error loading cached data: {e}")
        return False

def semantic_search(query: str, top_k: int = 5) -> List[Dict]:
    """Search using pre-computed embeddings"""
    if not load_cached_data():
        return []
    
    try:
        # Create query embedding
        response = genai.embed_content(
            model=f"models/{embedding_model}",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = np.array(response['embedding'])
        
        # Calculate similarities
        similarities = np.dot(embeddings_matrix, query_embedding)
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Minimum threshold
                item = content_items[idx].copy()
                item['similarity'] = float(similarities[idx])
                results.append(item)
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def analyze_intent(question: str) -> Dict[str, Any]:
    """Quick intent analysis"""
    question_lower = question.lower()
    
    return {
        "type": "technical" if any(word in question_lower for word in 
                                 ["docker", "api", "code", "error", "install", "deploy"]) else "general",
        "is_about_future": any(word in question_lower for word in 
                             ["will", "future", "exam", "test", "quiz"]),
        "requires_specific_info": "?" in question,
        "confidence_required": "medium",
        "needs_technical_details": any(word in question_lower for word in 
                                     ["how", "setup", "configure", "implement"]),
        "is_troubleshooting": any(word in question_lower for word in 
                                ["error", "problem", "issue", "fix", "debug"])
    }

def generate_answer(question: str, context_items: List[Dict], intent: Dict) -> str:
    """Generate answer using Gemini with context"""
    try:
        # Build context
        context_parts = []
        for item in context_items[:3]:  # Limit context for token efficiency
            title = item.get('title', '')
            content = item.get('content', '')[:500]  # Truncate for efficiency
            context_parts.append(f"Title: {title}\nContent: {content}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful Teaching Assistant for the "Tools in Data Science" course at IIT Madras.

CONTEXT:
{context}

QUESTION: {question}

Please provide a helpful, accurate answer based on the context provided. If the context doesn't contain relevant information, provide a general helpful response but mention that specific details should be checked in the course materials.

Keep responses concise but informative."""

        # Generate response with Flash model for speed
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        print(f"Answer generation error: {e}")
        return "I apologize, but I'm having trouble generating an answer right now. Please try again or check the course materials directly."

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "Gemini 2.0 Flash",
        "embeddings": "Text Embedding 004",
        "optimization": "Pre-computed Embeddings + Serverless"
    })

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        results = semantic_search(query, top_k=5)
        
        return jsonify({
            "query": query,
            "results": results
        })
        
    except Exception as e:
        print(f"Search endpoint error: {e}")
        return jsonify({"error": "Search failed"}), 500

@app.route('/api/', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Analyze intent
        intent = analyze_intent(question)
        
        # Search for relevant content
        search_results = semantic_search(question, top_k=5)
        
        # Generate answer
        answer = generate_answer(question, search_results, intent)
        
        # Extract links from search results (matching main virtual_ta.py logic)
        links = []
        for result in search_results[:6]:  # Get more links like main version
            link_data = {}
            if result.get('type') == 'discourse_topic':
                link_data['url'] = result.get('data', {}).get('url', '')
                link_data['text'] = result.get('data', {}).get('title', '')
            elif result.get('type') == 'qa_pair':
                link_data['url'] = result.get('topic_url', '')
                link_data['text'] = result.get('title', '')
            elif result.get('type') == 'course_content':
                link_data['url'] = result.get('data', {}).get('url', '')
                link_data['text'] = f"Course: {result.get('data', {}).get('title', '')}"
            
            if link_data.get('url'):
                links.append(link_data)
        
        return jsonify({
            "answer": answer,
            "links": links,
            "search_results_count": len(search_results),
            "intent_analysis": intent
        })
        
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        return jsonify({
            "answer": "I apologize, but I'm having trouble generating an answer right now. Please try again or check the course materials directly.",
            "links": [],
            "search_results_count": 0,
            "intent_analysis": {"type": "error"}
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "TDS Virtual TA API",
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/ (POST)",
            "search": "/api/search (POST)"
        }
    })

# For local testing
if __name__ == "__main__":
    print("🚀 Starting TDS Virtual TA (Serverless)...")
    app.run(host="0.0.0.0", port=5000, debug=False) 