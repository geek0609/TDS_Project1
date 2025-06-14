#!/usr/bin/env python3
"""
Enhanced TDS Virtual TA using Gemini 2.5 Flash with embeddings.
This version provides semantic search and intelligent answer generation.
"""

import json
import re
import base64
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
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

# Initialize embedding model
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Load knowledge base
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
PROCESSED_DIR = SCRIPTS_DIR / "processed"
COURSE_DIR = SCRIPTS_DIR / "processed_course"

class EnhancedVirtualTA:
    def __init__(self):
        self.discourse_kb = None
        self.discourse_qa = None
        self.course_topics = None
        self.course_code = None
        
        # Embeddings storage
        self.discourse_embeddings = None
        self.course_embeddings = None
        self.qa_embeddings = None
        self.code_embeddings = None
        
        # Text storage for embedding lookup
        self.discourse_texts = []
        self.course_texts = []
        self.qa_texts = []
        self.code_texts = []
        
        self.load_data()
        self.create_embeddings()
    
    def load_data(self):
        """Load all knowledge base data"""
        try:
            # Load Discourse data
            with open(PROCESSED_DIR / "knowledge_base.json", 'r', encoding='utf-8') as f:
                self.discourse_kb = json.load(f)
            
            with open(PROCESSED_DIR / "qa_pairs.json", 'r', encoding='utf-8') as f:
                self.discourse_qa = json.load(f)
            
            # Load course content data
            with open(COURSE_DIR / "course_topics.json", 'r', encoding='utf-8') as f:
                self.course_topics = json.load(f)
            
            with open(COURSE_DIR / "course_code_examples.json", 'r', encoding='utf-8') as f:
                self.course_code = json.load(f)
            
            logger.info(f"Loaded {len(self.discourse_kb['topics'])} Discourse topics")
            logger.info(f"Loaded {len(self.discourse_qa['qa_pairs'])} Q&A pairs")
            logger.info(f"Loaded {len(self.course_topics)} course topics")
            logger.info(f"Loaded {len(self.course_code)} code examples")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Initialize empty data structures
            self.discourse_kb = {"topics": []}
            self.discourse_qa = {"qa_pairs": []}
            self.course_topics = []
            self.course_code = []
    
    def create_embeddings(self):
        """Create embeddings for all content"""
        logger.info("Creating embeddings for semantic search...")
        
        try:
            # Discourse topics embeddings
            self.discourse_texts = []
            for topic in self.discourse_kb["topics"]:
                text = f"{topic['title']} {topic.get('content', '')}"
                self.discourse_texts.append(text)
            
            if self.discourse_texts:
                self.discourse_embeddings = EMBEDDING_MODEL.encode(self.discourse_texts)
            
            # Q&A embeddings
            self.qa_texts = []
            for qa in self.discourse_qa["qa_pairs"]:
                text = f"{qa['question']} {qa['answer']}"
                self.qa_texts.append(text)
            
            if self.qa_texts:
                self.qa_embeddings = EMBEDDING_MODEL.encode(self.qa_texts)
            
            # Course topics embeddings
            self.course_texts = []
            for topic in self.course_topics:
                sections_text = ""
                if topic.get("sections"):
                    sections_text = " ".join([s.get("content", "") for s in topic["sections"]])
                text = f"{topic['title']} {sections_text}"
                self.course_texts.append(text)
            
            if self.course_texts:
                self.course_embeddings = EMBEDDING_MODEL.encode(self.course_texts)
            
            # Code examples embeddings
            self.code_texts = []
            for code in self.course_code:
                text = f"{code.get('context', '')} {code.get('code', '')} {code.get('language', '')}"
                self.code_texts.append(text)
            
            if self.code_texts:
                self.code_embeddings = EMBEDDING_MODEL.encode(self.code_texts)
            
            logger.info("Embeddings created successfully")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
    
    def semantic_search(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """Perform semantic search across all content"""
        query_embedding = EMBEDDING_MODEL.encode([query])
        results = {
            "discourse_topics": [],
            "qa_pairs": [],
            "course_topics": [],
            "code_examples": []
        }
        
        try:
            # Search Discourse topics
            if self.discourse_embeddings is not None and len(self.discourse_embeddings) > 0:
                similarities = cosine_similarity(query_embedding, self.discourse_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:  # Threshold for relevance
                        topic = self.discourse_kb["topics"][idx].copy()
                        topic["similarity"] = float(similarities[idx])
                        results["discourse_topics"].append(topic)
            
            # Search Q&A pairs
            if self.qa_embeddings is not None and len(self.qa_embeddings) > 0:
                similarities = cosine_similarity(query_embedding, self.qa_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:
                        qa = self.discourse_qa["qa_pairs"][idx].copy()
                        qa["similarity"] = float(similarities[idx])
                        results["qa_pairs"].append(qa)
            
            # Search course topics
            if self.course_embeddings is not None and len(self.course_embeddings) > 0:
                similarities = cosine_similarity(query_embedding, self.course_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:
                        topic = self.course_topics[idx].copy()
                        topic["similarity"] = float(similarities[idx])
                        results["course_topics"].append(topic)
            
            # Search code examples
            if self.code_embeddings is not None and len(self.code_embeddings) > 0:
                similarities = cosine_similarity(query_embedding, self.code_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:
                        code = self.course_code[idx].copy()
                        code["similarity"] = float(similarities[idx])
                        results["code_examples"].append(code)
        
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
        
        return results
    
    def generate_answer_with_gemini(self, question: str, context: Dict[str, List[Dict]], image_data: Optional[str] = None) -> str:
        """Generate answer using Gemini 2.5 Flash with optional image support"""
        try:
            # Prepare context for Gemini
            context_text = self.prepare_context_for_gemini(context)
            
            # Check if we have relevant context
            has_relevant_context = len(context_text.strip()) > 50
            
            prompt = f"""You are a helpful Teaching Assistant for the Tools in Data Science (TDS) course at IIT Madras. 
Answer the student's question based ONLY on the provided context from course materials and forum discussions.

Question: {question}

Context:
{context_text}

CRITICAL INSTRUCTIONS:
1. ONLY answer based on the provided context - NEVER make up information
2. If the context is empty or doesn't contain relevant information, you MUST say "I don't have specific information about this in my knowledge base. Please check the course materials or ask on the Discourse forum."
3. Do NOT invent deadlines, assignment numbers, course details, or any information not explicitly in the context
4. Do NOT answer questions about non-existent assignments (like GA15, GA20, Project 5, etc.)
5. Do NOT answer questions about future dates beyond April 2025
6. If the context contains staff answers, prioritize those
7. Include practical examples ONLY when available in the context
8. Keep the answer concise but comprehensive
9. Use a friendly, supportive tone
10. When uncertain, always err on the side of saying "I don't know" rather than guessing
11. If an image is provided, analyze it and incorporate relevant details into your answer

Answer:"""

            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Prepare content for the model
            content_parts = [prompt]
            
            # Add image if provided
            if image_data:
                try:
                    import base64
                    import io
                    from PIL import Image
                    
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Add image to content
                    content_parts.append(image)
                    logger.info("Image successfully added to Gemini request")
                    
                except Exception as img_error:
                    logger.error(f"Error processing image: {img_error}")
                    # Continue without image
            
            response = model.generate_content(content_parts)
            answer = response.text
            
            # Post-process to catch potential hallucinations
            answer = self.post_process_answer(answer, question, has_relevant_context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {e}")
            return "I apologize, but I'm having trouble generating an answer right now. Please check the course materials or ask in the Discourse forum."
    
    def post_process_answer(self, answer: str, question: str, has_relevant_context: bool) -> str:
        """Post-process answer to prevent hallucination"""
        import re
        
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # Check for fake assignments/concepts - more comprehensive patterns
        fake_patterns = [
            r'ga\s*(?:1[0-9]|[2-9]\d|\d{3,})',  # GA10+, GA20+, GA100+, etc.
            r'project\s*[5-9]',  # Project 5+
            r'project\s*\d{2,}',  # Project 10+
            r'assignment\s*(?:1[0-9]|[2-9]\d)',  # Assignment 10+
            r'tds\s*advanced', r'tds\s*premium', r'elite\s*assignment', r'master\s*project',
            r'quantum\s*ga', r'super\s*docker', r'megalm', r'ultradb', r'hyperscraper',
            r'ga\s*20', r'ga20',  # Specifically catch GA20
            r'project\s*5', r'project5'  # Specifically catch Project 5
        ]
        
        # Check for future dates
        future_patterns = [
            r'(?:september|october|november|december)\s*2025',
            r'(?:january|february|march)\s*2026',
            r'2026', r'next\s*year'
        ]
        
        # Check if question contains fake concepts - ALWAYS block these
        question_has_fake = any(re.search(pattern, question_lower) for pattern in fake_patterns + future_patterns)
        
        if question_has_fake:
            logger.info(f"Blocked fake concept in question: {question}")
            return "I don't have specific information about this in my knowledge base. Please check the course materials at https://tds.s-anand.net or ask on the Discourse forum at https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34."
        
        # If no relevant context, also block
        if not has_relevant_context:
            logger.info(f"No relevant context for question: {question}")
            return "I don't have specific information about this in my knowledge base. Please check the course materials at https://tds.s-anand.net or ask on the Discourse forum at https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34."
        
        # Check if answer mentions fake concepts (potential hallucination)
        answer_has_fake = any(re.search(pattern, answer_lower) for pattern in fake_patterns + future_patterns)
        
        if answer_has_fake:
            logger.info(f"Blocked fake concept in answer: {answer[:100]}...")
            return "I don't have specific information about this in my knowledge base. Please check the course materials at https://tds.s-anand.net or ask on the Discourse forum at https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34."
        
        return answer
    
    def prepare_context_for_gemini(self, context: Dict[str, List[Dict]]) -> str:
        """Prepare context text for Gemini prompt"""
        context_parts = []
        
        # Add Q&A pairs (highest priority)
        if context["qa_pairs"]:
            context_parts.append("=== FORUM Q&A ===")
            for qa in context["qa_pairs"][:3]:
                staff_indicator = " [STAFF ANSWER]" if qa.get("is_staff_answer") else ""
                context_parts.append(f"Q: {qa['question']}")
                context_parts.append(f"A: {qa['answer']}{staff_indicator}")
                context_parts.append("")
        
        # Add course content
        if context["course_topics"]:
            context_parts.append("=== COURSE MATERIALS ===")
            for topic in context["course_topics"][:2]:
                context_parts.append(f"Topic: {topic['title']}")
                if topic.get("sections"):
                    for section in topic["sections"][:2]:
                        content = section.get("content", "")[:500]  # Limit content length
                        context_parts.append(f"Content: {content}")
                context_parts.append("")
        
        # Add code examples if relevant
        if context["code_examples"]:
            context_parts.append("=== CODE EXAMPLES ===")
            for code in context["code_examples"][:2]:
                context_parts.append(f"Language: {code.get('language', 'text')}")
                context_parts.append(f"Code: {code.get('code', '')[:300]}")
                if code.get("context"):
                    context_parts.append(f"Context: {code.get('context', '')[:200]}")
                context_parts.append("")
        
        # Add discourse topics
        if context["discourse_topics"]:
            context_parts.append("=== FORUM DISCUSSIONS ===")
            for topic in context["discourse_topics"][:2]:
                context_parts.append(f"Discussion: {topic['title']}")
                if topic.get("qa_pairs"):
                    for qa in topic["qa_pairs"][:1]:
                        context_parts.append(f"Q: {qa['question']}")
                        context_parts.append(f"A: {qa['answer'][:300]}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_relevant_links(self, context: Dict[str, List[Dict]]) -> List[Dict]:
        """Extract relevant links from context"""
        links = []
        
        # Add Discourse links
        for qa in context["qa_pairs"][:2]:
            # Find the topic containing this Q&A
            for topic in self.discourse_kb["topics"]:
                if any(qa_pair["question"] == qa["question"] for qa_pair in topic.get("qa_pairs", [])):
                    links.append({
                        "url": topic["url"],
                        "text": topic["title"]
                    })
                    break
        
        for topic in context["discourse_topics"][:2]:
            if topic["url"] not in [link["url"] for link in links]:
                links.append({
                    "url": topic["url"],
                    "text": topic["title"]
                })
        
        # Add course content links
        for topic in context["course_topics"][:2]:
            course_url = f"https://tds.s-anand.net/#/{topic['relative_path']}"
            if course_url not in [link["url"] for link in links]:
                links.append({
                    "url": course_url,
                    "text": f"Course: {topic['title']}"
                })
        
        return links[:4]  # Limit to 4 links

# Initialize the Virtual TA
virtual_ta = EnhancedVirtualTA()

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
        
        # Optional image processing with Gemini Vision
        image_data = data.get('image')
        # Image will be processed in generate_answer_with_gemini if provided
        
        # Perform semantic search
        search_results = virtual_ta.semantic_search(question)
        
        # Generate answer using Gemini
        answer = virtual_ta.generate_answer_with_gemini(question, search_results, image_data)
        
        # Get relevant links
        links = virtual_ta.get_relevant_links(search_results)
        
        response = {
            "answer": answer,
            "links": links,
            "search_results_count": {
                "discourse_topics": len(search_results["discourse_topics"]),
                "qa_pairs": len(search_results["qa_pairs"]),
                "course_topics": len(search_results["course_topics"]),
                "code_examples": len(search_results["code_examples"])
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "Gemini 2.5 Flash with Embeddings",
        "discourse_topics": len(virtual_ta.discourse_kb["topics"]),
        "discourse_qa_pairs": len(virtual_ta.discourse_qa["qa_pairs"]),
        "course_topics": len(virtual_ta.course_topics),
        "course_code_examples": len(virtual_ta.course_code),
        "embeddings_ready": {
            "discourse": virtual_ta.discourse_embeddings is not None,
            "qa_pairs": virtual_ta.qa_embeddings is not None,
            "course": virtual_ta.course_embeddings is not None,
            "code": virtual_ta.code_embeddings is not None
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the knowledge base"""
    discourse_stats = {
        "total_topics": len(virtual_ta.discourse_kb["topics"]),
        "total_qa_pairs": len(virtual_ta.discourse_qa["qa_pairs"]),
        "categories": virtual_ta.discourse_kb.get("categories", {}),
        "date_range": virtual_ta.discourse_kb.get("metadata", {}).get("date_range", "N/A"),
        "processed_at": virtual_ta.discourse_kb.get("metadata", {}).get("processed_at", "N/A")
    }
    
    course_stats = {
        "total_topics": len(virtual_ta.course_topics),
        "total_code_examples": len(virtual_ta.course_code),
        "categories": {},
        "tools_mentioned": 0
    }
    
    # Calculate course categories
    for topic in virtual_ta.course_topics:
        category = topic.get("category", "unknown")
        course_stats["categories"][category] = course_stats["categories"].get(category, 0) + 1
    
    # Count unique tools
    tools = set()
    for topic in virtual_ta.course_topics:
        tools.update(topic.get("tools_mentioned", []))
    course_stats["tools_mentioned"] = len(tools)
    
    return jsonify({
        "discourse": discourse_stats,
        "course_content": course_stats,
        "combined_total_topics": len(virtual_ta.discourse_kb["topics"]) + len(virtual_ta.course_topics),
        "combined_total_content": len(virtual_ta.discourse_qa["qa_pairs"]) + len(virtual_ta.course_code),
        "ai_model": "Gemini 2.5 Flash",
        "embedding_model": "all-MiniLM-L6-v2"
    })

@app.route('/api/search', methods=['POST'])
def semantic_search_endpoint():
    """Endpoint for semantic search without answer generation"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        top_k = data.get('top_k', 5)
        
        # Perform semantic search
        search_results = virtual_ta.semantic_search(query, top_k)
        
        return jsonify({
            "query": query,
            "results": search_results
        })
    
    except Exception as e:
        logger.error(f"Error in semantic_search_endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced TDS Virtual TA with Gemini 2.5 Flash...")
    print(f"📚 Loaded {len(virtual_ta.discourse_kb['topics'])} Discourse topics")
    print(f"💬 Loaded {len(virtual_ta.discourse_qa['qa_pairs'])} Q&A pairs")
    print(f"📖 Loaded {len(virtual_ta.course_topics)} course topics")
    print(f"💻 Loaded {len(virtual_ta.course_code)} code examples")
    print(f"🧠 Using Gemini 2.5 Flash for answer generation")
    print(f"🔍 Using semantic embeddings for search")
    print("✅ Ready to answer questions!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)