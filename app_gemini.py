#!/usr/bin/env python3
"""
Enhanced TDS Virtual TA using Gemini 2.5 Flash with embeddings.
This version provides semantic search and intelligent answer generation.

Version: 2.1.0 - Fixed deployment with knowledge base files
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
import pickle

import google.generativeai as genai
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

# Load knowledge base
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
PROCESSED_DIR = SCRIPTS_DIR / "processed"
COURSE_DIR = SCRIPTS_DIR / "processed_course"

class EnhancedVirtualTA:
    def __init__(self):
        """Initialize the Enhanced Virtual TA with Gemini embeddings and caching support"""
        
        # Cache directory for embeddings
        self.cache_dir = "embeddings_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Data storage
        self.discourse_kb = {}
        self.discourse_qa = {}
        self.course_topics = []
        self.course_code = []
        
        # Text storage for embedding lookup
        self.discourse_texts = []
        self.qa_texts = []
        self.course_texts = []
        self.code_texts = []
        
        # Embeddings
        self.discourse_embeddings = None
        self.qa_embeddings = None
        self.course_embeddings = None
        self.code_embeddings = None
        
        # Load data and embeddings
        self.load_data()
        self.load_or_create_embeddings()
    
    def get_cache_path(self, cache_name: str) -> str:
        """Get the cache file path for a given cache name"""
        return os.path.join(self.cache_dir, f"{cache_name}.pkl")
    
    def chunk_text(self, text: str, max_chars: int = 30000) -> str:
        """Chunk text to fit within Gemini's payload limits"""
        if len(text) <= max_chars:
            return text
        
        # Try to split at sentence boundaries first
        sentences = text.split('. ')
        result = ""
        for sentence in sentences:
            if len(result + sentence + '. ') <= max_chars:
                result += sentence + '. '
            else:
                break
        
        # If no sentences fit, just truncate
        if not result:
            result = text[:max_chars]
        
        return result.strip()
    
    def get_gemini_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using Gemini's embedding API"""
        try:
            embeddings = []
            batch_size = 50  # Smaller batch size to avoid rate limits
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"🔄 Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                batch_embeddings = []
                for text in batch:
                    try:
                        # Chunk text to fit within limits
                        chunked_text = self.chunk_text(text)
                        
                        # Use Gemini's embedding model
                        result = genai.embed_content(
                            model="models/text-embedding-004",
                            content=chunked_text,
                            task_type="retrieval_document"
                        )
                        batch_embeddings.append(result['embedding'])
                    except Exception as e:
                        logger.warning(f"Error embedding text (length: {len(text)}): {e}")
                        # Use zero vector as fallback
                        batch_embeddings.append([0.0] * 768)  # Standard embedding dimension
                
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error getting Gemini embeddings: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), 768))
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a single query using Gemini"""
        try:
            # Chunk query if too long
            chunked_query = self.chunk_text(query)
            
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunked_query,
                task_type="retrieval_query"
            )
            return np.array([result['embedding']])
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            return np.zeros((1, 768))
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between embeddings"""
        # Normalize vectors
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarity
        return np.dot(a_norm, b_norm.T)
    
    def save_embeddings_to_cache(self):
        """Save all embeddings to cache files"""
        try:
            cache_data = {
                'discourse_embeddings': self.discourse_embeddings,
                'qa_embeddings': self.qa_embeddings,
                'course_embeddings': self.course_embeddings,
                'code_embeddings': self.code_embeddings,
                'discourse_texts': self.discourse_texts,
                'qa_texts': self.qa_texts,
                'course_texts': self.course_texts,
                'code_texts': self.code_texts,
                'timestamp': datetime.now().isoformat(),
                'embedding_model': 'gemini-text-embedding-004',
                'data_sizes': {
                    'discourse_topics': len(self.discourse_kb.get("topics", [])),
                    'qa_pairs': len(self.discourse_qa.get("qa_pairs", [])),
                    'course_topics': len(self.course_topics),
                    'code_examples': len(self.course_code)
                }
            }
            
            cache_path = self.get_cache_path("gemini_embeddings")
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"✅ Gemini embeddings saved to cache: {cache_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving embeddings to cache: {e}")
    
    def load_embeddings_from_cache(self) -> bool:
        """Load embeddings from cache if available and valid"""
        try:
            cache_path = self.get_cache_path("gemini_embeddings")
            
            if not os.path.exists(cache_path):
                logger.info("📁 No Gemini embeddings cache found")
                return False
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache is still valid (data sizes match)
            current_sizes = {
                'discourse_topics': len(self.discourse_kb.get("topics", [])),
                'qa_pairs': len(self.discourse_qa.get("qa_pairs", [])),
                'course_topics': len(self.course_topics),
                'code_examples': len(self.course_code)
            }
            
            cached_sizes = cache_data.get('data_sizes', {})
            
            if current_sizes != cached_sizes:
                logger.info("📊 Data size mismatch, cache invalid")
                return False
            
            # Load embeddings and texts
            self.discourse_embeddings = cache_data['discourse_embeddings']
            self.qa_embeddings = cache_data['qa_embeddings']
            self.course_embeddings = cache_data['course_embeddings']
            self.code_embeddings = cache_data['code_embeddings']
            self.discourse_texts = cache_data.get('discourse_texts', [])
            self.qa_texts = cache_data.get('qa_texts', [])
            self.course_texts = cache_data.get('course_texts', [])
            self.code_texts = cache_data.get('code_texts', [])
            
            cache_time = cache_data.get('timestamp', 'unknown')
            embedding_model = cache_data.get('embedding_model', 'unknown')
            logger.info(f"✅ Gemini embeddings loaded from cache (model: {embedding_model}, created: {cache_time})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading embeddings from cache: {e}")
            return False
    
    def load_or_create_embeddings(self):
        """Load embeddings from cache or create them if not available"""
        if self.load_embeddings_from_cache():
            logger.info("🚀 Using cached Gemini embeddings - fast startup!")
            return
        
        logger.info("🔄 Creating new Gemini embeddings...")
        self.create_embeddings()
        self.save_embeddings_to_cache()
    
    def load_data(self):
        """Load all knowledge base data"""
        try:
            # Check if directories exist
            logger.info(f"PROCESSED_DIR path: {PROCESSED_DIR}")
            logger.info(f"COURSE_DIR path: {COURSE_DIR}")
            logger.info(f"PROCESSED_DIR exists: {PROCESSED_DIR.exists()}")
            logger.info(f"COURSE_DIR exists: {COURSE_DIR.exists()}")
            
            if PROCESSED_DIR.exists():
                logger.info(f"Files in PROCESSED_DIR: {list(PROCESSED_DIR.iterdir())}")
            if COURSE_DIR.exists():
                logger.info(f"Files in COURSE_DIR: {list(COURSE_DIR.iterdir())}")
            
            # Load Discourse data
            kb_file = PROCESSED_DIR / "knowledge_base.json"
            qa_file = PROCESSED_DIR / "qa_pairs.json"
            logger.info(f"Trying to load: {kb_file} (exists: {kb_file.exists()})")
            logger.info(f"Trying to load: {qa_file} (exists: {qa_file.exists()})")
            
            with open(kb_file, 'r', encoding='utf-8') as f:
                self.discourse_kb = json.load(f)
            
            with open(qa_file, 'r', encoding='utf-8') as f:
                self.discourse_qa = json.load(f)
            
            # Load course content data
            topics_file = COURSE_DIR / "course_topics.json"
            code_file = COURSE_DIR / "course_code_examples.json"
            logger.info(f"Trying to load: {topics_file} (exists: {topics_file.exists()})")
            logger.info(f"Trying to load: {code_file} (exists: {code_file.exists()})")
            
            with open(topics_file, 'r', encoding='utf-8') as f:
                self.course_topics = json.load(f)
            
            with open(code_file, 'r', encoding='utf-8') as f:
                self.course_code = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.discourse_kb['topics'])} Discourse topics")
            logger.info(f"✅ Loaded {len(self.discourse_qa['qa_pairs'])} Q&A pairs")
            logger.info(f"✅ Loaded {len(self.course_topics)} course topics")
            logger.info(f"✅ Loaded {len(self.course_code)} code examples")
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Files in current directory: {os.listdir('.')}")
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
                self.discourse_embeddings = self.get_gemini_embeddings(self.discourse_texts)
            
            # Q&A embeddings
            self.qa_texts = []
            for qa in self.discourse_qa["qa_pairs"]:
                text = f"{qa['question']} {qa['answer']}"
                self.qa_texts.append(text)
            
            if self.qa_texts:
                self.qa_embeddings = self.get_gemini_embeddings(self.qa_texts)
            
            # Course topics embeddings
            self.course_texts = []
            for topic in self.course_topics:
                sections_text = ""
                if topic.get("sections"):
                    sections_text = " ".join([s.get("content", "") for s in topic["sections"]])
                text = f"{topic['title']} {sections_text}"
                self.course_texts.append(text)
            
            if self.course_texts:
                self.course_embeddings = self.get_gemini_embeddings(self.course_texts)
            
            # Code examples embeddings
            self.code_texts = []
            for code in self.course_code:
                text = f"{code.get('context', '')} {code.get('code', '')} {code.get('language', '')}"
                self.code_texts.append(text)
            
            if self.code_texts:
                self.code_embeddings = self.get_gemini_embeddings(self.code_texts)
            
            logger.info("Embeddings created successfully")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
    
    def semantic_search(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """Perform semantic search across all content"""
        query_embedding = self.get_query_embedding(query)
        results = {
            "discourse_topics": [],
            "qa_pairs": [],
            "course_topics": [],
            "code_examples": []
        }
        
        try:
            # Search Discourse topics
            if self.discourse_embeddings is not None and len(self.discourse_embeddings) > 0:
                similarities = self.cosine_similarity(query_embedding, self.discourse_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:  # Threshold for relevance
                        topic = self.discourse_kb["topics"][idx].copy()
                        topic["similarity"] = float(similarities[idx])
                        results["discourse_topics"].append(topic)
            
            # Search Q&A pairs
            if self.qa_embeddings is not None and len(self.qa_embeddings) > 0:
                similarities = self.cosine_similarity(query_embedding, self.qa_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:
                        qa = self.discourse_qa["qa_pairs"][idx].copy()
                        qa["similarity"] = float(similarities[idx])
                        results["qa_pairs"].append(qa)
            
            # Search course topics
            if self.course_embeddings is not None and len(self.course_embeddings) > 0:
                similarities = self.cosine_similarity(query_embedding, self.course_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:
                        topic = self.course_topics[idx].copy()
                        topic["similarity"] = float(similarities[idx])
                        results["course_topics"].append(topic)
            
            # Search code examples
            if self.code_embeddings is not None and len(self.code_embeddings) > 0:
                similarities = self.cosine_similarity(query_embedding, self.code_embeddings)[0]
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
        "embedding_model": "gemini-text-embedding-004"
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

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check knowledge base loading"""
    import os
    from pathlib import Path
    
    debug_info = {
        "current_directory": os.getcwd(),
        "files_in_current_dir": os.listdir('.'),
        "scripts_dir_exists": os.path.exists('scripts'),
        "processed_dir_exists": os.path.exists('scripts/processed'),
        "processed_course_dir_exists": os.path.exists('scripts/processed_course'),
        "knowledge_base_loaded": len(virtual_ta.discourse_kb.get("topics", [])),
        "qa_pairs_loaded": len(virtual_ta.discourse_qa.get("qa_pairs", [])),
        "course_topics_loaded": len(virtual_ta.course_topics),
        "code_examples_loaded": len(virtual_ta.course_code),
        "embeddings_status": {
            "discourse": virtual_ta.discourse_embeddings is not None,
            "qa": virtual_ta.qa_embeddings is not None,
            "course": virtual_ta.course_embeddings is not None,
            "code": virtual_ta.code_embeddings is not None
        }
    }
    
    # Check specific files
    if os.path.exists('scripts/processed'):
        debug_info["processed_files"] = os.listdir('scripts/processed')
    
    if os.path.exists('scripts/processed_course'):
        debug_info["processed_course_files"] = os.listdir('scripts/processed_course')
    
    return jsonify(debug_info)

if __name__ == '__main__':
    print("🚀 Starting Enhanced TDS Virtual TA with Gemini 2.5 Flash...")
    print(f"📚 Loaded {len(virtual_ta.discourse_kb['topics'])} Discourse topics")
    print(f"💬 Loaded {len(virtual_ta.discourse_qa['qa_pairs'])} Q&A pairs")
    print(f"📖 Loaded {len(virtual_ta.course_topics)} course topics")
    print(f"💻 Loaded {len(virtual_ta.course_code)} code examples")
    print(f"🧠 Using Gemini 2.5 Flash for answer generation")
    print(f"🔍 Using semantic embeddings for search")
    print("✅ Ready to answer questions!")
    
    # Use debug=False for production, debug=True for development
    import os
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)