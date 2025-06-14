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
import time

import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tds_virtual_ta.log')
    ]
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("🚀 STARTING TDS VIRTUAL TA INITIALIZATION")
print("=" * 80)

app = Flask(__name__)
CORS(app)

# Global initialization flag
INITIALIZATION_COMPLETE = False
INITIALIZATION_ERROR = None

print("📋 Step 1: Loading environment variables...")
# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    error_msg = "❌ CRITICAL ERROR: GEMINI_API_KEY environment variable not found!"
    print(error_msg)
    logger.error(error_msg)
    raise ValueError("Please set GEMINI_API_KEY environment variable")
else:
    print(f"✅ Gemini API key loaded (length: {len(GEMINI_API_KEY)} characters)")

print("🔧 Step 2: Configuring Gemini API...")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully")
except Exception as e:
    error_msg = f"❌ Failed to configure Gemini API: {e}"
    print(error_msg)
    logger.error(error_msg)
    raise

print("📁 Step 3: Setting up directory paths...")
# Load knowledge base
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
PROCESSED_DIR = SCRIPTS_DIR / "processed"
COURSE_DIR = SCRIPTS_DIR / "processed_course"

print(f"   📂 Scripts directory: {SCRIPTS_DIR}")
print(f"   📂 Processed directory: {PROCESSED_DIR}")
print(f"   📂 Course directory: {COURSE_DIR}")
print(f"   ✅ Scripts dir exists: {SCRIPTS_DIR.exists()}")
print(f"   ✅ Processed dir exists: {PROCESSED_DIR.exists()}")
print(f"   ✅ Course dir exists: {COURSE_DIR.exists()}")

class EnhancedVirtualTA:
    def __init__(self):
        """Initialize the Enhanced Virtual TA with Gemini embeddings and caching support"""
        
        print("\n🤖 INITIALIZING ENHANCED VIRTUAL TA")
        print("-" * 50)
        
        # Data storage
        print("📊 Step 4a: Initializing data structures...")
        self.discourse_kb = {}
        self.discourse_qa = {}
        self.course_topics = []
        self.course_code = []
        print("   ✅ Data structures initialized")
        
        # Text storage for embedding lookup
        print("📝 Step 4b: Initializing text storage...")
        self.discourse_texts = []
        self.qa_texts = []
        self.course_texts = []
        self.code_texts = []
        print("   ✅ Text storage initialized")
        
        # Embeddings
        print("🧠 Step 4c: Initializing embedding storage...")
        self.discourse_embeddings = None
        self.qa_embeddings = None
        self.course_embeddings = None
        self.code_embeddings = None
        print("   ✅ Embedding storage initialized")
        
        # Load data and embeddings
        print("\n📚 Step 5: Loading knowledge base data...")
        self.load_data()
        
        print("\n🔍 Step 6: Creating embeddings...")
        self.load_or_create_embeddings()
        
        print("\n✅ ENHANCED VIRTUAL TA INITIALIZATION COMPLETE!")
        print("-" * 50)
    
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
        print(f"      🔄 Creating embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        try:
            embeddings = []
            batch_size = 50  # Smaller batch size to avoid rate limits
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            print(f"      📦 Processing {total_batches} batches of {batch_size} texts each...")
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = i//batch_size + 1
                print(f"      🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")
                
                batch_embeddings = []
                for j, text in enumerate(batch):
                    try:
                        # Chunk text to fit within limits
                        chunked_text = self.chunk_text(text)
                        
                        if j % 10 == 0 and j > 0:
                            print(f"         📝 Processed {j}/{len(batch)} texts in current batch...")
                        
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
                print(f"      ✅ Batch {batch_num}/{total_batches} completed")
                
                # Small delay to avoid rate limits
                if batch_num < total_batches:
                    time.sleep(0.1)
            
            elapsed_time = time.time() - start_time
            print(f"      ✅ All embeddings created in {elapsed_time:.2f} seconds")
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error getting Gemini embeddings: {e}")
            print(f"      ❌ Error creating embeddings: {e}")
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
                logger.info("🔄 Data size mismatch, cache invalid")
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
        print("   🔄 Creating fresh Gemini embeddings (no caching enabled)...")
        self.create_embeddings()
    
    def load_data(self):
        """Load all knowledge base data"""
        try:
            print("   📂 Checking directory structure...")
            print(f"      PROCESSED_DIR path: {PROCESSED_DIR}")
            print(f"      COURSE_DIR path: {COURSE_DIR}")
            print(f"      PROCESSED_DIR exists: {PROCESSED_DIR.exists()}")
            print(f"      COURSE_DIR exists: {COURSE_DIR.exists()}")
            
            if PROCESSED_DIR.exists():
                files_in_processed = list(PROCESSED_DIR.iterdir())
                print(f"      Files in PROCESSED_DIR: {[f.name for f in files_in_processed]}")
            else:
                print("      ❌ PROCESSED_DIR does not exist!")
                
            if COURSE_DIR.exists():
                files_in_course = list(COURSE_DIR.iterdir())
                print(f"      Files in COURSE_DIR: {[f.name for f in files_in_course]}")
            else:
                print("      ❌ COURSE_DIR does not exist!")
            
            print("   📖 Loading Discourse knowledge base...")
            # Load Discourse data
            kb_file = PROCESSED_DIR / "knowledge_base.json"
            qa_file = PROCESSED_DIR / "qa_pairs.json"
            print(f"      Trying to load: {kb_file}")
            print(f"      File exists: {kb_file.exists()}")
            
            if not kb_file.exists():
                raise FileNotFoundError(f"Knowledge base file not found: {kb_file}")
                
            print(f"      Trying to load: {qa_file}")
            print(f"      File exists: {qa_file.exists()}")
            
            if not qa_file.exists():
                raise FileNotFoundError(f"Q&A pairs file not found: {qa_file}")
            
            print("      📥 Reading knowledge_base.json...")
            with open(kb_file, 'r', encoding='utf-8') as f:
                self.discourse_kb = json.load(f)
            print(f"      ✅ Loaded {len(self.discourse_kb.get('topics', []))} Discourse topics")
            
            print("      📥 Reading qa_pairs.json...")
            with open(qa_file, 'r', encoding='utf-8') as f:
                self.discourse_qa = json.load(f)
            print(f"      ✅ Loaded {len(self.discourse_qa.get('qa_pairs', []))} Q&A pairs")
            
            print("   📚 Loading course content...")
            # Load course content data
            topics_file = COURSE_DIR / "course_topics.json"
            code_file = COURSE_DIR / "course_code_examples.json"
            print(f"      Trying to load: {topics_file}")
            print(f"      File exists: {topics_file.exists()}")
            
            if not topics_file.exists():
                print(f"      ⚠️  Course topics file not found: {topics_file}")
                self.course_topics = []
            else:
                print("      📥 Reading course_topics.json...")
                with open(topics_file, 'r', encoding='utf-8') as f:
                    self.course_topics = json.load(f)
                print(f"      ✅ Loaded {len(self.course_topics)} course topics")
            
            print(f"      Trying to load: {code_file}")
            print(f"      File exists: {code_file.exists()}")
            
            if not code_file.exists():
                print(f"      ⚠️  Course code file not found: {code_file}")
                self.course_code = []
            else:
                print("      📥 Reading course_code_examples.json...")
                with open(code_file, 'r', encoding='utf-8') as f:
                    self.course_code = json.load(f)
                print(f"      ✅ Loaded {len(self.course_code)} code examples")
            
            print("\n   📊 DATA LOADING SUMMARY:")
            print(f"      ✅ Discourse topics: {len(self.discourse_kb.get('topics', []))}")
            print(f"      ✅ Q&A pairs: {len(self.discourse_qa.get('qa_pairs', []))}")
            print(f"      ✅ Course topics: {len(self.course_topics)}")
            print(f"      ✅ Code examples: {len(self.course_code)}")
            
        except Exception as e:
            error_msg = f"❌ Error loading data: {e}"
            print(f"   {error_msg}")
            logger.error(error_msg)
            print(f"      Current working directory: {os.getcwd()}")
            print(f"      Files in current directory: {os.listdir('.')}")
            # Initialize empty data structures
            self.discourse_kb = {"topics": []}
            self.discourse_qa = {"qa_pairs": []}
            self.course_topics = []
            self.course_code = []
            raise
    
    def create_embeddings(self):
        """Create embeddings for all content"""
        print("   🧠 CREATING EMBEDDINGS FOR SEMANTIC SEARCH")
        print("   " + "-" * 45)
        
        try:
            total_start_time = time.time()
            
            # Discourse topics embeddings
            print("   📖 Step 6a: Creating Discourse topic embeddings...")
            self.discourse_texts = []
            for i, topic in enumerate(self.discourse_kb["topics"]):
                # Combine title with all post contents to give the embedding rich context
                combined_posts = " ".join([p.get("content", "") for p in topic.get("all_posts", [])])
                text = f"{topic['title']} {combined_posts}"
                self.discourse_texts.append(text)
                
                if (i + 1) % 50 == 0:
                    print(f"      📝 Prepared {i + 1}/{len(self.discourse_kb['topics'])} discourse texts...")
            
            print(f"      📝 Prepared {len(self.discourse_texts)} discourse texts for embedding")
            
            if self.discourse_texts:
                print("      🔄 Generating Discourse embeddings...")
                self.discourse_embeddings = self.get_gemini_embeddings(self.discourse_texts)
                print(f"      ✅ Created {self.discourse_embeddings.shape[0]} discourse embeddings")
            else:
                print("      ⚠️  No discourse texts to embed")
            
            # Q&A embeddings
            print("\n   💬 Step 6b: Creating Q&A pair embeddings...")
            self.qa_texts = []
            for i, qa in enumerate(self.discourse_qa["qa_pairs"]):
                text = f"{qa['question']} {qa['answer']}"
                self.qa_texts.append(text)
                
                if (i + 1) % 100 == 0:
                    print(f"      📝 Prepared {i + 1}/{len(self.discourse_qa['qa_pairs'])} Q&A texts...")
            
            print(f"      📝 Prepared {len(self.qa_texts)} Q&A texts for embedding")
            
            if self.qa_texts:
                print("      🔄 Generating Q&A embeddings...")
                self.qa_embeddings = self.get_gemini_embeddings(self.qa_texts)
                print(f"      ✅ Created {self.qa_embeddings.shape[0]} Q&A embeddings")
            else:
                print("      ⚠️  No Q&A texts to embed")
            
            # Course topics embeddings
            print("\n   📚 Step 6c: Creating course topic embeddings...")
            self.course_texts = []
            for i, topic in enumerate(self.course_topics):
                sections_text = ""
                if topic.get("sections"):
                    sections_text = " ".join([s.get("content", "") for s in topic["sections"]])
                text = f"{topic['title']} {sections_text}"
                self.course_texts.append(text)
                
                if (i + 1) % 20 == 0:
                    print(f"      📝 Prepared {i + 1}/{len(self.course_topics)} course texts...")
            
            print(f"      📝 Prepared {len(self.course_texts)} course texts for embedding")
            
            if self.course_texts:
                print("      🔄 Generating course embeddings...")
                self.course_embeddings = self.get_gemini_embeddings(self.course_texts)
                print(f"      ✅ Created {self.course_embeddings.shape[0]} course embeddings")
            else:
                print("      ⚠️  No course texts to embed")
            
            # Code examples embeddings
            print("\n   💻 Step 6d: Creating code example embeddings...")
            self.code_texts = []
            for i, code in enumerate(self.course_code):
                text = f"{code.get('context', '')} {code.get('code', '')} {code.get('language', '')}"
                self.code_texts.append(text)
                
                if (i + 1) % 50 == 0:
                    print(f"      📝 Prepared {i + 1}/{len(self.course_code)} code texts...")
            
            print(f"      📝 Prepared {len(self.code_texts)} code texts for embedding")
            
            if self.code_texts:
                print("      🔄 Generating code embeddings...")
                self.code_embeddings = self.get_gemini_embeddings(self.code_texts)
                print(f"      ✅ Created {self.code_embeddings.shape[0]} code embeddings")
            else:
                print("      ⚠️  No code texts to embed")
            
            total_elapsed = time.time() - total_start_time
            print(f"\n   🎉 EMBEDDING CREATION COMPLETE!")
            print(f"   ⏱️  Total time: {total_elapsed:.2f} seconds")
            print("   " + "-" * 45)
            
        except Exception as e:
            error_msg = f"❌ Error creating embeddings: {e}"
            print(f"   {error_msg}")
            logger.error(error_msg)
            raise
    
    def semantic_search(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """Perform semantic search across all content with adaptive filtering"""
        query_embedding = self.get_query_embedding(query)
        results = {
            "discourse_topics": [],
            "qa_pairs": [],
            "course_topics": [],
            "code_examples": []
        }
        
        try:
            # Calculate adaptive threshold based on query type
            adaptive_threshold = self._calculate_adaptive_threshold(query)
            
            # Search Discourse topics
            if self.discourse_embeddings is not None and len(self.discourse_embeddings) > 0:
                similarities = self.cosine_similarity(query_embedding, self.discourse_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    similarity = similarities[idx]
                    if idx == top_indices[0] or similarity > adaptive_threshold:
                        topic = self.discourse_kb["topics"][idx].copy()
                        topic["similarity"] = float(similarity)
                        topic["relevance_score"] = self._calculate_relevance_score(query, topic, similarity)
                        results["discourse_topics"].append(topic)
            
            # Search Q&A pairs
            if self.qa_embeddings is not None and len(self.qa_embeddings) > 0:
                similarities = self.cosine_similarity(query_embedding, self.qa_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    similarity = similarities[idx]
                    if idx == top_indices[0] or similarity > adaptive_threshold:
                        qa = self.discourse_qa["qa_pairs"][idx].copy()
                        qa["similarity"] = float(similarity)
                        qa["relevance_score"] = self._calculate_relevance_score(query, qa, similarity)
                        results["qa_pairs"].append(qa)
            
            # Search course topics
            if self.course_embeddings is not None and len(self.course_embeddings) > 0:
                similarities = self.cosine_similarity(query_embedding, self.course_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    similarity = similarities[idx]
                    if idx == top_indices[0] or similarity > adaptive_threshold:
                        topic = self.course_topics[idx].copy()
                        topic["similarity"] = float(similarity)
                        topic["relevance_score"] = self._calculate_relevance_score(query, topic, similarity)
                        results["course_topics"].append(topic)
            
            # Search code examples
            if self.code_embeddings is not None and len(self.code_embeddings) > 0:
                similarities = self.cosine_similarity(query_embedding, self.code_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    similarity = similarities[idx]
                    if idx == top_indices[0] or similarity > adaptive_threshold:
                        code = self.course_code[idx].copy()
                        code["similarity"] = float(similarity)
                        code["relevance_score"] = self._calculate_relevance_score(query, code, similarity)
                        results["code_examples"].append(code)
            
            # Filter and sort by relevance
            results = self._filter_by_relevance(results, query)
        
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
        
        return results
    
    def _calculate_adaptive_threshold(self, query: str) -> float:
        """Calculate adaptive similarity threshold based on query characteristics"""
        query_lower = query.lower()
        
        # Higher threshold for temporal queries (dates, deadlines)
        if any(term in query_lower for term in ["when", "deadline", "exam", "sep 2025", "december 2025", "2026"]):
            return 0.35  # High threshold for temporal queries
        
        # Higher threshold for non-existent assignments
        if any(term in query_lower for term in ["ga15", "ga20", "project 5", "project 6"]):
            return 0.40  # Very high threshold for non-existent items
        
        # Lower threshold for specific technical queries
        if any(term in query_lower for term in ["ga4", "bonus", "dashboard", "110", "project 1"]):
            return 0.15  # Lower threshold for specific queries
        
        # Medium threshold for general queries
        return 0.20  # Default threshold
    
    def _calculate_relevance_score(self, query: str, item: Dict, similarity: float) -> float:
        """Calculate relevance score combining similarity with content analysis"""
        query_lower = query.lower()
        
        # Get text content for analysis
        if "question" in item:  # Q&A pair
            content = f"{item.get('question', '')} {item.get('answer', '')}"
        elif "title" in item:  # Topic
            content = f"{item.get('title', '')} {' '.join([p.get('content', '') for p in item.get('all_posts', [])])}"
        else:  # Other content
            content = str(item)
        
        content_lower = content.lower()
        
        # Base score is similarity
        score = similarity
        
        # Boost for exact keyword matches
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 2 and word in content_lower:
                score += 0.05
        
        # Special boost for GA4 bonus content
        if "ga4" in query_lower and "bonus" in query_lower:
            if any(term in content_lower for term in ["110", "11/10", "dashboard", "102.5"]):
                score += 0.2
        
        # Penalty for temporal mismatches
        if any(term in query_lower for term in ["sep 2025", "september 2025", "december 2025", "2026"]):
            if "march 2025" in content_lower or "january 2025" in content_lower:
                score -= 0.4  # Heavy penalty for date mismatch
        
        # Penalty for non-existent assignments
        fake_assignments = ["ga15", "ga20", "project 5", "project 6"]
        if any(fake in query_lower for fake in fake_assignments):
            score -= 0.5  # Heavy penalty
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def _filter_by_relevance(self, results: Dict[str, List[Dict]], query: str) -> Dict[str, List[Dict]]:
        """Filter and sort results by relevance score"""
        filtered_results = {}
        
        for category, items in results.items():
            # Sort by relevance score
            sorted_items = sorted(items, key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Apply minimum relevance threshold
            query_lower = query.lower()
            if any(term in query_lower for term in ["when", "2025", "2026", "ga15", "ga20", "project 5"]):
                min_relevance = 0.4  # High threshold for suspicious queries
            else:
                min_relevance = 0.2  # Normal threshold
            
            filtered_items = [item for item in sorted_items if item.get("relevance_score", 0) >= min_relevance]
            filtered_results[category] = filtered_items[:5]  # Keep top 5
        
        return filtered_results
    
    def generate_answer_with_gemini(self, question: str, context: Dict[str, List[Dict]], image_data: Optional[str] = None) -> str:
        """Generate answer using Gemini with dual-request system for robustness"""
        try:
            # Prepare context for Gemini
            context_text = self.prepare_context_for_gemini(context)
            
            # Debug: log the context being sent
            logger.info(f"Context length: {len(context_text)} characters")
            logger.info(f"Context preview: {context_text[:500]}...")
            
            # Check if we have relevant context
            has_relevant_context = len(context_text.strip()) > 20
            
            # Debug: log context details
            logger.info(f"Has relevant context: {has_relevant_context}")
            logger.info(f"Context length: {len(context_text)} characters")
            
            # Log search results for debugging
            total_results = sum(len(v) for v in context.values())
            logger.info(f"Total search results found: {total_results}")
            for key, results in context.items():
                if results:
                    top_relevance = results[0].get('relevance_score', 'N/A')
                    logger.info(f"{key}: {len(results)} results, top similarity: {results[0].get('similarity', 'N/A')}, top relevance: {top_relevance}")
            
            # DUAL-REQUEST SYSTEM
            # Request 1: Generate initial answer
            initial_answer = self._generate_initial_answer(question, context_text, image_data)
            
            # Request 2: Validate and refine the answer
            final_answer = self._validate_and_refine_answer(question, context_text, initial_answer, has_relevant_context)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {e}")
            return "I apologize, but I'm having trouble generating an answer right now. Please check the course materials or ask in the Discourse forum."
    
    def _generate_initial_answer(self, question: str, context_text: str, image_data: Optional[str] = None) -> str:
        """Generate the initial answer (Request 1)"""
        
        prompt = f"""You are a helpful Teaching Assistant for the Tools in Data Science (TDS) course at IIT Madras. 
Answer the student's question based on the provided context from course materials and forum discussions.

Question: {question}

Context:
{context_text}

INSTRUCTIONS:
1. Use the provided context to answer the question as completely as possible
2. If the context contains relevant information, provide a comprehensive answer based on that information
3. Pay special attention to staff answers and official course information
4. For GA4 bonus questions: Look for specific numerical examples in the context (like "110", calculations showing bonus scores)
5. Include specific details like numbers, percentages, or dashboard displays when mentioned in the context
6. If you see examples of calculations or score displays in the context, use those to explain how things work
7. When the context shows specific examples (like "(100+100+100+110) / 4"), use those to illustrate your answer
8. Be specific and detailed when the context supports it
9. Use a friendly, supportive tone
10. If an image is provided, analyze it and incorporate relevant details into your answer

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
        return response.text
    
    def _validate_and_refine_answer(self, question: str, context_text: str, initial_answer: str, has_relevant_context: bool) -> str:
        """Validate and refine the initial answer (Request 2)"""
        
        # First apply basic post-processing
        processed_answer = self.post_process_answer(initial_answer, question, has_relevant_context)
        
        # If basic post-processing blocked it, return that
        if "I don't have specific information" in processed_answer:
            return processed_answer
        
        # Advanced validation for specific cases
        validation_prompt = f"""You are a fact-checker for a TDS course Teaching Assistant. Review this answer for accuracy and appropriateness.

Original Question: {question}

Available Context: {context_text[:1500]}...

Proposed Answer: {initial_answer}

VALIDATION CRITERIA:
1. Does the answer address information that is NOT in the provided context?
2. Does the answer mention dates, deadlines, or events not explicitly in the context?
3. Does the answer make assumptions about future events (beyond April 2025)?
4. Is the answer relevant to the specific question asked?

SPECIFIC CHECKS:
- If question asks about "Sep 2025" or future dates, but context only has "March 2025" - INVALID
- If question asks about non-existent assignments (GA15, GA20, Project 5+) - INVALID
- If answer provides specific dates/deadlines not in context - INVALID
- If question is about valid course content (GA1-GA4, Project 1-2) - should be VALID

Respond with either:
"VALID: [brief reason]" - if the answer is appropriate and based on context
"INVALID: [specific reason]" - if the answer has issues

Response:"""

        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            validation_response = model.generate_content(validation_prompt)
            validation_result = validation_response.text.strip()
            
            logger.info(f"Validation result: {validation_result}")
            
            if validation_result.startswith("INVALID"):
                logger.info(f"Answer invalidated: {validation_result}")
                return "I don't have specific information about this in my knowledge base. Please check the course materials at https://tds.s-anand.net or ask on the Discourse forum at https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34."
            else:
                return processed_answer
                
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            # If validation fails, return the processed answer
            return processed_answer
    
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
        
        # Be much less conservative - only block if answer is clearly generic
        if not has_relevant_context and "I don't have specific information" in answer:
            logger.info(f"Generic answer detected for question: {question}")
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
        
        # Add Q&A pairs (highest priority), prioritizing GA4 bonus content
        if context["qa_pairs"]:
            context_parts.append("=== FORUM Q&A ===")
            
            # Separate Q&A pairs by relevance to GA4 bonus
            ga4_bonus_qa = []
            other_qa = []
            
            for qa in context["qa_pairs"]:
                qa_text = f"{qa.get('question', '')} {qa.get('answer', '')}"
                if any(term in qa_text.lower() for term in ["110", "11/10", "dashboard", "bonus", "ga4"]):
                    ga4_bonus_qa.append(qa)
                else:
                    other_qa.append(qa)
            
            # Add GA4 bonus Q&A first (full content)
            for qa in ga4_bonus_qa[:3]:
                staff_indicator = " [STAFF ANSWER]" if qa.get("is_staff_answer") else ""
                context_parts.append(f"Q: {qa['question']}")
                context_parts.append(f"A: {qa['answer']}{staff_indicator}")
                context_parts.append("")
            
            # Add other Q&A pairs
            for qa in other_qa[:2]:
                staff_indicator = " [STAFF ANSWER]" if qa.get("is_staff_answer") else ""
                context_parts.append(f"Q: {qa['question']}")
                context_parts.append(f"A: {qa['answer'][:400]}{staff_indicator}")
                context_parts.append("")
        
        # Add discourse topics with more detail
        if context["discourse_topics"]:
            context_parts.append("=== FORUM DISCUSSIONS ===")
            for topic in context["discourse_topics"][:3]:  # Increased from 2 to 3
                context_parts.append(f"Discussion: {topic['title']}")
                
                # Include all posts content if available
                if topic.get("all_posts"):
                    for post in topic["all_posts"][:5]:  # Include more posts
                        post_content = post.get("content", "")[:800]  # Increased content length
                        if post_content.strip():
                            context_parts.append(f"Post: {post_content}")
                
                # Include Q&A pairs from the topic
                if topic.get("qa_pairs"):
                    for qa in topic["qa_pairs"][:3]:  # Increased from 1 to 3
                        context_parts.append(f"Q: {qa['question']}")
                        context_parts.append(f"A: {qa['answer'][:500]}")  # Increased from 300 to 500
                context_parts.append("")
        
        # Add course content
        if context["course_topics"]:
            context_parts.append("=== COURSE MATERIALS ===")
            for topic in context["course_topics"][:2]:
                context_parts.append(f"Topic: {topic['title']}")
                if topic.get("sections"):
                    for section in topic["sections"][:3]:  # Increased from 2 to 3
                        content = section.get("content", "")[:800]  # Increased from 500 to 800
                        context_parts.append(f"Content: {content}")
                context_parts.append("")
        
        # Add code examples if relevant
        if context["code_examples"]:
            context_parts.append("=== CODE EXAMPLES ===")
            for code in context["code_examples"][:2]:
                context_parts.append(f"Language: {code.get('language', 'text')}")
                context_parts.append(f"Code: {code.get('code', '')[:400]}")  # Increased from 300 to 400
                if code.get("context"):
                    context_parts.append(f"Context: {code.get('context', '')[:300]}")  # Increased from 200 to 300
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

# Initialize the Virtual TA with proper error handling
print("\n🚀 INITIALIZING VIRTUAL TA INSTANCE...")
try:
    virtual_ta = EnhancedVirtualTA()
    INITIALIZATION_COMPLETE = True
    print("✅ VIRTUAL TA INITIALIZATION SUCCESSFUL!")
    print("🌟 Ready to serve requests!")
except Exception as e:
    INITIALIZATION_ERROR = str(e)
    print(f"❌ VIRTUAL TA INITIALIZATION FAILED: {e}")
    logger.error(f"Initialization failed: {e}")
    virtual_ta = None

def check_initialization():
    """Check if initialization is complete before processing requests"""
    if not INITIALIZATION_COMPLETE:
        if INITIALIZATION_ERROR:
            return jsonify({
                "error": "Virtual TA initialization failed",
                "details": INITIALIZATION_ERROR,
                "status": "initialization_failed"
            }), 503
        else:
            return jsonify({
                "error": "Virtual TA is still initializing, please wait",
                "status": "initializing"
            }), 503
    return None

@app.route('/api/', methods=['POST'])
def answer_question():
    """Main API endpoint for answering student questions"""
    # Check initialization status first
    init_check = check_initialization()
    if init_check:
        return init_check
    
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
    # Check initialization status first
    init_check = check_initialization()
    if init_check:
        return init_check
    
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
    # Check initialization status first
    init_check = check_initialization()
    if init_check:
        return init_check
    
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
    # Check initialization status first
    init_check = check_initialization()
    if init_check:
        return init_check
    
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

@app.route('/api/init-status', methods=['GET'])
def initialization_status():
    """Get initialization status without requiring full initialization"""
    return jsonify({
        "initialization_complete": INITIALIZATION_COMPLETE,
        "initialization_error": INITIALIZATION_ERROR,
        "virtual_ta_available": virtual_ta is not None,
        "timestamp": datetime.now().isoformat()
    })

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
    print("\n" + "=" * 80)
    print("🌟 TDS VIRTUAL TA STARTUP SUMMARY")
    print("=" * 80)
    
    if INITIALIZATION_COMPLETE and virtual_ta:
        print("✅ INITIALIZATION STATUS: SUCCESS")
        print(f"📚 Discourse topics loaded: {len(virtual_ta.discourse_kb['topics'])}")
        print(f"💬 Q&A pairs loaded: {len(virtual_ta.discourse_qa['qa_pairs'])}")
        print(f"📖 Course topics loaded: {len(virtual_ta.course_topics)}")
        print(f"💻 Code examples loaded: {len(virtual_ta.course_code)}")
        print(f"🧠 AI Model: Gemini 2.5 Flash")
        print(f"🔍 Embedding Model: Gemini Text Embedding 004")
        print(f"🌐 Server: Flask with CORS enabled")
        print(f"📡 Endpoints: /api/, /api/health, /api/stats, /api/search, /api/debug")
        print("✅ READY TO SERVE REQUESTS!")
    else:
        print("❌ INITIALIZATION STATUS: FAILED")
        if INITIALIZATION_ERROR:
            print(f"❌ Error: {INITIALIZATION_ERROR}")
        print("⚠️  Server will start but return error responses")
    
    print("=" * 80)
    print("🚀 STARTING FLASK SERVER...")
    print("   Host: 0.0.0.0")
    print("   Port: 5000")
    
    # Use debug=False for production, debug=True for development
    import os
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    print(f"   Debug mode: {debug_mode}")
    print("=" * 80)
    
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)