#!/usr/bin/env python3
"""
Professional TDS Virtual TA with Advanced LLM Optimization
Uses sophisticated semantic search, dynamic context ranking, and intelligent prompt engineering.
"""

import os
import json
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Mock numpy for basic operations
    class MockNumpy:
        def array(self, data):
            return data
        def zeros(self, shape):
            if isinstance(shape, int):
                return [0.0] * shape
            return [0.0] * shape[0]
        def load(self, file):
            raise ImportError("Numpy not available for loading embeddings")
        def save(self, file, data):
            pass  # Skip saving
        def linalg(self):
            return self
        def norm(self, data, axis=None):
            return 1.0
        def dot(self, a, b):
            return 0.5
        def any(self, data):
            return True
        def max(self, data, axis=None):
            return [0.5] * len(data[0]) if axis == 0 else 0.5
    np = MockNumpy()
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import time
import re
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TDSVirtualTA:
    def __init__(self):
        print("🚀 Initializing TDS Virtual TA...")
        
        # Load environment
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        print("✅ Gemini API configured")
        
        # Initialize data
        self.discourse_data = None
        self.course_data = None
        self.embeddings = None
        
        # Load data
        self.load_data()
        self.create_embeddings()
        
        print("✅ TDS Virtual TA ready!")
    
    def load_data(self):
        """Load discourse and course data"""
        print("📚 Loading data...")
        
        # Load discourse data
        discourse_file = Path("data/discourse_data.json")
        if discourse_file.exists():
            with open(discourse_file, 'r', encoding='utf-8') as f:
                self.discourse_data = json.load(f)
            print(f"✅ Loaded {len(self.discourse_data['topics'])} discourse topics")
        else:
            print("❌ Discourse data not found. Run scrape_discourse.py first.")
            self.discourse_data = {"topics": [], "all_qa_pairs": []}
        
        # Augment discourse data with any additional raw topics present in scripts/raw
        raw_dir = Path("scripts/raw")
        if raw_dir.exists():
            existing_ids = {t["topic_id"] for t in self.discourse_data.get("topics", []) if "topic_id" in t}
            new_topics_added = 0
            for raw_file in raw_dir.glob("topic-*.json"):
                try:
                    with open(raw_file, "r", encoding="utf-8") as rf:
                        raw_topic = json.load(rf)
                    topic_id = raw_topic.get("topic_id") or raw_topic.get("id")
                    if topic_id is None or topic_id in existing_ids:
                        continue
                        
                    # Correctly get slug: from data first, then fallback to title
                    title = raw_topic.get("title", f"Topic {topic_id}")
                    slug = raw_topic.get("slug") or raw_topic.get("topic_slug")
                    if not slug:
                        slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
                    
                    # Sanitize slug from either source to ensure it's clean
                    if slug:
                        slug = re.sub(r'-+', '-', slug)

                    posts = raw_topic.get("post_stream", {}).get("posts", [])
                    full_content = "\n\n".join(p.get("cooked", "") for p in posts)
                    url = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}"
                    self.discourse_data["topics"].append({
                        "topic_id": topic_id,
                        "title": title,
                        "full_content": full_content,
                        "url": url
                    })
                    existing_ids.add(topic_id)
                    new_topics_added += 1
                except Exception as e:
                    logger.warning(f"Error processing {raw_file}: {e}")
            if new_topics_added:
                print(f"✅ Added {new_topics_added} raw discourse topics from scripts/raw")
        
        # Load course data from the tools-in-data-science-public directory
        course_dir = Path("tools-in-data-science-public")
        self.course_data = []
        
        if course_dir.exists():
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
        else:
            print("❌ Course data directory not found")
            self.course_data = []
    
    def create_embeddings(self):
        """Create embeddings for all content using Gemini"""
        # If cache doesn't exist, create embeddings
        print("🧠 Creating embeddings with Gemini...")
        
        # Check for cached embeddings first
        embeddings_file = Path("embeddings_cache/embeddings.npy")
        metadata_file = Path("embeddings_cache/metadata.json")
        
        if embeddings_file.exists() and metadata_file.exists() and HAS_NUMPY:
            print("📁 Loading cached embeddings...")
            try:
                self.embeddings = np.load(embeddings_file)
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.metadata = cache_data['items']
                print(f"✅ Loaded cached embeddings: {self.embeddings.shape}")
                return
            except Exception as e:
                print(f"⚠️ Error loading cached embeddings: {e}")
        elif not HAS_NUMPY:
            print("⚠️ Numpy not available - embeddings will be generated fresh")
        
        # Prepare all texts for embedding (local development only)
        texts = []
        metadata = []
        
        # Add discourse topics
        for topic in self.discourse_data.get("topics", []):
            # Combine title and content for better context
            text = f"{topic['title']} {topic['full_content'][:2000]}"  # Limit length
            texts.append(text)
            metadata.append({
                "type": "discourse_topic",
                "topic_id": topic["topic_id"],
                "title": topic["title"],
                "url": topic["url"],
                "data": topic
            })
        
        # Add Q&A pairs
        for qa in self.discourse_data.get("all_qa_pairs", []):
            text = f"Q: {qa['question']} A: {qa['answer']}"
            texts.append(text)
            metadata.append({
                "type": "qa_pair",
                "topic_url": qa["topic_url"],
                "title": qa["topic_title"],
                "data": qa
            })
        
        # Add course content
        for course in self.course_data:
            text = f"{course['title']} {course['content'][:2000]}"  # Limit length
            texts.append(text)
            metadata.append({
                "type": "course_content",
                "title": course["title"],
                "url": course["url"],
                "data": course
            })
        
        print(f"📊 Creating embeddings for {len(texts)} items...")
        
        # Create embeddings in batches
        embeddings = []
        batch_size = 100  # Gemini API limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
            
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=batch,
                    task_type="retrieval_document"
                )
                embeddings.extend(result['embedding'])
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i}: {e}")
                # Add zero vectors as fallback
                embeddings.extend([np.zeros(768).tolist() for _ in batch])
        
        self.embeddings = np.array(embeddings)
        self.metadata = metadata
        
        print(f"✅ Created embeddings: {self.embeddings.shape}")
        
        # Cache the embeddings
        try:
            os.makedirs("embeddings_cache", exist_ok=True)
            np.save(embeddings_file, self.embeddings)
            
            cache_data = {
                'model': 'text-embedding-004',
                'created_at': time.time(),
                'items': self.metadata
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            print("💾 Cached embeddings for future use")
        except Exception as e:
            logger.warning(f"Could not cache embeddings: {e}")
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms for better semantic matching"""
        query_lower = query.lower()
        expanded_queries = [query]
        
        # Technical term expansions
        expansions = {
            'gpt': ['gpt-3.5-turbo', 'gpt-4o-mini', 'openai model', 'language model'],
            'ga4': ['graded assignment 4', 'assignment 4', 'ga-4'],
            'ga5': ['graded assignment 5', 'assignment 5', 'ga-5'],
            'docker': ['containerization', 'container technology'],
            'podman': ['container runtime', 'docker alternative'],
            'bonus': ['extra credit', 'additional points', 'bonus points'],
            'exam': ['test', 'assessment', 'evaluation'],
            'dashboard': ['gradebook', 'scores', 'grades'],
            'model': ['ai model', 'machine learning model', 'llm']
        }
        
        for term, synonyms in expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    if synonym not in query_lower:
                        expanded_queries.append(f"{query} {synonym}")
        
        return expanded_queries[:3]  # Limit to avoid too many queries
    
    def _extract_key_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract key entities from query for better matching"""
        query_lower = query.lower()
        
        entities = {
            'assignments': [],
            'models': [],
            'tools': [],
            'concepts': [],
            'numbers': []
        }
        
        # Assignment patterns
        assignment_patterns = [r'ga\s*(\d+)', r'assignment\s*(\d+)', r'graded\s*assignment\s*(\d+)']
        for pattern in assignment_patterns:
            matches = re.findall(pattern, query_lower)
            entities['assignments'].extend([f"ga{m}" for m in matches])
        
        # Model patterns
        model_patterns = [r'gpt-[\d\.]+-?\w*', r'gpt\s*[\d\.]+', r'openai']
        for pattern in model_patterns:
            matches = re.findall(pattern, query_lower)
            entities['models'].extend(matches)
        
        # Tool patterns
        tool_patterns = [r'docker', r'podman', r'container']
        for pattern in tool_patterns:
            if re.search(pattern, query_lower):
                entities['tools'].append(pattern)
        
        # Number patterns (scores, dates, etc.)
        number_patterns = [r'\b\d{2,4}\b']
        for pattern in number_patterns:
            matches = re.findall(pattern, query_lower)
            entities['numbers'].extend(matches)
        
        return entities
    
    def _calculate_semantic_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate semantic similarity with improved normalization"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return np.array([])
        
        # Normalize embeddings
        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        # Handle zero vectors
        valid_indices = embedding_norms > 0
        if not np.any(valid_indices) or query_norm == 0:
            return np.zeros(len(self.embeddings))
        
        # Calculate cosine similarity
        similarities = np.zeros(len(self.embeddings))
        similarities[valid_indices] = np.dot(self.embeddings[valid_indices], query_embedding) / (
            embedding_norms[valid_indices] * query_norm
        )
        
        return similarities
    
    def _calculate_lexical_similarity(self, query: str, metadata: Dict) -> float:
        """Calculate lexical similarity using TF-IDF-like approach"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Get text content based on metadata type
        if metadata['type'] == 'discourse_topic':
            content = f"{metadata['title']} {metadata['data'].get('full_content', '')}"
        elif metadata['type'] == 'qa_pair':
            content = f"{metadata['data'].get('question', '')} {metadata['data'].get('answer', '')}"
        elif metadata['type'] == 'course_content':
            content = f"{metadata['title']} {metadata['data'].get('content', '')}"
        else:
            content = ""
        
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
        if not query_words or not content_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_entity_match_score(self, entities: Dict, metadata: Dict) -> float:
        """Calculate score based on entity matching"""
        score = 0.0
        
        # Get content for entity matching
        if metadata['type'] == 'discourse_topic':
            content = f"{metadata['title']} {metadata['data'].get('full_content', '')}".lower()
            url = metadata['data'].get('url', '').lower()
        elif metadata['type'] == 'qa_pair':
            content = f"{metadata['data'].get('question', '')} {metadata['data'].get('answer', '')}".lower()
            url = metadata.get('topic_url', '').lower()
        elif metadata['type'] == 'course_content':
            content = f"{metadata['title']} {metadata['data'].get('content', '')}".lower()
            url = metadata['data'].get('url', '').lower()
        else:
            content = ""
            url = ""
        
        # Assignment matching
        for assignment in entities['assignments']:
            if assignment in content or assignment in url:
                score += 0.3
        
        # Model matching
        for model in entities['models']:
            if model in content:
                score += 0.25
        
        # Tool matching
        for tool in entities['tools']:
            if tool in content:
                score += 0.2
        
        # Number matching (for scores, dates, etc.)
        for number in entities['numbers']:
            if number in content:
                score += 0.15
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _rank_results_dynamically(self, query: str, similarities: np.ndarray, entities: Dict) -> List[Tuple[int, float]]:
        """Dynamically rank results using multiple signals"""
        ranked_results = []
        
        for idx, semantic_sim in enumerate(similarities):
            if semantic_sim < 0.01:  # Skip very low similarity
                continue
            
            metadata = self.metadata[idx]
            
            # Calculate different similarity components
            lexical_sim = self._calculate_lexical_similarity(query, metadata)
            entity_score = self._calculate_entity_match_score(entities, metadata)
            
            # Content type weighting
            type_weights = {
                'qa_pair': 1.2,      # Prioritize Q&A pairs for specific questions
                'discourse_topic': 1.0,
                'course_content': 0.9
            }
            type_weight = type_weights.get(metadata['type'], 1.0)
            
            # Calculate final score with adaptive weighting
            final_score = (
                0.5 * semantic_sim +      # Semantic understanding
                0.25 * lexical_sim +      # Exact term matching
                0.25 * entity_score       # Entity recognition
            ) * type_weight
            
            ranked_results.append((idx, final_score))
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return ranked_results
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Advanced semantic search with query expansion and dynamic ranking"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        try:
            # Expand query for better coverage
            expanded_queries = self._expand_query(query)
            
            # Extract key entities
            entities = self._extract_key_entities(query)
            
            # Get embeddings for all query variations
            all_similarities = []
            for expanded_query in expanded_queries:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=expanded_query,
                    task_type="retrieval_query"
                )
                query_embedding = np.array(result['embedding'])
                similarities = self._calculate_semantic_similarity(query_embedding)
                all_similarities.append(similarities)
            
            # Combine similarities (max pooling for best match across expansions)
            combined_similarities = np.max(all_similarities, axis=0)
            
            # Dynamic ranking
            ranked_results = self._rank_results_dynamically(query, combined_similarities, entities)
            
            # Prepare final results
            results = []
            for idx, final_score in ranked_results[:top_k * 2]:  # Get more candidates
                metadata = self.metadata[idx].copy()
                metadata["similarity"] = final_score
                metadata["semantic_similarity"] = float(combined_similarities[idx])
                results.append(metadata)
            
            # Keep all results (diversity) and rely on ranking to pick top_k
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    def _analyze_question_intent(self, question: str) -> Dict[str, any]:
        """Analyze question to understand intent and requirements"""
        question_lower = question.lower()
        
        intent = {
            'type': 'general',
            'requires_specific_info': False,
            'is_about_future': False,
            'needs_technical_details': False,
            'is_troubleshooting': False,
            'confidence_required': 'medium'
        }
        
        # Future-related questions
        future_indicators = ['2025', 'future', 'upcoming', 'next', 'when will']
        if any(indicator in question_lower for indicator in future_indicators):
            intent['is_about_future'] = True
            intent['confidence_required'] = 'high'
        
        # Technical questions
        technical_indicators = ['model', 'gpt', 'docker', 'podman', 'api', 'code']
        if any(indicator in question_lower for indicator in technical_indicators):
            intent['needs_technical_details'] = True
            intent['type'] = 'technical'
        
        # Specific assignment questions
        assignment_indicators = ['ga4', 'ga5', 'assignment', 'bonus', 'score']
        if any(indicator in question_lower for indicator in assignment_indicators):
            intent['requires_specific_info'] = True
            intent['type'] = 'assignment'
        
        # Troubleshooting questions
        trouble_indicators = ['error', 'problem', 'issue', 'not working', 'help']
        if any(indicator in question_lower for indicator in trouble_indicators):
            intent['is_troubleshooting'] = True
            intent['type'] = 'troubleshooting'
        
        return intent
    
    def _build_dynamic_context(self, search_results: List[Dict], intent: Dict) -> str:
        """Build context dynamically based on question intent and result relevance"""
        if not search_results:
            return "No relevant context found."
        
        context_parts = []
        
        # Sort results by relevance
        sorted_results = sorted(search_results, key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Adaptive context building based on intent
        if intent['type'] == 'assignment':
            # Prioritize Q&A pairs and specific discussions
            qa_results = [r for r in sorted_results if r["type"] == "qa_pair"]
            topic_results = [r for r in sorted_results if r["type"] == "discourse_topic"]
            
            if qa_results:
                context_parts.append("=== RELEVANT Q&A FROM FORUM ===")
                for qa in qa_results[:2]:
                    data = qa["data"]
                    context_parts.append(f"Q: {data['question']}")
                    context_parts.append(f"A: {data['answer']}")
                    context_parts.append(f"[Relevance: {qa.get('similarity', 0):.3f}]")
                context_parts.append("")
                
            if topic_results:
                context_parts.append("=== FORUM DISCUSSIONS ===")
                for topic in topic_results[:1]:
                    data = topic["data"]
                    context_parts.append(f"Topic: {data['title']}")
                    context_parts.append(f"Content: {data['full_content'][:1000]}...")
                    context_parts.append("")
        
        elif intent['type'] == 'technical':
            # Prioritize course content and technical discussions
            course_results = [r for r in sorted_results if r["type"] == "course_content"]
            other_results = [r for r in sorted_results if r["type"] != "course_content"]
            
            if course_results:
                context_parts.append("=== COURSE MATERIALS ===")
                for course in course_results[:2]:
                    data = course["data"]
                    context_parts.append(f"Topic: {data['title']}")
                    context_parts.append(f"Content: {data['content'][:1000]}...")
                    context_parts.append("")
            
            if other_results:
                context_parts.append("=== FORUM DISCUSSIONS ===")
                for result in other_results[:2]:
                    if result["type"] == "qa_pair":
                        data = result["data"]
                        context_parts.append(f"Q: {data['question']}")
                        context_parts.append(f"A: {data['answer']}")
                    elif result["type"] == "discourse_topic":
                        data = result["data"]
                        context_parts.append(f"Topic: {data['title']}")
                        context_parts.append(f"Content: {data['full_content'][:800]}...")
                        context_parts.append("")
                
        else:
            # General approach - balanced mix
            context_parts.append("=== RELEVANT INFORMATION ===")
            for result in sorted_results[:4]:
                if result["type"] == "qa_pair":
                    data = result["data"]
                    context_parts.append(f"Q: {data['question']}")
                    context_parts.append(f"A: {data['answer']}")
                elif result["type"] == "discourse_topic":
                    data = result["data"]
                    context_parts.append(f"Topic: {data['title']}")
                    context_parts.append(f"Content: {data['full_content'][:600]}...")
                elif result["type"] == "course_content":
                    data = result["data"]
                    context_parts.append(f"Course Topic: {data['title']}")
                    context_parts.append(f"Content: {data['content'][:600]}...")
                
                context_parts.append(f"[Relevance: {result.get('similarity', 0):.3f}]")
                context_parts.append("")
                
        return "\n".join(context_parts)
    
    def _create_adaptive_prompt(self, question: str, context: str, intent: Dict) -> str:
        """Create adaptive prompt based on question intent and context quality"""
        
        base_instructions = """You are a professional Teaching Assistant for the Tools in Data Science (TDS) course at IIT Madras.

Your primary goal is to provide accurate, helpful answers based ONLY on the provided context from course materials and forum discussions."""
        
        # This function is now a stub, returning a simplified generic prompt.
        # The complex logic has been removed to rely on embedding quality.
        specific_instructions = """
Provide a comprehensive answer by:
1. Using all relevant information from the context
2. Organizing your response clearly
3. Being helpful and supportive in tone
4. Acknowledging any limitations in the available information"""
        
        return f"""{base_instructions}

{specific_instructions}

Question: {question}

Context:
{context}

Guidelines:
- Use ONLY the provided context - do not add external information
- Be precise and accurate in your response
- Include relevant links and references when available in context
- If context is insufficient, clearly state the limitations
- Maintain a helpful and professional tone

Answer:"""

    def generate_answer(self, question: str, search_results: List[Dict], image_path: str = None) -> Dict:
        """Simplified answer generation with minimal processing"""
        
        # Handle image if provided
        image_content = None
        if image_path and image_path.startswith('file://'):
            try:
                import base64
                from pathlib import Path
                
                # Remove file:// prefix and get actual path
                actual_path = image_path.replace('file://', '')
                image_file = Path(actual_path)
                
                if image_file.exists():
                    with open(image_file, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    image_content = {
                        'mime_type': 'image/webp',
                        'data': image_data
                    }
                    logger.info(f"Loaded image: {actual_path}")
                else:
                    logger.warning(f"Image file not found: {actual_path}")
            except Exception as e:
                logger.error(f"Error loading image: {e}")
        
        # Simple context building
        context_parts = []
        link_candidates = []
        # Increase the number of results used for context to provide more info to the LLM
        for result in search_results[:20]:
            r_type = result.get("type")
            # Fallback: some legacy cached entries may not have a nested `data` dict
            payload = result.get("data", result)
            # if no explicit type but url/title present treat generically
            if not r_type and payload.get("url"):
                link_candidates.append({"url": payload.get("url"), "text": payload.get("title", payload.get("url"))})
                continue
            if r_type in ("discourse_topic", "discourse"):
                title = payload.get("title", "(untitled)")
                content = payload.get("full_content") or payload.get("content") or ""
                url = payload.get("url")
                context_parts.append(f"Topic: {title}")
                context_parts.append(f"Content: {content[:800]}...")
                link_candidates.append({"url": url, "text": title})
            elif r_type == "qa_pair":
                context_parts.append(f"Q: {payload.get('question','')}")
                context_parts.append(f"A: {payload.get('answer','')}")
                link_candidates.append({"url": result.get("topic_url"), "text": result.get("title")})
            elif r_type in ("course_content", "course"):
                title = payload.get("title", "Course material")
                content = payload.get("content", "")
                url = payload.get("url")
                if not url:
                    # Build canonical course content URL using slug derived from id or file metadata
                    file_name = payload.get("id") or payload.get("metadata", {}).get("file")
                    if file_name:
                        slug = re.sub(r"\.md$", "", file_name)
                        url = f"https://tds.s-anand.net/#/{slug}"
                context_parts.append(f"Course: {title}")
                context_parts.append(f"Content: {content[:800]}...")
                link_candidates.append({"url": url, "text": f"Course: {title}"})
            context_parts.append("")

        context_text = "\n".join(context_parts)
        
        # Simple prompt
        prompt_text = f"""You are a helpful teaching assistant. Answer the question based on the provided context.
The context is extensive, so read it carefully to find the most relevant information.

Question: {question}

Context:
{context_text}

Answer:"""
        
        # Generate answer with minimal configuration
        try:
            # Use single model for all queries to avoid compatibility issues
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Minimal generation config with safety settings
            generation_config = {
                'temperature': 0.3,
                'max_output_tokens': 1000
            }
            
            # Prepare content for the model
            if image_content:
                # Ultra-simple prompt for image + text to avoid safety filters
                simple_prompt = f"Question: {question}\n\nAnswer based on context:"
                content = [
                    simple_prompt,
                    {
                        'mime_type': image_content['mime_type'],
                        'data': image_content['data']
                    }
                ]
                response = model.generate_content(content, generation_config=generation_config)
            else:
                response = model.generate_content(prompt_text, generation_config=generation_config)
            
            answer = response.text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = "I apologize, but I'm having trouble generating an answer right now. Please try again or check the course materials directly."
        
        # Extract links
        links = []
        for link in link_candidates[:25]:
            if link and link.get("url") and link not in links:
                links.append(link)
        
        return {
            "answer": answer,
            "links": links,
            "search_results_count": len(search_results),
        }

# Initialize the Virtual TA lazily for Vercel
print("🚀 Starting TDS Virtual TA...")
virtual_ta = None

def get_virtual_ta():
    global virtual_ta
    if virtual_ta is None:
        virtual_ta = TDSVirtualTA()
    return virtual_ta

# Create Flask app
app = Flask(__name__)
CORS(app)

@app.route('/api/', methods=['POST'])
def answer_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        image = data.get('image', None)
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Retrieve more search results to ensure important links (e.g., GA5 / GA4 threads) are captured
        search_results = get_virtual_ta().search(question, top_k=50)
        
        # Generate answer (with optional image support)
        result = get_virtual_ta().generate_answer(question, search_results, image)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = get_virtual_ta().search(query, top_k)
        
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
        'endpoints': {
            'health': '/api/health',
            'chat': '/api/ (POST)',
            'search': '/api/search (POST)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'Gemini 2.5 Flash Preview',
        'embeddings': 'Text Embedding 004',
        'optimization': 'Advanced Semantic Search'
    })

# For Vercel deployment, the app needs to be available at module level
# For local development
if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    print("✅ Ready to serve requests!")
    app.run(host="0.0.0.0", port=5000, debug=False) 