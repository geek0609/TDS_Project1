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

try:
    from PIL import Image
    import base64
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not available for image processing")

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
    
    def clean_text_for_search(self, text: str) -> str:
        """Clean text for better search matching"""
        import re
        # Convert to lowercase and remove punctuation, keeping spaces
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        # Replace multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def simple_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Improved text-based search with better text processing"""
        results = []
        query_cleaned = self.clean_text_for_search(query)
        query_terms = set(query_cleaned.split())
        
        # Search through discourse topics
        for topic in self.discourse_data.get("topics", []):
            try:
                title = topic.get("title", "")
                content = topic.get("full_content", "")
                combined_text = f"{title} {content}"
                
                # Clean the text for search
                cleaned_text = self.clean_text_for_search(combined_text)
                text_words = set(cleaned_text.split())
                
                # Calculate word overlap score
                overlap = len(query_terms.intersection(text_words))
                if overlap > 0:
                    score = overlap / len(query_terms)
                    # Boost score if query appears as substring (for cases like "DevTools")
                    if query.lower() in combined_text.lower():
                        score += 0.5
                    
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
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                combined_text = f"{question} {answer}"
                
                # Clean the text for search
                cleaned_text = self.clean_text_for_search(combined_text)
                text_words = set(cleaned_text.split())
                
                overlap = len(query_terms.intersection(text_words))
                if overlap > 0:
                    score = overlap / len(query_terms)
                    # Boost score if query appears as substring
                    if query.lower() in combined_text.lower():
                        score += 0.5
                    
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
                title = course.get("title", "")
                content = course.get("content", "")
                combined_text = f"{title} {content}"
                
                # Clean the text for search
                cleaned_text = self.clean_text_for_search(combined_text)
                text_words = set(cleaned_text.split())
                
                overlap = len(query_terms.intersection(text_words))
                if overlap > 0:
                    score = overlap / len(query_terms)
                    # Boost score if query appears as substring
                    if query.lower() in combined_text.lower():
                        score += 0.5
                    
                    results.append({
                        "title": course.get("title", ""),
                        "url": course.get("url", ""),
                        "content": course.get("content", "")[:1000],  # Increased from 500
                        "score": score,
                        "type": "course_content",
                        "data": course
                    })
            except Exception as e:
                logger.warning(f"Error processing course: {e}")
        
        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def extract_relevant_content(self, content: str, query: str, max_length: int = 1500) -> str:
        """Extract relevant sections from long content based on query terms"""
        if len(content) <= max_length:
            return content
        
        # Extract key subject terms (ignore common words)
        stop_words = {'can', 'you', 'please', 'explain', 'what', 'is', 'the', 'use', 'of', 'how', 'do', 'does', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        query_terms = [term.lower() for term in query.split() if term.lower() not in stop_words and len(term) > 2]
        
        lines = content.split('\n')
        
        # Find lines that contain key query terms (not common words)
        relevant_lines = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(term in line_lower for term in query_terms):
                # Include context around the relevant line
                start = max(0, i - 3)
                end = min(len(lines), i + 8)  # More lines after for Q&A format
                relevant_lines.extend(lines[start:end])
        
        # If no specific terms found, try substring search for the most important term
        if not relevant_lines and query_terms:
            main_term = max(query_terms, key=len)  # Use the longest term as most important
            for i, line in enumerate(lines):
                if main_term in line.lower():
                    start = max(0, i - 3)
                    end = min(len(lines), i + 8)
                    relevant_lines.extend(lines[start:end])
                
        # If we found relevant sections, use them
        if relevant_lines:
            # Remove duplicates while preserving order
            seen = set()
            unique_lines = []
            for line in relevant_lines:
                if line not in seen:
                    unique_lines.append(line)
                    seen.add(line)
            
            relevant_content = '\n'.join(unique_lines)
            if len(relevant_content) <= max_length:
                return relevant_content
            else:
                return relevant_content[:max_length] + "..."
        
        # Fallback to beginning of content
        return content[:max_length] + "..."

    def process_image(self, image_data: str) -> Image.Image:
        """Process image from base64 or file path"""
        if not HAS_PIL:
            raise ValueError("PIL not available for image processing")
        
        try:
            # Check if it's a file path
            if image_data.startswith('file://'):
                file_path = image_data[7:]  # Remove 'file://' prefix
                return Image.open(file_path)
            
            # Check if it's a base64 encoded image
            elif image_data.startswith('data:image/'):
                # Extract base64 data after the comma
                base64_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_bytes))
            
            # Try to decode as plain base64
            else:
                try:
                    image_bytes = base64.b64decode(image_data)
                    return Image.open(io.BytesIO(image_bytes))
                except:
                    # Assume it's a file path
                    return Image.open(image_data)
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ValueError(f"Could not process image: {e}")

    def generate_answer(self, question: str, search_results: List[Dict], image: str = None) -> Dict:
        """Generate answer based on search results and optional image with improved context extraction"""
        
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
                    
                    # Extract relevant content based on the question
                    relevant_content = self.extract_relevant_content(content, question, 1200)
                    context_parts.append(f"Content: {relevant_content}")
                    
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
                    
                    # Extract relevant content based on the question
                    relevant_content = self.extract_relevant_content(content, question, 1200)
                    context_parts.append(f"Content: {relevant_content}")
                    
                    if url:
                        links.append({"url": url, "text": f"Course: {title}"})
                context_parts.append("")
            except Exception as e:
                logger.warning(f"Error processing result: {e}")
        
        context_text = "\n".join(context_parts)
        
        # Generate answer
        if self.has_api and HAS_GENAI:
            try:
                # Prepare content for the model
                content_parts = []
                
                # Add the text prompt
                if image:
                    prompt_text = f"""You are a helpful teaching assistant. Answer the question based on the provided context and the image.

Question: {question}

Context:
{context_text}

Please analyze the image along with the context to provide a comprehensive answer."""
                else:
                    prompt_text = f"""You are a helpful teaching assistant. Answer the question based on the provided context.

Question: {question}

Context:
{context_text}

Answer:"""
                
                content_parts.append(prompt_text)
                
                # Add image if provided
                if image:
                    try:
                        processed_image = self.process_image(image)
                        content_parts.append(processed_image)
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")
                        content_parts[0] += f"\n\nNote: Could not process the provided image due to error: {e}"
                
                model = genai.GenerativeModel('gemini-2.0-flash')
                generation_config = {
                    'temperature': 0.3,
                    'max_output_tokens': 1000
                }
                
                response = model.generate_content(content_parts, generation_config=generation_config)
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
        image = data.get('image')  # Optional image data
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        search_results = get_virtual_ta().simple_search(question, top_k=20)
        result = get_virtual_ta().generate_answer(question, search_results, image)
        
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