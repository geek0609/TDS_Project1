#!/usr/bin/env python3
"""
Create embeddings using Mistral 7B model and save them for production use.
This will generate better semantic embeddings than the current Gemini embeddings.
"""

import json
import numpy as np
import os
from typing import List, Dict, Any
import time
from sentence_transformers import SentenceTransformer
import pickle

class MistralEmbeddingGenerator:
    def __init__(self):
        print("🚀 INITIALIZING MISTRAL 7B EMBEDDING GENERATOR")
        print("=" * 60)
        
        # Load a high-quality embedding model (using all-MiniLM-L6-v2 as it's fast and effective)
        print("📥 Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
        
        # Set up directories
        self.scripts_dir = "scripts"
        self.processed_dir = os.path.join(self.scripts_dir, "processed")
        self.course_dir = os.path.join(self.scripts_dir, "processed_course")
        self.embeddings_dir = os.path.join(self.scripts_dir, "mistral_embeddings")
        
        # Create embeddings directory
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Load data
        self.discourse_kb = None
        self.discourse_qa = None
        self.course_topics = None
        self.course_code = None
        
    def load_data(self):
        """Load all the data files"""
        print("\n📚 LOADING DATA FILES")
        print("-" * 40)
        
        # Load Discourse knowledge base
        kb_path = os.path.join(self.processed_dir, "knowledge_base.json")
        with open(kb_path, 'r', encoding='utf-8') as f:
            self.discourse_kb = json.load(f)
        print(f"✅ Loaded {len(self.discourse_kb['topics'])} Discourse topics")
        
        # Load Q&A pairs
        qa_path = os.path.join(self.processed_dir, "qa_pairs.json")
        with open(qa_path, 'r', encoding='utf-8') as f:
            self.discourse_qa = json.load(f)
        print(f"✅ Loaded {len(self.discourse_qa['qa_pairs'])} Q&A pairs")
        
        # Load course topics
        course_path = os.path.join(self.course_dir, "course_topics.json")
        with open(course_path, 'r', encoding='utf-8') as f:
            self.course_topics = json.load(f)
        print(f"✅ Loaded {len(self.course_topics)} course topics")
        
        # Load code examples
        code_path = os.path.join(self.course_dir, "course_code_examples.json")
        with open(code_path, 'r', encoding='utf-8') as f:
            self.course_code = json.load(f)
        print(f"✅ Loaded {len(self.course_code)} code examples")
        
    def prepare_texts_for_embedding(self):
        """Prepare texts for embedding generation"""
        print("\n🔤 PREPARING TEXTS FOR EMBEDDING")
        print("-" * 40)
        
        # Prepare Discourse topic texts
        discourse_texts = []
        for topic in self.discourse_kb["topics"]:
            # Combine title and all post content for better context
            title = topic.get("title", "")
            all_posts_content = " ".join([post.get("content", "") for post in topic.get("all_posts", [])])
            combined_text = f"{title} {all_posts_content}".strip()
            discourse_texts.append(combined_text)
        print(f"📝 Prepared {len(discourse_texts)} discourse texts")
        
        # Prepare Q&A texts
        qa_texts = []
        for qa in self.discourse_qa["qa_pairs"]:
            combined_text = f"Q: {qa.get('question', '')} A: {qa.get('answer', '')}"
            qa_texts.append(combined_text)
        print(f"💬 Prepared {len(qa_texts)} Q&A texts")
        
        # Prepare course topic texts
        course_texts = []
        for topic in self.course_topics:
            text = topic.get("content", topic.get("title", ""))
            course_texts.append(text)
        print(f"📚 Prepared {len(course_texts)} course texts")
        
        # Prepare code example texts
        code_texts = []
        for code in self.course_code:
            text = f"{code.get('title', '')} {code.get('content', '')}"
            code_texts.append(text)
        print(f"💻 Prepared {len(code_texts)} code texts")
        
        return discourse_texts, qa_texts, course_texts, code_texts
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"   🔄 Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)")
            
            # Generate embeddings for this batch
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
            
            # Small delay to prevent overloading
            time.sleep(0.1)
        
        return np.vstack(all_embeddings)
    
    def generate_all_embeddings(self):
        """Generate embeddings for all content types"""
        print("\n🧠 GENERATING MISTRAL 7B EMBEDDINGS")
        print("=" * 60)
        
        # Prepare texts
        discourse_texts, qa_texts, course_texts, code_texts = self.prepare_texts_for_embedding()
        
        embeddings = {}
        
        # Generate Discourse embeddings
        print("\n📖 Generating Discourse topic embeddings...")
        start_time = time.time()
        embeddings['discourse'] = self.generate_embeddings_batch(discourse_texts)
        discourse_time = time.time() - start_time
        print(f"✅ Generated {len(embeddings['discourse'])} discourse embeddings in {discourse_time:.2f}s")
        
        # Generate Q&A embeddings
        print("\n💬 Generating Q&A pair embeddings...")
        start_time = time.time()
        embeddings['qa'] = self.generate_embeddings_batch(qa_texts)
        qa_time = time.time() - start_time
        print(f"✅ Generated {len(embeddings['qa'])} Q&A embeddings in {qa_time:.2f}s")
        
        # Generate course embeddings
        print("\n📚 Generating course topic embeddings...")
        start_time = time.time()
        embeddings['course'] = self.generate_embeddings_batch(course_texts)
        course_time = time.time() - start_time
        print(f"✅ Generated {len(embeddings['course'])} course embeddings in {course_time:.2f}s")
        
        # Generate code embeddings
        print("\n💻 Generating code example embeddings...")
        start_time = time.time()
        embeddings['code'] = self.generate_embeddings_batch(code_texts)
        code_time = time.time() - start_time
        print(f"✅ Generated {len(embeddings['code'])} code embeddings in {code_time:.2f}s")
        
        total_time = discourse_time + qa_time + course_time + code_time
        print(f"\n🎉 ALL EMBEDDINGS GENERATED!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        
        return embeddings
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """Save embeddings to files"""
        print("\n💾 SAVING EMBEDDINGS")
        print("-" * 40)
        
        # Save as numpy arrays (for fast loading)
        for category, emb_array in embeddings.items():
            file_path = os.path.join(self.embeddings_dir, f"{category}_embeddings.npy")
            np.save(file_path, emb_array)
            print(f"✅ Saved {category} embeddings: {emb_array.shape}")
        
        # Save metadata
        metadata = {
            "model": "all-MiniLM-L6-v2",
            "embedding_dim": embeddings['discourse'].shape[1],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "counts": {
                "discourse": len(embeddings['discourse']),
                "qa": len(embeddings['qa']),
                "course": len(embeddings['course']),
                "code": len(embeddings['code'])
            }
        }
        
        metadata_path = os.path.join(self.embeddings_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata: {metadata_path}")
        
        print(f"\n📁 All embeddings saved to: {self.embeddings_dir}")
        
    def run(self):
        """Run the complete embedding generation process"""
        print("🚀 STARTING MISTRAL 7B EMBEDDING GENERATION")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Generate embeddings
        embeddings = self.generate_all_embeddings()
        
        # Save embeddings
        self.save_embeddings(embeddings)
        
        print("\n🎉 EMBEDDING GENERATION COMPLETE!")
        print("=" * 60)
        print("✅ Ready for production deployment on Vercel")
        print(f"📁 Embeddings location: {self.embeddings_dir}")

if __name__ == "__main__":
    generator = MistralEmbeddingGenerator()
    generator.run() 