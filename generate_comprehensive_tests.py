#!/usr/bin/env python3
"""
Comprehensive Test Generator for TDS Virtual TA
Creates 1000 realistic test questions based on actual data and tests for hallucination
"""

import json
import random
import requests
import time
from pathlib import Path
from typing import List, Dict, Any
import re

class ComprehensiveTestGenerator:
    def __init__(self):
        self.scripts_dir = Path(__file__).resolve().parent / "scripts"
        self.processed_dir = self.scripts_dir / "processed"
        self.course_dir = self.scripts_dir / "processed_course"
        
        # Load data
        self.discourse_kb = self.load_json(self.processed_dir / "knowledge_base.json")
        self.discourse_qa = self.load_json(self.processed_dir / "qa_pairs.json")
        self.course_topics = self.load_json(self.course_dir / "course_topics.json")
        self.course_code = self.load_json(self.course_dir / "course_code_examples.json")
        
        # Common TDS terms and concepts
        self.tds_concepts = [
            "Docker", "Podman", "LLM", "prompt engineering", "vector databases",
            "ChromaDB", "LanceDB", "Selenium", "web scraping", "API", "FastAPI",
            "Gemini", "OpenAI", "GPT", "embeddings", "semantic search",
            "graded assignment", "GA1", "GA2", "GA3", "GA4", "GA5", "GA6", "GA7",
            "project", "ROE", "end-term exam", "deadline", "submission",
            "Python", "JavaScript", "HTML", "CSS", "JSON", "CSV", "pandas",
            "numpy", "matplotlib", "seaborn", "plotly", "streamlit", "gradio"
        ]
        
        # Fake/non-existent concepts for hallucination testing
        self.fake_concepts = [
            "GA15", "GA20", "Project 5", "TDS Advanced Course", "Super Docker",
            "MegaLLM", "UltraDB", "HyperScraper", "QuantumAPI", "NeoPrompt",
            "DataMaster 3000", "CodeGenius Pro", "WebCrawler Ultimate",
            "September 2025 exam", "December 2025 deadline", "Advanced ROE",
            "TDS Premium", "Elite Assignment", "Master Project", "Quantum GA"
        ]
    
    def load_json(self, file_path: Path) -> Dict:
        """Load JSON file safely"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        return {}
    
    def generate_discourse_questions(self, count: int = 200) -> List[Dict]:
        """Generate questions based on actual discourse topics"""
        questions = []
        topics = self.discourse_kb.get("topics", [])
        qa_pairs = self.discourse_qa.get("qa_pairs", [])
        
        # Question templates
        templates = [
            "How do I {action}?",
            "What is the deadline for {topic}?",
            "I'm getting an error with {concept}. Can you help?",
            "Can you explain {concept}?",
            "What's the difference between {concept1} and {concept2}?",
            "How do I submit {assignment}?",
            "Is {concept} allowed in this course?",
            "What are the requirements for {topic}?",
            "Can I use {tool} for {purpose}?",
            "How do I install {tool}?",
            "What's the best way to {action}?",
            "I missed the deadline for {assignment}. What should I do?",
            "Can you provide an example of {concept}?",
            "How do I debug {error_type} errors?",
            "What's the grading criteria for {assignment}?"
        ]
        
        actions = ["submit my project", "install Docker", "use the API", "scrape data", 
                  "create embeddings", "deploy my app", "test my code", "debug errors"]
        
        for i in range(count):
            if i < len(qa_pairs):
                # Use actual Q&A pairs
                qa = qa_pairs[i]
                questions.append({
                    "question": qa["question"],
                    "expected_answer_contains": qa["answer"][:100],
                    "category": "discourse_qa",
                    "should_find_answer": True,
                    "difficulty": "medium"
                })
            else:
                # Generate variations based on topics
                topic = random.choice(topics) if topics else {"title": "assignment"}
                template = random.choice(templates)
                concept = random.choice(self.tds_concepts)
                
                question = template.format(
                    action=random.choice(actions),
                    topic=topic.get("title", "assignment"),
                    concept=concept,
                    concept1=concept,
                    concept2=random.choice(self.tds_concepts),
                    assignment=f"GA{random.randint(1,7)}",
                    tool=random.choice(["Docker", "Python", "Selenium", "FastAPI"]),
                    purpose=random.choice(["web scraping", "API development", "data analysis"]),
                    error_type=random.choice(["Docker", "Python", "API", "installation"])
                )
                
                questions.append({
                    "question": question,
                    "category": "discourse_generated",
                    "should_find_answer": True,
                    "difficulty": "medium",
                    "topic_reference": topic.get("url", "")
                })
        
        return questions
    
    def generate_hallucination_tests(self, count: int = 200) -> List[Dict]:
        """Generate questions designed to test for hallucination"""
        questions = []
        
        # Questions about non-existent things
        fake_templates = [
            "What is the deadline for {fake_concept}?",
            "How do I submit {fake_concept}?",
            "Can you explain {fake_concept}?",
            "What are the requirements for {fake_concept}?",
            "When is the {fake_concept} exam?",
            "How do I install {fake_concept}?",
            "What's the difference between {fake_concept} and {real_concept}?",
            "Can I use {fake_concept} for this course?",
            "What's the grading criteria for {fake_concept}?",
            "How do I access {fake_concept}?"
        ]
        
        # Future dates that don't exist
        future_templates = [
            "When is the TDS exam in {future_date}?",
            "What's the deadline for assignments in {future_date}?",
            "Are there any courses in {future_date}?",
            "What's the syllabus for TDS {future_date}?",
            "When do {future_date} admissions open?"
        ]
        
        future_dates = ["September 2025", "December 2025", "January 2026", "March 2026"]
        
        for i in range(count):
            if i < count // 2:
                # Fake concept questions
                template = random.choice(fake_templates)
                question = template.format(
                    fake_concept=random.choice(self.fake_concepts),
                    real_concept=random.choice(self.tds_concepts)
                )
                
                questions.append({
                    "question": question,
                    "category": "hallucination_fake",
                    "should_find_answer": False,
                    "difficulty": "hard",
                    "expected_response": "don't know|not available|cannot find|no information"
                })
            else:
                # Future date questions
                template = random.choice(future_templates)
                question = template.format(future_date=random.choice(future_dates))
                
                questions.append({
                    "question": question,
                    "category": "hallucination_future",
                    "should_find_answer": False,
                    "difficulty": "hard",
                    "expected_response": "don't know|not available|cannot find|no information"
                })
        
        return questions
    
    def generate_all_tests(self) -> List[Dict]:
        """Generate all 1000 test questions"""
        print("🔄 Generating comprehensive test suite...")
        
        all_questions = []
        
        # Generate different categories
        all_questions.extend(self.generate_discourse_questions(500))
        print(f"✅ Generated {len(all_questions)} discourse questions")
        
        all_questions.extend(self.generate_hallucination_tests(500))
        print(f"✅ Generated {len(all_questions)} total questions (added hallucination tests)")
        
        # Shuffle the questions
        random.shuffle(all_questions)
        
        # Add IDs and metadata
        for i, q in enumerate(all_questions):
            q["id"] = i + 1
            q["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"🎯 Total questions generated: {len(all_questions)}")
        return all_questions
    
    def save_tests(self, questions: List[Dict], filename: str = "comprehensive_test_suite.json"):
        """Save test questions to file"""
        test_data = {
            "metadata": {
                "total_questions": len(questions),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "categories": {},
                "difficulty_levels": {}
            },
            "questions": questions
        }
        
        # Calculate statistics
        for q in questions:
            category = q.get("category", "unknown")
            difficulty = q.get("difficulty", "unknown")
            
            test_data["metadata"]["categories"][category] = test_data["metadata"]["categories"].get(category, 0) + 1
            test_data["metadata"]["difficulty_levels"][difficulty] = test_data["metadata"]["difficulty_levels"].get(difficulty, 0) + 1
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Test suite saved to: {filename}")
        print(f"📊 Categories: {test_data['metadata']['categories']}")
        print(f"📈 Difficulty levels: {test_data['metadata']['difficulty_levels']}")

def main():
    """Generate comprehensive test suite"""
    generator = ComprehensiveTestGenerator()
    questions = generator.generate_all_tests()
    generator.save_tests(questions)
    
    print("\n🚀 Comprehensive test suite generated successfully!")
    print("📝 Ready to run comprehensive API testing")

if __name__ == "__main__":
    main() 