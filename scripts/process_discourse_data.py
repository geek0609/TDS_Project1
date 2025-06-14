import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode common HTML entities
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&amp;', '&').replace('&quot;', '"')
    text = text.replace('&#39;', "'").replace('&nbsp;', ' ')
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from text"""
    code_blocks = []
    
    # Find code blocks with triple backticks
    code_pattern = r'```[\s\S]*?```'
    matches = re.findall(code_pattern, text)
    code_blocks.extend(matches)
    
    # Find inline code with single backticks
    inline_code_pattern = r'`[^`]+`'
    matches = re.findall(inline_code_pattern, text)
    code_blocks.extend(matches)
    
    return code_blocks


def categorize_topic(title: str, tags: List[str]) -> str:
    """Categorize topics based on title and tags"""
    title_lower = title.lower()
    
    if any(tag in ["graded-assignment", "graded-question"] for tag in tags):
        return "graded_assignment"
    elif any(tag in ["tds-project-1", "tds-project-2", "tds-project-one"] for tag in tags):
        return "project"
    elif "ga" in title_lower and any(char.isdigit() for char in title_lower):
        return "graded_assignment"
    elif any(word in title_lower for word in ["project", "assignment"]):
        return "project"
    elif any(word in title_lower for word in ["error", "issue", "problem", "bug"]):
        return "technical_issue"
    elif any(word in title_lower for word in ["clarification", "doubt", "question"]):
        return "clarification"
    elif any(word in title_lower for word in ["exam", "mock", "test"]):
        return "exam"
    elif any(word in title_lower for word in ["score", "marks", "grade"]):
        return "grading"
    else:
        return "general"


def extract_qa_pairs(posts: List[Dict]) -> List[Dict]:
    """Extract question-answer pairs from posts"""
    qa_pairs = []
    
    if not posts:
        return qa_pairs
    
    # First post is usually the question
    question_post = posts[0]
    question_text = clean_html(question_post.get("cooked", ""))
    
    if not question_text:
        return qa_pairs
    
    # Look for answers from staff/TAs
    for post in posts[1:]:
        if post.get("staff") or post.get("moderator") or post.get("admin"):
            answer_text = clean_html(post.get("cooked", ""))
            if answer_text:
                qa_pairs.append({
                    "question": question_text,
                    "answer": answer_text,
                    "question_author": question_post.get("display_username", ""),
                    "answer_author": post.get("display_username", ""),
                    "is_staff_answer": True,
                    "post_number": post.get("post_number", 0)
                })
    
    # If no staff answers, look for accepted answers
    for post in posts[1:]:
        if post.get("accepted_answer"):
            answer_text = clean_html(post.get("cooked", ""))
            if answer_text:
                qa_pairs.append({
                    "question": question_text,
                    "answer": answer_text,
                    "question_author": question_post.get("display_username", ""),
                    "answer_author": post.get("display_username", ""),
                    "is_staff_answer": False,
                    "is_accepted": True,
                    "post_number": post.get("post_number", 0)
                })
    
    return qa_pairs


def process_topic_file(file_path: Path) -> Dict[str, Any]:
    """Process a single topic JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    topic_id = data.get("id")
    title = data.get("title", "")
    created_at = data.get("created_at", "")
    tags = data.get("tags", [])
    posts = data.get("post_stream", {}).get("posts", [])
    
    category = categorize_topic(title, tags)
    qa_pairs = extract_qa_pairs(posts)
    
    # Extract all post content for full-text search
    all_posts_content = []
    for post in posts:
        content = clean_html(post.get("cooked", ""))
        if content:
            all_posts_content.append({
                "content": content,
                "author": post.get("display_username", ""),
                "post_number": post.get("post_number", 0),
                "created_at": post.get("created_at", ""),
                "is_staff": post.get("staff", False) or post.get("moderator", False) or post.get("admin", False)
            })
    
    # Extract code examples
    code_examples = []
    for post in posts:
        content = post.get("cooked", "")
        if content:
            codes = extract_code_blocks(content)
            code_examples.extend(codes)
    
    return {
        "topic_id": topic_id,
        "title": title,
        "created_at": created_at,
        "category": category,
        "tags": tags,
        "posts_count": len(posts),
        "views": data.get("views", 0),
        "qa_pairs": qa_pairs,
        "all_posts": all_posts_content,
        "code_examples": code_examples,
        "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}"
    }


def create_knowledge_base():
    """Process all topic files and create knowledge base"""
    knowledge_base = {
        "topics": [],
        "qa_pairs": [],
        "categories": {},
        "code_examples": [],
        "metadata": {
            "total_topics": 0,
            "total_qa_pairs": 0,
            "processed_at": datetime.now().isoformat(),
            "date_range": "2025-01-01 to 2025-04-14"
        }
    }
    
    topic_files = list(RAW_DIR.glob("topic-*.json"))
    print(f"Processing {len(topic_files)} topic files...")
    
    for file_path in topic_files:
        print(f"Processing {file_path.name}...")
        topic_data = process_topic_file(file_path)
        
        if topic_data:
            knowledge_base["topics"].append(topic_data)
            knowledge_base["qa_pairs"].extend(topic_data["qa_pairs"])
            knowledge_base["code_examples"].extend(topic_data["code_examples"])
            
            # Count by category
            category = topic_data["category"]
            if category not in knowledge_base["categories"]:
                knowledge_base["categories"][category] = 0
            knowledge_base["categories"][category] += 1
    
    # Update metadata
    knowledge_base["metadata"]["total_topics"] = len(knowledge_base["topics"])
    knowledge_base["metadata"]["total_qa_pairs"] = len(knowledge_base["qa_pairs"])
    
    return knowledge_base


def create_search_index(knowledge_base: Dict) -> Dict:
    """Create search index for quick lookups"""
    search_index = {
        "by_category": {},
        "by_keywords": {},
        "by_tags": {}
    }
    
    # Index by category
    for topic in knowledge_base["topics"]:
        category = topic["category"]
        if category not in search_index["by_category"]:
            search_index["by_category"][category] = []
        search_index["by_category"][category].append(topic["topic_id"])
    
    # Index by keywords (from titles and content)
    for topic in knowledge_base["topics"]:
        title_words = re.findall(r'\w+', topic["title"].lower())
        for word in title_words:
            if len(word) > 3:  # Skip short words
                if word not in search_index["by_keywords"]:
                    search_index["by_keywords"][word] = []
                if topic["topic_id"] not in search_index["by_keywords"][word]:
                    search_index["by_keywords"][word].append(topic["topic_id"])
    
    # Index by tags
    for topic in knowledge_base["topics"]:
        for tag in topic["tags"]:
            if tag not in search_index["by_tags"]:
                search_index["by_tags"][tag] = []
            search_index["by_tags"][tag].append(topic["topic_id"])
    
    return search_index


def main():
    print("Creating knowledge base from raw Discourse data...")
    
    knowledge_base = create_knowledge_base()
    
    print(f"Created knowledge base with:")
    print(f"  - {knowledge_base['metadata']['total_topics']} topics")
    print(f"  - {knowledge_base['metadata']['total_qa_pairs']} Q&A pairs")
    print(f"  - {len(knowledge_base['code_examples'])} code examples")
    print(f"  - Categories: {knowledge_base['categories']}")
    
    # Save knowledge base
    kb_file = OUTPUT_DIR / "knowledge_base.json"
    with open(kb_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    print(f"Saved knowledge base to {kb_file}")
    
    # Create and save search index
    search_index = create_search_index(knowledge_base)
    index_file = OUTPUT_DIR / "search_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(search_index, f, indent=2, ensure_ascii=False)
    print(f"Saved search index to {index_file}")
    
    # Create Q&A only file for easier access
    qa_only = {
        "qa_pairs": knowledge_base["qa_pairs"],
        "metadata": knowledge_base["metadata"]
    }
    qa_file = OUTPUT_DIR / "qa_pairs.json"
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(qa_only, f, indent=2, ensure_ascii=False)
    print(f"Saved Q&A pairs to {qa_file}")


if __name__ == "__main__":
    main() 