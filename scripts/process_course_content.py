#!/usr/bin/env python3
"""
Process Tools in Data Science course content into a structured knowledge base.
This script processes markdown files from the course repository to create
searchable content for the Virtual TA.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CourseContentProcessor:
    def __init__(self, course_dir: str = "tools-in-data-science-public"):
        self.course_dir = Path(course_dir)
        self.output_dir = Path("scripts/processed_course")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data structures
        self.topics = []
        self.code_examples = []
        self.tutorials = []
        self.tools_info = []
        self.project_info = []
        
    def clean_markdown_content(self, content: str) -> str:
        """Clean markdown content by removing excessive whitespace and formatting."""
        # Remove multiple consecutive newlines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        # Remove leading/trailing whitespace
        content = content.strip()
        # Normalize whitespace
        content = re.sub(r'[ \t]+', ' ', content)
        return content
    
    def extract_code_blocks(self, content: str, filename: str) -> List[Dict[str, Any]]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        
        # Pattern to match code blocks with language specification
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for i, (language, code) in enumerate(matches):
            if code.strip():  # Only include non-empty code blocks
                code_blocks.append({
                    'id': f"{filename}_code_{i}",
                    'language': language or 'text',
                    'code': code.strip(),
                    'source_file': filename,
                    'context': self.extract_context_around_code(content, code)
                })
        
        return code_blocks
    
    def extract_context_around_code(self, content: str, code: str) -> str:
        """Extract context around a code block."""
        # Find the position of the code block
        code_pos = content.find(code)
        if code_pos == -1:
            return ""
        
        # Extract surrounding text (up to 200 chars before and after)
        start = max(0, code_pos - 200)
        end = min(len(content), code_pos + len(code) + 200)
        context = content[start:end]
        
        # Clean up the context
        context = re.sub(r'```\w*\n.*?\n```', '[CODE_BLOCK]', context, flags=re.DOTALL)
        return context.strip()
    
    def extract_headings_and_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract headings and their content sections."""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if line is a heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Save previous section if exists
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    if current_section['content']:  # Only add non-empty sections
                        sections.append(current_section)
                
                # Start new section
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                current_section = {
                    'level': level,
                    'title': title,
                    'content': ''
                }
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Add the last section
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            if current_section['content']:
                sections.append(current_section)
        
        return sections
    
    def categorize_content(self, filename: str, content: str) -> str:
        """Categorize content based on filename and content."""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Project-related content
        if 'project' in filename_lower:
            return 'project'
        
        # LLM and AI related
        if any(term in filename_lower for term in ['llm', 'prompt', 'ai', 'gpt', 'claude']):
            return 'llm_ai'
        
        # Data processing and analysis
        if any(term in filename_lower for term in ['data', 'analysis', 'processing', 'pandas', 'numpy']):
            return 'data_processing'
        
        # Web scraping and APIs
        if any(term in filename_lower for term in ['scraping', 'api', 'web', 'requests']):
            return 'web_scraping'
        
        # Visualization
        if any(term in filename_lower for term in ['visual', 'chart', 'plot', 'graph']):
            return 'visualization'
        
        # Databases
        if any(term in filename_lower for term in ['sql', 'database', 'db', 'sqlite']):
            return 'database'
        
        # Development tools
        if any(term in filename_lower for term in ['git', 'github', 'docker', 'deployment']):
            return 'development_tools'
        
        # Live sessions
        if 'live-session' in filename_lower:
            return 'live_session'
        
        return 'general'
    
    def extract_tools_and_libraries(self, content: str) -> List[str]:
        """Extract mentioned tools and libraries from content."""
        tools = set()
        
        # Common patterns for tool mentions
        patterns = [
            r'`([a-zA-Z][a-zA-Z0-9_-]+)`',  # Code-formatted tools
            r'\*\*([a-zA-Z][a-zA-Z0-9_-]+)\*\*',  # Bold tools
            r'## ([a-zA-Z][a-zA-Z0-9_\s-]+)',  # Section headers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) > 2 and len(match) < 30:  # Reasonable tool name length
                    tools.add(match.strip())
        
        # Filter out common words that aren't tools
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        tools = {tool for tool in tools if tool.lower() not in common_words}
        
        return list(tools)
    
    def process_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Process a single markdown file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return None
            
            filename = filepath.name
            relative_path = str(filepath.relative_to(self.course_dir))
            
            # Clean content
            cleaned_content = self.clean_markdown_content(content)
            
            # Extract various components
            sections = self.extract_headings_and_sections(cleaned_content)
            code_blocks = self.extract_code_blocks(cleaned_content, filename)
            tools = self.extract_tools_and_libraries(cleaned_content)
            category = self.categorize_content(filename, cleaned_content)
            
            # Create topic entry
            topic = {
                'id': f"course_{filename.replace('.md', '').replace('-', '_')}",
                'title': sections[0]['title'] if sections else filename.replace('.md', '').replace('-', ' ').title(),
                'filename': filename,
                'relative_path': relative_path,
                'category': category,
                'content': cleaned_content,
                'sections': sections,
                'tools_mentioned': tools,
                'code_blocks_count': len(code_blocks),
                'word_count': len(cleaned_content.split()),
                'processed_at': datetime.now().isoformat()
            }
            
            # Store code blocks separately
            self.code_examples.extend(code_blocks)
            
            # Categorize special content types
            if category == 'project':
                self.project_info.append(topic)
            elif 'tutorial' in filename.lower() or 'guide' in filename.lower():
                self.tutorials.append(topic)
            elif len(tools) > 3:  # Files with many tool mentions
                self.tools_info.append(topic)
            
            return topic
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return None
    
    def process_all_files(self):
        """Process all markdown files in the course directory."""
        logger.info(f"Processing course content from {self.course_dir}")
        
        # Find all markdown files
        md_files = list(self.course_dir.glob("**/*.md"))
        logger.info(f"Found {len(md_files)} markdown files")
        
        processed_count = 0
        for filepath in md_files:
            topic = self.process_file(filepath)
            if topic:
                self.topics.append(topic)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count} files...")
        
        logger.info(f"Successfully processed {processed_count} files")
    
    def create_search_indices(self):
        """Create search indices for efficient querying."""
        logger.info("Creating search indices...")
        
        # Create keyword index
        keyword_index = {}
        for topic in self.topics:
            words = re.findall(r'\b\w+\b', topic['content'].lower())
            for word in words:
                if len(word) > 3:  # Only index meaningful words
                    if word not in keyword_index:
                        keyword_index[word] = []
                    keyword_index[word].append(topic['id'])
        
        # Create category index
        category_index = {}
        for topic in self.topics:
            category = topic['category']
            if category not in category_index:
                category_index[category] = []
            category_index[category].append(topic['id'])
        
        # Create tools index
        tools_index = {}
        for topic in self.topics:
            for tool in topic['tools_mentioned']:
                tool_lower = tool.lower()
                if tool_lower not in tools_index:
                    tools_index[tool_lower] = []
                tools_index[tool_lower].append(topic['id'])
        
        return {
            'keywords': keyword_index,
            'categories': category_index,
            'tools': tools_index
        }
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate statistics about the processed content."""
        stats = {
            'total_topics': len(self.topics),
            'total_code_examples': len(self.code_examples),
            'total_tutorials': len(self.tutorials),
            'total_tools_info': len(self.tools_info),
            'total_project_info': len(self.project_info),
            'categories': {},
            'languages': {},
            'tools_mentioned': {},
            'processing_date': datetime.now().isoformat()
        }
        
        # Category distribution
        for topic in self.topics:
            category = topic['category']
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        # Programming language distribution
        for code_block in self.code_examples:
            lang = code_block['language']
            stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
        
        # Most mentioned tools
        for topic in self.topics:
            for tool in topic['tools_mentioned']:
                stats['tools_mentioned'][tool] = stats['tools_mentioned'].get(tool, 0) + 1
        
        # Sort tools by frequency
        stats['tools_mentioned'] = dict(sorted(
            stats['tools_mentioned'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:50])  # Top 50 tools
        
        return stats
    
    def save_processed_data(self):
        """Save all processed data to files."""
        logger.info("Saving processed data...")
        
        # Save main topics
        with open(self.output_dir / "course_topics.json", 'w', encoding='utf-8') as f:
            json.dump(self.topics, f, indent=2, ensure_ascii=False)
        
        # Save code examples
        with open(self.output_dir / "course_code_examples.json", 'w', encoding='utf-8') as f:
            json.dump(self.code_examples, f, indent=2, ensure_ascii=False)
        
        # Save categorized content
        with open(self.output_dir / "course_tutorials.json", 'w', encoding='utf-8') as f:
            json.dump(self.tutorials, f, indent=2, ensure_ascii=False)
        
        with open(self.output_dir / "course_tools_info.json", 'w', encoding='utf-8') as f:
            json.dump(self.tools_info, f, indent=2, ensure_ascii=False)
        
        with open(self.output_dir / "course_project_info.json", 'w', encoding='utf-8') as f:
            json.dump(self.project_info, f, indent=2, ensure_ascii=False)
        
        # Create and save search indices
        indices = self.create_search_indices()
        with open(self.output_dir / "course_search_indices.json", 'w', encoding='utf-8') as f:
            json.dump(indices, f, indent=2, ensure_ascii=False)
        
        # Generate and save statistics
        stats = self.generate_statistics()
        with open(self.output_dir / "course_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed data saved to {self.output_dir}")
        
        # Print summary
        print(f"\n=== Course Content Processing Summary ===")
        print(f"Total topics processed: {len(self.topics)}")
        print(f"Code examples extracted: {len(self.code_examples)}")
        print(f"Tutorials identified: {len(self.tutorials)}")
        print(f"Tool-focused content: {len(self.tools_info)}")
        print(f"Project-related content: {len(self.project_info)}")
        print(f"\nCategory distribution:")
        for category, count in stats['categories'].items():
            print(f"  {category}: {count}")
        print(f"\nTop programming languages:")
        for lang, count in sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {lang}: {count}")
        print(f"\nOutput directory: {self.output_dir}")

def main():
    """Main function to process course content."""
    processor = CourseContentProcessor()
    
    # Check if course directory exists
    if not processor.course_dir.exists():
        logger.error(f"Course directory {processor.course_dir} not found!")
        print("Please ensure the tools-in-data-science-public directory exists in the current directory.")
        return
    
    try:
        # Process all files
        processor.process_all_files()
        
        # Save processed data
        processor.save_processed_data()
        
        print("\n✅ Course content processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"❌ Processing failed: {e}")

if __name__ == "__main__":
    main()