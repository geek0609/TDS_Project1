# Enhanced TDS Virtual TA with Gemini 2.5 Flash

🚀 **An AI-powered Teaching Assistant for the Tools in Data Science course at IIT Madras, enhanced with Google's Gemini 2.5 Flash and semantic embeddings.**

## 🌟 Features

### 🧠 AI-Powered Answer Generation
- **Gemini 2.5 Flash Integration**: Uses Google's latest language model for intelligent, contextual responses
- **Semantic Search**: Advanced embedding-based search using `sentence-transformers`
- **Multi-source Knowledge**: Combines Discourse forum discussions with official course materials
- **Contextual Understanding**: Provides relevant answers based on semantic similarity, not just keyword matching

### 📚 Comprehensive Knowledge Base
- **114 Discourse Topics**: Historical student discussions and Q&A from Jan-Apr 2025
- **36 Q&A Pairs**: Curated question-answer pairs from forum discussions
- **137 Course Topics**: Official course materials covering all TDS concepts
- **184 Code Examples**: Practical code snippets with context and explanations

### 🔍 Advanced Search Capabilities
- **Semantic Search**: Find relevant content based on meaning, not just keywords
- **Multi-category Search**: Searches across Discourse topics, Q&A pairs, course content, and code examples
- **Similarity Scoring**: Results ranked by semantic relevance
- **Real-time Processing**: Fast response times with efficient embedding lookup

### 🎯 Smart Answer Prioritization
1. **Staff Answers**: Prioritizes official responses from teaching staff
2. **Course Materials**: Uses authoritative course content
3. **Community Discussions**: Includes relevant student discussions
4. **Code Examples**: Provides practical implementation examples

## 🏗️ Architecture

```
Enhanced Virtual TA
├── Gemini 2.5 Flash (Answer Generation)
├── Sentence Transformers (Embeddings)
├── Knowledge Base
│   ├── Discourse Data (114 topics, 36 Q&A pairs)
│   └── Course Content (137 topics, 184 code examples)
├── Semantic Search Engine
└── Flask API Server
```

## 📊 Knowledge Base Statistics

| Category | Count | Description |
|----------|-------|-------------|
| **Discourse Topics** | 114 | Forum discussions from Jan-Apr 2025 |
| **Q&A Pairs** | 36 | Curated question-answer pairs |
| **Course Topics** | 137 | Official course materials |
| **Code Examples** | 184 | Practical code snippets |
| **Total Keywords** | 6,054 | Indexed for semantic search |
| **Content Categories** | 9 | LLM/AI, Data Processing, Web Scraping, etc. |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd TDSProject/P1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the knowledge base** (if not already processed)
   ```bash
   python scripts/process_discourse.py
   python scripts/process_course_content.py
   ```

4. **Start the Enhanced Virtual TA**
   ```bash
   python app_gemini.py
   ```

The server will start on `http://localhost:5000` with the following endpoints:

## 📡 API Endpoints

### Main Question Answering
```http
POST /api/
Content-Type: application/json

{
  "question": "What is prompt engineering and how do I use it?",
  "image": "base64_encoded_image_optional"
}
```

**Response:**
```json
{
  "answer": "Prompt engineering is the art and science of crafting effective prompts for Large Language Models...",
  "links": [
    {
      "url": "https://tds.s-anand.net/#/prompt-engineering.md",
      "text": "Course: Prompt Engineering"
    }
  ],
  "search_results_count": {
    "discourse_topics": 2,
    "qa_pairs": 1,
    "course_topics": 3,
    "code_examples": 2
  }
}
```

### Semantic Search
```http
POST /api/search
Content-Type: application/json

{
  "query": "vector databases",
  "top_k": 5
}
```

### Health Check
```http
GET /api/health
```

### Statistics
```http
GET /api/stats
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_gemini_api.py
```

This will test:
- ✅ Health and statistics endpoints
- ✅ Semantic search functionality
- ✅ Question answering with various types of queries
- ✅ Discourse-based questions (project submission, deadlines, technical issues)
- ✅ Course content questions (prompt engineering, vector databases, web scraping)
- ✅ Complex technical comparisons

## 💡 Example Questions

### Discourse-Based Questions
- "How do I submit my TDS project?"
- "What is the deadline for GA1?"
- "I'm getting a Docker error when running my code"
- "My assignment score is showing 0"

### Course Content Questions
- "What is prompt engineering and how do I use it?"
- "Explain vector databases and their use cases"
- "How do I scrape data with Python? Show me examples"
- "What tools are available for data visualization?"
- "How do I use SQLite for data analysis?"

### Complex Questions
- "What's the difference between ChromaDB and LanceDB for vector storage?"
- "How do I deploy my TDS project using Docker and what are common issues?"
- "Show me how to use LLMs for text analysis with code examples"

## 🔧 Configuration

### Gemini API Setup
The system uses Google's Gemini 2.5 Flash model. The API key is configured in `app_gemini.py`:

```python
GEMINI_API_KEY = "AIzaSyDb-kro99IFG9PssLKALzzt70q6p6lL3cQ"
```

### Embedding Model
Uses `all-MiniLM-L6-v2` from Sentence Transformers for semantic embeddings:

```python
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
```

## 📁 Project Structure

```
TDSProject/P1/
├── app_gemini.py              # Enhanced Gemini-powered Virtual TA
├── test_gemini_api.py         # Comprehensive test suite
├── requirements.txt           # Python dependencies
├── scripts/
│   ├── process_discourse.py   # Discourse data processing
│   ├── process_course_content.py  # Course content processing
│   ├── processed/             # Processed Discourse data
│   │   ├── knowledge_base.json
│   │   └── qa_pairs.json
│   └── processed_course/      # Processed course data
│       ├── course_topics.json
│       └── course_code_examples.json
└── tools-in-data-science-public/  # Course materials
```

## 🎯 Key Improvements Over Basic Version

1. **AI-Powered Responses**: Uses Gemini 2.5 Flash instead of simple keyword matching
2. **Semantic Understanding**: Embedding-based search finds relevant content by meaning
3. **Comprehensive Context**: Combines multiple data sources for better answers
4. **Real-time Processing**: Fast response times with efficient caching
5. **Enhanced Testing**: Comprehensive test suite covering all functionality
6. **Better Error Handling**: Robust error handling and logging
7. **Scalable Architecture**: Designed for easy extension and maintenance

## 🚀 Performance

- **Response Time**: 3-12 seconds for complex questions
- **Accuracy**: High relevance through semantic search
- **Scalability**: Efficient embedding lookup and caching
- **Reliability**: Comprehensive error handling and fallbacks

## 🔮 Future Enhancements

- **Image Analysis**: Integrate Gemini Vision for image-based questions
- **Conversation Memory**: Add conversation context for follow-up questions
- **Advanced RAG**: Implement more sophisticated retrieval-augmented generation
- **Real-time Updates**: Automatic knowledge base updates from new forum posts
- **Multi-language Support**: Support for questions in multiple languages

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For questions or issues:
- Check the test examples in `test_gemini_api.py`
- Review the API documentation above
- Create an issue in the repository

---

**Built with ❤️ for the IIT Madras Tools in Data Science course**

*Powered by Google Gemini 2.5 Flash, Sentence Transformers, and comprehensive course knowledge* 