# TDS Virtual TA 🤖

**An AI-powered Teaching Assistant for the Tools in Data Science course at IIT Madras**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Vercel](https://img.shields.io/badge/Vercel-Deploy-black.svg)](https://vercel.com/)
[![Gemini 2.5 Flash](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-purple.svg)](https://ai.google.dev/)

## 🌟 Features

### 🧠 AI-Powered Answer Generation
- **Gemini 2.5 Flash Integration**: Uses Google's latest language model for intelligent, contextual responses
- **Image Processing**: Full support for image analysis using Gemini Vision (screenshots, diagrams, code images)
- **Gemini Embeddings**: Advanced semantic search using Google's text-embedding-004 model
- **Multi-source Knowledge**: Combines Discourse forum discussions with official course materials
- **Smart Prioritization**: Staff answers → Course content → Community discussions → Code examples
- **Lightweight Deployment**: No heavy ML dependencies, uses Gemini API for all AI tasks

### 📚 Comprehensive Knowledge Base
- **163 Discourse Topics**: Historical student discussions and Q&A from Jan-Apr 2025
- **49 Q&A Pairs**: Curated question-answer pairs from forum discussions
- **137 Course Topics**: Official course materials covering all TDS concepts
- **184 Code Examples**: Practical code snippets with context and explanations

### 🚀 Deployment Options
- **Local Development**: Flask server with full semantic search capabilities
- **Vercel**: Serverless deployment with global edge distribution
- **Docker Support**: Containerized deployment for easy scaling

## 🏗️ Architecture

```
TDS Virtual TA
├── 🧠 Gemini 2.5 Flash (Answer Generation)
├── 🔍 Semantic Search Engine
├── 📚 Knowledge Base
│   ├── Discourse Data (163 topics, 49 Q&A pairs)
│   └── Course Content (137 topics, 184 code examples)
├── 🌐 API Endpoints
│   ├── Local Flask Server (app_gemini.py)
│   └── Vercel Serverless Functions
└── 🧪 Testing & Evaluation
    ├── Comprehensive Test Suite
    └── PromptFoo Integration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/tds-virtual-ta.git
cd tds-virtual-ta

# Install Python dependencies
pip install -r requirements.txt

# Setup environment variables
cp env.example .env
# Edit .env and add your GEMINI_API_KEY (get from https://aistudio.google.com/app/apikey)
```

### 2. Process Knowledge Base

```bash
# Process Discourse forum data
python scripts/process_discourse.py

# Process course content
python scripts/process_course_content.py
```

### 3. Run Locally

```bash
# Start the enhanced Gemini-powered API
python app_gemini.py

# Test the API
python test_gemini_api.py
```

The server will start on `http://localhost:5000`

## 🌐 Vercel Deployment

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Deploy to Vercel

```bash
# Login to Vercel
vercel login

# Deploy the application
vercel

# Set environment variables in Vercel dashboard or CLI
vercel env add GEMINI_API_KEY
```

Your API will be available at `https://your-project.vercel.app`

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
  "answer": "Prompt engineering is the art and science of crafting effective prompts...",
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

### Health Check
```http
GET /api/health
```

### Statistics
```http
GET /api/stats
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

## 📷 Image Processing with Gemini Vision

The TDS Virtual TA supports image analysis using Gemini Vision. You can send screenshots, diagrams, code images, or any visual content along with your questions.

### Supported Image Formats
- **WebP, PNG, JPEG, GIF**: All common image formats
- **Screenshots**: Error messages, code snippets, terminal outputs
- **Diagrams**: Architecture diagrams, flowcharts, data visualizations
- **Code Images**: Screenshots of code from IDEs or notebooks

### Usage Examples

**Python (using requests):**
```python
import requests
import base64

# Read and encode image
with open("screenshot.png", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# Send request with image
response = requests.post("http://localhost:5000/api/", json={
    "question": "What's wrong with this code?",
    "image": image_data
})
```

**PowerShell:**
```powershell
# Read and encode image
$imageBytes = [System.IO.File]::ReadAllBytes("error_screenshot.png")
$imageBase64 = [System.Convert]::ToBase64String($imageBytes)

# Send request
$response = Invoke-RestMethod -Uri "http://localhost:5000/api/" -Method Post -Body (@{
    question = "How do I fix this error?"
    image = $imageBase64
} | ConvertTo-Json) -ContentType "application/json"
```

**cURL (Linux/Mac):**
```bash
curl "http://localhost:5000/api/" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Explain this diagram\", \"image\": \"$(base64 -w0 diagram.png)\"}"
```

### Real Example from Project Description

The API can analyze the provided screenshot and answer questions about it:

```bash
# Question: "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?"
# With image: project-tds-virtual-ta-q1.webp

# Response: "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports 
# `gpt-4o-mini`. Use the OpenAI API directly for this question."
```

### Test Image Processing

```bash
# Test with the provided image
python test_image_processing.py

# Or use PowerShell script
powershell -ExecutionPolicy Bypass -File test_image_powershell.ps1
```

## 🧪 Testing & Evaluation

### Run Test Suite
```bash
# Test the enhanced Gemini API
python test_gemini_api.py

# Test with PromptFoo (requires Node.js)
npx promptfoo eval --config project-tds-virtual-ta-promptfoo.yaml
```

### Example Questions

**Discourse-Based Questions:**
- "How do I submit my TDS project?"
- "What is the deadline for GA1?"
- "I'm getting a Docker error when running my code"

**Course Content Questions:**
- "What is prompt engineering and how do I use it?"
- "Explain vector databases and their use cases"
- "How do I scrape data with Python? Show me examples"

**Complex Questions:**
- "What's the difference between ChromaDB and LanceDB?"
- "How do I deploy my TDS project using Docker?"

## 📊 Knowledge Base Statistics

| Category | Count | Description |
|----------|-------|-------------|
| **Discourse Topics** | 163 | Forum discussions from Jan-Apr 2025 |
| **Q&A Pairs** | 49 | Curated question-answer pairs |
| **Course Topics** | 137 | Official course materials |
| **Code Examples** | 184 | Practical code snippets |
| **Total Keywords** | 6,054 | Indexed for semantic search |
| **Content Categories** | 9 | LLM/AI, Data Processing, Web Scraping, etc. |

## 🔧 Configuration

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here
```

### Vercel Configuration

Set environment variables in your Vercel dashboard or using CLI:
- `GEMINI_API_KEY`: Your Google Gemini API key

## 📁 Project Structure

```
tds-virtual-ta/
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 .gitignore                   # Git ignore rules
├── 📄 requirements.txt             # Python dependencies
├── 📄 env.example                  # Environment variables template
├── 🐍 app_gemini.py                # Enhanced Flask API with Gemini
├── 🧪 test_gemini_api.py           # Comprehensive test suite
├── ⚙️ project-tds-virtual-ta-promptfoo.yaml  # PromptFoo evaluation config
├── 📁 scripts/                     # Data processing scripts
│   ├── process_discourse.py        # Discourse data processing
│   ├── process_course_content.py   # Course content processing
│   ├── processed/                  # Processed Discourse data
│   └── processed_course/           # Processed course data
└── 📁 tools-in-data-science-public/  # Course materials (git submodule)
```

## 🎯 Performance

- **Response Time**: 3-12 seconds for complex questions
- **Accuracy**: High relevance through semantic search
- **Scalability**: Serverless deployment with Vercel's global edge network
- **Reliability**: Comprehensive error handling and fallbacks

## 🔮 Future Enhancements

- **Conversation Memory**: Add conversation context for follow-up questions
- **Advanced RAG**: Implement more sophisticated retrieval-augmented generation
- **Real-time Updates**: Automatic knowledge base updates from new forum posts
- **Multi-language Support**: Support for questions in multiple languages
- **Batch Processing**: Support for processing multiple questions simultaneously
- **Enhanced Caching**: Implement intelligent caching for faster response times

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IIT Madras** for the Tools in Data Science course
- **Google** for the Gemini 2.5 Flash API
- **Vercel** for serverless deployment platform
- **Sentence Transformers** for semantic search capabilities
- **PromptFoo** for evaluation framework

## 📞 Support

- 📧 Create an issue in this repository
- 📚 Check the [course materials](https://tds.s-anand.net/)
- 💬 Ask in the [TDS Discourse forum](https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34)

---

**Built with ❤️ for the IIT Madras Tools in Data Science course**

*Powered by Google Gemini 2.5 Flash, Vercel, and comprehensive course knowledge* 