# TDS Virtual TA - Local Deployment

A fully functional Virtual Teaching Assistant for the Tools in Data Science (TDS) course.

## Features

✅ **AI-Powered Responses**: Uses Gemini 2.0 Flash for intelligent answers  
✅ **Semantic Search**: 309 pre-computed embeddings for course content  
✅ **Image Support**: Can analyze screenshots and images  
✅ **Course Integration**: Direct links to relevant course materials and discussions  
✅ **Fast Performance**: ~1-2 second response times  

## Quick Start

### Prerequisites
- Python 3.8+
- Git
- Google Gemini API key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd P1
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   ```bash
   # Copy the example file
   cp env.example .env
   
   # Edit .env and add your API keys:
   # GEMINI_API_KEY=your_gemini_api_key_here
   # AIPIPE_TOKEN=your_aipipe_token_here (optional, for evaluations)
   ```

4. **Run the Server**:
   ```bash
   python simple_virtual_ta.py
   ```

5. **Test the Web Interface**:
   Open `tds_virtual_ta_chat.html` in your browser or visit deployed URL

### For Evaluations (Optional)

Install promptfoo for testing:
```bash
npm install -g promptfoo
promptfoo eval --config project-tds-virtual-ta-aipipe-working.yaml
```

## API Usage

**POST** `http://localhost:5000/api/`

```json
{
  "question": "What is the GPT model?",
  "image": "base64_encoded_image_optional"
}
```

**Response**:
```json
{
  "answer": "Detailed AI-generated response...",
  "links": [
    {"text": "Course Material", "url": "https://..."},
    {"text": "Discussion Thread", "url": "https://..."}
  ],
  "search_results_count": 6
}
```

## Test Results

All 5 test cases pass with high-quality responses:
- GPT model guidance ✅
- GA4 scoring information ✅  
- Docker/Podman recommendations ✅
- Future exam date handling ✅
- Comprehensive technical answers ✅

## Architecture

- **Flask API** for HTTP endpoints
- **Gemini 2.0 Flash** for text generation
- **Gemini 2.0 Flash** for image analysis
- **Text Embedding 004** for semantic search
- **NumPy** for efficient similarity computation
- **Pre-computed embeddings** for fast retrieval

## Data

- **165 Discourse topics** from course discussions
- **143 course files** from official materials
- **309 total embeddings** (768 dimensions each)
- **1.8MB embeddings cache** for instant startup