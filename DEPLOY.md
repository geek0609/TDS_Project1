# TDS Virtual TA - Deployment Guide

## Vercel Deployment

The project is ready for Vercel deployment with pre-computed embeddings for fast serverless execution.

### Prerequisites
1. Vercel account
2. Gemini API key

### Deployment Steps

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Set Environment Variables**:
   Create a `.env` file or set in Vercel dashboard:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Deploy to Vercel**:
   ```bash
   vercel --prod
   ```

### File Structure for Deployment
```
├── api/
│   ├── index.py              # Vercel entry point
│   └── virtual_ta_serverless.py  # Optimized serverless implementation
├── embeddings_cache/
│   ├── embeddings.npy        # Pre-computed embeddings (1.8MB)
│   ├── metadata.json         # Content metadata (968KB)
│   └── gemini_embeddings.pkl # Backup embeddings
├── vercel.json              # Vercel configuration
├── requirements-vercel.txt  # Python dependencies
└── data/
    └── discourse_data.json  # Course data
```

### API Endpoints
- `GET /api/health` - Health check
- `POST /api/` - Main chat endpoint
- `POST /api/search` - Search endpoint

### Features
✅ **Pre-computed embeddings** - Fast cold starts  
✅ **Optimized for serverless** - <50MB package size  
✅ **6 relevant links per response** - Better user experience  
✅ **Image support** - Handles multimodal queries  
✅ **Intent analysis** - Smart response routing  
✅ **CORS enabled** - Frontend integration ready  

### Performance
- **Cold start**: ~2-3 seconds
- **Warm response**: ~500ms-1s
- **Memory usage**: ~200MB
- **Package size**: ~45MB

### Testing
After deployment, test with:
```bash
curl -X POST https://your-app.vercel.app/api/ \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I use Docker for this course?"}'
```

### Environment Variables in Vercel
1. Go to your Vercel project dashboard
2. Settings → Environment Variables
3. Add: `GEMINI_API_KEY` = `your_api_key`
4. Redeploy if needed

### Monitoring
- Check logs in Vercel dashboard
- Monitor `/api/health` endpoint
- Track response times and errors

---

## Local Development

For local testing of the serverless version:
```bash
cd api
python virtual_ta_serverless.py
```

The server will start on `http://localhost:5000` 