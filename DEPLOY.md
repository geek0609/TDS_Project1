# Vercel Deployment Guide for TDS Virtual TA

## Prerequisites

1. **Vercel CLI** installed:
   ```bash
   npm install -g vercel
   ```

2. **Gemini API Key** from Google AI Studio

## Deployment Steps

### 1. Set Environment Variables

In your Vercel dashboard or via CLI, set:
```bash
vercel env add GEMINI_API_KEY
# Enter your actual Gemini API key when prompted
```

### 2. Deploy to Vercel

```bash
# Login to Vercel (if not already)
vercel login

# Deploy the project
vercel --prod
```

### 3. Configuration Details

The project is already configured with:
- `vercel.json` - Vercel deployment configuration
- `api/index.py` - Serverless function entry point
- `requirements.txt` - Python dependencies
- Data files in `data/` and `tools-in-data-science-public/`

### 4. Endpoints

After deployment, your API will be available at:
- `https://your-domain.vercel.app/api/` - Main chat endpoint
- `https://your-domain.vercel.app/api/search` - Search endpoint  
- `https://your-domain.vercel.app/api/health` - Health check

### 5. Testing

Test with curl:
```bash
curl "https://your-domain.vercel.app/api/health"

curl "https://your-domain.vercel.app/api/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Docker?"}'
```

## Important Notes

- **Cold Start**: First request may take 10-15 seconds
- **Size Limit**: Lambda is configured for 50MB max
- **Timeout**: Vercel has 10-second timeout for hobby plan
- **Data**: Course materials and discourse data are included in deployment

## Troubleshooting

1. **Import errors**: Check that `data/` directory is included
2. **API key issues**: Verify environment variable is set correctly
3. **Timeout**: Consider upgrading to Pro plan for longer timeouts
4. **Size issues**: Current deployment should be under 50MB limit 