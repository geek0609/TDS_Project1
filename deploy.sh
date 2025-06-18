#!/bin/bash

echo "🚀 Deploying TDS Virtual TA to Vercel..."

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null
then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Deploy to Vercel
echo "📦 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo "🌐 Your TDS Virtual TA API should now be available on Vercel"
echo "📝 Update the chat interface if needed with your new Vercel URL" 