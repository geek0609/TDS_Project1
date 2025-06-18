# TDS Virtual TA - Chat Interface

A modern, interactive ChatGPT-like web interface for the TDS Virtual TA API.

## Features

- 🎨 **Modern Design**: Clean, responsive ChatGPT-inspired interface
- 💬 **Real-time Chat**: Interactive messaging with typing indicators
- 📱 **Mobile Responsive**: Works perfectly on all device sizes
- 🔗 **Smart Links**: Displays relevant course materials and discourse links
- ⚡ **Fast Responses**: Optimized for quick API interactions
- 🎯 **Status Monitoring**: Real-time connection status indicator
- 🌐 **Multiple API Sources**: Switch between Vercel hosted and local development
- ⚙️ **Custom API URLs**: Support for custom API endpoints

## How to Use

### Option 1: Use Hosted Vercel API (Recommended)

The chat interface is pre-configured to use the hosted Vercel API at `https://tds-project1-nh0zbupt1-geek0609s-projects.vercel.app`.

### Option 2: Use Local Development Server

If you want to use a local server:

```bash
python virtual_ta.py
```

The server will start on `http://localhost:5000`

### Step 2: Open the Chat Interface

Open one of these HTML files in your web browser:

- `tds_virtual_ta_chat.html` - **Main chat interface** (with API selector)
- `chat.html` - Alternative interface
- `index.html` - Another interface option

You can open it by:
1. Double-clicking the HTML file
2. Right-clicking and selecting "Open with" → Your browser
3. Dragging the file into your browser window

### Step 3: Choose Your API Source

In `tds_virtual_ta_chat.html`, you can choose between:
- **Vercel (Hosted)** - Uses the deployed API (default)
- **Local Development** - Uses your local server on port 5000
- **Custom URL** - Enter any custom API endpoint

### Step 4: Start Chatting!

- Type your question in the input field at the bottom
- Press **Enter** to send (or click the send button)
- Use **Shift+Enter** for new lines
- The TA will respond with helpful answers and relevant links

## Example Questions

Try asking questions like:

- "How do I install Docker?"
- "What's the difference between Git and GitHub?"
- "Help me with Python data analysis"
- "Explain the assignment requirements"
- "How to use pandas for data manipulation?"

## Interface Features

### Status Indicator
- **Green dot + "Connected"**: API is working properly
- **Red dot + "Disconnected"**: API is not responding

### Message Types
- **Your messages**: Appear on the right with blue background
- **TA responses**: Appear on the left with white background
- **Resource links**: Clickable links to course materials and discussions

### Smart Formatting
The interface supports basic markdown formatting:
- **Bold text** with `**text**`
- *Italic text* with `*text*`
- `Code snippets` with backticks

## Troubleshooting

### "Disconnected" Status
If you see "Disconnected" status:
1. Make sure `virtual_ta.py` is running on port 5000
2. Check that there are no firewall issues
3. Refresh the page after starting the server

### CORS Issues
If you encounter CORS errors:
1. The `virtual_ta.py` already includes CORS headers
2. Try opening the HTML file through a local server instead of file://
3. Check browser console for specific error messages

### No Response from API
If the TA doesn't respond:
1. Check the browser's developer console (F12) for errors
2. Verify the API is running: visit `http://localhost:5000/api/health`
3. Make sure you have the required dependencies installed

## Technical Details

### API Endpoints Used
- `GET /api/health` - Connection status check
- `POST /api/ask` - Send questions and get responses

### Browser Compatibility
- Chrome/Edge/Safari: Full support
- Firefox: Full support
- Mobile browsers: Responsive design supported

### Performance
- Automatic textarea resizing
- Smooth animations and transitions
- Efficient message rendering
- Typing indicators for better UX

## Customization

You can easily customize the interface by modifying the CSS in the HTML file:

- **Colors**: Change the gradient colors in the CSS
- **Layout**: Modify container sizes and spacing
- **API URL**: Change the `apiUrl` in the JavaScript if using a different port

## Deployment

### Deploy to Vercel

To deploy your own instance:

1. Install Vercel CLI: `npm install -g vercel`
2. Run the deployment script: `./deploy.sh` (or `vercel --prod`)
3. Update the API URL in the chat interface if needed

### Environment Variables

For the Vercel deployment, make sure to set:
- `GEMINI_API_KEY` - Your Google Gemini API key

## Files

- `tds_virtual_ta_chat.html` - **Main chat interface** (with API selector)
- `chat.html` - Alternative interface
- `index.html` - Another interface option
- `virtual_ta.py` - Local development API server
- `simple_virtual_ta.py` - Vercel-optimized API server
- `vercel.json` - Vercel deployment configuration
- `deploy.sh` - Deployment script

## Default API Configuration

The chat interfaces are configured with these default API URLs:
- **Primary**: `https://tds-project1-nh0zbupt1-geek0609s-projects.vercel.app` (Vercel hosted)
- **Fallback**: `http://localhost:5000` (Local development)

Enjoy chatting with your TDS Virtual TA! 🤖📚 