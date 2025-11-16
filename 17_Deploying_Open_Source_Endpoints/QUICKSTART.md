# Quick Start Guide

Get your chatbot API up and running in 3 minutes! ⚡

## Prerequisites
- Docker Desktop installed and running
- Together AI API key ready

---

## 🚀 Setup (One-time)

### Windows
```cmd
setup.bat
```

### Mac/Linux
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup
1. Create `.env` file:
   ```bash
   TOGETHER_API_KEY=your-key-here
   RAG_DATA_DIR=data
   ```

2. Update dependencies:
   ```bash
   uv sync
   ```

---

## ▶️ Run the API

```bash
docker-compose up -d
```

The API will be available at: **http://localhost:8000**

---

## 🧪 Test the API

### Method 1: Test Script (Easiest)
```bash
python test_api.py
```

### Method 2: Interactive Docs
Open in browser: **http://localhost:8000/docs**

### Method 3: cURL
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

### Method 4: Python
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "What is machine learning?"}
)
print(response.json()["response"])
```

---

## 📊 View Logs

```bash
docker-compose logs -f
```

Press `Ctrl+C` to stop viewing logs.

---

## 🔄 Restart After Code Changes

```bash
docker-compose up --build -d
```

---

## 🛑 Stop the API

```bash
docker-compose down
```

---

## 🐛 Troubleshooting

### Container won't start?
```bash
docker-compose logs
```

### Port 8000 already in use?
**Windows:**
```cmd
netstat -ano | findstr :8000
```

**Mac/Linux:**
```bash
lsof -i :8000
```

### API returns errors?
1. Check if `.env` has your correct `TOGETHER_API_KEY`
2. Check if your endpoint identifier in `app/agent.py` is correct
3. View logs: `docker-compose logs -f`

---

## 📝 Useful Commands

| Command | Description |
|---------|-------------|
| `docker-compose up -d` | Start API in background |
| `docker-compose down` | Stop API |
| `docker-compose logs -f` | View logs (real-time) |
| `docker-compose restart` | Restart API |
| `docker-compose ps` | Check status |
| `docker-compose up --build -d` | Rebuild and restart |

---

## 🎯 What Your Chatbot Can Do

Your API includes an agentic RAG system with:

1. **RAG Tool** - Answers questions from your PDF documents in `data/`
2. **Tavily Search** - Web search for current information
3. **Arxiv Tool** - Search academic papers

Try asking:
- "What information do you have about AI in daily work?" (uses RAG)
- "What's the latest news about AI?" (uses Tavily)
- "Find papers about machine learning" (uses Arxiv)

---

## 📚 More Information

- Full Docker guide: See `DOCKER_SETUP.md`
- API documentation: http://localhost:8000/docs (when running)
- Project README: See `README.md`


