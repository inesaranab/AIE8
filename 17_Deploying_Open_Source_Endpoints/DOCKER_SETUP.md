# Docker Setup Guide

This guide will help you containerize and run your Open Source Chatbot API using Docker.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (included with Docker Desktop)
- Together AI API key

## Quick Start

### 1. Create Environment File

Create a `.env` file in the project root:

```bash
TOGETHER_API_KEY=your-together-api-key-here
RAG_DATA_DIR=data
```

> **⚠️ Important:** Never commit your `.env` file to git! It's already in `.gitignore`.

### 2. Update Dependencies

First, update your dependencies:

```bash
uv sync
```

### 3. Build and Run with Docker Compose

```bash
# Build and start the container in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### 4. Test the API

Once running, test your API:

```bash
# Using the test script
python test_api.py

# Or with a custom message
python test_api.py "What is machine learning?"

# Or using curl
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

## API Endpoints

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Chat Endpoint

**POST** `/chat`

Request body:
```json
{
  "message": "Your question here"
}
```

Response:
```json
{
  "response": "AI's answer here"
}
```

## Docker Commands Reference

### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up --build -d

# View running containers
docker-compose ps
```

### Using Docker Directly

```bash
# Build the image
docker build -t open-source-chatbot .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e TOGETHER_API_KEY=your-api-key-here \
  -e RAG_DATA_DIR=/app/data \
  -v $(pwd)/data:/app/data:ro \
  --name chatbot-api \
  open-source-chatbot

# View logs
docker logs -f chatbot-api

# Stop and remove
docker stop chatbot-api
docker rm chatbot-api
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs

# Check if port 8000 is already in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Mac/Linux
```

### API returns 500 errors
- Check if your TOGETHER_API_KEY is set correctly
- Check if your dedicated endpoint identifier is correct in `app/agent.py` and `app/rag.py`
- View logs: `docker-compose logs -f`

### Updates not reflecting
```bash
# Rebuild the image
docker-compose up --build -d
```

### Access container shell
```bash
# Using docker-compose
docker-compose exec api bash

# Using docker
docker exec -it chatbot-api bash
```

## Development Workflow

1. Make changes to your code
2. Rebuild the container: `docker-compose up --build -d`
3. Test with: `python test_api.py`
4. Check logs: `docker-compose logs -f`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TOGETHER_API_KEY` | Your Together AI API key | Required |
| `RAG_DATA_DIR` | Directory containing PDF documents | `/app/data` |

## Production Considerations

For production deployment:

1. Use a production-grade ASGI server configuration
2. Add health check endpoints
3. Implement proper logging
4. Use secrets management (not `.env` files)
5. Set up monitoring and alerting
6. Configure auto-scaling based on load
7. Use container orchestration (Kubernetes, ECS, etc.)

## Notes

- The `data/` directory is mounted read-only in the container
- The API runs on port 8000 by default
- Logs are output to stdout/stderr for Docker to capture
- The container will automatically restart unless stopped manually

## Query the Endpoint

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! What can you do?"}'


