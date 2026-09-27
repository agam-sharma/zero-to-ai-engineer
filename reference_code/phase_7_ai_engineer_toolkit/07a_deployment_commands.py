# Source: AI_Learning_Cursor lines 29662-29698
# Original transcript phase: 6 - ADVANCED TOPICS & PORTFOLIO
# Nearest header: #### CODE: Dockerfile and docker-compose
# Title: DEPLOYMENT COMMANDS
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
DEPLOYMENT COMMANDS
===================
"""

deployment_commands = """
# Build and run with Docker

# 1. Build the image
docker build -t mini-gpt-api .

# 2. Run the container
docker run -p 8000:8000 -v $(pwd)/models:/app/models mini-gpt-api

# Using docker-compose (recommended)
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Test the API
curl http://localhost:8000/health

curl -X POST http://localhost:8000/generate \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "To be, or not to be", "max_tokens": 50}'
"""

print(deployment_commands)
