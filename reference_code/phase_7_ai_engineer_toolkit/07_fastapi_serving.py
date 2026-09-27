# Source: AI_Learning_Cursor lines 29248-29573
# Original transcript phase: 6 - ADVANCED TOPICS & PORTFOLIO
# Nearest header: #### CODE: Model Serving API
# Title: MODEL SERVING WITH FASTAPI
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
MODEL SERVING WITH FASTAPI
==========================
Deploy your GPT model as a REST API.
"""

# Save this as: api.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import asyncio
import time
import uvicorn

# =================================
# 1. API SETUP
# =================================

app = FastAPI(
    title="Mini-GPT API",
    description="API for text generation with your custom GPT model",
    version="1.0.0",
)

# =================================
# 2. REQUEST/RESPONSE MODELS
# =================================

class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="The text prompt to complete")
    max_tokens: int = Field(100, ge=1, le=500, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(50, ge=1, le=100, description="Top-k sampling")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop generation at these strings")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "To be, or not to be",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_k": 50,
            }
        }


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    prompt: str
    tokens_generated: int
    generation_time: float


class ModelInfo(BaseModel):
    """Model information."""
    model_name: str
    parameters: int
    vocab_size: int
    max_context_length: int
    device: str


# =================================
# 3. MODEL LOADING
# =================================

class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_info = None
    
    def load_model(self, model_path: str):
        """Load the trained model."""
        print(f"Loading model from {model_path}...")
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # In production, load your actual model
        # self.model = GPT.load(model_path).to(self.device)
        # self.tokenizer = CharacterTokenizer.load(model_path)
        
        # For demo, we'll simulate
        self.model_info = ModelInfo(
            model_name="mini-gpt-shakespeare",
            parameters=10_800_000,
            vocab_size=65,
            max_context_length=256,
            device=str(self.device),
        )
        
        print(f"Model loaded on {self.device}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> tuple[str, int]:
        """Generate text from prompt."""
        
        # In production, use your actual model
        # generated = self.model.generate(...)
        
        # Demo: simulate generation
        import random
        demo_completions = [
            ", that is the question:\nWhether 'tis nobler in the mind to suffer",
            ":\nWhether to suffer the slings and arrows of outrageous fortune",
            ", whether to sleep, perchance to dream",
        ]
        
        # Simulate generation time
        time.sleep(0.5 + random.random())
        
        generated = prompt + random.choice(demo_completions)
        tokens_generated = len(generated) - len(prompt)
        
        return generated, tokens_generated
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ):
        """Generate text with streaming."""
        
        # Simulate streaming generation
        demo_text = ", that is the question:\nWhether 'tis nobler in the mind to suffer"
        
        yield prompt
        
        for char in demo_text:
            await asyncio.sleep(0.05)  # Simulate token generation
            yield char


# Initialize model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_manager.load_model("./gpt_shakespeare")


# =================================
# 4. API ENDPOINTS
# =================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Mini-GPT API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model_info is not None,
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if model_manager.model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_manager.model_info


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from a prompt."""
    
    if model_manager.model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        generated_text, tokens_generated = model_manager.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            stop_sequences=request.stop_sequences,
        )
        
        generation_time = time.time() - start_time
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream")
async def generate_text_stream(request: GenerationRequest):
    """Generate text with streaming response."""
    
    if model_manager.model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def stream_generator():
        async for token in model_manager.generate_stream(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            yield token
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
    )


# =================================
# 5. RATE LIMITING AND SECURITY
# =================================

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
import time as time_module

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple rate limiter
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time_module.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        self.requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter(requests_per_minute=60)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )
    
    response = await call_next(request)
    return response


# =================================
# 6. RUN SERVER
# =================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════╗
    ║          Mini-GPT API Server             ║
    ╠══════════════════════════════════════════╣
    ║  Endpoints:                              ║
    ║    GET  /          - Welcome             ║
    ║    GET  /health    - Health check        ║
    ║    GET  /model/info- Model info          ║
    ║    POST /generate  - Generate text       ║
    ║    POST /generate/stream - Streaming     ║
    ║                                          ║
    ║  Docs: http://localhost:8000/docs        ║
    ╚══════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
