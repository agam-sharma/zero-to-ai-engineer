# Source: AI_Learning_Cursor lines 29706-29943
# Original transcript phase: 6 - ADVANCED TOPICS & PORTFOLIO
# Nearest header: #### CODE: Production Best Practices
# Title: PRODUCTION BEST PRACTICES
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
PRODUCTION BEST PRACTICES
=========================
Things to consider for production deployment.
"""

# =================================
# 1. LOGGING
# =================================

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger("mini-gpt-api")

# Log requests
def log_request(request_id: str, prompt: str, response_time: float):
    logger.info(f"Request {request_id}: prompt_len={len(prompt)}, time={response_time:.3f}s")


# =================================
# 2. METRICS
# =================================

from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

TOKENS_GENERATED = Counter(
    'tokens_generated_total',
    'Total tokens generated'
)


# =================================
# 3. CACHING
# =================================

from functools import lru_cache
import hashlib

class ResponseCache:
    """Simple in-memory cache for responses."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def _get_key(self, prompt: str, params: dict) -> str:
        """Generate cache key."""
        param_str = str(sorted(params.items()))
        return hashlib.sha256(f"{prompt}{param_str}".encode()).hexdigest()
    
    def get(self, prompt: str, params: dict):
        """Get cached response."""
        key = self._get_key(prompt, params)
        return self.cache.get(key)
    
    def set(self, prompt: str, params: dict, response: str):
        """Cache a response."""
        if len(self.cache) >= self.max_size:
            # Remove oldest (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self._get_key(prompt, params)
        self.cache[key] = response


# =================================
# 4. BATCHING
# =================================

import asyncio
from typing import List, Tuple
import queue

class BatchProcessor:
    """Process multiple requests in batches for efficiency."""
    
    def __init__(self, model, batch_size: int = 8, max_wait_ms: int = 50):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()
        self.running = False
    
    async def add_request(self, prompt: str, params: dict) -> str:
        """Add request to batch queue."""
        future = asyncio.Future()
        await self.queue.put((prompt, params, future))
        return await future
    
    async def process_batches(self):
        """Process batches continuously."""
        self.running = True
        
        while self.running:
            batch = []
            
            # Collect batch
            try:
                while len(batch) < self.batch_size:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.max_wait_ms / 1000
                    )
                    batch.append(item)
            except asyncio.TimeoutError:
                pass
            
            if batch:
                # Process batch
                prompts = [item[0] for item in batch]
                params = [item[1] for item in batch]
                futures = [item[2] for item in batch]
                
                # Generate all at once (if model supports batching)
                results = await self._generate_batch(prompts, params)
                
                # Return results
                for future, result in zip(futures, results):
                    future.set_result(result)
    
    async def _generate_batch(self, prompts: List[str], params: List[dict]) -> List[str]:
        """Generate for a batch of prompts."""
        # In practice, use model's batch generation
        results = []
        for prompt, param in zip(prompts, params):
            # Simulate
            result = prompt + "... [generated]"
            results.append(result)
        return results


# =================================
# 5. SECURITY CHECKLIST
# =================================

security_checklist = """
PRODUCTION SECURITY CHECKLIST:

✓ Input Validation
  - Limit prompt length
  - Sanitize inputs
  - Validate all parameters

✓ Rate Limiting
  - Per-IP limits
  - Per-API-key limits
  - Burst protection

✓ Authentication
  - API keys for production
  - JWT for user-specific access
  - OAuth for third-party access

✓ HTTPS
  - Always use HTTPS in production
  - Valid SSL certificates
  - HSTS headers

✓ Monitoring
  - Request logging
  - Error tracking (Sentry)
  - Performance metrics (Prometheus)

✓ Content Filtering
  - Block harmful prompts
  - Filter inappropriate outputs
  - Rate limit sensitive topics

✓ Resource Limits
  - Max tokens per request
  - Timeout limits
  - Memory limits per container
"""

print(security_checklist)


# =================================
# 6. SCALING
# =================================

scaling_considerations = """
SCALING YOUR API:

1. HORIZONTAL SCALING
   - Run multiple container replicas
   - Load balancer (nginx, HAProxy)
   - Kubernetes for orchestration

2. GPU OPTIMIZATION
   - Use mixed precision (FP16)
   - Batch requests together
   - Model parallelism for large models

3. CACHING LAYERS
   - Redis for response caching
   - CDN for static content
   - Model weight caching

4. ASYNC PROCESSING
   - Queue long-running requests
   - WebSocket for real-time updates
   - Background job processing

5. MODEL OPTIMIZATION
   - Quantization (INT8, INT4)
   - Distillation (smaller student model)
   - ONNX Runtime for inference
"""

print(scaling_considerations)
