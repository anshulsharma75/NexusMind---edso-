import httpx
import json
import time
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from collections import defaultdict

app = FastAPI(
    title="ConvAI — API Gateway",
    description="Production-grade conversational AI gateway with intent classification",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://ml_service:8001")
LOG_FILE = "query_logs.jsonl"
MAX_QUERY_LENGTH = 500
RATE_LIMIT = 10
RATE_WINDOW = 60

request_counts = defaultdict(list)
query_history = []


def is_rate_limited(user_id: str) -> bool:
    now = time.time()
    request_counts[user_id] = [
        t for t in request_counts[user_id] if now - t < RATE_WINDOW
    ]
    if len(request_counts[user_id]) >= RATE_LIMIT:
        return True
    request_counts[user_id].append(now)
    return False


def log_query(entry: dict):
    query_history.append(entry)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous"


class QueryResponse(BaseModel):
    user_id: str
    query: str
    intent: str
    confidence: float
    response: str
    model_type: str
    timestamp: str


@app.get("/")
def root():
    return {
        "service": "ConvAI API Gateway",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": ["/query", "/health", "/history", "/stats"],
    }


@app.get("/health")
async def health():
    ml_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"{ML_SERVICE_URL}/health")
            ml_status = "ok" if res.status_code == 200 else "degraded"
    except Exception:
        ml_status = "unreachable"
    return {
        "gateway": "ok",
        "ml_service": ml_status,
        "total_requests": len(query_history),
    }


@app.get("/history")
def get_history(limit: int = 20, user_id: str = None):
    data = query_history[-limit:]
    if user_id:
        data = [q for q in data if q.get("user_id") == user_id]
    return {"count": len(data), "history": data}


@app.get("/stats")
def get_stats():
    if not query_history:
        return {"message": "No queries yet"}

    intent_counts = defaultdict(int)
    confidences = []
    for q in query_history:
        intent_counts[q.get("intent", "unknown")] += 1
        confidences.append(q.get("confidence", 0))

    return {
        "total_queries": len(query_history),
        "intent_distribution": dict(intent_counts),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "unique_users": len(set(q.get("user_id") for q in query_history)),
    }


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(request.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long. Max {MAX_QUERY_LENGTH} characters allowed."
        )

    if is_rate_limited(request.user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT} requests per minute."
        )

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            res = await client.post(
                f"{ML_SERVICE_URL}/classify",
                json={"query": request.query},
            )
            res.raise_for_status()
            ml_result = res.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"ML service unavailable: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"ML service error: {str(e)}")

    timestamp = datetime.now(timezone.utc).isoformat()

    log_entry = {
        "user_id": request.user_id,
        "query": request.query,
        "intent": ml_result["intent"],
        "confidence": ml_result["confidence"],
        "response": ml_result["response"],
        "model_type": ml_result.get("model_type", "unknown"),
        "timestamp": timestamp,
    }
    log_query(log_entry)

    return QueryResponse(**log_entry)
