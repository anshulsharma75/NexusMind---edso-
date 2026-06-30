# ConvAI — End-to-End Conversational AI System
### Edxso AI Engineer Intern — Assignment 3 · Expert Level

> **Author:** Anshul Sharma  
> **Stack:** FastAPI · HuggingFace Transformers · Scikit-learn · MLflow · Docker

---

## System Architecture

```
  User / Client
       │
       │  POST /query
       ▼
┌──────────────────────────────────────────────────────────┐
│                     Docker Network                        │
│                                                            │
│   ┌──────────────────────────────────┐                    │
│   │         API Gateway               │                    │
│   │         FastAPI  ·  :8000         │                    │
│   │                                   │                    │
│   │  • Input validation               │                    │
│   │  • Rate limiting (10 req/min)     │                    │
│   │  • Request logging (JSONL)        │                    │
│   │  • Query history & stats          │                    │
│   │  • ML service health check        │                    │
│   └──────────────┬────────────────────┘                    │
│                  │  HTTP POST /classify                    │
│                  ▼                                         │
│   ┌──────────────────────────────────┐                    │
│   │         ML Service                │                    │
│   │         FastAPI  ·  :8001         │                    │
│   │                                   │                    │
│   │  Primary:  facebook/bart-large-   │                    │
│   │            mnli (zero-shot NLI)   │                    │
│   │                                   │                    │
│   │  Fallback: TF-IDF +               │                    │
│   │            LogisticRegression     │                    │
│   │                                   │                    │
│   │  • Confidence threshold gating    │                    │
│   └──────────────────────────────────┘                    │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

---

## Services

| Service | Port | Tech | Role |
|---|---|---|---|
| `api_gateway` | 8000 | FastAPI | Public endpoint, rate limiting, logging, routing |
| `ml_service` | 8001 | FastAPI + HuggingFace | Zero-shot intent classification |

**Supported Intents:**
`greeting` · `farewell` · `product_inquiry` · `complaint` · `support_request` · `general_question`

---

## Quick Start

### With Docker (recommended)

```bash
git clone https://github.com/anshulsharma75/convai-edxso
cd convai-edxso
docker-compose up --build
```

> First build downloads `facebook/bart-large-mnli` (~1.6GB). Subsequent builds use Docker cache.

API: `http://localhost:8000`  
Swagger docs: `http://localhost:8000/docs`

### Without Docker

```bash
# Terminal 1 — ML Service
cd ml_service
pip install -r requirements.txt
uvicorn main:app --port 8001 --reload

# Terminal 2 — API Gateway
cd api_gateway
pip install -r requirements.txt
uvicorn main:app --port 8000 --reload
```

---

## API Reference

### `POST /query` — Classify a query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hi, I need help with my order", "user_id": "user_001"}'
```

```json
{
  "user_id": "user_001",
  "query": "Hi, I need help with my order",
  "intent": "support_request",
  "confidence": 0.94,
  "response": "Our support team is ready to help. Let me connect you immediately.",
  "model_type": "facebook/bart-large-mnli (zero-shot)",
  "timestamp": "2026-06-30T10:30:00+00:00"
}
```

### `GET /health` — Service health

```bash
curl http://localhost:8000/health
```
```json
{"gateway": "ok", "ml_service": "ok", "total_requests": 42}
```

### `GET /history` — Recent query log

```bash
curl http://localhost:8000/history?limit=10
```

### `GET /stats` — Aggregate analytics

```bash
curl http://localhost:8000/stats
```
```json
{
  "total_queries": 42,
  "intent_distribution": {"greeting": 12, "support_request": 8, "...": "..."},
  "avg_confidence": 0.81,
  "unique_users": 5
}
```

### Rate Limiting

10 requests per minute per `user_id`. Exceeding returns `429 Too Many Requests`.

---

## Data Pipeline & MLflow

```bash
cd data_pipeline
pip install mlflow scikit-learn numpy
python train_eval.py

mlflow ui
# Open http://localhost:5000
```

Runs 3 experiments with progressively tuned hyperparameters:

| Run | C | N-gram | Features | F1 Score |
|---|---|---|---|---|
| baseline | 0.5 | (1,1) | 500 | ~0.52 |
| ngram-tuned | 1.0 | (1,2) | 1000 | ~0.59 |
| best-config | 2.0 | (1,2) | 2000 | ~0.59 |

Tracks: accuracy, precision, recall, F1, 5-fold cross-validation.

---

## ML Model

**Primary:** `facebook/bart-large-mnli` (HuggingFace zero-shot classification)
- No fine-tuning required — generalizes to unseen phrasing
- Uses Natural Language Inference to score each candidate intent

**Fallback:** TF-IDF + Logistic Regression
- Trained on 91 labeled samples across 6 intent classes
- Activates automatically if HuggingFace model fails to load
- Guarantees the system never goes down regardless of environment

**Confidence Gating:** Responses below 0.35 confidence return a clarification prompt instead of a potentially wrong intent response — prevents confidently-wrong answers reaching the user.

---

## Production Considerations

- **Logging:** every query is logged to `query_logs.jsonl` for audit and analytics
- **Rate limiting:** per-user sliding window prevents abuse
- **Graceful degradation:** ML service falls back to a local model if the transformer fails to load — no hard dependency on external downloads at runtime
- **Health checks:** Docker Compose won't start the gateway until the ML service is verified healthy

This is a prototype trained on a small labeled set for demonstration. In production, the dataset would scale to thousands of labeled examples and the model would be fine-tuned rather than zero-shot, while the architecture stays the same.

---

## File Structure

```
.
├── api_gateway/
│   ├── main.py              # Gateway: routing, rate limit, logging, stats
│   ├── Dockerfile
│   └── requirements.txt
├── ml_service/
│   ├── main.py              # HuggingFace + fallback classifier
│   ├── Dockerfile
│   └── requirements.txt
├── data_pipeline/
│   └── train_eval.py        # MLflow experiment tracking
├── docker-compose.yml
└── README.md
```

---

## Video Walkthrough

🎥 [Link to async demo](#)
