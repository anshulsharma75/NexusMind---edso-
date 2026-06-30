import re
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ConvAI ML Service", version="2.0.0")

INTENTS = [
    "greeting",
    "farewell",
    "product inquiry",
    "complaint",
    "support request",
    "general question",
]

INTENT_KEY_MAP = {
    "greeting":         "greeting",
    "farewell":         "farewell",
    "product inquiry":  "product_inquiry",
    "complaint":        "complaint",
    "support request":  "support_request",
    "general question": "general_question",
}

INTENT_RESPONSES = {
    "greeting":         "Hello! Welcome to Edxso. How can I assist you today?",
    "farewell":         "Thank you for reaching out. Have a great day!",
    "product_inquiry":  "I'd be happy to walk you through our products and pricing plans.",
    "complaint":        "I sincerely apologize for the inconvenience. Let me escalate this right away.",
    "support_request":  "Our support team is ready to help. Let me connect you immediately.",
    "general_question": "Great question! Let me find the most accurate information for you.",
}

LOW_CONFIDENCE_RESPONSE = (
    "I'm not entirely sure I understood your request. "
    "Could you rephrase it? I can help with product info, support, or general questions."
)

CONFIDENCE_THRESHOLD = 0.35

TRAINING_DATA = [
    ("hi there", "greeting"), ("hello", "greeting"), ("good morning", "greeting"),
    ("hey how are you", "greeting"), ("greetings", "greeting"), ("howdy", "greeting"),
    ("good evening", "greeting"), ("hey", "greeting"), ("hi", "greeting"),
    ("what's up", "greeting"), ("nice to meet you", "greeting"), ("yo", "greeting"),
    ("hello there", "greeting"), ("hi good morning", "greeting"),
    ("bye", "farewell"), ("goodbye", "farewell"), ("see you later", "farewell"),
    ("take care", "farewell"), ("have a good day", "farewell"), ("catch you later", "farewell"),
    ("nice talking to you", "farewell"), ("until next time", "farewell"),
    ("so long", "farewell"), ("farewell", "farewell"), ("cya", "farewell"),
    ("thanks bye", "farewell"), ("goodbye for now", "farewell"),
    ("what is the price", "product_inquiry"), ("how much does it cost", "product_inquiry"),
    ("what services do you offer", "product_inquiry"), ("pricing details", "product_inquiry"),
    ("what features are included", "product_inquiry"), ("any discounts available", "product_inquiry"),
    ("do you have any offers", "product_inquiry"), ("tell me about your plans", "product_inquiry"),
    ("show me your products", "product_inquiry"), ("what plans do you have", "product_inquiry"),
    ("how much is the subscription", "product_inquiry"), ("enterprise pricing", "product_inquiry"),
    ("what does it cost per month", "product_inquiry"), ("is there a free trial", "product_inquiry"),
    ("annual plan pricing", "product_inquiry"), ("compare your plans", "product_inquiry"),
    ("this is not working", "complaint"), ("I have a problem", "complaint"),
    ("bad experience", "complaint"), ("very disappointed", "complaint"),
    ("your service is terrible", "complaint"), ("I want to complain", "complaint"),
    ("unacceptable quality", "complaint"), ("wrong order received", "complaint"),
    ("I am not happy", "complaint"), ("this is broken", "complaint"),
    ("worst experience ever", "complaint"), ("I demand a refund", "complaint"),
    ("you guys messed up", "complaint"), ("this is unacceptable", "complaint"),
    ("I am frustrated", "complaint"), ("terrible service", "complaint"),
    ("I need help", "support_request"), ("can you assist me", "support_request"),
    ("fix my issue please", "support_request"), ("help me with this error", "support_request"),
    ("I am stuck", "support_request"), ("technical support needed", "support_request"),
    ("how do I reset my password", "support_request"), ("I cannot login", "support_request"),
    ("I need assistance", "support_request"), ("please help", "support_request"),
    ("my account is locked", "support_request"), ("I need urgent help", "support_request"),
    ("I cannot access my account", "support_request"), ("app is crashing", "support_request"),
    ("how do I update my profile", "support_request"), ("I forgot my password", "support_request"),
    ("what is AI", "general_question"), ("how does this work", "general_question"),
    ("can you explain", "general_question"), ("what do you do", "general_question"),
    ("what is edxso", "general_question"), ("tell me about yourself", "general_question"),
    ("what can you do", "general_question"), ("how can you help me", "general_question"),
    ("I have a question", "general_question"), ("tell me more", "general_question"),
    ("what is machine learning", "general_question"), ("explain your services", "general_question"),
    ("how does your system work", "general_question"), ("what technology do you use", "general_question"),
    ("are you a bot", "general_question"), ("who built you", "general_question"),
]

classifier = None
USE_TRANSFORMERS = False


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def load_model():
    global classifier, USE_TRANSFORMERS
    try:
        from transformers import pipeline
        print("[INFO] Loading facebook/bart-large-mnli...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
        USE_TRANSFORMERS = True
        print("[INFO] HuggingFace model ready.")
    except Exception as e:
        print(f"[WARN] HuggingFace unavailable: {e}")
        print("[INFO] Loading TF-IDF fallback classifier...")
        load_fallback()


def load_fallback():
    global classifier, USE_TRANSFORMERS
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts  = [preprocess(t) for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    X   = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=500, C=2.0, random_state=42)
    clf.fit(X, labels)

    classifier = {"vectorizer": vec, "model": clf}
    USE_TRANSFORMERS = False
    print("[INFO] Fallback classifier ready.")


load_model()


class ClassifyRequest(BaseModel):
    query: str


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    model_type: str
    low_confidence: bool


@app.get("/")
def root():
    return {
        "service": "ConvAI ML Service",
        "version": "2.0.0",
        "model": "facebook/bart-large-mnli" if USE_TRANSFORMERS else "TF-IDF + LogisticRegression",
        "type": "zero-shot-classification" if USE_TRANSFORMERS else "supervised",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.get("/intents")
def list_intents():
    return {"supported_intents": list(INTENT_KEY_MAP.values())}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    clean = preprocess(request.query)

    if USE_TRANSFORMERS:
        result     = classifier(clean, INTENTS)
        raw_intent = result["labels"][0]
        confidence = round(float(result["scores"][0]), 4)
        intent_key = INTENT_KEY_MAP[raw_intent]
        model_type = "facebook/bart-large-mnli (zero-shot)"
    else:
        vec    = classifier["vectorizer"]
        clf    = classifier["model"]
        v      = vec.transform([clean])
        proba  = clf.predict_proba(v)[0]
        idx    = int(np.argmax(proba))
        intent_key = clf.classes_[idx]
        confidence = round(float(proba[idx]), 4)
        model_type = "TF-IDF + LogisticRegression (fallback)"

    is_low = confidence < CONFIDENCE_THRESHOLD
    final_response = LOW_CONFIDENCE_RESPONSE if is_low else INTENT_RESPONSES[intent_key]

    return ClassifyResponse(
        intent=intent_key,
        confidence=confidence,
        response=final_response,
        model_type=model_type,
        low_confidence=is_low,
    )
