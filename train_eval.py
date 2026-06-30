import re
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
)

DATASET = [
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


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def run_experiments():
    texts  = [preprocess(t) for t, _ in DATASET]
    labels = [l for _, l in DATASET]

    print(f"\n[INFO] Dataset: {len(texts)} samples | {len(set(labels))} intent classes")
    print(f"[INFO] Classes: {sorted(set(labels))}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    mlflow.set_experiment("convai-intent-classification")

    configs = [
        {"name": "baseline",    "C": 0.5, "ngram": (1, 1), "features": 500},
        {"name": "ngram-tuned", "C": 1.0, "ngram": (1, 2), "features": 1000},
        {"name": "best-config", "C": 2.0, "ngram": (1, 2), "features": 2000},
    ]

    best_f1, best_cfg = 0, None

    for cfg in configs:
        with mlflow.start_run(run_name=cfg["name"]):
            vec  = TfidfVectorizer(ngram_range=cfg["ngram"], max_features=cfg["features"])
            X_tr = vec.fit_transform(X_train)
            X_te = vec.transform(X_test)

            clf = LogisticRegression(max_iter=500, C=cfg["C"], random_state=42)
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            cv   = cross_val_score(clf, vec.transform(texts), labels, cv=5, scoring="f1_weighted")

            mlflow.log_params({
                "model":        "LogisticRegression",
                "vectorizer":   "TF-IDF",
                "C":            cfg["C"],
                "ngram_range":  str(cfg["ngram"]),
                "max_features": cfg["features"],
                "train_size":   len(X_train),
                "test_size":    len(X_test),
                "num_classes":  len(set(labels)),
            })
            mlflow.log_metrics({
                "accuracy":   round(acc,  4),
                "precision":  round(prec, 4),
                "recall":     round(rec,  4),
                "f1_score":   round(f1,   4),
                "cv_f1_mean": round(float(cv.mean()), 4),
                "cv_f1_std":  round(float(cv.std()),  4),
            })

            print(f"[Run: {cfg['name']}]")
            print(f"  Accuracy  : {acc:.4f}")
            print(f"  Precision : {prec:.4f}")
            print(f"  Recall    : {rec:.4f}")
            print(f"  F1 Score  : {f1:.4f}")
            print(f"  CV F1     : {cv.mean():.4f} ± {cv.std():.4f}\n")

            if f1 > best_f1:
                best_f1, best_cfg = f1, cfg["name"]

    print(f"{'='*55}")
    print(f"  Best Config   : {best_cfg}")
    print(f"  Best F1 Score : {best_f1:.4f}")
    print(f"{'='*55}")
    print("\n[INFO] All runs logged to MLflow.")
    print("[INFO] Run `mlflow ui` to open dashboard → http://localhost:5000\n")


if __name__ == "__main__":
    run_experiments()
