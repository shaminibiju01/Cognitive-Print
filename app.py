from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from transformers import pipeline
import networkx as nx
from datetime import datetime

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")
sentiment = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

G = nx.Graph()
entries = []

DISTORTIONS = {
    "catastrophizing": ["never", "always", "worst", "terrible", "ruined", "hopeless", "disaster"],
    "black_and_white": ["everyone", "nobody", "everything", "nothing", "perfect", "failure", "completely"],
    "mind_reading": ["they think", "she thinks", "he thinks", "they hate", "nobody cares"],
    "self_blame": ["my fault", "i ruined", "because of me", "i always mess", "i can't do anything"],
}

def detect_distortions(text):
    found = []
    lower = text.lower()
    for distortion, phrases in DISTORTIONS.items():
        if any(p in lower for p in phrases):
            found.append(distortion)
    return found

def compute_health_score(entry_list):
    if not entry_list:
        return 1.0
    recent = entry_list[-7:]
    distortion_counts = [len(e["distortions"]) for e in recent]
    neg_sentiments = [1 if e["emotion"] in ["sadness", "fear", "anger", "disgust"] else 0 for e in recent]
    distortion_score = min(sum(distortion_counts) / (len(recent) * 4), 1.0)
    sentiment_score = sum(neg_sentiments) / len(recent)
    health = 1.0 - (0.6 * distortion_score + 0.4 * sentiment_score)
    return round(health, 3)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    date = data.get("date", datetime.now().strftime("%Y-%m-%d"))

    doc = nlp(text)
    emotion_result = sentiment(text[:512])[0]
    distortions = detect_distortions(text)
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB"] and not token.is_stop]

    entry = {
        "id": len(entries),
        "date": date,
        "text": text,
        "emotion": emotion_result["label"].lower(),
        "emotion_score": round(emotion_result["score"], 3),
        "distortions": distortions,
        "keywords": keywords[:10],
    }
    entries.append(entry)

    G.add_node(entry["id"], **entry)
    if len(entries) > 1:
        G.add_edge(entry["id"] - 1, entry["id"], weight=1)

    health = compute_health_score(entries)
    nodes = [{"id": n, **G.nodes[n]} for n in G.nodes]
    edges = [{"source": u, "target": v} for u, v in G.edges]

    return jsonify({
        "entry": entry,
        "health_score": health,
        "graph": {"nodes": nodes, "edges": edges},
        "total_entries": len(entries)
    })

@app.route("/entries", methods=["GET"])
def get_entries():
    return jsonify({"entries": entries, "health_score": compute_health_score(entries)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    