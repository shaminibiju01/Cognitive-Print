# Cognitive-Print
NLP-based mental health pattern detection web app using RoBERTa and D3.js
Built solo at Divergent Teams Boston 2026 — Third Place, Mental Health and Sustainability Track



## What It Does

CognitivePrint analyzes free-form journal entries to:

- Detect emotions using a fine-tuned DistilRoBERTa model (7 emotion classes: joy, sadness, anger, fear, disgust, surprise, neutral)
- Identify cognitive distortions using patterns of irrational thinking linked to anxiety and depression, including catastrophizing, black-and-white thinking, mind reading, and self blame
- Track mental health over time using a weighted health score algorithm across rolling 7 entry windows
- Visualize emotional patterns as an interactive knowledge graph built with D3.js, mapping how mood and distortions evolve across entries

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask, Flask-CORS |
| NLP | spaCy (en_core_web_sm), HuggingFace Transformers |
| Emotion Model | j-hartmann/emotion-english-distilroberta-base |
| Graph Engine | NetworkX |
| Frontend | HTML, CSS, JavaScript, D3.js |

---

## How the Health Score Works

The health score (0–1) is computed over the last 7 journal entries using:
```
health = 1.0 - (0.6 × distortion_score + 0.4 × sentiment_score)
```

- distortion_score: average number of cognitive distortions detected per entry, normalized to [0,1]
- sentiment_score: proportion of entries with negative emotions (sadness, fear, anger, disgust)

A score near 1.0 indicates low distortion and positive emotional patterns. A declining score over time flags potential mental health concerns worth attention.

---

## Cognitive Distortions Detected

| Distortion | Example Triggers |
|---|---|
| Catastrophizing | "never", "worst", "hopeless", "ruined" |
| Black-and-White Thinking | "everyone", "nothing", "perfect", "failure" |
| Mind Reading | "they think", "they hate", "nobody cares" |
| Self-Blame | "my fault", "I ruined", "I can't do anything" |

---

## Running Locally

```bash
# Install dependencies
pip install flask flask-cors spacy transformers networkx
python -m spacy download en_core_web_sm

# Run the backend
python app.py

# Open frontend
open index.html
```

API runs on `http://localhost:5000`


## API Endpoints

### `POST /analyze`
Analyzes a journal entry and returns emotion, distortions, health score, and updated graph.

**Request:**
```json
{
  "text": "I always mess everything up. Nobody cares about me.",
  "date": "2026-03-30"
}
```

**Response:**
```json
{
  "entry": {
    "emotion": "sadness",
    "emotion_score": 0.91,
    "distortions": ["catastrophizing", "black_and_white", "mind_reading"],
    "keywords": ["mess", "care"]
  },
  "health_score": 0.62,
  "graph": { "nodes": [...], "edges": [...] },
  "total_entries": 3
}
```



## Why I Built This

Cognitive distortions are a core concept in Cognitive Behavioral Therapy (CBT). They're patterns of thinking that reinforce anxiety and depression. Most mental health apps track mood with emoji sliders. CognitivePrint goes deeper, using NLP to surface how someone is thinking. 68% of psychiatrists use journal entries and their language to flag any early signs of mental health issues, and this website app allows for this process to become less manual, thus improving the efficiency of someone getting the help they need. Visualizing that connection is the first step toward recognizing it.

---

## Future Directions

- Longitudinal trend analysis with regression modeling
- Personalized distortion pattern clustering across users
- Integration with validated clinical scales (PHQ-9, GAD-7)
- Privacy-preserving local inference (on-device model)

---

Built by Shamini Biju | Shrewsbury, MA
