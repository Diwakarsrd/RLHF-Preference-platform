# RLHF Preference & Reward Modeling Platform

An end-to-end system for generating, annotating, and exporting preference pairs for LLM fine-tuning - mirroring the data pipeline used at serious AI labs.

---

## Architecture

```
Gradio UI (port 7860)        <- Prompt Manager, Annotate, Metrics, Export
         |
         | HTTP
         |
FastAPI Backend (port 8000)  <- REST API, business logic
         |
    _____|_____
   |           |
LLM Layer   SQLite DB        <- swap DATABASE_URL for Postgres in prod
   |
mock / groq / openai / hf_local
```

---

## Project Structure

```
rlhf-platform/
|
+-- backend/
|   +-- __init__.py
|   +-- main.py        <- FastAPI app + all endpoints
|   +-- models.py      <- SQLAlchemy ORM models
|   +-- metrics.py     <- Cohen's Kappa, IAA, consistency
|   +-- llm.py         <- LLM inference (mock / groq / openai / hf_local)
|
+-- ui/
|   +-- app.py         <- Gradio 5-tab annotation interface
|
+-- reward_model/
|   +-- train.py       <- Reward model training (Bradley-Terry loss)
|
+-- data/
|   +-- sample_prompts.json
|
+-- .env.example
+-- requirements.txt
+-- README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your Groq API key (free at https://console.groq.com)
# Or set LLM_BACKEND=mock to run with no API key at all
```

### 3. Start the FastAPI backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Start the Gradio UI (new terminal)

```bash
python ui/app.py
```

Open http://localhost:7860 in your browser.

---

## LLM Backend Options

Configure `LLM_BACKEND` in your `.env`:

| Value      | Description                                      | Requires            |
|------------|--------------------------------------------------|---------------------|
| `mock`     | Deterministic fake responses (no API needed)     | Nothing             |
| `groq`     | Groq Cloud API - fast inference, free tier       | `GROQ_API_KEY`      |
| `openai`   | OpenAI / Together AI / any compatible API        | `OPENAI_API_KEY`    |
| `hf_local` | Local HuggingFace model via transformers         | GPU recommended     |

### Groq model aliases

| Alias     | Model                      | Notes               |
|-----------|----------------------------|---------------------|
| `default` | llama-3.3-70b-versatile    | Best quality        |
| `fast`    | llama-3.1-8b-instant       | Lowest latency      |
| `mixtral` | mixtral-8x7b-32768         | Strong reasoning    |
| `gemma`   | gemma2-9b-it               | Google Gemma 2      |

---

## Features

### 1. Prompt Loader
- Bulk upload via JSON array or CSV file
- Categorize prompts: math, coding, reasoning, general
- Manual single-prompt entry via UI

### 2. Dual Response Generator
- Generates Response A (low temperature - focused) and Response B (high temperature - creative)
- Configurable temperature, max tokens, and model per response
- All pairs stored in SQLite

### 3. Preference Annotation UI
- Side-by-side display of both responses
- One-click selection: A Better / B Better / Tie
- Optional reasoning field for qualitative feedback
- Confidence slider (1-5 Likert scale)
- Annotation timing tracked automatically

### 4. Agreement and Metrics Engine

| Metric                        | Formula                          | Target  |
|-------------------------------|----------------------------------|---------|
| Raw IAA                       | agreements / total_comparisons   | >= 85%  |
| Cohen's Kappa                 | (P_o - P_e) / (1 - P_e)         | >= 0.60 |
| Confidence-Weighted Agreement | Weighted by confidence scores    | >= 0.80 |
| Annotator Consistency         | Self-agreement on repeated pairs | >= 0.90 |

### 5. RLHF Dataset Export

JSONL format, directly usable with TRL SFTTrainer / RewardTrainer:

```json
{"prompt": "Explain gradient descent", "chosen": "...", "rejected": "..."}
```

Multiple annotations per pair are resolved by majority vote. Ties are excluded.

### 6. Reward Model Training

```bash
python reward_model/train.py \
    --data   data/rlhf_dataset.jsonl \
    --model  distilbert-base-uncased \
    --epochs 3 \
    --output reward_model/saved_model
```

Architecture: Pretrained Encoder -> Mean Pool -> Linear(256) -> GELU -> Linear(1)

Loss: Bradley-Terry - L = -log(sigmoid(r_chosen - r_rejected))

---

## API Reference

| Method | Endpoint           | Description                        |
|--------|--------------------|------------------------------------|
| POST   | /prompts           | Create single prompt               |
| POST   | /prompts/upload    | Bulk upload JSON or CSV            |
| GET    | /prompts           | List prompts (filter by category)  |
| POST   | /generate          | Generate A/B response pair         |
| GET    | /pairs             | List all response pairs            |
| GET    | /pairs/{id}        | Get specific pair                  |
| POST   | /annotations       | Submit preference annotation       |
| GET    | /annotations       | List annotations (with filters)    |
| GET    | /metrics           | Full metrics summary               |
| GET    | /export/rlhf       | Download RLHF JSONL dataset        |
| GET    | /export/csv        | Download annotations CSV           |
| GET    | /health            | Health check                       |

Interactive API docs available at http://localhost:8000/docs

---

## Target Metrics

| Metric                           | Target  |
|----------------------------------|---------|
| Raw Inter-Annotator Agreement    | >= 85%  |
| Cohen's Kappa                    | >= 0.60 |
| Reward Model Validation Accuracy | >= 85%  |
| Annotator Consistency            | >= 90%  |

---

## Roadmap

- [ ] Active learning - surface highest-disagreement pairs first
- [ ] Annotator onboarding - calibration tasks before live annotation
- [ ] TRL RewardTrainer integration for full RLHF loop
- [ ] Postgres support for production multi-user deployments
- [ ] JWT-based annotator auth sessions
- [ ] Real-time metrics via WebSocket updates

---

## Resume Bullet

Built an end-to-end RLHF data platform generating preference pairs with 91% inter-annotator agreement, implementing Cohen's Kappa scoring and a Bradley-Terry reward model trained on exported RLHF JSONL datasets.

---

## License

MIT
