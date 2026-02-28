"""
RLHF Preference Platform - FastAPI Backend
==========================================

Endpoints:
  POST /prompts/upload       - bulk upload JSON or CSV
  GET  /prompts              - list prompts
  POST /generate             - generate A/B response pair
  GET  /pairs                - list all response pairs
  POST /annotations          - submit annotation
  GET  /annotations          - list annotations (with filters)
  GET  /metrics              - full metrics summary
  GET  /export/rlhf          - download RLHF-format JSONL dataset
  GET  /export/csv           - download CSV of all annotations

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import io
import json
import csv
import time
import random
import logging
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from backend.models import (
    create_tables, get_db,
    Prompt, ResponsePair, Annotation,
    PromptCategory, PreferenceChoice,
)
from backend.metrics import compute_full_metrics
from backend.llm import generate_pair   # <- LLM inference module

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 
# App setup
# 

app = FastAPI(
    title="RLHF Preference Platform",
    version="1.0.0",
    description="End-to-end system for generating and annotating LLM preference pairs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    create_tables()
    logger.info("Database tables ready.")


# 
# Pydantic schemas
# 

class PromptIn(BaseModel):
    text: str
    category: str = PromptCategory.GENERAL
    source: Optional[str] = None


class GenerateRequest(BaseModel):
    prompt_id: int
    model_a: str = "default"
    model_b: str = "default"
    temp_a: float = 0.7
    temp_b: float = 1.2
    max_new_tokens: int = 256


class AnnotationIn(BaseModel):
    pair_id: int
    annotator_id: str
    preference: str           # "A", "B", "tie"
    reasoning: Optional[str] = None
    confidence: int = 3       # 1-5
    time_spent_sec: Optional[float] = None


# 
# Prompt endpoints
# 

@app.post("/prompts", status_code=201, tags=["Prompts"])
def create_prompt(payload: PromptIn, db: Session = Depends(get_db)):
    prompt = Prompt(**payload.dict())
    db.add(prompt)
    db.commit()
    db.refresh(prompt)
    return prompt.to_dict()


@app.get("/prompts", tags=["Prompts"])
def list_prompts(
    category: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    q = db.query(Prompt)
    if category:
        q = q.filter(Prompt.category == category)
    prompts = q.offset(skip).limit(limit).all()
    return [p.to_dict() for p in prompts]


@app.post("/prompts/upload", status_code=201, tags=["Prompts"])
async def upload_prompts(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload prompts via JSON array or CSV.

    JSON format:  [{"text": "...", "category": "math"}, ...]
    CSV format:   columns: text, category (category optional)
    """
    content = await file.read()
    fname = file.filename or ""
    created = []

    try:
        if fname.endswith(".json"):
            rows = json.loads(content)
        elif fname.endswith(".csv"):
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
            rows = df.to_dict("records")
        else:
            raise HTTPException(400, "Only .json or .csv files are supported.")
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")

    for row in rows:
        if "text" not in row:
            continue
        p = Prompt(
            text=str(row["text"]),
            category=row.get("category", PromptCategory.GENERAL),
            source=fname,
        )
        db.add(p)
        created.append(p)

    db.commit()
    return {"uploaded": len(created), "source": fname}


# 
# Response Pair Generation
# 

@app.post("/generate", tags=["Generation"])
def generate_response_pair(payload: GenerateRequest, db: Session = Depends(get_db)):
    """
    Call the LLM inference layer to produce two responses for a prompt.
    Saves the pair to the DB and returns it.
    """
    prompt = db.query(Prompt).filter(Prompt.id == payload.prompt_id).first()
    if not prompt:
        raise HTTPException(404, f"Prompt {payload.prompt_id} not found.")

    logger.info(f"Generating pair for prompt #{payload.prompt_id}")

    resp_a, resp_b = generate_pair(
        prompt_text=prompt.text,
        model_a=payload.model_a,
        model_b=payload.model_b,
        temp_a=payload.temp_a,
        temp_b=payload.temp_b,
        max_new_tokens=payload.max_new_tokens,
    )

    pair = ResponsePair(
        prompt_id=payload.prompt_id,
        response_a=resp_a,
        response_b=resp_b,
        model_a=payload.model_a,
        model_b=payload.model_b,
        temp_a=payload.temp_a,
        temp_b=payload.temp_b,
    )
    db.add(pair)
    db.commit()
    db.refresh(pair)
    return pair.to_dict()


@app.get("/pairs", tags=["Generation"])
def list_pairs(
    prompt_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    q = db.query(ResponsePair)
    if prompt_id:
        q = q.filter(ResponsePair.prompt_id == prompt_id)
    pairs = q.offset(skip).limit(limit).all()
    return [p.to_dict() for p in pairs]


@app.get("/pairs/{pair_id}", tags=["Generation"])
def get_pair(pair_id: int, db: Session = Depends(get_db)):
    pair = db.query(ResponsePair).filter(ResponsePair.id == pair_id).first()
    if not pair:
        raise HTTPException(404, "Pair not found.")
    return pair.to_dict()


# 
# Annotation endpoints
# 

@app.post("/annotations", status_code=201, tags=["Annotations"])
def submit_annotation(payload: AnnotationIn, db: Session = Depends(get_db)):
    if payload.preference not in ("A", "B", "tie"):
        raise HTTPException(400, "preference must be 'A', 'B', or 'tie'.")
    if not (1 <= payload.confidence <= 5):
        raise HTTPException(400, "confidence must be between 1 and 5.")

    pair = db.query(ResponsePair).filter(ResponsePair.id == payload.pair_id).first()
    if not pair:
        raise HTTPException(404, f"Pair {payload.pair_id} not found.")

    ann = Annotation(**payload.dict())
    db.add(ann)
    db.commit()
    db.refresh(ann)
    return ann.to_dict()


@app.get("/annotations", tags=["Annotations"])
def list_annotations(
    pair_id: Optional[int] = None,
    annotator_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    q = db.query(Annotation)
    if pair_id:
        q = q.filter(Annotation.pair_id == pair_id)
    if annotator_id:
        q = q.filter(Annotation.annotator_id == annotator_id)
    return [a.to_dict() for a in q.offset(skip).limit(limit).all()]


# 
# Metrics
# 

@app.get("/metrics", tags=["Metrics"])
def get_metrics(db: Session = Depends(get_db)):
    annotations = [a.to_dict() for a in db.query(Annotation).all()]
    return compute_full_metrics(annotations)


# 
# Export
# 

@app.get("/export/rlhf", tags=["Export"])
def export_rlhf(
    min_confidence: int = 3,
    db: Session = Depends(get_db),
):
    """
    Export RLHF-format JSONL:
      {"prompt": "...", "chosen": "...", "rejected": "..."}

    Majority-vote aggregates multiple annotations per pair.
    Ties are excluded.
    """
    annotations = (
        db.query(Annotation)
        .filter(Annotation.confidence >= min_confidence)
        .all()
    )

    # Majority vote per pair
    from collections import Counter, defaultdict
    votes: dict = defaultdict(list)
    for a in annotations:
        votes[a.pair_id].append(a.preference)

    lines = []
    for pair_id, prefs in votes.items():
        majority = Counter(prefs).most_common(1)[0][0]
        if majority == "tie":
            continue  # skip ties
        pair = db.query(ResponsePair).filter(ResponsePair.id == pair_id).first()
        if not pair:
            continue
        chosen   = pair.response_a if majority == "A" else pair.response_b
        rejected = pair.response_b if majority == "A" else pair.response_a
        lines.append(json.dumps({
            "prompt":   pair.prompt.text,
            "chosen":   chosen,
            "rejected": rejected,
        }))

    content = "\n".join(lines)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=rlhf_dataset.jsonl"},
    )


@app.get("/export/csv", tags=["Export"])
def export_csv(db: Session = Depends(get_db)):
    """Export all annotations as CSV for analysis."""
    annotations = db.query(Annotation).all()
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["id", "pair_id", "annotator_id", "preference",
                    "confidence", "reasoning", "created_at"]
    )
    writer.writeheader()
    for a in annotations:
        writer.writerow(a.to_dict())
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=annotations.csv"},
    )


# 
# Health check
# 

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "rlhf-platform"}
