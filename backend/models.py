"""
Database models for the RLHF Preference Platform.
Uses SQLAlchemy with SQLite (swap URL for Postgres in production).
"""

from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, 
    ForeignKey, create_engine, Enum
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from datetime import datetime
import enum

Base = declarative_base()


# 
# Enums
# 

class PreferenceChoice(str, enum.Enum):
    A_BETTER   = "A"
    B_BETTER   = "B"
    TIE        = "tie"


class PromptCategory(str, enum.Enum):
    MATH       = "math"
    CODING     = "coding"
    REASONING  = "reasoning"
    GENERAL    = "general"


# 
# Tables
# 

class Prompt(Base):
    __tablename__ = "prompts"

    id          = Column(Integer, primary_key=True, index=True)
    text        = Column(Text, nullable=False)
    category    = Column(String(50), default=PromptCategory.GENERAL)
    source      = Column(String(100))          # e.g. filename
    created_at  = Column(DateTime, default=datetime.utcnow)

    pairs       = relationship("ResponsePair", back_populates="prompt", cascade="all, delete")

    def to_dict(self):
        return {
            "id":         self.id,
            "text":       self.text,
            "category":   self.category,
            "source":     self.source,
            "created_at": str(self.created_at),
        }


class ResponsePair(Base):
    __tablename__ = "response_pairs"

    id              = Column(Integer, primary_key=True, index=True)
    prompt_id       = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    response_a      = Column(Text, nullable=False)
    response_b      = Column(Text, nullable=False)
    model_a         = Column(String(100), default="model_a")
    model_b         = Column(String(100), default="model_b")
    temp_a          = Column(Float, default=0.7)
    temp_b          = Column(Float, default=1.2)
    created_at      = Column(DateTime, default=datetime.utcnow)

    prompt          = relationship("Prompt", back_populates="pairs")
    annotations     = relationship("Annotation", back_populates="pair", cascade="all, delete")

    def to_dict(self):
        return {
            "id":         self.id,
            "prompt_id":  self.prompt_id,
            "prompt":     self.prompt.text if self.prompt else "",
            "response_a": self.response_a,
            "response_b": self.response_b,
            "model_a":    self.model_a,
            "model_b":    self.model_b,
        }


class Annotation(Base):
    __tablename__ = "annotations"

    id              = Column(Integer, primary_key=True, index=True)
    pair_id         = Column(Integer, ForeignKey("response_pairs.id"), nullable=False)
    annotator_id    = Column(String(100), nullable=False)   # username / session token
    preference      = Column(String(10), nullable=False)    # "A", "B", "tie"
    reasoning       = Column(Text)
    confidence      = Column(Integer, default=3)            # 1-5 Likert scale
    time_spent_sec  = Column(Float)                         # annotation duration
    created_at      = Column(DateTime, default=datetime.utcnow)

    pair            = relationship("ResponsePair", back_populates="annotations")

    def to_dict(self):
        return {
            "id":           self.id,
            "pair_id":      self.pair_id,
            "annotator_id": self.annotator_id,
            "preference":   self.preference,
            "reasoning":    self.reasoning,
            "confidence":   self.confidence,
            "created_at":   str(self.created_at),
        }


# 
# DB helpers
# 

DATABASE_URL = "sqlite:///./rlhf_platform.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},   # SQLite only
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency - yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
