"""
LLM Inference Layer
===================

Supports four backends (configured via .env):
  1. MOCK       - deterministic fake responses (default, no GPU needed)
  2. GROQ       - Groq Cloud API (fast inference, free tier available)
  3. OPENAI     - any OpenAI-compatible API (OpenAI, Together, etc.)
  4. HF_LOCAL   - local HuggingFace model via transformers pipeline

Set LLM_BACKEND in your .env file.
"""

import os
import logging
import random
from typing import Tuple

logger = logging.getLogger(__name__)

BACKEND = os.getenv("LLM_BACKEND", "mock").lower()


# 
# 1. Mock Backend (zero dependencies)
# 

_MOCK_RESPONSES = [
    "The answer involves carefully considering the constraints of the problem. "
    "First, we decompose the question into sub-problems, solve each independently, "
    "then combine the results to form the final answer.",

    "To solve this, we apply first principles. The key insight is that the "
    "underlying structure allows for a systematic approach: enumerate cases, "
    "verify each one, then synthesize the conclusion.",

    "Let me walk through this step by step. \n\n"
    "Step 1: Identify the core objective.\n"
    "Step 2: Gather necessary context.\n"
    "Step 3: Apply the relevant algorithm or formula.\n"
    "Step 4: Verify the solution against edge cases.",

    "This is a classic problem in the domain. The standard approach yields "
    "an O(n log n) solution by leveraging a divide-and-conquer strategy.",

    "After analyzing the prompt carefully, the most accurate response draws "
    "on domain knowledge in {domain}. The conclusion follows logically from "
    "the stated premises.",
]


def _mock_response(prompt_text: str, temperature: float) -> str:
    """Simulate variability with temperature."""
    random.seed(hash(prompt_text + str(round(temperature, 1))) % (2**32))
    base = random.choice(_MOCK_RESPONSES).replace("{domain}", "this field")
    if temperature > 1.0:
        # Higher temp -> noisier / less structured
        return base + "\n\nNote: There may be alternative interpretations worth exploring."
    return base


# 
# 2. Groq Backend
# 

# Available Groq models (as of 2025):
GROQ_MODELS = {
    "default":          "llama-3.3-70b-versatile",   # best quality
    "fast":             "llama-3.1-8b-instant",       # fastest / lowest latency
    "mixtral":          "mixtral-8x7b-32768",         # good for reasoning
    "gemma":            "gemma2-9b-it",               # Google Gemma 2
}


def _groq_response(
    prompt_text: str,
    temperature: float,
    model: str,
    max_tokens: int,
) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("pip install groq to use GROQ backend.")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in your .env file.")

    client = Groq(api_key=api_key)

    # Resolve model alias or use the value directly
    actual_model = GROQ_MODELS.get(model, model)
    # Allow override via env var
    actual_model = os.getenv("GROQ_MODEL", actual_model)

    resp = client.chat.completions.create(
        model=actual_model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()




def _openai_response(
    prompt_text: str,
    temperature: float,
    model: str,
    max_tokens: int,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai to use OPENAI backend.")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    actual_model = os.getenv("OPENAI_MODEL", model) if model == "default" else model

    resp = client.chat.completions.create(
        model=actual_model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# 
# 3. Local HuggingFace Backend
# 

_hf_pipeline = None  # lazy load


def _get_hf_pipeline():
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline
        model_id = os.getenv("HF_MODEL_ID", "gpt2")
        logger.info(f"Loading HuggingFace model: {model_id}")
        _hf_pipeline = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
        )
    return _hf_pipeline


def _hf_response(
    prompt_text: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    pipe = _get_hf_pipeline()
    outputs = pipe(
        prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-7),
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    # Strip the original prompt from the output
    full_text = outputs[0]["generated_text"]
    return full_text[len(prompt_text):].strip()


# 
# Public interface
# 

def generate_single(
    prompt_text: str,
    model: str = "default",
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> str:
    """Generate a single response for a prompt."""
    if BACKEND == "mock":
        return _mock_response(prompt_text, temperature)

    elif BACKEND == "groq":
        return _groq_response(prompt_text, temperature, model, max_new_tokens)

    elif BACKEND == "openai":
        return _openai_response(prompt_text, temperature, model, max_new_tokens)

    elif BACKEND == "hf_local":
        return _hf_response(prompt_text, temperature, max_new_tokens)

    else:
        raise ValueError(f"Unknown LLM_BACKEND: {BACKEND!r}. Choose mock/groq/openai/hf_local.")


def generate_pair(
    prompt_text: str,
    model_a: str = "default",
    model_b: str = "default",
    temp_a: float = 0.7,
    temp_b: float = 1.2,
    max_new_tokens: int = 256,
) -> Tuple[str, str]:
    """
    Generate two independent responses for the same prompt.
    Response A uses a lower temperature (more focused).
    Response B uses a higher temperature (more creative/noisy).
    """
    resp_a = generate_single(prompt_text, model_a, temp_a, max_new_tokens)
    resp_b = generate_single(prompt_text, model_b, temp_b, max_new_tokens)
    return resp_a, resp_b
