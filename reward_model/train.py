"""
Reward Model Training
======================

Trains a simple reward model on RLHF preference data exported
from the platform (JSONL format).

Architecture:
  Pretrained transformer encoder -> mean pooling -> linear head -> scalar reward

The model learns: reward(chosen) > reward(rejected)

Usage:
    python reward_model/train.py \
        --data   data/rlhf_dataset.jsonl \
        --model  distilbert-base-uncased \
        --epochs 3 \
        --output reward_model/saved_model
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# 
# Dataset
# 

class PreferencePairDataset(Dataset):
    """
    Each sample: (prompt + chosen, prompt + rejected)
    Label: chosen > rejected -> model must assign higher reward to chosen.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples: List[Dict] = []

        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} preference pairs from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def _encode(self, prompt: str, response: str) -> Dict:
        text = f"Prompt: {prompt}\n\nResponse: {response}"
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        chosen   = self._encode(sample["prompt"], sample["chosen"])
        rejected = self._encode(sample["prompt"], sample["rejected"])
        return {
            "chosen_input_ids":      chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(0),
            "rejected_input_ids":      rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
        }


# 
# Model
# 

class RewardModel(nn.Module):
    """
    Encoder + mean pooling + scalar reward head.
    Outputs a single scalar per input sequence.
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embed     = (token_embeddings * mask_expanded).sum(1)
        sum_mask      = mask_expanded.sum(1).clamp(min=1e-9)
        return sum_embed / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = self.mean_pool(outputs.last_hidden_state, attention_mask)
        reward  = self.reward_head(pooled).squeeze(-1)   # shape: (batch,)
        return reward


# 
# Loss
# 

def preference_loss(
    reward_chosen:   torch.Tensor,  # (batch,)
    reward_rejected: torch.Tensor,  # (batch,)
) -> torch.Tensor:
    """
    Bradley-Terry preference loss:
      L = -log(sigma(r_chosen - r_rejected))

    Minimizing this encourages r_chosen > r_rejected.
    """
    return -torch.nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()


# 
# Training loop
# 

def evaluate(model: RewardModel, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    total_loss   = 0.0
    total_correct = 0
    total         = 0

    with torch.no_grad():
        for batch in loader:
            r_chosen = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            r_rejected = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )
            loss = preference_loss(r_chosen, r_rejected)
            total_loss += loss.item()
            total_correct += (r_chosen > r_rejected).sum().item()
            total += len(r_chosen)

    model.train()
    return {
        "loss":     round(total_loss / max(len(loader), 1), 4),
        "accuracy": round(total_correct / max(total, 1), 4),
    }


def train(
    data_path:  str,
    model_name: str,
    output_dir: str,
    epochs:     int = 3,
    batch_size: int = 8,
    lr:         float = 2e-5,
    max_length: int = 512,
    val_split:  float = 0.1,
    warmup_steps: int = 50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Base model: {model_name}")

    #  Tokenizer & Dataset 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset   = PreferencePairDataset(data_path, tokenizer, max_length)

    val_size   = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info(f"Train: {train_size} | Val: {val_size}")

    #  Model 
    model = RewardModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    #  Training 
    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            r_chosen = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            r_rejected = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )

            loss = preference_loss(r_chosen, r_rejected)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss    += loss.item()
            epoch_correct += (r_chosen > r_rejected).sum().item()
            epoch_total   += len(r_chosen)

            if (step + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        train_acc = round(epoch_correct / max(epoch_total, 1), 4)
        val_metrics = evaluate(model, val_loader, device)

        history.append({
            "epoch":     epoch,
            "train_loss": round(epoch_loss / len(train_loader), 4),
            "train_acc":  train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        logger.info(
            f"Epoch {epoch} | "
            f"Train Acc: {train_acc:.1%} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.1%}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            _save(model, tokenizer, output_dir)
            logger.info(f"   New best val accuracy: {best_val_acc:.1%} - model saved.")

    logger.info(f"\n Training complete. Best Val Accuracy: {best_val_acc:.1%}")
    _save_history(history, output_dir)
    return history


def _save(model: RewardModel, tokenizer, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "reward_model.pt"))
    tokenizer.save_pretrained(output_dir)


def _save_history(history: List[Dict], output_dir: str):
    path = os.path.join(output_dir, "training_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {path}")


# 
# Inference helper
# 

def load_reward_model(model_dir: str, base_model: str) -> Tuple:
    """Load a saved reward model for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = RewardModel(base_model)
    state     = torch.load(
        os.path.join(model_dir, "reward_model.pt"),
        map_location="cpu",
    )
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def score_response(
    model: RewardModel,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int = 512,
) -> float:
    """Return the reward score for a single prompt-response pair."""
    text = f"Prompt: {prompt}\n\nResponse: {response}"
    enc  = tokenizer(text, max_length=max_length, truncation=True,
                     padding="max_length", return_tensors="pt")
    with torch.no_grad():
        reward = model(enc["input_ids"], enc["attention_mask"])
    return reward.item()


# 
# CLI entry point
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RLHF Reward Model")
    parser.add_argument("--data",       default="data/rlhf_dataset.jsonl")
    parser.add_argument("--model",      default="distilbert-base-uncased",
                        help="HuggingFace model name for the encoder backbone.")
    parser.add_argument("--output",     default="reward_model/saved_model")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_length", type=int,   default=512)
    parser.add_argument("--val_split",  type=float, default=0.1)
    args = parser.parse_args()

    history = train(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        val_split=args.val_split,
    )
