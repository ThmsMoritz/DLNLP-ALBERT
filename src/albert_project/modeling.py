from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import (
    AlbertConfig,
    AlbertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
)


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    model_label: str
    notes: str = ""


def count_parameters(model: Any, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def count_embedding_parameters(model: Any) -> int:
    """Count parameters that belong to embedding modules by name.

    This is intentionally name-based because BERT and ALBERT expose embeddings
    slightly differently.
    """
    total = 0
    for name, param in model.named_parameters():
        if "embeddings" in name:
            total += param.numel()
    return total


def _albert_config_from_project_config(config: dict[str, Any], num_labels: int) -> AlbertConfig:
    model_cfg = dict(config.get("model_config", {}))
    sharing_strategy = config.get("sharing_strategy", "full_sharing")

    if sharing_strategy in {"shared_attention", "shared_ffn"}:
        raise NotImplementedError(
            "TODO: Hugging Face AlbertForSequenceClassification supports grouped/full "
            "sharing via num_hidden_groups, but not attention-only or FFN-only sharing "
            "out of the box. Implement a custom AlbertLayer variant if these should be "
            "trained exactly as in the paper."
        )

    num_hidden_layers = int(model_cfg.get("num_hidden_layers", 12))
    if sharing_strategy == "full_sharing":
        model_cfg.setdefault("num_hidden_groups", 1)
    elif sharing_strategy == "no_sharing":
        model_cfg.setdefault("num_hidden_groups", num_hidden_layers)
    else:
        model_cfg.setdefault("num_hidden_groups", 1)

    defaults = {
        "vocab_size": 30000,
        "embedding_size": 128,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_hidden_groups": model_cfg.get("num_hidden_groups", 1),
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "classifier_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "num_labels": num_labels,
    }
    defaults.update(model_cfg)
    defaults["num_labels"] = num_labels
    return AlbertConfig(**defaults)


def _bert_config_from_project_config(config: dict[str, Any], num_labels: int) -> BertConfig:
    model_cfg = dict(config.get("model_config", {}))
    defaults = {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "num_labels": num_labels,
    }
    defaults.update(model_cfg)
    defaults["num_labels"] = num_labels
    return BertConfig(**defaults)


def load_model_and_tokenizer(config: dict[str, Any], num_labels: int) -> LoadedModel:
    """Load a pretrained model or instantiate a controlled architecture.

    Supported modes:
    - model_source: pretrained -> use AutoModelForSequenceClassification.from_pretrained
    - model_source: from_config -> instantiate BERT/ALBERT from config with random weights

    The ablation experiments use from_config by default because exact pretrained
    checkpoints for every custom configuration are usually not available.
    """
    model_source = config.get("model_source", "pretrained")
    model_name = config.get("model_name")
    model_label = config.get("model_label", model_name or config.get("architecture", "model"))
    tokenizer_name = config.get("tokenizer_name", model_name or "albert-base-v2")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if model_source == "pretrained":
        if not model_name:
            raise ValueError("model_name is required when model_source='pretrained'.")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        return LoadedModel(model=model, tokenizer=tokenizer, model_label=model_label)

    if model_source == "from_config":
        architecture = config.get("architecture", "albert").lower()
        if architecture == "albert":
            hf_config = _albert_config_from_project_config(config, num_labels)
            model = AlbertForSequenceClassification(hf_config)
        elif architecture == "bert":
            hf_config = _bert_config_from_project_config(config, num_labels)
            model = BertForSequenceClassification(hf_config)
        else:
            raise ValueError(f"Unsupported architecture '{architecture}'.")
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            model_label=model_label,
            notes="Randomly initialized controlled architecture; not a pretrained checkpoint.",
        )

    raise ValueError("model_source must be either 'pretrained' or 'from_config'.")
