"""Portable, dependency-free validation of MBT pretraining recipes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re

from .config import ModelConfig


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_name(value: str) -> str:
    require(isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9._-]{1,120}", value)),
            "Run names/IDs must contain 1–120 ASCII letters, digits, '.', '_' or '-'")
    return value


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    train_steps: int
    global_batch_size: int
    sequence_length: int = 4096
    micro_batch_size: int = 16
    min_lr_ratio: float = 0.1
    warmup_fraction: float = 0.01
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    clip_grad: float = 1.0
    seed: int = 1234
    eval_global_batch_size: int = 128
    eval_micro_batch_size: int = 2
    eval_steps: int = 95
    checkpoint_fraction: float = 0.1
    keep_last_checkpoints: int = 1

    def __post_init__(self):
        for key in ("train_steps", "global_batch_size", "sequence_length", "micro_batch_size",
                    "eval_global_batch_size", "eval_micro_batch_size", "eval_steps", "keep_last_checkpoints"):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, f"{key} must be a positive integer")
        require(type(self.seed) is int and self.seed >= 0, "seed must be a nonnegative integer")
        for key in ("learning_rate", "adam_epsilon", "clip_grad"):
            require(math.isfinite(getattr(self, key)) and getattr(self, key) > 0, f"Invalid {key}")
        for key in ("min_lr_ratio", "warmup_fraction", "checkpoint_fraction"):
            require(math.isfinite(getattr(self, key)) and 0 < getattr(self, key) <= 1, f"Invalid {key}")
        require(0 <= self.weight_decay and math.isfinite(self.weight_decay), "Invalid weight decay")
        require(0 <= self.adam_beta1 < 1 and 0 <= self.adam_beta2 < 1, "Invalid Adam betas")

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        require(not unknown, f"Unknown training keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    @property
    def training_tokens(self):
        return self.train_steps * self.global_batch_size * self.sequence_length

    @property
    def checkpoint_interval(self):
        return max(1, math.ceil(self.train_steps * self.checkpoint_fraction))

    def accumulation_steps(self, world_size):
        require(type(world_size) is int and world_size > 0, "world_size must be positive")
        for prefix in ("", "eval_"):
            batch = getattr(self, prefix + "global_batch_size")
            micro = getattr(self, prefix + "micro_batch_size")
            require(batch % (world_size * micro) == 0,
                    f"{prefix or 'train_'}GBS must be divisible by world_size × micro_batch_size")
        return self.global_batch_size // (world_size * self.micro_batch_size)


@dataclass(frozen=True)
class Recipe:
    name: str
    model: ModelConfig
    training: TrainingConfig
    description: str = ""

    @classmethod
    def from_dict(cls, data):
        require(set(data) <= {"name", "model", "training", "description"}, "Unknown recipe keys")
        return cls(safe_name(data["name"]), ModelConfig.from_dict(data["model"]),
                   TrainingConfig.from_dict(data["training"]), data.get("description", ""))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return {"name": self.name, "model": self.model.to_dict(),
                "training": self.training.to_dict(), "description": self.description}


@dataclass(frozen=True)
class DataConfig:
    """Ordered weighted Megatron indexed datasets; no automatic downloads."""
    train: tuple
    validation: tuple
    tokenizer_model: str

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        require(set(data) == {"train", "validation", "tokenizer_model"},
                "Data config requires train, validation, tokenizer_model only")
        groups = []
        for split in ("train", "validation"):
            require(isinstance(data[split], list) and data[split], f"Empty {split} blend")
            entries = []
            for item in data[split]:
                require(set(item) == {"weight", "prefix"}, "Blend entry requires weight and prefix")
                weight = item["weight"]
                require(type(weight) in (int, float) and math.isfinite(weight) and weight > 0,
                        "Dataset weights must be finite and positive")
                prefix = Path(item["prefix"])
                require(prefix.is_absolute(), "Dataset prefixes must be absolute paths")
                require(prefix.suffix not in (".bin", ".idx"), "Pass indexed dataset prefix, without .bin/.idx")
                entries.append((float(weight), str(prefix)))
            groups.append(tuple(entries))
        require(Path(data["tokenizer_model"]).is_absolute(), "Tokenizer path must be absolute")
        return cls(*groups, data["tokenizer_model"])

    def validate_files(self):
        for _, prefix in (*self.train, *self.validation):
            for suffix in (".bin", ".idx"):
                path = Path(prefix + suffix)
                require(path.is_file() and path.stat().st_size > 0, f"Missing/empty indexed data: {path}")
        require(Path(self.tokenizer_model).is_file(), "Tokenizer model is missing")

    def backend_args(self):
        args = []
        for flag, blend in (("--train-data-path", self.train), ("--valid-data-path", self.validation)):
            args.append(flag)
            for weight, prefix in blend:
                args.extend((str(weight), prefix))
        return args + ["--tokenizer-type", "TikTokenizer", "--tokenizer-model", self.tokenizer_model,
                       "--tiktoken-num-special-tokens", "256"]

    def to_dict(self):
        return {"train": [{"weight": w, "prefix": p} for w, p in self.train],
                "validation": [{"weight": w, "prefix": p} for w, p in self.validation],
                "tokenizer_model": self.tokenizer_model}
