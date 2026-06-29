from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np

from data import RerankerExample, load_dataset_splits, load_examples
from metrics import compute_all_metrics, is_better_metric
from modeling import (
    DEFAULT_MODEL_NAME,
    append_answer_prompt,
    load_causal_training_model,
    predict_causal_model,
    predict_sequence_classification_model,
    save_reranker_config,
    torch,
)


logger = logging.getLogger(__name__)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pointwise BCE soft-label distillation for MemReranker-style Qwen3 reranking."
    )
    parser.add_argument("--train_file", required=True, help="JSON/JSONL data file, or train split if dev/test files are set.")
    parser.add_argument("--dev_file", default=None, help="Optional explicit dev JSON/JSONL split.")
    parser.add_argument("--test_file", default=None, help="Optional explicit test JSON/JSONL split.")
    parser.add_argument("--split_file", default=None, help="Optional JSON split file with train/dev/test group keys.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--backend", default="auto", choices=["auto", "cross_encoder", "causal_lm"])
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    add_bool_arg(parser, "use_lora", default=True, help_text="Use LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--default_instruction", default="")
    add_bool_arg(parser, "gradient_checkpointing", default=True, help_text="Enable gradient checkpointing")
    parser.add_argument("--load_in_4bit", action="store_true", help="Enable QLoRA-style 4-bit loading for causal backend.")
    parser.add_argument("--logging_steps", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_train_dev_test(args: argparse.Namespace) -> tuple[dict[str, list[RerankerExample]], dict[str, Any]]:
    if args.dev_file or args.test_file:
        train = load_examples(args.train_file, default_instruction=args.default_instruction)
        dev = load_examples(args.dev_file, default_instruction=args.default_instruction) if args.dev_file else []
        test = load_examples(args.test_file, default_instruction=args.default_instruction) if args.test_file else []
        split_info = {
            "strategy": "explicit_files",
            "splits": {
                "train": {"num_examples": len(train), "num_groups": len({ex.group_key for ex in train})},
                "dev": {"num_examples": len(dev), "num_groups": len({ex.group_key for ex in dev})},
                "test": {"num_examples": len(test), "num_groups": len({ex.group_key for ex in test})},
            },
        }
        return {"train": train, "dev": dev, "test": test}, split_info
    return load_dataset_splits(
        args.train_file,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_file=args.split_file,
        default_instruction=args.default_instruction,
    )


def examples_to_records(examples: list[RerankerExample], scores: list[float]) -> list[dict[str, Any]]:
    rows = []
    for ex, score in zip(examples, scores, strict=False):
        rows.append(
            {
                "group_key": ex.group_key,
                "query": ex.query,
                "query_id": ex.query_id,
                "doc": ex.doc,
                "label": ex.label,
                "raw_label": ex.raw_label,
                "score": float(score),
                "reason": ex.reason,
            }
        )
    return rows


def evaluate_examples(
    examples: list[RerankerExample],
    predict_fn: Callable[[list[str]], list[float]],
    relevance_threshold: float,
) -> dict[str, float]:
    input_texts = [ex.input_text for ex in examples]
    scores = predict_fn(input_texts)
    records = examples_to_records(examples, scores)
    overall, _ = compute_all_metrics(
        records,
        query_key="group_key",
        relevance_threshold=relevance_threshold,
    )
    return overall


def get_device() -> Any:
    if torch is None:
        raise RuntimeError("torch is required for training")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def autocast_context(args: argparse.Namespace, device: Any) -> Any:
    if torch is None or device.type != "cuda" or not (args.bf16 or args.fp16):
        return nullcontext()
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_output_dir(args: argparse.Namespace, split_info: dict[str, Any]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "training_args.json", vars(args))
    save_json(output_dir / "split_info.json", split_info)


def maybe_apply_lora_to_sequence_model(model: Any, args: argparse.Namespace) -> Any:
    if not args.use_lora:
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def make_optimizer_and_scheduler(model: Any, args: argparse.Namespace, num_batches: int) -> tuple[Any, Any]:
    if torch is None:
        raise RuntimeError("torch is required for training")
    from transformers import get_linear_schedule_with_warmup

    update_steps_per_epoch = max(1, math.ceil(num_batches / args.gradient_accumulation_steps))
    total_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def train_cross_encoder(
    args: argparse.Namespace,
    splits: dict[str, list[RerankerExample]],
) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("torch is required for CrossEncoder training")
    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader

    logger.info("Trying CrossEncoder backend with model %s", args.model_name_or_path)
    ce = CrossEncoder(args.model_name_or_path, num_labels=1, max_length=args.max_length)
    model = maybe_apply_lora_to_sequence_model(ce.model, args)
    tokenizer = ce.tokenizer
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    device = get_device()
    model.to(device)

    train_examples = splits["train"]
    eval_examples = splits["dev"] or splits["test"] or splits["train"]
    if not splits["dev"]:
        logger.warning("Dev split is empty; evaluating on %s split.", "test" if splits["test"] else "train")

    def collate(batch: list[RerankerExample]) -> tuple[dict[str, Any], Any]:
        encoded = tokenizer(
            [ex.input_text for ex in batch],
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.float32)
        return encoded, labels

    loader = DataLoader(
        train_examples,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    if len(loader) == 0:
        raise ValueError("Empty training dataloader")

    optimizer, scheduler = make_optimizer_and_scheduler(model, args, len(loader))
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_metrics: dict[str, float] | None = None
    best_dir = Path(args.output_dir) / "best"
    history_path = Path(args.output_dir) / "metrics_history.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, (encoded, labels) in enumerate(loader, start=1):
            encoded = move_batch_to_device(encoded, device)
            labels = labels.to(device)
            with autocast_context(args, device):
                outputs = model(**encoded)
                logits = outputs.logits.squeeze(-1)
                loss = loss_fn(logits.float(), labels.float())
                loss_to_backprop = loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            running_loss += float(loss.detach().cpu())
            should_step = step % args.gradient_accumulation_steps == 0 or step == len(loader)
            if should_step:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if args.logging_steps > 0 and step % args.logging_steps == 0:
                logger.info("epoch=%d step=%d train_loss=%.6f", epoch, step, running_loss / step)

        predict_fn = lambda texts: predict_sequence_classification_model(
            model,
            tokenizer,
            texts,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            device=str(device),
        )
        metrics = evaluate_examples(eval_examples, predict_fn, args.relevance_threshold)
        metrics["epoch"] = float(epoch)
        logger.info("epoch=%d dev_metrics=%s", epoch, json.dumps(metrics, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"backend": "cross_encoder", **metrics}, ensure_ascii=False) + "\n")

        if is_better_metric(metrics, best_metrics):
            best_metrics = metrics
            if best_dir.exists():
                shutil.rmtree(best_dir)
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_reranker_config(
                best_dir,
                {
                    "backend": "cross_encoder",
                    "base_model_name_or_path": args.model_name_or_path,
                    "max_length": args.max_length,
                    "score_activation": "sigmoid",
                    "loss": "BCEWithLogitsLoss",
                    "label_normalization": "labels / 10 clipped to [0, 1]",
                },
            )
            save_json(Path(args.output_dir) / "best_metrics.json", best_metrics)
            logger.info("Saved new best checkpoint to %s", best_dir)

    return best_metrics or {}


def train_causal_lm(
    args: argparse.Namespace,
    splits: dict[str, list[RerankerExample]],
) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("torch is required for causal LM training")
    from torch.utils.data import DataLoader

    logger.info("Using causal LM yes/no-logit backend with model %s", args.model_name_or_path)
    wrapper, tokenizer = load_causal_training_model(
        args.model_name_or_path,
        bf16=args.bf16,
        fp16=args.fp16,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    device = get_device()
    if not args.load_in_4bit:
        wrapper.to(device)

    train_examples = splits["train"]
    eval_examples = splits["dev"] or splits["test"] or splits["train"]
    if not splits["dev"]:
        logger.warning("Dev split is empty; evaluating on %s split.", "test" if splits["test"] else "train")

    def collate(batch: list[RerankerExample]) -> dict[str, Any]:
        encoded = tokenizer(
            [append_answer_prompt(ex.input_text) for ex in batch],
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([ex.label for ex in batch], dtype=torch.float32)
        return encoded

    loader = DataLoader(
        train_examples,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    if len(loader) == 0:
        raise ValueError("Empty training dataloader")

    optimizer, scheduler = make_optimizer_and_scheduler(wrapper, args, len(loader))
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    best_metrics: dict[str, float] | None = None
    best_dir = Path(args.output_dir) / "best"
    history_path = Path(args.output_dir) / "metrics_history.jsonl"

    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            if not args.load_in_4bit:
                batch = move_batch_to_device(batch, device)
            else:
                target_device = next(wrapper.parameters()).device
                batch = move_batch_to_device(batch, target_device)

            with autocast_context(args, device):
                outputs = wrapper(**batch)
                loss = outputs["loss"]
                loss_to_backprop = loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            running_loss += float(loss.detach().cpu())
            should_step = step % args.gradient_accumulation_steps == 0 or step == len(loader)
            if should_step:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if args.logging_steps > 0 and step % args.logging_steps == 0:
                logger.info("epoch=%d step=%d train_loss=%.6f", epoch, step, running_loss / step)

        predict_fn = lambda texts: predict_causal_model(
            wrapper,
            tokenizer,
            texts,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            device=str(device),
        )
        metrics = evaluate_examples(eval_examples, predict_fn, args.relevance_threshold)
        metrics["epoch"] = float(epoch)
        logger.info("epoch=%d dev_metrics=%s", epoch, json.dumps(metrics, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"backend": "causal_lm", **metrics}, ensure_ascii=False) + "\n")

        if is_better_metric(metrics, best_metrics):
            best_metrics = metrics
            if best_dir.exists():
                shutil.rmtree(best_dir)
            best_dir.mkdir(parents=True, exist_ok=True)
            wrapper.model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_reranker_config(
                best_dir,
                {
                    "backend": "causal_lm",
                    "base_model_name_or_path": args.model_name_or_path,
                    "max_length": args.max_length,
                    "answer_prompt": "<Answer>:",
                    "score": "sigmoid(logit_yes - logit_no)",
                    "loss": "BCEWithLogitsLoss",
                    "label_normalization": "labels / 10 clipped to [0, 1]",
                    "use_lora": args.use_lora,
                    "load_in_4bit": args.load_in_4bit,
                },
            )
            save_json(Path(args.output_dir) / "best_metrics.json", best_metrics)
            logger.info("Saved new best checkpoint to %s", best_dir)

    return best_metrics or {}


def train(args: argparse.Namespace, splits: dict[str, list[RerankerExample]]) -> dict[str, float]:
    if args.backend == "auto":
        backend_order = ["cross_encoder", "causal_lm"]
    else:
        backend_order = [args.backend]

    errors = []
    for backend in backend_order:
        try:
            if backend == "cross_encoder":
                return train_cross_encoder(args, splits)
            if backend == "causal_lm":
                return train_causal_lm(args, splits)
        except Exception as exc:
            if args.backend != "auto":
                raise
            logger.exception("Backend %s failed; trying next backend if available.", backend)
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("All training backends failed: " + " | ".join(errors))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    set_seed(args.seed)

    splits, split_info = load_train_dev_test(args)
    if not splits["train"]:
        raise ValueError("Train split is empty.")
    logger.info(
        "Split sizes: train=%d dev=%d test=%d",
        len(splits["train"]),
        len(splits["dev"]),
        len(splits["test"]),
    )
    prepare_output_dir(args, split_info)
    best_metrics = train(args, splits)
    logger.info("Best metrics: %s", json.dumps(best_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
