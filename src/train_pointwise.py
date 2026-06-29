from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil

from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm

from data import RerankerExample, load_dataset_splits, load_examples
from metrics import compute_all_metrics, is_better_metric
from modeling import (
    DEFAULT_MODEL_NAME,
    append_answer_prompt,
    load_causal_training_model,
    model_load_help,
    normalize_model_name_or_path,
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
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Set DDP find_unused_parameters=True. Usually keep this off for LoRA.",
    )
    return parser.parse_args()


def create_accelerator(args: argparse.Namespace) -> Any:
    try:
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs
    except ImportError as exc:
        raise RuntimeError("accelerate is required for training. Install requirements.txt first.") from exc

    mixed_precision = "no"
    if args.bf16:
        mixed_precision = "bf16"
    elif args.fp16:
        mixed_precision = "fp16"

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(args.ddp_find_unused_parameters)
    )
    return Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )


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


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_output_dir(args: argparse.Namespace, split_info: dict[str, Any], accelerator: Any) -> None:
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "training_args.json", vars(args))
        save_json(output_dir / "split_info.json", split_info)
    accelerator.wait_for_everyone()


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
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def make_optimizer(model: Any, args: argparse.Namespace) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for training")
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def make_scheduler(optimizer: Any, args: argparse.Namespace, num_batches: int) -> Any:
    from transformers import get_linear_schedule_with_warmup

    update_steps_per_epoch = max(1, math.ceil(num_batches / args.gradient_accumulation_steps))
    total_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def save_best_cross_encoder(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    accelerator: Any,
    metrics: dict[str, float],
) -> None:
    best_dir = Path(args.output_dir) / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(best_dir, save_function=accelerator.save)
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
            "distributed": {
                "num_processes": accelerator.num_processes,
                "mixed_precision": accelerator.mixed_precision,
            },
        },
    )
    save_json(Path(args.output_dir) / "best_metrics.json", metrics)
    logger.info("Saved new best checkpoint to %s", best_dir)


def save_best_causal(
    wrapper: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    accelerator: Any,
    metrics: dict[str, float],
) -> None:
    best_dir = Path(args.output_dir) / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(wrapper)
    unwrapped.model.save_pretrained(best_dir, save_function=accelerator.save)
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
            "distributed": {
                "num_processes": accelerator.num_processes,
                "mixed_precision": accelerator.mixed_precision,
            },
        },
    )
    save_json(Path(args.output_dir) / "best_metrics.json", metrics)
    logger.info("Saved new best checkpoint to %s", best_dir)


def train_cross_encoder(
    args: argparse.Namespace,
    splits: dict[str, list[RerankerExample]],
    accelerator: Any,
) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("torch is required for CrossEncoder training")
    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader

    if accelerator.is_main_process:
        logger.info("Trying CrossEncoder backend with model %s", args.model_name_or_path)
    ce = CrossEncoder(args.model_name_or_path, num_labels=1, max_length=args.max_length)
    model = maybe_apply_lora_to_sequence_model(ce.model, args)
    tokenizer = ce.tokenizer
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_examples = splits["train"]
    eval_examples = splits["dev"] or splits["test"] or splits["train"]
    if accelerator.is_main_process and not splits["dev"]:
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

    optimizer = make_optimizer(model, args)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    scheduler = make_scheduler(optimizer, args, len(loader))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_metrics: dict[str, float] | None = None
    history_path = Path(args.output_dir) / "metrics_history.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(
            enumerate(loader, start=1),
            total=len(loader),
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            ascii=True,
            disable=args.disable_tqdm or not accelerator.is_main_process,
        )
        for step, (encoded, labels) in progress:
            with accelerator.accumulate(model):
                outputs = model(**encoded)
                logits = outputs.logits.squeeze(-1)
                loss = loss_fn(logits.float(), labels.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += float(accelerator.gather(loss.detach()).mean().cpu())
            if accelerator.is_main_process and not args.disable_tqdm:
                progress.set_postfix(loss=f"{running_loss / step:.4f}")
            if accelerator.is_main_process and args.logging_steps > 0 and step % args.logging_steps == 0:
                logger.info("epoch=%d step=%d train_loss=%.6f", epoch, step, running_loss / step)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            eval_model = accelerator.unwrap_model(model)
            predict_fn = lambda texts: predict_sequence_classification_model(
                eval_model,
                tokenizer,
                texts,
                max_length=args.max_length,
                batch_size=args.eval_batch_size,
                device=str(accelerator.device),
            )
            metrics = evaluate_examples(eval_examples, predict_fn, args.relevance_threshold)
            metrics["epoch"] = float(epoch)
            logger.info("epoch=%d dev_metrics=%s", epoch, json.dumps(metrics, ensure_ascii=False))
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"backend": "cross_encoder", **metrics}, ensure_ascii=False) + "\n")

            if is_better_metric(metrics, best_metrics):
                best_metrics = metrics
                save_best_cross_encoder(model, tokenizer, args, accelerator, metrics)
        accelerator.wait_for_everyone()

    return best_metrics or {}


def _kbit_device_map_for_process(args: argparse.Namespace, accelerator: Any) -> Any | None:
    if not args.load_in_4bit or accelerator.num_processes <= 1:
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
    return {"": local_rank}


def train_causal_lm(
    args: argparse.Namespace,
    splits: dict[str, list[RerankerExample]],
    accelerator: Any,
) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("torch is required for causal LM training")
    from torch.utils.data import DataLoader

    if accelerator.is_main_process:
        logger.info("Using causal LM yes/no-logit backend with model %s", args.model_name_or_path)
    try:
        wrapper, tokenizer = load_causal_training_model(
            args.model_name_or_path,
            bf16=args.bf16,
            fp16=args.fp16,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            load_in_4bit=args.load_in_4bit,
            device_map=_kbit_device_map_for_process(args, accelerator),
            gradient_checkpointing=args.gradient_checkpointing,
        )
    except Exception as exc:
        raise RuntimeError(model_load_help(args.model_name_or_path, exc)) from exc

    train_examples = splits["train"]
    eval_examples = splits["dev"] or splits["test"] or splits["train"]
    if accelerator.is_main_process and not splits["dev"]:
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

    optimizer = make_optimizer(wrapper, args)
    wrapper, optimizer, loader = accelerator.prepare(wrapper, optimizer, loader)
    scheduler = make_scheduler(optimizer, args, len(loader))
    best_metrics: dict[str, float] | None = None
    history_path = Path(args.output_dir) / "metrics_history.jsonl"

    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(
            enumerate(loader, start=1),
            total=len(loader),
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            ascii=True,
            disable=args.disable_tqdm or not accelerator.is_main_process,
        )
        for step, batch in progress:
            with accelerator.accumulate(wrapper):
                outputs = wrapper(**batch)
                loss = outputs["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += float(accelerator.gather(loss.detach()).mean().cpu())
            if accelerator.is_main_process and not args.disable_tqdm:
                progress.set_postfix(loss=f"{running_loss / step:.4f}")
            if accelerator.is_main_process and args.logging_steps > 0 and step % args.logging_steps == 0:
                logger.info("epoch=%d step=%d train_loss=%.6f", epoch, step, running_loss / step)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            eval_wrapper = accelerator.unwrap_model(wrapper)
            predict_fn = lambda texts: predict_causal_model(
                eval_wrapper,
                tokenizer,
                texts,
                max_length=args.max_length,
                batch_size=args.eval_batch_size,
                device=str(accelerator.device),
            )
            metrics = evaluate_examples(eval_examples, predict_fn, args.relevance_threshold)
            metrics["epoch"] = float(epoch)
            logger.info("epoch=%d dev_metrics=%s", epoch, json.dumps(metrics, ensure_ascii=False))
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"backend": "causal_lm", **metrics}, ensure_ascii=False) + "\n")

            if is_better_metric(metrics, best_metrics):
                best_metrics = metrics
                save_best_causal(wrapper, tokenizer, args, accelerator, metrics)
        accelerator.wait_for_everyone()

    return best_metrics or {}


def train(
    args: argparse.Namespace,
    splits: dict[str, list[RerankerExample]],
    accelerator: Any,
) -> dict[str, float]:
    backend_order = ["cross_encoder", "causal_lm"] if args.backend == "auto" else [args.backend]
    errors = []
    for backend in backend_order:
        try:
            if backend == "cross_encoder":
                return train_cross_encoder(args, splits, accelerator)
            if backend == "causal_lm":
                return train_causal_lm(args, splits, accelerator)
        except Exception as exc:
            if args.backend != "auto":
                raise
            if accelerator.is_main_process:
                logger.exception("Backend %s failed; trying next backend if available.", backend)
            errors.append(f"{backend}: {exc}")
            accelerator.wait_for_everyone()
    raise RuntimeError("All training backends failed: " + " | ".join(errors))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    args.model_name_or_path = normalize_model_name_or_path(args.model_name_or_path)

    accelerator = create_accelerator(args)
    set_seed(args.seed)

    splits, split_info = load_train_dev_test(args)
    if not splits["train"]:
        raise ValueError("Train split is empty.")
    if accelerator.is_main_process:
        logger.info(
            "Split sizes: train=%d dev=%d test=%d",
            len(splits["train"]),
            len(splits["dev"]),
            len(splits["test"]),
        )
        logger.info(
            "Accelerate: num_processes=%d mixed_precision=%s device=%s",
            accelerator.num_processes,
            accelerator.mixed_precision,
            accelerator.device,
        )
    prepare_output_dir(args, split_info, accelerator)
    best_metrics = train(args, splits, accelerator)
    if accelerator.is_main_process:
        logger.info("Best metrics: %s", json.dumps(best_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
