from __future__ import annotations

import json
import logging
import math

from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_4B_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
ANSWER_PROMPT = "\n<Answer>:"
MODEL_NAME_ALIASES = {
    "qwen/qwen3-reranker-0.6": DEFAULT_MODEL_NAME,
    "qwen/qwen3-reranker-0.6b": DEFAULT_MODEL_NAME,
    "qwen/qwen3-reranker-06b": DEFAULT_MODEL_NAME,
    "qwen/qwen3-reranker-4": DEFAULT_4B_MODEL_NAME,
    "qwen/qwen3-reranker-4b": DEFAULT_4B_MODEL_NAME,
}


try:
    import torch
except Exception:  # pragma: no cover - lets mock mode work without torch.
    torch = None  # type: ignore[assignment]


def append_answer_prompt(text: str) -> str:
    return f"{text.rstrip()}{ANSWER_PROMPT}"


def normalize_model_name_or_path(model_name_or_path: str) -> str:
    """Correct common Qwen3 reranker Hub ID typos without touching local paths."""
    path = Path(model_name_or_path)
    if path.exists():
        return model_name_or_path
    normalized = model_name_or_path.strip()
    alias = MODEL_NAME_ALIASES.get(normalized.lower())
    if alias:
        if normalized != alias:
            logger.warning("Normalized model name %r to %r", model_name_or_path, alias)
        return alias
    return normalized


def model_load_help(model_name_or_path: str, exc: BaseException) -> str:
    return (
        f"Failed to load model/tokenizer from {model_name_or_path!r}: {exc}\n\n"
        "Checklist:\n"
        "1. Use the exact Hugging Face id, for example Qwen/Qwen3-Reranker-0.6B "
        "or Qwen/Qwen3-Reranker-4B.\n"
        "2. Install all dependencies: pip install -r requirements.txt\n"
        "3. Qwen3 requires transformers>=4.51.0; the baseline CrossEncoder path also "
        "requires sentence-transformers.\n"
        "4. If the machine cannot reach Hugging Face, download the model on a machine "
        "with network access and pass the local directory via --model_path or "
        "--model_name_or_path.\n"
        "5. If your cluster uses a proxy or mirror, fix HTTPS_PROXY/HTTP_PROXY or set "
        "HF_ENDPOINT before running."
    )


def sigmoid_array(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def resolve_yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    def choose(candidates: list[str]) -> int:
        fallback: list[int] | None = None
        for token in candidates:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                if fallback is None:
                    fallback = ids
                if len(ids) == 1:
                    return int(ids[0])
        if fallback:
            return int(fallback[-1])
        raise ValueError(f"Could not resolve token id from candidates: {candidates}")

    yes_id = choose(["yes", "Yes", " yes", " Yes"])
    no_id = choose(["no", "No", " no", " No"])
    logger.info("Resolved reranker yes/no token ids: yes=%s no=%s", yes_id, no_id)
    return yes_id, no_id


class MockRerankerScorer:
    """Deterministic lexical scorer for smoke tests; it is not a model."""

    backend = "mock"

    def __init__(self, query: str | None = None):
        self.query = query or ""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return {tok for tok in normalized.split() if tok}

    def predict(self, input_texts: list[str], batch_size: int = 32) -> list[float]:
        scores = []
        for text in input_texts:
            query = self.query
            if "<Query>:" in text and "<Document>:" in text:
                query = text.split("<Query>:", 1)[1].split("<Document>:", 1)[0]
                doc = text.split("<Document>:", 1)[1]
            else:
                doc = text
            q_tokens = self._tokens(query)
            d_tokens = self._tokens(doc)
            if not q_tokens or not d_tokens:
                scores.append(0.0)
                continue
            overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
            length_bonus = min(0.15, math.log1p(len(d_tokens)) / 100.0)
            scores.append(float(min(1.0, overlap + length_bonus)))
        return scores


class CrossEncoderScorer:
    backend = "cross_encoder"

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 4096,
        device: str | None = None,
        torch_dtype: Any | None = None,
    ):
        if torch is None:
            raise RuntimeError("torch is required for CrossEncoderScorer")
        model_name_or_path = normalize_model_name_or_path(model_name_or_path)
        path = Path(model_name_or_path)
        adapter_config = path / "adapter_config.json"
        if adapter_config.exists():
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            adapter_data = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_path = adapter_data.get("base_model_name_or_path") or DEFAULT_MODEL_NAME
            tokenizer_path = model_name_or_path if (path / "tokenizer_config.json").exists() else base_path
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                padding_side="right",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_path,
                num_labels=1,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            self.model = PeftModel.from_pretrained(base_model, model_name_or_path)
        else:
            from sentence_transformers import CrossEncoder

            kwargs: dict[str, Any] = {"max_length": max_length}
            if device:
                kwargs["device"] = device
            if torch_dtype is not None:
                kwargs["automodel_args"] = {"torch_dtype": torch_dtype}
            self.cross_encoder = CrossEncoder(model_name_or_path, num_labels=1, **kwargs)
            self.model = self.cross_encoder.model
            self.tokenizer = self.cross_encoder.tokenizer
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, input_texts: list[str], batch_size: int = 16) -> list[float]:
        return predict_sequence_classification_model(
            self.model,
            self.tokenizer,
            input_texts,
            max_length=self.max_length,
            batch_size=batch_size,
            device=self.device,
        )


if torch is not None:

    class QwenCausalReranker(torch.nn.Module):
        def __init__(
            self,
            model: Any,
            tokenizer: Any,
            yes_token_id: int | None = None,
            no_token_id: int | None = None,
        ):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.yes_token_id, self.no_token_id = (
                (yes_token_id, no_token_id)
                if yes_token_id is not None and no_token_id is not None
                else resolve_yes_no_token_ids(tokenizer)
            )
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            labels: Any | None = None,
        ) -> dict[str, Any]:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_indices = attention_mask.size(1) - 1 - torch.argmax(
                attention_mask.flip(dims=[1]).long(),
                dim=1,
            )
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            last_logits = outputs.logits[batch_indices, last_indices, :]
            yes_logits = last_logits[:, self.yes_token_id]
            no_logits = last_logits[:, self.no_token_id]
            logits = yes_logits - no_logits
            result = {"logits": logits}
            if labels is not None:
                result["loss"] = self.loss_fn(logits.float(), labels.float())
            return result

else:
    QwenCausalReranker = None  # type: ignore[assignment]


class CausalLMScorer:
    backend = "causal_lm"

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 4096,
        device: str | None = None,
        torch_dtype: Any | None = None,
    ):
        if torch is None or QwenCausalReranker is None:
            raise RuntimeError("torch is required for CausalLMScorer")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name_or_path = normalize_model_name_or_path(model_name_or_path)
        path = Path(model_name_or_path)
        adapter_config = path / "adapter_config.json"
        load_path = model_name_or_path
        tokenizer_path = model_name_or_path
        is_adapter = adapter_config.exists()
        if is_adapter:
            adapter_data = json.loads(adapter_config.read_text(encoding="utf-8"))
            load_path = adapter_data.get("base_model_name_or_path") or DEFAULT_MODEL_NAME
            tokenizer_path = model_name_or_path if (path / "tokenizer_config.json").exists() else load_path

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if is_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, model_name_or_path)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wrapper = QwenCausalReranker(model, tokenizer)
        self.wrapper.eval()
        self.wrapper.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def predict(self, input_texts: list[str], batch_size: int = 4) -> list[float]:
        return predict_causal_model(
            self.wrapper,
            self.tokenizer,
            input_texts,
            max_length=self.max_length,
            batch_size=batch_size,
            device=self.device,
        )


def predict_sequence_classification_model(
    model: Any,
    tokenizer: Any,
    input_texts: list[str],
    max_length: int = 4096,
    batch_size: int = 16,
    device: str | None = None,
) -> list[float]:
    if torch is None:
        raise RuntimeError("torch is required for sequence-classification prediction")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(input_texts), batch_size):
            batch = input_texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            logits = outputs.logits.squeeze(-1).detach().float().cpu().numpy()
            scores.extend(sigmoid_array(np.asarray(logits)).tolist())
    return [float(score) for score in scores]


def predict_causal_model(
    wrapper: Any,
    tokenizer: Any,
    input_texts: list[str],
    max_length: int = 4096,
    batch_size: int = 4,
    device: str | None = None,
) -> list[float]:
    if torch is None:
        raise RuntimeError("torch is required for causal prediction")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.eval()
    scores: list[float] = []
    prompts = [append_answer_prompt(text) for text in input_texts]
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = wrapper(**encoded)
            logits = outputs["logits"].detach().float().cpu().numpy()
            scores.extend(sigmoid_array(np.asarray(logits)).tolist())
    return [float(score) for score in scores]


def torch_dtype_from_flags(bf16: bool = False, fp16: bool = False) -> Any | None:
    if torch is None:
        return None
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return None


def load_causal_training_model(
    model_name_or_path: str,
    bf16: bool = False,
    fp16: bool = False,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_4bit: bool = False,
    device_map: Any | None = None,
    gradient_checkpointing: bool = True,
) -> tuple[Any, Any]:
    if torch is None or QwenCausalReranker is None:
        raise RuntimeError("torch is required for training")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name_or_path = normalize_model_name_or_path(model_name_or_path)
    dtype = torch_dtype_from_flags(bf16=bf16, fp16=fp16)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = dtype or torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = device_map if device_map is not None else "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
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
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return QwenCausalReranker(model, tokenizer), tokenizer


def detect_backend(model_path: str, requested_backend: str = "auto") -> str:
    if requested_backend != "auto":
        return requested_backend
    model_path = normalize_model_name_or_path(model_path)
    path = Path(model_path)
    config_path = path / "reranker_config.json"
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        backend = data.get("backend")
        if backend:
            return str(backend)
    if (path / "adapter_config.json").exists():
        return "causal_lm"
    return "cross_encoder"


def load_scorer(
    model_path: str = DEFAULT_MODEL_NAME,
    backend: str = "auto",
    max_length: int = 4096,
    batch_query: str | None = None,
    bf16: bool = False,
    fp16: bool = False,
    mock: bool = False,
) -> Any:
    if mock:
        return MockRerankerScorer(query=batch_query)

    model_path = normalize_model_name_or_path(model_path)
    dtype = torch_dtype_from_flags(bf16=bf16, fp16=fp16)
    backend = detect_backend(model_path, backend)
    logger.info("Loading reranker scorer backend=%s model=%s", backend, model_path)

    if backend == "cross_encoder":
        try:
            return CrossEncoderScorer(model_path, max_length=max_length, torch_dtype=dtype)
        except Exception as exc:
            if detect_backend(model_path, "auto") == "cross_encoder":
                logger.warning("CrossEncoder load failed, falling back to causal LM: %s", exc)
                try:
                    return CausalLMScorer(model_path, max_length=max_length, torch_dtype=dtype)
                except Exception as causal_exc:
                    raise RuntimeError(model_load_help(model_path, causal_exc)) from causal_exc
            raise
    if backend == "causal_lm":
        try:
            return CausalLMScorer(model_path, max_length=max_length, torch_dtype=dtype)
        except Exception as exc:
            raise RuntimeError(model_load_help(model_path, exc)) from exc
    raise ValueError(f"Unknown backend: {backend}")


def save_reranker_config(output_dir: str | Path, config: dict[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "reranker_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
