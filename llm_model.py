"""
llm_model.py
============
Wrapper for google/flan-t5-base used to generate analytical comments from
Ames Housing analysis prompts.

The imports from HuggingFace are intentionally done inside the class so the
whole analytical project can still run when LLM dependencies are not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MODEL_ID = "google/flan-t5-base"
MAX_INPUT_LEN = 512
MAX_NEW_TOKENS = 150


@dataclass(frozen=True)
class GenerationConfig:
    max_input_len: int = MAX_INPUT_LEN
    max_new_tokens: int = MAX_NEW_TOKENS
    do_sample: bool = False


class AmesFlanT5:
    """Small wrapper around FLAN-T5 for deterministic text generation."""

    def __init__(
        self,
        checkpoint: Optional[str | Path] = None,
        device: Optional[str] = None,
        config: GenerationConfig | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Missing LLM dependencies. Install them with: pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        self.config = config or GenerationConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = str(checkpoint) if checkpoint else MODEL_ID

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_len,
        ).to(self.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def generate_batch(self, prompts: list[str], batch_size: int = 4) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            part = prompts[start : start + batch_size]
            inputs = self.tokenizer(
                part,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_input_len,
            ).to(self.device)

            with self.torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                )

            outputs.extend(
                text.strip()
                for text in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            )
        return outputs
