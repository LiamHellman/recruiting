"""Versioned prompt files: app/llm/prompts/<name>/v<N>.md.

Each file has YAML front matter (model, max_tokens, effort) followed by the system prompt.
A changed prompt or model is a new file, never an edit to an existing one.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    model: str
    max_tokens: int
    effort: Effort
    system: str

    @property
    def ref(self) -> str:
        return f"{self.name}/v{self.version}"


class PromptError(ValueError):
    pass


@lru_cache
def load_prompt(name: str, version: int, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    path = prompts_dir / name / f"v{version}.md"
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise PromptError(f"{path}: missing YAML front matter")
    _, front, body = text.split("---\n", 2)
    meta = yaml.safe_load(front) or {}
    for key in ("model", "max_tokens", "effort"):
        if key not in meta:
            raise PromptError(f"{path}: front matter needs '{key}'")
    if meta["effort"] not in ("low", "medium", "high", "xhigh", "max"):
        raise PromptError(f"{path}: unknown effort {meta['effort']!r}")
    system = body.strip()
    if not system:
        raise PromptError(f"{path}: empty system prompt")
    return Prompt(
        name=name,
        version=version,
        model=str(meta["model"]),
        max_tokens=int(meta["max_tokens"]),
        effort=cast(Effort, meta["effort"]),
        system=system,
    )
