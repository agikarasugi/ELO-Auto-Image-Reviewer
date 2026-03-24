from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_BUILTIN_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class PromptTemplate:
    name: str
    system: str


def default_prompt_path() -> Path:
    return _BUILTIN_PROMPTS_DIR / "default.md"


def list_builtin_prompts() -> list[Path]:
    return sorted(_BUILTIN_PROMPTS_DIR.glob("*.md"))


def resolve_prompt_path(value: Path) -> Path:
    """Resolve a user-supplied value to a prompt file path.

    Accepts:
    - A bare name like ``illustrations`` → looks up in the built-in prompts dir
    - An explicit file path → used as-is
    """
    if value.parent == Path(".") and value.suffix == "":
        return _BUILTIN_PROMPTS_DIR / f"{value}.md"
    return value


def load_prompt(path: Path) -> PromptTemplate:
    """Load a markdown prompt file.  The entire file content is the system prompt.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if the file is empty.
    """
    if not path.exists():
        builtin_names = [p.stem for p in list_builtin_prompts()]
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Built-in templates: {', '.join(builtin_names)}"
        )

    system = path.read_text(encoding="utf-8").strip()
    if not system:
        raise ValueError(f"Prompt file '{path}' is empty.")

    return PromptTemplate(name=path.stem, system=system)
