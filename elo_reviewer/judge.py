import random
import re
import time
from pathlib import Path
from typing import Literal

import openai
from rich.markup import escape

from . import console as cm
from .image_utils import build_image_content_block

SYSTEM_PROMPT = """\
You are an expert image quality judge. Your task is to compare two images \
and determine which is superior. Evaluate based on:
- Technical quality (sharpness, focus, exposure, noise)
- Composition (framing, balance, rule of thirds)
- Visual appeal (color, contrast, aesthetic coherence)
- Subject clarity (is the subject well-presented?)

You will ultimately give a final answer of exactly one capital letter: 'A' or 'B'.\
"""

TURN1_TEXT = """\
Compare the following two images carefully.

Image A:"""

TURN1_SUFFIX = """\

Image B:"""

TURN1_CLOSING = """\

Provide a thorough analysis of both images across all evaluation criteria. \
Explain the strengths and weaknesses of each.\
"""

TURN2_TEXT = """\
Review your analysis critically. Consider:
1. Did you weight all criteria fairly?
2. Could there be any ordering or position bias in how you viewed them?
3. Is there anything technically significant you may have missed?

Revise your assessment if needed, then summarize your final reasoning.\
"""

TURN3_TEXT = """\
Final answer only. Which image is better overall?
Respond with a single capital letter: A or B\
"""


def _parse_decision(text: str) -> Literal["A", "B"] | None:
    stripped = text.strip()
    if stripped in ("A", "B"):
        return stripped  # type: ignore[return-value]
    matches = re.findall(r"\b([AB])\b", stripped)
    if matches:
        return matches[-1]  # type: ignore[return-value]
    return None


class Judge:
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.verbose = verbose

    def _chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> str:
        kwargs: dict = {"model": self.model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise
            except openai.APIError:
                raise

    def compare(
        self,
        image_a: Path,
        image_b: Path,
    ) -> tuple[Literal["A", "B"], list[dict], bool]:
        """
        Run a 3-turn comparison conversation.

        Returns:
            decision: 'A' or 'B'
            history: full message history
            used_fallback: True if parse failed and random choice was used
        """
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Turn 1: show both images and ask for analysis
        turn1_content = [
            {"type": "text", "text": TURN1_TEXT},
            build_image_content_block(image_a),
            {"type": "text", "text": TURN1_SUFFIX},
            build_image_content_block(image_b),
            {"type": "text", "text": TURN1_CLOSING},
        ]
        messages.append({"role": "user", "content": turn1_content})

        try:
            reply1 = self._chat(messages)
        except openai.APIError as e:
            cm.console.print(f"  [bold red][API error turn 1: {escape(str(e))}] — using random fallback[/bold red]")
            decision = random.choice(["A", "B"])
            return decision, messages, True  # type: ignore[return-value]

        messages.append({"role": "assistant", "content": reply1})
        if self.verbose:
            cm.console.print(f"\n  [dim cyan][Turn 1][/dim cyan]\n{escape(reply1)}\n")

        # Turn 2: self-review
        messages.append({"role": "user", "content": TURN2_TEXT})
        try:
            reply2 = self._chat(messages, temperature=0.3)
        except openai.APIError as e:
            cm.console.print(f"  [bold red][API error turn 2: {escape(str(e))}] — using random fallback[/bold red]")
            decision = random.choice(["A", "B"])
            return decision, messages, True  # type: ignore[return-value]

        messages.append({"role": "assistant", "content": reply2})
        if self.verbose:
            cm.console.print(f"  [dim cyan][Turn 2][/dim cyan]\n{escape(reply2)}\n")

        # Turn 3: final decision
        messages.append({"role": "user", "content": TURN3_TEXT})
        try:
            reply3 = self._chat(messages, temperature=0.0)
        except openai.APIError as e:
            cm.console.print(f"  [bold red][API error turn 3: {escape(str(e))}] — using random fallback[/bold red]")
            decision = random.choice(["A", "B"])
            return decision, messages, True  # type: ignore[return-value]

        messages.append({"role": "assistant", "content": reply3})
        if self.verbose:
            cm.console.print(f"  [dim cyan][Turn 3 — decision][/dim cyan]\n{escape(reply3)}\n")

        decision = _parse_decision(reply3)
        if decision is None:
            cm.console.print(f"  [yellow][Could not parse '{escape(reply3)}'] — using random fallback[/yellow]")
            decision = random.choice(["A", "B"])  # type: ignore[assignment]
            return decision, messages, True  # type: ignore[return-value]

        return decision, messages, False
