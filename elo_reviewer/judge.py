import re
import time
from pathlib import Path
from typing import Literal

import openai
from rich.markup import escape

from . import console as cm
from .image_utils import build_image_content_block
from .prompt_loader import PromptTemplate

SYSTEM_SUFFIX = """\

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

# Fixed parsing instruction — not user-customisable so the tool can always
# extract a clear A/B verdict regardless of the evaluation template.
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
        prompt: PromptTemplate,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.prompt = prompt
        self.verbose = verbose

    def _chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> tuple[str, int]:
        kwargs: dict = {"model": self.model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                tokens = response.usage.total_tokens if response.usage else 0
                return content, tokens
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
    ) -> tuple[Literal["A", "B"] | None, list[dict], bool, int]:
        """
        Run a 3-turn comparison conversation.

        Returns:
            decision: 'A' or 'B'
            history: full message history
            used_fallback: True if parse failed and random choice was used
            tokens: total tokens consumed across all API calls this round
        """
        messages: list[dict] = [
            {"role": "system", "content": self.prompt.system + SYSTEM_SUFFIX},
        ]
        round_tokens = 0

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
            reply1, tokens1 = self._chat(messages)
            round_tokens += tokens1
        except openai.APIError as e:
            cm.log(f"  [bold red][API error turn 1: {escape(str(e))}] — skipping round[/bold red]")
            return None, messages, True, round_tokens

        messages.append({"role": "assistant", "content": reply1})
        if self.verbose:
            cm.console.print(f"\n  [dim cyan][Turn 1][/dim cyan]\n{escape(reply1)}\n")

        # Turn 2: self-review
        messages.append({"role": "user", "content": TURN2_TEXT})
        try:
            reply2, tokens2 = self._chat(messages, temperature=0.3)
            round_tokens += tokens2
        except openai.APIError as e:
            cm.log(f"  [bold red][API error turn 2: {escape(str(e))}] — skipping round[/bold red]")
            return None, messages, True, round_tokens

        messages.append({"role": "assistant", "content": reply2})
        if self.verbose:
            cm.console.print(f"  [dim cyan][Turn 2][/dim cyan]\n{escape(reply2)}\n")

        # Turn 3: fixed parsing instruction — extract final verdict
        messages.append({"role": "user", "content": TURN3_TEXT})
        try:
            reply3, tokens3 = self._chat(messages, temperature=0.0)
            round_tokens += tokens3
        except openai.APIError as e:
            cm.log(f"  [bold red][API error turn 3: {escape(str(e))}] — skipping round[/bold red]")
            return None, messages, True, round_tokens

        messages.append({"role": "assistant", "content": reply3})
        if self.verbose:
            cm.console.print(f"  [dim cyan][Turn 3 — decision][/dim cyan]\n{escape(reply3)}\n")

        decision = _parse_decision(reply3)
        if decision is None:
            cm.log(f"  [yellow][Could not parse '{escape(reply3)}'] — skipping round[/yellow]")
            return None, messages, True, round_tokens

        return decision, messages, False, round_tokens
