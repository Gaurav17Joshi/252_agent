"""
utils/shared.py — Shared utilities used across both pipeline stages.

Provides:
  - run_dir()          : create / return a timestamped workspace directory
  - save_state()       : write any dict to <run_dir>/<name>.json
  - load_state()       : read back a JSON state file
  - get_logger()       : consistent logger factory
  - ClaudeAgent        : thin base class wrapping the Anthropic SDK tool-use loop
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, cfg.LOG_LEVEL, logging.INFO))
    return logger


_log = get_logger("utils.shared")


# ── Workspace ─────────────────────────────────────────────────────────────────

def run_dir(run_id: str | None = None) -> Path:
    """Return (and create) a workspace directory for this pipeline run."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = cfg.WORKSPACE_ROOT / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_state(directory: Path, name: str, data: dict[str, Any]) -> Path:
    """Serialize *data* to <directory>/<name>.json and return the path."""
    path = directory / f"{name}.json"
    # FIX: create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    _log.debug("Saved state → %s", path)
    return path


def load_state(directory: Path, name: str) -> dict[str, Any]:
    """Load and return a previously saved JSON state file."""
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    with open(path) as fh:
        return json.load(fh)


# ── Claude Agent base ─────────────────────────────────────────────────────────

class ClaudeAgent:
    """
    Thin wrapper around the Anthropic Messages API that implements the
    tool-use agentic loop:

        1. Send messages + tool definitions to Claude.
        2. If Claude returns tool_use blocks, invoke the matching Python
           function and feed results back.
        3. Repeat until Claude returns a plain text response (no tool calls).

    Subclasses register tools via ``self.tools`` (list[dict]) and
    ``self.tool_handlers`` (dict[str, callable]).
    """

    name: str = "base_agent"

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(
            api_key=cfg.ANTHROPIC_API_KEY,
            base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        )
        self.tools: list[dict] = []
        self.tool_handlers: dict[str, Any] = {}
        self.logger = get_logger(f"agent.{self.name}")

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, system: str, user_message) -> str:
        """
        Execute the agentic tool-use loop and return the final text response.

        user_message can be str (plain text) or list (multimodal with image+text).
        """
        messages: list[dict] = [{"role": "user", "content": user_message}]
        self.logger.info("Starting agent loop (%s)", self.name)

        while True:
            kwargs: dict[str, Any] = dict(
                model=cfg.CLAUDE_MODEL,
                max_tokens=cfg.MAX_TOKENS,
                system=system,
                messages=messages,
            )
            if self.tools:
                kwargs["tools"] = self.tools

            response = self.client.messages.create(**kwargs)
            self.logger.debug("Stop reason: %s", response.stop_reason)

            # FIX: guard against None content (empty response from gateway)
            content = response.content or []

            # Collect tool_use blocks (if any)
            tool_uses = [b for b in content if b.type == "tool_use"]

            if not tool_uses:
                # Claude finished — extract text
                text_blocks = [b for b in content if b.type == "text"]
                final = "\n".join(b.text for b in text_blocks).strip()

                # FIX: if no text blocks either, fall back to stored tool results
                if not final:
                    self.logger.warning(
                        "Agent returned empty final response — "
                        "no text blocks in content: %s", content
                    )

                self.logger.info("Agent finished (%s)", self.name)
                return final

            # Append Claude's response turn
            messages.append({"role": "assistant", "content": content})

            # Execute each tool call and collect results
            tool_results = []
            for tu in tool_uses:
                self.logger.info("Tool call: %s(%s)", tu.name, list(tu.input.keys()))
                handler = self.tool_handlers.get(tu.name)
                if handler is None:
                    result_content = f"ERROR: unknown tool '{tu.name}'"
                    self.logger.error(result_content)
                else:
                    try:
                        result = handler(**tu.input)
                        result_content = json.dumps(result, default=str)
                    except Exception as exc:  # noqa: BLE001
                        result_content = f"ERROR: {exc}"
                        self.logger.exception("Tool %s raised", tu.name)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_content,
                })

            # Feed results back as the next user turn
            messages.append({"role": "user", "content": tool_results})