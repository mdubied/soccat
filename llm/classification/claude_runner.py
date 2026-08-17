"""
Thin wrapper around the `claude` CLI in headless mode ("claude -p ...").

Used instead of the Claude API so that classification runs draw on a Claude
Code subscription (session credits) rather than metered API credits. Shared
by classify_step1.py and (later) classify_step2.py.
"""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


def resolve_claude_executable() -> str:
    path = shutil.which("claude")
    if path is None:
        sys.exit(
            "error: 'claude' CLI not found on PATH. Install Claude Code and make sure "
            "the 'claude' command works in this terminal before running this script."
        )
    return path


@dataclass
class ClaudeResponse:
    structured_output: dict
    cost_usd: float
    input_tokens: int
    output_tokens: int
    duration_ms: int


@dataclass
class UsageTotals:
    n_calls: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0

    def add(self, r: ClaudeResponse) -> None:
        self.n_calls += 1
        self.cost_usd += r.cost_usd
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.duration_ms += r.duration_ms

    @classmethod
    def load(cls, path: Path) -> "UsageTotals":
        """Load cumulative usage totals persisted by a previous (possibly interrupted)
        run, so cost reporting stays accurate across resumed/extended runs."""
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__), encoding="utf-8")


def call_claude(
    user_message: str,
    system_prompt_file: str,
    schema: str,
    model: str,
    timeout: int,
    claude_path: str,
) -> ClaudeResponse:
    """Run one non-interactive claude -p call and return its structured output + usage.

    The user message is piped via stdin and the system prompt is loaded from a
    file (rather than passed as CLI arguments) because the installed `claude`
    CLI is a .cmd shim on Windows: subprocess routes it through cmd.exe, which
    mangles multi-line / quote-heavy arguments. --system-prompt fully replaces
    the default system prompt so Claude's coding-assistant identity doesn't
    interfere with plain classification.
    """
    cmd = [
        claude_path,
        "-p",
        "--system-prompt-file", system_prompt_file,
        "--output-format", "json",
        "--json-schema", schema,
        "--tools", "",
        "--no-session-persistence",
        "--model", model,
    ]

    proc = subprocess.run(
        cmd, input=user_message, capture_output=True, text=True, encoding="utf-8", timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()}")

    envelope = json.loads(proc.stdout)
    if envelope.get("is_error"):
        raise RuntimeError(f"claude reported an error: {envelope}")

    structured = envelope.get("structured_output")
    if structured is None:
        raise ValueError(f"no structured_output in response: {envelope.get('result')!r}")

    usage = envelope.get("usage", {})
    return ClaudeResponse(
        structured_output=structured,
        cost_usd=envelope.get("total_cost_usd", 0.0) or 0.0,
        input_tokens=usage.get("input_tokens", 0) or 0,
        output_tokens=usage.get("output_tokens", 0) or 0,
        duration_ms=envelope.get("duration_ms", 0) or 0,
    )


def write_system_prompt(tmpdir: Path, prompt_text: str) -> str:
    path = tmpdir / "system_prompt.txt"
    path.write_text(prompt_text, encoding="utf-8")
    return str(path)
