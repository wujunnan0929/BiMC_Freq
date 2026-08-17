"""Generate CUB-200 frequency descriptions with the official DeepSeek API.

This provider-specific entry point reuses the prompt construction, validation,
checkpointing, and retry loop from ``generate_cub200_frequency_descriptions``.
It keeps DeepSeek credentials in a separate environment variable so an OpenAI
key can never be sent accidentally to the DeepSeek endpoint.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

try:
    from tools import generate_cub200_frequency_descriptions as common
except ImportError:  # Direct execution puts the tools directory on sys.path.
    import generate_cub200_frequency_descriptions as common


DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_REASONING_EFFORT = "max"
DEFAULT_DEEPSEEK_OUTPUT = "description/cub200_frequency_descriptions_deepseek.json"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"


def normalize_deepseek_effort(reasoning_effort: str) -> Optional[str]:
    """Map compatible effort names to DeepSeek's current high/max values."""
    effort = reasoning_effort.lower().strip()
    if effort in ("", "none"):
        return None
    if effort in ("low", "medium", "high"):
        return "high"
    if effort in ("xhigh", "max"):
        return "max"
    raise ValueError("DeepSeek reasoning effort must map to high or max.")


def build_deepseek_request_body(
    model: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str,
    json_mode: bool,
) -> Dict[str, object]:
    """Build a DeepSeek V4 OpenAI-format Chat Completions request."""
    normalized_effort = normalize_deepseek_effort(reasoning_effort)
    body: Dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": common.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_output_tokens,
        "thinking": {
            "type": "enabled" if normalized_effort is not None else "disabled"
        },
    }
    if normalized_effort is not None:
        body["reasoning_effort"] = normalized_effort
    else:
        body["temperature"] = temperature
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def call_deepseek_chat_completions(
    api_base: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str,
    timeout: float,
    json_mode: bool,
) -> str:
    body = build_deepseek_request_body(
        model=model,
        user_prompt=user_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        json_mode=json_mode,
    )
    request = Request(
        common.endpoint_url(api_base),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "HTTP {} from DeepSeek API: {}".format(error.code, detail)
        ) from error
    except URLError as error:
        raise RuntimeError(
            "Could not reach DeepSeek API: {}".format(error.reason)
        ) from error

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "Unexpected DeepSeek chat-completions response: {}".format(result)
        ) from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "DeepSeek returned empty final content; the common retry loop will retry."
        )
    return content


def option_present(arguments: Sequence[str], option: str) -> bool:
    return any(argument == option or argument.startswith(option + "=") for argument in arguments)


def apply_deepseek_defaults(arguments: Sequence[str]) -> List[str]:
    """Add provider defaults while preserving explicit command-line values."""
    result = list(arguments)
    defaults = (
        ("--api-base", os.environ.get("DEEPSEEK_API_BASE", DEFAULT_DEEPSEEK_API_BASE)),
        ("--model", os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)),
        ("--api-key-env", DEEPSEEK_API_KEY_ENV),
        (
            "--reasoning-effort",
            os.environ.get(
                "DEEPSEEK_REASONING_EFFORT", DEFAULT_DEEPSEEK_REASONING_EFFORT
            ),
        ),
        ("--batch-size", os.environ.get("DEEPSEEK_BATCH_SIZE", "5")),
        (
            "--max-output-tokens",
            os.environ.get("DEEPSEEK_MAX_OUTPUT_TOKENS", "32768"),
        ),
        ("--max-retries", os.environ.get("DEEPSEEK_MAX_RETRIES", "8")),
        ("--timeout", os.environ.get("DEEPSEEK_TIMEOUT", "300")),
        ("--output", os.environ.get("DEEPSEEK_OUTPUT", DEFAULT_DEEPSEEK_OUTPUT)),
    )
    injected: List[str] = []
    for option, value in defaults:
        if not option_present(result, option):
            injected.extend((option, value))
    return injected + result


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    common.call_chat_completions = call_deepseek_chat_completions
    return common.main(apply_deepseek_defaults(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
