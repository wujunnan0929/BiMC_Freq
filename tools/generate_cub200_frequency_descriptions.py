"""Generate explicit low/middle/high semantic descriptions for CUB-200.

The generator intentionally uses only Python's standard library.  It talks to
an OpenAI-compatible ``/chat/completions`` endpoint, validates every batch, and
writes an atomic checkpoint after each successful response so a long CUB-200
generation job can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BANDS: Tuple[str, ...] = ("low", "middle", "high")
DEFAULT_SOURCE = Path("description/modified_cub200_output.json")
DEFAULT_OUTPUT = Path("description/cub200_frequency_descriptions.json")

SYSTEM_PROMPT = """You are an expert annotator for CLIP-based bird image recognition.

Generate visually observable, class-discriminative descriptions at three
spatial scales. Return valid JSON only.

Rules:
1. Describe only properties visible in a single photograph.
2. Do not mention habitat, behavior, diet, sound, geographic distribution,
   nesting, migration, or non-visual factual knowledge.
3. Do not use the words Fourier, low-frequency, middle-frequency, or
   high-frequency in a description.
4. Do not mention any class other than the target class.
5. Use positive descriptions; do not write "not a" or "unlike" comparisons.
6. Begin every description with "a photo of" followed by a natural article
   (a, an, or the) and the exact class name.
7. Keep cues conservative: omit uncertain attributes instead of inventing them.
8. Avoid repeating the same visual cue across low, middle, and high scales.
9. Return exactly the requested number of distinct descriptions per scale.
10. Preserve each supplied class name exactly as the JSON key and in sentences.
"""

NON_VISUAL_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bhabitat\b|\blives? in\b|\bfound in\b", "habitat"),
    (r"\bfeeds? on\b|\beats?\b|\bdiet\b|\bprey\b", "diet"),
    (r"\bnests?\b|\bmigrat\w*\b", "nesting or migration"),
    (r"\bsongs?\b|\bcalls?\b|\bsounds?\b", "sound"),
    (r"\bknown (?:for|to)\b|\boften (?:seen|found)\b", "non-visual knowledge"),
)


def normalize_name(name: str) -> str:
    """Match dataset names while preserving canonical spelling in output."""
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def load_class_names(source_path: Path) -> List[str]:
    with source_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    if not isinstance(source, dict) or not source:
        raise ValueError(f"Expected a non-empty JSON object in {source_path}.")
    class_names = list(source.keys())
    normalized = [normalize_name(name) for name in class_names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Class names become ambiguous after normalization.")
    if len(class_names) != 200:
        raise ValueError(
            f"Expected 200 CUB-200 classes in {source_path}, found {len(class_names)}."
        )
    return class_names


def build_user_prompt(class_names: Sequence[str], candidates: int, max_words: int) -> str:
    schema = {
        name: {band: ["..."] * candidates for band in BANDS}
        for name in class_names
    }
    return f"""Dataset: CUB-200-2011

Class names:
{json.dumps(list(class_names), ensure_ascii=False)}

For every class, generate these complementary visual descriptions:

low:
Properties that survive strong blur: global silhouette, overall proportions,
pose, and large color regions.

middle:
Properties visible after fine texture is removed: named body parts, their
arrangement and proportions, and medium-scale shapes or markings.

high:
Fine local properties: feather texture, thin boundaries, small markings,
spots, streaks, bars, and local color transitions.

Each description must contain at most {max_words} English words. Generate
exactly {candidates} descriptions for each of low, middle, and high. Return one
JSON object with no Markdown and exactly this shape:

{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def strip_markdown_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


def remove_trailing_json_commas(content: str) -> str:
    """Remove commas before closing braces/brackets without touching strings."""
    output = []
    in_string = False
    escaped = False
    for index, character in enumerate(content):
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            output.append(character)
            continue
        if character == ",":
            next_index = index + 1
            while next_index < len(content) and content[next_index].isspace():
                next_index += 1
            if next_index < len(content) and content[next_index] in "}]":
                continue
        output.append(character)
    return "".join(output)


def parse_response_content(content: str) -> Mapping[str, object]:
    stripped = strip_markdown_fence(content)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as original_error:
        # Some JSON-mode providers still emit a trailing comma or prose around
        # an otherwise complete object. Repair only these conservative cases.
        object_start = stripped.find("{")
        object_end = stripped.rfind("}")
        candidate = (
            stripped[object_start : object_end + 1]
            if object_start >= 0 and object_end > object_start
            else stripped
        )
        repaired = remove_trailing_json_commas(candidate)
        if repaired == stripped:
            raise original_error
        parsed = json.loads(repaired)
    if isinstance(parsed, dict) and set(parsed) == {"classes"}:
        parsed = parsed["classes"]
    if not isinstance(parsed, dict):
        raise ValueError("The model response must be a JSON object.")
    return parsed


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def validate_class_payload(
    class_name: str,
    payload: object,
    candidates: int,
    max_words: int,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return [f"{class_name}: value must be an object"]
    if set(payload) != set(BANDS):
        errors.append(
            f"{class_name}: expected bands {list(BANDS)}, found {list(payload)}"
        )
        return errors

    prefix = "a photo of "
    natural_prefixes = tuple(
        "{}{} {}".format(prefix, article, class_name).lower()
        for article in ("a", "an", "the")
    )
    all_descriptions: List[str] = []
    for band in BANDS:
        descriptions = payload[band]
        if not isinstance(descriptions, list):
            errors.append(f"{class_name}/{band}: descriptions must be a list")
            continue
        if len(descriptions) != candidates:
            errors.append(
                f"{class_name}/{band}: expected {candidates} descriptions, "
                f"found {len(descriptions)}"
            )
        normalized_descriptions = []
        for index, description in enumerate(descriptions):
            label = f"{class_name}/{band}[{index}]"
            if not isinstance(description, str) or not description.strip():
                errors.append(f"{label}: description must be a non-empty string")
                continue
            description = " ".join(description.split())
            lowered = description.lower()
            # A class name can legitimately contain a policy keyword (for
            # example, "Song Sparrow"). Remove the target name before testing
            # the remaining descriptive content for non-visual information.
            policy_text = lowered.replace(class_name.lower(), " target-class ")
            normalized_descriptions.append(lowered.rstrip("."))
            all_descriptions.append(lowered.rstrip("."))
            if not lowered.startswith(natural_prefixes):
                errors.append(
                    '{}: must start with "a photo of a/an/the {}"'.format(
                        label, class_name.lower()
                    )
                )
            word_count = count_words(description)
            if word_count > max_words:
                errors.append(
                    f"{label}: {word_count} words exceeds the limit {max_words}"
                )
            if any(term in lowered for term in ("fourier", "low-frequency", "middle-frequency", "high-frequency")):
                errors.append(f"{label}: contains forbidden frequency jargon")
            if " not a " in f" {lowered} " or "unlike" in lowered:
                errors.append(f"{label}: contains a negative/comparative description")
            for pattern, category in NON_VISUAL_PATTERNS:
                if re.search(pattern, policy_text):
                    errors.append(f"{label}: contains {category} information")
        if len(set(normalized_descriptions)) != len(normalized_descriptions):
            errors.append(f"{class_name}/{band}: contains duplicate descriptions")

    if len(set(all_descriptions)) != len(all_descriptions):
        errors.append(f"{class_name}: repeats a description across bands")
    return errors


def canonicalize_and_validate_batch(
    expected_names: Sequence[str],
    response: Mapping[str, object],
    candidates: int,
    max_words: int,
) -> Tuple[Dict[str, object], List[str]]:
    response_by_normalized = {normalize_name(name): value for name, value in response.items()}
    canonical: Dict[str, object] = {}
    errors: List[str] = []
    expected_normalized = {normalize_name(name) for name in expected_names}
    unexpected = sorted(set(response_by_normalized) - expected_normalized)
    if unexpected:
        errors.append(f"unexpected class keys: {unexpected}")

    for class_name in expected_names:
        key = normalize_name(class_name)
        if key not in response_by_normalized:
            errors.append(f"missing class key: {class_name}")
            continue
        payload = response_by_normalized[key]
        class_errors = validate_class_payload(
            class_name, payload, candidates, max_words
        )
        errors.extend(class_errors)
        if not class_errors:
            canonical[class_name] = payload
    return canonical, errors


def endpoint_url(api_base: str) -> str:
    api_base = api_base.rstrip("/")
    if api_base.endswith("/chat/completions"):
        return api_base
    return api_base + "/chat/completions"


def call_chat_completions(
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
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if reasoning_effort:
        # Reasoning models use completion tokens for both hidden reasoning and
        # the visible JSON answer. Sampling temperature is intentionally
        # omitted when reasoning is enabled.
        body["reasoning_effort"] = reasoning_effort
        body["max_completion_tokens"] = max_output_tokens
    else:
        body["temperature"] = temperature
        body["max_tokens"] = max_output_tokens
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    request = Request(
        endpoint_url(api_base),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from LLM endpoint: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach LLM endpoint: {error.reason}") from error

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Unexpected chat-completions response: {result}") from error
    if not isinstance(content, str):
        raise RuntimeError("Expected choices[0].message.content to be a string.")
    return content


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def batched(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def validate_existing_results(
    path: Path,
    class_names: Sequence[str],
    candidates: int,
    max_words: int,
) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    if not isinstance(existing, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    canonical_names = {normalize_name(name): name for name in class_names}
    results: Dict[str, object] = {}
    errors: List[str] = []
    for supplied_name, payload in existing.items():
        normalized = normalize_name(supplied_name)
        if normalized not in canonical_names:
            errors.append(f"unexpected existing class key: {supplied_name}")
            continue
        canonical_name = canonical_names[normalized]
        errors.extend(
            validate_class_payload(canonical_name, payload, candidates, max_words)
        )
        results[canonical_name] = payload
    if errors:
        raise ValueError("Existing output is invalid:\n- " + "\n- ".join(errors[:30]))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate validated CUB-200 low/middle/high CLIP descriptions."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-base", default=os.environ.get("LLM_API_BASE", ""))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument(
        "--reasoning-effort",
        choices=("", "none", "low", "medium", "high", "xhigh", "max"),
        default=os.environ.get("LLM_REASONING_EFFORT", ""),
        help="Optional reasoning effort; use xhigh for the UI's 'extremely high'.",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=22)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-json-mode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.candidates <= 0 or args.max_words <= 0:
        raise ValueError("batch-size, candidates, and max-words must be positive.")
    if args.max_retries <= 0 or args.timeout <= 0.0 or args.max_output_tokens <= 0:
        raise ValueError("max-retries, timeout, and max-output-tokens must be positive.")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("temperature must lie in [0, 2].")

    class_names = load_class_names(args.source)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive when supplied.")
        class_names = class_names[: args.limit]

    if args.dry_run:
        preview_names = class_names[: args.batch_size]
        print(SYSTEM_PROMPT)
        print(build_user_prompt(preview_names, args.candidates, args.max_words))
        return 0

    if args.overwrite and args.output.exists() and not args.validate_only:
        results: Dict[str, object] = {}
    else:
        results = validate_existing_results(
            args.output, class_names, args.candidates, args.max_words
        )

    if args.validate_only:
        missing = [name for name in class_names if name not in results]
        if missing:
            raise ValueError(
                f"Validated {len(results)} classes, but {len(missing)} are missing; "
                f"first missing class: {missing[0]}"
            )
        print(f"Validated {len(results)} CUB-200 classes in {args.output}.")
        return 0

    pending = [name for name in class_names if name not in results]
    if not pending:
        print(f"Nothing to generate; {args.output} already contains {len(results)} classes.")
        return 0
    api_key = os.environ.get(args.api_key_env, "")
    if not args.api_base or not args.model or not api_key:
        raise ValueError(
            "Generation requires --api-base, --model, and an API key in "
            f"the {args.api_key_env} environment variable."
        )

    total = len(class_names)
    for batch in batched(pending, args.batch_size):
        remaining_batch = list(batch)
        last_error: Optional[Exception] = None
        for attempt in range(1, args.max_retries + 1):
            base_prompt = build_user_prompt(
                remaining_batch, args.candidates, args.max_words
            )
            retry_note = ""
            if last_error is not None:
                retry_note = (
                    "\nThe previous response failed validation. Correct every issue and "
                    f"return the complete batch again. Error summary: {last_error}\n"
                )
            try:
                content = call_chat_completions(
                    api_base=args.api_base,
                    api_key=api_key,
                    model=args.model,
                    user_prompt=base_prompt + retry_note,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    reasoning_effort=args.reasoning_effort,
                    timeout=args.timeout,
                    json_mode=not args.disable_json_mode,
                )
                response = parse_response_content(content)
                canonical, errors = canonicalize_and_validate_batch(
                    remaining_batch, response, args.candidates, args.max_words
                )
                if canonical:
                    results.update(canonical)
                    remaining_batch = [
                        name for name in remaining_batch if name not in canonical
                    ]
                    ordered_results = {
                        name: results[name]
                        for name in class_names
                        if name in results
                    }
                    atomic_write_json(args.output, ordered_results)
                    print(
                        f"Saved {len(ordered_results)}/{total} classes to {args.output}",
                        flush=True,
                    )
                if errors and remaining_batch:
                    raise ValueError("; ".join(errors[:20]))
                if errors:
                    print(
                        "Ignoring response warnings after all requested classes "
                        "validated: " + "; ".join(errors[:5]),
                        file=sys.stderr,
                        flush=True,
                    )
                last_error = None
                break
            except (ValueError, RuntimeError, json.JSONDecodeError) as error:
                last_error = error
                print(
                    f"Batch beginning with {remaining_batch[0]!r}, attempt "
                    f"{attempt}/{args.max_retries} failed: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt < args.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 16))
        if last_error is not None:
            raise RuntimeError(
                f"Could not generate a valid batch beginning with "
                f"{remaining_batch[0]!r}. "
                f"Completed classes remain checkpointed in {args.output}."
            ) from last_error

    print(f"Completed {len(results)} CUB-200 classes in {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
