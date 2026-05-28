"""Mode B style survey.

Step 1 of the Mode B rebuild track (HANDOVER Part 6 item 17).

Captures the user's editing preferences across six high-variation sliders
and emits a JSON file mapping each answer to a slider offset. Step 2
(preset-to-checkpoint converter) consumes this file to set output-head
biases on the initial Mode B checkpoint.

Public surface:
- OFFSET_MAGNITUDES, QUESTION_SLIDER_MAP, QUESTION_ORDER, QUESTIONS
- compute_offset(): maps an answer (-2..+2) to a slider offset.
- parse_answers_string(): parses the CLI --answers string.
- build_survey_payload(): constructs the JSON-serialisable dict.
- write_survey() / load_survey(): file I/O.
- run_interactive(): prompts the user through the 6 questions.
- format_summary(): human-readable summary for confirmation.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Callable, Final, TextIO

from sonna_editor import config

SURVEY_SCHEMA_VERSION: Final[str] = "1.0"

# Maximum offset per question key (applied when |answer| == 2).
# Half-magnitude applied when |answer| == 1. Zero when answer == 0.
# Units match the destination slider's native units (see SLIDER_RANGES).
OFFSET_MAGNITUDES: Final[dict[str, float]] = {
    "exposure":    1.0,     # stops  — Exposure2012  range (-5.0, 5.0)
    "temperature": 1000.0,  # Kelvin — Temperature   range (2000.0, 50000.0)
    "tint":        20.0,    #        — Tint          range (-150.0, 150.0)
    "contrast":    30.0,    #        — Contrast2012  range (-100.0, 100.0)
    "saturation":  20.0,    #        — Saturation    range (-100.0, 100.0)
    "shadows":     40.0,    #        — Shadows2012   range (-100.0, 100.0)
}

# Maps the survey's question key to the SLIDER_FIELDS entry it modifies.
QUESTION_SLIDER_MAP: Final[dict[str, str]] = {
    "exposure":    "Exposure2012",
    "temperature": "Temperature",
    "tint":        "Tint",
    "contrast":    "Contrast2012",
    "saturation":  "Saturation",
    "shadows":     "Shadows2012",
}

# Display order for interactive prompt + payload construction.
QUESTION_ORDER: Final[tuple[str, ...]] = (
    "exposure", "temperature", "tint", "contrast", "saturation", "shadows",
)

VALID_ANSWERS: Final[tuple[int, ...]] = (-2, -1, 0, 1, 2)

# Module-init consistency checks.
_missing_fields = set(QUESTION_SLIDER_MAP.values()) - set(config.SLIDER_FIELDS)
assert not _missing_fields, (
    f"QUESTION_SLIDER_MAP references unknown slider fields: {_missing_fields}"
)
assert set(OFFSET_MAGNITUDES.keys()) == set(QUESTION_SLIDER_MAP.keys()), (
    "OFFSET_MAGNITUDES and QUESTION_SLIDER_MAP must share the same keys"
)
assert set(QUESTION_ORDER) == set(QUESTION_SLIDER_MAP.keys()), (
    "QUESTION_ORDER must cover all QUESTION_SLIDER_MAP keys"
)


QUESTIONS: Final[dict[str, dict]] = {
    "exposure": {
        "title": "Exposure",
        "prompt": "How bright are your edits relative to the preset baseline?",
        "options": {
            -2: "Much darker (moody, underexposed look)",
            -1: "Slightly darker",
             0: "Match the preset",
            +1: "Slightly brighter",
            +2: "Much brighter (bright, airy look)",
        },
    },
    "temperature": {
        "title": "Temperature",
        "prompt": "What's your typical white-balance bias?",
        "options": {
            -2: "Much cooler (blue-shifted, clean / editorial)",
            -1: "Slightly cooler",
             0: "Match the preset",
            +1: "Slightly warmer",
            +2: "Much warmer (golden, vintage)",
        },
    },
    "tint": {
        "title": "Tint",
        "prompt": "What's your typical green/magenta colour cast?",
        "options": {
            -2: "Strongly green-shifted (cooler greens, can flatten skin)",
            -1: "Slightly green-shifted",
             0: "Match the preset",
            +1: "Slightly magenta-shifted",
            +2: "Strongly magenta-shifted (warmer tones, lifted skin)",
        },
    },
    "contrast": {
        "title": "Contrast",
        "prompt": "Contrast feel?",
        "options": {
            -2: "Flat / soft (low contrast, film-like)",
            -1: "Slightly flat",
             0: "Match the preset",
            +1: "Slightly punchy",
            +2: "Very punchy (high-contrast, commercial)",
        },
    },
    "saturation": {
        "title": "Saturation",
        "prompt": "Colour saturation feel?",
        "options": {
            -2: "Muted / desaturated (editorial)",
            -1: "Slightly muted",
             0: "Match the preset",
            +1: "Slightly vibrant",
            +2: "Very vibrant (commercial / pop)",
        },
    },
    "shadows": {
        "title": "Shadows",
        "prompt": "Shadow handling?",
        "options": {
            -2: "Deep / crushed (dramatic, detail loss in shadows)",
            -1: "Slightly deep",
             0: "Match the preset",
            +1: "Slightly lifted",
            +2: "Strongly lifted (open shadows, airy)",
        },
    },
}
assert set(QUESTIONS.keys()) == set(QUESTION_SLIDER_MAP.keys()), (
    "QUESTIONS keys must match QUESTION_SLIDER_MAP"
)


def compute_offset(key: str, answer: int) -> float:
    """Map an answer (-2..+2) to the slider offset for `key`."""
    if key not in OFFSET_MAGNITUDES:
        raise KeyError(f"Unknown survey key: {key!r}")
    if answer not in VALID_ANSWERS:
        raise ValueError(
            f"Invalid answer {answer!r} for {key}; must be one of {VALID_ANSWERS}"
        )
    return float(answer) * (OFFSET_MAGNITUDES[key] / 2.0)


def parse_answers_string(s: str) -> dict[str, int]:
    """Parse a --answers string into a {key: answer} dict.

    Format: comma-separated key=value pairs. Whitespace tolerated around
    commas and equals. Key order is irrelevant. All 6 keys required.
    Unknown keys, duplicate keys, non-integer values, and out-of-range
    values are rejected with descriptive ValueError.
    """
    if not s or not s.strip():
        raise ValueError("--answers string is empty")

    pairs: dict[str, int] = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Expected key=value in --answers, got: {chunk!r}")
        key, _, raw = chunk.partition("=")
        key = key.strip()
        raw = raw.strip()
        if key in pairs:
            raise ValueError(f"Duplicate key in --answers: {key!r}")
        if key not in QUESTION_SLIDER_MAP:
            raise ValueError(
                f"Unknown key {key!r}; valid keys: {sorted(QUESTION_SLIDER_MAP)}"
            )
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"Value for {key!r} must be an integer, got {raw!r}"
            ) from exc
        if value not in VALID_ANSWERS:
            raise ValueError(
                f"Value for {key!r} must be one of {VALID_ANSWERS}, got {value}"
            )
        pairs[key] = value

    missing = set(QUESTION_SLIDER_MAP) - set(pairs)
    if missing:
        raise ValueError(f"Missing keys in --answers: {sorted(missing)}")

    return pairs


def build_survey_payload(
    answers: dict[str, int],
    *,
    created: _dt.datetime | None = None,
) -> dict:
    """Build the JSON-serialisable survey payload."""
    if set(answers.keys()) != set(QUESTION_SLIDER_MAP.keys()):
        raise ValueError(
            f"Answers must cover exactly {sorted(QUESTION_SLIDER_MAP)}; "
            f"got {sorted(answers)}"
        )

    if created is None:
        created = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)

    questions: dict[str, dict] = {}
    summary: dict[str, float] = {}
    for key in QUESTION_ORDER:
        answer = answers[key]
        offset = compute_offset(key, answer)
        questions[key] = {
            "slider_field": QUESTION_SLIDER_MAP[key],
            "answer": answer,
            "offset": offset,
        }
        summary[f"{key}_offset"] = offset

    return {
        "version": SURVEY_SCHEMA_VERSION,
        "created": created.isoformat(),
        "questions": questions,
        "summary": summary,
    }


def write_survey(payload: dict, path: Path) -> None:
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_survey(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- Interactive prompt ---


def _emit(line: str, stream: TextIO | None) -> None:
    if stream is not None:
        stream.write(line + "\n")
    else:
        print(line)


def _read_answer(
    key: str,
    *,
    input_fn: Callable[[str], str] = input,
    output_stream: TextIO | None = None,
) -> int:
    q = QUESTIONS[key]
    options = q["options"]

    _emit(f"  {q['prompt']}", output_stream)
    _emit("", output_stream)
    for ans in (-2, -1, 0, 1, 2):
        # Display as 1..5 to keep the visible menu conventional;
        # accept either form on input.
        menu_number = ans + 3
        _emit(f"    {menu_number}) {options[ans]}", output_stream)
    _emit("", output_stream)

    while True:
        raw = input_fn(f"  Your choice for {q['title']} [1-5 or -2..+2]: ").strip()
        if not raw:
            _emit("    (empty input; please enter 1-5 or -2..+2)", output_stream)
            continue
        try:
            n = int(raw)
        except ValueError:
            _emit(f"    Invalid input {raw!r}; expected an integer.", output_stream)
            continue
        if 1 <= n <= 5:
            return n - 3
        if n in VALID_ANSWERS:
            return n
        _emit(f"    Out of range: {n}. Use 1-5 or -2..+2.", output_stream)


def run_interactive(
    *,
    input_fn: Callable[[str], str] = input,
    output_stream: TextIO | None = None,
) -> dict[str, int]:
    _emit("Sonna Editor — Style Survey", output_stream)
    _emit("6 quick questions about your editing preferences.", output_stream)
    _emit("", output_stream)

    answers: dict[str, int] = {}
    for i, key in enumerate(QUESTION_ORDER, start=1):
        q = QUESTIONS[key]
        _emit(f"Question {i} of 6: {q['title']}", output_stream)
        answers[key] = _read_answer(
            key, input_fn=input_fn, output_stream=output_stream,
        )
        _emit("", output_stream)

    return answers


def format_summary(payload: dict) -> str:
    unit_hint = {
        "exposure":    "stops",
        "temperature": "K",
        "tint":        "",
        "contrast":    "",
        "saturation":  "",
        "shadows":     "",
    }
    lines = ["Summary:"]
    for key in QUESTION_ORDER:
        entry = payload["questions"][key]
        label = QUESTIONS[key]["options"][entry["answer"]]
        offset_str = f"{entry['offset']:+g}"
        hint = unit_hint[key]
        if hint:
            offset_str = f"{offset_str} {hint}"
        lines.append(
            f"  {QUESTIONS[key]['title']:12s} : {label}  "
            f"(offset = {offset_str} on {entry['slider_field']})"
        )
    return "\n".join(lines)
