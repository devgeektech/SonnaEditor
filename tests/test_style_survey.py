"""Tests for the Mode B style survey (Step 1 of Mode B rebuild)."""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
from pathlib import Path

import pytest

from sonna_editor import config
from sonna_editor.mode_b.survey import (
    OFFSET_MAGNITUDES,
    QUESTION_ORDER,
    QUESTION_SLIDER_MAP,
    SURVEY_SCHEMA_VERSION,
    VALID_ANSWERS,
    build_survey_payload,
    compute_offset,
    load_survey,
    parse_answers_string,
    write_survey,
)

# Load the CLI script as a module so we can invoke main() directly.
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_style_survey.py"
_spec = importlib.util.spec_from_file_location("run_style_survey", _SCRIPT_PATH)
_run_style_survey = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_run_style_survey)
_cli_main = _run_style_survey.main


ALL_KEYS = list(QUESTION_SLIDER_MAP.keys())


def _all_zero_answers() -> dict[str, int]:
    return {k: 0 for k in ALL_KEYS}


def _example_answers_string() -> str:
    return "exposure=0,temperature=1,tint=0,contrast=2,saturation=-1,shadows=1"


# --- compute_offset ---


@pytest.mark.parametrize("key", ALL_KEYS)
def test_compute_offset_zero_is_no_change(key: str) -> None:
    assert compute_offset(key, 0) == 0.0


@pytest.mark.parametrize("key", ALL_KEYS)
def test_compute_offset_at_extremes(key: str) -> None:
    mag = OFFSET_MAGNITUDES[key]
    assert compute_offset(key, +2) == pytest.approx(+mag)
    assert compute_offset(key, -2) == pytest.approx(-mag)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_compute_offset_at_half(key: str) -> None:
    mag = OFFSET_MAGNITUDES[key]
    assert compute_offset(key, +1) == pytest.approx(+mag / 2.0)
    assert compute_offset(key, -1) == pytest.approx(-mag / 2.0)


@pytest.mark.parametrize("key", ALL_KEYS)
@pytest.mark.parametrize("n", [1, 2])
def test_compute_offset_symmetric(key: str, n: int) -> None:
    assert compute_offset(key, +n) == pytest.approx(-compute_offset(key, -n))


# --- parse_answers_string ---


def test_parse_answers_happy_path() -> None:
    result = parse_answers_string(_example_answers_string())
    assert result == {
        "exposure": 0, "temperature": 1, "tint": 0,
        "contrast": 2, "saturation": -1, "shadows": 1,
    }


def test_parse_answers_whitespace_tolerated() -> None:
    s = (
        "  exposure = 0 , temperature= 1, tint =0 , "
        "contrast=2,saturation= -1 ,shadows=1  "
    )
    result = parse_answers_string(s)
    assert result["exposure"] == 0
    assert result["saturation"] == -1
    assert len(result) == 6


def test_parse_answers_signed_values() -> None:
    s = "exposure=+2,temperature=-2,tint=+1,contrast=-1,saturation=0,shadows=+0"
    result = parse_answers_string(s)
    assert result["exposure"] == 2
    assert result["temperature"] == -2
    assert result["tint"] == 1
    assert result["contrast"] == -1
    assert result["shadows"] == 0


def test_parse_answers_order_independent() -> None:
    forward = parse_answers_string(
        "exposure=1,temperature=-1,tint=2,contrast=-2,saturation=0,shadows=1"
    )
    reverse = parse_answers_string(
        "shadows=1,saturation=0,contrast=-2,tint=2,temperature=-1,exposure=1"
    )
    mixed = parse_answers_string(
        "tint=2,exposure=1,shadows=1,temperature=-1,saturation=0,contrast=-2"
    )
    assert forward == reverse == mixed


def test_parse_answers_rejects_missing_keys() -> None:
    s = "exposure=0,temperature=0,tint=0,contrast=0,saturation=0"
    with pytest.raises(ValueError, match="Missing keys"):
        parse_answers_string(s)


def test_parse_answers_rejects_unknown_key() -> None:
    s = (
        "exposure=0,temperature=0,tint=0,contrast=0,"
        "saturation=0,shadows=0,bogus=1"
    )
    with pytest.raises(ValueError, match="Unknown key"):
        parse_answers_string(s)


def test_parse_answers_rejects_out_of_range() -> None:
    s = "exposure=5,temperature=0,tint=0,contrast=0,saturation=0,shadows=0"
    with pytest.raises(ValueError, match="must be one of"):
        parse_answers_string(s)


def test_parse_answers_rejects_non_integer() -> None:
    s = "exposure=hot,temperature=0,tint=0,contrast=0,saturation=0,shadows=0"
    with pytest.raises(ValueError, match="must be an integer"):
        parse_answers_string(s)


# --- build_survey_payload ---


def test_build_payload_schema() -> None:
    payload = build_survey_payload(_all_zero_answers())
    assert payload["version"] == SURVEY_SCHEMA_VERSION
    assert isinstance(payload["created"], str)
    _dt.datetime.fromisoformat(payload["created"])  # parses as ISO 8601
    assert set(payload["questions"].keys()) == set(ALL_KEYS)
    for k in ALL_KEYS:
        entry = payload["questions"][k]
        assert set(entry.keys()) == {"slider_field", "answer", "offset"}
    assert set(payload["summary"].keys()) == {f"{k}_offset" for k in ALL_KEYS}


def test_build_payload_summary_matches_questions() -> None:
    answers = {
        "exposure": 2, "temperature": -1, "tint": 0,
        "contrast": 1, "saturation": -2, "shadows": 1,
    }
    payload = build_survey_payload(answers)
    for k in ALL_KEYS:
        assert (
            payload["summary"][f"{k}_offset"]
            == payload["questions"][k]["offset"]
        )


def test_build_payload_slider_field_matches_config() -> None:
    payload = build_survey_payload(_all_zero_answers())
    for k in ALL_KEYS:
        assert payload["questions"][k]["slider_field"] in config.SLIDER_FIELDS


def test_build_payload_all_zeros() -> None:
    payload = build_survey_payload(_all_zero_answers())
    for k in ALL_KEYS:
        assert payload["questions"][k]["offset"] == 0.0
        assert payload["summary"][f"{k}_offset"] == 0.0


# --- write_survey / load_survey ---


def test_write_survey_creates_file(tmp_path: Path) -> None:
    payload = build_survey_payload(_all_zero_answers())
    out = tmp_path / "survey.json"
    write_survey(payload, out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == SURVEY_SCHEMA_VERSION


def test_write_survey_roundtrip(tmp_path: Path) -> None:
    answers = {
        "exposure": 2, "temperature": -2, "tint": 1,
        "contrast": 0, "saturation": -1, "shadows": 2,
    }
    payload = build_survey_payload(answers)
    out = tmp_path / "survey.json"
    write_survey(payload, out)
    loaded = load_survey(out)
    assert loaded == payload


# --- CLI ---


def test_cli_non_interactive_happy_path(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "survey.json"
    rc = _cli_main([
        "--output", str(out),
        "--non-interactive",
        "--answers", _example_answers_string(),
    ])
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["version"] == SURVEY_SCHEMA_VERSION
    assert payload["questions"]["exposure"]["answer"] == 0
    assert payload["questions"]["contrast"]["answer"] == 2
    assert payload["questions"]["saturation"]["answer"] == -1


def test_cli_non_interactive_requires_answers(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "survey.json"
    with pytest.raises(SystemExit) as excinfo:
        _cli_main(["--output", str(out), "--non-interactive"])
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "answers" in captured.err.lower()


def test_cli_answers_requires_non_interactive(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "survey.json"
    with pytest.raises(SystemExit) as excinfo:
        _cli_main(["--output", str(out), "--answers", _example_answers_string()])
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "non-interactive" in captured.err.lower()


def test_cli_missing_output_parent_dir_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    bogus = tmp_path / "no_such_dir" / "survey.json"
    with pytest.raises(SystemExit) as excinfo:
        _cli_main([
            "--output", str(bogus),
            "--non-interactive",
            "--answers", _example_answers_string(),
        ])
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    err_lower = captured.err.lower()
    assert "parent" in err_lower or "directory" in err_lower
