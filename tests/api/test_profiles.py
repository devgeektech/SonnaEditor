"""Tests for /api/profiles, /api/profiles/{id}/activate, /api/profiles/lite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sonna_editor.api.routes import profiles as profiles_route


def _make_ckpt(dir_: Path, name: str, sidecar: dict | None = None) -> Path:
    p = dir_ / name
    p.write_bytes(b"fake-checkpoint-bytes")
    if sidecar is not None:
        p.with_suffix(".json").write_text(json.dumps(sidecar))
    return p


def _make_preset(path: Path) -> Path:
    """Write a minimal .xmp file usable by the Lite-creation route. The route
    only validates extension + existence; the fake builder doesn't read it."""
    path.write_text("<x:xmpmeta xmlns:x='adobe:ns:meta/'/>")
    return path


def _valid_survey_answers() -> dict[str, int]:
    return {
        "exposure": 1, "temperature": -1, "tint": 0,
        "contrast": 2, "saturation": -2, "shadows": 0,
    }


@pytest.fixture
def fake_mode_b_builder(monkeypatch: pytest.MonkeyPatch):
    """Replace build_mode_b_checkpoint with a stub that writes a real-looking
    ckpt + sidecar without invoking torch. Keeps route tests fast and isolated
    from the model machinery (which has its own test coverage)."""
    calls: list[dict] = []

    def _fake(
        *, preset_path, survey_path, base_ckpt_path, output_ckpt_path,
        profile_name, profile_id=None, skip_verification=False,
    ):
        calls.append({
            "preset_path": Path(preset_path),
            "survey_path": Path(survey_path),
            "base_ckpt_path": Path(base_ckpt_path),
            "output_ckpt_path": Path(output_ckpt_path),
            "profile_name": profile_name,
        })
        output_ckpt_path = Path(output_ckpt_path)
        output_ckpt_path.write_bytes(b"fake-mode-b-ckpt-bytes")
        sidecar_path = output_ckpt_path.with_suffix(".json")
        pid = profile_id or f"mode-b-{profile_name.replace(' ', '-').lower()}-test"
        sidecar_path.write_text(json.dumps({
            "profile_type": "mode_b_initial",
            "profile_id": pid,
            "display_name": profile_name,
            "resolution": 256,
            "base_checkpoint": str(base_ckpt_path),
            "default_skip_fields": ["Tint"],
            "date_iso": "2026-05-14T12:00:00+00:00",
        }, indent=2))
        return sidecar_path

    monkeypatch.setattr(
        profiles_route.mode_b_builder, "build_mode_b_checkpoint", _fake,
    )
    return calls


def test_profiles_returns_v101_checkpoint(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.0.1.ckpt")

    resp = client.get("/api/profiles")
    assert resp.status_code == 200
    profiles = resp.json()
    assert len(profiles) == 1

    p = profiles[0]
    assert p["id"] == "dp-event-v1.0.1"
    assert p["version"] == "v1.0.1"
    assert p["checkpoint_path"].endswith("model-v1.0.1.ckpt")
    assert p["trained_at"] is not None
    assert p["is_active"] is True  # only one profile, defaults to active


def test_profiles_uses_sidecar_when_present(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.0.2.ckpt", sidecar={
        "version": "model-v1.0.2",
        "date_iso": "2026-05-09T12:00:00+00:00",
        "ft_val_loss": 0.00098,
        "n_capture_rows": 100,
        "n_original_rows": 12857,
    })

    resp = client.get("/api/profiles")
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["val_loss"] == 0.00098
    assert payload[0]["photo_count"] == 12957
    assert payload[0]["trained_at"] == "2026-05-09T12:00:00+00:00"


def test_profile_type_none_when_sidecar_missing_field(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """Mode A v1.2.3 production sidecar predates `profile_type` — must surface as None."""
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.2.3.ckpt", sidecar={
        "version": "model-v1.2.3",
        "date_iso": "2026-05-09T12:00:00+00:00",
        "ft_val_loss": 0.00098,
    })

    payload = client.get("/api/profiles").json()
    assert len(payload) == 1
    assert payload[0]["profile_type"] is None


def test_profile_type_mode_b_initial_flows_through(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """Mode B initial ckpt sidecar carries `profile_type` → Profile surfaces it verbatim."""
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.3.0.ckpt", sidecar={
        "version": "model-v1.3.0",
        "date_iso": "2026-05-14T10:00:00+00:00",
        "profile_type": "mode_b_initial",
    })

    payload = client.get("/api/profiles").json()
    assert len(payload) == 1
    assert payload[0]["profile_type"] == "mode_b_initial"


def test_profiles_legacy_fallback_when_sidecar_missing_id_and_name(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """Mode A v1.x production sidecars predate profile_id/display_name —
    discovery must fall back to dp-event-{version} and the LEGACY_PROFILE_NAME_FALLBACK label."""
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.2.3.ckpt", sidecar={
        "date_iso": "2026-04-01T12:00:00+00:00",
        "ft_val_loss": 0.0008,
    })

    payload = client.get("/api/profiles").json()
    assert len(payload) == 1
    p = payload[0]
    assert p["id"] == "dp-event-v1.2.3"
    assert p["name"] == "DP Event"
    assert p["version"] == "v1.2.3"
    # No display_name in sidecar → field stays None; frontend renders
    # "{name} {version}".
    assert p["display_name"] is None


def test_profiles_uses_sidecar_profile_id_and_display_name(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """Mode B ckpt sidecar carries profile_id + display_name → Profile uses both."""
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v0.1.0.ckpt", sidecar={
        "profile_type": "mode_b_initial",
        "profile_id": "mode-b-wedding-lite-20260514-1102",
        "display_name": "Mode B — Wedding Lite",
        "date_iso": "2026-05-14T11:02:00+00:00",
    })

    payload = client.get("/api/profiles").json()
    assert len(payload) == 1
    p = payload[0]
    assert p["id"] == "mode-b-wedding-lite-20260514-1102"
    assert p["name"] == "Mode B — Wedding Lite"
    assert p["display_name"] == "Mode B — Wedding Lite"
    assert p["version"] == "v0.1.0"
    assert p["profile_type"] == "mode_b_initial"


def test_activate_sidecar_derived_id_round_trips(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """Activating a Mode B profile by its sidecar-derived id must persist and re-read."""
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v0.1.0.ckpt", sidecar={
        "profile_type": "mode_b_initial",
        "profile_id": "mode-b-lite-001",
        "display_name": "Lite #1",
    })
    _make_ckpt(ckpts, "model-v1.2.3.ckpt")  # legacy Mode A, no sidecar

    resp = client.post("/api/profiles/mode-b-lite-001/activate")
    assert resp.status_code == 200
    assert resp.json()["id"] == "mode-b-lite-001"
    assert resp.json()["is_active"] is True

    listing = client.get("/api/profiles").json()
    by_id = {p["id"]: p for p in listing}
    assert by_id["mode-b-lite-001"]["is_active"] is True
    assert by_id["dp-event-v1.2.3"]["is_active"] is False


def test_profiles_empty_when_dir_empty(client: TestClient) -> None:
    resp = client.get("/api/profiles")
    assert resp.status_code == 200
    assert resp.json() == []


def test_profiles_ignores_lightning_intermediates(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.0.1.ckpt")
    # These naming patterns must NOT appear in /api/profiles
    _make_ckpt(ckpts, "epoch=017-val_loss=0.0010.ckpt")
    _make_ckpt(ckpts, "sonna-v1.0.1-epoch017-val0.0010.ckpt")

    profiles = client.get("/api/profiles").json()
    assert len(profiles) == 1
    assert profiles[0]["id"] == "dp-event-v1.0.1"


def test_activate_writes_file_and_returns_profile(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.0.1.ckpt")
    _make_ckpt(ckpts, "model-v1.0.2.ckpt")

    resp = client.post("/api/profiles/dp-event-v1.0.1/activate")
    assert resp.status_code == 200
    p = resp.json()
    assert p["id"] == "dp-event-v1.0.1"
    assert p["is_active"] is True

    active_file = isolated_paths["saha_dir"] / "active_profile.txt"
    assert active_file.read_text().strip() == "dp-event-v1.0.1"

    # And the listing now reflects the explicit activation
    listing = client.get("/api/profiles").json()
    by_id = {p["id"]: p for p in listing}
    assert by_id["dp-event-v1.0.1"]["is_active"] is True
    assert by_id["dp-event-v1.0.2"]["is_active"] is False


def test_activate_unknown_id_returns_404(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    resp = client.post("/api/profiles/dp-event-v9.9.9/activate")
    assert resp.status_code == 404


def test_delete_profile_removes_generated_files(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    ckpts = isolated_paths["checkpoints_dir"]
    older_ckpt = _make_ckpt(
        ckpts,
        "model-v1.2.2.ckpt",
        sidecar={"profile_id": "dp-event-v1.2.2"},
    )
    _make_ckpt(ckpts, "model-v1.2.3.ckpt")
    sidecar = older_ckpt.with_suffix(".json")
    survey = ckpts / "model-v1.2.2-survey.json"
    preset = ckpts / "model-v1.2.2-preset.xmp"
    survey.write_text("survey")
    preset.write_text("preset")

    resp = client.delete("/api/profiles/dp-event-v1.2.2")

    assert resp.status_code == 200
    body = resp.json()
    assert body["profile_id"] == "dp-event-v1.2.2"
    assert body["deleted_paths"] == [
        str(older_ckpt),
        str(sidecar),
        str(survey),
        str(preset),
    ]
    assert not older_ckpt.exists()
    assert not sidecar.exists()
    assert not survey.exists()
    assert not preset.exists()


def test_delete_active_profile_rejects(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.2.3.ckpt")

    resp = client.delete("/api/profiles/dp-event-v1.2.3")

    assert resp.status_code == 409
    assert "Deactivate this profile first" in resp.json()["detail"]


# ── /api/profiles/lite ──────────────────────────────────────────────────────

def test_lite_create_happy_path(
    client: TestClient, isolated_paths: dict[str, Path],
    tmp_path: Path, fake_mode_b_builder,
) -> None:
    """Active Mode A profile + valid preset + valid survey → new Mode B ckpt
    appears in /api/profiles with the Lite badge fields."""
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.2.3.ckpt")  # Mode A, auto-activated as only profile
    preset = _make_preset(tmp_path / "wedding.xmp")

    resp = client.post("/api/profiles/lite", json={
        "profile_name": "Wedding Lite",
        "preset_path": str(preset),
        "survey_answers": _valid_survey_answers(),
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["profile_id"] == "mode-b-wedding-lite-test"
    assert body["ckpt_path"].endswith("model-v0.1.0.ckpt")
    assert body["sidecar_path"].endswith("model-v0.1.0.json")

    # Builder was invoked with the copied preset + survey, not the original.
    assert len(fake_mode_b_builder) == 1
    invocation = fake_mode_b_builder[0]
    assert invocation["preset_path"].parent == ckpts
    assert invocation["survey_path"].parent == ckpts
    assert invocation["base_ckpt_path"].name == "model-v1.2.3.ckpt"
    assert invocation["profile_name"] == "Wedding Lite"

    # Survey + preset were persisted under CHECKPOINTS_DIR.
    assert (ckpts / "model-v0.1.0-survey.json").exists()
    assert (ckpts / "model-v0.1.0-preset.xmp").exists()

    # And the new profile is now discoverable via /api/profiles.
    listing = client.get("/api/profiles").json()
    by_id = {p["id"]: p for p in listing}
    assert "mode-b-wedding-lite-test" in by_id
    assert by_id["mode-b-wedding-lite-test"]["profile_type"] == "mode_b_initial"
    assert by_id["mode-b-wedding-lite-test"]["display_name"] == "Wedding Lite"


def test_lite_create_rejects_when_no_active_personal_ai(
    client: TestClient, isolated_paths: dict[str, Path],
    tmp_path: Path, fake_mode_b_builder,
) -> None:
    """No profiles at all → guard fires with a clear error message."""
    preset = _make_preset(tmp_path / "any.xmp")
    resp = client.post("/api/profiles/lite", json={
        "profile_name": "X",
        "preset_path": str(preset),
        "survey_answers": _valid_survey_answers(),
    })
    assert resp.status_code == 400
    assert "Personal AI profile" in resp.json()["detail"]
    assert len(fake_mode_b_builder) == 0  # builder never invoked


def test_lite_create_rejects_when_only_mode_b_active(
    client: TestClient, isolated_paths: dict[str, Path],
    tmp_path: Path, fake_mode_b_builder,
) -> None:
    """Active profile is itself a Mode B initial → guard rejects."""
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v0.1.0.ckpt", sidecar={
        "profile_type": "mode_b_initial",
        "profile_id": "mode-b-existing",
        "display_name": "Existing Lite",
    })
    preset = _make_preset(tmp_path / "any.xmp")

    resp = client.post("/api/profiles/lite", json={
        "profile_name": "Derived Lite",
        "preset_path": str(preset),
        "survey_answers": _valid_survey_answers(),
    })
    assert resp.status_code == 400
    assert "Personal AI profile" in resp.json()["detail"]
    assert len(fake_mode_b_builder) == 0


def test_lite_create_counter_increments_past_existing(
    client: TestClient, isolated_paths: dict[str, Path],
    tmp_path: Path, fake_mode_b_builder,
) -> None:
    """Existing v0.5.0 ckpt → next Lite create produces v0.6.0."""
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.2.3.ckpt")  # Mode A active
    _make_ckpt(ckpts, "model-v0.2.0.ckpt", sidecar={
        "profile_type": "mode_b_initial",
        "profile_id": "mode-b-old-1",
        "display_name": "Old Lite 1",
    })
    _make_ckpt(ckpts, "model-v0.5.0.ckpt", sidecar={
        "profile_type": "mode_b_initial",
        "profile_id": "mode-b-old-2",
        "display_name": "Old Lite 2",
    })

    # Ensure Mode A v1.2.3 is the active one (it is by auto-pick since
    # trained_at falls back to file mtime and the test created it most
    # recently among Mode A ckpts — only Mode A is a candidate for active
    # under the guard anyway).
    client.post("/api/profiles/dp-event-v1.2.3/activate")

    preset = _make_preset(tmp_path / "next.xmp")
    resp = client.post("/api/profiles/lite", json={
        "profile_name": "Next Lite",
        "preset_path": str(preset),
        "survey_answers": _valid_survey_answers(),
    })
    assert resp.status_code == 200
    assert resp.json()["ckpt_path"].endswith("model-v0.6.0.ckpt")


def test_lite_create_rejects_invalid_survey(
    client: TestClient, isolated_paths: dict[str, Path],
    tmp_path: Path, fake_mode_b_builder,
) -> None:
    """Survey missing keys / out-of-range value → 400 before builder runs."""
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.2.3.ckpt")
    preset = _make_preset(tmp_path / "p.xmp")

    # Missing 'shadows'.
    bad = {k: 0 for k in ("exposure", "temperature", "tint", "contrast", "saturation")}
    resp = client.post("/api/profiles/lite", json={
        "profile_name": "X",
        "preset_path": str(preset),
        "survey_answers": bad,
    })
    assert resp.status_code == 400
    assert "shadows" in resp.json()["detail"]

    # Out of range.
    bad2 = _valid_survey_answers() | {"exposure": 5}
    resp2 = client.post("/api/profiles/lite", json={
        "profile_name": "X",
        "preset_path": str(preset),
        "survey_answers": bad2,
    })
    assert resp2.status_code == 400
    assert "exposure" in resp2.json()["detail"]

    assert len(fake_mode_b_builder) == 0


def test_lite_create_rejects_non_xmp_preset(
    client: TestClient, isolated_paths: dict[str, Path],
    tmp_path: Path, fake_mode_b_builder,
) -> None:
    """Preset must be .xmp; other extensions rejected at the route."""
    ckpts = isolated_paths["checkpoints_dir"]
    _make_ckpt(ckpts, "model-v1.2.3.ckpt")
    not_preset = tmp_path / "preset.txt"
    not_preset.write_text("nope")

    resp = client.post("/api/profiles/lite", json={
        "profile_name": "X",
        "preset_path": str(not_preset),
        "survey_answers": _valid_survey_answers(),
    })
    assert resp.status_code == 400
    assert ".xmp" in resp.json()["detail"]
    assert len(fake_mode_b_builder) == 0
