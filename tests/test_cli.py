"""Tests for src.cli (uses the tracked artifacts in outputs/, not data/)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cli import (
    EXAMPLE_APPLICANT,
    OUTPUT_DIR,
    display_path,
    find_in_frame,
    format_report,
    format_value,
    main,
    parse_args,
    waterfall_figure,
    waterfall_steps,
)
from src.pipeline import ApplicantResult, load_bundle, score_applicant


@pytest.fixture(scope="module")
def result() -> ApplicantResult:
    """Example applicant scored with the real scorecard."""
    return score_applicant(EXAMPLE_APPLICANT, load_bundle(OUTPUT_DIR))


def test_example_covers_all_scorecard_features() -> None:
    assert set(EXAMPLE_APPLICANT) == set(load_bundle(OUTPUT_DIR).features)


def test_parse_args_needs_exactly_one_source() -> None:
    assert parse_args(["--id", "5"]).id == 5
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--id", "5", "--template"])


def test_format_value() -> None:
    assert format_value(None) == "missing"
    assert format_value(np.nan) == "missing"
    assert format_value(450000) == "450,000"
    assert format_value(0.123456) == "0.1235"
    assert format_value("Laborers") == "Laborers"


def test_find_in_frame() -> None:
    df = pd.DataFrame({"SK_ID_CURR": [1, 2], "x": [10, 20]})
    assert find_in_frame(df, 2)["x"] == 20
    assert find_in_frame(df, 3) is None


def test_waterfall_steps_end_at_score(result: ApplicantResult) -> None:
    steps = waterfall_steps(result, max_features=5)
    assert len(steps) == 6
    assert steps["label"].iloc[-1] == f"{len(result.detail) - 5} other features"
    assert steps["start"].iloc[0] == pytest.approx(result.base_score)
    assert steps["end"].iloc[-1] == pytest.approx(result.score)
    assert np.allclose(steps["start"].iloc[1:].to_numpy(), steps["end"].iloc[:-1].to_numpy())


def test_waterfall_figure_has_one_bar_per_step(result: ApplicantResult) -> None:
    fig = waterfall_figure(result, "test", 540, 560, max_features=5)
    assert len(fig.axes[0].patches) >= 6


def test_format_report_plain(result: ApplicantResult) -> None:
    text = format_report(result, "Applicant X", color=False, notes=["note line"])
    assert "\033" not in text
    assert result.decision.upper() in text
    assert f"{result.score:.0f}" in text
    assert all(f in text for f in result.adverse.index)
    assert text.endswith("note line")


def test_main_json_prints_report_and_saves_plot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    applicant = {k: v for k, v in EXAMPLE_APPLICANT.items() if k != "OWN_CAR_AGE"} | {"TYPO_KEY": 1}
    src = tmp_path / "khach.json"
    src.write_text(json.dumps(applicant))
    png = tmp_path / "w.png"
    assert main(["--json", str(src), "--plot", str(png), "--no-color"]) == 0
    out = capsys.readouterr().out
    assert png.exists()
    assert "Decision" in out
    assert "OWN_CAR_AGE" in out and "TYPO_KEY" in out


def test_main_rejects_bad_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "bad.json"
    src.write_text(json.dumps({"EXT_SOURCE_3": "abc"}))
    assert main(["--json", str(src), "--no-plot"]) == 2
    assert "EXT_SOURCE_3 must be numeric" in capsys.readouterr().err


def test_main_template_is_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--template"]) == 0
    assert json.loads(capsys.readouterr().out) == EXAMPLE_APPLICANT


def test_display_path(tmp_path: Path) -> None:
    assert display_path(tmp_path / "a" / "b.png", tmp_path) == Path("a/b.png")
    assert display_path(Path("/elsewhere/b.png"), tmp_path) == Path("/elsewhere/b.png")


def test_format_report_hides_zero_reasons(result: ApplicantResult) -> None:
    tiny = ApplicantResult(**{**result.__dict__, "favourable": pd.Series({"REGION_RATING_CLIENT_W_CITY": 0.01})})
    text = format_report(tiny, "X")
    assert "REGION_RATING_CLIENT_W_CITY" not in text.split("raising")[1]
    assert "(none)" in text.split("raising")[1]
