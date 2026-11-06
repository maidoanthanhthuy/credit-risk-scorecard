"""Command-line scoring of one applicant with the champion scorecard.

Run from the project root (works in any shell, including fish):

    python -m src.cli --id 100002              # applicant from data/test.pkl or data/train.pkl
    python -m src.cli --template > khach.json  # example applicant to edit
    python -m src.cli --json khach.json        # new applicant

Prints the score, PD, decision and main reasons, and saves a waterfall chart
of the score (``outputs/cli/applicant_<name>.png`` by default).

The JSON holds one applicant in the cleaned format (see :mod:`src.pipeline`):
ages and durations in years, categories as labels; ``null`` or an absent key
means missing.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.pipeline import ApplicantResult, load_bundle, missing_features, score_applicant

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
ID_COL = "SK_ID_CURR"

EXAMPLE_APPLICANT: dict[str, object] = {
    "EXT_SOURCE_1": 0.45,
    "EXT_SOURCE_2": 0.55,
    "EXT_SOURCE_3": 0.40,
    "YEARS_EMPLOYED": 3.5,
    "OCCUPATION_TYPE": "Laborers",
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    "REGION_RATING_CLIENT_W_CITY": 2,
    "AMT_GOODS_PRICE": 450000,
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "YEARS_LAST_PHONE_CHANGE": 2.5,
    "FLOORSMAX_AVG": None,
    "YEARS_ID_PUBLISH": 8.0,
    "FLAG_DOCUMENT_3": 1,
    "YEARS_REGISTRATION": 12.0,
    "REG_CITY_NOT_LIVE_CITY": 0,
    "OWN_CAR_AGE": None,
}

# + points (safer) / - points (riskier), decision colors as in 13_cutoff_strategy
BLUE, ORANGE = "#2a78d6", "#eb6834"
DECISION_COLORS = {"Reject": "#eb6834", "Review": "#eda100", "Approve": "#1baf7a"}
MUTED, INK2 = "#8a8985", "#52514e"
MIN_REASON_POINTS = 0.05  # contributions that round to 0.0 are not reasons
ANSI = {"Approve": "\033[32m", "Review": "\033[33m", "Reject": "\033[31m", "bold": "\033[1m", "reset": "\033[0m"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        Parsed arguments; exactly one of ``id``, ``json``, ``template`` is set.
    """
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Score one applicant with the credit scorecard.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--id", type=int, help=f"{ID_COL} of an applicant in data/test.pkl or data/train.pkl")
    source.add_argument("--json", type=Path, help="JSON file with one applicant (cleaned format)")
    source.add_argument("--template", action="store_true", help="print an example applicant as JSON and exit")
    parser.add_argument("--top", type=int, default=4, help="number of reasons to show (default 4)")
    parser.add_argument("--plot", type=Path, help="PNG path for the waterfall (default outputs/cli/applicant_<name>.png)")
    parser.add_argument("--no-plot", action="store_true", help="do not save the waterfall chart")
    parser.add_argument("--no-color", action="store_true", help="plain text output")
    return parser.parse_args(argv)


def format_value(value: object) -> str:
    """Readable feature value.

    Args:
        value: Raw feature value.

    Returns:
        ``"missing"`` for null values, thousands separators for large
        numbers, 4 significant digits for other numbers, ``str`` otherwise.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"
    return str(value)


def display_path(path: Path, cwd: Path) -> Path:
    """Path relative to ``cwd`` when it lies inside it, else unchanged.

    Args:
        path: File path.
        cwd: Current working directory.

    Returns:
        Shorter path for printing.
    """
    try:
        return path.resolve().relative_to(cwd.resolve())
    except ValueError:
        return path


def find_in_frame(df: pd.DataFrame, applicant_id: int, id_col: str = ID_COL) -> pd.Series | None:
    """Row of one applicant.

    Args:
        df: Applicants with an ID column.
        applicant_id: ID to look for.
        id_col: Name of the ID column.

    Returns:
        The applicant's row, or ``None`` if the ID is absent.
    """
    match = df[df[id_col] == applicant_id]
    return None if match.empty else match.iloc[0]


def load_applicant_by_id(applicant_id: int, data_dir: Path = DATA_DIR) -> tuple[pd.Series, str]:
    """Look an applicant up in the cleaned test set, then the train set.

    Args:
        applicant_id: ``SK_ID_CURR`` to look for.
        data_dir: Folder with ``test.pkl`` and ``train.pkl``.

    Returns:
        ``(row, set_name)`` with ``set_name`` ``"test"`` or ``"train"``.

    Raises:
        KeyError: If the ID is in neither set.
    """
    for name in ("test", "train"):
        row = find_in_frame(pd.read_pickle(data_dir / f"{name}.pkl"), applicant_id)
        if row is not None:
            return row, name
    raise KeyError(f"{ID_COL} {applicant_id} not found in test.pkl or train.pkl")


def format_report(result: ApplicantResult, title: str, color: bool = False, notes: list[str] | None = None) -> str:
    """Text report of one scored applicant.

    Args:
        result: Output of :func:`src.pipeline.score_applicant`.
        title: First line (who was scored).
        color: Use ANSI colors for the decision.
        notes: Extra lines printed at the end (warnings, file paths).

    Returns:
        Multi-line report.
    """
    def paint(text: str, key: str) -> str:
        return f"{ANSI[key]}{text}{ANSI['reset']}" if color else text

    def reason_lines(reasons: pd.Series) -> list[str]:
        reasons = reasons[reasons.abs() >= MIN_REASON_POINTS]
        if reasons.empty:
            return ["  (none)"]
        labels = [f"{f} = {format_value(result.detail.loc[f, 'value'])}" for f in reasons.index]
        width = max(len(label) for label in labels)
        return [
            f"  {i}. {label:<{width}}  {pts:+6.1f} pts"
            for i, (label, pts) in enumerate(zip(labels, reasons), start=1)
        ]

    lines = [
        paint(title, "bold"),
        "-" * max(len(title), 40),
        f"Score     {result.score:.0f}   (average applicant {result.base_score:.0f})",
        f"PD        {result.pd:.1%}",
        f"Decision  {paint(result.decision.upper(), result.decision)}",
        "",
        "Main reasons lowering the score (vs average applicant):",
        *reason_lines(result.adverse),
        "",
        "Main reasons raising the score:",
        *reason_lines(result.favourable),
    ]
    if notes:
        lines += ["", *notes]
    return "\n".join(lines)


def waterfall_steps(result: ApplicantResult, max_features: int = 8) -> pd.DataFrame:
    """Steps of the score waterfall: average applicant -> this applicant.

    Args:
        result: Output of :func:`src.pipeline.score_applicant`.
        max_features: Largest contributions shown one by one; the rest are
            lumped into one step.

    Returns:
        DataFrame with ``label``, ``contribution``, ``start`` and ``end``
        (score before / after the step); the last ``end`` is the score.
    """
    contrib = result.detail["contribution"]
    order = contrib.abs().sort_values(ascending=False).index
    shown, rest = order[:max_features], order[max_features:]
    labels = [f"{f} = {format_value(result.detail.loc[f, 'value'])}" for f in shown]
    values = contrib[shown].tolist()
    if len(rest):
        labels.append(f"{len(rest)} other features")
        values.append(float(contrib[rest].sum()))
    end = result.base_score + np.cumsum(values)
    return pd.DataFrame({"label": labels, "contribution": values, "start": end - np.array(values), "end": end})


def waterfall_figure(
    result: ApplicantResult, title: str, reject_below: float, approve_from: float, max_features: int = 8
) -> Figure:
    """Waterfall chart of the score with the decision bands shaded.

    Args:
        result: Output of :func:`src.pipeline.score_applicant`.
        title: Chart title.
        reject_below: Reject threshold (band edge).
        approve_from: Approve threshold (band edge).
        max_features: See :func:`waterfall_steps`.

    Returns:
        Matplotlib figure (not attached to pyplot; save with ``savefig``).
    """
    steps = waterfall_steps(result, max_features)
    lo = min(steps["start"].min(), steps["end"].min(), reject_below)
    hi = max(steps["start"].max(), steps["end"].max(), approve_from)
    pad = 0.15 * (hi - lo)
    x0, x1 = lo - pad, hi + pad

    fig = Figure(figsize=(8, 0.45 * len(steps) + 1.8))
    ax = fig.subplots()
    bands = {"Reject": (x0, reject_below), "Review": (reject_below, approve_from), "Approve": (approve_from, x1)}
    for band, (a, b) in bands.items():
        if b > a:
            ax.axvspan(a, b, color=DECISION_COLORS[band], alpha=0.12, linewidth=0)
            # Band name above the plot area, in the band's color
            ax.text((a + b) / 2, 1.01, band, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=DECISION_COLORS[band])
    for y, row in enumerate(steps.itertuples()):
        v = row.contribution
        ax.barh(y, v, left=row.start, color=BLUE if v >= 0 else ORANGE, height=0.7)
        ax.text(row.end + (0.01 if v >= 0 else -0.01) * (x1 - x0), y, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    ax.axvline(result.base_score, color=MUTED, linewidth=1, linestyle="--",
               label=f"Average applicant ({result.base_score:.0f})")
    ax.axvline(result.score, color=INK2, linewidth=1.5, label=f"This applicant ({result.score:.0f})")
    ax.set_xlim(x0, x1)
    ax.set_yticks(range(len(steps)), steps["label"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Credit score (Reject < {reject_below:g} <= Review < {approve_from:g} <= Approve)")
    ax.set_title(f"{title}\nscore {result.score:.0f}: {result.decision}", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    """Entry point: load, score, print, save the chart.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        Exit code: 0 on success, 2 on bad input.
    """
    args = parse_args(argv)
    if args.template:
        print(json.dumps(EXAMPLE_APPLICANT, indent=2, ensure_ascii=False))
        return 0

    bundle = load_bundle(OUTPUT_DIR)
    notes: list[str] = []
    try:
        if args.id is not None:
            applicant, set_name = load_applicant_by_id(args.id)
            title, name = f"Applicant {args.id} ({set_name} set)", str(args.id)
            if "TARGET" in applicant.index:
                notes.append(f"Actual outcome: {'default' if applicant['TARGET'] == 1 else 'repaid'}")
        else:
            applicant = json.loads(args.json.read_text(encoding="utf-8"))
            if not isinstance(applicant, dict):
                raise ValueError("the JSON file must hold one object {...}")
            title, name = f"Applicant from {args.json.name}", args.json.stem
            absent = missing_features(pd.DataFrame([applicant]), bundle.features)
            unknown = [k for k in applicant if k not in bundle.features]
            if absent:
                notes.append(f"Not in input (scored as Missing): {', '.join(absent)}")
            if unknown:
                notes.append(f"Ignored keys (not scorecard features): {', '.join(unknown)}")
        result = score_applicant(applicant, bundle, k=args.top)
    except (KeyError, ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    if not args.no_plot:
        path = args.plot or OUTPUT_DIR / "cli" / f"applicant_{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        waterfall_figure(result, title, bundle.reject_below, bundle.approve_from).savefig(path, dpi=150, bbox_inches="tight")
        notes.append(f"Waterfall saved: {display_path(path, Path.cwd())}")

    print(format_report(result, title, color=not args.no_color and sys.stdout.isatty(), notes=notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
