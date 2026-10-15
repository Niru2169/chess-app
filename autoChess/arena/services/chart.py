from __future__ import annotations

"""Simple, dependency-free SVG line chart for rating history."""

from html import escape
from typing import Iterable


def rating_chart_svg(
    history: Iterable[object],
    *,
    width: int = 560,
    height: int = 190,
    padding: int = 30,
) -> str:
    """SVG line chart of the rating history (Elo + BayesElo).

    Args:
        history: RatingHistory instances in chronological order (elo_rating, bayeselo_rating).

    Returns:
        SVG markup string (timestamps in ``created_at`` are not plotted; only the order is).
    """
    points = list(history)

    outer = (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="Rating trend" '
        f'style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px">'
    )
    footer_legend = (
        f'<rect x="{width - 176}" y="8" width="12" height="12" rx="3" fill="#4f46e5"/>'
        f'<text x="{width - 158}" y="18" font-size="11" fill="#334155">Elo</text>'
        f'<rect x="{width - 96}" y="8" width="12" height="12" rx="3" fill="#0d9488"/>'
        f'<text x="{width - 78}" y="18" font-size="11" fill="#334155">BayesElo</text>'
    )
    close = "</svg>"

    if len(points) < 2:
        return (
            outer
            + footer_legend
            + f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'font-size="13" fill="#64748b">Not enough data for a chart</text>'
            + close
        )

    elo = [float(p.elo_rating) for p in points]
    bayes = [float(p.bayeselo_rating) for p in points]

    lo = min(min(elo), min(bayes))
    hi = max(max(elo), max(bayes))
    if hi - lo < 1.0:
        hi = lo + 1.0
    span = hi - lo

    inner_w = width - 2 * padding
    inner_h = height - 2 * padding
    n = len(points)

    def xy(index: int, rating: float) -> tuple[float, float]:
        x = padding + index * inner_w / (n - 1)
        y = height - padding - (rating - lo) / span * inner_h
        return x, y

    def polyline(values: list[float]) -> str:
        return " ".join(f"{xy(i, r)[0]:.1f},{xy(i, r)[1]:.1f}" for i, r in enumerate(values))

    grid = []
    for frac in (0.0, 0.5, 1.0):
        y = padding + frac * inner_h
        grid.append(
            f'<line x1="{padding}" y1="{y:.1f}" x2="{width - padding}" y2="{y:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )

    def label(y_value: float) -> str:
        return f'<text x="6" y="{y_value + 4:.1f}" font-size="10" fill="#94a3b8">{round(hi - (y_value - padding) / inner_h * span)}</text>'

    y_top = label(padding)
    y_mid = label(padding + inner_h / 2.0)
    y_bot = label(height - padding)

    return (
        outer
        + footer_legend
        + "".join(grid)
        + y_top
        + y_mid
        + y_bot
        + f'<polyline fill="none" stroke="#4f46e5" stroke-width="2" points="{polyline(elo)}"/>'
        + f'<polyline fill="none" stroke="#0d9488" stroke-width="2" points="{polyline(bayes)}"/>'
        + f'<text x="{width // 2}" y="{height - 8}" text-anchor="middle" font-size="10" '
        f'fill="#94a3b8">{escape(str(n))} matches</text>'
        + close
    )
