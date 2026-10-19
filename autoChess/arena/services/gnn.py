from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import chess

from django.conf import settings

DEFAULT_TOP_P = 0.9
DEFAULT_MIN_PROB = 0.001
MAX_CANDIDATES = 8


def _summarise_error(text: str, limit: int = 220) -> str:
    """Collapse a multi-line exception dump into one readable line.

    Checkpoint and tensor errors carry dozens of ``state_dict`` lines; the panel
    and the forfeit reason only need the actionable one.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "GNN unavailable"
    picked = next((line for line in lines if "size mismatch" in line), lines[0])
    return picked if len(picked) <= limit else picked[: limit - 1].rstrip() + "…"


class GnnUnavailable(RuntimeError):
    """Raised when the GNN model cannot be loaded or used."""


@dataclass(frozen=True)
class MovePrediction:
    """One candidate move from the GNN policy."""

    uci: str
    san: str
    probability: float

    @property
    def percent(self) -> float:
        return self.probability * 100.0


@dataclass(frozen=True)
class Prediction:
    """The GNN's view of a position: nucleus moves + predicted result."""

    available: bool
    top_p: float
    moves: tuple[MovePrediction, ...] = ()
    white_win: float = 0.0
    draw: float = 0.0
    black_win: float = 0.0
    error: str = ""

    @property
    def total_mass(self) -> float:
        """Combined probability of the listed candidate moves."""
        return sum(move.probability for move in self.moves)

    @property
    def white_pct(self) -> float:
        return self.white_win * 100.0

    @property
    def draw_pct(self) -> float:
        return self.draw * 100.0

    @property
    def black_pct(self) -> float:
        return self.black_win * 100.0


_lock = threading.Lock()
_state: dict = {}


def _gnn_dir() -> Path:
    return Path(str(getattr(settings, "AUTOCHESS_GNN_DIR", "") or "")).expanduser()


def _model_path() -> Path:
    configured = str(getattr(settings, "AUTOCHESS_GNN_MODEL", "") or "")
    if configured:
        return Path(configured).expanduser()
    return _gnn_dir() / "chessgnn.pt"


def _load() -> dict:
    """Import torch + the GNN modules and load the checkpoint exactly once.

    The import is done lazily (and cached, including failures) so the web app
    still runs on machines without PyTorch or without the model file.
    """
    with _lock:
        status = _state.get("status")
        if status == "ready":
            return _state
        if status == "failed":
            raise GnnUnavailable(_state.get("error", "GNN unavailable"))

        try:
            import sys

            gnn_dir = _gnn_dir()
            if not gnn_dir.is_dir():
                raise GnnUnavailable(f"GNN directory not found: {gnn_dir}")

            if str(gnn_dir) not in sys.path:
                sys.path.insert(0, str(gnn_dir))

            import torch  # noqa: PLC0415  (optional, heavy import)
            # Import `plays` first: it adds the sibling `ours_gpu_extras` folder to
            # sys.path, which is where `graph_builder` (and the other helpers) live.
            from plays import load_model, model_move, predict  # noqa: PLC0415
            from search import search_move  # noqa: PLC0415
            from graph_builder import GraphBuilder  # noqa: PLC0415

            model_file = _model_path()
            if not model_file.is_file():
                raise GnnUnavailable(f"GNN checkpoint not found: {model_file}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(str(model_file), device)
            builder = GraphBuilder(device)
        except GnnUnavailable as exc:
            _state.update(status="failed", error=str(exc))
            raise
        except Exception as exc:  # pragma: no cover - depends on optional deps
            _state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise GnnUnavailable(_state["error"]) from exc

        _state.update(
            status="ready", model=model, builder=builder,
            predict=predict, model_move=model_move, search_move=search_move,
        )
        return _state


def is_available() -> bool:
    """Whether the GNN can be loaded (used by templates/tests)."""
    try:
        _load()
    except GnnUnavailable:
        return False
    return True


def unavailable_reason() -> str:
    """Why the GNN cannot be loaded (empty string when it can).

    Handy when the checkpoint exists but no longer fits the model, e.g. after an
    architecture change: the caller can report the mismatch instead of guessing.
    """
    try:
        _load()
    except GnnUnavailable as exc:
        return str(exc)
    return ""


def reset_cache() -> None:
    """Drop the cached model/checkpoint (mainly for tests)."""
    with _lock:
        _state.clear()


def predict_position(fen: str, top_p: float | None = None) -> Prediction:
    """Predict the position: nucleus moves (top-p mass) and the game outcome.

    Never raises for expected failure modes; instead returns a ``Prediction``
    with ``available=False`` and a human-readable ``error``.
    """
    resolved_top_p = DEFAULT_TOP_P if top_p is None else max(0.0, min(1.0, float(top_p)))

    if not getattr(settings, "AUTOCHESS_GNN_ENABLED", True):
        return Prediction(False, resolved_top_p, error="GNN prediction is disabled.")

    try:
        state = _load()
    except GnnUnavailable as exc:
        return Prediction(False, resolved_top_p, error=_summarise_error(str(exc)))

    try:
        board = chess.Board(fen)
    except ValueError:
        return Prediction(False, resolved_top_p, error="Invalid position.")

    if not any(board.legal_moves):
        return Prediction(True, resolved_top_p, error="No legal moves in this position.")

    try:
        candidates, result = state["predict"](state["model"], state["builder"], board, resolved_top_p)
    except Exception as exc:  # pragma: no cover - depends on model behaviour
        return Prediction(False, resolved_top_p, error=_summarise_error(f"{type(exc).__name__}: {exc}"))

    moves = []
    for move, probability in candidates[:MAX_CANDIDATES]:
        try:
            san = board.san(move)
        except Exception:
            san = move.uci()
        moves.append(MovePrediction(move.uci(), san, float(probability)))

    return Prediction(
        True,
        resolved_top_p,
        tuple(moves),
        white_win=float(result.get("white", 0.0)),
        draw=float(result.get("draw", 0.0)),
        black_win=float(result.get("black", 0.0)),
    )


def best_move(fen: str, top_p: float | None = None, depth: int = 0,
              min_prob: float | None = None) -> str:
    """Pick a move for ``fen`` with the GNN model.

    ``depth`` selects a **minimax search** on top of the policy/value heads (the
    usual 3 / 5 / 8 settings). ``depth=0`` skips the search and samples from the
    top-p (nucleus) policy instead. ``min_prob`` is the search's width control: a
    child is kept while its cumulative path probability is at least this value.

    Raises ``GnnUnavailable`` if the model cannot be used, so callers can report
    a clear engine error (e.g. a forfeit) instead of an opaque crash.
    """
    resolved_top_p = (
        float(getattr(settings, "AUTOCHESS_GNN_TOP_P", DEFAULT_TOP_P))
        if top_p is None
        else float(top_p)
    )
    resolved_depth = max(0, int(depth or 0))
    resolved_min_prob = (
        float(getattr(settings, "AUTOCHESS_GNN_MIN_PROB", DEFAULT_MIN_PROB))
        if min_prob is None
        else float(min_prob)
    )

    if not getattr(settings, "AUTOCHESS_GNN_ENABLED", True):
        raise GnnUnavailable("GNN prediction is disabled.")

    try:
        state = _load()
    except GnnUnavailable as exc:
        raise GnnUnavailable(_summarise_error(str(exc))) from exc

    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise GnnUnavailable("Invalid position.") from exc

    if not any(board.legal_moves):
        return "(none)"

    try:
        if resolved_depth > 0:
            move = state["search_move"](
                state["model"], state["builder"], board,
                max_depth=resolved_depth, min_prob=resolved_min_prob,
            )
        else:
            move = state["model_move"](state["model"], state["builder"], board, resolved_top_p)
    except Exception as exc:  # pragma: no cover - depends on model behaviour
        raise GnnUnavailable(_summarise_error(f"{type(exc).__name__}: {exc}")) from exc
    return move.uci()
