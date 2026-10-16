from __future__ import annotations

from django.conf import settings
from django.db import models
from django.utils import timezone

class Side(models.TextChoices):
    WHITE = "WHITE", "White"
    BLACK = "BLACK", "Black"

class GameStatus(models.TextChoices):
    CONFIGURED = "CONFIGURED", "Configured"
    RUNNING = "RUNNING", "Running"
    PAUSED = "PAUSED", "Paused"
    FINISHED = "FINISHED", "Finished"

class TerminationReason(models.TextChoices):
    CHECKMATE = "CHECKMATE", "Checkmate"
    STALEMATE = "STALEMATE", "Stalemate"
    REPETITION = "REPETITION", "Threefold repetition"
    FIFTY_MOVE = "FIFTY_MOVE", "Fifty-move rule"
    INSUFFICIENT = "INSUFFICIENT", "Insufficient material"
    RESIGN = "RESIGN", "Forfeit / engine error"
    MAX_PLIES = "MAX_PLIES", "Max plies reached"
    UNKNOWN = "UNKNOWN", "Unknown"

class StrengthMode(models.TextChoices):
    SKILL = "SKILL", "Skill (1-20)"
    ELO = "ELO", "Elo (UCI_Elo)"

class EngineType(models.TextChoices):
    STOCKFISH = "STOCKFISH", "Stockfish (UCI)"
    PYCHESS = "PYCHESS", "PyChess (python-chess minimax)"
    GNN = "GNN", "GNN (BayesianELO)"
    PLAYER = "PLAYER", "Player"

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

class Game(models.Model):
    """State and metadata of one automated match."""

    status = models.CharField(max_length=16, choices=GameStatus.choices, default=GameStatus.CONFIGURED)
    fen = models.TextField(default=STARTPOS_FEN)

    side_to_move = models.CharField(max_length=5, default="w")  # 'w' or 'b' from FEN
    ply_count = models.PositiveIntegerField(default=0)

    move_interval_ms = models.PositiveIntegerField(default=500)
    preview_ms = models.PositiveIntegerField(default=350)
    next_action_at = models.DateTimeField(null=True, blank=True)

    last_move_uci = models.CharField(max_length=8, blank=True, default="")
    last_move_san = models.CharField(max_length=16, blank=True, default="")

    # Move preview (two-phase tick): first highlight the planned move, then apply it on the next tick.
    pending_move_uci = models.CharField(max_length=8, blank=True, default="")
    pending_move_set_at = models.DateTimeField(null=True, blank=True)

    # UI: square selections for human moves
    ui_selected_from = models.CharField(max_length=2, blank=True, default="")
    ui_selected_to = models.CharField(max_length=2, blank=True, default="")

    # Tick concurrency guard (helps avoid double-tick / sqlite lock bursts on demos)
    tick_lock = models.BooleanField(default=False)
    tick_lock_at = models.DateTimeField(null=True, blank=True)

    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    result = models.CharField(max_length=7, blank=True, default="")  # 1-0 | 0-1 | 1/2-1/2
    termination_reason = models.CharField(max_length=16, choices=TerminationReason.choices, blank=True, default="")

    initial_position_key = models.CharField(max_length=64, blank=True, default="")  # Stockfish 'Key:' for the start position.

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def mark_started(self) -> None:
        """Mark the match as running and initialize timing."""
        now = timezone.now()
        if self.started_at is None:
            self.started_at = now
        self.status = GameStatus.RUNNING
        self.next_action_at = now

    def mark_paused(self) -> None:
        """Stop the run without closing the match."""
        self.status = GameStatus.PAUSED
        # Cancel any pending preview move so a paused game never "jumps" on resume.
        self.pending_move_uci = ""
        self.pending_move_set_at = None

    def mark_finished(self, *, result: str, reason: str) -> None:
        """Close the match."""
        self.status = GameStatus.FINISHED
        # Clear preview state
        self.pending_move_uci = ""
        self.pending_move_set_at = None
        self.result = result
        self.termination_reason = reason
        self.finished_at = timezone.now()

class EngineConfig(models.Model):
    """Engine configuration for one side of a game."""

    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name="engines")
    side = models.CharField(max_length=5, choices=Side.choices)

    engine_type = models.CharField(max_length=16, choices=EngineType.choices, default=EngineType.STOCKFISH)
    strength_mode = models.CharField(max_length=8, choices=StrengthMode.choices, default=StrengthMode.SKILL)
    strength_value = models.PositiveIntegerField(default=10)  # skill 1..20 or elo value

    movetime_ms = models.PositiveIntegerField(default=150)
    uci_options = models.JSONField(blank=True, default=dict)

    # Rated participant for this side (human player or engine). Optional: unrated games leave it NULL.
    player = models.ForeignKey(
        "Player",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="engine_configs",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("game", "side")]

class Move(models.Model):
    """Record of one half-move (ply)."""

    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name="moves")
    ply_index = models.PositiveIntegerField()

    uci = models.CharField(max_length=8)
    san = models.CharField(max_length=16, blank=True, default="")

    fen_after = models.TextField()
    position_key_after = models.CharField(max_length=64, blank=True, default="")
    is_check = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("game", "ply_index")]
        ordering = ["ply_index"]

class MatchRecord(models.Model):
    """Summary row for finished matches, used by the dashboard."""

    game = models.OneToOneField(Game, on_delete=models.CASCADE, related_name="match_record")

    result = models.CharField(max_length=7)
    termination_reason = models.CharField(max_length=16)

    white_strength_mode = models.CharField(max_length=8)
    white_strength_value = models.PositiveIntegerField()
    black_strength_mode = models.CharField(max_length=8)
    black_strength_value = models.PositiveIntegerField()
    move_interval_ms = models.PositiveIntegerField()

    pgn = models.TextField(blank=True, default="")  # Filled in on game completion.

    finished_at = models.DateTimeField()

    created_at = models.DateTimeField(auto_now_add=True)


DEFAULT_RATING = 1500.0


class Player(models.Model):
    """A rated participant (human player or engine).

    The Elo rating is updated incrementally after every finished match, while the
    BayesElo value is the result of a full batch fit over all rated matches.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="player",
    )

    name = models.CharField(max_length=64, unique=True)
    is_engine = models.BooleanField(default=False)

    elo_rating = models.FloatField(default=DEFAULT_RATING)
    bayeselo_rating = models.FloatField(default=DEFAULT_RATING)

    games_played = models.PositiveIntegerField(default=0)
    wins = models.PositiveIntegerField(default=0)
    losses = models.PositiveIntegerField(default=0)
    draws = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-elo_rating", "name"]

    def __str__(self) -> str:
        return self.name

    @property
    def score(self) -> float:
        """Total score (win=1, draw=0.5)."""
        return float(self.wins) + 0.5 * float(self.draws)

    @property
    def score_rate(self) -> float:
        """Score rate (0..1); 0 if no games have been played."""
        if not self.games_played:
            return 0.0
        return self.score / float(self.games_played)


class RatingHistory(models.Model):
    """Snapshot of a player's rating after a finished match."""

    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="rating_history")
    game = models.ForeignKey(
        Game,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="rating_changes",
    )

    elo_rating = models.FloatField()
    bayeselo_rating = models.FloatField()

    elo_delta = models.FloatField(default=0.0)
    bayeselo_delta = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]
        # A game may produce at most one snapshot per player (idempotency).
        unique_together = [("player", "game")]

    def __str__(self) -> str:
        return f"{self.player.name} @ game {self.game_id}"
