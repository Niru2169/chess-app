from django.contrib import admin
from .models import Game, EngineConfig, Move, MatchRecord, Player, RatingHistory

@admin.register(Game)
class GameAdmin(admin.ModelAdmin):
    list_display = ("id","status","result","termination_reason","ply_count","move_interval_ms","started_at","finished_at")
    search_fields = ("id","result","termination_reason","status")

@admin.register(EngineConfig)
class EngineConfigAdmin(admin.ModelAdmin):
    list_display = ("id","game","side","engine_type","strength_mode","strength_value","movetime_ms")

@admin.register(Move)
class MoveAdmin(admin.ModelAdmin):
    list_display = ("id","game","ply_index","uci","is_check","created_at")

@admin.register(MatchRecord)
class MatchRecordAdmin(admin.ModelAdmin):
    list_display = ("id","game","result","termination_reason","finished_at")


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = (
        "id", "name", "is_engine", "elo_rating", "bayeselo_rating",
        "games_played", "wins", "losses", "draws",
    )
    list_filter = ("is_engine",)
    search_fields = ("name",)


@admin.register(RatingHistory)
class RatingHistoryAdmin(admin.ModelAdmin):
    list_display = ("id", "player", "game", "elo_rating", "bayeselo_rating", "elo_delta", "bayeselo_delta", "created_at")
    list_filter = ("player",)
    search_fields = ("player__name",)
