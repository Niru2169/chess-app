from django.urls import path
from . import views

app_name = "arena"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("players/", views.players_view, name="players"),
    path("players/<int:player_id>/", views.player_detail_view, name="player_detail"),
    path("games/new", views.create_game, name="create_game"),
    path("games/<int:game_id>/", views.game_view, name="game"),
    path("games/<int:game_id>/control/<str:action>/", views.control, name="control"),
    path("games/<int:game_id>/tick/", views.tick_view, name="tick"),
    path("games/<int:game_id>/click-square/", views.click_square_view, name="click_square"),
    path("games/<int:game_id>/moves/", views.moves_view, name="moves"),
    path("games/<int:game_id>/config/", views.update_config, name="update_config"),
    path("games/<int:game_id>/speed/", views.update_speed, name="update_speed"),
    path("games/<int:game_id>/pgn/", views.export_game_pgn, name="export_game_pgn"),
    path("exports/pvp.pgn", views.export_pvp_pgn, name="export_pvp_pgn"),
]
