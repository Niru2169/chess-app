from __future__ import annotations

from django import forms

from .models import Side, StrengthMode, EngineType, GNN_DEPTH_CHOICES


class NewGameForm(forms.Form):
    """Create a new match with simple UI parameters."""

    MODE_ENGINE = "ENGINE_VS_ENGINE"
    MODE_HUMAN_WHITE = "HUMAN_VS_ENGINE"
    MODE_HUMAN_VS_HUMAN = "HUMAN_VS_HUMAN"
    MODE_CHOICES = [
        (MODE_ENGINE, "Engine vs Engine"),
        (MODE_HUMAN_WHITE, "Human (White) vs Engine (Black)"),
        (MODE_HUMAN_VS_HUMAN, "Human vs Human (hot-seat)"),
    ]

    mode = forms.ChoiceField(choices=MODE_CHOICES, initial=MODE_ENGINE)

    white_strength = forms.IntegerField(min_value=1, max_value=20, initial=10)
    black_strength = forms.IntegerField(min_value=1, max_value=20, initial=10)
    move_interval_ms = forms.IntegerField(min_value=200, max_value=5000, initial=600)
    preview_ms = forms.IntegerField(min_value=0, max_value=2000, initial=350, required=False)
    movetime_ms = forms.IntegerField(min_value=50, max_value=2000, initial=150)


class UpdateConfigForm(forms.Form):
    """Modify parameters while a match is running or paused.

    Note: for PLAYER (human) the strength/movetime fields are irrelevant,
    so they are not required.
    """

    side = forms.ChoiceField(choices=Side.choices)
    engine_type = forms.ChoiceField(choices=EngineType.choices)

    strength_mode = forms.ChoiceField(choices=StrengthMode.choices, required=False)
    strength_value = forms.IntegerField(min_value=1, max_value=3200, required=False)
    movetime_ms = forms.IntegerField(min_value=50, max_value=2000, required=False)

    # GNN only: minimax search depth (0 = policy only, no search).
    search_depth = forms.TypedChoiceField(
        choices=GNN_DEPTH_CHOICES, coerce=int, required=False,
    )

    def clean(self):
        cleaned = super().clean()
        engine_type = cleaned.get("engine_type")

        # Human players and the GNN do not use the UCI strength / thinking time.
        if engine_type in (EngineType.PLAYER, EngineType.GNN):
            return cleaned

        mode = cleaned.get("strength_mode")
        val = cleaned.get("strength_value")
        mt = cleaned.get("movetime_ms")

        if mode is None or val is None or mt is None:
            raise forms.ValidationError("Missing engine parameters.")

        if mode == StrengthMode.SKILL and not (1 <= val <= 20):
            raise forms.ValidationError("In skill mode the strength value must be between 1 and 20.")

        if mode == StrengthMode.ELO and not (800 <= val <= 3200):
            raise forms.ValidationError("In Elo mode the strength value must be between 800 and 3200.")

        return cleaned


class UpdateSpeedForm(forms.Form):
    """Set the display speed (move interval + preview)."""

    move_interval_ms = forms.IntegerField(min_value=200, max_value=5000, initial=600)
    preview_ms = forms.IntegerField(min_value=0, max_value=2000, initial=350, required=False)
