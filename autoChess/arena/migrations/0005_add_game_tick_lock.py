from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("arena", "0004_add_player_ui_selection"),
    ]

    operations = [
        migrations.AddField(
            model_name="game",
            name="tick_lock",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="game",
            name="tick_lock_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
