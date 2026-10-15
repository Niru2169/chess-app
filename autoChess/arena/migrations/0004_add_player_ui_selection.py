from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('arena', '0003_add_game_preview_ms'),
    ]

    operations = [
        migrations.AddField(
            model_name='game',
            name='ui_selected_from',
            field=models.CharField(blank=True, default='', max_length=2),
        ),
        migrations.AddField(
            model_name='game',
            name='ui_selected_to',
            field=models.CharField(blank=True, default='', max_length=2),
        ),
    ]
