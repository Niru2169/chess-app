from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('arena', '0002_add_pending_move_preview'),
    ]

    operations = [
        migrations.AddField(
            model_name='game',
            name='preview_ms',
            field=models.PositiveIntegerField(default=350),
        ),
    ]
