from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('arena', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='game',
            name='pending_move_uci',
            field=models.CharField(blank=True, default='', max_length=8),
        ),
        migrations.AddField(
            model_name='game',
            name='pending_move_set_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
