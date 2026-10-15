from __future__ import annotations

from django.core.management.base import BaseCommand

from arena.services.rating import rebuild_ratings


class Command(BaseCommand):
    help = "Recompute every Elo/BayesElo rating from finished matches (in chronological order)."

    def handle(self, *args, **options) -> None:
        applied = rebuild_ratings()
        self.stdout.write(self.style.SUCCESS(f"Re-rated {applied} match(es)."))
