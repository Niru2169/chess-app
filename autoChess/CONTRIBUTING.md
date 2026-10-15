# Contributing

## Development setup

- Python 3.12+
- Django 6.0+
- Stockfish installed locally (or available on PATH)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
python manage.py migrate
python manage.py test
```

## Style

- Type hints everywhere (public APIs).
- Google-style docstrings: section headers in English; explanations in Hungarian.
- Keep modules cohesive; avoid “god objects”.

## Tests

- Unit tests: FEN parsing, move highlighting, draw detection logic.
- Integration tests: only run if Stockfish is configured (skipped otherwise).
