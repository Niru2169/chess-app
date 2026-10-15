# Third-party licenses

This repository includes or references the following third-party components.

## Python dependencies

| Name | License | Notes |
|---|---|---|
| Django | BSD-3-Clause | Web framework |
| python-dotenv | BSD-3-Clause | Loads `.env` in development/runtime entrypoints |
| python-chess (package name: `chess`) | GPL-3.0-or-later | Used for move generation/rendering and the built-in PyChess minimax engine |

## Frontend libraries (loaded via CDN)

| Name | License | Notes |
|---|---|---|
| htmx | BSD-2-Clause | Loaded via CDN |
| Tailwind CSS (Play CDN) | MIT | Loaded via CDN |

## External engine component (separately installed)

| Name | License | Notes |
|---|---|---|
| Stockfish | GPL-3.0-or-later | UCI chess engine binary (not vendored in this repo) |

## Piece artwork used by python-chess SVG rendering

The SVG piece shapes used by `python-chess` are based on the “Cburnett” chess pieces by Colin M. L. Burnett.
These are distributed by python-chess under a tri-license (GFDL 1.2+, BSD, GPL 2+).

If you need a different piece set/license profile, replace the SVG rendering layer with a separately vendored SVG
asset pack and update this document accordingly.
