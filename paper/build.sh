#!/usr/bin/env bash
# Build the revised paper: regenerate tables/macros from results, then compile.
set -euo pipefail
cd "$(dirname "$0")"
python3 make_tables.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
cp main.pdf MultiCardioFusion_Revised_v5.pdf
echo "built paper/MultiCardioFusion_Revised_v5.pdf"
