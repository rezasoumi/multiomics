#!/usr/bin/env bash
set -euo pipefail
export PATH="${HOME}/Library/TinyTeX/bin/universal-darwin:${PATH}"
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode thesistemplate.tex
bibtex thesistemplate || true
pdflatex -interaction=nonstopmode thesistemplate.tex
pdflatex -interaction=nonstopmode thesistemplate.tex
echo "Built: $(pwd)/thesistemplate.pdf"
