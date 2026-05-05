#!/usr/bin/env bash
set -euo pipefail
mkdir -p build
TEXMFOUTPUT=$PWD/build pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build paper/main.tex
cp build/main.pdf kernelweave-neurips-paper.pdf
