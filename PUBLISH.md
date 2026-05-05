# Publishing KernelWeave to NeurIPS and GitHub

## What is included
- `paper/main.tex` — NeurIPS-style manuscript source
- `kernelweave-neurips-paper.pdf` — compiled PDF
- `kernelweave-neurips-codebase.zip` — codebase archive for reviewers
- `tests/` — automated validation
- `docs/` — architecture and algorithm notes

## NeurIPS page limit
NeurIPS 2025 main submissions are limited to **9 pages** of content. Put any extra material in the appendix or supplementary material. Keep the main paper tight and let the appendix carry the grind.

## Step-by-step publishing workflow

### 1) Verify the codebase locally
```bash
cd /home/workspace/Projects/kernelweave
python -m pytest -q
```

### 2) Build the paper
```bash
cd /home/workspace/Projects/kernelweave
./paper/build.sh
```

### 3) Inspect the deliverables
Make sure these files exist:
- `paper/main.tex`
- `kernelweave-neurips-paper.pdf`
- `kernelweave-neurips-codebase.zip`

### 4) Create the GitHub repository
If it doesn't exist yet:
```bash
gh repo create AmanSachan/kernelweave --public --source=. --remote=origin --push
```
If it already exists, just push:
```bash
git push -u origin main
```

### 5) Prepare the submission bundle
Upload to OpenReview / NeurIPS:
- the PDF manuscript
- the appendix if permitted
- the codebase zip as supplementary material if allowed

### 6) GitHub Pages / public landing page
For reviewers and readers, create a tiny docs site in `docs/` that links:
- the PDF
- the codebase zip
- a short project summary

Then publish the site from the repo settings or via `gh pages publish`.

### 7) Final checks before submission
- paper fits the 9-page limit
- figures and equations render cleanly
- tests pass
- zip includes the whole codebase
- README explains what the prototype actually does

## Files to hand over
- `file 'paper/main.tex'`
- `file 'kernelweave-neurips-paper.pdf'`
- `file 'kernelweave-neurips-codebase.zip'`
