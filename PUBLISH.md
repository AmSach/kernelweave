# Publishing KernelWeave to GitHub Pages and GitHub

## What you have
- NeurIPS-style paper source: `paper/main.tex`
- compiled PDF: `kernelweave-neurips-paper.pdf`
- codebase zip: `kernelweave-neurips-codebase.zip`
- prototype code under `kernelweave/`

## Step-by-step publish guide

### 1) Create the GitHub repo
If the repo does not already exist, create it under your GitHub account:

```bash
gh repo create AmanSachan/kernelweave --public --source=. --remote=origin --push
```

If you want it private first, use `--private`.

### 2) Push the code
If the remote already exists:

```bash
git add .
git commit -m "Publish KernelWeave prototype"
git push -u origin main
```

### 3) Attach the NeurIPS files
Keep these in the repo root so reviewers can find them easily:
- `paper/main.tex`
- `kernelweave-neurips-paper.pdf`
- `kernelweave-neurips-codebase.zip`

### 4) Create a GitHub Pages branch or site
If you want a project page:

```bash
gh pages publish --branch main --path docs
```

Better: build a tiny `docs/` site or `index.html` that links the PDF and zip.

### 5) Put the paper and zip online
Recommended repo layout:
- `/paper/main.tex`
- `/kernelweave-neurips-paper.pdf`
- `/kernelweave-neurips-codebase.zip`
- `/docs/index.html`

### 6) Use the paper with NeurIPS submission
- Upload the PDF to OpenReview when submitting.
- Upload the technical appendix / codebase zip as supplementary material if permitted.
- Keep the main manuscript within the page limit.

### 7) Final sanity checks
- compile the TeX locally
- open the PDF and inspect equations/figures
- verify the zip contains the full codebase
- verify the repo renders on GitHub

## Direct links in this workspace
- `file 'kernelweave-neurips-paper.pdf'`
- `file 'kernelweave-neurips-codebase.zip'`
- `file 'paper/main.tex'`
