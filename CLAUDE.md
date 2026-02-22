# CLAUDE.md

This file provides guidance for AI assistants working in this repository.

## Project Overview

This is a graduate thesis data science project that retrieves geospatial / satellite imagery data from **Google Earth Engine (GEE)** and exports results to **Dropbox** for storage and analysis. The project is in active, early-stage development and intentionally kept minimal.

**GEE Project ID:** `grad-thesis-475918`

---

## Repository Structure

```
grad_thesis/
├── CLAUDE.md          # This file
├── README.md          # One-line project description
├── scraping.py        # Local Python script for GEE data retrieval setup
└── analysis.ipynb     # Primary analysis notebook (designed for Google Colab)
```

There are no subdirectories, build artifacts, or generated files tracked in git.

---

## Key Files

### `analysis.ipynb`
The **main working file**. Designed to run in **Google Colab**. Contains:
- Dependency installation (`dropbox`, `earthengine-api`)
- GEE and Dropbox authentication
- Two reusable export utility functions:
  - `export_df_to_dropbox(df, dropbox_path)` — serializes a pandas DataFrame to CSV and uploads it to Dropbox
  - `export_file_to_dropbox(local_path, dropbox_path)` — uploads any local file to Dropbox
- Commented-out template code for GEE image collection queries and batch export tasks

### `scraping.py`
A local Python script (not Colab-specific) that:
- Installs `earthengine-api` and `dropbox` via subprocess
- Authenticates and initializes the Earth Engine client with the project ID
- Uses `#%%` cell markers (VS Code Jupyter support)
- Currently contains only setup/boilerplate; data processing logic has not been filled in yet

---

## Development Environment

**Primary environment: Google Colab**
- The notebook uses `from google.colab import userdata` for secret management
- Dropbox token is stored as a Colab Secret under the key `DROPBOX_TOKEN`
- GEE authentication runs interactively via `ee.Authenticate()`

**Secondary environment: Local Python**
- `scraping.py` can run in a standard Python environment
- Requires Python 3, pip, and internet access to authenticate with GEE

**No virtual environment or `requirements.txt` is maintained.** Dependencies are installed at runtime with `pip install` directly inside notebooks/scripts.

---

## Authentication

Both files require manual authentication steps that cannot be automated:

| Service | Method |
|---|---|
| Google Earth Engine | `ee.Authenticate()` — opens a browser OAuth flow |
| Dropbox | Access token stored in Colab Secrets as `DROPBOX_TOKEN` |

When working locally (outside Colab), replace `userdata.get('DROPBOX_TOKEN')` with an environment variable or hardcoded token (never commit tokens to git).

---

## Common Patterns

### Installing dependencies inside a notebook/script
```python
import subprocess
subprocess.run(["pip3", "install", "earthengine-api", "--quiet"])
subprocess.run(["pip3", "install", "dropbox"])
```
Or in Colab cells:
```python
!pip install dropbox earthengine-api
```

### Initializing GEE
```python
import ee
ee.Authenticate()
ee.Initialize(project='grad-thesis-475918')
```

### Exporting a DataFrame to Dropbox
```python
export_df_to_dropbox(my_df, '/grad_thesis/results.csv')
```

### Exporting a local file to Dropbox
```python
export_file_to_dropbox('/content/output.tif', '/grad_thesis/output.tif')
```

### GEE image collection query template
```python
collection = ee.ImageCollection('YOUR_COLLECTION_HERE') \
    .filterDate('2023-01-01', '2023-12-31') \
    .filterBounds(your_region)
```

---

## Code Conventions

- **Python style:** No enforced linter or formatter. Keep code readable and well-commented.
- **Cell markers:** `#%%` is used in `.py` files to mark Jupyter-compatible cells (for VS Code).
- **Error handling:** Minimal — direct API calls without try/except unless the operation is explicitly fragile.
- **Dropbox upload mode:** Always use `WriteMode('overwrite')` for idempotent uploads.
- **Secrets:** Never hardcode API keys or access tokens. Use Colab Secrets or environment variables.

---

## What Does Not Exist (by design)

- No `requirements.txt` or `pyproject.toml` — dependencies are installed at runtime
- No test suite — this is academic research code, not a production application
- No CI/CD pipeline — no `.github/workflows/` or equivalent
- No `Makefile` or build scripts
- No `docs/` directory

Do not add these unless explicitly requested. Keep the project minimal.

---

## Git Workflow

- **Default branch:** `master`
- **AI-generated branches** follow the pattern: `claude/<feature>-<hash>`
- Commit messages are plain English, no strict convention enforced
- The remote is accessed through a local proxy at `127.0.0.1:27674`

When making changes as an AI assistant:
1. Work on the designated `claude/` branch
2. Write clear, descriptive commit messages
3. Push with `git push -u origin <branch-name>`
4. Open a pull request targeting `master`

---

## Project Status

Early development. The authentication scaffolding and export utilities are in place. The core GEE data retrieval queries and analysis logic remain to be implemented based on the specific thesis research questions.
