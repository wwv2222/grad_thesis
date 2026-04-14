import os
import io
import re
from pathlib import Path
import pandas as pd
import dropbox
from dotenv import load_dotenv

load_dotenv()

# ── Auth ──────────────────────────────────────────────────────────────────────
dbx = dropbox.Dropbox(
    oauth2_refresh_token=os.environ.get('DROPBOX_REFRESH_TOKEN'),
    app_key=os.environ.get('DROPBOX_APP_KEY'),
    app_secret=os.environ.get('DROPBOX_APP_SECRET')
)

HOUSTON_FOLDER = '/20 Cities/Houston'
OUTPUT_DROPBOX = '/20 Cities/houston_panel.csv'
OUTPUT_LOCAL   = Path.home() / 'houston_panel.csv'

# ── List all files in the Houston folder ──────────────────────────────────────
result = dbx.files_list_folder(HOUSTON_FOLDER)
entries = result.entries
while result.has_more:
    result = dbx.files_list_folder_continue(result.cursor)
    entries += result.entries

all_files = {
    e.name: e for e in entries
    if isinstance(e, dropbox.files.FileMetadata) and e.name.endswith('.csv')
}
print(f"Found {len(all_files)} CSV files in {HOUSTON_FOLDER}:")
for name in sorted(all_files):
    print(f"  {name}")
print()


def download_csv(entry):
    _, response = dbx.files_download(entry.path_lower)
    return pd.read_csv(io.BytesIO(response.content))


# ── 1. Load LST + NDVI:  houston_{year}.csv  (scale=150, years 1984-2024) ─────
lst_ndvi_files = sorted(
    [e for name, e in all_files.items()
     if re.fullmatch(r'houston_\d{4}\.csv', name, re.IGNORECASE)],
    key=lambda e: e.name
)
print(f"LST/NDVI files: {len(lst_ndvi_files)}")
lst_ndvi_dfs = []
for entry in lst_ndvi_files:
    df = download_csv(entry)
    lst_ndvi_dfs.append(df)
    print(f"  {entry.name}: {len(df):,} rows")

base = pd.concat(lst_ndvi_dfs, ignore_index=True)
print(f"  → {len(base):,} total rows\n")

# ── 2. Load MNDWI:  houston_mndwi_{year}.csv  (scale=150) ────────────────────
mndwi_files = sorted(
    [e for name, e in all_files.items()
     if re.fullmatch(r'houston_mndwi_\d{4}\.csv', name, re.IGNORECASE)],
    key=lambda e: e.name
)
print(f"MNDWI files: {len(mndwi_files)}")
mndwi_dfs = []
for entry in mndwi_files:
    df = download_csv(entry)
    mndwi_dfs.append(df[['latitude', 'longitude', 'year', 'MNDWI']])
    print(f"  {entry.name}: {len(df):,} rows")

mndwi = pd.concat(mndwi_dfs, ignore_index=True)
print(f"  → {len(mndwi):,} total rows\n")

# ── 3. Merge LST/NDVI + MNDWI on exact (lat, lon, year) ─────────────────────
panel = base.merge(mndwi, on=['latitude', 'longitude', 'year'], how='left')
unmatched = panel['MNDWI'].isna().sum()
if unmatched:
    print(f"WARNING: {unmatched:,} rows had no MNDWI match.")
else:
    print(f"✓ LST/NDVI × MNDWI merge complete — all rows matched.\n")

# ── 4. Load park similarity file ──────────────────────────────────────────────
similarity_matches = [
    (name, e) for name, e in all_files.items()
    if 'similarity' in name.lower() or 'park' in name.lower()
]
if not similarity_matches:
    print("WARNING: No park similarity file found — skipping.")
else:
    sim_name, sim_entry = similarity_matches[0]
    if len(similarity_matches) > 1:
        print(f"Multiple similarity files found; using: {sim_name}")
        print(f"  Others skipped: {[n for n, _ in similarity_matches[1:]]}")

    sim_df = download_csv(sim_entry)
    print(f"Park similarity file: {sim_name}")
    print(f"  Rows   : {len(sim_df):,}")
    print(f"  Columns: {list(sim_df.columns)}")
    print(f"  Sample :\n{sim_df.head(3).to_string(index=False)}\n")

    # Determine merge keys: use whichever of year/latitude/longitude are present
    key_candidates = ['latitude', 'longitude', 'year']
    merge_keys = [k for k in key_candidates if k in sim_df.columns]
    print(f"Merging on: {merge_keys}")

    # Drop any columns already in panel (except merge keys) to avoid conflicts
    drop_cols = [c for c in sim_df.columns if c in panel.columns and c not in merge_keys]
    if drop_cols:
        print(f"  Dropping duplicate columns from similarity file: {drop_cols}")
        sim_df = sim_df.drop(columns=drop_cols)

    panel = panel.merge(sim_df, on=merge_keys, how='left')
    sim_col = [c for c in sim_df.columns if c not in merge_keys]
    unmatched_sim = panel[sim_col].isna().all(axis=1).sum()
    if unmatched_sim:
        print(f"WARNING: {unmatched_sim:,} rows had no similarity match.")
    else:
        print(f"✓ Park similarity merge complete — all rows matched.\n")

# ── 5. Sanity checks ──────────────────────────────────────────────────────────
dupes = panel.duplicated(subset=['latitude', 'longitude', 'year'])
print(f"Duplicate (latitude, longitude, year) rows: {dupes.sum()}")
print(f"\nShape  : {panel.shape}")
print(f"Columns: {list(panel.columns)}")
print(f"Years  : {sorted(panel['year'].unique())}")
print(f"\nMissing values:\n{panel.isnull().sum()}")

# ── 6. Save locally ───────────────────────────────────────────────────────────
panel.to_csv(OUTPUT_LOCAL, index=False)
print(f"\nSaved locally to {OUTPUT_LOCAL}")

# ── 7. Upload to Dropbox ──────────────────────────────────────────────────────
buf = io.BytesIO()
panel.to_csv(buf, index=False)
data = buf.getvalue()
CHUNK = 150 * 1024 * 1024  # 150 MB

if len(data) <= CHUNK:
    dbx.files_upload(data, OUTPUT_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
else:
    session = dbx.files_upload_session_start(data[:CHUNK])
    cursor  = dropbox.files.UploadSessionCursor(session_id=session.session_id, offset=CHUNK)
    offset  = CHUNK
    while offset < len(data):
        chunk = data[offset:offset + CHUNK]
        if offset + CHUNK >= len(data):
            commit = dropbox.files.CommitInfo(path=OUTPUT_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
            dbx.files_upload_session_finish(chunk, cursor, commit)
        else:
            dbx.files_upload_session_append_v2(chunk, cursor)
            cursor = dropbox.files.UploadSessionCursor(
                session_id=cursor.session_id,
                offset=cursor.offset + len(chunk)
            )
        offset += CHUNK

print(f"Uploaded to Dropbox: {OUTPUT_DROPBOX}")
