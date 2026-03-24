
import os
import io
import re
import getpass
from pathlib import Path
import pandas as pd
import dropbox

# ── Auth ──────────────────────────────────────────────────────────────────────
token = os.environ.get("DROPBOX_TOKEN") or getpass.getpass("Dropbox access token: ")
dbx = dropbox.Dropbox(token)

DROPBOX_FOLDER = "/20 Cities"   # path inside Dropbox (case-sensitive)
OUTPUT_PATH    = "/panel_data.csv"

# ── Helper: list all entries in a Dropbox folder ──────────────────────────────
def list_folder(dbx, path):
    result = dbx.files_list_folder(path)
    entries = result.entries
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries += result.entries
    return entries

# ── Main loop ─────────────────────────────────────────────────────────────────
all_dfs = []

city_folders = [
    e for e in list_folder(dbx, DROPBOX_FOLDER)
    if isinstance(e, dropbox.files.FolderMetadata)
]
print(f"Found {len(city_folders)} city folders\n")

for city_entry in sorted(city_folders, key=lambda e: e.name):
    city_name = city_entry.name
    city_path = city_entry.path_lower
    city_dfs  = []

    csv_files = [
        e for e in list_folder(dbx, city_path)
        if isinstance(e, dropbox.files.FileMetadata) and e.name.endswith(".csv")
    ]

    for csv_entry in sorted(csv_files, key=lambda e: e.name):
        match = re.search(r'_(\d{4})\.csv$', csv_entry.name, re.IGNORECASE)
        if not match:
            print(f"  SKIP (no year): {csv_entry.name}")
            continue
        year = int(match.group(1))

        _, response = dbx.files_download(csv_entry.path_lower)
        df = pd.read_csv(io.BytesIO(response.content))
        df['year'] = year
        df['city'] = city_name
        city_dfs.append(df)

    if city_dfs:
        city_panel = pd.concat(city_dfs, ignore_index=True)
        all_dfs.append(city_panel)
        print(f"{city_name}: {len(city_dfs)} files, {len(city_panel):,} rows")
    else:
        print(f"WARNING: no matching CSVs in {city_name}")

if not all_dfs:
    raise RuntimeError("No data loaded. Check the DROPBOX_FOLDER path and file naming.")

# ── Combine & save ────────────────────────────────────────────────────────────
panel = pd.concat(all_dfs, ignore_index=True)

# Upload using chunked session (handles files of any size)
buf = io.BytesIO()
panel.to_csv(buf, index=False)
buf.seek(0)
data = buf.read()
total = len(data)
CHUNK = 100 * 1024 * 1024  # 100 MB chunks

if total <= CHUNK:
    dbx.files_upload(data, OUTPUT_PATH, mode=dropbox.files.WriteMode.overwrite)
else:
    session = dbx.files_upload_session_start(data[:CHUNK])
    cursor  = dropbox.files.UploadSessionCursor(session.session_id, offset=CHUNK)
    offset  = CHUNK
    while offset < total:
        chunk = data[offset: offset + CHUNK]
        offset += len(chunk)
        if offset >= total:
            commit = dropbox.files.CommitInfo(
                path=OUTPUT_PATH,
                mode=dropbox.files.WriteMode.overwrite
            )
            dbx.files_upload_session_finish(chunk, cursor, commit)
        else:
            dbx.files_upload_session_append_v2(chunk, cursor)
            cursor.offset = offset

print(f"\nUploaded panel_data.csv to Dropbox:{OUTPUT_PATH}")

# ── Save locally on server for fast access in shapeup.py ─────────────────────
LOCAL_PATH = Path.home() / "panel_data.csv"
panel.to_csv(LOCAL_PATH, index=False)
print(f"Also saved locally to {LOCAL_PATH}")

# ── Working dataset ───────────────────────────────────────────────────────────
twenty_cities = panel

print(f"\nShape  : {twenty_cities.shape}")
print(f"Columns: {list(twenty_cities.columns)}")
print(f"Cities : {sorted(twenty_cities['city'].unique())}")
print(f"Years  : {sorted(twenty_cities['year'].unique())}")
print(f"\nMissing values:\n{twenty_cities.isnull().sum()}")
print(f"\nSample:\n{twenty_cities.head()}")
