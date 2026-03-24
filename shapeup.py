import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

# %% 
# ── Load panel dataset ────────────────────────────────────────────────────────
LOCAL_PATH = Path.home() / "panel_data.csv"
twenty_cities = pd.read_csv(LOCAL_PATH)

# %%
# ── Convert LST from raw Landsat DN to degrees Celsius ───────────────────────
# Landsat Collection 2 formula: LST_K = DN × 0.00341802 + 149.0
# Then: LST_C = LST_K - 273.15
twenty_cities['LST'] = twenty_cities['LST'] * 0.00341802 + 149.0 - 273.15

#%% 
# Set physically impossible LST values to NaN (fill/sentinel values from satellite)
bad_lst = (twenty_cities['LST'] < -20) | (twenty_cities['LST'] > 90)
print(f"LST values outside [-20, 90]°C set to NaN: {bad_lst.sum():,} rows")
twenty_cities.loc[bad_lst, 'LST'] = float('nan')

#%%
# ── Section 1: Unique spatial points per year ─────────────────────────────────
print("=" * 60)
print("SECTION 1: Unique (latitude, longitude) points per year")
print("=" * 60)
pts_per_year = twenty_cities.groupby('year').apply(
    lambda x: x[['latitude', 'longitude']].drop_duplicates().shape[0]
)
print(pts_per_year.to_string())
print(f"\nMin: {pts_per_year.min()}  Max: {pts_per_year.max()}  "
      f"Consistent: {pts_per_year.nunique() == 1}")

dupes = twenty_cities.duplicated(subset=['city', 'year', 'latitude', 'longitude'])
print(f"Duplicate (city, year, latitude, longitude) rows: {dupes.sum()}")

#%%
# ── Section 1b: Per-city grid consistency ─────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1b: Grid consistency per city")
print("(% of master grid covered each year — ideally close to 100%)")
print("=" * 60)

for city, grp in twenty_cities.groupby('city'):
    # Master grid = all unique lat/lon seen in ANY year for this city
    master = set(zip(grp['latitude'], grp['longitude']))
    n_master = len(master)

    # Coverage per year
    coverage = grp.groupby('year').apply(
        lambda x: len(set(zip(x['latitude'], x['longitude'])) ) / n_master * 100,
        include_groups=False
    ).round(1)

    print(f"\n{city}  (master grid: {n_master:,} points)")
    print(f"  Min coverage: {coverage.min()}%  (year {coverage.idxmin()})")
    print(f"  Max coverage: {coverage.max()}%  (year {coverage.idxmax()})")
    print(f"  Years < 90% coverage: {sorted(coverage[coverage < 90].index.tolist())}")

# %%
# ── Section 2: Summary statistics ────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: Summary statistics — LST and NDVI")
print("=" * 60)
pd.set_option('display.float_format', '{:.4f}'.format)
print(twenty_cities[['LST', 'NDVI']].describe())
pd.reset_option('display.float_format')


# %%
# ── Section 3: Missing values per city ───────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: % Missing by city (sorted worst first)")
print("=" * 60)
missing_by_city = (
    twenty_cities.groupby('city')[['LST', 'NDVI']]
    .apply(lambda x: x.isnull().mean() * 100)
    .round(2)
)
print(missing_by_city.sort_values('LST', ascending=False).to_string())

# %%
# ── Section 4: Missing values per year ───────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4: % Missing by year (sorted worst first)")
print("=" * 60)
missing_by_year = (
    twenty_cities.groupby('year')[['LST', 'NDVI']]
    .apply(lambda x: x.isnull().mean() * 100)
    .round(2)
)
print(missing_by_year.sort_values('LST', ascending=False).to_string())

# %%
# ── Section 5: Missing value pattern — city × year pivot ─────────────────────
print("\n" + "=" * 60)
print("SECTION 5: % Missing LST — city × year pivot")
print("(High entire row = bad year; high entire col = bad city; scattered = clouds)")
print("=" * 60)
missing_pivot = (
    twenty_cities.groupby(['year', 'city'])['LST']
    .apply(lambda x: x.isnull().mean() * 100)
    .unstack('city')
    .round(1)
)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(missing_pivot.to_string())

# %%
# ── Section 6: Flag city-year cells exceeding 20% missing ────────────────────
print("\n" + "=" * 60)
print("SECTION 6: City-year cells with > 20% missing (LST or NDVI)")
print("=" * 60)
THRESHOLD = 20.0
cell_missing = (
    twenty_cities.groupby(['city', 'year'])[['LST', 'NDVI']]
    .apply(lambda x: x.isnull().mean() * 100)
    .round(2)
)
bad_cells = cell_missing[
    (cell_missing['LST'] > THRESHOLD) | (cell_missing['NDVI'] > THRESHOLD)
]
print(f"{len(bad_cells)} city-year cells exceed {THRESHOLD}% missing:\n")
print(bad_cells.sort_values('LST', ascending=False).to_string())

# %%
# ── Section 7: Water classification and proximity control ─────────────────────
print("\n" + "=" * 60)
print("SECTION 7: Water classification and near-water proximity")
print("=" * 60)

NDVI_WATER_THRESHOLD = 0.05  # NDVI below this → classified as water
WATER_BUFFER_M = 500         # meters; near_water=1 if within this distance

# Step 1 — Classify water pixels
# Water absorbs NIR strongly → very low/negative NDVI
# Original LST column is NOT modified; use is_water==0 to filter in regressions
twenty_cities['is_water'] = (twenty_cities['NDVI'] < NDVI_WATER_THRESHOLD).astype(int)

n_water = twenty_cities['is_water'].sum()
print(f"\nNDVI water threshold: {NDVI_WATER_THRESHOLD}")
print(f"Total water pixels flagged: {n_water:,} ({n_water / len(twenty_cities) * 100:.1f}% of all rows)")

print("\nWater pixel share by city (% of all city rows):")
water_by_city = (
    twenty_cities.groupby('city')['is_water']
    .mean()
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
    .rename('% water')
)
print(water_by_city.to_string())

# Step 2 — Distance to nearest water pixel (per city-year)
# Converts lat/lon to approximate meters; builds cKDTree on water pixels
# and queries every pixel's distance to its nearest water neighbor.
def _water_dist(group):
    lat_rad = np.radians(group['latitude'].mean())
    lat_m = group['latitude'].values * 111_000
    lon_m = group['longitude'].values * 111_000 * np.cos(lat_rad)
    coords = np.column_stack([lat_m, lon_m])
    water_mask = group['is_water'].values == 1
    if water_mask.sum() == 0:
        return pd.Series(np.inf, index=group.index)
    tree = cKDTree(coords[water_mask])
    dist, _ = tree.query(coords)
    return pd.Series(dist, index=group.index)

twenty_cities['dist_to_water_m'] = (
    twenty_cities
    .groupby(['city', 'year'], group_keys=False)
    .apply(_water_dist)
)

# Step 3 — Binary proximity indicator
twenty_cities['near_water'] = (twenty_cities['dist_to_water_m'] < WATER_BUFFER_M).astype(int)

n_near = twenty_cities['near_water'].sum()
print(f"\nNear-water pixels (within {WATER_BUFFER_M}m): {n_near:,} ({n_near / len(twenty_cities) * 100:.1f}%)")

print(f"\nNear-water share by city (% of all city rows):")
near_by_city = (
    twenty_cities.groupby('city')['near_water']
    .mean()
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
    .rename('% near water')
)
print(near_by_city.to_string())

print("\nNew columns added (LST unchanged):")
print("  is_water       — 1 if NDVI < threshold (exclude from LST regressions)")
print("  dist_to_water_m — distance in meters to nearest water pixel (same city-year)")
print("  near_water     — 1 if dist_to_water_m < buffer (use as regression control)")
