import pandas as pd
from linearmodels.panel import PanelOLS

# ── Base load ─────────────────────────────────────────────────────────────────
raw = pd.read_csv("/home/wwdelvalle/20_cities_panel_clean.csv", low_memory=False)

# Model 1: all non-water pixels across all cities
df_all = raw[raw["is_water"] == 0].copy()
print(f"Observations (non-water, all cities): {len(df_all):,}")
df_all = df_all.set_index(["city", "year"])

# Models 2-4: non-water pixels in cities that have dist_to_park_m data
df = raw[(raw["is_water"] == 0) & raw["dist_to_park_m"].notna()].copy()
print(f"Observations (non-water, with park distance): {len(df):,}")

# Map nearest_park_size (1/2/3) to labels; reference category = small
df["park_size_cat"] = pd.Categorical(
    df["nearest_park_size"].map({1: "small", 2: "medium", 3: "large"}),
    categories=["small", "medium", "large"],
    ordered=True
)
print("Park size distribution:\n", df["park_size_cat"].value_counts(), "\n")

# Two-way FE index: city (entity) + year (time)
df = df.set_index(["city", "year"])

# ── Model 1: LST ~ NDVI + near_water | city + year ───────────────────────────
print("="*80)
print("MODEL 1: LST ~ NDVI + near_water  (city + year FE)")
print("="*80)

model1 = PanelOLS.from_formula(
    "LST ~ NDVI + near_water + EntityEffects + TimeEffects",
    data=df_all
)
result1 = model1.fit(cov_type="clustered", cluster_entity=True)
print(result1.summary)

# ── Model 2: LST ~ dist_to_park_m + NDVI + near_water ────────────────────────
# Note: in_park_patch is dropped — dist_to_park_m is only recorded for pixels
# outside park patches, so after the notna() filter all remaining observations
# have in_park_patch == 0 (constant), making it collinear.
print("\n" + "="*80)
print("MODEL 2: LST ~ dist_to_park_m + NDVI + near_water  (city + year FE)")
print("="*80)

model2 = PanelOLS.from_formula(
    "LST ~ dist_to_park_m + NDVI + near_water + EntityEffects + TimeEffects",
    data=df
)
result2 = model2.fit(cov_type="clustered", cluster_entity=True)
print(result2.summary)

# ── Model 3: Heterogeneity — dist_to_park_m × park size ──────────────────────
# The interaction coefficient answers: does an extra meter from a medium/large
# park cool LST differently than an extra meter from a small park?
# Reference category = small; positive interaction = weaker cooling vs. small.
print("\n" + "="*80)
print("MODEL 3: LST ~ dist_to_park_m * park_size_cat + NDVI + near_water")
print("         (city + year FE)  — cooling by distance, heterogeneous across park sizes")
print("="*80)

model3 = PanelOLS.from_formula(
    "LST ~ dist_to_park_m * C(park_size_cat) + NDVI + near_water"
    " + EntityEffects + TimeEffects",
    data=df
)
result3 = model3.fit(cov_type="clustered", cluster_entity=True)
print(result3.summary)

# ── Model 4: Regression Discontinuity — park boundary threshold ──────────────
# Running variable: distance from park boundary (signed)
#   inside park (in_park_patch == 1)  →  0  (at/inside boundary)
#   outside park                      →  dist_to_park_m (positive)
# Treatment D: 1 = inside park, 0 = outside park
# Bandwidth: 1,000 m (only outside pixels within 1 km of a park boundary)
# Sample: same 5 cities as above + park pixels in those cities
# Note: NDVI is excluded — it is mechanically higher inside parks, so controlling
# for it would absorb the very effect we are estimating. near_water is kept.
print("\n" + "="*80)
print("MODEL 4: RDD — park boundary threshold (bandwidth = 1,000 m)")
print("         LST ~ D(in_park) + running_var + near_water  (city + year FE)")
print("="*80)

# Identify the 5 cities with park distance data
park_cities = df.index.get_level_values("city").unique().tolist()

rdd = raw[
    (raw["is_water"] == 0) &
    (raw["city"].isin(park_cities)) &
    ((raw["in_park_patch"] == 1) | (raw["dist_to_park_m"] <= 1000))
].copy()

# Build the signed running variable and treatment indicator
rdd["running_var"] = rdd["dist_to_park_m"].fillna(0)   # park pixels → 0
rdd["D"] = rdd["in_park_patch"].astype(int)             # 1 = inside park

print(f"RDD observations (bandwidth 1km): {len(rdd):,}")
print("Treatment breakdown:\n", rdd["D"].value_counts(), "\n")

rdd = rdd.set_index(["city", "year"])

model4 = PanelOLS.from_formula(
    "LST ~ D + running_var + near_water + EntityEffects + TimeEffects",
    data=rdd
)
result4 = model4.fit(cov_type="clustered", cluster_entity=True)
print(result4.summary)
