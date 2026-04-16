import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
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
print("MODEL 2: LST ~ dist_to_park_m + near_water  (city + year FE)")
print("="*80)

model2 = PanelOLS.from_formula(
    "LST ~ dist_to_park_m + near_water + EntityEffects + TimeEffects",
    data=df
)
result2 = model2.fit(cov_type="clustered", cluster_entity=True)
print(result2.summary)

# ── Model 3: Heterogeneity — dist_to_park_m × park size ──────────────────────
# The interaction coefficient answers: does an extra meter from a medium/large
# park cool LST differently than an extra meter from a small park?
# Reference category = small; positive interaction = weaker cooling vs. small.
print("\n" + "="*80)
print("MODEL 3: LST ~ dist_to_park_m * park_size_cat + near_water")
print("         (city + year FE)  — cooling by distance, heterogeneous across park sizes")
print("="*80)

model3 = PanelOLS.from_formula(
    "LST ~ dist_to_park_m * C(park_size_cat) + near_water"
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

# ── Plot 1: Residualized scatter (0–500 m bandwidth) ─────────────────────────
# Partial out city-year FE and NDVI via within-demean + OLS, then plot
# the pure distance gradient in the near-park window (< 500 m).
print("\n" + "="*80)
print("PLOT 1: Residualized LST vs dist_to_park_m  (0–500 m, city-year FE + NDVI removed)")
print("="*80)

_res = raw[
    (raw["is_water"] == 0) &
    raw["dist_to_park_m"].notna() &
    (raw["dist_to_park_m"] > 0) &
    (raw["dist_to_park_m"] <= 500) &
    raw["LST"].notna() &
    raw["NDVI"].notna()
].copy()

# Step 1: within-demean by city-year (removes city-year FE)
for col in ["LST", "NDVI"]:
    _res[f"{col}_dm"] = (
        _res[col] - _res.groupby(["city", "year"])[col].transform("mean")
    )

# Step 2: partial out demeaned NDVI → residual LST
_X = np.column_stack([np.ones(len(_res)), _res["NDVI_dm"].values])
_beta = np.linalg.lstsq(_X, _res["LST_dm"].values, rcond=None)[0]
_res["resid_LST"] = _res["LST_dm"] - (_X @ _beta)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(_res["dist_to_park_m"], _res["resid_LST"],
           alpha=0.05, s=1, color="steelblue", rasterized=True)

_xv = _res["dist_to_park_m"].values
_yv = _res["resid_LST"].values
_m, _b = np.polyfit(_xv, _yv, 1)
_xl = np.linspace(_xv.min(), _xv.max(), 200)
ax.plot(_xl, _m * _xl + _b, color="firebrick", linewidth=1.5,
        label=f"OLS slope: {_m * 1_000:.4f} °C/km")
ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
ax.set_xlabel("Distance to nearest park patch (m)", fontsize=11)
ax.set_ylabel("Residual LST (°C)\n[city-year FE + NDVI removed]", fontsize=10)
ax.set_title("Distance to Park vs Residual LST  (0–500 m bandwidth)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()

resid_path = Path.home() / "dist_park_vs_resid_lst.png"
plt.savefig(resid_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Residualized scatter saved to: {resid_path}")

# ── Plot 2: Binned means (150 m bins, 0–3 km) ────────────────────────────────
# Collapses vertical scatter into a clean cooling curve.
# Expect: coolest near boundary, gradual warming, flattening ~600–900 m.
print("\n" + "="*80)
print("PLOT 2: Mean LST by distance bin  (150 m bins, 0–3 km)")
print("="*80)

BIN_WIDTH_M = 150
BIN_MAX_M   = 3_000

_bin = raw[
    (raw["is_water"] == 0) &
    raw["dist_to_park_m"].notna() &
    (raw["dist_to_park_m"] > 0) &
    (raw["dist_to_park_m"] <= BIN_MAX_M) &
    raw["LST"].notna()
].copy()

_bins = np.arange(0, BIN_MAX_M + BIN_WIDTH_M, BIN_WIDTH_M)
_bin["dist_bin"] = pd.cut(_bin["dist_to_park_m"], bins=_bins)

_binned = (
    _bin.groupby("dist_bin", observed=True)["LST"]
    .agg(["mean", "std", "count"])
    .dropna()
)
_binned["ci95"] = 1.96 * _binned["std"] / np.sqrt(_binned["count"])
_bin_centers = np.array([iv.mid for iv in _binned.index]) / 1_000  # km

print(_binned[["mean", "count", "ci95"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(_bin_centers, _binned["mean"], color="steelblue", linewidth=2,
        marker="o", markersize=4)
ax.fill_between(_bin_centers,
                _binned["mean"] - _binned["ci95"],
                _binned["mean"] + _binned["ci95"],
                color="steelblue", alpha=0.25, label="95% CI")
ax.set_xlabel("Distance to nearest park patch (km)", fontsize=11)
ax.set_ylabel("Mean LST (°C)", fontsize=11)
ax.set_title(f"Mean LST by Distance to Park  ({BIN_WIDTH_M} m bins)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()

binned_path = Path.home() / "dist_park_binned_lst.png"
plt.savefig(binned_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Binned means plot saved to: {binned_path}")

# ── Near/far comparison table ─────────────────────────────────────────────────
# Compares mean LST across three distance bands, raw and residualized.
# Residualized = within-demean by city-year so city/seasonal differences
# don't drive the result — only the distance gradient matters.
print("\n" + "="*80)
print("NEAR/FAR COMPARISON: Mean LST by distance band")
print("="*80)

_comp = raw[
    (raw["is_water"] == 0) &
    raw["dist_to_park_m"].notna() &
    (raw["dist_to_park_m"] > 0) &
    raw["LST"].notna()
].copy()

_comp["band"] = pd.cut(
    _comp["dist_to_park_m"],
    bins=[0, 300, 600, np.inf],
    labels=["0–300 m (near park)", "300–600 m (transition)", "600 m+ (far from park)"]
)

# Raw means
_raw_tbl = (
    _comp.groupby("band", observed=True)["LST"]
    .agg(mean="mean", std="std", n="count")
)
_raw_tbl["se"]   = _raw_tbl["std"] / np.sqrt(_raw_tbl["n"])
_raw_tbl["ci95"] = 1.96 * _raw_tbl["se"]

# Residualized means (within city-year demean)
_comp["LST_dm"] = (
    _comp["LST"] - _comp.groupby(["city", "year"])["LST"].transform("mean")
)
_resid_tbl = (
    _comp.groupby("band", observed=True)["LST_dm"]
    .agg(mean="mean", std="std", n="count")
)
_resid_tbl["se"]   = _resid_tbl["std"] / np.sqrt(_resid_tbl["n"])
_resid_tbl["ci95"] = 1.96 * _resid_tbl["se"]

print("\nRaw mean LST (°C) by distance band:")
print(_raw_tbl[["mean", "std", "n", "ci95"]].round(3).to_string())

print("\nResidual mean LST (°C) by distance band  [city-year FE removed]:")
print(_resid_tbl[["mean", "std", "n", "ci95"]].round(3).to_string())

# Pairwise difference: near vs far (residualized)
_near = _comp.loc[_comp["band"] == "0–300 m (near park)", "LST_dm"]
_far  = _comp.loc[_comp["band"] == "600 m+ (far from park)", "LST_dm"]
_diff = _far.mean() - _near.mean()
_se_diff = np.sqrt(_near.var() / len(_near) + _far.var() / len(_far))
_t    = _diff / _se_diff
print(f"\nFar minus near (residualized): {_diff:+.3f} °C  (t = {_t:.2f})")
print("Positive t = far pixels are warmer → supports cooling hypothesis")
