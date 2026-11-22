"""Multi-criteria (AHP) selection of elephant water-conservation sites.

Settlement rule: distance from the supplied village locations only (1-5 km scenarios; 2 km is the base case).
"""
import json
import warnings
from pathlib import Path

import geopandas as gpd
import numba as nb
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "outputs" / "02_layers"
V = ROOT / "outputs" / "02_vectors"
OUT = ROOT / "outputs" / "03_results"
OUT.mkdir(parents=True, exist_ok=True)
CELL, SPACING_M, TOP_N, BASE_KM = 30.0, 2000, 50, 2
SCENARIOS = (1, 2, 3, 4, 5)
LIM = dict(slope=11.3, catch_min=5, catch_max=1000, soil=40, mine=1000, metalled=300, unmetalled=100, road=50, water=60)


def r(n):
    with rasterio.open(L / f"{n}.tif") as s:
        return s.read(1)


with rasterio.open(L / "aoi.tif") as s:
    PROFILE, T = s.profile.copy(), s.transform
aoi = r("aoi") > 0
H, W = aoi.shape


def save(name, a):
    with rasterio.open(OUT / f"{name}.tif", "w", **PROFILE) as d:
        d.write(np.asarray(a, dtype="float32"), 1)


def up(a, lo, hi):
    return np.clip((a - lo) / (hi - lo), 0, 1)


def down(a, lo, hi):
    return 1 - up(a, lo, hi)


def pct(a):
    """Percentile rank (0-1) relative to the reserve."""
    v = np.sort(a[aoi & np.isfinite(a)])
    return (np.searchsorted(v, np.nan_to_num(a, nan=v[0])) / v.size).astype("float32")


def pref(n):
    return ((np.nan_to_num(r(n), nan=1) - 1) / 4).clip(0, 1).astype("float32")


def ahp(M):
    M = np.array(M, float)
    vals, vecs = np.linalg.eig(M)
    i = np.argmax(vals.real)
    w = np.abs(vecs[:, i].real)
    w /= w.sum()
    n = len(M)
    RI = {3: .58, 4: .90, 5: 1.12, 6: 1.24, 7: 1.32}[n]
    return w, ((vals[i].real - n) / (n - 1)) / RI


d_village = r("dist_village_only_m")
slope, upa = r("slope_deg"), r("upstream_area_ha")
cat_a, cat_q = r("catchment_soil_loss_tha"), r("catchment_runoff_mm")
lulc, wc = r("lulc"), r("worldcover") if (L / "worldcover.tif").exists() else None
if wc is None:
    with rasterio.open(ROOT / "outputs" / "01_gee" / "worldcover.tif") as s:
        wc = s.read(1)
d_water = r("dist_perennial_water_m")
d_road, d_met, d_unm = r("dist_road_any_m"), r("dist_road_metalled_m"), r("dist_road_unmetalled_m")
d_mine, d_crop = r("dist_mine_m"), r("dist_cropland_m")
zone = r("zonation")

GROUPS = {
    "Water availability": {
        "Catchment runoff": pct(cat_q),
        "Catchment size": (up(upa, 5, 25) * down(upa, 200, 1000) * .7 + .3 * up(upa, 5, 25)).astype("float32"),
        "Soil water retention (clay)": up(r("soil_clay_pct"), 20, 40).astype("float32"),
        "Topographic wetness": pct(r("twi")),
        "Rainfall": up(r("rain_annual_mm"), *np.nanpercentile(r("rain_annual_mm")[aoi], [1, 99])).astype("float32"),
    },
    "Elephant need and use": {
        "Water gap": up(d_water, 500, 3000).astype("float32"),
        "Elephant use density": pct(r("elephant_density")),
        "Habitat suitability": pref("habitat_suitability"),
        "Forest type preference": pref("pref_forest_type"),
        "Forest cover preference": pref("pref_forest_cover"),
        "Terrain preference": pref("pref_terrain"),
        "Elevation preference": pref("pref_elevation"),
    },
    "Terrain and erosion": {
        "Catchment soil loss": down(cat_a, 5, LIM["soil"]).astype("float32"),
        "Slope": down(slope, 3, LIM["slope"]).astype("float32"),
        "Local soil loss": down(r("rusle_A_tha"), 5, LIM["soil"]).astype("float32"),
    },
    "Human disturbance": {
        "Village distance": up(d_village, 1000, 4000).astype("float32"),
        "Cropland distance": up(d_crop, 200, 1500).astype("float32"),
        "Conservation zonation": np.select([zone == 3, zone == 2, zone == 1], [1., .6, .3], .3).astype("float32"),
        "Road distance": up(d_road, 100, 1500).astype("float32"),
        "Mine distance": up(d_mine, 1000, 5000).astype("float32"),
    },
}
MATS = {
    "groups": [[1, 1, 2, 2], [1, 1, 2, 2], [.5, .5, 1, 1], [.5, .5, 1, 1]],
    "Water availability": [[1, 2, 3, 4, 5], [1/2, 1, 2, 3, 4], [1/3, 1/2, 1, 2, 3], [1/4, 1/3, 1/2, 1, 2], [1/5, 1/4, 1/3, 1/2, 1]],
    "Elephant need and use": [[1, 2, 2, 4, 4, 5, 6], [1/2, 1, 1, 3, 3, 4, 5], [1/2, 1, 1, 3, 3, 4, 5], [1/4, 1/3, 1/3, 1, 1, 2, 3],
                              [1/4, 1/3, 1/3, 1, 1, 2, 3], [1/5, 1/4, 1/4, 1/2, 1/2, 1, 2], [1/6, 1/5, 1/5, 1/3, 1/3, 1/2, 1]],
    "Terrain and erosion": [[1, 2, 3], [1/2, 1, 2], [1/3, 1/2, 1]],
    "Human disturbance": [[1, 2, 3, 3, 4], [1/2, 1, 2, 2, 3], [1/3, 1/2, 1, 1, 2], [1/3, 1/2, 1, 1, 2], [1/4, 1/3, 1/2, 1/2, 1]],
}
gw, gcr = ahp(MATS["groups"])
rows, cr = [], {"groups": float(gcr)}
for (g, crit), gv in zip(GROUPS.items(), gw):
    w, c = ahp(MATS[g])
    cr[g] = float(c)
    rows += [{"group": g, "criterion": k, "group_weight": gv, "within_group_weight": x, "final_weight": gv * x}
             for k, x in zip(crit, w)]
weights = pd.DataFrame(rows)
weights.to_csv(OUT / "ahp_weights.csv", index=False)
names = weights.criterion.tolist()
wv = weights.final_weight.values
layers = {k: v for g in GROUPS.values() for k, v in g.items()}

score = np.zeros((H, W), "float32")
for n, x in zip(names, wv):
    score += layers[n] * float(x)
score = np.where(aoi, score * 100, np.nan)
# Existing perennial water is not a candidate surface, and flat reservoir DEM cells give artificial flow lines.
save("suitability_score", np.where(r("water_perennial") > 0, np.nan, score))
for n in names:
    save("criterion_" + n.lower().replace(" ", "_").replace("(", "").replace(")", ""), np.where(aoi, layers[n], np.nan))

water_near = ndi.distance_transform_edt(~(r("water_any") > 0), sampling=CELL) < LIM["water"]
base = {
    "Inside reserve": aoi,
    "Slope <= 11.3 deg": slope <= LIM["slope"],
    "Not on or next to existing water": ~water_near & ~np.isin(lulc, [4]),
    "Not built-up, road, mine, crop or fallow": ~np.isin(lulc, [2, 5, 6, 7, 8, 9, 10]) & ~np.isin(wc, [40, 50]),
    "Mine >= 1 km": d_mine >= LIM["mine"],
    "Roads clear": (d_met >= LIM["metalled"]) & (d_unm >= LIM["unmetalled"]) & (d_road >= LIM["road"]),
    "Catchment soil loss <= 40 t/ha/yr": cat_a <= LIM["soil"],
}
land = np.ones((H, W), bool)
funnel = [{"step": "Reserve", "area_km2": aoi.sum() * CELL ** 2 / 1e6}]
for k, m in base.items():
    land &= m
    if k != "Inside reserve":
        funnel.append({"step": k, "area_km2": land.sum() * CELL ** 2 / 1e6})
drain = (upa >= LIM["catch_min"]) & (upa <= LIM["catch_max"])


@nb.njit(cache=True)
def greedy(idx, h, w, rad):
    blocked = np.zeros((h, w), np.bool_)
    out = []
    r2 = rad * rad
    for i in idx:
        y, x = i // w, i % w
        if blocked[y, x]:
            continue
        out.append(i)
        for yy in range(max(0, y - rad), min(h, y + rad + 1)):
            for xx in range(max(0, x - rad), min(w, x + rad + 1)):
                if (yy - y) ** 2 + (xx - x) ** 2 <= r2:
                    blocked[yy, xx] = True
    return np.array(out, dtype=np.int64)


def pick(mask, s=score):
    c = np.flatnonzero(mask.ravel())
    o = c[np.argsort(-s.ravel()[c], kind="stable")]
    return greedy(o, H, W, int(SPACING_M / CELL))


rid, cid = r("range_id").astype(int), r("compartment_id").astype(int)
rng = gpd.read_file(V / "ranges.gpkg").set_index("range_id")
comp = gpd.read_file(V / "compartments.gpkg").set_index("comp_id")
vil = gpd.read_file(V / "villages_dedup.gpkg")
vt = cKDTree(np.c_[vil.geometry.x, vil.geometry.y])
extra = {k: r(k) for k in ("dem_m", "habitat_suitability", "rain_annual_mm", "rusle_A_tha", "catchment_soil_loss_tableC_tha",
                           "stream_order", "soil_clay_pct", "elephant_density")}
ZN = {3: "Elephant Conservation", 2: "Human-Elephant Management", 1: "Human-Elephant Co-existence", 0: "Unzoned"}


def structure(a, sl, order):
    if a < 25:
        return "Water hole / saucer pond"
    if a <= 200 and sl <= 5:
        return "Percolation tank / earthen pond"
    return "Check dam / stop dam" if order >= 2 else "Percolation tank / earthen pond"


def table(idx, km):
    y, x = np.divmod(idx, W)
    px, py = rasterio.transform.xy(T, y, x)
    px, py = np.array(px), np.array(py)
    dist, near = vt.query(np.c_[px, py])
    d = gpd.GeoDataFrame({
        "scenario_km": km, "score": score[y, x].round(2),
        "range": [rng.RANGENAME.get(i, "") for i in rid[y, x]],
        "beat": [comp.Beat_Name.get(i, "") if i else "" for i in cid[y, x]],
        "compartment": [comp.New_Compt.get(i, "") if i else "" for i in cid[y, x]],
        "zone": [ZN[int(i)] for i in zone[y, x]],
        "structure": [structure(a, s, o) for a, s, o in zip(upa[y, x], slope[y, x], extra["stream_order"][y, x])],
        "elev_m": extra["dem_m"][y, x].round(0), "slope_deg": slope[y, x].round(1),
        "catchment_ha": upa[y, x].round(1), "runoff_mm": cat_q[y, x].round(0),
        "yield_m3_yr": (upa[y, x] * 10 * cat_q[y, x]).round(-1),
        "catch_soil_loss_tha": cat_a[y, x].round(1), "catch_soil_loss_tableC": extra["catchment_soil_loss_tableC_tha"][y, x].round(1),
        "rain_mm": extra["rain_annual_mm"][y, x].round(0), "clay_pct": extra["soil_clay_pct"][y, x].round(0),
        "habitat_class": extra["habitat_suitability"][y, x], "elephant_density": extra["elephant_density"][y, x].round(3),
        "village_km": (d_village[y, x] / 1000).round(2), "nearest_village": vil.NAME.values[near],
        "water_km": (d_water[y, x] / 1000).round(2), "road_km": (d_road[y, x] / 1000).round(2),
        "crop_km": (d_crop[y, x] / 1000).round(2), "mine_km": (d_mine[y, x] / 1000).round(2),
    }, geometry=gpd.points_from_xy(px, py), crs=32644)
    ll = d.to_crs(4326)
    d.insert(1, "lat", ll.geometry.y.round(6))
    d.insert(2, "lon", ll.geometry.x.round(6))
    return d.sort_values("score", ascending=False).reset_index(drop=True)


summary, all_sites = [], {}
for km in SCENARIOS:
    allowed = land & (d_village >= km * 1000)
    candidates = allowed & drain
    idx = pick(candidates)
    t = table(idx, km) if idx.size else gpd.GeoDataFrame()
    all_sites[km] = t
    save(f"eligible_land_{km}km", np.where(aoi, allowed, np.nan))
    if len(t):
        t.insert(0, "rank", np.arange(1, len(t) + 1))
        t.to_file(OUT / f"sites_{km}km.gpkg", driver="GPKG")
        t.drop(columns="geometry").to_csv(OUT / f"sites_{km}km.csv", index=False)
    summary.append({"buffer_km": km, "reserve_beyond_buffer_pct": (d_village[aoi] >= km * 1000).mean() * 100,
                    "eligible_land_km2": allowed.sum() * CELL ** 2 / 1e6, "eligible_drainage_km2": candidates.sum() * CELL ** 2 / 1e6,
                    "sites": len(t), "ranges_with_sites": t["range"].nunique() if len(t) else 0,
                    "mean_score": t.score.mean() if len(t) else np.nan, "top50_mean_score": t.score.head(TOP_N).mean() if len(t) else np.nan})
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "scenario_summary.csv", index=False)
funnel.append({"step": "Village distance >= 2 km", "area_km2": (land & (d_village >= 2000)).sum() * CELL ** 2 / 1e6})
funnel.append({"step": "On a suitable drainage line", "area_km2": (land & (d_village >= 2000) & drain).sum() * CELL ** 2 / 1e6})
pd.DataFrame(funnel).to_csv(OUT / "constraint_funnel_2km.csv", index=False)

# ------------------------------------------------------------- robustness, base case
b = all_sites[BASE_KM].copy()
y, x = np.divmod(np.ravel_multi_index(rasterio.transform.rowcol(T, b.geometry.x.values, b.geometry.y.values), (H, W)), W)
Xs = np.column_stack([layers[n][y, x] for n in names])
rng_np = np.random.default_rng(42)
draws = rng_np.dirichlet(wv * 60, 2000)
ranks = (-(Xs @ draws.T)).argsort(axis=0).argsort(axis=0)
limit = min(TOP_N, len(b))
b["top50_stability_pct"] = ((ranks < limit).mean(axis=1) * 100).round(1)
b["passes_tableC_soil_limit"] = b.catch_soil_loss_tableC <= LIM["soil"]
b["priority"] = np.select([b["rank"] <= TOP_N, b["rank"] <= TOP_N * 2], ["Priority 1", "Priority 2"], "Reserve list")
b.to_file(OUT / "sites_2km_final.gpkg", driver="GPKG")
b.drop(columns="geometry").to_csv(OUT / "sites_2km_final.csv", index=False)
top = b.head(TOP_N)
top.drop(columns="geometry").to_csv(OUT / "top50_sites_2km.csv", index=False)
top.drop(columns="geometry").to_excel(OUT / "top50_sites_2km.xlsx", index=False)

# ------------------------------------------------------------- water-access benefit
mask = d_water == 0
mask[y[:limit], x[:limit]] = True
d_after = ndi.distance_transform_edt(~mask, sampling=CELL)
save("dist_water_after_top50_m", np.where(aoi, d_after, np.nan))
el = gpd.read_file(V / "elephant_presence_clean.gpkg")
ec, er = (~T) * (el.geometry.x.values, el.geometry.y.values)
ec, er = ec.astype(int), er.astype(int)
k = (ec >= 0) & (ec < W) & (er >= 0) & (er < H)
ec, er = ec[k], er[k]
k = aoi[er, ec]
ec, er = ec[k], er[k]
benefit = {f"{t}km": {"reserve_before_pct": float((d_water[aoi] <= t * 1000).mean() * 100),
                      "reserve_after_pct": float((d_after[aoi] <= t * 1000).mean() * 100),
                      "elephant_records_before_pct": float((d_water[er, ec] <= t * 1000).mean() * 100),
                      "elephant_records_after_pct": float((d_after[er, ec] <= t * 1000).mean() * 100)} for t in (1, 2, 3)}
report = {
    "ahp_consistency_ratio": cr, "limits": LIM, "site_spacing_m": SPACING_M,
    "base_sites": len(b), "top50_ranges": top["range"].value_counts().to_dict(),
    "top50_structures": top.structure.value_counts().to_dict(), "top50_zones": top.zone.value_counts().to_dict(),
    "top50_score_range": [float(top.score.min()), float(top.score.max())],
    "top50_stability_median": float(top.top50_stability_pct.median()),
    "top50_stable_ge80": int((top.top50_stability_pct >= 80).sum()),
    "top50_pass_tableC": int(top.passes_tableC_soil_limit.sum()),
    "top50_total_yield_Mm3": float(top.yield_m3_yr.sum() / 1e6),
    "top50_min_village_km": float(top.village_km.min()),
    "water_access": benefit,
}
json.dump(report, open(OUT / "model_report.json", "w"), indent=1, default=str)
print(weights.round(3).to_string())
print(summary.round(2).to_string())
print(pd.DataFrame(funnel).round(1).to_string())
print(json.dumps(report, indent=1, default=str))
print(top[["rank", "score", "range", "compartment", "zone", "structure", "catchment_ha", "slope_deg", "village_km", "water_km", "top50_stability_pct"]].head(12).to_string())
