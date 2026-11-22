"""Align local layers to the 30 m grid; derive hydrology, SCS-CN runoff and RUSLE."""
import json
import warnings
from pathlib import Path

import geopandas as gpd
import numba as nb
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "Data_ELEPHANT_CONSERVATION_PLAN"
B = ROOT / "LEMRU ELEPHANR RESERVE-WATER MANAGEMENT"
G = ROOT / "outputs" / "01_gee"
L = ROOT / "outputs" / "02_layers"
V = ROOT / "outputs" / "02_vectors"
L.mkdir(parents=True, exist_ok=True)
V.mkdir(parents=True, exist_ok=True)
CRS = 32644
CELL = 30.0

with rasterio.open(G / "dem_m.tif") as s:
    PROFILE = s.profile.copy()
    T, H, W = s.transform, s.height, s.width
    dem = s.read(1).astype("float64")
PROFILE.update(dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True)


def gee(name):
    with rasterio.open(G / f"{name}.tif") as s:
        return s.read(1).astype("float32")


def save(name, a):
    with rasterio.open(L / f"{name}.tif", "w", **PROFILE) as d:
        d.write(np.asarray(a, dtype="float32"), 1)


def burn(pairs, fill=0, all_touched=False):
    return rasterize(pairs, out_shape=(H, W), transform=T, fill=fill, dtype="int32", all_touched=all_touched)


def warp(path, resampling=Resampling.nearest):
    out = np.full((H, W), np.nan, dtype="float32")
    with rasterio.open(path) as s:
        src = s.read(1, masked=True).astype("float32").filled(np.nan)
        reproject(src, out, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                  dst_transform=T, dst_crs=f"EPSG:{CRS}", dst_nodata=np.nan, resampling=resampling)
    return out


def drop_small(mask, min_cells=6):
    """Remove connected patches smaller than about 0.5 ha (6 cells at 30 m)."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (sizes >= min_cells)[lab]


# ------------------------------------------------------------------ hydrology
@nb.njit(cache=True)
def priority_flood(z, eps=1e-3):
    h, w = z.shape
    n = h * w
    f = z.ravel().copy()
    closed = np.zeros(n, dtype=np.bool_)
    keys = np.empty(n + 1, dtype=np.float64)
    vals = np.empty(n + 1, dtype=np.int64)
    size = 0
    dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1])

    def _noop():
        return None

    for r in range(h):
        for c in range(w):
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                i = r * w + c
                closed[i] = True
                k = f[i]
                p = size
                size += 1
                while p > 0:
                    q = (p - 1) // 2
                    if keys[q] <= k:
                        break
                    keys[p] = keys[q]
                    vals[p] = vals[q]
                    p = q
                keys[p] = k
                vals[p] = i
    while size > 0:
        i = vals[0]
        zi = keys[0]
        size -= 1
        k = keys[size]
        v = vals[size]
        p = 0
        while True:
            l = 2 * p + 1
            if l >= size:
                break
            if l + 1 < size and keys[l + 1] < keys[l]:
                l += 1
            if keys[l] >= k:
                break
            keys[p] = keys[l]
            vals[p] = vals[l]
            p = l
        if size > 0:
            keys[p] = k
            vals[p] = v
        r, c = i // w, i % w
        for d in range(8):
            rr, cc = r + dr[d], c + dc[d]
            if rr < 0 or cc < 0 or rr >= h or cc >= w:
                continue
            j = rr * w + cc
            if closed[j]:
                continue
            closed[j] = True
            if f[j] <= zi:
                f[j] = zi + eps
            k2 = f[j]
            p = size
            size += 1
            while p > 0:
                q = (p - 1) // 2
                if keys[q] <= k2:
                    break
                keys[p] = keys[q]
                vals[p] = vals[q]
                p = q
            keys[p] = k2
            vals[p] = j
    return f.reshape(h, w)


@nb.njit(cache=True)
def d8_receivers(f, cell):
    h, w = f.shape
    rec = np.full(h * w, -1, dtype=np.int64)
    dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    dist = np.array([1.41421356, 1., 1.41421356, 1., 1., 1.41421356, 1., 1.41421356]) * cell
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            best, target = 0.0, -1
            for d in range(8):
                s = (f[r, c] - f[r + dr[d], c + dc[d]]) / dist[d]
                if s > best:
                    best, target = s, (r + dr[d]) * w + c + dc[d]
            rec[r * w + c] = target
    return rec


@nb.njit(cache=True)
def accumulate(order, rec, weight):
    acc = weight.copy()
    for i in order:
        j = rec[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc


@nb.njit(cache=True)
def strahler(order, rec, stream):
    n = rec.size
    so = np.zeros(n, dtype=np.int8)
    mx = np.zeros(n, dtype=np.int8)
    cnt = np.zeros(n, dtype=np.int8)
    for i in order:
        if not stream[i]:
            continue
        so[i] = 1 if mx[i] == 0 else (mx[i] + 1 if cnt[i] >= 2 else mx[i])
        j = rec[i]
        if j >= 0 and stream[j]:
            if so[i] > mx[j]:
                mx[j], cnt[j] = so[i], 1
            elif so[i] == mx[j]:
                cnt[j] += 1
    return so


print("aligning local layers", flush=True)
ranges = gpd.read_file(B / "Lemru_Elephant_Conservation_Zone-Range_Boundary.shp").to_crs(CRS)
ranges["geometry"] = ranges.geometry.force_2d().make_valid()
ranges["range_id"] = np.arange(1, len(ranges) + 1)
ranges[["range_id", "RANGENAME", "DIVISION", "geometry"]].to_file(V / "ranges.gpkg")
range_id = burn(zip(ranges.geometry, ranges.range_id))
aoi = range_id > 0
save("aoi", aoi)
save("range_id", range_id)

comp = gpd.read_file(B / "Compartment_boundary.shp").to_crs(CRS)
comp["geometry"] = comp.geometry.force_2d().make_valid()
comp["comp_id"] = np.arange(1, len(comp) + 1)
comp[["comp_id", "Range_Name", "Beat_Name", "Old_Compt", "New_Compt", "Legal_Stat", "geometry"]].to_file(V / "compartments.gpkg")
save("compartment_id", burn(zip(comp.geometry, comp.comp_id)))

zon = gpd.read_file(A / "Lemru Elephant Reserve - Conservation Unit - Zonation-compartment.shp").to_crs(CRS)
zon["geometry"] = zon.geometry.force_2d().make_valid()
ZMAP = {"Elephant Conservation Zone": 3, "Human-Elephant Management Zone": 2, "Human-Elephant Co-existense Zone": 1}
zone = burn(zip(zon.geometry, zon.Zonation.map(ZMAP).fillna(0).astype(int)))
save("zonation", zone)

lu = gpd.read_file(B / "Landuse_Land_cover.shp", columns=["Class", "Shape_Area"]).to_crs(CRS)
LUMAP = {"Moderately dense forest": 1, "Roads": 2, "Barren land": 3, "Waterbodies/Streams": 4, "Industry": 5,
         "Build-up_Urban Area": 6, "Build-up_ Rural": 7, "Agriculture": 8, "Mine": 9, "Fallow land": 10}
lu["code"] = lu.Class.map(LUMAP)
print("unmapped LULC:", int(lu.code.isna().sum()), flush=True)
lu = lu.dropna(subset=["code"]).sort_values("Shape_Area", ascending=False)
lulc = burn(zip(lu.geometry.force_2d(), lu.code.astype(int)))
# The built-up classes contain thousands of single-pixel polygons (classification speckle).
# Keep a patch only when its connected polygons hold at least 0.5 ha of mapped built-up area.
bu = lu[lu.code.isin([5, 6, 7])]
raw_built = burn(zip(bu.geometry.force_2d(), [1] * len(bu)), all_touched=True) > 0
blab, nblab = ndi.label(raw_built, structure=np.ones((3, 3)))
pt = bu.geometry.representative_point()
pc, pr = (~T) * (pt.x.values, pt.y.values)
pc, pr = pc.astype(int), pr.astype(int)
pk = (pc >= 0) & (pc < W) & (pr >= 0) & (pr < H)
patch_area = np.bincount(blab[pr[pk], pc[pk]], weights=bu.geometry.area.values[pk], minlength=nblab + 1)
patch_area[0] = 0
built_touch = (patch_area >= 5000)[blab]
BUILT_QA = {"built_polygons": len(bu), "built_patches_raw": int(nblab), "built_patches_kept": int((patch_area >= 5000).sum())}
save("lulc", lulc)
del lu

PREF = {"Very highly preferred": 5, "Highly preferred": 4, "Moderately preferred": 3, "Low preferred": 2,
        "Very low preferred": 1, "No occurrence": 1}
fc = gpd.read_file(B / "Forest_cover_preference.shp").to_crs(CRS)
ft = gpd.read_file(B / "Forest_types_preference.shp").to_crs(CRS)
fc_s = burn([(g, PREF[p]) for g, p in zip(fc.geometry.force_2d(), fc.Preference) if p in PREF])
ft_s = burn([(g, PREF[p]) for g, p in zip(ft.geometry.force_2d(), ft.Preferrenc) if p in PREF])
save("pref_forest_cover", np.where(fc_s > 0, fc_s, np.nan))
save("pref_forest_type", np.where(ft_s > 0, ft_s, np.nan))

ev = warp(B / "elevation_preference.tif")
save("pref_elevation", 6 - ev)  # source class 1 = very highly preferred
tr = warp(B / "terrain_intensity_slope_preference.tif")
tmap = np.full(14, np.nan)
tmap[[1, 2]] = 5
tmap[[3, 4]] = 4
tmap[5] = 3
tmap[[6, 7, 8]] = 2
tmap[[9, 10, 11]] = 1
save("pref_terrain", np.where(np.isfinite(tr), tmap[np.nan_to_num(tr, nan=13).astype(int).clip(0, 13)], np.nan))
save("habitat_suitability", warp(B / "Elpehant habitat suitability.tif"))
legacy_slope = warp(A / "slope_rec.tif", Resampling.bilinear)
save("legacy_slope_input", legacy_slope)
save("legacy_terrain_intensity", warp(A / "terrain intensity.tif"))

# roads, mines, villages, buildings, elephants
roads = gpd.read_file(A / "road network.shp").to_crs(CRS)
roads["geometry"] = roads.geometry.force_2d()
rcode = roads.CATEGORY.map({"Metalled Road": 1, "Unmetalled Road": 2, "Cart Track": 3}).fillna(3).astype(int)
save("dist_road_metalled_m", ndi.distance_transform_edt(burn(zip(roads[rcode == 1].geometry, [1] * int((rcode == 1).sum())), all_touched=True) == 0, sampling=CELL))
save("dist_road_unmetalled_m", ndi.distance_transform_edt(burn(zip(roads[rcode == 2].geometry, [1] * int((rcode == 2).sum())), all_touched=True) == 0, sampling=CELL))
save("dist_road_any_m", ndi.distance_transform_edt(burn(zip(roads.geometry, [1] * len(roads)), all_touched=True) == 0, sampling=CELL))

mines = gpd.read_file(A / "mining_sites.shp").to_crs(CRS)
mines["geometry"] = mines.geometry.force_2d().make_valid()
mine = burn(zip(mines.geometry, [1] * len(mines)), all_touched=True) > 0
mine |= drop_small(lulc == 9)  # single-pixel "Mine" polygons are classification speckle
save("dist_mine_m", ndi.distance_transform_edt(~mine, sampling=CELL))

vil = gpd.read_file(A / "village location.shp").to_crs(CRS)
vil["geometry"] = vil.geometry.force_2d()
nv = len(vil)
vil = vil.drop_duplicates("geometry")
vil[["NAME", "CENSUSCODE", "RANGE", "geometry"]].to_file(V / "villages_dedup.gpkg")
inv = ~T
vr = np.array([inv * (x, y) for x, y in zip(vil.geometry.x, vil.geometry.y)]).astype(int)
ok = (vr[:, 0] >= 0) & (vr[:, 0] < W) & (vr[:, 1] >= 0) & (vr[:, 1] < H)
village = np.zeros((H, W), bool)
village[vr[ok, 1], vr[ok, 0]] = True

# Settlement rule (user decision): supplied village locations only.
dist_village = ndi.distance_transform_edt(~village, sampling=CELL)
save("dist_village_only_m", dist_village)

crop = drop_small(np.isin(lulc, [8, 10]) | (gee("worldcover") == 40))
save("dist_cropland_m", ndi.distance_transform_edt(~crop, sampling=CELL))

el = gpd.read_file(A / "elephant_occurrence1.shp")
el = el[(el.geometry.x > 80) & (el.geometry.y > 20) & (el.occurence == 1)].set_crs(4326, allow_override=True).to_crs(CRS)
el["geometry"] = el.geometry.force_2d()
el = el.drop_duplicates("geometry")
el[["Year", "RANGE_NAME", "geometry"]].to_file(V / "elephant_presence_clean.gpkg")
ec, er = inv * (el.geometry.x.values, el.geometry.y.values)
ec, er = ec.astype(int), er.astype(int)
k = (ec >= 0) & (ec < W) & (er >= 0) & (er < H)
ecount = np.zeros((H, W), dtype="float32")
np.add.at(ecount, (er[k], ec[k]), 1)
kde = ndi.gaussian_filter(ecount, sigma=1500 / CELL, mode="constant") / (CELL / 1000) ** 2  # records / km2
save("elephant_density", kde)
n_el_aoi = int(ecount[aoi].sum())

# ------------------------------------------------------------------ terrain and flow
print("hydrology", flush=True)
smooth = ndi.gaussian_filter(dem, 1.0)  # reduces canopy/radar noise in the 30 m surface model
gy, gx = np.gradient(smooth, CELL)
slope_rad = np.arctan(np.hypot(gx, gy))
slope_deg = np.degrees(slope_rad)
save("dem_m", dem)
save("slope_deg", slope_deg)

filled = priority_flood(smooth)
save("depth_filled_m", filled - smooth)
rec = d8_receivers(filled, CELL)
order = np.argsort(filled.ravel(), kind="stable")[::-1]
acc = accumulate(order, rec, np.ones(H * W))
upa_ha = (acc * CELL ** 2 / 1e4).reshape(H, W)
save("upstream_area_ha", upa_ha)
stream = upa_ha.ravel() >= 5.0
so = strahler(order, rec, stream).reshape(H, W)
save("stream_order", so)
dist_stream = ndi.distance_transform_edt(so == 0, sampling=CELL)
save("dist_stream_m", dist_stream)
sca = acc.reshape(H, W) * CELL
twi = np.log(sca / np.maximum(np.tan(slope_rad), 0.001))
save("twi", twi)

# ------------------------------------------------------------------ existing water
occ, sea = gee("jrc_occurrence"), gee("jrc_seasonality")
mndwi = gee("mndwi_dry_marmay")
wc = gee("worldcover")
perennial = (sea >= 9) | (occ >= 50) | ((mndwi > 0.15) & (wc == 80))
lab, n = ndi.label(perennial)
sizes = ndi.sum(perennial, lab, np.arange(1, n + 1))
keep = np.zeros(n + 1, bool)
keep[1:] = sizes >= 3  # at least ~0.27 ha
perennial = keep[lab]
any_water = perennial | (np.nan_to_num(occ) > 0) | (wc == 80)
save("water_perennial", perennial)
save("water_any", any_water)
save("dist_perennial_water_m", ndi.distance_transform_edt(~perennial, sampling=CELL))

# ------------------------------------------------------------------ rainfall and soil gap-fill
def fill_nearest(a):
    idx = ndi.distance_transform_edt(~np.isfinite(a), return_distances=False, return_indices=True)
    return a[tuple(idx)]


rain = gee("rain_annual_mm")
for n_ in ["rain_annual_mm", "rain_cv_pct", "rain_monsoon_mm", "rain_dry_febmay_mm"]:
    save(n_, gee(n_))
sand, clay, oc = [fill_nearest(gee(x)) for x in ("soil_sand_pct", "soil_clay_pct", "soil_oc_gkg")]
silt = np.clip(100 - sand - clay, 0, 100)
save("soil_sand_pct", sand)
save("soil_clay_pct", clay)
save("soil_silt_pct", silt)
save("soil_oc_gkg", oc)
ndvi = gee("ndvi_median")
save("ndvi_median", ndvi)
save("ndvi_dry_marmay", gee("ndvi_dry_marmay"))

# ------------------------------------------------------------------ RUSLE
print("RUSLE and runoff", flush=True)
# R: Renard & Freimund (1994), P > 850 mm, MJ mm ha-1 h-1 yr-1
R = np.where(rain > 850, 587.8 - 1.219 * rain + 0.004105 * rain ** 2, 0.0483 * rain ** 1.61)
# K: Williams EPIC (Sharpley & Williams 1990), converted to t ha h ha-1 MJ-1 mm-1
ocp = np.clip(oc / 10.0, 0, 4.0)  # organic carbon %, capped at the equation's practical range
sn = 1 - sand / 100
K = (0.2 + 0.3 * np.exp(-0.256 * sand * (1 - silt / 100))) \
    * (silt / np.maximum(clay + silt, 1)) ** 0.3 \
    * (1 - 0.25 * ocp / (ocp + np.exp(3.72 - 2.95 * ocp))) \
    * (1 - 0.7 * sn / (sn + np.exp(-5.51 + 22.9 * sn))) * 0.1317
# LS: Moore & Burch (1986), slope length capped at 300 m
As = np.minimum(acc.reshape(H, W) * CELL, 300.0)
LS = (As / 22.13) ** 0.4 * (np.sin(slope_rad) / 0.0896) ** 1.3
# C: van der Knijff et al. (2000)
nd = np.clip(np.nan_to_num(ndvi), 0, 0.95)
C = np.clip(np.exp(-2 * nd / (1 - nd)), 0.001, 1)
# P: Wischmeier & Smith contour factors on cropland, 1 elsewhere
sp = np.tan(slope_rad) * 100
Pf = np.ones((H, W), dtype="float32")
cr = np.isin(lulc, [8, 10]) | (wc == 40)
Pf[cr] = np.select([sp[cr] < 2, sp[cr] < 7, sp[cr] < 12, sp[cr] < 18, sp[cr] < 24], [0.6, 0.5, 0.6, 0.8, 0.9], 1.0)
soil_loss = R * K * LS * C * Pf
nonsoil = perennial | np.isin(wc, [50, 80]) | np.isin(lulc, [4, 5, 6, 7])
soil_loss = np.where(nonsoil, 0, soil_loss)
for n_, a in [("rusle_R", R), ("rusle_K", K), ("rusle_LS", LS), ("rusle_C", C), ("rusle_P", Pf), ("rusle_A_tha", soil_loss)]:
    save(n_, a)
# Sensitivity: land-cover lookup C factor (the NDVI equation is known to be high under dense canopy)
C_table = np.select([lulc == 1, np.isin(lulc, [8]), lulc == 10, lulc == 3], [0.006, 0.25, 0.30, 0.45], C)
soil_loss_alt = np.where(nonsoil, 0, R * K * LS * C_table * Pf)
save("rusle_A_tableC_tha", soil_loss_alt)

# SCS-CN: daily reserve rainfall gives the runoff ratio of each curve number
daily = pd.read_csv(G / "chirps_daily_aoi_mean.csv").rain_mm.fillna(0).values
def runoff_ratio(cn):
    if cn >= 99.5:
        return 1.0
    S = 25400 / cn - 254
    q = np.where(daily > .2 * S, (daily - .2 * S) ** 2 / (daily + .8 * S), 0)
    return q.sum() / daily.sum()

hsg = np.select([(clay < 10) & (sand > 85), (clay < 20) & (sand > 50), clay < 40], [0, 1, 2], 3).astype("int8")
CN = {10: (30, 55, 70, 77), 20: (35, 56, 70, 77), 30: (49, 69, 79, 84), 40: (67, 78, 85, 89),
      50: (77, 85, 90, 92), 60: (77, 86, 91, 94), 80: (100, 100, 100, 100), 90: (98, 98, 98, 98), 95: (98, 98, 98, 98)}
cn = np.full((H, W), 70, dtype="float32")
for cls, vals in CN.items():
    m = wc == cls
    cn[m] = np.array(vals, dtype="float32")[hsg[m]]
lut = np.array([runoff_ratio(x) if x > 0 else 0 for x in range(0, 101)])
ratio = lut[cn.astype(int)]
Q = rain * ratio
save("hsg", hsg + 1)
save("curve_number", cn)
save("runoff_mm", Q)

# upstream means, routed down the same D8 network
up_q = accumulate(order, rec, Q.ravel().astype("float64")) / acc
up_a = accumulate(order, rec, soil_loss.ravel().astype("float64")) / acc
save("catchment_runoff_mm", up_q.reshape(H, W))
save("catchment_soil_loss_tha", up_a.reshape(H, W))
up_alt = accumulate(order, rec, soil_loss_alt.ravel().astype("float64")) / acc
save("catchment_soil_loss_tableC_tha", up_alt.reshape(H, W))
np.save(L / "d8_receivers.npy", rec.astype("int32"))

# ------------------------------------------------------------------ QA
merit = gee("merit_upa_km2")
mk = aoi & (merit > 50) & (upa_ha / 100 > 5)
small = aoi & (merit >= 1) & (merit <= 10) & (merit == ndi.maximum_filter(merit, 7))
ours = ndi.maximum_filter(upa_ha / 100, 7)[small]
qa = {
    **BUILT_QA,
    "merit_check_1_10km2": {"n": int(small.sum()), "median_ratio": float(np.median(ours / merit[small])),
                            "within_factor2_pct": float((np.abs(np.log2(ours / merit[small])) < 1).mean() * 100)},
    "A_mean_tableC_tha": float(soil_loss_alt[aoi].mean()),
    "villages_raw": nv, "villages_dedup": len(vil),
    "elephant_presence_unique_all": len(el), "elephant_presence_unique_aoi": n_el_aoi,
    "aoi_km2": float(aoi.sum() * CELL ** 2 / 1e6),
    "area_beyond_village_buffer_pct": {f"{d}km": float((dist_village[aoi] >= d * 1000).mean() * 100) for d in (1, 2, 3, 4, 5)},
    "rain_aoi": [float(rain[aoi].min()), float(rain[aoi].mean()), float(rain[aoi].max())],
    "perennial_water_km2_aoi": float(perennial[aoi].sum() * CELL ** 2 / 1e6),
    "slope_deg_pct": [float(x) for x in np.percentile(slope_deg[aoi], [10, 50, 90])],
    "R_mean": float(R[aoi].mean()), "K_mean": float(K[aoi].mean()), "LS_median": float(np.median(LS[aoi])),
    "C_median": float(np.median(C[aoi])),
    "A_mean_tha": float(soil_loss[aoi].mean()), "A_pct": [float(x) for x in np.percentile(soil_loss[aoi], [50, 90, 99])],
    "A_class_pct": {k_: float(v * 100) for k_, v in zip(["<5", "5-10", "10-20", "20-40", ">40"], np.histogram(soil_loss[aoi], [0, 5, 10, 20, 40, 1e9])[0] / aoi.sum())},
    "runoff_mm_mean": float(Q[aoi].mean()), "runoff_ratio_mean": float(ratio[aoi].mean()),
    "hsg_counts": {int(a): int(b) for a, b in zip(*np.unique(hsg[aoi] + 1, return_counts=True))},
    "upa_vs_merit_log_corr": float(np.corrcoef(np.log(upa_ha[mk] / 100), np.log(merit[mk]))[0, 1]) if mk.sum() > 10 else None,
    "slope_vs_supplied_corr": float(np.corrcoef(slope_deg[aoi & np.isfinite(legacy_slope)], legacy_slope[aoi & np.isfinite(legacy_slope)])[0, 1]),
}
json.dump(qa, open(L / "qa_preprocess.json", "w"), indent=1)
print(json.dumps(qa, indent=1), flush=True)
