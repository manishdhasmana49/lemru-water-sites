"""Publication maps (300 dpi PNG + PDF) for every variable and result."""
import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from scipy.ndimage import maximum_filter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
L, V, R = [ROOT / "outputs" / x for x in ("02_layers", "02_vectors", "03_results")]
G = ROOT / "outputs" / "01_gee"
OUT = ROOT / "outputs" / "04_maps"
(OUT / "png").mkdir(parents=True, exist_ok=True)
(OUT / "pdf").mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f2a24", "#5c6861", "#d9dfdb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": .6,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
STEP = 3

with rasterio.open(L / "aoi.tif") as s:
    aoi_full = s.read(1) > 0
    TR = s.transform
ranges = gpd.read_file(V / "ranges.gpkg")
villages = gpd.read_file(V / "villages_dedup.gpkg")
sites = gpd.read_file(R / "sites_2km_final.gpkg")
x0, y0, x1, y1 = ranges.total_bounds
PAD = 2000
EXT = (x0 - PAD, x1 + PAD, y0 - PAD, y1 + PAD)
villages_v = villages.cx[EXT[0]:EXT[1], EXT[2]:EXT[3]]
FULL_EXT = (TR.c, TR.c + TR.a * aoi_full.shape[1], TR.f + TR.e * aoi_full.shape[0], TR.f)
to_utm = Transformer.from_crs(4326, 32644, always_xy=True)
to_ll = Transformer.from_crs(32644, 4326, always_xy=True)
aoi_small = aoi_full[::STEP, ::STEP]


def load(name, folder=L):
    with rasterio.open(folder / f"{name}.tif") as s:
        a = s.read(1)
    if name == "stream_order":  # keep one-cell-wide channels visible after downsampling
        a = maximum_filter(a, STEP)
    a = a[::STEP, ::STEP].astype("float32")
    # Open water hides land runoff, and flat reservoir DEM cells produce artificial straight flow lines.
    if name in ("runoff_mm", "stream_order", "upstream_area_ha", "twi"):
        with rasterio.open(L / "water_perennial.tif") as s:
            a = np.where(s.read(1)[::STEP, ::STEP] > 0, np.nan, a)
    return np.where(aoi_small, a, np.nan)


def graticule(ax):
    lo0, la0 = to_ll.transform(EXT[0], EXT[2])
    lo1, la1 = to_ll.transform(EXT[1], EXT[3])
    lons = np.arange(np.ceil(lo0 * 5) / 5, lo1, .2)
    lats = np.arange(np.ceil(la0 * 10) / 10, la1, .1)
    xt, yt = [], []
    for lo in lons:
        xx, yy = to_utm.transform(np.full(30, lo), np.linspace(la0 - .05, la1 + .05, 30))
        ax.plot(xx, yy, color=GRID, lw=.35, zorder=1)
        xt.append(np.interp(EXT[2], yy, xx))
    for la in lats:
        xx, yy = to_utm.transform(np.linspace(lo0 - .05, lo1 + .05, 30), np.full(30, la))
        ax.plot(xx, yy, color=GRID, lw=.35, zorder=1)
        yt.append(np.interp(EXT[0], xx, yy))
    ax.set_xticks(xt, [f"{v:.1f}°E" for v in lons], fontsize=6.5)
    ax.set_yticks(yt, [f"{v:.1f}°N" for v in lats], fontsize=6.5, rotation=90, va="center")
    ax.tick_params(length=2, width=.4, pad=1.5)


def furniture(ax):
    bx, by = EXT[1] - 25000, EXT[2] + 2600
    for i in range(4):
        ax.add_patch(Rectangle((bx + i * 5000, by), 5000, 650, fc=INK if i % 2 == 0 else "white", ec=INK, lw=.4, zorder=8))
    for d in (0, 10, 20):
        ax.text(bx + d * 1000, by + 1000, f"{d}" + (" km" if d == 20 else ""), ha="center", va="bottom", fontsize=6, zorder=8)
    ax.annotate("N", xy=(.965, .95), xytext=(.965, .83), xycoords="axes fraction", ha="center", va="center", fontsize=8,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, lw=.9), zorder=8)


def frame(ax):
    ranges.boundary.plot(ax=ax, color="#2b332e", lw=.45, zorder=5)
    ax.set_xlim(EXT[0], EXT[1])
    ax.set_ylim(EXT[2], EXT[3])
    ax.set_aspect("equal")
    graticule(ax)
    furniture(ax)


def finish(fig, name):
    fig.savefig(OUT / "png" / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "pdf" / f"{name}.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def base(title, source):
    fig = plt.figure(figsize=(7.4, 5.0))
    ax = fig.add_axes([.06, .19, .91, .72])
    fig.text(.06, .945, title, fontsize=10.5, fontweight="bold", va="center")
    fig.text(.06, .022, f"Source: {source}. UTM 44N (WGS 84), 30 m analysis grid.", fontsize=5.8, color=MUTED)
    return fig, ax


def continuous(name, title, unit, source, layer=None, cmap="Blues", log=False, q=(2, 98), folder=L, vmin=None, vmax=None, extras=None):
    a = load(layer or name, folder)
    f = a[np.isfinite(a)]
    lo, hi = np.percentile(f[f > 0] if log else f, q)
    lo, hi = lo if vmin is None else vmin, hi if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-3
    fig, ax = base(title, source)
    im = ax.imshow(a, extent=FULL_EXT, origin="upper", cmap=cmap, interpolation="nearest", zorder=2,
                   norm=LogNorm(max(lo, 1e-3), hi) if log else Normalize(lo, hi))
    frame(ax)
    if extras:
        extras(ax)
    cax = fig.add_axes([.27, .105, .46, .022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.outline.set_linewidth(.4)
    cb.ax.tick_params(labelsize=6.5, length=2, width=.4)
    cb.set_label(unit, fontsize=7, labelpad=2)
    finish(fig, name)


def classed(name, title, source, layer, classes, folder=L, transform=None, extras=None):
    """classes: list of (value, label, colour)."""
    a = load(layer, folder)
    if transform:
        a = transform(a)
    vals = [c[0] for c in classes]
    cm = ListedColormap([c[2] for c in classes])
    norm = BoundaryNorm([v - .5 for v in vals] + [vals[-1] + .5], cm.N)
    fig, ax = base(title, source)
    ax.imshow(np.where(np.isin(a, vals), a, np.nan), extent=FULL_EXT, origin="upper", cmap=cm, norm=norm, interpolation="nearest", zorder=2)
    frame(ax)
    if extras:
        extras(ax)
    total = np.isin(a, vals).sum()
    show_pct = layer != "stream_order"  # channel widening for display would distort class shares
    h = [Patch(fc=c[2], ec="#6b746e", lw=.3, label=f"{c[1]} ({np.sum(a == c[0]) / total * 100:.1f}%)" if show_pct else c[1]) for c in classes]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(.51, .045), ncol=min(4, len(h)), fontsize=6.3, frameon=False,
               handlelength=1.1, handleheight=.9, columnspacing=1.1)
    finish(fig, name)


PREF_COL = ["#edf3df", "#c5dd9e", "#8fbf63", "#4f9345", "#1b5e33"]
PREF = [(1, "Very low", PREF_COL[0]), (2, "Low", PREF_COL[1]), (3, "Moderate", PREF_COL[2]), (4, "High", PREF_COL[3]), (5, "Very high", PREF_COL[4])]
CH = "CHIRPS v2.0 daily, 1996–2025"
LOCAL = "Lemru Elephant Reserve project data"
S2 = "Sentinel-2 SR, 2023–2025"
OLM = "OpenLandMap soil, 0 cm"
DEM = "Copernicus GLO-30 DEM"

continuous("01_elevation", "Elevation", "Elevation (m a.s.l.)", DEM, "dem_m", "YlOrBr")
continuous("02_slope", "Slope", "Slope (degrees)", DEM, "slope_deg", "Oranges", vmin=0, vmax=35)
continuous("03_rainfall_annual", "Mean annual rainfall", "Rainfall (mm/yr)", CH, "rain_annual_mm", "Blues", q=(0, 100))
continuous("04_rainfall_monsoon", "Mean monsoon rainfall (June–September)", "Rainfall (mm)", CH, "rain_monsoon_mm", "Blues", q=(0, 100))
continuous("05_rainfall_dry", "Mean dry-season rainfall (February–May)", "Rainfall (mm)", CH, "rain_dry_febmay_mm", "Blues", q=(0, 100))
continuous("06_rainfall_variability", "Inter-annual rainfall variability", "Coefficient of variation (%)", CH, "rain_cv_pct", "Purples", q=(0, 100))
continuous("07_rusle_R", "RUSLE R: rainfall erosivity", "R (MJ mm ha⁻¹ h⁻¹ yr⁻¹)", CH + "; Renard & Freimund (1994)", "rusle_R", "Blues", q=(0, 100))
continuous("08_rusle_K", "RUSLE K: soil erodibility", "K (t ha h ha⁻¹ MJ⁻¹ mm⁻¹)", OLM + "; EPIC (Williams)", "rusle_K", "YlOrBr", q=(1, 99))
continuous("09_rusle_LS", "RUSLE LS: slope length and steepness", "LS (dimensionless)", DEM + "; Moore & Burch (1986)", "rusle_LS", "Oranges", log=True, q=(5, 99))
continuous("10_rusle_C", "RUSLE C: cover management", "C (dimensionless)", S2 + "; van der Knijff et al. (2000)", "rusle_C", "Greens_r", log=True, q=(2, 98))
continuous("11_rusle_P", "RUSLE P: support practice", "P (dimensionless)", LOCAL + "; Wischmeier & Smith (1978)", "rusle_P", "Purples", vmin=.5, vmax=1)
SOIL = [(1, "<5 slight", "#fff1dc"), (2, "5–10 moderate", "#fdd08e"), (3, "10–20 high", "#f8a04b"), (4, "20–40 very high", "#d9601a"), (5, ">40 severe", "#8a2f07")]
tc = lambda a: np.where(np.isfinite(a), np.digitize(a, [5, 10, 20, 40]) + 1, np.nan)
classed("12_rusle_soil_loss", "RUSLE soil loss (t ha⁻¹ yr⁻¹)", "RUSLE, NDVI-based C factor", "rusle_A_tha", SOIL, transform=tc)
classed("13_rusle_soil_loss_alt_C", "RUSLE soil loss, land-cover C sensitivity (t ha⁻¹ yr⁻¹)", "RUSLE, land-cover lookup C factor", "rusle_A_tableC_tha", SOIL, transform=tc)
classed("14_catchment_soil_loss", "Mean upstream catchment soil loss (t ha⁻¹ yr⁻¹)", "RUSLE routed on D8 flow network", "catchment_soil_loss_tha", SOIL, transform=tc)
continuous("15_soil_sand", "Soil sand content", "Sand (%)", OLM, "soil_sand_pct", "YlOrBr", q=(0, 100))
continuous("16_soil_clay", "Soil clay content", "Clay (%)", OLM, "soil_clay_pct", "Oranges", q=(0, 100))
continuous("17_soil_silt", "Soil silt content", "Silt (%)", OLM, "soil_silt_pct", "Purples", q=(0, 100))
continuous("18_soil_organic_carbon", "Soil organic carbon", "Organic carbon (g/kg)", OLM, "soil_oc_gkg", "Greens", q=(1, 99))
continuous("19_ndvi_annual", "Median NDVI", "NDVI", S2, "ndvi_median", "Greens", vmin=0, vmax=.85)
continuous("20_ndvi_dry", "Dry-season NDVI (March–May)", "NDVI", S2, "ndvi_dry_marmay", "Greens", vmin=0, vmax=.85)
continuous("21_curve_number", "SCS curve number", "Curve number", "ESA WorldCover 2021; " + OLM + "; USDA TR-55", "curve_number", "Blues", q=(0, 100))
continuous("22_runoff", "Mean annual surface runoff", "Runoff (mm/yr)", "SCS-CN with " + CH, "runoff_mm", "Blues", q=(1, 99))
continuous("23_twi", "Topographic wetness index", "TWI", DEM, "twi", "Blues", q=(2, 99))
continuous("24_upstream_area", "Upstream contributing area", "Area (ha, log scale)", DEM + "; D8 flow routing", "upstream_area_ha", "Blues", log=True, q=(40, 99.9))
ORD = [(i, str(i), c) for i, c in zip(range(1, 8), ["#d6e6f4", "#abd0e6", "#6faed6", "#3d8dc3", "#1a67ae", "#084a91", "#08306b"])]
classed("25_stream_order", "Strahler stream order (catchment ≥ 5 ha)", DEM + "; D8 flow routing", "stream_order", ORD)
continuous("26_distance_perennial_water", "Distance to perennial water", "Distance (km)", "JRC Global Surface Water; Sentinel-2 MNDWI; ESA WorldCover", "dist_perennial_water_m", "Blues_r", vmin=0, vmax=8000)


def villages_layer(ax):
    villages_v.plot(ax=ax, color=INK, markersize=2, zorder=6, linewidth=0)


continuous("27_distance_villages", "Distance to village locations", "Distance (m)", LOCAL, "dist_village_only_m", "Purples", vmin=0, vmax=5000, extras=villages_layer)
continuous("28_distance_roads", "Distance to roads", "Distance (m)", LOCAL, "dist_road_any_m", "Purples", vmin=0, vmax=3000)
continuous("29_distance_cropland", "Distance to cropland", "Distance (m)", LOCAL + "; ESA WorldCover", "dist_cropland_m", "Purples", vmin=0, vmax=3000)
continuous("30_distance_mines", "Distance to mining sites", "Distance (km)", LOCAL, "dist_mine_m", "Purples", vmin=0, vmax=25000)
LU = [(1, "Forest", "#2f7d4a"), (3, "Barren", "#c9b48a"), (4, "Water/streams", "#2f7fc1"), (8, "Agriculture", "#e5c33f"),
      (10, "Fallow", "#f0e2a2"), (2, "Roads", "#7c7c7c"), (6, "Built-up urban", "#b4262c"), (7, "Built-up rural", "#e4572e"), (5, "Industry", "#7a3e9d"), (9, "Mine", "#2b2b2b")]
remap = {c[0]: i + 1 for i, c in enumerate(LU)}
classed("31_land_use", "Land use and land cover", LOCAL, "lulc", [(remap[c[0]], c[1], c[2]) for c in LU],
        transform=lambda a: np.vectorize(lambda v: remap.get(int(v), np.nan) if np.isfinite(v) else np.nan, otypes=[float])(a))
classed("32_pref_forest_cover", "Elephant preference: forest cover", LOCAL, "pref_forest_cover", PREF)
classed("33_pref_forest_type", "Elephant preference: forest type", LOCAL, "pref_forest_type", PREF)
classed("34_pref_elevation", "Elephant preference: elevation", LOCAL, "pref_elevation", PREF)
classed("35_pref_terrain", "Elephant preference: terrain", LOCAL, "pref_terrain", PREF)
classed("36_habitat_suitability", "Elephant habitat suitability", LOCAL, "habitat_suitability", PREF)
classed("37_zonation", "Reserve management zonation", LOCAL, "zonation",
        [(1, "Human-Elephant Co-existence", "#e8c37a"), (2, "Human-Elephant Management", "#8fbf63"), (3, "Elephant Conservation", "#1b5e33")])
continuous("38_elephant_density", "Elephant record density (1.5 km kernel)", "Records per km²", LOCAL + "; cleaned presence records", "elephant_density", "Greens", q=(2, 99))
continuous("39_suitability_score", "Composite water-site suitability", "AHP suitability score (0–100)", "This study", "suitability_score", "YlGnBu", folder=R, q=(1, 99.5))

# Scenario small multiples
fig, axs = plt.subplots(3, 2, figsize=(7.4, 7.0))
axs = axs.ravel()
fig.subplots_adjust(left=.03, right=.98, top=.93, bottom=.04, hspace=.16, wspace=.05)
fig.text(.03, .968, "Eligible land and candidate sites by village buffer", fontsize=10.5, fontweight="bold", va="center")
scen = pd.read_csv(R / "scenario_summary.csv")
for ax, km in zip(axs, (1, 2, 3, 4, 5)):
    e = load(f"eligible_land_{km}km", R)
    ax.imshow(np.where(aoi_small, .12, np.nan), extent=FULL_EXT, cmap="Greys", vmin=0, vmax=1, zorder=1, interpolation="nearest")
    ax.imshow(np.where(e > 0, 1, np.nan), extent=FULL_EXT, cmap=ListedColormap(["#4f9345"]), interpolation="nearest", zorder=2)
    ranges.boundary.plot(ax=ax, color="#2b332e", lw=.25, zorder=3)
    g = gpd.read_file(R / f"sites_{km}km.gpkg")
    g.plot(ax=ax, color="#0b4f8a", markersize=4, edgecolor="white", linewidth=.2, zorder=4)
    q = scen[scen.buffer_km == km].iloc[0]
    ax.set_title(f"{km} km buffer: {int(q.sites)} sites, {q.eligible_land_km2:.0f} km² eligible", fontsize=7.5, loc="left", pad=2)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
axs[5].axis("off")
axs[5].legend(handles=[Patch(fc="#4f9345", label="Eligible land"), Patch(fc="#e0e0e0", label="Excluded land"),
                       Line2D([], [], marker="o", ls="", mfc="#0b4f8a", mec="white", ms=5, label="Candidate site (2 km spacing)")],
              loc="center", frameon=False, fontsize=8)
finish(fig, "40_scenarios_village_buffers")

# Final sites
STR = {"Percolation tank / earthen pond": ("o", "#0b4f8a"), "Check dam / stop dam": ("s", "#c2410c"), "Water hole / saucer pond": ("^", "#6d28d9")}


def site_layers(ax):
    rest = sites[sites.priority != "Priority 1"]
    rest.plot(ax=ax, color="white", edgecolor="#55615a", markersize=7, linewidth=.4, zorder=6)
    for k, (m, c) in STR.items():
        g = sites[(sites.priority == "Priority 1") & (sites.structure == k)]
        if len(g):
            g.plot(ax=ax, marker=m, color=c, edgecolor="white", markersize=26, linewidth=.45, zorder=7)
    for _, p in sites.head(10).iterrows():
        ax.annotate(str(int(p["rank"])), (p.geometry.x, p.geometry.y), xytext=(3, 3), textcoords="offset points", fontsize=5.8,
                    fontweight="bold", zorder=9, bbox=dict(boxstyle="round,pad=.12", fc="white", ec="none", alpha=.85))


fig, ax = base("Recommended water-conservation sites, 2 km village buffer", "This study. Desk-based candidates requiring field verification")
ax.imshow(load("suitability_score", R), extent=FULL_EXT, cmap="Greens", alpha=.55, interpolation="nearest", vmin=45, vmax=85, zorder=2)
frame(ax)
site_layers(ax)
h = [Line2D([], [], marker=m, ls="", mfc=c, mec="white", ms=6, label=f"Priority 1: {k}") for k, (m, c) in STR.items() if ((sites.priority == "Priority 1") & (sites.structure == k)).any()]
h.append(Line2D([], [], marker="o", ls="", mfc="white", mec="#55615a", ms=4, label="Other candidate sites"))
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(.51, .055), ncol=3, fontsize=6.6, frameon=False)
finish(fig, "41_final_sites_2km")

# Water access before and after
fig, axs = plt.subplots(2, 1, figsize=(7.0, 7.3))
fig.subplots_adjust(left=.07, right=.97, top=.93, bottom=.1, hspace=.13)
fig.text(.07, .968, "Distance to dry-season water", fontsize=10.5, fontweight="bold", va="center")
for ax, lay, folder, ti in [(axs[0], "dist_perennial_water_m", L, "Existing perennial water"), (axs[1], "dist_water_after_top50_m", R, "With the 50 Priority 1 sites, if each holds water through the dry season")]:
    im = ax.imshow(load(lay, folder) / 1000, extent=FULL_EXT, cmap="Blues_r", vmin=0, vmax=8, interpolation="nearest", zorder=2)
    frame(ax)
    ax.set_title(ti, fontsize=8, loc="left", pad=3)
sites.head(50).plot(ax=axs[1], color="#c2410c", edgecolor="white", linewidth=.25, markersize=9, zorder=7)
cax = fig.add_axes([.27, .045, .46, .014])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
cb.set_label("Distance to water (km)", fontsize=7, labelpad=2)
cb.ax.tick_params(labelsize=6.5, length=2)
finish(fig, "42_water_access_before_after")


# ------------------------------------------------------------------ supporting charts
def clean(ax):
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


w = pd.read_csv(R / "ahp_weights.csv").sort_values("final_weight")
GC = {"Water availability": "#0b6fa4", "Elephant need and use": "#2f7d4a", "Terrain and erosion": "#c2570c", "Human disturbance": "#6d4bb3"}
fig, ax = plt.subplots(figsize=(7, 5.4))
ax.barh(w.criterion, w.final_weight * 100, color=w.group.map(GC), height=.72, edgecolor="white", linewidth=1)
for y, v in enumerate(w.final_weight * 100):
    ax.text(v + .15, y, f"{v:.1f}", va="center", fontsize=6.5, color=INK)
clean(ax)
ax.set_xlabel("Final AHP weight (%)")
ax.set_title("Criterion weights", loc="left", fontsize=10.5, fontweight="bold")
ax.legend(handles=[Patch(fc=c, label=k) for k, c in GC.items()], frameon=False, loc="lower right", fontsize=7)
fig.tight_layout()
finish(fig, "43_ahp_weights")

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3))
ax[0].bar(scen.buffer_km.astype(str) + " km", scen.sites, color="#0b6fa4", width=.68)
ax[1].bar(scen.buffer_km.astype(str) + " km", scen.eligible_land_km2, color="#2f7d4a", width=.68)
for a, col, ti, fmt in [(ax[0], scen.sites, "Candidate sites", "{:.0f}"), (ax[1], scen.eligible_land_km2, "Eligible land (km²)", "{:.0f}")]:
    for i, v in enumerate(col):
        a.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=7)
    a.set_title(ti, loc="left", fontsize=9, fontweight="bold")
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=.4); a.set_axisbelow(True); a.tick_params(length=0)
fig.tight_layout()
finish(fig, "44_scenario_comparison")

d = pd.read_csv(G / "chirps_daily_aoi_mean.csv", parse_dates=["date"])
m = d.groupby([d.date.dt.year, d.date.dt.month]).rain_mm.sum().groupby(level=1).mean()
y = d.groupby(d.date.dt.year).rain_mm.sum()
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
ax[0].bar(["J", "F", "M", "A", "M ", "J ", "J  ", "A ", "S", "O", "N", "D"], m.values, color="#0b6fa4", width=.7)
ax[0].set_title("Mean monthly rainfall (mm)", loc="left", fontsize=9, fontweight="bold")
ax[1].plot(y.index, y.values, color="#0b6fa4", lw=1.6)
ax[1].axhline(y.mean(), color=MUTED, lw=.7, ls="--")
ax[1].text(y.index.max(), y.mean(), f"mean {y.mean():.0f}", ha="right", va="bottom", fontsize=6.5, color=MUTED)
ax[1].set_title("Annual rainfall (mm)", loc="left", fontsize=9, fontweight="bold")
for a in ax:
    a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color=GRID, lw=.4); a.set_axisbelow(True); a.tick_params(length=0)
fig.tight_layout()
finish(fig, "45_rainfall_climate")
json.dump({"rain_mean": float(y.mean()), "rain_min": float(y.min()), "rain_max": float(y.max()),
           "jjas_share_pct": float(m.loc[6:9].sum() / m.sum() * 100), "feb_may_mm": float(m.loc[2:5].sum())},
          open(R / "rain_summary.json", "w"), indent=1)
print("maps", len(list((OUT / "png").glob("*.png"))))
