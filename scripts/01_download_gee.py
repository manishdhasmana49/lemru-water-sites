"""Download all Earth Engine inputs for the Lemru water-site model to a common 30 m grid.

Grid: EPSG:32644, reserve boundary + 10 km margin, snapped to 30 m.
Re-running skips layers that already exist.
"""
import concurrent.futures as cf
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

warnings.filterwarnings("ignore")

EE_PROJECT = os.environ["EE_PROJECT"]  # set this to your Google Cloud project ID
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "LEMRU ELEPHANR RESERVE-WATER MANAGEMENT"
OUT = ROOT / "outputs" / "01_gee"
OUT.mkdir(parents=True, exist_ok=True)
CRS, RES, MARGIN, TILE = "EPSG:32644", 30, 10_000, 768
Y0, Y1 = 1996, 2025  # 30-year rainfall normal
S2_START, S2_END = "2023-01-01", "2026-01-01"

ee.Initialize(project=EE_PROJECT)


def grid():
    b = gpd.read_file(SRC / "Lemru_Elephant_Conservation_Zone-Range_Boundary.shp").to_crs(CRS)
    x0, y0, x1, y1 = b.total_bounds
    x0 = np.floor((x0 - MARGIN) / RES) * RES
    y0 = np.floor((y0 - MARGIN) / RES) * RES
    x1 = np.ceil((x1 + MARGIN) / RES) * RES
    y1 = np.ceil((y1 + MARGIN) / RES) * RES
    w, h = int((x1 - x0) / RES), int((y1 - y0) / RES)
    return Affine(RES, 0, x0, 0, -RES, y1), w, h, (x0, y0, x1, y1)


TRANSFORM, W, H, BOUNDS = grid()
REGION = ee.Geometry.Rectangle(list(BOUNDS), CRS, False)
REGION_LL = REGION.transform("EPSG:4326", 100)


def fetch_tile(img, col, row, w, h, tries=6):
    t = TRANSFORM * Affine.translation(col, row)
    req = {
        "expression": img,
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {"scaleX": t.a, "shearX": t.b, "translateX": t.c,
                                "shearY": t.d, "scaleY": t.e, "translateY": t.f},
            "crsCode": CRS,
        },
    }
    for i in range(tries):
        try:
            raw = ee.data.computePixels(req)
            with rasterio.MemoryFile(io.BytesIO(raw)) as m, m.open() as s:
                return col, row, s.read(1, masked=True).filled(np.nan).astype("float32")
        except Exception as e:
            if i == tries - 1:
                raise RuntimeError(f"tile {col},{row}: {str(e)[:250]}")
            time.sleep(2 ** i + 1)


def download(name, img):
    path = OUT / f"{name}.tif"
    if path.exists():
        print("skip", name, flush=True)
        return
    t0 = time.time()
    img = ee.Image(img).toFloat().unmask(-9999, False)
    arr = np.full((H, W), np.nan, dtype="float32")
    jobs = [(c, r, min(TILE, W - c), min(TILE, H - r))
            for r in range(0, H, TILE) for c in range(0, W, TILE)]
    with cf.ThreadPoolExecutor(8) as ex:
        for c, r, a in ex.map(lambda j: fetch_tile(img, *j), jobs):
            arr[r:r + a.shape[0], c:c + a.shape[1]] = a
    arr[arr <= -9998] = np.nan
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=1, dtype="float32",
                       crs=CRS, transform=TRANSFORM, nodata=np.nan, compress="deflate",
                       predictor=3, tiled=True) as d:
        d.write(arr, 1)
    v = arr[np.isfinite(arr)]
    print(f"{name}: {time.time()-t0:.0f}s valid={v.size/arr.size:.1%} "
          f"min={v.min() if v.size else None:.3f} max={v.max() if v.size else None:.3f}", flush=True)


def chirps_stack():
    c = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    years = ee.List.sequence(Y0, Y1)

    def season(months=None):
        def f(y):
            q = c.filter(ee.Filter.calendarRange(y, y, "year"))
            if months:
                q = q.filter(ee.Filter.calendarRange(months[0], months[1], "month"))
            return q.sum().set("year", y)
        return ee.ImageCollection.fromImages(years.map(f))

    annual = season()
    return {
        "rain_annual_mm": annual.mean().resample("bilinear"),
        "rain_cv_pct": annual.reduce(ee.Reducer.stdDev()).divide(annual.mean()).multiply(100).resample("bilinear"),
        "rain_monsoon_mm": season((6, 9)).mean().resample("bilinear"),
        "rain_dry_febmay_mm": season((2, 5)).mean().resample("bilinear"),
    }


def s2_layers():
    def mask(i):
        s = i.select("SCL")
        ok = s.neq(1).And(s.neq(3)).And(s.neq(8)).And(s.neq(9)).And(s.neq(10)).And(s.neq(11))
        return i.updateMask(ok).divide(10000).copyProperties(i, ["system:time_start"])

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(REGION_LL)
           .filterDate(S2_START, S2_END).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
           .select(["B3", "B4", "B8", "B11", "SCL"]).map(mask))
    add = lambda i: i.addBands(i.normalizedDifference(["B8", "B4"]).rename("ndvi")) \
                     .addBands(i.normalizedDifference(["B3", "B11"]).rename("mndwi"))
    col = col.map(add)
    dry = col.filter(ee.Filter.calendarRange(3, 5, "month"))
    return {
        "ndvi_median": col.select("ndvi").median(),
        "ndvi_dry_marmay": dry.select("ndvi").median(),
        "mndwi_dry_marmay": dry.select("mndwi").median(),
    }


def dem():
    for cid in ("COPERNICUS/DEM/GLO30_2024_1", "COPERNICUS/DEM/GLO30"):
        try:
            c = ee.ImageCollection(cid).filterBounds(REGION_LL).select("DEM")
            if c.size().getInfo() > 0:
                print("DEM source:", cid, flush=True)
                proj = c.first().projection()
                return c.mosaic().setDefaultProjection(proj).resample("bilinear"), cid
        except Exception:
            continue
    raise RuntimeError("No DEM")


def daily_series():
    p = OUT / "chirps_daily_aoi_mean.csv"
    if p.exists():
        return
    b = gpd.read_file(SRC / "Lemru_Elephant_Conservation_Zone-Range_Boundary.shp").to_crs(4326)
    x0, y0, x1, y1 = b.total_bounds
    geom = ee.Geometry.Rectangle([float(x0), float(y0), float(x1), float(y1)])
    frames = []
    for y in range(Y0, Y1 + 1):
        c = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(f"{y}-01-01", f"{y+1}-01-01").select("precipitation")
        fc = c.map(lambda i: ee.Feature(None, {
            "date": i.date().format("YYYY-MM-dd"),
            "p": i.reduceRegion(ee.Reducer.mean(), geom, 5566).get("precipitation")}))
        rows = fc.reduceColumns(ee.Reducer.toList(2), ["date", "p"]).get("list").getInfo()
        frames.append(pd.DataFrame(np.array(rows).reshape(-1, 2), columns=["date", "rain_mm"]))
    d = pd.concat(frames)
    d["rain_mm"] = d["rain_mm"].astype(float)
    d.to_csv(p, index=False)
    print("daily rainfall rows", len(d), flush=True)


if __name__ == "__main__":
    print(f"grid {W}x{H} cells, bounds {BOUNDS}", flush=True)
    d, dem_id = dem()
    layers = {"dem_m": d}
    layers.update(chirps_stack())
    ol = "OpenLandMap/SOL/SOL_{}/v02"
    layers.update({
        "soil_sand_pct": ee.Image(ol.format("SAND-WFRACTION_USDA-3A1A1A_M")).select("b0"),
        "soil_clay_pct": ee.Image(ol.format("CLAY-WFRACTION_USDA-3A1A1A_M")).select("b0"),
        "soil_oc_gkg": ee.Image(ol.format("ORGANIC-CARBON_USDA-6A1C_M")).select("b0").multiply(5),
    })
    layers.update(s2_layers())
    g = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    layers.update({"jrc_occurrence": g.select("occurrence"), "jrc_seasonality": g.select("seasonality")})
    layers["worldcover"] = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    layers["merit_upa_km2"] = ee.Image("MERIT/Hydro/v1_0_1").select("upa")

    failed = []
    for n, i in layers.items():
        try:
            download(n, i)
        except Exception as e:
            failed.append(n)
            print("FAILED", n, str(e)[:300], flush=True)
    try:
        daily_series()
    except Exception as e:
        failed.append("daily")
        print("FAILED daily", str(e)[:300], flush=True)
    json.dump({"project_grid": CRS, "res_m": RES, "dem": dem_id, "rain_period": [Y0, Y1],
               "s2_period": [S2_START, S2_END], "failed": failed},
              open(OUT / "download_manifest.json", "w"), indent=1)
    print("DONE failed=", failed, flush=True)
    sys.exit(1 if failed else 0)
