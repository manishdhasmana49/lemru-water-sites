"""Self-contained interactive map. Usage: 05_dynamic_map.py <leaflet.js> <leaflet.css>"""
import base64
import io
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
L, V, R = [ROOT / "outputs" / x for x in ("02_layers", "02_vectors", "03_results")]
OUT = ROOT / "outputs" / "05_dynamic_map"
OUT.mkdir(parents=True, exist_ok=True)
leaflet_js, leaflet_css = Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text()

with rasterio.open(L / "aoi.tif") as s:
    SRC_T, SRC_CRS, H, W, B = s.transform, s.crs, s.height, s.width, s.bounds
    aoi = s.read(1) > 0
DT, DW, DH = calculate_default_transform(SRC_CRS, "EPSG:3857", W, H, *B, resolution=90)
west, north = DT.c, DT.f
east, south = DT.c + DT.a * DW, DT.f + DT.e * DH
lon0, lat0, lon1, lat1 = transform_bounds("EPSG:3857", "EPSG:4326", west, south, east, north)
BOUNDS = [[lat0, lon0], [lat1, lon1]]


def warp(path, method=Resampling.nearest):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
    a = np.where(aoi, a, np.nan)
    o = np.full((DH, DW), np.nan, "float32")
    reproject(a, o, src_transform=SRC_T, src_crs=SRC_CRS, src_nodata=np.nan, dst_transform=DT, dst_crs="EPSG:3857",
              dst_nodata=np.nan, resampling=method)
    return o


def png(rgba):
    b = io.BytesIO()
    Image.fromarray((rgba * 255).astype("uint8"), "RGBA").quantize(128, method=Image.Quantize.FASTOCTREE).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def hexes(cm, n=6):
    return [matplotlib.colors.to_hex(cm(x)) for x in np.linspace(0, 1, n)]


def cont(path, label, unit, cmap, q=(2, 98), vmin=None, vmax=None, log=False, scale=1):
    a = warp(path) * scale
    if path.name == "runoff_mm.tif":  # open water has 100% runoff and would hide variation on land
        a = np.where(warp(L / "water_perennial.tif") > 0, np.nan, a)
    f = a[np.isfinite(a)]
    lo, hi = np.percentile(f[f > 0] if log else f, q)
    lo, hi = lo if vmin is None else vmin, hi if vmax is None else vmax
    cm = plt.get_cmap(cmap)
    norm = LogNorm(max(lo, 1e-3), hi) if log else Normalize(lo, hi)
    rgba = cm(norm(np.nan_to_num(a, nan=lo)))
    rgba[..., 3] = np.isfinite(a)
    return {"label": label, "img": png(rgba), "kind": "ramp", "unit": unit, "min": f"{lo:.3g}", "max": f"{hi:.3g}", "colors": hexes(cm)}


def cat(path, label, classes, f=None):
    a = warp(path)
    if f:
        a = f(a)
    rgba = np.zeros((*a.shape, 4))
    for v, _, c in classes:
        rgba[a == v] = matplotlib.colors.to_rgba(c)
    return {"label": label, "img": png(rgba), "kind": "classes", "classes": [[n, c] for _, n, c in classes]}


PREF = [(1, "Very low", "#edf3df"), (2, "Low", "#c5dd9e"), (3, "Moderate", "#8fbf63"), (4, "High", "#4f9345"), (5, "Very high", "#1b5e33")]
SOIL = [(1, "<5 slight", "#fff1dc"), (2, "5–10 moderate", "#fdd08e"), (3, "10–20 high", "#f8a04b"), (4, "20–40 very high", "#d9601a"), (5, ">40 severe", "#8a2f07")]
tc = lambda a: np.where(np.isfinite(a), np.digitize(a, [5, 10, 20, 40]) + 1, 0)
LU = [(1, "Forest", "#2f7d4a"), (3, "Barren", "#c9b48a"), (4, "Water/streams", "#2f7fc1"), (8, "Agriculture", "#e5c33f"), (10, "Fallow", "#f0e2a2"),
      (2, "Roads", "#7c7c7c"), (6, "Built-up urban", "#b4262c"), (7, "Built-up rural", "#e4572e"), (5, "Industry", "#7a3e9d"), (9, "Mine", "#2b2b2b")]
layers = {
    "suit": cont(R / "suitability_score.tif", "Composite suitability", "Score 0–100", "YlGnBu", (1, 99.5)),
    "soil": cat(L / "rusle_A_tha.tif", "RUSLE soil loss", SOIL, tc),
    "catch": cat(L / "catchment_soil_loss_tha.tif", "Catchment soil loss", SOIL, tc),
    "slope": cont(L / "slope_deg.tif", "Slope", "Degrees", "Oranges", vmin=0, vmax=35),
    "rain": cont(L / "rain_annual_mm.tif", "Annual rainfall", "mm/yr", "Blues", (0, 100)),
    "runoff": cont(L / "runoff_mm.tif", "Surface runoff", "mm/yr", "Blues", (1, 99)),
    "water": cont(L / "dist_perennial_water_m.tif", "Distance to perennial water", "km", "Blues_r", vmin=0, vmax=8, scale=.001),
    "village": cont(L / "dist_village_only_m.tif", "Distance to villages", "km", "Purples", vmin=0, vmax=5, scale=.001),
    "habitat": cat(L / "habitat_suitability.tif", "Elephant habitat suitability", PREF),
    "elephant": cont(L / "elephant_density.tif", "Elephant record density", "Records/km²", "Greens", (2, 99)),
    "ndvi": cont(L / "ndvi_median.tif", "NDVI", "Index", "Greens", vmin=0, vmax=.85),
    "lulc": cat(L / "lulc.tif", "Land use and land cover", LU),
    "zone": cat(L / "zonation.tif", "Management zonation", [(1, "Co-existence", "#e8c37a"), (2, "Management", "#8fbf63"), (3, "Elephant Conservation", "#1b5e33")]),
}
eligible = {}
for km in range(1, 6):
    a = warp(R / f"eligible_land_{km}km.tif")
    rgba = np.zeros((*a.shape, 4))
    rgba[a > 0] = matplotlib.colors.to_rgba("#39b54a")
    eligible[km] = png(rgba)

KEEP = ["rank", "score", "lat", "lon", "range", "beat", "compartment", "zone", "structure", "elev_m", "slope_deg", "catchment_ha", "yield_m3_yr",
        "catch_soil_loss_tha", "village_km", "nearest_village", "water_km", "road_km"]
sites = {}
for km in range(1, 6):
    d = pd.DataFrame(gpd.read_file(R / (f"sites_{km}km.gpkg" if km != 2 else "sites_2km_final.gpkg")).drop(columns="geometry"))
    k = KEEP + (["priority", "top50_stability_pct"] if km == 2 else [])
    sites[km] = json.loads(d[k].round(3).to_json(orient="records"))
scen = json.loads(pd.read_csv(R / "scenario_summary.csv").round(3).to_json(orient="records"))
rg = gpd.read_file(V / "ranges.gpkg").to_crs(4326)
rg["geometry"] = rg.geometry.simplify(.0006)
ranges = json.loads(rg[["RANGENAME", "geometry"]].to_json(drop_id=True))
for f in ranges["features"]:
    def rnd(c):
        return [round(c[0], 5), round(c[1], 5)] if isinstance(c[0], (int, float)) else [rnd(x) for x in c]
    f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
DATA = json.dumps({"bounds": BOUNDS, "layers": layers, "eligible": eligible, "sites": sites, "scenarios": scen, "ranges": ranges}, separators=(",", ":")).replace("</", "<\\/")

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lemru Water Sites</title><style>__CSS__</style><style>
:root{--ink:#17241d;--muted:#59675f;--line:#d7dfda;--panel:#fff;--soft:#eef3f0;--brand:#14533a;--accent:#c2410c;--bg:#f6f8f7}
*{box-sizing:border-box}html,body{height:100%;margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 Arial,Helvetica,sans-serif}
#app{display:grid;grid-template-columns:350px 1fr;height:100%}aside{background:var(--panel);border-right:1px solid var(--line);overflow:auto;padding:16px}
h1{font-size:21px;line-height:1.15;margin:0 0 4px}.sub{color:var(--muted);font-size:12.5px;margin-bottom:14px}h2{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:16px 0 7px}
.seg{display:grid;grid-template-columns:repeat(5,1fr);gap:4px}button{font:inherit;cursor:pointer}.seg button{padding:8px 0;border:1px solid var(--line);background:#fff;border-radius:6px;font-weight:700}.seg button.on{background:var(--brand);border-color:var(--brand);color:#fff}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}.stat{background:var(--soft);border-radius:6px;padding:8px}.stat b{display:block;font-size:19px}.stat span{font-size:11px;color:var(--muted)}
select,input[type=range]{width:100%}select{font:inherit;padding:7px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink)}label.row{display:flex;gap:7px;align-items:center;margin:5px 0}
.ramp{height:10px;border-radius:3px}.rl{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-top:2px}.cl{display:grid;grid-template-columns:1fr 1fr;gap:2px 8px;font-size:11.5px}.cl i{display:inline-block;width:11px;height:11px;margin-right:5px;border:1px solid #8a948e;vertical-align:-1px}
#list{max-height:260px;overflow:auto;border:1px solid var(--line);border-radius:6px}#list button{display:grid;grid-template-columns:32px 1fr 42px;width:100%;padding:6px 8px;border:0;border-bottom:1px solid var(--line);background:#fff;text-align:left;color:var(--ink)}#list button:hover,#list button:focus{background:var(--soft)}
#map{height:100%;background:#e8ece9}.note{font-size:11.5px;color:var(--muted);margin-top:12px}.pop{min-width:235px}.pop h3{margin:0 0 5px;font-size:15px}.pop table{border-collapse:collapse;width:100%;font-size:12px}.pop td{padding:2px 0;border-bottom:1px solid #edf0ee}.pop td:last-child{text-align:right;font-weight:600}
@media(max-width:760px){#app{grid-template-columns:1fr;grid-template-rows:55% 45%}#map{grid-row:1}aside{grid-row:2;border-right:0;border-top:1px solid var(--line)}}
</style></head><body><div id="app"><aside><h1>Lemru water sites</h1><div class="sub">Candidate elephant water-conservation sites by village buffer</div>
<h2>Village buffer</h2><div class="seg" id="seg"></div><div class="stats" id="stats"></div>
<h2>Map layer</h2><select id="layer" aria-label="Map layer"></select><div style="margin-top:8px" id="legend"></div>
<label class="row" style="margin-top:9px"><span style="white-space:nowrap">Opacity</span><input id="op" type="range" min="0" max="100" value="75" aria-label="Layer opacity"></label>
<label class="row"><input id="elig" type="checkbox">Show eligible land</label><label class="row"><input id="sitesOn" type="checkbox" checked>Show candidate sites</label>
<h2>Basemap</h2><select id="base" aria-label="Basemap"><option value="img">Satellite imagery</option><option value="osm">OpenStreetMap</option><option value="none">None (works offline)</option></select>
<h2>Top sites</h2><div id="list"></div><div class="note">Desk-based candidates. Verify soil, dam feasibility, ownership and elephant movement in the field before construction. Treat site coordinates as sensitive.</div></aside><div id="map" role="application" aria-label="Interactive reserve map"></div></div>
<script>__JS__</script><script>const D=__DATA__;
const $=id=>document.getElementById(id),esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const map=L.map("map",{preferCanvas:true,zoomSnap:.25}).fitBounds(D.bounds);L.control.scale({imperial:false}).addTo(map);
const bases={img:L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",{maxZoom:18,attribution:"Imagery: Esri"}),osm:L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:18,attribution:"© OpenStreetMap contributors"})};
let base=bases.img.addTo(map),km=2,over,elig,markers=L.layerGroup().addTo(map),byRank={};
L.geoJSON(D.ranges,{style:{color:"#fff",weight:1.2,fill:false},interactive:false}).addTo(map);
for(const[k,v]of Object.entries(D.layers)){const o=document.createElement("option");o.value=k;o.textContent=v.label;$("layer").append(o)}
const seg=$("seg");[1,2,3,4,5].forEach(n=>{const b=document.createElement("button");b.textContent=n+" km";b.onclick=()=>{km=n;draw()};b.dataset.k=n;seg.append(b)});
function layer(){const v=D.layers[$("layer").value];if(over)map.removeLayer(over);over=L.imageOverlay(v.img,D.bounds,{opacity:$("op").value/100}).addTo(map);
$("legend").innerHTML=v.kind==="ramp"?`<div class="ramp" style="background:linear-gradient(90deg,${v.colors.join(",")})"></div><div class="rl"><span>${esc(v.min)}</span><span>${esc(v.unit)}</span><span>${esc(v.max)}</span></div>`:`<div class="cl">${v.classes.map(c=>`<span><i style="background:${c[1]}"></i>${esc(c[0])}</span>`).join("")}</div>`;if(elig)elig.bringToFront()}
function color(s){return /Check/.test(s.structure)?"#c2410c":/Water hole/.test(s.structure)?"#6d28d9":"#ffc21a"}
function pop(s){const r=[["Score",s.score],["Range",s.range],["Beat",s.beat],["Compartment",s.compartment],["Zone",s.zone],["Slope",s.slope_deg+"°"],["Catchment",s.catchment_ha+" ha"],["Estimated yield",Math.round(s.yield_m3_yr).toLocaleString()+" m³/yr"],["Catchment soil loss",s.catch_soil_loss_tha+" t/ha/yr"],["Nearest village",`${s.nearest_village} (${s.village_km} km)`],["Perennial water",s.water_km+" km"],["Road",s.road_km+" km"]];if(s.top50_stability_pct!=null)r.splice(1,0,["Top-50 stability",s.top50_stability_pct+"%"]);return `<div class="pop"><h3>Site ${s.rank}: ${esc(s.structure)}</h3><table>${r.map(x=>`<tr><td>${esc(x[0])}</td><td>${esc(x[1])}</td></tr>`).join("")}</table><div style="margin-top:5px;font-size:11px">${s.lat}, ${s.lon}</div></div>`}
function draw(){[...seg.children].forEach(b=>b.classList.toggle("on",+b.dataset.k===km));const q=D.scenarios.find(x=>x.buffer_km===km);$("stats").innerHTML=`<div class="stat"><b>${q.sites}</b><span>sites</span></div><div class="stat"><b>${Math.round(q.eligible_land_km2)}</b><span>km² eligible</span></div><div class="stat"><b>${q.ranges_with_sites}/12</b><span>ranges</span></div>`;
if(elig){map.removeLayer(elig);elig=null}if($("elig").checked)elig=L.imageOverlay(D.eligible[km],D.bounds,{opacity:.85}).addTo(map);markers.clearLayers();byRank={};const S=D.sites[km];if($("sitesOn").checked)S.forEach(s=>{const top=s.rank<=50;const m=L.circleMarker([s.lat,s.lon],{radius:top?7:4,color:"#fff",weight:top?1.5:1,fillColor:top?color(s):"#f7fbf8",fillOpacity:1}).bindPopup(pop(s));byRank[s.rank]=m;markers.addLayer(m)});
$("list").innerHTML="";S.slice(0,50).forEach(s=>{const b=document.createElement("button");b.innerHTML=`<b>${s.rank}</b><span>${esc(s.range)} · ${esc(s.compartment)}</span><span>${Number(s.score).toFixed(1)}</span>`;b.onclick=()=>{map.flyTo([s.lat,s.lon],14);if(!$("sitesOn").checked){$("sitesOn").checked=true;draw()}setTimeout(()=>byRank[s.rank]?.openPopup(),700)};$("list").append(b)})}
$("layer").onchange=layer;$("op").oninput=e=>over&&over.setOpacity(e.target.value/100);$("elig").onchange=draw;$("sitesOn").onchange=draw;$("base").onchange=e=>{map.removeLayer(base);base=bases[e.target.value];if(base)base.addTo(map).bringToBack()};layer();draw();
</script></body></html>"""
html = HTML.replace("__CSS__", leaflet_css).replace("__JS__", leaflet_js.replace("</script", "<\\/script")).replace("__DATA__", DATA)
p = OUT / "Lemru_Water_Sites_Dynamic_Map.html"
p.write_text(html, encoding="utf-8")
print(p, round(p.stat().st_size / 1e6, 1), "MB")
