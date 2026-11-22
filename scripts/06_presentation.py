"""Build the PowerPoint deck from model outputs."""
import json
from pathlib import Path

import pandas as pd
import rasterio
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "outputs" / "03_results"
M = ROOT / "outputs" / "04_maps" / "png"
OUT = ROOT / "outputs" / "06_presentation"
OUT.mkdir(parents=True, exist_ok=True)
rep = json.load(open(R / "model_report.json"))
rain = json.load(open(R / "rain_summary.json"))
qa = json.load(open(ROOT / "outputs" / "02_layers" / "qa_preprocess.json"))
scen = pd.read_csv(R / "scenario_summary.csv")
fun = pd.read_csv(R / "constraint_funnel_2km.csv")
wts = pd.read_csv(R / "ahp_weights.csv").sort_values("final_weight")
top = pd.read_csv(R / "top50_sites_2km.csv")
daily = pd.read_csv(ROOT / "outputs" / "01_gee" / "chirps_daily_aoi_mean.csv", parse_dates=["date"])
month = daily.groupby([daily.date.dt.year, daily.date.dt.month]).rain_mm.sum().groupby(level=1).mean()
acc = rep["water_access"]
with rasterio.open(ROOT / "outputs" / "02_layers" / "runoff_mm.tif") as _q, rasterio.open(ROOT / "outputs" / "02_layers" / "aoi.tif") as _a, \
        rasterio.open(ROOT / "outputs" / "02_layers" / "water_perennial.tif") as _w:
    LAND_RUNOFF = float(_q.read(1)[(_a.read(1) > 0) & (_w.read(1) == 0)].mean())

FOREST, DEEP, WATER, EMBER, INK, MUTED, TINT, WHITE = [RGBColor.from_string(x) for x in
    ("14533A", "0D2F24", "0B6FA4", "C2410C", "17241D", "59675F", "EAF2EE", "FFFFFF")]
HEAD, BODY = "Cambria", "Calibri"
p = Presentation()
p.slide_width, p.slide_height = Inches(13.333), Inches(7.5)
TITLE_ONLY = p.slide_layouts[5]


def text(s, x, y, w, h, t, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None):
    b = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        b.name = name
    f = b.text_frame
    f.word_wrap = True
    f.vertical_anchor = anchor
    f.margin_left = f.margin_right = f.margin_top = f.margin_bottom = 0
    items = t if isinstance(t, list) else [t]
    for i, x_ in enumerate(items):
        para = f.paragraphs[0] if i == 0 else f.add_paragraph()
        para.alignment = align
        para.space_after = Pt(7 if len(items) > 1 else 0)
        r = para.add_run()
        r.text = x_
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
        r.font.color.rgb = color
    return b


def slide(title, notes, dark=False):
    s = p.slides.add_slide(TITLE_ONLY)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = DEEP if dark else WHITE
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(.55), Inches(.33), Inches(12.2), Inches(.8)
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = 0
    tf.text = title
    q = tf.paragraphs[0]
    q.alignment = PP_ALIGN.LEFT
    q.runs[0].font.size, q.runs[0].font.bold, q.runs[0].font.name = Pt(30), True, HEAD
    q.runs[0].font.color.rgb = WHITE if dark else DEEP
    s.notes_slide.notes_text_frame.text = notes
    n = len(p.slides)
    if n > 1:
        text(s, 12.2, 7.08, .6, .22, str(n), 10, MUTED if not dark else WHITE, align=PP_ALIGN.RIGHT, name="Slide number")
    return s


def image(s, name, x, y, w, h, alt):
    path = M / f"{name}.png"
    iw, ih = Image.open(path).size
    k = min(w / iw, h / ih)
    pw, ph = iw * k, ih * k
    pic = s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def card(s, x, y, w, h, value, label, color=FOREST):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    c.adjustments[0] = .07
    c.fill.solid()
    c.fill.fore_color.rgb = TINT
    c.line.fill.background()
    c.shadow.inherit = False
    text(s, x + .18, y + .12, w - .36, h * .54, value, 30, color, True, HEAD, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + .18, y + h * .62, w - .36, h * .34, label, 13, MUTED)


def bar(s, x, y, w, h, cats, vals, color, title, horizontal=False, fmt="0"):
    d = CategoryChartData()
    d.categories = list(cats)
    d.add_series(title, [float(v) for v in vals])
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED,
                            Inches(x), Inches(y), Inches(w), Inches(h), d).chart
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = title
    r = ch.chart_title.text_frame.paragraphs[0].runs[0]
    r.font.size, r.font.bold, r.font.name = Pt(14), True, BODY
    r.font.color.rgb = INK
    pl = ch.plots[0]
    pl.gap_width = 45
    pl.has_data_labels = True
    pl.data_labels.number_format, pl.data_labels.number_format_is_linked = fmt, False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size, pl.data_labels.font.name = Pt(11), BODY
    pl.data_labels.font.color.rgb = INK
    pl.series[0].format.fill.solid()
    pl.series[0].format.fill.fore_color.rgb = color
    ch.value_axis.visible = False
    ch.value_axis.has_major_gridlines = False
    ch.category_axis.tick_labels.font.size, ch.category_axis.tick_labels.font.name = Pt(11), BODY
    ch.category_axis.tick_labels.font.color.rgb = MUTED
    ch.category_axis.format.line.color.rgb = RGBColor.from_string("C9D2CD")
    return ch


b2 = scen[scen.buffer_km == 2].iloc[0]

# 1 Title
s = slide("Water Conservation Site Selection", "Purpose: present candidate water-conservation sites for elephants in the Lemru reserve, and the evidence behind them.", dark=True)
s.shapes.title.top, s.shapes.title.width, s.shapes.title.height = Inches(2.05), Inches(5.3), Inches(1.75)
s.shapes.title.text_frame.paragraphs[0].runs[0].font.size = Pt(40)
text(s, .55, 3.95, 5.1, .8, "Lemru Elephant Reserve, Chhattisgarh", 20, RGBColor.from_string("BFE3D0"))
text(s, .55, 4.65, 5.1, .8, "A GIS, RUSLE and AHP assessment using project data and Google Earth Engine", 15, WHITE)
image(s, "41_final_sites_2km", 6.0, .75, 7.0, 6.0, "Map of recommended water-conservation sites across the reserve")

# 2 Key result
s = slide("50 priority sites could double nearby water access",
          f"The base case keeps every site at least 2 km from village locations. {int(b2.sites)} candidates remain and the top 50 form Priority 1. The access figures assume each site holds water through the dry season, which must be confirmed in the field.")
image(s, "41_final_sites_2km", .45, 1.25, 8.2, 5.65, "Priority 1 and other candidate sites")
card(s, 8.95, 1.35, 3.85, 1.55, f"{int(b2.sites)}", "candidate sites at the 2 km buffer")
card(s, 8.95, 3.08, 3.85, 1.55, "50", "Priority 1 sites in 6 ranges", WATER)
card(s, 8.95, 4.81, 3.85, 1.85, f"{acc['2km']['reserve_before_pct']:.0f}% → {acc['2km']['reserve_after_pct']:.0f}%", "of reserve within 2 km of dry-season water, if sites hold water", EMBER)

# 3 Climate
s = slide("Water is scarce when elephants need it most",
          f"Mean annual rainfall is {rain['rain_mean']:.0f} mm. June to September supplies {rain['jjas_share_pct']:.0f} percent. February to May averages only {rain['feb_may_mm']:.0f} mm. Rainfall varies by only about 36 mm across the reserve, so it receives a low weight.")
bar(s, .55, 1.3, 8.0, 5.45, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], month.values, WATER, "Mean monthly rainfall, 1996–2025 (mm)")
card(s, 8.95, 1.35, 3.85, 1.6, f"{rain['rain_mean']:.0f} mm", "mean annual rainfall")
card(s, 8.95, 3.15, 3.85, 1.6, f"{rain['jjas_share_pct']:.0f}%", "falls in June–September", WATER)
card(s, 8.95, 4.95, 3.85, 1.6, f"{rain['feb_may_mm']:.0f} mm", "falls in February–May", EMBER)

# 4 Method
s = slide("Method: exclude unsuitable land, then score the rest",
          "Step one applies hard limits. Step two scores remaining drainage cells using 20 criteria and AHP weights. Step three selects the highest-scoring sites at least 2 km apart. Step four tests sensitivity.")
steps = [("1", "Build inputs", "Project layers plus rainfall, DEM, soil, NDVI and water from Earth Engine"),
         ("2", "Model processes", "Flow routing, SCS-CN runoff and RUSLE soil loss on a 30 m grid"),
         ("3", "Apply limits", "Village buffer, slope, soil loss, mines, roads, land use and existing water"),
         ("4", "Score and rank", "20 criteria weighted by AHP, with sites spaced at least 2 km apart")]
for i, (n, h, b) in enumerate(steps):
    x = .55 + i * 3.1
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.65), Inches(2.85), Inches(3.75))
    c.adjustments[0] = .05
    c.fill.solid(); c.fill.fore_color.rgb = TINT; c.line.fill.background(); c.shadow.inherit = False
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + .22), Inches(1.88), Inches(.62), Inches(.62))
    o.fill.solid(); o.fill.fore_color.rgb = FOREST; o.line.fill.background(); o.shadow.inherit = False
    o.text_frame.text = n
    o.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    o.text_frame.paragraphs[0].runs[0].font.size, o.text_frame.paragraphs[0].runs[0].font.bold = Pt(18), True
    o.text_frame.paragraphs[0].runs[0].font.color.rgb = WHITE
    text(s, x + .22, 2.7, 2.42, .45, h, 20, DEEP, True, HEAD)
    text(s, x + .22, 3.3, 2.42, 1.9, b, 16, INK)
text(s, .55, 5.85, 12.2, .8, "Settlement rule: distance from the supplied village locations, tested at 1, 2, 3, 4 and 5 km.", 18, FOREST, True)

# 5 Funnel
s = slide("Hard limits narrow 1,991 km² to 30 km² of drainage lines",
          "Each bar shows the land still eligible after that limit is applied in sequence. The final bar is the drainage network available for structures in the 2 km scenario.")
labels = ["Reserve", "Slope ≤ 11.3°", "Away from water", "Suitable land use", "Mines ≥ 1 km", "Road setbacks", "Soil loss ≤ 40 t/ha/yr", "Villages ≥ 2 km", "Drainage lines"]
bar(s, .55, 1.25, 12.2, 5.6, labels[::-1], fun.area_km2.values[::-1], FOREST, "Eligible area after each limit (km²)", True)

# 6 scenarios
s = slide("A 2 km buffer balances separation and coverage",
          "Stricter buffers reduce options quickly. At 2 km all 12 ranges retain candidate sites. At 3 km two ranges lose all candidates, and at 5 km only four ranges retain any.")
image(s, "40_scenarios_village_buffers", .45, 1.2, 6.5, 5.8, "Eligible land and sites for each village buffer")
bar(s, 7.15, 1.25, 5.7, 3.3, [f"{k} km" for k in scen.buffer_km], scen.sites, WATER, "Candidate sites by village buffer")
rows = [["Buffer", "Eligible km²", "Sites", "Ranges"]] + [[f"{int(r.buffer_km)} km", f"{r.eligible_land_km2:.0f}", f"{int(r.sites)}", f"{int(r.ranges_with_sites)}/12"] for r in scen.itertuples()]
tb = s.shapes.add_table(6, 4, Inches(7.3), Inches(4.65), Inches(5.4), Inches(2.25)).table
for i, row in enumerate(rows):
    for j, v in enumerate(row):
        c = tb.cell(i, j); c.text = v
        c.fill.solid(); c.fill.fore_color.rgb = FOREST if i == 0 else (TINT if i == 2 else WHITE)
        q = c.text_frame.paragraphs[0]; q.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
        q.runs[0].font.size, q.runs[0].font.name, q.runs[0].font.bold = Pt(12), BODY, i in (0, 2)
        q.runs[0].font.color.rgb = WHITE if i == 0 else INK
        c.margin_top = c.margin_bottom = Inches(.025)

# 7 RUSLE factors
s = slide("RUSLE combines rainfall, soil, terrain and cover",
          "A equals R times K times LS times C times P. R uses Renard and Freimund, K uses the EPIC equation, LS uses Moore and Burch, C uses the van der Knijff NDVI equation, and P uses contour factors on cropland.")
for i, (n, a) in enumerate([("07_rusle_R", "R factor"), ("08_rusle_K", "K factor"), ("09_rusle_LS", "LS factor"), ("10_rusle_C", "C factor"), ("11_rusle_P", "P factor"), ("12_rusle_soil_loss", "Soil loss")]):
    image(s, n, .45 + (i % 3) * 4.16, 1.2 + (i // 3) * 2.92, 4.08, 2.85, f"Map of RUSLE {a}")

# 8 Soil loss
s = slide("Soil loss is concentrated on steep northern slopes",
          f"Mean modelled loss is {qa['A_mean_tha']:.1f} t/ha/yr with the NDVI C factor and {qa['A_mean_tableC_tha']:.1f} with a land-cover C factor. Treat absolute values as uncertain and use the map to compare areas. {rep['top50_pass_tableC']} of 50 priority sites meet the 40 t/ha/yr limit under both methods.")
image(s, "12_rusle_soil_loss", .45, 1.25, 8.2, 5.65, "RUSLE soil loss classes")
card(s, 8.95, 1.35, 3.85, 1.6, f"{qa['A_mean_tha']:.1f}", "t/ha/yr mean, NDVI-based C", EMBER)
card(s, 8.95, 3.15, 3.85, 1.6, f"{qa['A_mean_tableC_tha']:.1f}", "t/ha/yr mean, land-cover C", WATER)
card(s, 8.95, 4.95, 3.85, 1.6, f"{rep['top50_pass_tableC']}/50", "priority sites pass under both C methods")

# 9 Hydrology
s = slide("Runoff and drainage show where water can be captured",
          f"The SCS curve-number method estimates mean land runoff near {LAND_RUNOFF:.0f} mm per year, excluding open water. It uses reserve-average daily rainfall, so local storm runoff may be underestimated.")
image(s, "22_runoff", .45, 1.3, 6.15, 4.7, "Mean annual surface runoff")
image(s, "25_stream_order", 6.72, 1.3, 6.15, 4.7, "Strahler stream order")
text(s, .55, 6.2, 12.2, .55, f"Mean modelled land runoff is about {LAND_RUNOFF:.0f} mm/yr. Candidate structures sit on drainage lines with 5–1,000 ha catchments.", 16)

# 10 Elephant
s = slide("Sites target elephant habitat with poor water access",
          "The elephant-need criteria combine distance to perennial water, cleaned presence records, habitat suitability and the supplied preference layers. Exact sighting points are intentionally not shown.")
for i, (n, a) in enumerate([("36_habitat_suitability", "Elephant habitat suitability"), ("38_elephant_density", "Elephant record density"), ("26_distance_perennial_water", "Distance to perennial water")]):
    image(s, n, .45 + i * 4.16, 1.5, 4.08, 3.3, a)
text(s, .55, 5.15, 12.2, 1.2, ["Only 17% of elephant records lie within 2 km of perennial water.", "Exact sighting points are withheld from maps to reduce poaching risk."], 18)

# 11 Weights
s = slide("Water supply and elephant need carry the most weight",
          f"AHP group weights are one third each for water availability and elephant need, and one sixth each for terrain and human disturbance. All consistency ratios are below the accepted 0.10 limit. The judgements should be reviewed by reserve experts.")
t = wts.tail(12)
bar(s, .55, 1.25, 8.25, 5.65, t.criterion, t.final_weight * 100, FOREST, "Top 12 of 20 criterion weights (%)", True, "0.0")
card(s, 9.1, 1.35, 3.7, 1.6, "33%", "water availability", WATER)
card(s, 9.1, 3.15, 3.7, 1.6, "33%", "elephant need and use")
card(s, 9.1, 4.95, 3.7, 1.6, "17% each", "terrain and disturbance", EMBER)

# 12 Suitability
s = slide("Suitability is highest across the southern ranges",
          "This surface shows the weighted score before site spacing. Darker areas combine stronger water potential, elephant need and lower disturbance.")
image(s, "39_suitability_score", 1.65, 1.15, 10.0, 5.85, "Composite water-site suitability map")

# 13 top table
s = slide("Ten highest-ranked sites for first field checks",
          "These are the first sites to inspect. Coordinates for all 50 priority sites are in the Excel file. Treat them as sensitive.")
cols = ["Rank", "Range", "Compartment", "Structure", "Catchment (ha)", "Slope (°)", "Village (km)", "Water (km)", "Score"]
tb = s.shapes.add_table(11, len(cols), Inches(.55), Inches(1.3), Inches(12.2), Inches(5.3)).table
for j, wd in enumerate((.7, 1.45, 1.45, 2.75, 1.6, .95, 1.2, 1.1, 1.0)):
    tb.columns[j].width = Inches(wd)
for i in range(11):
    vals = cols if i == 0 else (lambda r: [str(int(r["rank"])), r["range"].title(), str(r.compartment), r.structure.replace(" / earthen pond", "").replace(" / stop dam", ""), f"{r.catchment_ha:.0f}", f"{r.slope_deg:.1f}", f"{r.village_km:.1f}", f"{r.water_km:.1f}", f"{r.score:.1f}"])(top.iloc[i - 1])
    for j, v in enumerate(vals):
        c = tb.cell(i, j); c.text = v; c.fill.solid(); c.fill.fore_color.rgb = FOREST if i == 0 else (TINT if i % 2 == 0 else WHITE)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        q = c.text_frame.paragraphs[0]; q.alignment = PP_ALIGN.RIGHT if j >= 4 or j == 0 else PP_ALIGN.LEFT
        q.runs[0].font.size, q.runs[0].font.name, q.runs[0].font.bold = Pt(13), BODY, i == 0
        q.runs[0].font.color.rgb = WHITE if i == 0 else INK

# 14 access
s = slide("Priority sites could close major water-access gaps",
          "These are potential gains. They assume all 50 sites are built and hold water through the dry season. Estimated inflow is not the same as storage.")
image(s, "42_water_access_before_after", .55, 1.15, 6.4, 5.9, "Distance to water before and after priority sites")
card(s, 7.3, 1.35, 5.5, 1.55, f"{acc['2km']['reserve_before_pct']:.0f}% → {acc['2km']['reserve_after_pct']:.0f}%", "of reserve within 2 km of water", WATER)
card(s, 7.3, 3.08, 5.5, 1.55, f"{acc['2km']['elephant_records_before_pct']:.0f}% → {acc['2km']['elephant_records_after_pct']:.0f}%", "of elephant records within 2 km of water")
card(s, 7.3, 4.81, 5.5, 1.55, f"{rep['top50_total_yield_Mm3']:.1f} million m³", "modelled annual catchment inflow, not storage", EMBER)

# 15 limits
s = slide("Results need field verification before construction",
          "The largest uncertainties are the expert AHP weights, the RUSLE C factor, coarse soil data, a surface DEM under forest canopy and village points that do not show the full settlement footprint.")
card(s, .55, 1.35, 3.85, 1.65, f"{rep['top50_stable_ge80']}/50", "sites stay in the top 50 in ≥80% of weight tests")
card(s, .55, 3.2, 3.85, 1.65, f"{qa['merit_check_1_10km2']['within_factor2_pct']:.0f}%", "of small catchments within 2× of MERIT Hydro", WATER)
card(s, .55, 5.05, 3.85, 1.65, f"{rep['top50_pass_tableC']}/50", "sites pass soil limit under both C methods", EMBER)
text(s, 4.85, 1.4, 7.9, 5.3, ["Village points do not capture full settlement extent or scattered houses.",
                               "Soil data are 250 m, so percolation and seepage need site tests.",
                               "The 30 m DEM includes canopy height, so dam axes need ground survey.",
                               "Rainfall is nearly uniform across the reserve at CHIRPS resolution.",
                               "Elephant records may reflect where patrols and reporting occurred.",
                               "AHP weights are analyst judgements and need expert review."], 20)

# 16 next steps
s = slide("Recommended next steps", "Ask the group to confirm the 2 km base case, review the AHP weights and approve field checks for the first priority sites.", dark=True)
ns = [("Confirm", "Adopt the 2 km base case and review weights with reserve staff"), ("Verify", "Inspect Priority 1 sites for soil, seepage, access and elephant movement"),
      ("Design", "Size structures for dry-season demand, evaporation and sediment"), ("Monitor", "Track water persistence, elephant use and conflict after construction")]
for i, (h, b) in enumerate(ns):
    x = .55 + i * 3.1
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.9), Inches(2.85), Inches(3.5))
    c.adjustments[0] = .05
    c.fill.solid(); c.fill.fore_color.rgb = FOREST; c.line.fill.background(); c.shadow.inherit = False
    text(s, x + .22, 2.15, 2.42, .5, h, 24, WHITE, True, HEAD)
    text(s, x + .22, 2.9, 2.42, 2.3, b, 17, WHITE)
text(s, .55, 5.85, 12.2, .6, "Use the interactive map during discussion to compare the 1–5 km scenarios.", 18, RGBColor.from_string("BFE3D0"))

f = OUT / "Lemru_Water_Conservation_Site_Selection.pptx"
p.save(f)
print(f, len(p.slides))
