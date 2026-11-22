# Lemru Elephant Reserve water-conservation sites

An interactive map and reproducible GIS workflow for identifying candidate water-conservation sites for elephants in the Lemru Elephant Reserve, Chhattisgarh, India.

The sites are desk-based candidates. Each one needs field verification of soil, seepage, dam feasibility, land status and elephant movement before any construction.

## Interactive map

The website is in [`docs/index.html`](docs/index.html). It lets you compare 1–5 km village buffers, switch between 13 map layers and inspect each candidate site.

## Method

1. Align project layers and Earth Engine data to a 30 m grid.
2. Model drainage, SCS-CN runoff and RUSLE soil loss.
3. Exclude land that fails limits for village distance, slope, soil loss, mines, roads, land use and existing water.
4. Score remaining drainage lines with 20 AHP-weighted criteria.
5. Select the highest-scoring sites at least 2 km apart and test sensitivity to the weights.

The base case keeps sites at least 2 km from the supplied village locations.

## Data

Earth Engine inputs are CHIRPS rainfall, Copernicus GLO-30 DEM, OpenLandMap soil, Sentinel-2, JRC Global Surface Water, ESA WorldCover and MERIT Hydro.

Project source data and generated rasters are not included in this repository, so the scripts need those local inputs to run.

## Run the workflow

Run the scripts in `scripts/` in numeric order from the project root. Set `EE_PROJECT` to your Google Cloud project ID before running the download step:

```bash
export EE_PROJECT="your-project-id"
python scripts/01_download_gee.py
```

`05_dynamic_map.py` takes the paths to `leaflet.js` and `leaflet.css` as arguments.

## Limitations

- AHP weights are expert judgements and should be reviewed by reserve staff.
- RUSLE values depend strongly on the C-factor method, so use them mainly to compare areas.
- Village points do not show full settlement extent.
- The DEM is a surface model that includes forest canopy.
- Modelled inflow is not the same as usable storage.
