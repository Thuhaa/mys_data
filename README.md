# mys_data

Building centroids for **Selangor, Kuala Lumpur and Putrajaya**, Malaysia: one point per building footprint, 2,337,721 in total. They are split into one FlatGeobuf file per district so each file stays under GitHub's 100 MB limit.

All files are served from `raw.githubusercontent.com`, which supports HTTP range requests and CORS. Tools that read FlatGeobuf over HTTP, such as GDAL `/vsicurl/` and flatgeobuf.js, can fetch only the area they need.

## Files

Base URL: `https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/`

| District | Code | Buildings | Size | URL |
|---|---|---:|---:|---|
| Klang | 1001 | 327,796 | 69.0 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1001_klang.fgb |
| Kuala Langat | 1002 | 179,606 | 37.8 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1002_kuala-langat.fgb |
| Kuala Selangor | 1004 | 138,971 | 29.3 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1004_kuala-selangor.fgb |
| Sabak Bernam | 1005 | 75,342 | 15.9 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1005_sabak-bernam.fgb |
| Ulu Langat | 1006 | 361,571 | 76.2 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1006_ulu-langat.fgb |
| Ulu Selangor | 1007 | 113,450 | 23.9 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1007_ulu-selangor.fgb |
| Petaling | 1008 | 435,549 | 91.8 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1008_petaling.fgb |
| Gombak | 1009 | 244,300 | 51.5 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1009_gombak.fgb |
| Sepang | 1010 | 134,961 | 28.4 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1010_sepang.fgb |
| W.P. Kuala Lumpur | 1400 | 279,874 | 61.1 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1400_kuala-lumpur.fgb |
| W.P. Putrajaya | 1600 | 13,857 | 3.0 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1600_putrajaya.fgb |
| Unassigned (outside every district) | – | 32,444 | 6.1 MB | https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/unassigned.fgb |

**Index:** https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/index.json lists every file with its URL, district, building count, size in bytes and bounding box, so code can discover the files without hard-coding them.

`unassigned.fgb` holds buildings that fall outside every district polygon. Most are within the 1 km buffer the source extract added around the three states, plus some along the coast.

## Attributes

CRS is EPSG:4326 and the geometry type is Point. Each point carries the attributes of its source footprint:

| Field | Description |
|---|---|
| `boundary_id`, `s2_id`, `geohash` | Source partitioning and location keys |
| `bf_source` | Footprint source: `google` or `microsoft` |
| `confidence` | Detection confidence (Google footprints only) |
| `area_in_meters` | Footprint area in m² |
| `country_iso` | `MYS` |
| `adm_name`, `adm_code` | State assigned in the source extract (e.g. `Selangor`, `MY-10`) |

The district split comes from the district boundaries in this repo (`outputs/adm2_district_2026.fgb`). It is independent of `adm_name` and `adm_code`, so the two can disagree for buildings near a state border.

## Usage

Python (GeoPandas / pyogrio), reading only a bounding box:

```python
import geopandas as gpd

url = "https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1400_kuala-lumpur.fgb"
gdf = gpd.read_file(url, bbox=(101.68, 3.13, 101.72, 3.17))
```

All districts, using the index:

```python
import geopandas as gpd, pandas as pd, requests

index = requests.get("https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/index.json").json()
gdf = pd.concat([gpd.read_file(t["url"]) for t in index["tiles"]], ignore_index=True)
```

GDAL / ogr2ogr:

```sh
ogrinfo -so /vsicurl/https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids/1600_putrajaya.fgb
```

QGIS: *Layer → Add Layer → Add Vector Layer → Protocol: HTTP(S)* and paste a file URL.

## How it was made

1. `src/vida_buildings.py` extracts footprints for the three states from the VIDA combined Google–Microsoft Open Buildings dataset.
2. `src/building_centroids.py` converts each footprint to its centroid. Pass `--inside` to use a point guaranteed to lie inside the footprint instead.
3. `src/split_centroids.py` assigns each point to a district and writes the per-district files and `index.json`.

```sh
uv run src/building_centroids.py
uv run src/split_centroids.py --base-url https://raw.githubusercontent.com/Thuhaa/mys_data/main/outputs/building_centroids
```

## Source and license

Footprints come from [Google–Microsoft Open Buildings, combined by VIDA](https://source.coop/vida/google-microsoft-open-buildings) (readme version 2.0, 2024-09-04), licensed under [ODbL-1.0](https://opendatacommons.org/licenses/odbl/1-0/). This derived data is shared under the same license. Attribute the source and keep derived databases under ODbL.
