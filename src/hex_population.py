#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "exactextract>=0.2",
#     "h3>=4.1",
#     "geopandas>=1.0",
#     "pyogrio>=0.10",
#     "rasterio>=1.3",
#     "shapely>=2.0",
#     "numpy",
#     "pandas",
#     "requests",
# ]
# ///
"""
Population and persons with disabilities per H3 hexagon and per admin unit,
from the WorldPop age-sex rasters held to official totals, using exactextract.

Study area
  Default: Selangor, Kuala Lumpur and Putrajaya from geoBoundaries ADM1.
  --mukims DIR: the UNDP mukim layer (Census 2020 geography). The mukims become
  the study area, and the script also writes admin 1 (state), 2 (district) and
  3 (mukim) outputs. Districts and states are dissolved from the mukims, so the
  three levels nest and add up. A file in the folder with KOD_SEKSYE and no
  MUKIMCODE is read as admin 4 (units inside a mukim, such as Putrajaya's
  precincts): each unit is clipped to its parent mukim and the units are held
  to the mukim's totals, so they add up to it.
  Neighbouring mukims in the source overlap and leave thin gaps. Overlaps go to
  the smaller unit, each gap goes to the neighbour sharing most of its edge, and
  edges snap to a 0.1 mm grid, so the mukims tile the study area and districts
  and states dissolve without seams. Admin 4 units get the same treatment
  inside their mukim.

How it works
  1. Covers the study area with every H3 cell that touches it, clipping the
     cells on the boundary so they only count people inside it.
  2. Runs exactextract's coverage-weighted sum of every raster over the cells
     and the admin units. The rasters stay on their own grid.
  3. Builds the CGT groups (total, 0 to 4, 60 and over, each by sex).
  4. Builds child groups by single year of age. WorldPop gives under 1, 1 to 4,
     5 to 9 and so on, so a single year inside a band is the band value times
     that year's share of the band in UN World Population Prospects 2024 for
     Malaysia in 2026, for the same sex.
  5. Estimates persons with disabilities:
       adults 18+   NHMS 2023 rates by age group, made sex-specific by raking
                    to the NHMS rates for women and men, then scaled so each
                    state or federal territory matches its own NHMS 2023 rate
       ages 2-17    NHMS 2019 Child Functioning Module rate, 4.7%
     Inside a territory the estimate follows the local age and sex mix.
     Hexagons crossing a state line are cut along it, so each part takes its
     own state's factor; summed by state, the hexagons match the state tables
     apart from people in the sliver gaps between mukims.
  6. Checks that the hexagons add back to the study-area totals, how much of
     each raster falls outside the study area, and whether the female, male
     and total rasters agree.

Definitions
  Adults: NHMS 2023 "overall disability", Washington Group Short Set: some
  difficulty in two domains, or a lot of difficulty or cannot do at all in one.
  This is broader than the Washington Group's recommended cut-off (a lot of
  difficulty or worse in at least one domain), so counts run higher than
  figures that use it.
  Children: Washington Group/UNICEF Child Functioning Module cut-offs.
  Children under 2 are not covered by either instrument.

Child groups (completed years, both ends included)
  lt1 = age 0, 0_2 = ages 0-1, 2_4 = ages 2-3 (with 0_2, the TASKA range,
  under 4), 4_6 = ages 4-6 (preschool age). Change with --child-groups.

Outputs, in --out-dir
  hex_r{res}_population_{year}.fgb            one row per H3 cell
  hex_r{res}_population_{year}_checks.json    inputs, rates and reconciliation
  admin_population_{year}.gpkg                adm1_state, adm2_district, adm3_mukim, adm4 (--mukims)
  adm1_state_{year}.csv ... adm4_{year}.csv   the same tables without geometry (--mukims)
  population_by_area_{year}.csv               per state (geoBoundaries mode)

Usage (uv installs the dependencies above on the first run)
  uv run hex_population.py --rasters data/cn_rasters --mukims data/mukim_2020
  uv run hex_population.py --rasters data/cn_rasters --mukims data/mukim_2020 --res 8 9
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
from exactextract import exact_extract

# Adding many group columns one by one only triggers pandas' fragmentation notice
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SCRIPT_VERSION = "2.2"
AGES = ["00", "01"] + [f"{a:02d}" for a in range(5, 95, 5)]  # 00, 01, 05 ... 90
GROUP_AGES = {"0_4": ["00", "01"], "60P": ["60", "65", "70", "75", "80", "85", "90"]}
DEFAULT_CODES = ["MY-10", "MY-14", "MY-16"]  # Selangor, Kuala Lumpur, Putrajaya
DEFAULT_BOUNDARIES = Path("vida_cache/boundaries/geoBoundaries-MYS-ADM1.geojson")
GB_GITHUB = ("https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/main/"
             "releaseData/gbOpen/{iso3}/ADM1/geoBoundaries-{iso3}-ADM1.geojson")
# Holes below this size in the dissolved study area are slivers between
# neighbouring boundaries; filling them matches the building download.
MIN_HOLE_M2 = 1_000_000
# Polygon parts smaller than this left by the overlap removal are specks and zero-width slivers
MIN_PART_M2 = 10
# A detached piece of a mukim smaller than this, sitting wholly inside other mukims, is an
# overlap-removal remnant; it goes to the unit around it
MIN_ENCLAVE_M2 = 10_000
# Unions for the output layers snap to this grid (about 0.1 mm) so shared edges merge without seams
SNAP_GRID_DEG = 1e-9

# WorldPop R2025A names: mys_f_00_2026_..., mys_t_60_2026_..., mys_T_F_2026_..., mys_pop_2026_...
NAME_PATTERNS = [
    (re.compile(r"^(?P<iso>[a-z]{3})_(?P<sex>[fmt])_(?P<age>\d{2})_(?P<year>\d{4})_.*\.tif$"), "{sex}_{age}"),
    (re.compile(r"^(?P<iso>[a-z]{3})_T_(?P<sex>[FM])_(?P<year>\d{4})_.*\.tif$"), "T_{sex}"),
    (re.compile(r"^(?P<iso>[a-z]{3})_pop_(?P<year>\d{4})_.*\.tif$"), "pop"),
]
EXPECTED = [f"{s}_{a}" for s in "fmt" for a in AGES] + ["T_F", "T_M", "pop"]

# UN World Population Prospects 2024, medium variant: Malaysia (UN code 458),
# 2026, population by single year of age in thousands as (female, male).
# From the UN Population Division's wpp2024 R package, popprojAge1dt.
UN_SOURCE = "UN World Population Prospects 2024, medium variant, Malaysia, 2026 (wpp2024 R package, popprojAge1dt)"
UN_YEAR = "2026"
UN_SINGLE_AGE = {
    0: (214.474, 228.043), 1: (215.774, 229.428), 2: (217.331, 231.119), 3: (216.245, 229.947),
    4: (214.094, 227.652), 5: (212.087, 225.503), 6: (225.952, 240.360), 7: (235.995, 248.061),
    8: (257.954, 267.085), 9: (269.594, 279.988), 10: (276.571, 290.924), 11: (282.589, 300.172),
    12: (275.799, 295.212), 13: (272.558, 296.825), 14: (272.450, 297.507), 15: (263.544, 286.111),
    16: (268.370, 291.554), 17: (267.644, 289.596), 18: (267.142, 288.201), 19: (274.519, 295.264),
}
# WorldPop band that holds each single year of age
BAND_OF_AGE = {0: "00", **{a: "01" for a in range(1, 5)}, **{a: f"{a // 5 * 5:02d}" for a in range(5, 20)}}
DEFAULT_CHILD_GROUPS = ["lt1=0", "0_2=0-1", "2_4=2-3", "4_6=4-6"]

# NHMS 2023 Technical Report (Institute for Public Health, MOH), Table 4.16.2:
# prevalence of overall disability, adults 18+, in percent. NHMS suppresses
# ages 18-19, so they take the 20-29 rate.
NHMS_SOURCE = "NHMS 2023 Technical Report, Institute for Public Health, Table 4.16.2 (overall disability, adults 18+)"
NHMS_DEFINITION = ("some difficulty in two domains, or a lot of difficulty or cannot do at all in one domain "
                   "(Washington Group Short Set)")
NHMS_AGE_RATES = {"18_19": 3.1, "20_29": 3.1, "30_39": 2.7, "40_49": 6.1, "50_59": 10.4, "60P": 26.0}
NHMS_SEX_RATES = {"f": 9.0, "m": 7.5}
NHMS_STATE_RATES = {  # code: (prevalence, lower 95% CI, upper 95% CI)
    "MY-01": (6.3, 4.66, 8.39), "MY-02": (7.0, 5.24, 9.42), "MY-03": (6.1, 3.40, 10.76),
    "MY-04": (4.5, 2.25, 8.92), "MY-05": (6.5, 4.05, 10.39), "MY-06": (8.1, 6.05, 10.76),
    "MY-07": (7.7, 5.43, 10.72), "MY-08": (11.2, 7.82, 15.84), "MY-09": (8.4, 6.14, 11.35),
    "MY-10": (5.9, 4.53, 7.55), "MY-11": (7.2, 4.30, 11.74), "MY-12": (6.9, 4.47, 10.57),
    "MY-13": (24.1, 19.23, 29.80), "MY-14": (6.3, 4.76, 8.38), "MY-15": (8.3, 5.39, 12.51),
    "MY-16": (10.2, 7.06, 14.61),
}
# WPP 2024: Malaysia 2023 population by NHMS age group, thousands (female, male),
# used to rake the age rates to the NHMS rates for women and men.
NATIONAL_ADULTS_2023 = {
    "18_19": (566.075, 621.785), "20_29": (2893.288, 3343.228), "30_39": (2844.318, 3386.227),
    "40_49": (2260.811, 2602.252), "50_59": (1645.357, 1718.660), "60P": (2065.243, 1973.240),
}
CHILD_SOURCE = "NHMS 2019, Washington Group/UNICEF Child Functioning Module, ages 2-17"
CHILD_DISABILITY_RATE = 4.7
# JKM registered persons with disabilities (active), 31 May 2025. Putrajaya has no row of its own.
JKM_SOURCE = "JKM, Statistik OKU (aktif) berdaftar mengikut negeri dan kategori, 31 May 2025"
JKM_REGISTERED = {"MY-10": 132544, "MY-14": 56402}

# WorldPop bands (and single ages inside a band) for each NHMS age group
ADULT_GROUPS = {
    "18_19": [("15", (18, 19))],
    "20_29": [("20", None), ("25", None)],
    "30_39": [("30", None), ("35", None)],
    "40_49": [("40", None), ("45", None)],
    "50_59": [("50", None), ("55", None)],
    "60P": [(b, None) for b in ("60", "65", "70", "75", "80", "85", "90")],
}
CHILD_PARTS = [("01", (2, 3, 4)), ("05", None), ("10", None), ("15", (15, 16, 17))]

# UNDP mukim layer (Census 2020 geography)
MUKIM_FIELDS = ("NAM", "MUKIMCODE", "KOD_NEGERI", "KOD_DAERAH", "NAMA_DAERA", "KOD_MUKIM", "KATEGORI")
# A file with KOD_SEKSYE and no MUKIMCODE holds admin 4 units inside a mukim (e.g. Putrajaya's precincts)
ADM4_FIELDS = ("NAM", "KOD_NEGERI", "KOD_DAERAH", "KOD_MUKIM", "KOD_SEKSYE")
STATE_NAMES = {
    "01": "Johor", "02": "Kedah", "03": "Kelantan", "04": "Melaka", "05": "Negeri Sembilan", "06": "Pahang",
    "07": "Pulau Pinang", "08": "Perak", "09": "Perlis", "10": "Selangor", "11": "Terengganu", "12": "Sabah",
    "13": "Sarawak", "14": "W.P. Kuala Lumpur", "15": "W.P. Labuan", "16": "W.P. Putrajaya",
}


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


# ----------------------------------------------------------------- rasters --

def find_rasters(folder: Path):
    found, years, skipped = {}, set(), []
    for path in sorted(folder.glob("*.tif")):
        for rx, key_format in NAME_PATTERNS:
            match = rx.match(path.name)
            if match:
                found[key_format.format(**match.groupdict())] = path
                years.add(match["year"])
                break
        else:
            skipped.append(path.name)
    if not found:
        sys.exit(f"No WorldPop-style rasters found in {folder}")
    if len(years) > 1:
        sys.exit(f"The folder mixes years {sorted(years)}; point --rasters at one year.")
    return found, years.pop(), skipped


def inspect_rasters(found: dict):
    """Grid, NoData and whole-raster totals; stops on problems that would corrupt sums."""
    reference, grid_info, fatal, notes, totals = None, None, [], [], {}
    for key, path in found.items():
        with rasterio.open(path) as src:
            grid = (src.crs.to_string() if src.crs else None, tuple(round(v, 10) for v in src.transform[:6]),
                    src.width, src.height)
            if src.count != 1:
                fatal.append(f"{path.name}: {src.count} bands, expected 1")
            if src.nodata is None:
                fatal.append(f"{path.name}: no NoData value set, so fill values would be summed")
            total, negative = 0.0, False
            for _, window in src.block_windows(1):  # block by block, so a national raster fits in memory
                data = src.read(1, window=window)
                valid = np.isfinite(data) if np.issubdtype(data.dtype, np.floating) else np.ones(data.shape, bool)
                if src.nodata is not None:
                    valid &= data != src.nodata
                values = data[valid]
                negative |= bool((values < 0).any())
                total += float(values.sum(dtype=np.float64))
            if negative:
                notes.append(f"{path.name}: has negative values")
            totals[key] = total
            if reference is None:
                reference = (grid, path.name)
                grid_info = {"crs": grid[0], "width": src.width, "height": src.height,
                             "pixel_size": [src.transform.a, -src.transform.e], "dtype": src.dtypes[0],
                             "nodata": src.nodata, "bounds": list(src.bounds)}
            elif grid != reference[0]:
                notes.append(f"{path.name}: grid differs from {reference[1]}")
    if fatal:
        sys.exit("Raster problems:\n  " + "\n  ".join(fatal))
    return grid_info, totals, notes


# -------------------------------------------------------------- study area --

def polygonal(geom):
    """Polygonal part of a geometry (drops stray lines and points), or None."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    polys = []
    for part in shapely.get_parts(geom):
        part = polygonal(part)
        if part is not None:
            polys.extend(shapely.get_parts(part))
    if not polys:
        return None
    return shapely.MultiPolygon(polys) if len(polys) > 1 else polys[0]


def load_geoboundaries(args):
    """States from geoBoundaries (or another boundary file) as parts with adm_name, adm_code, state_code."""
    path = args.boundaries
    if not path.exists():
        if path != DEFAULT_BOUNDARIES:
            sys.exit(f"Boundary file not found: {path}")
        url = GB_GITHUB.format(iso3=args.country)
        log(f"Boundaries not cached, downloading {url}")
        r = requests.get(url, timeout=180, headers={"User-Agent": f"cgt-hex-population/{SCRIPT_VERSION}"})
        r.raise_for_status()
        if not r.content.lstrip().startswith(b"{"):
            sys.exit("The boundary download is not GeoJSON. Pass a boundary file with --boundaries.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    gdf = gpd.read_file(path)

    if args.all:
        selected = gdf
    else:
        if args.code_field not in gdf.columns:
            sys.exit(f"No '{args.code_field}' field in {path.name}. Use --code-field, or --all.")
        codes = [c.upper() for c in (args.iso or DEFAULT_CODES)]
        column = gdf[args.code_field].astype(str).str.upper()
        missing = sorted(set(codes) - set(column))
        if missing:
            names = gdf[args.name_field].astype(str) if args.name_field in gdf.columns else column
            available = ", ".join(f"{n} ({c})" for n, c in zip(names, column))
            sys.exit(f"Not found: {', '.join(missing)}\nAvailable: {available}")
        selected = gdf[column.isin(codes)]
    selected = selected.set_crs("EPSG:4326") if selected.crs is None else selected.to_crs("EPSG:4326")

    n = len(selected)
    names = (selected[args.name_field].astype(str).to_numpy() if args.name_field in selected.columns
             else [f"area {i + 1}" for i in range(n)])
    codes = (selected[args.code_field].astype(str).to_numpy() if args.code_field in selected.columns
             else [f"A{i + 1}" for i in range(n)])
    parts = gpd.GeoDataFrame(
        {"adm_name": names, "adm_code": codes,
         "state_code": [c if c in NHMS_STATE_RATES else None for c in codes], "state_name": names},
        geometry=[polygonal(g) for g in selected.geometry.make_valid()], crs="EPSG:4326",
    )
    return parts[parts.geometry.notna()].reset_index(drop=True), str(path.resolve())


def assign_gaps(geoms, gaps):
    """Merge each gap piece into the polygon that shares the longest stretch of its edge."""
    shapes = np.empty(len(geoms), dtype=object)
    shapes[:] = [g if g is not None else shapely.Polygon() for g in geoms]
    tree = shapely.STRtree(shapes)
    extra: dict = {}
    for gap in gaps:
        for piece in shapely.get_parts(gap):
            near = tree.query(piece, predicate="intersects")
            if not near.size:
                near = tree.query(shapely.buffer(piece, 1e-7), predicate="intersects")  # about 1 cm
            if not near.size:
                continue
            shared = shapely.length(shapely.intersection(piece.boundary, shapes[near]))
            if shared.max() <= 0:
                shared = shapely.length(shapely.intersection(piece.boundary, shapely.buffer(shapes[near], 1e-7)))
            extra.setdefault(int(near[np.argmax(shared)]), []).append(piece)
    out = list(geoms)
    for i, pieces in extra.items():
        out[i] = safe_union([out[i]] + pieces)
    return out, sum(len(v) for v in extra.values())


def tidy_parts(geoms, utm):
    """Drop specks under MIN_PART_M2, and take out detached pieces under MIN_ENCLAVE_M2 that sit
    wholly inside other mukims, so fill_gaps hands them to the unit around them."""
    owner, pieces = [], []
    for i, geom in enumerate(geoms):
        if geom is not None:
            for piece in shapely.get_parts(geom):
                owner.append(i)
                pieces.append(piece)
    if not pieces:
        return geoms, 0, 0
    owner = np.array(owner)
    size = gpd.GeoSeries(pieces, crs="EPSG:4326").to_crs(utm).area.to_numpy()
    keep = size >= MIN_PART_M2
    largest = {}
    for k, (i, s) in enumerate(zip(owner, size)):
        if s > size[largest.get(i, k)] or i not in largest:
            largest[i] = k
    shapes = np.empty(len(geoms), dtype=object)
    shapes[:] = [g if g is not None else shapely.Polygon() for g in geoms]
    tree = shapely.STRtree(shapes)
    enclaves = 0
    for k in np.flatnonzero(keep & (size < MIN_ENCLAVE_M2)):
        if largest[owner[k]] == k:
            continue
        near = [j for j in tree.query(pieces[k], predicate="intersects") if j != owner[k]]
        if near:
            shared = shapely.length(shapely.intersection(pieces[k].boundary, safe_union(shapes[near])))
            if shared >= 0.99 * pieces[k].boundary.length:
                keep[k] = False
                enclaves += 1
    kept: dict = {}
    for i, piece, k in zip(owner, pieces, keep):
        if k:
            kept.setdefault(i, []).append(piece)
    out = [None if i not in kept else kept[i][0] if len(kept[i]) == 1 else shapely.MultiPolygon(kept[i])
           for i in range(len(geoms))]
    return out, int((size < MIN_PART_M2).sum()), enclaves


def fill_gaps(geoms, utm):
    """Give each sliver gap between neighbours (a hole under MIN_HOLE_M2 in their union) to the
    neighbour sharing most of its edge."""
    union = safe_union(geoms)
    holes = [shapely.Polygon(ring) for poly in shapely.get_parts(union) for ring in poly.interiors] \
        if union is not None else []
    if not holes:
        return geoms, 0, 0.0
    sizes = gpd.GeoSeries(holes, crs="EPSG:4326").to_crs(utm).area.to_numpy()
    gaps = [polygonal(shapely.difference(h, union)) for h, s in zip(holes, sizes) if s < MIN_HOLE_M2]
    gaps = [g for g in gaps if g is not None]
    area = gpd.GeoSeries(gaps, crs="EPSG:4326").to_crs(utm).area.sum() / 1e6 if gaps else 0.0
    out, placed = assign_gaps(geoms, gaps)
    return out, placed, area


def load_mukims(folder: Path):
    """UNDP mukim layer: every vector file in the folder, repaired, with overlaps removed.
    Files with KOD_SEKSYE and no MUKIMCODE are admin 4 layers and are returned apart."""
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".shp", ".gpkg", ".geojson", ".fgb"))
    if not files:
        sys.exit(f"No vector files in {folder}")
    frames, adm4_frames = [], []
    for path in files:
        g = gpd.read_file(path)
        g = g.set_crs("EPSG:4326") if g.crs is None else g.to_crs("EPSG:4326")
        if "KOD_SEKSYE" in g.columns and "MUKIMCODE" not in g.columns:
            missing = [f for f in ADM4_FIELDS if f not in g.columns]
            if missing:
                sys.exit(f"{path.name} looks like an admin 4 layer but lacks the fields {', '.join(missing)}")
            adm4_frames.append(g.assign(source_file=path.name))
            continue
        missing = [f for f in MUKIM_FIELDS if f not in g.columns]
        if missing:
            sys.exit(f"{path.name} lacks the fields {', '.join(missing)}")
        frames.append(g.assign(source_file=path.name))
    if not frames:
        sys.exit(f"No mukim layer in {folder}")
    g = pd.concat(frames, ignore_index=True)
    if not g["MUKIMCODE"].is_unique:
        dupes = g.loc[g["MUKIMCODE"].duplicated(), "MUKIMCODE"].unique()[:5]
        sys.exit(f"MUKIMCODE repeats across the files, e.g. {', '.join(map(str, dupes))}")
    invalid = int((~g.geometry.is_valid).sum())
    geoms = [polygonal(x) for x in g.geometry.make_valid()]

    # Overlapping slivers go to the smaller unit, so towns keep their full area
    # and no area is counted twice when mukims are added up to districts.
    utm = g.estimate_utm_crs()
    area = gpd.GeoSeries(geoms, crs="EPSG:4326").to_crs(utm).area.to_numpy()
    tree = shapely.STRtree([x if x is not None else shapely.Polygon() for x in geoms])
    fixed: dict = {}
    for i in np.argsort(area, kind="stable"):
        geom = geoms[i]
        if geom is None:
            continue
        for j in tree.query(geom):
            if j != i and j in fixed and fixed[j] is not None and geom.intersects(fixed[j]):
                geom = polygonal(geom.difference(fixed[j]))
                if geom is None:
                    break
        fixed[i] = geom
    clean = [fixed.get(i) for i in range(len(geoms))]
    overlap_km2 = (np.nansum(area) - gpd.GeoSeries([x for x in clean if x is not None], crs="EPSG:4326")
                   .to_crs(utm).area.sum()) / 1e6
    # Specks left by the overlap removal go, and sliver gaps between neighbours join the
    # neighbour sharing most of their edge. Snapping to a 0.1 mm grid then makes shared edges
    # identical, so the mukims tile the study area and districts and states dissolve without seams.
    clean, specks, enclaves = tidy_parts(clean, utm)
    clean, gaps, gap_km2 = fill_gaps(clean, utm)
    snapped = np.empty(len(clean), dtype=object)
    snapped[:] = clean
    clean = [polygonal(x) for x in shapely.set_precision(snapped, SNAP_GRID_DEG)]

    state = g["KOD_NEGERI"].astype(str).str.zfill(2)
    district = g["KOD_DAERAH"].astype(str).str.zfill(2)
    district_name = g["NAMA_DAERA"].astype(str).str.strip()
    no_district = district_name.str.upper().isin(["TIADA DAERAH", "NONE", "NAN", ""])
    district_name = district_name.where(~no_district, state.map(STATE_NAMES).fillna(state))
    parts = gpd.GeoDataFrame(
        {
            "adm_name": g["NAM"].astype(str).str.strip().to_numpy(),
            "adm_code": g["MUKIMCODE"].astype(str).to_numpy(),
            "category": g["KATEGORI"].astype(str).to_numpy(),
            "district_code": (state + district).to_numpy(),
            "district_name": district_name.to_numpy(),
            "state_code": ("MY-" + state).to_numpy(),
            "state_name": state.map(STATE_NAMES).fillna(state).to_numpy(),
            "mukim_key": (state + district + g["KOD_MUKIM"].astype(str).str.zfill(2)).to_numpy(),
        },
        geometry=clean, crs="EPSG:4326",
    )
    dropped = int(parts.geometry.isna().sum())
    parts = parts[parts.geometry.notna()].sort_values("adm_code").reset_index(drop=True)
    info = {"files": sorted(g["source_file"].unique()), "units": len(parts), "invalid_repaired": invalid,
            "overlap_removed_km2": round(overlap_km2, 3), "specks_dropped": specks,
            "enclave_remnants_moved": enclaves, "gaps_filled": gaps,
            "gap_area_filled_km2": round(gap_km2, 3), "units_dropped_empty": dropped}
    return parts, adm4_frames, info


def load_adm4(frames, parts):
    """Admin 4 units clipped to their parent mukim, matched on KOD_NEGERI, KOD_DAERAH and
    KOD_MUKIM (or, if the codes match no mukim, the mukim holding most of the unit)."""
    g = pd.concat(frames, ignore_index=True)
    key = (g["KOD_NEGERI"].astype(str).str.zfill(2) + g["KOD_DAERAH"].astype(str).str.zfill(2)
           + g["KOD_MUKIM"].astype(str).str.zfill(2))
    by_key = {k: i for i, k in enumerate(parts["mukim_key"])}
    mukims = parts.geometry.to_numpy()
    tree = shapely.STRtree(mukims)
    raw = [polygonal(x) for x in g.geometry.make_valid()]
    parent, clipped, by_overlap = [], [], 0
    for k, geom in zip(key, raw):
        j = by_key.get(k)
        if j is None and geom is not None:
            near = tree.query(geom, predicate="intersects")
            if near.size:
                j = int(near[np.argmax(shapely.area(shapely.intersection(geom, mukims[near])))])
                by_overlap += 1
        parent.append(-1 if j is None else j)
        clipped.append(polygonal(shapely.intersection(geom, mukims[j])) if j is not None and geom is not None else None)
    parent = np.array(parent)
    keep = (parent >= 0) & np.array([c is not None for c in clipped])
    mukim = parts.iloc[parent[keep]].reset_index(drop=True)
    section = g.loc[keep, "KOD_SEKSYE"].astype(str).str.strip().str.zfill(3).to_numpy()
    units = gpd.GeoDataFrame(
        {"adm4_code": mukim["adm_code"].to_numpy() + section,
         "adm4_name": g.loc[keep, "NAM"].astype(str).str.strip().to_numpy(),
         **{c: mukim[c].to_numpy() for c in ("adm_code", "adm_name", "district_code", "district_name",
                                              "state_code", "state_name")}},
        geometry=[c for c, k in zip(clipped, keep) if k], crs="EPSG:4326",
    )
    if not units["adm4_code"].is_unique:
        dupes = units.loc[units["adm4_code"].duplicated(), "adm4_code"].unique()[:5]
        sys.exit(f"Admin 4 codes (mukim code + KOD_SEKSYE) repeat, e.g. {', '.join(dupes)}")
    units = units.sort_values("adm4_code").reset_index(drop=True)

    utm = parts.estimate_utm_crs()
    # Strips of the mukim outside every unit (where the two layers' outlines differ) join the
    # unit sharing most of their edge, so the units tile their mukim.
    for code, rows in units.groupby("adm_code").groups.items():
        rows = list(rows)
        whole = parts.geometry.iloc[int(np.flatnonzero(parts["adm_code"].to_numpy() == code)[0])]
        rest = polygonal(shapely.difference(whole, safe_union(units.geometry.iloc[rows].to_numpy())))
        if rest is None:
            continue
        pieces = list(shapely.get_parts(rest))
        sizes = gpd.GeoSeries(pieces, crs="EPSG:4326").to_crs(utm).area.to_numpy()
        geoms, _ = assign_gaps(list(units.geometry.iloc[rows]), [p for p, s in zip(pieces, sizes) if s < MIN_HOLE_M2])
        units.loc[rows, "geometry"] = gpd.GeoSeries(geoms, index=rows, crs="EPSG:4326")
    inside = units.to_crs(utm)
    raw_km2 = gpd.GeoSeries(raw, crs="EPSG:4326").to_crs(utm).area.sum() / 1e6
    by_mukim = {}
    for code, grp in inside.groupby("adm_code"):
        whole = parts.loc[parts["adm_code"] == code].to_crs(utm).geometry.iloc[0]
        by_mukim[code] = {"units": len(grp), "mukim_km2": round(whole.area / 1e6, 3),
                          "mukim_area_outside_units_km2": round(
                              shapely.difference(whole, shapely.union_all(grp.geometry.to_numpy())).area / 1e6, 3)}
    info = {"files": sorted(g["source_file"].unique()), "units": len(units),
            "units_outside_every_mukim": int((~keep).sum()), "parents_found_by_overlap": by_overlap,
            "area_clipped_off_outside_parent_km2": round(raw_km2 - inside.area.sum() / 1e6, 3),
            "by_mukim": by_mukim}
    log(f"Admin 4: {len(units)} units from {', '.join(info['files'])} in {len(by_mukim)} mukim(s); "
        f"{info['area_clipped_off_outside_parent_km2']} km2 outside the parent mukim clipped off")
    return units, info


def hold_to_parent(child: pd.DataFrame, parent: pd.DataFrame, wp_cols, key="adm_code"):
    """Scale each band so the admin 4 units in a mukim add up to the mukim's own figure. Within
    a mukim the units keep their extracted shares (area shares for a band that is empty)."""
    raw = child[wp_cols].to_numpy()
    total = child.groupby(key)[wp_cols].transform("sum").to_numpy()
    area_share = (child["land_km2"] / child.groupby(key)["land_km2"].transform("sum")).to_numpy()[:, None]
    share = np.where(total > 0, raw / np.where(total > 0, total, 1.0), area_share)
    child[wp_cols] = parent.set_index(key).loc[child[key], wp_cols].to_numpy() * share


def admin4_table(units, mukim_df, found, wp_cols, laea, in_raster_crs, info):
    """Raster sums for the admin 4 units, held to their mukim's totals."""
    units["land_km2"] = units.to_crs(laea).area.to_numpy() / 1e6
    log(f"exactextract: {len(found)} rasters over {len(units)} admin 4 units")
    df = units.drop(columns="geometry").merge(
        zonal_sums(found, in_raster_crs(units[["adm4_code", "geometry"]]), ["adm4_code"]), on="adm4_code", how="left")
    df[wp_cols] = df[wp_cols].fillna(0.0)
    if "wp_pop" in wp_cols:
        ratio = (df.groupby("adm_code")["wp_pop"].sum() / mukim_df.set_index("adm_code")["wp_pop"]).dropna()
        info["extracted_vs_mukim_before_holding"] = {code: float(v - 1) for code, v in ratio.items()}
        for code, v in info["extracted_vs_mukim_before_holding"].items():
            log(f"  admin 4 units in {code} summed to {v:+.2%} of the mukim before being held to its totals")
    hold_to_parent(df, mukim_df, wp_cols)
    return df


def tag_children(zones, units, mukim_codes):
    """Admin 4 unit holding each zone's representative point. A point in a mukim that has admin 4
    units but outside all of them (a sliver) takes that mukim's nearest unit; -1 elsewhere."""
    points = shapely.point_on_surface(zones)
    geoms = units.geometry.to_numpy()
    point_idx, unit_idx = shapely.STRtree(geoms).query(points, predicate="intersects")
    out = np.full(len(zones), -1)
    out[point_idx[::-1]] = unit_idx[::-1]
    parent = units["adm_code"].to_numpy()
    for code in np.unique(parent):
        miss = np.flatnonzero((out < 0) & (mukim_codes == code))
        if miss.size:
            members = np.flatnonzero(parent == code)
            query_idx, nearest = shapely.STRtree(geoms[members]).query_nearest(points[miss], all_matches=False)
            out[miss[query_idx]] = members[nearest]
    return out


def safe_union(geoms, grid_size=None):
    """Union that survives the near-degenerate edges boundary files often have. With grid_size,
    vertices snap to that grid, which merges near-identical shared edges without seams."""
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    try:
        return polygonal(shapely.union_all(geoms, grid_size=grid_size))
    except shapely.errors.GEOSException:
        fixed = shapely.make_valid(np.array(geoms + [None], dtype=object)[:-1])
        return polygonal(shapely.union_all(fixed, grid_size=grid_size or SNAP_GRID_DEG))


def dissolve(parts: gpd.GeoDataFrame):
    """Union of the parts with sliver holes filled (sizes measured in a local UTM CRS), in EPSG:4326."""
    union = safe_union(parts.geometry.to_numpy())
    if union is None:
        sys.exit("The selected boundaries contain no polygons.")
    utm = parts.estimate_utm_crs()
    polygons = []
    for poly in shapely.get_parts(union):
        holes = list(poly.interiors)
        if holes:
            sizes = gpd.GeoSeries([shapely.Polygon(r) for r in holes], crs="EPSG:4326").to_crs(utm).area.to_numpy()
            holes = [ring for ring, size in zip(holes, sizes) if size >= MIN_HOLE_M2]
        polygons.append(shapely.Polygon(poly.exterior, holes))
    return shapely.MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]


def equal_area_crs(geom) -> str:
    """Lambert azimuthal equal-area centred on the study area, for km2 figures."""
    c = geom.centroid
    return f"+proj=laea +lat_0={c.y:.4f} +lon_0={c.x:.4f} +datum=WGS84 +units=m +no_defs"


# --------------------------------------------------------------------- hex --

def build_cells(aoi, res: int):
    shape = h3.geo_to_h3shape(shapely.geometry.mapping(aoi))
    returned = h3.h3shape_to_cells_experimental(shape, res, contain="overlap")
    # Overlap mode lists a cell once per polygon part it touches, so the coast's
    # islands produce duplicates; without this they would be counted twice.
    cells = np.array(sorted(set(returned)))
    hexes = np.array([shapely.Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells],
                     dtype=object)
    return cells, hexes, len(returned) - len(cells)


def clip_cells(hexes, aoi):
    shapely.prepare(aoi)
    inside = shapely.contains(aoi, hexes)
    zones = hexes.copy()
    edge = np.flatnonzero(~inside)
    if edge.size:
        zones[edge] = [polygonal(g) for g in shapely.intersection(hexes[edge], aoi)]
    keep = np.array([z is not None and not z.is_empty and z.area > 0 for z in zones])
    return zones, ~inside, keep


def tag_areas(zones, parts):
    """Index of the part holding each zone's representative point (the nearest part if none
    holds it, e.g. a point in a sliver gap), and how many needed the nearest part."""
    points = shapely.point_on_surface(zones)
    tree = shapely.STRtree(parts.geometry.to_numpy())
    point_idx, part_idx = tree.query(points, predicate="intersects")
    out = np.full(len(zones), -1)
    out[point_idx[::-1]] = part_idx[::-1]
    miss = np.flatnonzero(out < 0)
    if miss.size:
        query_idx, nearest = tree.query_nearest(points[miss], all_matches=False)
        out[miss[query_idx]] = nearest
    return out, int(miss.size)


def split_by_state(zones, parts):
    """Cut zones along state lines, so each piece takes its own state's disability factor.

    The pieces of a zone cover all of it: area outside every state (a sliver gap
    between units) stays with the zone's largest piece. Returns, per piece, the
    zone index, the state key and the geometry."""
    key = parts["state_code"].fillna(parts["adm_code"]).astype(str).to_numpy()
    codes = np.array(sorted(set(key)), dtype=object)
    geoms = parts.geometry.to_numpy()
    polys = np.empty(len(codes), dtype=object)
    polys[:] = [safe_union(geoms[key == c]) for c in codes]
    tree = shapely.STRtree(polys)
    zone_idx, state_idx = tree.query(zones, predicate="intersects")
    count = np.bincount(zone_idx, minlength=len(zones))
    state_of = np.full(len(zones), -1)
    state_of[zone_idx] = state_idx
    none = np.flatnonzero(count == 0)
    if none.size:
        query_idx, nearest = tree.query_nearest(zones[none], all_matches=False)
        state_of[none[query_idx]] = nearest

    whole = np.flatnonzero(count <= 1)
    piece_zone, piece_state, piece_geom = [whole], [state_of[whole]], list(zones[whole])
    touching: dict = {}
    sel = count[zone_idx] > 1
    for z, s in zip(zone_idx[sel].tolist(), state_idx[sel].tolist()):
        touching.setdefault(z, []).append(s)
    extra_zone, extra_state, split = [], [], 0
    for z, states in touching.items():
        cut = [(s, polygonal(shapely.intersection(zones[z], polys[s]))) for s in states]
        cut = sorted(((s, g) for s, g in cut if g is not None and g.area > 0), key=lambda t: -t[1].area)
        if len(cut) <= 1:  # only touches the other states along a line
            extra_zone.append(z)
            extra_state.append(cut[0][0] if cut else states[0])
            piece_geom.append(zones[z])
            continue
        split += 1
        rest = polygonal(shapely.difference(zones[z], shapely.union_all([g for _, g in cut[1:]])))
        for i, (s, g) in enumerate(cut):
            extra_zone.append(z)
            extra_state.append(s)
            piece_geom.append(rest if i == 0 and rest is not None else g)
    piece_zone.append(np.array(extra_zone, dtype=int))
    piece_state.append(np.array(extra_state, dtype=int))
    geom_array = np.empty(len(piece_geom), dtype=object)
    geom_array[:] = piece_geom
    return np.concatenate(piece_zone), codes[np.concatenate(piece_state)], geom_array, split


# ------------------------------------------------------------- extraction --

def zonal_sums(found: dict, zones: gpd.GeoDataFrame, include_cols) -> pd.DataFrame:
    keys = list(found)
    stems = {Path(found[k]).stem: k for k in keys}
    df = exact_extract([str(found[k]) for k in keys], zones, "sum", include_cols=include_cols,
                       output="pandas", strategy="raster-sequential")
    df = df.rename(columns={f"{stem}_sum": f"wp_{key}" for stem, key in stems.items()})
    value_cols = [f"wp_{k}" for k in keys]
    df[value_cols] = df[value_cols].fillna(0.0)
    return df


def add_groups(df: pd.DataFrame, year: str):
    """CGT groups from the raster columns; returns the column names made and any skipped."""
    made, skipped = [], []

    def put(name, cols):
        if all(c in df.columns for c in cols):
            df[name] = df[cols].sum(axis=1)
            made.append(name)
        else:
            skipped.append(name)

    put(f"cgt_pop_{year}", ["wp_pop"])
    put(f"cgt_pop_f_{year}", ["wp_T_F"])
    put(f"cgt_pop_m_{year}", ["wp_T_M"])
    for group, ages in GROUP_AGES.items():
        for sex in "tfm":
            put(f"cgt_{group}_{sex}_{year}", [f"wp_{sex}_{a}" for a in ages])
    return made, skipped


def load_single_age(path: Path | None):
    """Single-year population (female, male) for ages 0 to 19: the built-in UN table or a CSV."""
    if path is None:
        return UN_SINGLE_AGE, UN_SOURCE
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if not {"age", "female", "male"} <= cols.keys():
        sys.exit(f"{path} needs the columns age, female and male")
    table = {int(r[cols["age"]]): (float(r[cols["female"]]), float(r[cols["male"]])) for _, r in df.iterrows()}
    missing = [a for a in range(20) if a not in table]
    if missing:
        sys.exit(f"{path} is missing ages {missing}; ages 0 to 19 are needed")
    return {a: table[a] for a in range(20)}, str(path.resolve())


def single_year_shares(table: dict) -> dict:
    """Share of each single year of age within its WorldPop band, for f, m and t."""
    shares = {}
    for sex in "fmt":
        value = {a: (f if sex == "f" else m if sex == "m" else f + m) for a, (f, m) in table.items()}
        band_total: dict = {}
        for age, band in BAND_OF_AGE.items():
            band_total[band] = band_total.get(band, 0.0) + value[age]
        shares[sex] = {age: value[age] / band_total[band] for age, band in BAND_OF_AGE.items()}
    return shares


def parse_child_groups(specs) -> dict:
    groups = {}
    for spec in specs:
        label, _, ages = spec.partition("=")
        match = re.fullmatch(r"(\d{1,2})(?:-(\d{1,2}))?", ages.strip())
        if not re.fullmatch(r"[A-Za-z0-9_]+", label) or not match:
            sys.exit(f"Bad --child-groups entry '{spec}'. Use label=age or label=first-last, e.g. 4_6=4-6")
        first, last = int(match[1]), int(match[2] or match[1])
        if first > last or last > 19:
            sys.exit(f"Bad age range in '{spec}'; ages 0 to 19 are supported")
        groups[label] = (first, last)
    return groups


def add_child_groups(df: pd.DataFrame, year: str, shares: dict, groups: dict):
    """Child groups and single years 0 to the oldest age used, split out of the WorldPop bands."""
    top = max(last for _, last in groups.values())
    group_cols, single_cols, skipped = [], [], []
    for sex in "tfm":
        if not {f"wp_{sex}_{BAND_OF_AGE[a]}" for a in range(top + 1)} <= set(df.columns):
            skipped += [f"cgt_{label}_{sex}_{year}" for label in groups]
            continue
        single = {a: df[f"wp_{sex}_{BAND_OF_AGE[a]}"] * shares[sex][a] for a in range(top + 1)}
        for label, (first, last) in groups.items():
            name = f"cgt_{label}_{sex}_{year}"
            df[name] = sum(single[a] for a in range(first, last + 1))
            group_cols.append(name)
        for a in range(top + 1):
            name = f"cgt_age{a}_{sex}_{year}"
            df[name] = single[a]
            single_cols.append(name)
    return group_cols + single_cols, skipped


# -------------------------------------------------------------- disability --

def band_population(df: pd.DataFrame, sex: str, parts, shares: dict):
    """Population in a set of WorldPop bands, keeping only the listed single ages where given."""
    total = 0.0
    for band, ages in parts:
        column = df[f"wp_{sex}_{band}"]
        total = total + (column * sum(shares[sex][a] for a in ages) if ages else column)
    return total


def fit_adult_rates(age_rates: dict, sex_rates: dict, structure: dict) -> dict:
    """Sex-specific rates by age group: rake age rates x population to the NHMS age and sex margins."""
    groups = list(age_rates)
    pop = {(g, s): structure[g][0 if s == "f" else 1] for g in groups for s in "fm"}
    fit = {(g, s): pop[g, s] * age_rates[g] / 100 for g in groups for s in "fm"}
    age_target = {g: age_rates[g] / 100 * (pop[g, "f"] + pop[g, "m"]) for g in groups}
    sex_target = {s: sex_rates[s] / 100 * sum(pop[g, s] for g in groups) for s in "fm"}
    scale = sum(age_target.values()) / sum(sex_target.values())
    sex_target = {s: v * scale for s, v in sex_target.items()}
    for _ in range(500):
        for g in groups:
            k = age_target[g] / (fit[g, "f"] + fit[g, "m"])
            fit[g, "f"] *= k
            fit[g, "m"] *= k
        for s in "fm":
            k = sex_target[s] / sum(fit[g, s] for g in groups)
            for g in groups:
                fit[g, s] *= k
    return {(g, s): fit[g, s] / pop[g, s] for g in groups for s in "fm"}


def disability_inputs_present(df: pd.DataFrame) -> bool:
    bands = {b for parts in list(ADULT_GROUPS.values()) + [CHILD_PARTS] for b, _ in parts}
    return all(f"wp_{s}_{b}" in df.columns for s in "fm" for b in bands)


def calibration_factors(states: pd.DataFrame, rates: dict, shares: dict, mode: str) -> dict:
    """Per state: factor that brings the modelled adult rate to the state's NHMS rate."""
    out = {}
    adults = sum(band_population(states, s, parts, shares) for parts in ADULT_GROUPS.values() for s in "fm")
    expected = sum(band_population(states, s, parts, shares) * rates[g, s]
                   for g, parts in ADULT_GROUPS.items() for s in "fm")
    for code, a, e in zip(states["state_code"], adults, expected):
        nhms = NHMS_STATE_RATES.get(code)
        factor = 1.0
        if mode == "state" and nhms and e > 0:
            factor = nhms[0] / 100 * a / e
        out[code] = {"factor": factor, "adults": float(a), "rate_before_pct": 100 * e / a if a else None,
                     "nhms_rate_pct": nhms[0] if nhms else None, "nhms_ci_pct": list(nhms[1:]) if nhms else None,
                     "rate_after_pct": 100 * e * factor / a if a else None}
    return out


def add_disability(df: pd.DataFrame, year: str, rates: dict, shares: dict, child_rate: float, factor):
    """Persons with disabilities: adults by NHMS age-sex rates times the state factor, plus ages 2-17."""
    if not disability_inputs_present(df):
        return [], [f"cgt_dis_*_{year}"]
    parts = {}
    for s in "fm":
        older = band_population(df, s, ADULT_GROUPS["60P"], shares) * rates["60P", s] * factor
        adult = sum(band_population(df, s, p, shares) * rates[g, s] for g, p in ADULT_GROUPS.items()) * factor
        child = band_population(df, s, CHILD_PARTS, shares) * child_rate
        parts[s] = {"": adult + child, "adult_": adult, "child_": child, "60P_": older}
    names = []
    for kind in ("", "adult_", "child_", "60P_"):
        for s in "tfm":
            name = f"cgt_dis_{kind}{s}_{year}"
            df[name] = parts["f"][kind] + parts["m"][kind] if s == "t" else parts[s][kind]
            names.append(name)
    return names, []


def consistency(totals: dict) -> dict:
    """Do the female, male and total rasters agree inside the study area?"""
    def rel(a, b):
        return None if not b else (a - b) / b

    out = {}
    per_age = {a: rel(totals[f"f_{a}"] + totals[f"m_{a}"], totals[f"t_{a}"])
               for a in AGES if all(f"{s}_{a}" in totals for s in "fmt")}
    if per_age:
        out["f_plus_m_vs_t_by_age"] = per_age
    if all(f"t_{a}" in totals for a in AGES) and "pop" in totals:
        out["sum_of_t_ages_vs_pop"] = rel(sum(totals[f"t_{a}"] for a in AGES), totals["pop"])
    for sex, key in (("f", "T_F"), ("m", "T_M")):
        if all(f"{sex}_{a}" in totals for a in AGES) and key in totals:
            out[f"sum_of_{sex}_ages_vs_{key}"] = rel(sum(totals[f"{sex}_{a}"] for a in AGES), totals[key])
    if {"T_F", "T_M", "pop"} <= totals.keys():
        out["T_F_plus_T_M_vs_pop"] = rel(totals["T_F"] + totals["T_M"], totals["pop"])
    return out


# -------------------------------------------------------------------- main --

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Population and disability per H3 hexagon and admin unit, with exactextract.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--rasters", type=Path, required=True, help="Folder with the age-sex rasters")
    p.add_argument("--res", type=int, nargs="+", default=[9], help="H3 resolution(s)")
    p.add_argument("--mukims", type=Path,
                   help="Folder with the UNDP mukim layer; becomes the study area and adds admin 1-3 outputs")
    p.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES,
                   help="Boundary file when --mukims is not used; defaults to the geoBoundaries copy cached by vida_buildings.py")
    p.add_argument("--country", default="MYS", help="ISO3 code, used if the boundaries need downloading")
    p.add_argument("--iso", nargs="+", metavar="CODE", help="Area codes to keep (default: MY-10 MY-14 MY-16)")
    p.add_argument("--all", action="store_true", help="Use every feature in --boundaries")
    p.add_argument("--code-field", default="shapeISO", help="Code field in --boundaries")
    p.add_argument("--name-field", default="shapeName", help="Name field in --boundaries")
    p.add_argument("--allow-missing", action="store_true",
                   help="Run even if some age-sex rasters are missing; groups that need them are skipped")
    p.add_argument("--child-groups", nargs="+", default=DEFAULT_CHILD_GROUPS, metavar="LABEL=AGES",
                   help="Child groups in completed years, both ends included, ages 0 to 19")
    p.add_argument("--single-age-csv", type=Path,
                   help="CSV with columns age, female, male (ages 0 to 19) to use instead of the built-in "
                        "UN WPP 2024 Malaysia 2026 figures")
    p.add_argument("--disability-calibration", choices=("state", "national"), default="state",
                   help="state scales adults to each territory's NHMS 2023 rate; national uses the national "
                        "age-sex rates everywhere")
    p.add_argument("--child-disability-rate", type=float, default=CHILD_DISABILITY_RATE,
                   help="Percent of children aged 2 to 17 with functional difficulty")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output folder")
    return p.parse_args(argv)


def add_rates(df: pd.DataFrame, year: str):
    """Densities per km2 of land and the disability share of the population."""
    made = []
    if "land_km2" in df.columns:
        land = df["land_km2"].astype(float)
        for label, col in (("pop", f"cgt_pop_{year}"), ("0_4", f"cgt_0_4_t_{year}"),
                           ("60P", f"cgt_60P_t_{year}"), ("dis", f"cgt_dis_t_{year}")):
            if col in df.columns:
                df[f"dens_{label}_{year}"] = df[col] / land.where(land > 0)
                made.append(f"dens_{label}_{year}")
    if f"cgt_dis_t_{year}" in df.columns and f"cgt_pop_{year}" in df.columns:
        pop = df[f"cgt_pop_{year}"].astype(float)
        df[f"dis_share_{year}"] = df[f"cgt_dis_t_{year}"] / pop.where(pop > 0)
        made.append(f"dis_share_{year}")
    return made


def finish(df: pd.DataFrame, year, shares, child_groups, rates, child_rate, factors):
    """All derived columns for one table; returns the derived column names in output order."""
    groups, skipped = add_groups(df, year)
    child_cols, skipped_child = add_child_groups(df, year, shares, child_groups)
    factor = df["state_code"].map(lambda c: factors.get(c, {}).get("factor", 1.0)).astype(float).to_numpy()
    dis_cols, skipped_dis = add_disability(df, year, rates, shares, child_rate, factor)
    return groups + child_cols + dis_cols + add_rates(df, year), skipped + skipped_child + skipped_dis


def main(argv=None) -> int:
    args = parse_args(argv)
    args.country = args.country.upper()
    started = time.time()

    # Rasters
    found, year, skipped_files = find_rasters(args.rasters)
    missing = [k for k in EXPECTED if k not in found]
    log(f"Rasters: {len(found)} for {year} in {args.rasters}")
    if skipped_files:
        log(f"  ignored (name not recognised): {', '.join(skipped_files)}")
    if missing:
        log(f"  missing: {', '.join(missing)}")
        if not args.allow_missing:
            sys.exit("Stopping, so no group is built from an incomplete set. "
                     "Add the missing rasters, or rerun with --allow-missing.")
    grid, raster_totals, raster_notes = inspect_rasters(found)
    for w in raster_notes:
        log(f"  warning: {w}")
    log(f"  grid {grid['width']} x {grid['height']}, {grid['crs']}, NoData {grid['nodata']}")
    child_groups = parse_child_groups(args.child_groups)
    single_age, single_age_source = load_single_age(args.single_age_csv)
    shares = single_year_shares(single_age)
    if args.single_age_csv is None and year != UN_YEAR:
        log(f"  warning: the built-in UN single-age figures are for {UN_YEAR} and the rasters for {year}; "
            "pass --single-age-csv to match the year")
    rates = fit_adult_rates(NHMS_AGE_RATES, NHMS_SEX_RATES, NATIONAL_ADULTS_2023)
    child_rate = args.child_disability_rate / 100

    # Study area
    mukim_info, adm4 = None, None
    if args.mukims:
        parts, adm4_frames, mukim_info = load_mukims(args.mukims)
        source = f"UNDP mukim layer ({args.mukims.resolve()})"
        log(f"Mukim layer: {mukim_info['units']} units from {len(mukim_info['files'])} files; repaired "
            f"{mukim_info['invalid_repaired']} invalid, removed {mukim_info['overlap_removed_km2']} km2 of overlap, "
            f"filled {mukim_info['gaps_filled']} sliver gaps ({mukim_info['gap_area_filled_km2']} km2)")
        if adm4_frames:
            adm4, mukim_info["admin4"] = load_adm4(adm4_frames, parts)
    else:
        parts, source = load_geoboundaries(args)
    aoi = dissolve(parts)
    laea = equal_area_crs(aoi)
    area_km2 = gpd.GeoSeries([aoi], crs="EPSG:4326").to_crs(laea).area.iloc[0] / 1e6
    parts["land_km2"] = parts.to_crs(laea).area.to_numpy() / 1e6
    states_in = sorted({c for c in parts["state_code"] if c})
    log(f"Study area: {area_km2:,.0f} km2, {len(parts)} units, states {', '.join(states_in) or 'unknown'}")
    raster_crs = grid["crs"] or "EPSG:4326"
    in_raster_crs = (lambda g: g) if raster_crs in ("EPSG:4326", "OGC:CRS84") else (lambda g: g.to_crs(raster_crs))

    # Unit and study-area totals
    log(f"exactextract: {len(found)} rasters over {len(parts)} units and the study area")
    aoi_gdf = gpd.GeoDataFrame({"adm_code": ["study_area"]}, geometry=[aoi], crs="EPSG:4326")
    aoi_totals_df = zonal_sums(found, in_raster_crs(aoi_gdf), ["adm_code"])
    aoi_totals = {k: float(aoi_totals_df[f"wp_{k}"].iloc[0]) for k in found}
    unit_df = zonal_sums(found, in_raster_crs(parts[["adm_code", "geometry"]]), ["adm_code"])
    attrs = parts.drop(columns="geometry")
    unit_df = attrs.merge(unit_df, on="adm_code", how="left")
    wp_cols = [f"wp_{k}" for k in EXPECTED if k in found]
    unit_df[wp_cols] = unit_df[wp_cols].fillna(0.0)

    state_df = (unit_df.groupby("state_code", dropna=False)[wp_cols + ["land_km2"]].sum().reset_index()
                if args.mukims else unit_df[["state_code", "state_name"] + wp_cols + ["land_km2"]].copy())
    if args.mukims:
        state_df.insert(1, "state_name", state_df["state_code"].map(
            lambda c: STATE_NAMES.get(str(c)[-2:], c) if c else None))
    factors = calibration_factors(state_df, rates, shares, args.disability_calibration) \
        if disability_inputs_present(state_df) else {}
    derived, skipped_groups = finish(state_df, year, shares, child_groups, rates, child_rate, factors)
    finish(unit_df, year, shares, child_groups, rates, child_rate, factors)
    adm4_df = None
    if adm4 is not None:
        adm4_df = admin4_table(adm4, unit_df, found, wp_cols, laea, in_raster_crs, mukim_info["admin4"])
        finish(adm4_df, year, shares, child_groups, rates, child_rate, factors)
    if skipped_groups:
        log(f"  groups skipped for missing rasters: {', '.join(skipped_groups)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    if args.mukims:
        district_df = unit_df.groupby(["state_code", "state_name", "district_code", "district_name"])[
            wp_cols + ["land_km2"]].sum().reset_index()
        finish(district_df, year, shares, child_groups, rates, child_rate, factors)
        gpkg = args.out_dir / f"admin_population_{year}.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        layers = {
            "adm1_state": (state_df, "state_code", ["state_code", "state_name"]),
            "adm2_district": (district_df, "district_code",
                              ["state_code", "state_name", "district_code", "district_name"]),
            "adm3_mukim": (unit_df, "adm_code", ["adm_code", "adm_name", "category", "district_code",
                                                 "district_name", "state_code", "state_name"]),
        }
        for layer, (table, key, ids) in layers.items():
            out = table[ids + ["land_km2"] + derived + wp_cols]
            shapes = {k: safe_union(g.geometry.to_numpy(), SNAP_GRID_DEG) for k, g in parts.groupby(key)}
            geometry = [shapes.get(k) for k in out[key]]
            gpd.GeoDataFrame(out, geometry=geometry, crs="EPSG:4326").to_file(gpkg, layer=layer, driver="GPKG")
            csv = args.out_dir / f"{layer}_{year}.csv"
            out.round(3).to_csv(csv, index=False)
            outputs.append(csv)
        if adm4_df is not None:
            ids = ["adm4_code", "adm4_name", "adm_code", "adm_name", "district_code", "district_name",
                   "state_code", "state_name"]
            out = adm4_df[ids + ["land_km2"] + derived + wp_cols]
            geometry = adm4.set_index("adm4_code").geometry.reindex(out["adm4_code"]).to_numpy()
            gpd.GeoDataFrame(out, geometry=geometry, crs="EPSG:4326").to_file(gpkg, layer="adm4", driver="GPKG")
            csv = args.out_dir / f"adm4_{year}.csv"
            out.round(3).to_csv(csv, index=False)
            outputs.append(csv)
        outputs.insert(0, gpkg)
    else:
        csv = args.out_dir / f"population_by_area_{year}.csv"
        counts = [c for c in derived if c.startswith("cgt_")]
        total = state_df[wp_cols + ["land_km2"] + counts].sum().to_frame().T
        total.insert(0, "state_code", "")
        total.insert(1, "state_name", "Study area")
        table = pd.concat([state_df, total], ignore_index=True)
        add_rates(table, year)
        table[["state_code", "state_name", "land_km2"] + derived + wp_cols].round(3).to_csv(csv, index=False)
        outputs.append(csv)

    checks_common = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": f"hex_population.py {SCRIPT_VERSION}",
        "rasters": {"folder": str(args.rasters.resolve()), "year": year, "count": len(found),
                    "missing": missing, "ignored": skipped_files, "grid": grid, "warnings": raster_notes},
        "study_area": {"source": source, "area_km2": round(area_km2, 1), "units": len(parts),
                       "mukim_layer": mukim_info, "sliver_holes_filled_below_m2": MIN_HOLE_M2},
        "outside_study_area": {k: raster_totals[k] - aoi_totals[k] for k in found},
        "units_vs_study_area": {k: float(unit_df[f"wp_{k}"].sum()) - aoi_totals[k] for k in found},
        "consistency_inside_study_area": consistency(aoi_totals),
        "groups_skipped": skipped_groups,
        "child_groups": {
            "ages_inclusive": {label: [first, last] for label, (first, last) in child_groups.items()},
            "single_year_source": single_age_source,
            "shares_within_worldpop_band": shares,
        },
        "disability": {
            "adult_source": NHMS_SOURCE, "adult_definition": NHMS_DEFINITION,
            "age_rates_pct": NHMS_AGE_RATES, "sex_rates_pct": NHMS_SEX_RATES,
            "national_structure_for_raking": "WPP 2024, Malaysia 2023, thousands (female, male)",
            "fitted_rates_pct": {f"{g}_{s}": 100 * v for (g, s), v in rates.items()},
            "calibration": args.disability_calibration, "states": factors,
            "child_source": CHILD_SOURCE, "child_rate_pct": args.child_disability_rate,
            "not_covered": "children under 2",
            "registered_benchmark": {"source": JKM_SOURCE, "counts": JKM_REGISTERED},
        },
        "software": {"exactextract": __import__("exactextract").__version__, "h3": h3.__version__,
                     "shapely": shapely.__version__, "rasterio": rasterio.__version__},
    }

    label_cols = [c for c in ("adm_code", "adm_name", "district_code", "district_name", "state_code", "state_name")
                  if c in parts.columns]
    counts = [c for c in derived if c.startswith("cgt_")] + wp_cols
    for res in args.res:
        log(f"H3 res {res}: building cells")
        cells, hexes, duplicates = build_cells(aoi, res)
        zones, edge, keep = clip_cells(hexes, aoi)
        zones, cells, hexes, edge = zones[keep], cells[keep], hexes[keep], edge[keep]
        land_km2 = gpd.GeoSeries(zones, crs="EPSG:4326").to_crs(laea).area.to_numpy() / 1e6
        tag, tagged_by_nearest = tag_areas(zones, parts)
        piece_zone, piece_state, piece_geom, split = split_by_state(zones, parts)
        log(f"  {len(cells):,} cells, {int(edge.sum()):,} on the boundary, {split:,} split along state lines, "
            f"{duplicates} duplicates from h3 removed")

        zone_gdf = gpd.GeoDataFrame({"zone": np.arange(len(cells))}, geometry=zones, crs="EPSG:4326")
        log(f"  exactextract: {len(found)} rasters over {len(zone_gdf):,} cells")
        values = zonal_sums(found, in_raster_crs(zone_gdf), ["zone"]).set_index("zone")[wp_cols]
        values = values.reindex(np.arange(len(cells)), fill_value=0.0)

        # Cells cut by a state line: the cell totals stay as extracted, and the parts
        # only set how each band divides between the states. exactextract can
        # misplace whole pixels on very thin boundary wedges, so part sums are not
        # used as totals.
        pieces = values.iloc[piece_zone].reset_index(drop=True)
        n_parts = np.bincount(piece_zone, minlength=len(cells))
        multi = np.flatnonzero(n_parts[piece_zone] > 1)
        parts_off = 0
        if multi.size:
            cut_gdf = gpd.GeoDataFrame({"piece": multi}, geometry=piece_geom[multi], crs="EPSG:4326")
            cut = zonal_sums(found, in_raster_crs(cut_gdf), ["piece"]).set_index("piece")[wp_cols].loc[multi]
            cut_zone = piece_zone[multi]
            got = cut.to_numpy()
            total = cut.groupby(cut_zone).transform("sum").to_numpy()
            even = (1.0 / n_parts[cut_zone])[:, None]
            share = np.where(total > 0, got / np.where(total > 0, total, 1.0), even)
            pieces.loc[multi, wp_cols] = pieces.loc[multi, wp_cols].to_numpy() * share
            if "wp_pop" in wp_cols:
                zone_pop = values["wp_pop"].to_numpy()[cut_zone]
                part_pop = total[:, wp_cols.index("wp_pop")]
                parts_off = len(set(cut_zone[np.abs(part_pop - zone_pop) > 1e-3 * np.maximum(zone_pop, 1.0)]))
        pieces["zone"] = piece_zone
        pieces["state_code"] = piece_state
        finish(pieces, year, shares, child_groups, rates, child_rate, factors)
        summed = pieces.groupby("zone")[counts].sum().reindex(np.arange(len(cells)), fill_value=0.0)

        df = parts[label_cols].iloc[tag].reset_index(drop=True)
        df.insert(0, "h3", cells)
        df.insert(1, "h3_res", res)
        df["edge"] = edge
        df["land_km2"] = land_km2
        df = pd.concat([df, summed.reset_index(drop=True)], axis=1)
        add_rates(df, year)
        df = df[["h3", "h3_res"] + label_cols + ["edge", "land_km2"] + derived + wp_cols]
        if adm4 is not None:
            unit = tag_children(zones, adm4, df["adm_code"].to_numpy())
            for i, col in enumerate(("adm4_code", "adm4_name")):
                df.insert(2 + i, col, np.where(unit >= 0, adm4[col].to_numpy()[np.maximum(unit, 0)], None))

        out = args.out_dir / f"hex_r{res}_population_{year}.fgb"
        gpd.GeoDataFrame(df, geometry=hexes, crs="EPSG:4326").to_file(out, driver="FlatGeobuf")
        outputs.append(out)

        recon = {k: (float(df[f"wp_{k}"].sum()) - aoi_totals[k]) / aoi_totals[k] if aoi_totals[k] else 0.0
                 for k in found}
        dis_col = f"cgt_dis_t_{year}"
        dis_check = {}
        if dis_col in pieces.columns:
            by_state = pieces.groupby("state_code")[dis_col].sum()
            for code, admin in zip(state_df["state_code"], state_df[dis_col]):
                hexed = float(by_state.get(code, 0.0))
                dis_check[str(code)] = {"hexagons": hexed, "admin": float(admin),
                                        "relative_difference": (hexed - admin) / admin if admin else None}
        checks = dict(checks_common)
        checks["hexagons"] = {"res": res, "cells": len(cells), "edge_cells": int(edge.sum()),
                              "cells_split_along_state_lines": split,
                              "split_cells_whose_parts_did_not_add_up": parts_off,
                              "duplicates_removed": duplicates,
                              "cells_labelled_by_nearest_unit": tagged_by_nearest, "file": out.name,
                              "max_relative_difference_vs_study_area": max(abs(v) for v in recon.values()),
                              "relative_difference_by_raster": recon,
                              "disability_by_state_hexagons_vs_admin": dis_check}
        check_path = args.out_dir / f"hex_r{res}_population_{year}_checks.json"
        check_path.write_text(json.dumps(checks, indent=2, default=str))
        outputs.append(check_path)
        log(f"  hexagons vs study area: max difference {checks['hexagons']['max_relative_difference_vs_study_area']:.1e}")

    # Summary
    log("")
    if "pop" in found:
        outside = raster_totals["pop"] - aoi_totals["pop"]
        share = outside / raster_totals["pop"] if raster_totals["pop"] else 0
        log(f"Outside the study area (not counted): {outside:,.0f} people in the pop raster ({share:.2%})")
        if args.mukims:
            gap = aoi_totals["pop"] - float(unit_df["wp_pop"].sum())
            log(f"Study area minus the sum of mukims (sliver gaps): {gap:,.0f} people")
    cons = checks_common["consistency_inside_study_area"]
    by_age = {a: v for a, v in cons.get("f_plus_m_vs_t_by_age", {}).items() if v is not None}
    if by_age:
        age = max(by_age, key=lambda a: abs(by_age[a]))
        log(f"Raster agreement: f + m vs t, largest gap {by_age[age]:+.2%} (age {age})")
    labels = {"sum_of_t_ages_vs_pop": "t ages vs pop", "sum_of_f_ages_vs_T_F": "f ages vs T_F",
              "sum_of_m_ages_vs_T_M": "m ages vs T_M", "T_F_plus_T_M_vs_pop": "T_F + T_M vs pop"}
    others = [f"{labels[k]} {v:+.2%}" for k, v in cons.items() if k in labels and v is not None]
    if others:
        log("                   " + ", ".join(others))

    def table_line(cells):
        return f"{cells[0]:<20}" + "".join(f"{c:>13}" for c in cells[1:])

    rows = state_df.to_dict(orient="records")
    total_row = {"state_name": "Study area", **state_df[[c for c in state_df.columns
                                                        if c.startswith(("cgt_", "wp_"))]].sum().to_dict()}
    show = [(f"cgt_pop_{year}", "total"), (f"cgt_0_4_t_{year}", "0 to 4"), (f"cgt_60P_t_{year}", "60+"),
            (f"cgt_dis_t_{year}", "disability")]
    show = [(c, h) for c, h in show if c in state_df.columns]
    if show:
        log("")
        log(table_line([""] + [h for _, h in show]))
        for row in rows + [total_row]:
            log(table_line([str(row.get("state_name") or row["state_code"])[:20]]
                           + [f"{row[c]:,.0f}" for c, _ in show]))
    child_show = [(label, span) for label, span in child_groups.items() if f"cgt_{label}_t_{year}" in state_df.columns]
    if child_show:
        log("")
        log(table_line(["children"] + [f"age {a}" if a == b else f"ages {a}-{b}" for _, (a, b) in child_show]))
        for row in rows + [total_row]:
            log(table_line([str(row.get("state_name") or row["state_code"])[:20]]
                           + [f"{row[f'cgt_{label}_t_{year}']:,.0f}" for label, _ in child_show]))
    if factors and f"cgt_dis_t_{year}" in state_df.columns:
        log("")
        log(f"Disability ({args.disability_calibration} calibration); rates are for adults 18+")
        log(table_line(["", "age-sex rate", "NHMS 2023", "factor", "adults 18+", "ages 2-17", "JKM reg."]))
        for row in rows:
            f = factors.get(row["state_code"], {})
            reg = JKM_REGISTERED.get(row["state_code"])
            log(table_line([str(row.get("state_name") or row["state_code"])[:20],
                            f"{f['rate_before_pct']:.1f}%" if f.get("rate_before_pct") is not None else "-",
                            f"{f['nhms_rate_pct']:.1f}%" if f.get("nhms_rate_pct") is not None else "-",
                            f"{f.get('factor', 1.0):.2f}", f"{row[f'cgt_dis_adult_t_{year}']:,.0f}",
                            f"{row[f'cgt_dis_child_t_{year}']:,.0f}", f"{reg:,}" if reg else "-"]))
        log("  JKM counts are registrations of all ages (31 May 2025); the KL figure may include Putrajaya.")
    log("")
    for path in outputs:
        log(f"Wrote {path}")
    log(f"Done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())