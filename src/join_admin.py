#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "geopandas>=1.0",
#     "pyogrio>=0.10",
#     "pandas",
#     "shapely>=2.0",
# ]
# ///
"""
Join the admin tables written by hex_population.py to your own boundary files,
one FlatGeobuf per level.

  level          your fields                       join
  adm1_state     KOD_NEGERI                         "MY-" + KOD_NEGERI = state_code
  adm2_district  KOD_NEGERI, KOD_DAERAH             KOD_NEGERI + KOD_DAERAH = district_code
  adm3_mukim     every field of your mukim files    MUKIMCODE = adm_code
  adm4           every field of your admin 4 file   parent MUKIMCODE + KOD_SEKSYE = adm4_code
                 (the parent is the mukim with the same KOD_NEGERI, KOD_DAERAH and KOD_MUKIM)

Geometry comes from the GeoPackage hex_population.py writes next to the CSVs:
the cleaned units the figures were computed on, which tile the study area
without overlaps, gaps or seams. --original-geometry keeps your own polygons
for the mukims and admin 4 units instead (repaired where invalid).

Codes are read as text, so 1001 and 017 keep their form. Any feature or
table row without a match is reported.

Usage (uv installs the dependencies above on the first run)
  uv run join_admin.py --tables outputs --boundaries data/boundaries
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely

# Joining wide tables only triggers pandas' fragmentation notice
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SCRIPT_VERSION = "1.1"
TEXT_COLUMNS = ["state_code", "state_name", "district_code", "district_name", "adm_code", "adm_name",
                "category", "adm4_code", "adm4_name"]
SUFFIXES = (".shp", ".gpkg", ".geojson", ".fgb")


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def code(g: pd.DataFrame, *fields_and_widths) -> pd.Series:
    """Text code from fields, each zero-padded to its width."""
    out = None
    for field, width in fields_and_widths:
        part = g[field].astype(str).str.strip().str.zfill(width)
        out = part if out is None else out + part
    return out


def polygonal(geom):
    """Polygonal part of a geometry (drops stray lines and points), or None."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    polys = [q for p in shapely.get_parts(geom) if p.geom_type in ("Polygon", "MultiPolygon")
             for q in shapely.get_parts(p)]
    if not polys:
        return None
    return shapely.MultiPolygon(polys) if len(polys) > 1 else polys[0]


def repair(g: gpd.GeoDataFrame, name: str, id_field: str) -> gpd.GeoDataFrame:
    """Make invalid polygons valid (same shape and area) and say which were repaired."""
    bad = ~g.geometry.is_valid
    if bad.any():
        ids = g.loc[bad, id_field].astype(str) if id_field in g.columns else g.index[bad].astype(str)
        log(f"  {name}: repaired {int(bad.sum())} invalid polygon(s): {', '.join(ids[:10])}")
        g = g.copy()
        g.loc[bad, "geometry"] = [polygonal(x) for x in shapely.make_valid(g.geometry[bad].to_numpy())]
    return g


def read_boundaries(folder: Path):
    mukims, units = [], []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES):
        g = gpd.read_file(path)
        g = g.set_crs("EPSG:4326") if g.crs is None else g.to_crs("EPSG:4326")
        if "MUKIMCODE" in g.columns:
            mukims.append(repair(g, path.name, "MUKIMCODE"))
        elif "KOD_SEKSYE" in g.columns:
            units.append(repair(g, path.name, "NAM"))
        else:
            log(f"  skipped {path.name}: it has neither MUKIMCODE nor KOD_SEKSYE")
    if not mukims:
        sys.exit(f"No mukim file (one with a MUKIMCODE field) in {folder}")
    mukims = gpd.GeoDataFrame(pd.concat(mukims, ignore_index=True), crs="EPSG:4326")
    units = gpd.GeoDataFrame(pd.concat(units, ignore_index=True), crs="EPSG:4326") if units else None
    return mukims, units


def with_geometry(features: gpd.GeoDataFrame, key: str, shapes: gpd.GeoSeries, label: str) -> gpd.GeoDataFrame:
    """Swap in the cleaned geometry, matched on the key."""
    geoms = features[key].astype(str).map(shapes)
    missing = features.loc[geoms.isna(), key].astype(str).tolist()
    if missing:
        log(f"  {label}: no cleaned polygon for {', '.join(missing[:10])}; these keep no geometry")
    return features.set_geometry(gpd.GeoSeries(geoms.to_numpy(), index=features.index, crs="EPSG:4326"))


def join(features: gpd.GeoDataFrame, left_key: str, table: pd.DataFrame, right_key: str, label: str):
    """Left join of the table onto the features, with a report of what did not match."""
    features = features.assign(**{left_key: features[left_key].astype(str)})
    left, right = features[left_key], table[right_key].astype(str)
    dupes = right[right.duplicated()].unique()
    if len(dupes):
        sys.exit(f"{label}: the table repeats {right_key} values, e.g. {', '.join(dupes[:5])}")
    merged = features.merge(table.assign(**{right_key: right}), left_on=left_key, right_on=right_key,
                            how="left", indicator=True)
    no_row = left[~left.isin(right)].unique()
    no_feature = right[~right.isin(left)].unique()
    log(f"{label}: {int((merged['_merge'] == 'both').sum())} of {len(features)} features matched "
        f"{len(table)} table rows")
    if len(no_row):
        log(f"  features with no table row: {', '.join(no_row[:10])}" + (" ..." if len(no_row) > 10 else ""))
    if len(no_feature):
        log(f"  table rows with no feature: {', '.join(no_feature[:10])}" + (" ..." if len(no_feature) > 10 else ""))
    return gpd.GeoDataFrame(merged.drop(columns="_merge"), geometry="geometry", crs=features.crs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Join the hex_population.py admin tables to your boundary files.")
    p.add_argument("--tables", type=Path, required=True,
                   help="Folder with adm1_state_{year}.csv ... adm4_{year}.csv and admin_population_{year}.gpkg")
    p.add_argument("--boundaries", type=Path, required=True,
                   help="Folder with your mukim files and any admin 4 file (the one passed to --mukims)")
    p.add_argument("--original-geometry", action="store_true",
                   help="Keep your own polygons for the mukims and admin 4 units")
    p.add_argument("--year", help="Year in the table names (default: found from adm3_mukim_*.csv)")
    p.add_argument("--out-dir", type=Path, help="Output folder (default: --tables)")
    args = p.parse_args(argv)
    out_dir = args.out_dir or args.tables
    out_dir.mkdir(parents=True, exist_ok=True)

    year = args.year
    if year is None:
        years = sorted({m[1] for f in args.tables.glob("adm3_mukim_*.csv")
                        if (m := re.fullmatch(r"adm3_mukim_(\d{4})\.csv", f.name))})
        if len(years) != 1:
            sys.exit(f"Found tables for years {years or 'none'} in {args.tables}; pass --year")
        year = years[0]
    gpkg = args.tables / f"admin_population_{year}.gpkg"
    if not gpkg.exists():
        sys.exit(f"{gpkg} not found; hex_population.py writes it next to the CSVs")
    layers = {name for name, _ in pyogrio.list_layers(gpkg)}

    def cleaned(layer, key):
        g = gpd.read_file(gpkg, layer=layer, columns=[key])
        return gpd.GeoSeries(g.geometry.to_numpy(), index=g[key].astype(str), crs="EPSG:4326")

    def table(name):
        path = args.tables / f"{name}_{year}.csv"
        return pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}) if path.exists() else None

    mukims, units = read_boundaries(args.boundaries)
    levels = []

    states = cleaned("adm1_state", "state_code")
    states = gpd.GeoDataFrame({"KOD_NEGERI": states.index.str[-2:], "state_code": states.index},
                              geometry=states.to_numpy(), crs="EPSG:4326")
    levels.append(("adm1_state", states, "state_code", "state_code"))
    districts = cleaned("adm2_district", "district_code")
    districts = gpd.GeoDataFrame({"KOD_NEGERI": districts.index.str[:2], "KOD_DAERAH": districts.index.str[2:],
                                  "district_code": districts.index}, geometry=districts.to_numpy(), crs="EPSG:4326")
    levels.append(("adm2_district", districts, "district_code", "district_code"))

    if not args.original_geometry:
        mukims = with_geometry(mukims, "MUKIMCODE", cleaned("adm3_mukim", "adm_code"), "adm3_mukim")
    levels.append(("adm3_mukim", mukims, "MUKIMCODE", "adm_code"))

    if units is not None:
        parent = dict(zip(code(mukims, ("KOD_NEGERI", 2), ("KOD_DAERAH", 2), ("KOD_MUKIM", 2)),
                          mukims["MUKIMCODE"].astype(str)))
        units = units.copy()
        mukim_code = code(units, ("KOD_NEGERI", 2), ("KOD_DAERAH", 2), ("KOD_MUKIM", 2)).map(parent)
        if mukim_code.isna().any():
            log(f"  {int(mukim_code.isna().sum())} admin 4 features match no mukim on KOD_NEGERI, KOD_DAERAH, KOD_MUKIM")
        units["_key"] = mukim_code + code(units, ("KOD_SEKSYE", 3))
        if not args.original_geometry and "adm4" in layers:
            units = with_geometry(units, "_key", cleaned("adm4", "adm4_code"), "adm4")
        levels.append(("adm4", units, "_key", "adm4_code"))

    for name, features, left_key, right_key in levels:
        data = table(name)
        if data is None:
            log(f"{name}: no {name}_{year}.csv in {args.tables}, skipped")
            continue
        joined = join(features, left_key, data, right_key, name)
        joined = joined.drop(columns=[c for c in ("_key",) if c in joined.columns])
        path = out_dir / f"{name}_{year}.fgb"
        joined.to_file(path, driver="FlatGeobuf")
        log(f"  wrote {path} ({len(joined)} features, {len(joined.columns) - 1} fields)")
    return 0


if __name__ == "__main__":
    sys.exit(main())