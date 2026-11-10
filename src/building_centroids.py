#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "geopandas>=1.0",
#     "pyogrio>=0.10",
#     "pyarrow",
#     "shapely>=2.0",
# ]
# ///
"""
Convert building footprints to one point per building, keeping every attribute.

The centroid is computed directly in the source CRS (EPSG:4326). Over the
extent of a single building the lon/lat -> metres mapping is effectively
affine, and centroids are preserved under affine maps, so projecting first
would move the points by far less than a millimetre.

A centroid can fall outside a concave footprint (L- or U-shaped buildings).
--inside uses shapely's point_on_surface instead, which is always inside the
polygon but is not the geometric centre.

Usage
  uv run src/building_centroids.py
  uv run src/building_centroids.py --inside -o outputs/buildings_points.fgb
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import pyogrio
import shapely

DEFAULT_INPUT = Path("data/vida_buildings_mys_selangor_kuala-lumpur_putrajaya.fgb")


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT, help=f"footprints (default: {DEFAULT_INPUT})")
    p.add_argument("-o", "--output", type=Path,
                   help="output file; format from the extension (default: outputs/<input>_centroids.fgb)")
    p.add_argument("--inside", action="store_true",
                   help="use point_on_surface so every point lies inside its footprint")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")
    out = args.output or Path("outputs") / f"{args.input.stem}_{'points' if args.inside else 'centroids'}.fgb"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    info = pyogrio.read_info(args.input)
    log(f"Reading {info['features']:,} features from {args.input} ({info['crs']})")
    gdf = gpd.read_file(args.input, engine="pyogrio", use_arrow=True)

    geom = gdf.geometry.values
    empty = shapely.is_empty(geom) | shapely.is_missing(geom)
    if empty.any():
        log(f"Dropping {int(empty.sum()):,} features with empty or missing geometry")
        gdf = gdf.loc[~empty]
        geom = gdf.geometry.values

    log("Computing " + ("points on surface" if args.inside else "centroids"))
    points = shapely.point_on_surface(geom) if args.inside else shapely.centroid(geom)
    gdf = gdf.set_geometry(gpd.GeoSeries(points, index=gdf.index, crs=gdf.crs))

    log(f"Writing {len(gdf):,} points to {out}")
    gdf.to_file(out, engine="pyogrio", use_arrow=True, layer=out.stem)
    log(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
