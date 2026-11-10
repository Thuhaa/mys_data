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
Split the building centroids into one FlatGeobuf per district so every file
stays under GitHub's 100 MB limit and can be served from raw.githubusercontent.com.

Points are assigned to the district polygon that contains them; points outside
every district (coast, boundary slivers) go to "unassigned.fgb". An index.json
lists every tile with its feature count, size and bounds.

Usage
  uv run src/split_centroids.py
  uv run src/split_centroids.py --base-url https://raw.githubusercontent.com/<user>/<repo>/main/outputs/building_centroids
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd

DEFAULT_INPUT = Path("outputs/vida_buildings_mys_selangor_kuala-lumpur_putrajaya_centroids.fgb")
DEFAULT_DISTRICTS = Path("outputs/adm2_district_2026.fgb")
DEFAULT_OUT = Path("outputs/building_centroids")
GITHUB_LIMIT = 100 * 1024 * 1024


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def slug(text: str) -> str:
    text = re.sub(r"^(DAERAH|W\.P\.)\s+", "", text.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--districts", type=Path, default=DEFAULT_DISTRICTS)
    p.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--base-url", default="", help="prefix for the url field in index.json")
    args = p.parse_args()

    log(f"Reading {args.input}")
    pts = gpd.read_file(args.input, engine="pyogrio", use_arrow=True)
    districts = gpd.read_file(args.districts, columns=["district_code", "district_name"]).to_crs(pts.crs)

    log("Assigning points to districts")
    joined = gpd.sjoin(pts[["geometry"]], districts, predicate="within", how="left")
    joined = joined[~joined.index.duplicated()]  # a point on a shared edge keeps its first match
    pts["_code"] = joined["district_code"].reindex(pts.index)
    pts["_name"] = joined["district_name"].reindex(pts.index)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("*.fgb"):
        old.unlink()

    tiles, too_big = [], []
    for (code, name), part in pts.groupby(["_code", "_name"], dropna=False, sort=True):
        assigned = isinstance(code, str)
        fname = f"{code}_{slug(name)}.fgb" if assigned else "unassigned.fgb"
        path = args.out_dir / fname
        part = part.drop(columns=["_code", "_name"])
        part.to_file(path, engine="pyogrio", use_arrow=True, layer=path.stem)
        size = path.stat().st_size
        if size >= GITHUB_LIMIT:
            too_big.append(fname)
        tiles.append({
            "file": fname,
            "url": f"{args.base_url.rstrip('/')}/{fname}" if args.base_url else fname,
            "district_code": code if assigned else None,
            "district_name": name if assigned else None,
            "features": len(part),
            "bytes": size,
            "bbox": [round(v, 6) for v in part.total_bounds],
        })
        log(f"  {fname:40s} {len(part):>9,} pts {size / 1e6:7.1f} MB")

    index = {"source": args.input.name, "crs": str(pts.crs), "features": len(pts), "tiles": tiles}
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    log(f"Wrote {len(tiles)} tiles + index.json to {args.out_dir} "
        f"({sum(t['features'] for t in tiles):,} of {len(pts):,} points)")
    if too_big:
        sys.exit(f"Over GitHub's 100 MB limit: {', '.join(too_big)}")


if __name__ == "__main__":
    main()
