#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "geopandas>=1.0",
#     "pyogrio>=0.10",
#     "pyarrow>=16",
#     "shapely>=2.0",
#     "numpy",
#     "pandas",
#     "requests>=2.31",
#     "tqdm>=4.66",
#     "h3>=4.1",
# ]
# ///
"""
GlobalBuildingAtlas (GBA) buildings with heights for a study area, as FlatGeobuf.

GBA (TUM; Zhu et al. 2025, ESSD 17, 6647-6668) gives each building a height
estimated from PlanetScope imagery, on footprints fused from Google, Microsoft,
OpenStreetMap and TUM's own extraction. This script reads the GeoParquet copy
on Source Cooperative (tge-labs/globalbuildingatlas-lod1, one file per 5 x 5
degree tile) and, over HTTP range requests, fetches only the row groups whose
bounding boxes overlap the study area.

What it does
  1. Builds the study area from your boundary files or folders (--aoi).
  2. Lists the tiles on Source Cooperative and keeps those overlapping it.
  3. Reads each tile's footer, then only the row groups that overlap.
  4. Keeps buildings whose centroid falls inside the study area.
  5. Adds footprint area, an estimated storey count (height divided by
     --storey-height, at least 1), floor area (footprint x storeys) and the
     H3 cell holding the centroid.
  6. Writes an indexed FlatGeobuf and a summary JSON, and prints the height
     distribution and the tallest buildings to check against what you know.

Licence: GBA heights and the non-ODbL polygons are CC BY-NC 4.0; polygons from
OpenStreetMap and Microsoft are ODbL.

Usage (uv installs the dependencies above on the first run)
  uv run gba_buildings.py --aoi data/boundaries
  uv run gba_buildings.py --aoi data/boundaries --download     # whole tiles, resumable
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
import shapely
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

SCRIPT_VERSION = "1.0"
DATA_BASE = "https://data.source.coop"
ACCOUNT = "tge-labs"
PRODUCT = "globalbuildingatlas-lod1"
DATASET_PAGE = f"https://source.coop/{ACCOUNT}/{PRODUCT}"
AWS_LIST = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop"
USER_AGENT = f"cgt-gba-buildings/{SCRIPT_VERSION}"
TILE_RX = re.compile(r"([ew])(\d{3})_([ns])(\d{2})_([ew])(\d{3})_([ns])(\d{2})")
VECTOR_SUFFIXES = (".shp", ".gpkg", ".geojson", ".fgb")
MIN_HOLE_M2 = 1_000_000  # sliver holes in the dissolved study area get filled
WANTED = ("id", "region", "source", "height", "var")


def log(msg: str = "") -> None:
    tqdm.write(msg, file=sys.stderr)


def human(n: float | None) -> str:
    if n is None:
        return "size unknown"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET", "HEAD"}))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# --------------------------------------------------------------------- AOI --

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


def load_aoi(paths):
    """Union of every polygon in the given files and folders, sliver holes filled, in EPSG:4326."""
    files = []
    for path in paths:
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.suffix.lower() in VECTOR_SUFFIXES)
        elif path.is_file():
            files.append(path)
        else:
            sys.exit(f"Not found: {path}")
    if not files:
        sys.exit("No boundary files found for --aoi")
    frames = []
    for path in files:
        g = gpd.read_file(path)
        frames.append(g.set_crs("EPSG:4326") if g.crs is None else g.to_crs("EPSG:4326"))
    geoms = [polygonal(x) for x in shapely.make_valid(pd.concat(frames).geometry.to_numpy())]
    geoms = [g for g in geoms if g is not None]
    try:
        union = shapely.union_all(geoms)
    except shapely.errors.GEOSException:
        union = shapely.union_all(geoms, grid_size=1e-9)
    union = polygonal(union)
    if union is None:
        sys.exit("The --aoi files contain no polygons")
    utm = gpd.GeoSeries([union], crs="EPSG:4326").estimate_utm_crs()
    polygons = []
    for poly in shapely.get_parts(union):
        holes = list(poly.interiors)
        if holes:
            sizes = gpd.GeoSeries([shapely.Polygon(r) for r in holes], crs="EPSG:4326").to_crs(utm).area.to_numpy()
            holes = [r for r, s in zip(holes, sizes) if s >= MIN_HOLE_M2]
        polygons.append(shapely.Polygon(poly.exterior, holes))
    aoi = shapely.MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    return aoi, [p.name for p in files]


def equal_area_crs(geom) -> str:
    c = geom.centroid
    return f"+proj=laea +lat_0={c.y:.4f} +lon_0={c.x:.4f} +datum=WGS84 +units=m +no_defs"


# ------------------------------------------------------------ remote files --

def tile_bounds(name: str):
    """(lon_min, lat_min, lon_max, lat_max) from a tile name such as e100_n05_e105_n00."""
    m = TILE_RX.search(name)
    if not m:
        return None
    ew1, x1, ns1, y1, ew2, x2, ns2, y2 = m.groups()
    xs = [int(x1) * (1 if ew1 == "e" else -1), int(x2) * (1 if ew2 == "e" else -1)]
    ys = [int(y1) * (1 if ns1 == "n" else -1), int(y2) * (1 if ns2 == "n" else -1)]
    return min(xs), min(ys), max(xs), max(ys)


def expected_tiles(bbox):
    """Tile names the 5 x 5 degree grid gives for a bounding box, for when listing fails."""
    def lon(v):
        return f"{'e' if v >= 0 else 'w'}{abs(v):03d}"

    def lat(v):
        return f"{'n' if v >= 0 else 's'}{abs(v):02d}"

    names = []
    for x in range(math.floor(bbox[0] / 5) * 5, math.ceil(bbox[2] / 5) * 5, 5):
        for y in range(math.floor(bbox[1] / 5) * 5, math.ceil(bbox[3] / 5) * 5, 5):
            names.append(f"{lon(x)}_{lat(y + 5)}_{lon(x + 5)}_{lat(y)}.parquet")
    return names


def s3_list(session, url: str, prefix: str):
    """S3 ListObjectsV2 against url, returning [(file name, size)] for .parquet keys."""
    params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    found = []
    while True:
        r = session.get(url, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for el in root.iter():
            if isinstance(el.tag, str) and "}" in el.tag:
                el.tag = el.tag.split("}", 1)[1]
        if root.tag != "ListBucketResult":
            raise ValueError(f"unexpected response <{root.tag}>")
        for item in root.findall("Contents"):
            key = item.findtext("Key") or ""
            if key.endswith(".parquet"):
                found.append((key, int(item.findtext("Size") or 0) or None))
        token = root.findtext("NextContinuationToken")
        if (root.findtext("IsTruncated") or "").lower() == "true" and token:
            params["continuation-token"] = token
        else:
            return found


def list_tiles(session, base: str):
    attempts = [(f"{base}/{ACCOUNT}/{PRODUCT}", ""), (f"{base}/{ACCOUNT}", f"{PRODUCT}/")]
    if base == DATA_BASE:
        attempts.append((AWS_LIST, f"{ACCOUNT}/{PRODUCT}/"))
    problems = []
    for url, prefix in attempts:
        try:
            found = s3_list(session, url, prefix)
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            problems.append(f"{url}: {exc}")
            continue
        if found:
            # keys may be relative to the product or carry the account and product in front
            strip = (f"{ACCOUNT}/{PRODUCT}/", f"{PRODUCT}/")
            return [(next((k[len(s):] for s in strip if k.startswith(s)), k), size) for k, size in found]
        problems.append(f"{url}: no .parquet files under '{prefix}'")
    raise RuntimeError("\n  ".join(problems))


def head_info(session, url: str) -> dict:
    try:
        r = session.head(url, allow_redirects=True, timeout=60)
        if r.ok:
            size = r.headers.get("Content-Length")
            return {"size": int(size) if size else None, "last_modified": r.headers.get("Last-Modified"),
                    "etag": r.headers.get("ETag")}
    except requests.RequestException:
        pass
    return {}


class RemoteFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests, with a small block cache."""

    def __init__(self, session, url: str, size: int, block: int = 8 << 20, max_blocks: int = 32):
        super().__init__()
        self.session, self.url, self.size, self.block = session, url, size, block
        self.max_blocks, self.pos, self.fetched, self.requests = max_blocks, 0, 0, 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        lo = index * self.block
        hi = min(lo + self.block, self.size) - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=180)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError("the server ignored the Range request; rerun with --download")
        data = r.content
        self.fetched += len(data)
        self.requests += 1
        self.cache[index] = data
        while len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        out = bytearray()
        while self.pos < end:
            index = self.pos // self.block
            data = self._block(index)
            start = self.pos - index * self.block
            take = min(end - self.pos, len(data) - start)
            if take <= 0:
                break
            out += data[start:start + take]
            self.pos += take
        return bytes(out)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def download(session, url: str, dest: Path, size: int | None) -> Path:
    """Resumable download with a byte progress bar. Partial data is kept in <dest>.part."""
    if dest.exists() and (size is None or dest.stat().st_size == size):
        log(f"  already downloaded: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 6):
        done = part.stat().st_size if part.exists() else 0
        if size is not None and done >= size:
            break
        try:
            headers = {"Range": f"bytes={done}-"} if done else {}
            with session.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
                if done and r.status_code == 200:
                    done = 0
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                with open(part, "ab") as fh, tqdm(total=size, initial=done, unit="B", unit_scale=True,
                                                   desc=dest.name, leave=False) as bar:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        bar.update(len(chunk))
            if size is None:
                break
        except requests.RequestException as exc:
            log(f"  connection problem ({exc}); resuming, attempt {attempt}")
            time.sleep(3 * attempt)
    if size is not None and part.stat().st_size != size:
        sys.exit(f"Download of {url} incomplete; rerun to resume")
    part.replace(dest)
    return dest


# ------------------------------------------------------------- reading -----

def geometry_column(schema: pa.Schema) -> str:
    for name in ("geometry", "geom", "wkb_geometry"):
        if name in schema.names:
            return name
    for field in schema:
        if pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
            return field.name
    sys.exit(f"No WKB geometry column in the tile; columns are {schema.names}")


def row_group_boxes(pf: pq.ParquetFile, bbox_col: str | None):
    """Per row group: (xmin, ymin, xmax, ymax) from the bbox column statistics, or None."""
    md = pf.metadata
    leaf = {md.schema.column(j).path: j for j in range(md.num_columns)}
    cols = {k: leaf.get(f"{bbox_col}.{k}") for k in ("xmin", "ymin", "xmax", "ymax")} if bbox_col else {}
    boxes = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)

        def stat(key, low):
            j = cols.get(key)
            s = rg.column(j).statistics if j is not None else None
            if s is None or not s.has_min_max:
                return None
            return float(s.min if low else s.max)

        box = (stat("xmin", True), stat("ymin", True), stat("xmax", False), stat("ymax", False))
        boxes.append(None if None in box else box)
    return boxes


def read_tile(pf: pq.ParquetFile, name: str, aoi, aoi_bbox, stats: dict):
    """Buildings of one tile whose centroid falls in the AOI, as (attributes, geometries)."""
    schema = pf.schema_arrow
    geom_col = geometry_column(schema)
    bbox_col = "bbox" if "bbox" in schema.names and pa.types.is_struct(schema.field("bbox").type) else None
    columns = [c for c in WANTED if c in schema.names] + [geom_col] + ([bbox_col] if bbox_col else [])
    boxes = row_group_boxes(pf, bbox_col)
    known = [b for b in boxes if b is not None]

    # GBA's own GeoJSON is lon/lat; guard against a copy written lat/lon.
    flipped = bool(known) and max(max(abs(b[1]), abs(b[3])) for b in known) > 90
    want = (aoi_bbox[1], aoi_bbox[0], aoi_bbox[3], aoi_bbox[2]) if flipped else aoi_bbox
    groups = [i for i, b in enumerate(boxes)
              if b is None or not (b[2] < want[0] or b[0] > want[2] or b[3] < want[1] or b[1] > want[3])]
    if not bbox_col:
        log(f"  {name}: no bbox column, so every row group is read")
    elif len(known) < len(boxes):
        log(f"  {name}: {len(boxes) - len(known)} row groups lack bbox statistics and are read whole")
    if flipped:
        log(f"  {name}: coordinates are stored lat/lon; swapping to lon/lat")
    log(f"  {name}: {pf.metadata.num_rows:,} buildings in {len(boxes):,} row groups; "
        f"reading {len(groups):,} that overlap the study area")
    stats["row_groups_total"] += len(boxes)
    stats["row_groups_read"] += len(groups)

    attrs, geoms_out = [], []
    for i in tqdm(groups, desc=name, unit="group", leave=False):
        table = pf.read_row_group(i, columns=columns, use_threads=False)
        stats["rows_scanned"] += table.num_rows
        mask = pc.is_valid(table[geom_col])
        if bbox_col:
            box = table[bbox_col]
            x0, y0 = pc.struct_field(box, "xmin"), pc.struct_field(box, "ymin")
            x1, y1 = pc.struct_field(box, "xmax"), pc.struct_field(box, "ymax")
            mask = pc.and_(mask, pc.and_(pc.and_(pc.less_equal(x0, want[2]), pc.greater_equal(x1, want[0])),
                                         pc.and_(pc.less_equal(y0, want[3]), pc.greater_equal(y1, want[1]))))
        table = table.filter(pc.fill_null(mask, False))
        if table.num_rows == 0:
            continue
        geoms = shapely.from_wkb(table[geom_col].combine_chunks().to_numpy(zero_copy_only=False))
        if flipped:
            geoms = shapely.transform(geoms, lambda xy: xy[:, ::-1])
        centres = shapely.centroid(geoms)
        ok = ~shapely.is_missing(centres) & ~shapely.is_empty(centres)
        inside = np.zeros(len(geoms), dtype=bool)
        cx, cy = shapely.get_x(centres[ok]), shapely.get_y(centres[ok])
        inside[np.flatnonzero(ok)] = shapely.contains_xy(aoi, cx, cy)
        if not inside.any():
            continue
        keep = table.filter(pa.array(inside)).select([c for c in WANTED if c in table.column_names])
        attrs.append(keep.to_pandas())
        geoms_out.append(geoms[inside])
    return attrs, geoms_out


# -------------------------------------------------------------------- main --

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GlobalBuildingAtlas buildings with heights for a study area.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--aoi", type=Path, nargs="+", required=True,
                   help="Boundary files, or folders of them, whose union is the study area")
    p.add_argument("--out", type=Path, default=Path("gba_buildings.fgb"), help="Output FlatGeobuf")
    p.add_argument("--storey-height", type=float, default=3.0, help="Metres per storey for the storey estimate")
    p.add_argument("--h3-res", type=int, default=9, help="H3 resolution of the cell column (0 to leave it out)")
    p.add_argument("--download", action="store_true",
                   help="Download whole tiles (resumable) and read them locally, instead of range requests")
    p.add_argument("--cache", type=Path, default=Path("gba_cache"), help="Folder for --download")
    p.add_argument("--base-url", default=DATA_BASE, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    session = make_session()

    aoi, aoi_files = load_aoi(args.aoi)
    shapely.prepare(aoi)
    aoi_bbox = aoi.bounds
    laea = equal_area_crs(aoi)
    log(f"Study area from {len(aoi_files)} file(s), bounds "
        f"{aoi_bbox[0]:.3f}, {aoi_bbox[1]:.3f}, {aoi_bbox[2]:.3f}, {aoi_bbox[3]:.3f}")

    base = f"{args.base_url}/{ACCOUNT}/{PRODUCT}"
    try:
        listing = list_tiles(session, args.base_url)
        tiles = [(n, s) for n, s in listing if (b := tile_bounds(n)) and
                 not (b[2] <= aoi_bbox[0] or b[0] >= aoi_bbox[2] or b[3] <= aoi_bbox[1] or b[1] >= aoi_bbox[3])]
        log(f"Source Cooperative lists {len(listing):,} tiles; {len(tiles)} overlap the study area")
    except RuntimeError as exc:
        log(f"Could not list the tiles ({exc}); trying the expected tile names")
        tiles = [(n, head_info(session, f"{base}/{n}").get("size")) for n in expected_tiles(aoi_bbox)]
        tiles = [(n, s) for n, s in tiles if s]
    if not tiles:
        sys.exit(f"No GBA tile overlaps the study area at {DATASET_PAGE}")

    stats = {"row_groups_total": 0, "row_groups_read": 0, "rows_scanned": 0, "bytes_fetched": 0, "requests": 0}
    provenance, attrs, geoms = [], [], []
    for name, size in tiles:
        url = f"{base}/{name}"
        info = head_info(session, url)
        size = info.get("size") or size
        log(f"Tile {name} ({human(size)})")
        if args.download:
            handle = open(download(session, url, args.cache / Path(name).name, size), "rb")
        else:
            if not size:
                sys.exit(f"Could not get the size of {url}; rerun with --download")
            handle = RemoteFile(session, url, size)
        with handle:
            part_attrs, part_geoms = read_tile(pq.ParquetFile(handle), name, aoi, aoi_bbox, stats)
            if isinstance(handle, RemoteFile):
                stats["bytes_fetched"] += handle.fetched
                stats["requests"] += handle.requests
        attrs += part_attrs
        geoms += part_geoms
        provenance.append({"tile": name, "url": url, "size": size, **{k: info.get(k) for k in ("last_modified", "etag")}})

    if not attrs:
        sys.exit("No GBA building has its centroid inside the study area")
    table = pd.concat(attrs, ignore_index=True)
    shapes = np.concatenate(geoms)
    gdf = gpd.GeoDataFrame(table, geometry=shapes, crs="EPSG:4326")
    if "id" in gdf.columns:
        gdf = gdf.rename(columns={"id": "gba_id"})
        gdf["gba_id"] = gdf["gba_id"].astype(str)

    log(f"Measuring {len(gdf):,} footprints")
    height = pd.to_numeric(gdf["height"], errors="coerce").to_numpy(float) if "height" in gdf else np.full(len(gdf), np.nan)
    known = np.isfinite(height) & (height > 0)
    area = gdf.geometry.to_crs(laea).area.to_numpy()
    storeys = np.where(known, np.maximum(1, np.rint(np.where(known, height, 0) / args.storey_height)), 1).astype(int)
    gdf["height"] = np.where(known, height, np.nan)
    gdf["height_known"] = known
    gdf["area_m2"] = area.round(1)
    gdf["storeys"] = storeys
    gdf["floor_area_m2"] = (area * storeys).round(1)
    if args.h3_res:
        centres = shapely.centroid(gdf.geometry.to_numpy())
        lat, lon = shapely.get_y(centres), shapely.get_x(centres)
        gdf[f"h3_r{args.h3_res}"] = [h3.latlng_to_cell(a, b, args.h3_res) for a, b in zip(lat, lon)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    log(f"Writing {args.out}")
    gdf.to_file(args.out, driver="FlatGeobuf")

    # Summary
    def pct(v):
        return float(np.nanpercentile(height[known], v)) if known.any() else None

    by_source = gdf["source"].value_counts().to_dict() if "source" in gdf else {}
    bands = pd.cut(storeys, [0, 1, 2, 5, 10, 20, 40, 10_000], labels=["1", "2", "3-5", "6-10", "11-20", "21-40", "41+"])
    storey_counts = pd.Series(bands).value_counts().reindex(bands.categories).to_dict()
    floor_by_band = (pd.Series(gdf["floor_area_m2"].to_numpy()).groupby(bands, observed=False).sum()
                     .reindex(bands.categories).fillna(0.0))
    tallest = gdf.iloc[np.argsort(-np.where(known, height, -1))[:15]]
    tall_rows = []
    for _, row in tallest.iterrows():
        c = row.geometry.centroid
        tall_rows.append({"lon": round(c.x, 5), "lat": round(c.y, 5), "height_m": round(float(row["height"]), 1),
                          "storeys": int(row["storeys"]), "footprint_m2": float(row["area_m2"]),
                          "source": row.get("source")})
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": f"gba_buildings.py {SCRIPT_VERSION}",
        "dataset": DATASET_PAGE,
        "citation": "Zhu, X. X., Chen, S., Zhang, F., Shi, Y. and Wang, Y. (2025). GlobalBuildingAtlas: an open "
                    "global and complete dataset of building polygons, heights and LoD1 3D models. ESSD 17, 6647-6668.",
        "licence": "Heights and non-ODbL polygons CC BY-NC 4.0; OpenStreetMap and Microsoft polygons ODbL",
        "aoi_files": aoi_files, "tiles": provenance, "transfer": stats,
        "buildings": len(gdf), "by_source": {str(k): int(v) for k, v in by_source.items()},
        "height_known_share": float(known.mean()),
        "height_m": {"median": pct(50), "p90": pct(90), "p99": pct(99), "max": pct(100)},
        "storey_height_m": args.storey_height,
        "buildings_by_storeys": {str(k): int(v) for k, v in storey_counts.items()},
        "floor_area_share_by_storeys": {str(k): float(v / floor_by_band.sum()) for k, v in floor_by_band.items()},
        "footprint_km2": float(area.sum() / 1e6), "floor_area_km2": float((area * storeys).sum() / 1e6),
        "tallest": tall_rows,
    }
    summary_path = args.out.with_name(args.out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    log("")
    log(f"{len(gdf):,} buildings; heights for {known.mean():.1%}")
    if by_source:
        log("by source: " + ", ".join(f"{k} {v:,}" for k, v in by_source.items()))
    if known.any():
        log(f"height m: median {pct(50):.1f}, 90th percentile {pct(90):.1f}, 99th {pct(99):.1f}, max {pct(100):.1f}")
    log("storeys   buildings   share of floor area")
    for band in bands.categories:
        log(f"{band:>7} {storey_counts[band]:>11,} {floor_by_band[band] / floor_by_band.sum():>12.1%}")
    log("tallest (check a few against what you know):")
    for row in tall_rows[:10]:
        log(f"  {row['height_m']:>6.1f} m  {row['storeys']:>3} storeys  {row['footprint_m2']:>9,.0f} m2  "
            f"{row['lat']:.5f}, {row['lon']:.5f}  {row['source']}")
    if not args.download:
        log(f"fetched {human(stats['bytes_fetched'])} in {stats['requests']} range requests; "
            f"read {stats['row_groups_read']:,} of {stats['row_groups_total']:,} row groups")
    log(f"Wrote {args.out} and {summary_path} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())