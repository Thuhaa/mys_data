#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "geopandas>=1.0",
#     "pyogrio>=0.10",
#     "pyarrow>=16",
#     "shapely>=2.0",
#     "requests>=2.31",
#     "tqdm>=4.66",
# ]
# ///
"""
Download Google-Microsoft Open Buildings (combined by VIDA) as FlatGeobuf for a
set of admin areas, keeping only the buildings inside their combined boundary.

Default area: Selangor, Kuala Lumpur and Putrajaya, taken from geoBoundaries
gbOpen ADM1 (ISO 3166-2 codes MY-10, MY-14 and MY-16).

What it does
  1. Builds the area of interest (AOI) from geoBoundaries, or from your own
     boundary file with --aoi, and dissolves it into one shape.
  2. Lists flatgeobuf/<layout>/country_iso=<ISO3>/ on Source Cooperative
     through its S3-compatible LIST API.
  3. Reads each file's FlatGeobuf header over HTTP and keeps only the files
     whose extent overlaps the AOI.
  4. Downloads those files with a resumable progress bar (default), or with
     --stream reads only the AOI bounding box from the remote files through
     the FlatGeobuf spatial index.
  5. Keeps buildings whose centroid falls inside the AOI (or that touch it,
     with --predicate intersects), tags each one with its admin area, and
     writes an indexed FlatGeobuf, the AOI used, and a provenance JSON.

Usage (uv installs the dependencies above on the first run)
  uv run vida_buildings.py --dry-run
  uv run vida_buildings.py
  uv run vida_buildings.py --stream
  uv run vida_buildings.py --iso MY-10 --out selangor_buildings.fgb
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyogrio
import pyogrio.raw
import requests
import shapely
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

SCRIPT_VERSION = "1.0"
DATA_BASE = "https://data.source.coop"
ACCOUNT = "vida"
PRODUCT = "google-microsoft-open-buildings"
DATASET_PAGE = f"https://source.coop/{ACCOUNT}/{PRODUCT}"
AWS_LIST = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop"
GB_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{level}/"
GB_GITHUB = (
    "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/main/"
    "releaseData/gbOpen/{iso3}/{level}/geoBoundaries-{iso3}-{level}.geojson"
)
# Selangor, Kuala Lumpur, Putrajaya
DEFAULT_CODES = {"MYS": ["MY-10", "MY-14", "MY-16"]}
USER_AGENT = f"cgt-vida-buildings/{SCRIPT_VERSION}"
# Holes smaller than this in the dissolved AOI are slivers between neighbouring
# boundaries (e.g. along the Selangor / Kuala Lumpur edge), so they get filled.
MIN_HOLE_M2 = 1_000_000

GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".fgb",
    "CPL_VSIL_CURL_CHUNK_SIZE": str(1024 * 1024),
    "CPL_VSIL_CURL_CACHE_SIZE": str(256 * 1024 * 1024),
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "GDAL_HTTP_USERAGENT": USER_AGENT,
}


# ----------------------------------------------------------------- helpers --

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


def count(n: int | None) -> str:
    return "?" if n is None else f"{n:,}"


def overlaps(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# --------------------------------------------------------------------- AOI --

def fetch_geoboundaries(session, iso3: str, level: str, cache_dir: Path) -> Path:
    path = cache_dir / "boundaries" / f"geoBoundaries-{iso3}-{level}.geojson"
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urls = []
    try:
        r = session.get(GB_API.format(iso3=iso3, level=level), timeout=60)
        r.raise_for_status()
        meta = r.json()
        meta = meta[0] if isinstance(meta, list) else meta
        if meta.get("gjDownloadURL"):
            urls.append(meta["gjDownloadURL"])
    except (requests.RequestException, ValueError) as exc:
        log(f"geoBoundaries API not reachable ({type(exc).__name__}), trying GitHub directly")
    urls.append(GB_GITHUB.format(iso3=iso3, level=level))
    for url in urls:
        try:
            r = session.get(url, timeout=180)
            r.raise_for_status()
            if not r.content.lstrip().startswith(b"{"):
                raise ValueError("response is not GeoJSON (Git LFS pointer or HTML page)")
            path.write_bytes(r.content)
            log(f"Boundaries downloaded from {url}")
            return path
        except (requests.RequestException, ValueError) as exc:
            log(f"  could not use {url}: {exc}")
    sys.exit("Could not download geoBoundaries. Pass your own boundary file with --aoi.")


def load_aoi(args, session, cache_dir: Path):
    """Selected areas as a GeoDataFrame (adm_name, adm_code, adm_id) in EPSG:4326."""
    if args.aoi:
        gdf = gpd.read_file(args.aoi)
        source = str(args.aoi.resolve())
    else:
        path = fetch_geoboundaries(session, args.country, args.adm_level, cache_dir)
        gdf = gpd.read_file(path)
        source = f"geoBoundaries gbOpen {args.country} {args.adm_level} ({path.name})"

    name_field = args.name_field or next(
        (c for c in ("shapeName", "name", "NAME_1", "NAME") if c in gdf.columns), None
    )
    code_field = "shapeISO" if "shapeISO" in gdf.columns else None
    id_field = "shapeID" if "shapeID" in gdf.columns else None

    codes, names = args.iso, args.names
    if not args.aoi and not codes and not names:
        codes = DEFAULT_CODES.get(args.country)
        if not codes:
            sys.exit(f"No default areas for {args.country}. Pass --iso or --names.")

    if codes:
        if not code_field:
            sys.exit("--iso needs a shapeISO field in the boundary file. Use --names instead.")
        wanted = [c.upper() for c in codes]
        column = gdf[code_field].astype(str).str.upper()
    elif names:
        if not name_field:
            sys.exit("No name field found in the boundary file. Pass --name-field.")
        wanted = [n.casefold() for n in names]
        column = gdf[name_field].astype(str).str.casefold()
    else:
        wanted = column = None

    if wanted is not None:
        mask = column.isin(wanted)
        missing = sorted(set(wanted) - set(column[mask]))
        if missing:
            labels = gdf[name_field].astype(str) if name_field else gdf.index.astype(str)
            tags = gdf[code_field].astype(str) if code_field else [""] * len(gdf)
            available = ", ".join(f"{n} ({c})" if c else n for n, c in zip(labels, tags))
            sys.exit(f"Not found in the boundaries: {', '.join(missing)}\nAvailable: {available}")
        gdf = gdf[mask]
    if gdf.empty:
        sys.exit("The boundary selection is empty.")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    n = len(gdf)
    parts = gpd.GeoDataFrame(
        {
            "adm_name": gdf[name_field].astype(str).to_numpy() if name_field else [f"area {i + 1}" for i in range(n)],
            "adm_code": gdf[code_field].astype(str).to_numpy() if code_field else [""] * n,
            "adm_id": gdf[id_field].astype(str).to_numpy() if id_field else [""] * n,
        },
        geometry=gdf.geometry.make_valid().to_numpy(),
        crs=gdf.crs,
    ).to_crs("EPSG:4326")
    return parts, source


def dissolve(parts: gpd.GeoDataFrame, buffer_m: float):
    """Union the areas in a local UTM CRS, fill sliver holes, optionally buffer."""
    utm = parts.estimate_utm_crs()
    union = shapely.union_all(np.asarray(parts.to_crs(utm).geometry.to_numpy()))
    polygons = []
    for poly in shapely.get_parts(union):
        if poly.geom_type != "Polygon":
            continue
        holes = [ring for ring in poly.interiors if shapely.Polygon(ring).area >= MIN_HOLE_M2]
        polygons.append(shapely.Polygon(poly.exterior, holes))
    if not polygons:
        sys.exit("The selected boundaries contain no polygons.")
    union = shapely.MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    if buffer_m:
        union = union.buffer(buffer_m)
    area_km2 = union.area / 1e6
    return gpd.GeoSeries([union], crs=utm).to_crs("EPSG:4326"), area_km2


def default_output(country: str, parts: gpd.GeoDataFrame) -> Path:
    if len(parts) <= 4:
        slug = "_".join(re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-") for n in parts["adm_name"])
    else:
        slug = f"{len(parts)}-areas"
    return Path(f"vida_buildings_{country.lower()}_{slug}.fgb")


# ------------------------------------------------------------ remote files --

def s3_list(session, url: str, prefix: str):
    """S3 ListObjectsV2 against url, returning [(file name, size)] for .fgb keys."""
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
            if key.endswith(".fgb"):
                size = int(item.findtext("Size") or 0)
                found.append((key.rsplit("/", 1)[-1], size or None))
        token = root.findtext("NextContinuationToken")
        if (root.findtext("IsTruncated") or "").lower() == "true" and token:
            params["continuation-token"] = token
        else:
            return found


def list_files(session, base: str, prefix: str):
    attempts = [
        (f"{base}/{ACCOUNT}/{PRODUCT}", prefix),  # current data proxy
        (f"{base}/{ACCOUNT}", f"{PRODUCT}/{prefix}"),  # older proxy form
    ]
    if base == DATA_BASE:
        attempts.append((AWS_LIST, f"{ACCOUNT}/{PRODUCT}/{prefix}"))  # bucket behind the proxy
    problems = []
    for url, pfx in attempts:
        try:
            found = s3_list(session, url, pfx)
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            problems.append(f"{url}: {exc}")
            continue
        if found:
            return found
        problems.append(f"{url}: no .fgb files under {pfx}")
    raise RuntimeError("\n  ".join(problems))


def head_info(session, url: str) -> dict:
    try:
        r = session.head(url, allow_redirects=True, timeout=60)
        if r.ok:
            size = r.headers.get("Content-Length")
            return {
                "size": int(size) if size else None,
                "last_modified": r.headers.get("Last-Modified"),
                "etag": r.headers.get("ETag"),
            }
    except requests.RequestException:
        pass
    return {}


def inspect_source(path: str) -> dict:
    """Feature count, extent, CRS and index flag from a FlatGeobuf header."""
    info = pyogrio.read_info(path)
    caps = info.get("capabilities") or {}
    bounds = info.get("total_bounds")
    if bounds is not None:
        bounds = tuple(float(v) for v in bounds)
        if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
            bounds = None
    features = info.get("features")
    return {
        "features": int(features) if features is not None and features >= 0 else None,
        "bounds": bounds,
        "crs": info.get("crs"),
        "geometry_type": info.get("geometry_type"),
        "indexed": bool(caps.get("fast_spatial_filter", False)),
    }


def content_total(r: requests.Response, done: int) -> int | None:
    match = re.match(r"bytes \d+-\d+/(\d+)", r.headers.get("Content-Range", ""))
    if match:
        return int(match.group(1))
    length = r.headers.get("Content-Length")
    return int(length) + done if length else None


def download(session, url: str, dest: Path, size: int | None) -> Path:
    """Resumable download with a byte progress bar. Partial data is kept in <dest>.part."""
    if dest.exists() and (size is None or dest.stat().st_size == size):
        log(f"Already downloaded: {dest} ({human(dest.stat().st_size)})")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    if size is not None and done > size:
        part.unlink()
        done = 0
    if done:
        log(f"Resuming {dest.name} from {human(done)}")
    bar = tqdm(total=size, initial=done, unit="B", unit_scale=True, unit_divisor=1024,
               desc=dest.name, dynamic_ncols=True)
    failures = 0
    try:
        while size is None or done < size:
            start = done
            try:
                headers = {"Range": f"bytes={done}-"} if done else {}
                with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                    if r.status_code == 416 and size is not None and done >= size:
                        break
                    r.raise_for_status()
                    if done and r.status_code != 206:
                        log("  the server ignored the resume request, restarting this file")
                        done = 0
                        bar.reset(total=size)
                    if size is None:
                        size = content_total(r, done)
                        bar.total = size
                        bar.refresh()
                    with open(part, "ab" if done else "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            fh.write(chunk)
                            done += len(chunk)
                            bar.update(len(chunk))
                if size is None:
                    break
                if done < size:
                    raise requests.ConnectionError("connection closed before the end of the file")
            except (requests.RequestException, OSError) as exc:
                failures = failures + 1 if done == start else 1
                if failures > 8:
                    raise
                wait = min(60, 2 ** failures)
                log(f"  {type(exc).__name__} at {human(done)}, resuming in {wait}s")
                time.sleep(wait)
    finally:
        bar.close()
    got = part.stat().st_size
    if size is not None and got != size:
        raise RuntimeError(f"{dest.name}: received {got} bytes, expected {size}")
    os.replace(part, dest)
    return dest


def readme_version(session, base: str) -> str | None:
    """Latest 'Version x - date' entry in the dataset README, for provenance."""
    try:
        r = session.get(f"{base}/{ACCOUNT}/{PRODUCT}/README.md", timeout=60)
        r.raise_for_status()
    except requests.RequestException:
        return None
    match = re.search(r"Version\s+([\d.]+)\s*[-:(]\s*(\d{4}-\d{2}-\d{2})", r.text)
    return f"{match.group(1)} ({match.group(2)})" if match else None


# -------------------------------------------------------------------- clip --

class Stats:
    def __init__(self):
        self.scanned = 0
        self.kept = 0
        self.by_area: Counter = Counter()
        self.by_source: Counter = Counter()


def geometry_column(meta: dict, schema: pa.Schema) -> str:
    name = meta.get("geometry_name") or "wkb_geometry"
    if name in schema.names:
        return name
    for field in schema:
        ext = (field.metadata or {}).get(b"ARROW:extension:name")
        if ext in (b"ogc.wkb", b"geoarrow.wkb"):
            return field.name
    raise RuntimeError("Could not find the geometry column in the source data.")


def clipped_stream(sources, bbox, aoi, parts, predicate, stats: Stats, bar, batch_size=65536):
    """Open the sources with a bbox filter and yield only the batches' rows inside the AOI."""
    with pyogrio.raw.open_arrow(sources[0], bbox=bbox, batch_size=batch_size, use_pyarrow=True) as (meta, reader):
        in_schema = reader.schema
    geom = geometry_column(meta, in_schema)
    out_schema = in_schema.append(pa.field("adm_name", pa.string())).append(pa.field("adm_code", pa.string()))
    has_source = "bf_source" in in_schema.names

    shapely.prepare(aoi)
    areas = list(zip(parts["adm_name"], parts["adm_code"], parts.geometry.to_numpy()))
    for _, _, g in areas:
        shapely.prepare(g)

    def clip(batch: pa.RecordBatch):
        geoms = shapely.from_wkb(batch.column(geom).to_numpy(zero_copy_only=False))
        centroids = shapely.centroid(geoms)
        x, y = shapely.get_x(centroids), shapely.get_y(centroids)
        if predicate == "centroid":
            keep = shapely.intersects_xy(aoi, x, y)
        else:
            keep = shapely.intersects(aoi, geoms)
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            return None
        xk, yk = x[idx], y[idx]
        names = np.full(idx.size, None, dtype=object)
        codes = np.full(idx.size, None, dtype=object)
        free = np.ones(idx.size, dtype=bool)
        for name, code, g in areas:
            hit = free & shapely.intersects_xy(g, xk, yk)
            names[hit] = name
            codes[hit] = code or None
            free &= ~hit
        kept = batch.take(pa.array(idx))
        return pa.RecordBatch.from_arrays(
            kept.columns + [pa.array(names, type=pa.string()), pa.array(codes, type=pa.string())],
            schema=out_schema,
        )

    def generate():
        for src in sources:
            with pyogrio.raw.open_arrow(src, bbox=bbox, batch_size=batch_size, use_pyarrow=True) as (_, reader):
                for batch in reader:
                    if batch.schema != in_schema:
                        batch = pa.Table.from_batches([batch]).cast(in_schema).to_batches()[0]
                    stats.scanned += batch.num_rows
                    out = clip(batch)
                    if out is not None:
                        stats.kept += out.num_rows
                        stats.by_area.update(out.column("adm_name").to_pylist())
                        if has_source:
                            for item in pc.value_counts(out.column("bf_source")).to_pylist():
                                stats.by_source[item["values"]] += item["counts"]
                    bar.set_postfix_str(f"kept {stats.kept:,}", refresh=False)
                    bar.update(batch.num_rows)
                    if out is not None:
                        yield out

    return meta, geom, out_schema, generate()


# -------------------------------------------------------------------- main --

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download VIDA Google-Microsoft Open Buildings (FlatGeobuf) for chosen admin areas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--country", default="MYS", help="ISO3 code, used for the VIDA folder and geoBoundaries")
    p.add_argument("--iso", nargs="+", metavar="CODE",
                   help="ISO 3166-2 codes of the areas to keep (MYS default: MY-10 MY-14 MY-16)")
    p.add_argument("--names", nargs="+", metavar="NAME", help="Area names to keep, instead of codes")
    p.add_argument("--adm-level", default="ADM1", help="geoBoundaries level to select from")
    p.add_argument("--aoi", type=Path, help="Your own boundary file (any OGR format) instead of geoBoundaries")
    p.add_argument("--name-field", help="Field in --aoi used to tag buildings with their area")
    p.add_argument("--buffer-m", type=float, default=0.0,
                   help="Grow the AOI by this many metres, e.g. to catch reclaimed coastal land")
    p.add_argument("--predicate", choices=("centroid", "intersects"), default="centroid",
                   help="centroid keeps buildings whose centroid is inside, so each building belongs "
                        "to one area; intersects keeps every building touching the AOI")
    p.add_argument("--stream", action="store_true",
                   help="Read only the AOI bounding box from the remote files instead of downloading them")
    p.add_argument("--layout", choices=("by_country_s2", "by_country"), default="by_country_s2",
                   help="VIDA FlatGeobuf folder layout")
    p.add_argument("--files", nargs="+", metavar="NAME", help="Skip listing and use these file names")
    p.add_argument("--out", type=Path, help="Output file; .fgb by default, .gpkg also works")
    p.add_argument("--cache-dir", type=Path, default=Path("vida_cache"),
                   help="Folder for raw downloads and boundaries; downloads resume from here")
    p.add_argument("--dry-run", action="store_true", help="Show the files and sizes, then stop")
    p.add_argument("--base-url", default=DATA_BASE, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.country = args.country.upper()
    args.base_url = args.base_url.rstrip("/")
    started = time.time()

    if tuple(pyogrio.__gdal_version__) < (3, 8, 0):
        sys.exit(f"Needs GDAL 3.8 or newer, pyogrio is using {pyogrio.__gdal_version_string__}. "
                 "Run it with 'uv run' to use the GDAL bundled with pyogrio.")
    pyogrio.set_gdal_config_options(GDAL_OPTIONS)
    session = make_session()
    cache = args.cache_dir

    # 1. Area of interest
    parts, aoi_source = load_aoi(args, session, cache)
    aoi, area_km2 = dissolve(parts, args.buffer_m)
    labels = [f"{n} ({c})" if c else n for n, c in zip(parts["adm_name"], parts["adm_code"])]
    log(f"AOI: {', '.join(labels)}")
    log(f"     {area_km2:,.0f} km2 from {aoi_source}"
        + (f", buffered by {args.buffer_m:g} m" if args.buffer_m else ""))

    # 2. Files in the VIDA folder
    prefix = f"flatgeobuf/{args.layout}/country_iso={args.country}/"
    folder_url = f"{args.base_url}/{ACCOUNT}/{PRODUCT}/{prefix}"
    if args.files:
        files = [(f.rsplit("/", 1)[-1], None) for f in args.files]
    else:
        log(f"Listing {folder_url}")
        try:
            files = list_files(session, args.base_url, prefix)
        except RuntimeError as exc:
            sys.exit(f"Could not list the folder:\n  {exc}\n"
                     f"Open {DATASET_PAGE} in a browser and pass the file names with --files.")

    # 3. Headers: which files overlap the AOI
    aoi_bbox_cache: dict = {}

    def aoi_bbox(crs):
        key = crs or "EPSG:4326"
        if key not in aoi_bbox_cache:
            aoi_bbox_cache[key] = tuple(float(v) for v in aoi.to_crs(key).total_bounds)
        return aoi_bbox_cache[key]

    log(f"{len(files)} file(s) {'given' if args.files else 'in the folder'}, reading headers")
    flag = {True: "yes", False: "no", None: "unknown"}
    entries = []
    for name, size in files:
        entry = {"name": name, "url": folder_url + name, "size": size}
        head = head_info(session, entry["url"])
        entry["size"] = entry["size"] or head.get("size")
        entry["last_modified"] = head.get("last_modified")
        entry["etag"] = head.get("etag")
        try:
            entry.update(inspect_source("/vsicurl/" + entry["url"]))
        except Exception as exc:  # GDAL raises several error types for network problems
            log(f"  {name}: header not readable over HTTP ({exc}); it will be checked after download")
            entry.update({"features": None, "bounds": None, "crs": None, "geometry_type": None, "indexed": False})
        entry["overlaps"] = None if entry["bounds"] is None else overlaps(entry["bounds"], aoi_bbox(entry["crs"]))
        entries.append(entry)
        log(f"  {name:<28} {human(entry.get('size')):>11}  {count(entry['features']):>12} buildings"
            f"  overlaps AOI: {flag[entry['overlaps']]:<7}  spatial index: {flag[entry['indexed']]}")

    selected = [e for e in entries if e["overlaps"] is not False]
    if not selected:
        sys.exit("None of the files overlap the AOI. Check --country and the area codes.")

    to_download = [e for e in selected if not (args.stream and e["indexed"])]
    for e in selected:
        if args.stream and not e["indexed"]:
            log(f"  {e['name']} has no readable spatial index, so it will be downloaded whole")
    sizes = [e.get("size") for e in to_download]
    if to_download:
        total = None if None in sizes else sum(sizes)
        log(f"Plan: download {len(to_download)} file(s), {human(total)}"
            + (f"; stream {len(selected) - len(to_download)}" if len(selected) > len(to_download) else ""))
    else:
        log(f"Plan: stream the AOI bounding box from {len(selected)} file(s), no full download")
    if args.dry_run:
        return 0

    # 4. Get the data
    sources, src_info = [], None
    for e in selected:
        if args.stream and e["indexed"]:
            e["mode"] = "stream"
            sources.append("/vsicurl/" + e["url"])
        else:
            local = cache / args.layout / f"country_iso={args.country}" / e["name"]
            download(session, e["url"], local, e.get("size"))
            e["mode"] = "download"
            e["local_path"] = str(local.resolve())
            if e["bounds"] is None:
                e.update(inspect_source(str(local)))
                e["overlaps"] = e["bounds"] is None or overlaps(e["bounds"], aoi_bbox(e["crs"]))
                if not e["overlaps"]:
                    log(f"  {e['name']} does not overlap the AOI, skipping it")
                    continue
            sources.append(str(local))
        src_info = src_info or e
    if not sources:
        sys.exit("None of the downloaded files overlap the AOI.")
    used = [e for e in selected if e.get("mode") and e.get("overlaps") is not False]

    # 5. Clip and write
    crs = src_info.get("crs") or "EPSG:4326"
    aoi_src = aoi.to_crs(crs).iloc[0]
    parts_src = parts.to_crs(crs)
    bbox = aoi_bbox(crs)

    out = args.out or default_output(args.country, parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    if tmp.exists():
        tmp.unlink()

    stats = Stats()
    desc = "Streaming + clipping" if any(s.startswith("/vsicurl/") for s in sources) else "Clipping"
    with tqdm(unit=" bldg", unit_scale=True, desc=desc, dynamic_ncols=True) as bar:
        meta, geom, schema, batches = clipped_stream(sources, bbox, aoi_src, parts_src, args.predicate, stats, bar)
        pyogrio.raw.write_arrow(
            pa.RecordBatchReader.from_batches(schema, batches),
            str(tmp),
            layer=out.stem,
            geometry_name=geom,
            geometry_type=meta.get("geometry_type") or src_info.get("geometry_type") or "Unknown",
            crs=meta.get("crs") or crs,
        )
    os.replace(tmp, out)

    # AOI and provenance next to the output
    aoi_path = out.with_name(f"{out.stem}_aoi.geojson")
    gpd.GeoDataFrame(
        {"areas": [", ".join(parts["adm_name"])], "codes": [", ".join(c for c in parts["adm_code"] if c)],
         "buffer_m": [args.buffer_m], "area_km2": [round(area_km2, 1)]},
        geometry=aoi.to_numpy(), crs="EPSG:4326",
    ).to_file(aoi_path, driver="GeoJSON")

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": f"vida_buildings.py {SCRIPT_VERSION}",
        "dataset": "Google-Microsoft Open Buildings, combined by VIDA",
        "dataset_page": DATASET_PAGE,
        "dataset_readme_version": readme_version(session, args.base_url),
        "license": "ODbL-1.0",
        "layout": args.layout,
        "country_iso": args.country,
        "source_files": [
            {k: e.get(k) for k in ("name", "url", "size", "last_modified", "etag", "features",
                                   "bounds", "indexed", "mode", "local_path")}
            for e in used
        ],
        "aoi": {
            "source": aoi_source,
            "areas": parts[["adm_name", "adm_code", "adm_id"]].to_dict(orient="records"),
            "buffer_m": args.buffer_m,
            "sliver_holes_filled_below_m2": MIN_HOLE_M2,
            "area_km2": round(area_km2, 1),
            "file": aoi_path.name,
        },
        "selection": {"bbox_prefilter": bbox, "predicate": args.predicate,
                      "area_tag": "area containing the building centroid; empty if none"},
        "result": {
            "file": out.name,
            "buildings": stats.kept,
            "scanned_in_bbox": stats.scanned,
            "by_area": {(k if k is not None else "(outside every area)"): v
                        for k, v in stats.by_area.most_common()},
            "by_source": dict(stats.by_source.most_common()),
        },
        "software": {"pyogrio": pyogrio.__version__, "gdal": pyogrio.__gdal_version_string__,
                     "shapely": shapely.__version__, "pyarrow": pa.__version__},
    }
    prov_path = out.with_name(f"{out.stem}_provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2, default=str))

    # Summary
    elapsed = time.time() - started
    log("")
    log(f"Done in {int(elapsed // 60)}m {int(elapsed % 60)}s")
    log(f"  output:     {out} ({human(out.stat().st_size)})")
    log(f"  buildings:  {stats.kept:,} kept of {stats.scanned:,} in the AOI bounding box")
    for name, code in zip(parts["adm_name"], parts["adm_code"]):
        label = f"{name} ({code})" if code else name
        log(f"    {label}: {stats.by_area.get(name, 0):,}")
    if stats.by_area.get(None):
        log(f"    outside every area (buffer zone or boundary sliver): {stats.by_area[None]:,}")
    if stats.by_source:
        log("  by source:  " + ", ".join(f"{k} {v:,}" for k, v in stats.by_source.most_common()))
    log(f"  AOI used:   {aoi_path}")
    log(f"  provenance: {prov_path}")
    if stats.kept == 0:
        log("WARNING: no buildings were kept. Check the area codes and --country.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\nInterrupted. Partial downloads stay in the cache folder and resume on the next run.")
        sys.exit(130)