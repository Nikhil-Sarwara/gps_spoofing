"""terrain_classifier.py — Step 1: Terrain Classifier

Classifies GPS coordinates into terrain types: flat, mountain, sea.
Primary method: Open-Elevation REST API.
Fallback method: heuristic using alt_m / rel_alt_m from the CSV.

Usage (CLI):
    python terrain_classifier.py --csv path/to/cleaned.csv \
                                  --output path/to/output.csv \
                                  --segment-km 1.0

Usage (module):
    from terrain_classifier import TerrainClassifier
    tc = TerrainClassifier()
    label = tc.classify_point(47.3979, 8.5461, alt_m=450.0)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Optional requests import (graceful fallback if missing)
# ---------------------------------------------------------------------------
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "Step_5_Data" / "artifacts" / "elevation_cache.json"
_API_URL            = "https://api.open-elevation.com/api/v1/lookup"
_API_BATCH_SIZE     = 100
_API_TIMEOUT_S      = 10
_CACHE_ROUND_DP     = 4     # decimal places for cache key

# Classification thresholds
_MOUNTAIN_ELEV_M    = 800.0
_SEA_ELEV_M         = 0.0
_MOUNTAIN_STD_M     = 150.0  # elevation std-dev across segment -> mountain
_HEURISTIC_WINDOW   = 30
_HEURISTIC_STD_M    = 50.0   # rolling std of alt_m -> mountain


# ===========================================================================
class TerrainClassifier:
    """Classify GPS coordinates into flat / mountain / sea terrain."""

    TERRAIN_FLAT     = "flat"
    TERRAIN_MOUNTAIN = "mountain"
    TERRAIN_SEA      = "sea"

    def __init__(
        self,
        cache_path: str = None,
        use_api: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        cache_path : str, optional
            Path to elevation_cache.json.  Defaults to Step_5_Data/artifacts/elevation_cache.json.
        use_api : bool
            If False, always use heuristic (offline / fast mode).
        """
        self.use_api    = use_api and _HAS_REQUESTS
        self.cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE_PATH
        self._cache: dict[str, float] = self._load_cache()
        self._api_available: bool = True   # toggled to False on first failure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_point(
        self,
        lat: float,
        lon: float,
        alt_m: float = None,
        rel_alt_m: float = None,
    ) -> str:
        """Classify a single GPS coordinate.

        Returns one of: "flat", "mountain", "sea"
        """
        elev = self._get_elevation(lat, lon, alt_m)
        return self._classify_elevation(elev, alt_m=alt_m, rel_alt_m=rel_alt_m)

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add terrain_type column to a cleaned GPS log DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: lat_deg, lon_deg, alt_m.
            rel_alt_m is used by the heuristic fallback if present.

        Returns
        -------
        pd.DataFrame with terrain_type column appended.
        """
        df = df.copy()

        # Fill NaN coordinates with previous valid value
        df["lat_deg"] = pd.to_numeric(df["lat_deg"], errors="coerce").ffill().fillna(0.0)
        df["lon_deg"] = pd.to_numeric(df["lon_deg"], errors="coerce").ffill().fillna(0.0)
        df["alt_m"]   = pd.to_numeric(df["alt_m"],   errors="coerce").ffill().fillna(0.0)
        has_rel = "rel_alt_m" in df.columns
        if has_rel:
            df["rel_alt_m"] = pd.to_numeric(df["rel_alt_m"], errors="coerce").ffill().fillna(0.0)

        lats = df["lat_deg"].values
        lons = df["lon_deg"].values
        alts = df["alt_m"].values
        rels = df["rel_alt_m"].values if has_rel else np.zeros(len(df))

        # Fetch elevations in batch (API or heuristic)
        elevations = self._fetch_elevations_batch(lats, lons, alts)

        # Row-level classification
        labels = np.array(
            [self._classify_elevation(e, alt_m=a, rel_alt_m=r)
             for e, a, r in zip(elevations, alts, rels)],
            dtype=object,
        )

        # Rolling window mountain override (heuristic)
        rolling_std = (
            pd.Series(alts)
            .rolling(_HEURISTIC_WINDOW, min_periods=5)
            .std()
            .fillna(0.0)
            .values
        )
        labels[rolling_std > _HEURISTIC_STD_M] = self.TERRAIN_MOUNTAIN

        df["terrain_type"] = labels
        return df

    def classify_csv(
        self,
        input_csv_path: str,
        output_csv_path: str = None,
    ) -> pd.DataFrame:
        """Read a cleaned CSV, classify, write output with terrain_type column.

        If output_csv_path is None, overwrites the input file.
        """
        input_csv_path  = Path(input_csv_path)
        output_csv_path = Path(output_csv_path) if output_csv_path else input_csv_path

        df = pd.read_csv(input_csv_path, low_memory=False)
        if len(df) == 0:
            import warnings
            warnings.warn(
                f"[TerrainClassifier] Skipping {input_csv_path.name}: 0 rows after cleaning. "
                f"File left unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return df  # return empty df without overwriting
        df = self.classify_dataframe(df)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        return df

    def get_segment_terrain(
        self,
        df: pd.DataFrame,
        segment_km: float = 1.0,
    ) -> list[dict]:
        """Split the flight path into ~segment_km chunks and assign terrain labels.

        Uses cumulative Haversine distance along (lat_deg, lon_deg).
        terrain_type column must already exist (call classify_dataframe first).

        Returns
        -------
        list of dicts with keys:
          segment_id, start_time_s, end_time_s,
          start_lat, start_lon, end_lat, end_lon,
          distance_km, terrain_type
        """
        df = df.copy()
        if "terrain_type" not in df.columns:
            df = self.classify_dataframe(df)

        lats  = df["lat_deg"].values
        lons  = df["lon_deg"].values
        times = df["time_s"].values if "time_s" in df.columns else np.arange(len(df)) * 0.1

        # Cumulative Haversine distance
        cum_dist_km = np.zeros(len(df))
        for i in range(1, len(df)):
            cum_dist_km[i] = cum_dist_km[i - 1] + _haversine_km(
                lats[i - 1], lons[i - 1], lats[i], lons[i]
            )

        segments       = []
        seg_id         = 0
        seg_start_i    = 0
        seg_start_dist = 0.0

        for i in range(1, len(df)):
            seg_dist = cum_dist_km[i] - seg_start_dist
            if seg_dist >= segment_km or i == len(df) - 1:
                seg_rows = df.iloc[seg_start_i: i + 1]
                majority = seg_rows["terrain_type"].mode()
                terrain  = majority.iloc[0] if len(majority) > 0 else self.TERRAIN_FLAT

                segments.append({
                    "segment_id":   seg_id,
                    "start_time_s": float(times[seg_start_i]),
                    "end_time_s":   float(times[i]),
                    "start_lat":    float(lats[seg_start_i]),
                    "start_lon":    float(lons[seg_start_i]),
                    "end_lat":      float(lats[i]),
                    "end_lon":      float(lons[i]),
                    "distance_km":  round(seg_dist, 4),
                    "terrain_type": terrain,
                })
                seg_id        += 1
                seg_start_i    = i
                seg_start_dist = cum_dist_km[i]

        return segments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_elevation(self, lat: float, lon: float, alt_m: float = None) -> float:
        """Return ground elevation for a single point (API or cache or heuristic)."""
        cache_key = _cache_key(lat, lon)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.use_api and self._api_available:
            elev = self._api_lookup_single(lat, lon)
            if elev is not None:
                self._cache[cache_key] = elev
                self._save_cache()
                return elev

        # Fallback: use alt_m as proxy
        return alt_m if alt_m is not None else 0.0

    def _fetch_elevations_batch(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        alts: np.ndarray,
    ) -> np.ndarray:
        """Fetch elevations for all rows, using cache + API batch calls."""
        n = len(lats)
        elevations = np.full(n, np.nan)

        # 1. Fill from cache
        miss_idx = []
        for i in range(n):
            key = _cache_key(lats[i], lons[i])
            if key in self._cache:
                elevations[i] = self._cache[key]
            else:
                miss_idx.append(i)

        # 2. Batch API for cache misses
        if miss_idx and self.use_api and self._api_available:
            for batch_start in range(0, len(miss_idx), _API_BATCH_SIZE):
                batch  = miss_idx[batch_start: batch_start + _API_BATCH_SIZE]
                coords = [{"latitude": float(lats[i]), "longitude": float(lons[i])}
                          for i in batch]
                results = self._api_lookup_batch(coords)
                if results is not None:
                    for j, i in enumerate(batch):
                        elev = results[j]
                        if elev is not None:
                            elevations[i] = elev
                            self._cache[_cache_key(lats[i], lons[i])] = elev
                else:
                    self._api_available = False
                    break

            self._save_cache()

        # 3. Fill remaining NaN with alt_m heuristic
        nan_mask = np.isnan(elevations)
        elevations[nan_mask] = alts[nan_mask]

        return elevations

    def _classify_elevation(
        self,
        elevation: float,
        alt_m: float = None,
        rel_alt_m: float = None,
    ) -> str:
        """Apply classification rules to a single elevation value.

        Ground elevation (from API/cache) is the authoritative signal:
          > 800m           -> mountain
          0m < elev <= 800m -> flat
          <= 0m            -> sea  (cache-seeded -1.0 for synthetic sea flights)
        """
        if elevation > _MOUNTAIN_ELEV_M:
            return self.TERRAIN_MOUNTAIN
        if elevation > _SEA_ELEV_M:
            # Positive ground elevation: flat.
            # (Note: Removed the legacy heuristic that treated elevation <= 5.0
            # as sea, as it caused false positives in low-lying coastal land
            # areas like the Baylands).
            return self.TERRAIN_FLAT

        # elevation <= 0 (sea level or below, including cache-seeded -1.0)
        return self.TERRAIN_SEA

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _api_lookup_single(self, lat: float, lon: float) -> Optional[float]:
        results = self._api_lookup_batch([{"latitude": lat, "longitude": lon}])
        if results and results[0] is not None:
            return results[0]
        return None

    def _api_lookup_batch(
        self, coords: list[dict]
    ) -> Optional[list[Optional[float]]]:
        """POST a batch of coords to Open-Elevation API."""
        try:
            resp = requests.post(
                _API_URL,
                json={"locations": coords},
                timeout=_API_TIMEOUT_S,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [r.get("elevation") for r in data.get("results", [])]
        except Exception:
            self._api_available = False
            return None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)


# ---------------------------------------------------------------------------
# Haversine distance (inline — no external geo libs)
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two WGS84 points."""
    R    = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = (math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, _CACHE_ROUND_DP)}_{round(lon, _CACHE_ROUND_DP)}"


# ---------------------------------------------------------------------------
# Spot-check validation (run before CLI logic)
# ---------------------------------------------------------------------------

def _run_spot_checks(tc: TerrainClassifier) -> bool:
    checks = [
        (47.3979, 8.5461,   450.0,  0.24,  "flat",     "Zurich suburb"),
        (46.8500, 9.5300,  1800.0, 1800.0, "mountain", "Swiss Alps"),
        (35.0000, 25.0000,   0.0,   0.0,   "sea",      "Mediterranean Sea"),
    ]
    print("\n--- Spot checks ---")
    all_pass = True
    for lat, lon, alt, rel, expected, desc in checks:
        result = tc.classify_point(lat, lon, alt_m=alt, rel_alt_m=rel)
        ok     = result == expected
        if not ok:
            all_pass = False
        print(f"  {'OK' if ok else 'FAIL'} classify_point({lat}, {lon}, alt={alt}) "
              f"-> {result!r}  (expected {expected!r}) [{desc}]")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    df: pd.DataFrame,
    segments: list[dict],
    method: str,
    csv_path: str,
) -> None:
    n      = len(df)
    counts = df["terrain_type"].value_counts()
    print(f"\n{'─'*55}")
    print(f"  File   : {csv_path}")
    print(f"  Rows   : {n}")
    print(f"  Method : {method}")
    print("  Terrain distribution:")
    for terrain, count in counts.items():
        print(f"    {terrain:<10} {count:5d} rows  ({count/n*100:.1f}%)")
    print(f"  Segments ({len(segments)} total):")
    for seg in segments:
        print(f"    seg {seg['segment_id']:02d} | "
              f"{seg['start_time_s']:.1f}-{seg['end_time_s']:.1f}s | "
              f"{seg['distance_km']:.3f} km | {seg['terrain_type']}")
    print(f"{'─'*55}\n")


# ---------------------------------------------------------------------------
# CLI __main__
# ---------------------------------------------------------------------------

def main(argv: list[str] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Terrain classifier for GPS spoofing detection project"
    )
    parser.add_argument("--csv",        required=True,         help="Path to cleaned CSV")
    parser.add_argument("--output",     default=None,          help="Output path (default: overwrite input)")
    parser.add_argument("--segment-km", type=float, default=1.0, help="Segment size km (default 1.0)")
    parser.add_argument("--no-api",     action="store_true",   help="Disable API, heuristic only")
    parser.add_argument("--cache",      default=None,          help="Path to elevation_cache.json")
    args = parser.parse_args(argv)

    tc = TerrainClassifier(cache_path=args.cache, use_api=not args.no_api)

    # Spot checks before processing
    _run_spot_checks(tc)

    print(f"Processing: {args.csv}")
    df = tc.classify_csv(args.csv, args.output)

    method   = "API" if (tc.use_api and tc._api_available) else "heuristic"
    segments = tc.get_segment_terrain(df, segment_km=args.segment_km)
    _print_summary(df, segments, method, args.csv)


if __name__ == "__main__":
    main()
