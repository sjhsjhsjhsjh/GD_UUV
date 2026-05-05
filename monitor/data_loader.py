"""Data loading helpers for the Bellhop run monitor.

This module discovers the latest env_output map directory, parses cache_index.json,
reads Bellhop env/bty files, and builds Plotly-compatible visualization data.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_OUTPUT_ROOT = PROJECT_ROOT / "env_output"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from myutils.shd_plotter import build_tl_surface_trace
from myutils.shd_plotter import ensure_shd_plot_cache


@dataclass(frozen=True)
class CacheEntry:
    """Normalized cache entry loaded from cache_index.json."""

    cache_key: str
    env_file_path: str
    source_position: Tuple[float, float, float] | None = None
    receiver_position: Tuple[float, float, float] | None = None

    @property
    def env_id(self) -> str:
        return Path(self.env_file_path).stem if self.env_file_path else ""


def discover_latest_env_dir(outputs_root: str | Path | None = None) -> Path:
    """Return the env cache directory to display in the monitor.

    The directory layout is expected to be env_output/<map_name>/, but the root
    may also point directly at one map directory.
    """

    root = Path(outputs_root) if outputs_root else DEFAULT_ENV_OUTPUT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Env output root not found: {root}")

    if _is_env_cache_dir(root):
        return root

    map_dirs = [p for p in root.iterdir() if p.is_dir() and _is_env_cache_dir(p)]
    if not map_dirs:
        raise FileNotFoundError(f"No map folders found in env output root: {root}")

    return max(map_dirs, key=lambda item: item.stat().st_mtime)


def discover_latest_run_dir(outputs_root: str | Path | None = None) -> Path:
    """Backward-compatible alias for discover_latest_env_dir()."""

    return discover_latest_env_dir(outputs_root)


def build_manifest(outputs_root: str | Path | None = None) -> Dict[str, Any]:
    """Build a manifest for the latest env cache directory."""

    env_dir = discover_latest_env_dir(outputs_root)
    cache_index_path = env_dir / "cache_index.json"
    cache_entries = load_cache_index(cache_index_path)

    env_summaries: List[Dict[str, Any]] = []
    for entry in cache_entries:
        env_path = Path(entry.env_file_path)
        if not env_path.exists() or env_path.suffix.lower() != ".env":
            continue
        env_info = parse_env_file(env_path)
        env_summaries.append(
            {
                "cache_key": entry.cache_key,
                "env_id": entry.env_id,
                "env_file_path": str(env_path),
                "source_position": _tuple_to_list(entry.source_position or env_info["source_position"]),
                "receiver_position": _tuple_to_list(entry.receiver_position or env_info["receiver_position"]),
                "beam": env_info["beam"],
                "receiver": env_info["receiver"],
                "has_bty": _resolve_bty_path(env_path).exists(),
                "has_shd": env_path.with_suffix(".shd").exists(),
                "has_shd_plot": _resolve_shd_plot_path(env_path).exists(),
            }
        )

    env_summaries.sort(key=lambda item: item["env_id"])

    return {
        "outputs_root": str(Path(outputs_root) if outputs_root else DEFAULT_ENV_OUTPUT_ROOT),
        "env_dir": str(env_dir),
        "map_name": env_dir.name,
        "env_count": len(env_summaries),
        "envs": env_summaries,
    }


def build_env_payload(env_id: str, outputs_root: str | Path | None = None) -> Dict[str, Any]:
    """Build visualization payload for one env file."""

    env_dir = discover_latest_env_dir(outputs_root)
    cache_entries = load_cache_index(env_dir / "cache_index.json")
    cache_entry = _find_cache_entry_by_env_id(cache_entries, env_id)
    if cache_entry is None:
        raise FileNotFoundError(f"Env not found in latest run: {env_id}")

    env_path = Path(cache_entry.env_file_path)
    env_info = parse_env_file(env_path)
    bty_info = parse_bty_file(_resolve_bty_path(env_path))
    shd_path = env_path.with_suffix(".shd")

    shd_plot = None
    if shd_path.exists():
        shd_plot = ensure_shd_plot_cache(
            shd_path,
            xs_km=env_info["source_position"][0],
            ys_km=env_info["source_position"][1],
            rd_m=env_info["receiver"]["depth_m"],
            env_file=env_path,
        )

    traces = build_scene_traces(env_path, env_info, bty_info)
    if shd_plot is not None:
        traces.append(build_tl_surface_trace(shd_plot))
    layout = build_layout(env_path, env_info, bty_info)

    return {
        "manifest": build_manifest(outputs_root),
        "env": {
            "cache_key": cache_entry.cache_key,
            "env_id": cache_entry.env_id,
            "env_file_path": str(env_path),
            "source_position": _tuple_to_list(cache_entry.source_position or env_info["source_position"]),
            "receiver_position": _tuple_to_list(cache_entry.receiver_position or env_info["receiver_position"]),
            "beam": env_info["beam"],
            "receiver": env_info["receiver"],
            "meta": env_info["meta"],
            "has_shd_plot": shd_plot is not None,
        },
        "shd_plot": None if shd_plot is None else {
            "cache_path": str(_resolve_shd_plot_path(env_path)),
            "schema_version": shd_plot.get("schema_version"),
            "receiver_depth_m": shd_plot.get("receiver_depth_m"),
            "freq_hz": shd_plot.get("freq_hz"),
        },
        "figure": {"data": traces, "layout": layout},
    }


def build_layout(env_path: Path, env_info: Dict[str, Any], bty_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build Plotly layout for the 3D scene.
    
    Args:
        env_path: Path to the env file
        env_info: Parsed env file information
        bty_data: Optional bathymetry data for z-axis range calculation
    """

    title = f"Bellhop Monitor - {env_path.stem}"
    
    # Calculate z-axis range from bathymetry data
    z_min = 0
    if bty_data and bty_data.get("depth_grid"):
        # Find minimum (most negative) depth value
        all_depths = []
        for row in bty_data["depth_grid"]:
            all_depths.extend(row)
        if all_depths:
            z_min = min(all_depths)  # This will be negative
    
    z_axis_config = {
        "title": "Z (m, depth)",
        "gridcolor": "#24334d",
        "color": "#c7d4ea",
    }
    
    if z_min < 0:
        z_axis_config["range"] = [z_min * 1.05, 0]  # Add 5% margin below deepest point
    else:
        z_axis_config["autorange"] = "reversed"
    
    return {
        "title": {"text": title, "font": {"size": 20, "color": "#e5eefb"}},
        "paper_bgcolor": "#07111f",
        "plot_bgcolor": "#07111f",
        "margin": {"l": 0, "r": 0, "t": 48, "b": 0},
        "scene": {
            "bgcolor": "#07111f",
            "xaxis": {
                "title": "X (m, 0-20 km)",
                "gridcolor": "#24334d",
                "color": "#c7d4ea",
                "range": [0, 20000],
            },
            "yaxis": {
                "title": "Y (m, 0-20 km)",
                "gridcolor": "#24334d",
                "color": "#c7d4ea",
                "range": [0, 20000],
            },
            "zaxis": z_axis_config,
            "aspectmode": "data",
        },
        "legend": {"orientation": "h", "y": -0.08, "font": {"color": "#dbe7f8"}},
        "font": {"color": "#dbe7f8"},
        "annotations": [
            {
                "text": (
                    f"Source: {env_info['source_position'][0]:.3f} km, {env_info['source_position'][1]:.3f} km, "
                    f"{env_info['source_position'][2]:.1f} m | "
                    f"Receiver depth: {env_info['receiver']['depth_m']:.1f} m"
                ),
                "x": 0.01,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"color": "#a7bddb", "size": 12},
            }
        ],
    }


def build_scene_traces(
    env_path: Path,
    env_info: Dict[str, Any],
    bty_info: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Build Plotly 3D traces for source, receiver sector, beam fan, and terrain."""

    source_x_m = env_info["source_position"][0] * 1000.0
    source_y_m = env_info["source_position"][1] * 1000.0
    source_z_m = -env_info["source_position"][2]

    receiver = env_info["receiver"]
    beam = env_info["beam"]

    traces: List[Dict[str, Any]] = []

    if bty_info and bty_info.get("x_grid") and bty_info.get("y_grid") and bty_info.get("depth_grid"):
        traces.append(
            {
                "type": "surface",
                "name": "Bottom terrain",
                "x": bty_info["x_grid"],
                "y": bty_info["y_grid"],
                "z": bty_info["depth_grid"],
                "showscale": False,
                "opacity": 0.35,
                "colorscale": [[0, "#443322"], [1, "#8d6e63"]],
                "hoverinfo": "skip",
            }
        )

    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers+text",
            "name": "Source / UUV",
            "x": [source_x_m],
            "y": [source_y_m],
            "z": [source_z_m],
            "text": ["Source"],
            "textposition": "top center",
            "marker": {"size": 7, "color": "#ff6b6b", "symbol": "diamond"},
        }
    )

    traces.extend(_build_receiver_sector_traces(source_x_m, source_y_m, receiver))
    traces.extend(_build_beam_fan_traces(source_x_m, source_y_m, source_z_m, receiver, beam))

    return traces


def parse_env_file(env_path: str | Path) -> Dict[str, Any]:
    """Parse geometry and run parameters from a Bellhop env file."""

    path = Path(env_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    info: Dict[str, Any] = {
        "meta": {
            "title": _parse_title(lines),
            "frequency_hz": _parse_first_float(_find_line(lines, "FREQ (Hz)")),
            "nmedia": _parse_first_int(_find_line(lines, "NMEDIA")),
            "step": _numbers_from_line(_find_line(lines, "STEP (m)")),
        },
        "source_position": (
            _parse_first_float(_find_line(lines, "x coordinate of source (km)")),
            _parse_first_float(_find_line(lines, "y coordinate of source (km)")),
            _parse_first_float(_find_line(lines, "SD(1:NSD)")),
        ),
        "receiver_position": (
            _parse_first_float(_find_line(lines, "x coordinate of source (km)")),
            _parse_first_float(_find_line(lines, "y coordinate of source (km)")),
            _parse_first_float(_find_line(lines, "RD(1:NRD)")),
        ),
        "receiver": {
            "depth_m": _parse_first_float(_find_line(lines, "RD(1:NRD)")),
            "range_count": _parse_first_int(_find_line_regex(lines, r"!\s*NR\b")),
            "r_min_km": _numbers_from_line(_find_line(lines, "R(1:NR ) (km)"))[0],
            "r_max_km": _numbers_from_line(_find_line(lines, "R(1:NR ) (km)"))[1],
            "bearing_start_deg": _numbers_from_line(_find_line(lines, "bearing angles (degrees)"))[0],
            "bearing_end_deg": _numbers_from_line(_find_line(lines, "bearing angles (degrees)"))[1],
        },
        "beam": {
            "alpha_count": _numbers_from_line(_find_line(lines, "Nalpha"))[0],
            "alpha_start_deg": _numbers_from_line(_find_line(lines, "alpha1, 2 (degrees)"))[0],
            "alpha_end_deg": _numbers_from_line(_find_line(lines, "alpha1, 2 (degrees)"))[1],
            "beta_count": _numbers_from_line(_find_line(lines, "Nbeta"))[0],
            "beta_start_deg": _numbers_from_line(_find_line(lines, "beta1, beta2"))[0],
            "beta_end_deg": _numbers_from_line(_find_line(lines, "beta1, beta2"))[1],
            "n_theta": _numbers_from_line(_find_line(lines, "Ntheta"))[0],
        },
    }
    return info


def parse_bty_file(bty_path: str | Path | None) -> Dict[str, Any] | None:
    """Parse an R-format Bellhop bathymetry file into grid arrays."""

    if not bty_path:
        return None

    path = Path(bty_path)
    if not path.exists():
        return None

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0].strip().strip("'") != "R":
        return None

    nx = int(lines[1])
    x_min, x_max = _numbers_from_text(lines[2])[:2]
    ny = int(lines[3])
    y_min, y_max = _numbers_from_text(lines[4])[:2]

    values: List[float] = []
    for line in lines[5:]:
        values.extend(_numbers_from_text(line))

    if len(values) < nx * ny:
        return None

    depth_grid: List[List[float]] = []
    index = 0
    for _ in range(ny):
        row = values[index : index + nx]
        depth_grid.append(row)
        index += nx

    x_coords_m = [x_min * 1000.0 + (x_max - x_min) * 1000.0 * i / max(nx - 1, 1) for i in range(nx)]
    y_coords_m = [y_min * 1000.0 + (y_max - y_min) * 1000.0 * j / max(ny - 1, 1) for j in range(ny)]

    x_grid = [x_coords_m[:] for _ in range(ny)]
    y_grid = [[value for _ in range(nx)] for value in y_coords_m]

    # Convert depth values to negative (sea bottom is below sea surface)
    for row in depth_grid:
        for i in range(len(row)):
            row[i] = -row[i]

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "depth_grid": depth_grid,
    }


def load_cache_index(cache_index_path: str | Path) -> List[CacheEntry]:
    """Load and normalize cache entries from cache_index.json."""

    path = Path(cache_index_path)
    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    entries: List[CacheEntry] = []
    if isinstance(raw, dict):
        for cache_key, value in raw.items():
            normalized = _normalize_cache_value(cache_key, value)
            if normalized and normalized.env_file_path:
                entries.append(normalized)
    return entries


def _normalize_cache_value(cache_key: str, value: Any) -> CacheEntry | None:
    if isinstance(value, str):
        return CacheEntry(cache_key=cache_key, env_file_path=value)
    if isinstance(value, dict):
        env_file_path = str(value.get("env_file_path", ""))
        source = _tuple_or_none(value.get("source_position"))
        receiver = _tuple_or_none(value.get("receiver_position"))
        return CacheEntry(
            cache_key=cache_key,
            env_file_path=env_file_path,
            source_position=source,
            receiver_position=receiver,
        )
    return None


def _find_cache_entry_by_env_id(entries: Iterable[CacheEntry], env_id: str) -> CacheEntry | None:
    for entry in entries:
        if entry.env_id == env_id:
            return entry
    return None


def _resolve_bty_path(env_path: Path) -> Path:
    candidate = env_path.with_suffix(".bty")
    if candidate.exists():
        return candidate

    matches = sorted(env_path.parent.glob("*.bty"), key=lambda item: item.name)
    if matches:
        return matches[0]
    return candidate


def _resolve_shd_plot_path(env_path: Path) -> Path:
    return env_path.with_suffix(".shdplot.json")


def _is_env_cache_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    cache_index_path = path / "cache_index.json"
    if cache_index_path.exists():
        return True

    return any(path.glob("*.env"))


def _build_receiver_sector_traces(source_x_m: float, source_y_m: float, receiver: Dict[str, Any]) -> List[Dict[str, Any]]:
    r_min_m = receiver["r_min_km"] * 1000.0
    r_max_m = receiver["r_max_km"] * 1000.0
    bearing_start = receiver["bearing_start_deg"]
    bearing_end = receiver["bearing_end_deg"]

    traces: List[Dict[str, Any]] = []
    traces.append(
        {
            "type": "surface",
            "name": "Receiver sector",
            "x": _sector_surface_x(source_x_m, r_min_m, r_max_m, bearing_start, bearing_end),
            "y": _sector_surface_y(source_y_m, r_min_m, r_max_m, bearing_start, bearing_end),
            "z": _sector_surface_z(receiver["depth_m"], bearing_steps=24, range_steps=18),
            "opacity": 0.45,
            "showscale": False,
            "colorscale": [[0, "#3fa7ff"], [1, "#8ad6ff"]],
            "hoverinfo": "skip",
        }
    )

    boundary_rays = [r_min_m, r_max_m]
    boundary_bearings = [bearing_start, bearing_end]
    for radius in boundary_rays:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "Receiver boundary",
                "x": [source_x_m + radius * math.cos(math.radians(angle)) for angle in _linspace(bearing_start, bearing_end, 36)],
                "y": [source_y_m + radius * math.sin(math.radians(angle)) for angle in _linspace(bearing_start, bearing_end, 36)],
                "z": [-receiver["depth_m"] for _ in range(36)],
                "line": {"color": "#8ad6ff", "width": 3},
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )

    for bearing in boundary_bearings:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "Receiver edge",
                "x": [source_x_m + radius * math.cos(math.radians(bearing)) for radius in _linspace(r_min_m, r_max_m, 18)],
                "y": [source_y_m + radius * math.sin(math.radians(bearing)) for radius in _linspace(r_min_m, r_max_m, 18)],
                "z": [-receiver["depth_m"] for _ in range(18)],
                "line": {"color": "#63c1ff", "width": 2, "dash": "dot"},
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )

    # Add receiver point marker (like Source marker) showing receiver position
    receiver_center_r_m = (r_min_m + r_max_m) / 2.0
    receiver_center_bearing = (bearing_start + bearing_end) / 2.0
    receiver_center_x = source_x_m + receiver_center_r_m * math.cos(math.radians(receiver_center_bearing))
    receiver_center_y = source_y_m + receiver_center_r_m * math.sin(math.radians(receiver_center_bearing))
    receiver_center_z = -receiver["depth_m"]
    
    receiver_label = (
        f"Receiver\n"
        f"RD: {receiver['depth_m']:.0f}m\n"
        f"r: {receiver['r_min_km']:.1f}-{receiver['r_max_km']:.1f}km\n"
        f"θ: {bearing_start:.0f}°-{bearing_end:.0f}°"
    )
    
    # Receiver point marker (with marker symbol and text)
    traces.append({
        "type": "scatter3d",
        "mode": "markers+text",
        "name": "Receiver / Hydrophone Array",
        "x": [receiver_center_x],
        "y": [receiver_center_y],
        "z": [receiver_center_z],
        "text": [receiver_label],
        "textposition": "top center",
        "marker": {"size": 7, "color": "#4cc9f0", "symbol": "diamond"},
        "showlegend": True,
    })

    return traces


def _build_beam_fan_traces(
    source_x_m: float,
    source_y_m: float,
    source_z_m: float,
    receiver: Dict[str, Any],
    beam: Dict[str, Any],
) -> List[Dict[str, Any]]:
    beam_length_m = max(receiver["r_max_km"] * 1000.0 * 1.15, 5000.0)
    alpha_start = beam["alpha_start_deg"]
    alpha_end = beam["alpha_end_deg"]
    beta_start = beam["beta_start_deg"]
    beta_end = beam["beta_end_deg"]

    corner_angles = [
        (alpha_start, beta_start),
        (alpha_start, beta_end),
        (alpha_end, beta_start),
        (alpha_end, beta_end),
    ]
    corner_points = [
        _beam_endpoint(source_x_m, source_y_m, source_z_m, beam_length_m, alpha, beta)
        for alpha, beta in corner_angles
    ]

    traces: List[Dict[str, Any]] = []
    for alpha, beta in [(alpha_start, beta_start), (alpha_start, beta_end), (alpha_end, beta_start), (alpha_end, beta_end)]:
        x_end, y_end, z_end = _beam_endpoint(source_x_m, source_y_m, source_z_m, beam_length_m, alpha, beta)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "Beam edge",
                "x": [source_x_m, x_end],
                "y": [source_y_m, y_end],
                "z": [source_z_m, z_end],
                "line": {"color": "#ffb347", "width": 4},
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )

    center_x, center_y, center_z = _beam_endpoint(
        source_x_m,
        source_y_m,
        source_z_m,
        beam_length_m,
        (alpha_start + alpha_end) / 2.0,
        (beta_start + beta_end) / 2.0,
    )
    traces.append(
        {
            "type": "scatter3d",
            "mode": "lines",
            "name": "Beam center",
            "x": [source_x_m, center_x],
            "y": [source_y_m, center_y],
            "z": [source_z_m, center_z],
            "line": {"color": "#ffd166", "width": 6},
            "hoverinfo": "skip",
        }
    )

    traces.append(
        {
            "type": "mesh3d",
            "name": "Beam fan",
            "x": [source_x_m] + [point[0] for point in corner_points],
            "y": [source_y_m] + [point[1] for point in corner_points],
            "z": [source_z_m] + [point[2] for point in corner_points],
            "i": [0, 0, 0, 0],
            "j": [1, 2, 3, 4],
            "k": [2, 3, 4, 1],
            "opacity": 0.22,
            "color": "#ffb347",
            "hoverinfo": "skip",
            "showlegend": False,
        }
    )

    return traces


def _beam_endpoint(
    source_x_m: float,
    source_y_m: float,
    source_z_m: float,
    distance_m: float,
    alpha_deg: float,
    beta_deg: float,
) -> Tuple[float, float, float]:
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    x = source_x_m + distance_m * math.cos(alpha) * math.cos(beta)
    y = source_y_m + distance_m * math.cos(alpha) * math.sin(beta)
    z = source_z_m - distance_m * math.sin(alpha)
    return x, y, z


def _sector_surface_x(source_x_m: float, r_min_m: float, r_max_m: float, bearing_start: float, bearing_end: float) -> List[List[float]]:
    radii = _linspace(r_min_m, r_max_m, 18)
    bearings = _linspace(bearing_start, bearing_end, 24)
    return [[source_x_m + radius * math.cos(math.radians(bearing)) for bearing in bearings] for radius in radii]


def _sector_surface_y(source_y_m: float, r_min_m: float, r_max_m: float, bearing_start: float, bearing_end: float) -> List[List[float]]:
    radii = _linspace(r_min_m, r_max_m, 18)
    bearings = _linspace(bearing_start, bearing_end, 24)
    return [[source_y_m + radius * math.sin(math.radians(bearing)) for bearing in bearings] for radius in radii]


def _sector_surface_z(depth_m: float, bearing_steps: int, range_steps: int) -> List[List[float]]:
    return [[-depth_m for _ in range(bearing_steps)] for _ in range(range_steps)]


def _parse_title(lines: List[str]) -> str:
    if not lines:
        return ""
    first = lines[0]
    if "!" in first:
        first = first.split("!")[0]
    return first.strip().strip("'")


def _find_line(lines: List[str], marker: str) -> str:
    for line in lines:
        if marker in line:
            return line
    raise ValueError(f"Missing env marker: {marker}")


def _find_line_regex(lines: List[str], pattern: str) -> str:
    compiled = re.compile(pattern)
    for line in lines:
        if compiled.search(line):
            return line
    raise ValueError(f"Missing env marker matching regex: {pattern}")


def _numbers_from_line(line: str) -> List[float]:
    prefix = line.split("!")[0]
    prefix = prefix.split("/")[0]
    return [float(item) for item in re.findall(r"[-+]?\d*\.?\d+", prefix)]


def _numbers_from_text(text: str) -> List[float]:
    return [float(item) for item in re.findall(r"[-+]?\d*\.?\d+", text)]


def _parse_first_float(line: str) -> float:
    values = _numbers_from_line(line)
    if not values:
        raise ValueError(f"No numeric value found in line: {line}")
    return float(values[0])


def _parse_first_int(line: str) -> int:
    return int(round(_parse_first_float(line)))


def _tuple_or_none(values: Any) -> Tuple[float, float, float] | None:
    if isinstance(values, list) and len(values) == 3:
        try:
            return float(values[0]), float(values[1]), float(values[2])
        except (TypeError, ValueError):
            return None
    return None


def _tuple_to_list(values: Tuple[float, float, float] | None) -> List[float]:
    if values is None:
        return []
    return [float(values[0]), float(values[1]), float(values[2])]


def _linspace(start: float, end: float, count: int) -> List[float]:
    if count <= 1:
        return [start]
    step = (end - start) / (count - 1)
    return [start + step * index for index in range(count)]
