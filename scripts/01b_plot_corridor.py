#!/usr/bin/env python3
"""Draw a quick corridor map: walk graph + landmarks (diagnostic only)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.graph import (  # noqa: E402
    AKADEMICHESKAYA_LON_LAT,
    HOME_LON_LAT,
    MISIS_LON_LAT,
    OKTYABRSKAYA_LON_LAT,
    load_graph,
)

OUT = ROOT / "data" / "processed" / "corridor_map.png"


def to_xy(transformer, lon_lat):
    return transformer.transform(lon_lat[0], lon_lat[1])


def main():
    g = load_graph()
    crs = g.graph["crs"]
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    fig, ax = plt.subplots(figsize=(8, 10))
    # draw edges (thin)
    xs, ys = [], []
    for u, v, _k, data in g.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is not None:
            coords = list(geom.coords)
            ax.plot([c[0] for c in coords], [c[1] for c in coords],
                    color="#9bb7c9", linewidth=0.3, zorder=1)

    points = {
        "MISIS (Leninsky, 4)": MISIS_LON_LAT,
        "Октябрьская": OKTYABRSKAYA_LON_LAT,
        "Дом": HOME_LON_LAT,
        "Академическая": AKADEMICHESKAYA_LON_LAT,
    }
    colors = {
        "MISIS (Leninsky, 4)": "#c45c26",
        "Октябрьская": "#1f6f8b",
        "Дом": "#2f8f4e",
        "Академическая": "#6b4c9a",
    }
    for label, lon_lat in points.items():
        x, y = to_xy(transformer, lon_lat)
        ax.scatter([x], [y], s=70, color=colors[label], zorder=3, edgecolors="white")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6),
                    fontsize=9, color="#1c2430")

    # corridor axis home → MISIS
    hx, hy = to_xy(transformer, HOME_LON_LAT)
    mx, my = to_xy(transformer, MISIS_LON_LAT)
    ax.plot([hx, mx], [hy, my], color="#c45c26", linewidth=1.2, linestyle="--",
            alpha=0.6, zorder=2, label="ось коридора")

    ax.set_aspect("equal")
    ax.set_title("Коридор МИСИС ↔ дом: пешеходный граф (OSM walk)", fontsize=12)
    ax.set_xlabel(f"X, м ({crs})")
    ax.set_ylabel(f"Y, м ({crs})")
    ax.legend(loc="lower left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
