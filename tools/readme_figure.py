"""Make docs/img/warp.png for the README.

Run from the repository root with wxy and matplotlib installed:

    python tools/readme_figure.py

The same figure as wxylib/r's README, made from Python.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wxy

# A global lon/lat grid at 0.5 degrees: a checkerboard of 15-degree cells
# with a wave on top, so every curve in the warped panels is a graticule
# line. Row 0 is the north edge.
lon = np.arange(-179.75, 180, 0.5)
lat = np.arange(89.75, -90, -0.5)
lo, la = np.meshgrid(lon, lat)
z = (np.floor(la / 15) + np.floor(lo / 15)) % 2 + 0.35 * np.sin(lo * np.pi / 30) * np.cos(
    la * np.pi / 30
)
ll = (-180, 180, -90, 90)

targets = [
    ("Mollweide", "+proj=moll +datum=WGS84", (-1.81e7, 1.81e7, -9.1e6, 9.1e6), (720, 360)),
    (
        "North polar Lambert azimuthal",
        "+proj=laea +lat_0=90 +lon_0=0 +datum=WGS84",
        (-9e6, 9e6, -9e6, 9e6),
        (480, 480),
    ),
    (
        "South polar stereographic (EPSG:3031 parameters)",
        "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +datum=WGS84",
        (-6e6, 6e6, -6e6, 6e6),
        (480, 480),
    ),
]

fig, axes = plt.subplots(2, 2, figsize=(12.3, 7.7), dpi=130)
panels = [("Source: longitude/latitude, 720 x 360", z)]
for title, crs, extent, dim in targets:
    panels.append((title, wxy.warp(z, ll, "EPSG:4326", extent, dim, crs, "bilinear")))

for ax, (title, m) in zip(axes.flat, panels):
    # imshow draws row 0 at the top, which is north here
    ax.imshow(m, cmap="viridis", interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()

fig.tight_layout()
fig.savefig("docs/img/warp.png", facecolor="white")
