import numpy as np
import pandas as pd

import pyvista as pv
import pathlib

cwd = pathlib.Path(__file__).parent.resolve()

path = cwd / "C_plasma_coordinates-10_mm_spacing-height_30-length_20-slit_100.dat"
df = pd.read_csv(path, delimiter=";")
df.columns = [
    "idx_sel_plas_points",
    "plasma_x",
    "plasma_y",
    "plasma_z",
    "total_intensity_fraction",
]
"""Plots observed plasma volume including distribution of radiation intensity regions."""
fig = pv.Plotter()
fig.set_background("black")

intensity = df["total_intensity_fraction"]
plasma_coordinates = df.to_numpy()[:, 1:4]
point_cloud = pv.PolyData(plasma_coordinates)
point_cloud["Intensity"] = intensity
fig.add_mesh(pv.PolyData(point_cloud), point_size=8, render_points_as_spheres=True)
fig.show()
