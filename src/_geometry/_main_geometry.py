import numpy as np
import pyvista as pv
import time
from pyvista import PolyData, Plotter
from scipy.spatial import ConvexHull
import dask.array as da
from simulation import Simulation
from mesh_calculation import PlasmaMesh

elements_list = ["B", "N"]
testing_settings = dict(
    slits_number=10,
    distance_between_points=50,
    crystal_height_step=5,
    crystal_length_step=5,
    savetxt=True,
    plot=False,
)


start = time.time()

if __name__ == "__main__":
    for element in elements_list:
        simul = Simulation(element, **testing_settings)

print(f"\nExecution time is {round((time.time() - start), 2)} s")
