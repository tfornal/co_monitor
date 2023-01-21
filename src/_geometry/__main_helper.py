import numpy as np
import pyvista as pv
import time
from pyvista import PolyData, Plotter
from scipy.spatial import ConvexHull
from scipy.spatial import ConvexHull, Delaunay
import dask.array as da
from datetime import datetime
from simulation_OOP import Simulation
from mesh_calculation_OOP import PlasmaMesh


def first_main():
    for slit in range(10, 9, -1):
        elements_list = ["B"]
        distance_between_points = 30
        starting_settings = dict(
            slits_number=slit,
            distance_between_points=distance_between_points,
            crystal_height_step=5,
            crystal_length_step=5,
            savetxt=False,
        )
        for element in elements_list:

            simul = Simulation(element, **starting_settings)
            total_plasma = simul.plasma_coordinates
            plas_points_indices = simul.plas_points_indices[
                "idx_sel_plas_points"
            ].values
            plas_points_indices = plas_points_indices.compute().tolist()
            unique_plas_points_indices = list(set(plas_points_indices))
            total_plasma_final = simul.plasma_coordinates
            plasma_coordinates = total_plasma_final[plas_points_indices]

            pm = PlasmaMesh(distance_between_points)
            loaded_coordinates = pm.outer_cube_mesh
            plasma_coordinates = da.from_array(
                loaded_coordinates[unique_plas_points_indices], chunks=(2000, 3)
            )

            def vizualization():
                fig = Plotter()
                fig.set_background("black")
                fig.add_mesh(
                    polyhull(plasma_coordinates), color="green", opacity=0.3
                )  # hull
                # fig.add_mesh(hullpoints, color = "red", opacity = 0.8) #all points
                fig.add_mesh(
                    plasma_coordinates.compute(), color="orange", opacity=0.3
                )  # all points
                fig.add_mesh(total_plasma.compute(), color="pink", opacity=0.1)
                fig.add_mesh(plasma_coordinates.compute(), color="red")

            # vizualization()

    def polyhull(calculation_input_points):

        hull = ConvexHull(calculation_input_points)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=np.int), hull.simplices)
        ).flatten()
        poly = PolyData(hull.points, faces)

        return poly


start = time.time()

if __name__ == "__main__":
    first_main()

print(f"\nExecution time is {round((time.time() - start), 2)} s")
