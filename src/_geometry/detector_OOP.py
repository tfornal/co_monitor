import numpy as np
import pyvista as pv
import pathlib
import json
from scipy.spatial import ConvexHull
from sympy import Point3D, Plane


class Detector:
    """_summary_"""

    def __init__(self, element, plot=False):
        """_summary_

        Args:
            element (_type_): _description_
            plot (bool, optional): _description_. Defaults to False.
        """
        self.det_vertices_coord = self.get_all_coordinates(element)[:4]
        self.orientation_vector = self.get_all_coordinates(element)[-1]
        self.spatial_det_coordinates = self.create_thick_det(self.det_vertices_coord)

    def get_all_coordinates(self, element):
        """_summary_

        Args:
            element (_type_): _description_

        Returns:
            _type_: _description_
        """
        det_coordinates = np.zeros([5, 3])
        with open(
            pathlib.Path.cwd() / "src" / "_geometry" / "coordinates.json"
        ) as file:
            json_file = json.load(file)

            for nr, vertex in enumerate(
                json_file["detector"]["element"][element]["vertex"]
            ):
                det_coordinates[nr] = json_file["detector"]["element"][element][
                    "vertex"
                ][vertex]
            det_coordinates[-1] = json_file["detector"]["element"][element][
                "orientation vector"
            ]
        return det_coordinates

    def create_thick_det(self, vertices_coordinates):
        """_summary_

        Args:
            vertices_coordinates (_type_): _description_

        Returns:
            _type_: _description_
        """
        det_vertices_with_depth = np.concatenate(
            (
                vertices_coordinates + self.orientation_vector,
                vertices_coordinates - self.orientation_vector,
            )
        ).reshape(8, 3)
        return det_vertices_with_depth

    def make_detectors_surface(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        hull = ConvexHull(self.spatial_det_coordinates)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)
        return poly


if __name__ == "__main__":

    def plot_all_detectors():
        fig = pv.Plotter()
        for element in ["B", "C", "N", "O"]:
            fig.set_background("black")
            det = Detector(element, plot=False)
            detector = det.make_detectors_surface()
            fig.add_mesh(detector, color="yellow", opacity=0.9)
        fig.show()

    plot_all_detectors()

    # det = Detector("B", plot=True)
