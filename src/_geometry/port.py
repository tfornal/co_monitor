import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

from json_reader import read_json_file


class Port:
    """Creates numerical representation of the W7-X port based on its edge vertices coordinates in  x, y, z (mm)."""

    def __init__(self, plot=False):
        """Constructs the hull of the port based on its vertices.

        Args:
            plot (bool): creates 3D visualization if set to "True"
            (default is False)
        """
        self.loaded_file = read_json_file()
        self.vertices_coordinates = self.get_vertices()
        self.orientation_vector = self.get_orientation_vector()
        self.spatial_port = self.calculate_port_thickness()
        self.port_hull = self.make_port_hull()
        if plot:
            self.plotter()

    def get_vertices(self) -> np.ndarray:
        """Reads coordinates of all port vertices.

        Returns:
            np.ndarray: n points representing port vertices (rows) and 3 columns (representing x,y,z)
        """
        port_coordinates = [
            self.loaded_file["port"]["vertex"][vertex]
            for vertex in self.loaded_file["port"]["vertex"]
        ]
        port_coordinates = np.vstack(port_coordinates)

        return port_coordinates

    def get_orientation_vector(self) -> np.ndarray:
        """Reads coordinates of orientation vector.

        Returns:
            np.ndarray: port orientation vector (x, y, z)
        """
        orientation_vector = np.array(self.loaded_file["port"]["orientation vector"])

        return orientation_vector

    def calculate_port_thickness(self) -> np.ndarray:
        """Takes np.ndarray of n points in cartesian coordinate system (mm) and creates a thick W7-X port object adding the thickness of its orientation vector.

        Returns:
            np.ndarray: set of 8 points in carthesian coordinate system representing W7-X port dimensions including its thickness.
        """
        det_vertices_with_depth = np.concatenate(
            (
                self.vertices_coordinates + np.array(self.orientation_vector) / 2,
                self.vertices_coordinates - np.array(self.orientation_vector) / 2,
            )
        )
        return det_vertices_with_depth

    def make_port_hull(self) -> pv.core.pointset.PolyData:
        """Creates surface geometry (poly data) representing port object."""

        hull = ConvexHull(self.spatial_port)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)

        return poly

    def plotter(self) -> None:
        """Plots 3D representaiton of calculated port hull."""
        fig = pv.Plotter()
        fig.set_background("black")
        fig.add_mesh(self.port_hull, color="yellow", opacity=0.9)
        fig.add_mesh(self.spatial_port, color="r")
        fig.show()


if __name__ == "__main__":
    port = Port(plot=True)
