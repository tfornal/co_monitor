import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

from geometry.json_reader import read_json_file


class Port:
    """Creates numerical representation of the W7-X port based on its edge vertices coordinates in  x, y, z (mm)."""

    def __init__(self, plot=False):
        """Constructs the hull of the port based on its vertices.

        Parameters
        ----------
        plot : bool, optional
            Creates 3D visualization if set to "True" (default is False).
        """
        self.loaded_file = read_json_file()
        self.vertices_coordinates = self.get_vertices()
        self.orientation_vector = self.get_orientation_vector()
        self.spatial_port = self.calculate_port_thickness()
        self.port_hull = self.make_port_hull()
        if plot:
            self.plotter()

    def get_vertices(self) -> np.ndarray:
        """
        Get the coordinates of all port vertices.

        Returns
        -------
        ndarray
            A n x 3 array representing the n port vertices, where each row represents a vertex and the columns represent the x, y, and z coordinates.

        """
        port_vertices = np.vstack(
            [
                self.loaded_file["port"]["vertex"][vertex]
                for vertex in self.loaded_file["port"]["vertex"]
            ]
        )
        return port_vertices

    def get_orientation_vector(self) -> np.ndarray:
        """
        Get the coordinates of the port orientation vector.

        Returns
        -------
        ndarray
            A 1 x 3 array representing the port orientation vector, where each column represents the x, y, and z coordinates.

        """
        orientation_vector = np.array(self.loaded_file["port"]["orientation vector"])
        return orientation_vector

    def calculate_port_thickness(self) -> np.ndarray:
        """
        Calculate the thick W7-X port object by adding the thickness of its orientation vector to its vertices coordinates.

        Returns
        -------
        np.ndarray
            An array of 8 points in the cartesian coordinate system representing the W7-X port dimensions including its thickness.
        """
        det_vertices_with_depth = np.concatenate(
            (
                self.vertices_coordinates + np.array(self.orientation_vector) / 2,
                self.vertices_coordinates - np.array(self.orientation_vector) / 2,
            )
        )
        return det_vertices_with_depth

    def make_port_hull(self) -> pv.core.pointset.PolyData:
        """Creates surface geometry (poly data) representing port object.

        Returns
        -------
        poly : pv.PolyData
            The poly data representing the surface geometry of the port object.
        """
        hull = ConvexHull(self.spatial_port)
        simplices = hull.simplices
        faces = np.column_stack(
            (np.ones((len(simplices), 1), dtype=int) * 3, simplices)
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
