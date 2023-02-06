import json
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull, Delaunay

from json_reader import read_json_file


class Collimator:
    """Creates a numerical representation of a grid collimator with the number of slits specifiec by a user.
    The 3D model of a single cuboid represents the single slit.
    """

    def __init__(
        self,
        element: str,
        closing_side: str,
        slits_number: int = 10,
        plot: bool = False,
    ):
        """
        Constructs the 3D representation of a collimator based on its cartesian coordinates.


        Parameters
        ----------
        element : str
            Element for which the representation of a grid collimator is calculated.
        closing_side : str
            Indicates which side (top or bottom) the collimator may be closed when performing simulations.
            Takes as an argument: "bottom closing side" or "top closing side".
        slits_number : int, optional
            Requires number of slits in the collimator, by default 10.
        plot : bool, optional
            Creates 3D representation of calculated collimator, by default False.
        """
        self.slits_number = slits_number
        self.element = element
        self.closing_side = closing_side
        self.loaded_file = read_json_file()
        self.collimator, self.vector_front_back = self.get_coordinates()
        print(self.vector_front_back)

        self.vector_top_bottom = np.array(self.collimator["vector_top_bottom"])
        self.A1 = np.array(self.collimator["vertex"]["A1"])
        self.A2 = np.array(self.collimator["vertex"]["A2"])
        self.B1 = np.array(self.collimator["vertex"]["B1"])
        self.B2 = np.array(self.collimator["vertex"]["B2"])
        if plot:
            self.visualization()

    def __repr__(self, *args, **kwargs):
        return f'Collimator(element="{self.element}", A={self.A1}, B={self.A2}, C={self.B1})'

    def get_coordinates(self) -> np.ndarray:
        """Returns the collimator and vector front-back coordinates for the specified element and closing side."""
        collimator = self.loaded_file["collimator"]["element"][f"{self.element}"][
            f"{self.closing_side}"
        ]
        vector_front_back = np.array(
            self.loaded_file["collimator"]["element"][f"{self.element}"][
                "vector front-back"
            ]
        )

        return collimator, vector_front_back

    def spatial_colimator(self, vertices_coordinates: np.ndarray) -> np.ndarray:
        """
        Creates a representation of an empty space between the collimator slits based on its defined coordinates.

        Parameters
        ----------
        vertices_coordinates : numpy.ndarray, shape (N, 3)
            Coordinates of the vertices of the collimator, where N is the number of vertices.

        Returns
        -------
        numpy.ndarray, shape (2N, 3)
            Coordinates of the collimator with depth.
        """
        collim_vertices_with_depth = np.concatenate(
            (
                vertices_coordinates + self.vector_front_back,
                vertices_coordinates - self.vector_front_back,
            )
        ).reshape(8, 3)

        return collim_vertices_with_depth

    def check_in_hull(
        self, points: np.ndarray, vertices_coordinates: np.ndarray
    ) -> np.ndarray:
        """
        Check if the given points are within the space defined by the vertices coordinates (its convex hull).

        Parameters
        ----------
        points : numpy.ndarray
            A 2D array of shape (n_points, n_dims) representing the points to be checked.
        vertices_coordinates : numpy.ndarray
            A 2D array of shape (n_vertices, n_dims) representing the coordinates of the vertices that define the space.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (n_points,) with the result of the in/out check for each point. The value is 1 if the point is inside the hull, -1 if outside and 0 if on the hull.
        """
        collim_vertices_with_depth = self.spatial_colimator(vertices_coordinates)
        hull = Delaunay(collim_vertices_with_depth)

        return hull.find_simplex(points) >= 0

    def read_colim_coord(self):
        """
        Calculates the comprehensive 3D array (slits_number, 8, 3) of collimator coordinates.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing the comprehensive 3D array of collimator coordinates, the crystal-side slit coordinates, and the plasma-side slit coordinates.
        """

        slit_coord_crys_side = np.empty((self.slits_number, 4, 3))
        for slit in range(self.slits_number):
            slit_coord_crys_side[slit, 0, :] = self.A1 + (
                self.vector_top_bottom * 2 * slit
            )
            slit_coord_crys_side[slit, 1, :] = (self.A1 + self.vector_top_bottom) + (
                self.vector_top_bottom * 2 * slit
            )
            slit_coord_crys_side[slit, 2, :] = self.A2 + (
                self.vector_top_bottom * 2 * slit
            )
            slit_coord_crys_side[slit, 3, :] = (
                self.A2
                + 0.001
                + self.vector_top_bottom  ### 0.0001 in order to add a depth to a surface
            ) + (self.vector_top_bottom * 2 * slit)

        slit_coord_plasma_side = slit_coord_crys_side + self.vector_front_back
        colimator_spatial = np.concatenate(
            (slit_coord_crys_side, slit_coord_plasma_side), axis=1
        )

        return colimator_spatial, slit_coord_crys_side, slit_coord_plasma_side

    def make_collimator(self, points: np.ndarray) -> pv.PolyData:
        """Creates a 3D representation of a collimator using the points array as vertices of the convex hull.

        Parameters
        ----------
        points : np.ndarray
            numerical representation of all empty spaces between the collimator slits

        Returns
        -------
        pv.PolyData
            A PolyData object of all slits - each represents empty space between in the collimator through which radiaion can pass freely
        """

        hull = ConvexHull(points)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)

        return poly

    def visualization(self):
        """
        Visualize the collimator.
        """
        fig = pv.Plotter()
        fig.set_background("black")
        (
            colimator_spatial,
            slit_coord_crys_side,
            slit_coord_plasma_side,
        ) = self.read_colim_coord()
        for slit in range(self.slits_number):
            collimator_points = self.make_collimator(colimator_spatial[slit])
            fig.add_mesh(collimator_points, color="yellow")
        fig.show()


if __name__ == "__main__":

    def plot_one_collimator():
        element = "B"
        col = Collimator(element, "top closing side", 10, plot=True)

    def plot_all_collimators():
        elements = [
            "B",
            "C",
            "N",
            "O",
        ]
        fig = pv.Plotter()
        fig.set_background("black")
        for element in elements:
            col = Collimator(element, "top closing side", 10)
            for slit in range(col.slits_number):
                (
                    colimator_spatial,
                    slit_coord_crys_side,
                    slit_coord_plasma_side,
                ) = col.read_colim_coord()
                collimator = col.make_collimator(colimator_spatial[slit])
                fig.add_mesh(collimator, color="yellow", opacity=0.9)
                fig.add_mesh(col.A1, color="red", point_size=10)
                fig.add_mesh(col.A2, color="blue", point_size=10)

            col = Collimator(element, "bottom closing side", 10)
            for slit in range(col.slits_number):
                (
                    colimator_spatial,
                    slit_coord_crys_side,
                    slit_coord_plasma_side,
                ) = col.read_colim_coord()
                collimator = col.make_collimator(colimator_spatial[slit])
                fig.add_mesh(collimator, color="red", opacity=0.9)
                fig.add_mesh(col.A1, color="red", point_size=10)
                fig.add_mesh(col.A2, color="blue", point_size=10)
        fig.show()

    plot_all_collimators()
    plot_one_collimator()
