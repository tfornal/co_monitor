import numpy as np
import pyvista as pv
import json
from scipy.spatial import ConvexHull, Delaunay
from optics_coordinates_OOP import CollimatorsCoordinates


class Collimator:
    def __init__(
        self,
        element: str,
        closing_side: str,
        slits_number: int = 10,
        plot: bool = False,
    ):
        """
        Creates a numerical representation of a grid collimator with the
        number of slits specifiec by a user.
        The 3D model of a single cuboid represents the single slit.
        In order to fulfill the requirements o

        Parameters
        ----------
        element : str
            Element for which the representation of a grid collimator is
            calculated.
        slits_number : int, optional
            Requires number of . The default is 10.
        plot : bool, optional
            Plots the 3D graph of calculated collimator. The default is False.
        """

        self.cc = CollimatorsCoordinates()
        print(self.cc)
        self.element = element
        # self.closing_side = closing_side
        self.collimators_coordinates = self.choose_element(element, closing_side)
        (
            self.vertices_coordinates,
            self.A1,
            self.B1,
            self.A2,
            self.B2,
            self.G,
            self.vector_top_bottom,
            self.vector_front_back,
        ) = self.collimators_coordinates
        self.collim_depth_vector = self.check_depth_vector(element)
        # sprawdza tyko w colim vertices with depth A1-A1-B1-B2 bez
        # przesuniec!! poprawic
        self.slits_number = slits_number
        self.visualization(plot)

    def __repr__(self, *args, **kwargs):
        return f'Collimator(element="{self.element}", A={self.A1}, B={self.A2}, C={self.B1})'

    def spatial_colimator(self, vertices_coordinates):
        """_summary_

        Args:
            vertices_coordinates (_type_): _description_

        Returns:
            _type_: _description_
        """
        collim_vertices_with_depth = np.concatenate(
            (
                vertices_coordinates + self.collim_depth_vector,
                vertices_coordinates - self.collim_depth_vector,
            )
        ).reshape(8, 3)

        return collim_vertices_with_depth

    def check_in_hull(self, points, vertices_coordinates):
        collim_vertices_with_depth = self.spatial_colimator(vertices_coordinates)
        hull = Delaunay(collim_vertices_with_depth)

        return hull.find_simplex(points) >= 0

    def choose_element(self, element: str, closing_side: str) -> tuple:

        """
        Checkout of element that needs to be investigated.

        Input - element symbol.
        Return - array of dispersive elements coordinates and orientation
                of each crystal (from its top to bottom) readed from the
                database.
        """

        vertices_dict = {
            "B": self.cc.vertices_collim_B(closing_side),
            "C": self.cc.vertices_collim_C(closing_side),
            "N": self.cc.vertices_collim_N(closing_side),
            "O": self.cc.vertices_collim_O(closing_side),
        }

        return vertices_dict.get(element)

    def check_depth_vector(self, element):
        """
        Checks the depth vector (perpendicular to the ray direction) for each
        energy channel.
        """
        if str(element) in ["B", "N"]:
            orientation = self.cc.plas_colim_orientation_vectors()[1] / 1000
        elif str(element) in ["C", "O"]:
            orientation = self.cc.plas_colim_orientation_vectors()[0] / 1000

        return orientation

    def read_colim_coord(self):
        """
        Calculates the comprehensive 3D array (slits_number, 8, 3) containing
        coordinates of the collimator.
        """
        slit_coord_crys_side = np.array([])
        # import time
        for slit in range(self.slits_number):
            # time.sleep(1)
            slit_coord_crys_side = np.append(
                slit_coord_crys_side, (self.A1 + (self.vector_top_bottom * 2 * slit))
            )
            slit_coord_crys_side = np.append(
                slit_coord_crys_side,
                (
                    (self.A1 + self.vector_top_bottom)
                    + (self.vector_top_bottom * 2 * slit)
                ),
            )
            slit_coord_crys_side = np.append(
                slit_coord_crys_side, (self.A2 + (self.vector_top_bottom * 2 * slit))
            )
            slit_coord_crys_side = np.append(
                slit_coord_crys_side,
                (
                    (self.A2 + 0.001 + self.vector_top_bottom)
                    + (self.vector_top_bottom * 2 * slit)
                ),
            )  # TODOwprowadzilem sztuczna wartosc
        breakpoint()
        print(slit_coord_crys_side)
        slit_coord_crys_side = slit_coord_crys_side.reshape(self.slits_number, 4, 3)
        slit_coord_plasma_side = (
            slit_coord_crys_side + self.vector_front_back
        ).reshape(self.slits_number, 4, 3)

        colimator_spatial = np.concatenate(
            (slit_coord_crys_side, slit_coord_plasma_side), axis=1
        )

        return colimator_spatial, slit_coord_crys_side, slit_coord_plasma_side

    def make_collimator(self, points):
        """
        Creates 3D representation of a collimator.
        """
        hull = ConvexHull(points)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)

        return poly

    def visualization(self, plot: bool) -> None:
        """
        Plots the results.
        """
        if plot == True:
            fig = pv.Plotter()
            fig.set_background("black")
            for slit in range(self.slits_number):
                (
                    colimator_spatial,
                    slit_coord_crys_side,
                    slit_coord_plasma_side,
                ) = self.read_colim_coord()
                x = self.make_collimator(colimator_spatial[slit])
                fig.add_mesh(x, color="pink")
            fig.show()
        else:
            pass


if __name__ == "__main__":
    fig = pv.Plotter()
    fig.set_background("black")
    elements = [
        # "B",
        "C",
        # "N",
        "O",
    ]

    for element in elements:
        col = Collimator(element, "top", 10, plot=False)
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

        col = Collimator(element, "bottom", 10, plot=False)
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
