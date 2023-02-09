import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

from json_reader import read_json_file
from rotation_matrix import rotation_matrix


class RadiationShield:
    """Class for creating 3D representation of ECRH protective shields against stray radiation."""

    def __init__(self, chamber_position: str, selected_shield: str):
        """
        Parameters
        ----------
        chamber_position : str
            The position of the chamber.
        selected_shield : str
            The selected shield.
        """
        self.chamber_position = chamber_position
        self.selected_shield = selected_shield
        self.loaded_file = read_json_file()
        self.shields_coordinates = self.get_coordinates()
        self._init_shield_coord()
        self.vertices_coordinates = self.make_cyllinder()
        self.radiation_shield = self.make_shield()

    def _init_shield_coord(self):
        """Retrieve radiation shield's coordinates."""

        self.radius_central_point = self.shields_coordinates["central point"]
        self.radius = self.shields_coordinates["radius"]
        self.orientation_vector = self.shields_coordinates["orientation vector"]

    def get_coordinates(self) -> np.ndarray:
        """
        Get the coordinates of all shield vertices.

        Returns
        -------
        numpy.ndarray
            An array with `n` points representing shield vertices (rows) and 3 columns (representing `x`, `y`, `z`).

        """
        shield_coordinates = self.loaded_file["ECRH shield"][
            f"{self.chamber_position}"
        ][f"{self.selected_shield}"]

        return shield_coordinates

    def make_cyllinder(self, length=1, nlength=2, alpha=360, nalpha=360) -> np.ndarray:
        """
        Create a numpy array of vertices coordinates for the cyllindrical shield.

        Parameters
        ----------
        length : float, optional
            Length of the cyllinder based on which the shield is created, by default 1.
        nlength : int, optional
            Number of points along the length, by default 2.
        alpha : float, optional
            Angle of the shield, by default 360.
        nalpha : int, optional
            Number of points along the angle, by default 360.

        Returns
        -------
        numpy.ndarray
            An array with `n` points representing shield vertices (rows) and 3 columns (representing `x`, `y`, `z`).

        """
        I = np.linspace(0, length, nlength)
        if int(alpha) == 360:
            A = np.linspace(0, alpha, num=nalpha, endpoint=False) / 180 * np.pi
        else:
            A = np.linspace(0, alpha, num=nalpha) / 180 * np.pi

        X = self.radius * np.cos(A)
        Y = self.radius * np.sin(A)

        px = np.repeat(X, nlength)
        py = np.repeat(Y, nlength)
        pz = np.tile(I, nalpha)
        points = np.vstack((pz, px, py)).T

        ovec = self.orientation_vector / (np.linalg.norm(self.orientation_vector))
        cylvec = np.array([1, 0, 0])

        if np.allclose(cylvec, ovec):
            return points

        oaxis = np.cross(ovec, cylvec)
        rot = np.arccos(np.dot(ovec, cylvec))

        R = rotation_matrix(rot, oaxis)
        cylinder_points = points.dot(R)

        shifted_cylinder_points = cylinder_points + self.radius_central_point

        return shifted_cylinder_points

    def make_shield(self) -> pv.PolyData:
        """Create a vtk PolyData object from the convex hull of the shield vertices.

        Returns:
            pv.PolyData: a 3D polygonal data representation of the shield.
        """
        hull = ConvexHull(self.vertices_coordinates)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)

        return poly


if __name__ == "__main__":

    def plotter(*args):
        """Helper function to plot all radiation shields at once."""
        fig = pv.Plotter()
        fig.set_background("black")
        for protective_shield in args:
            fig.add_mesh(
                protective_shield.vertices_coordinates, color="blue", opacity=0.9
            )
            fig.add_mesh(protective_shield.radiation_shield, color="green", opacity=0.9)

        fig.show()

    rad1 = RadiationShield("upper chamber", "1st shield")
    rad2 = RadiationShield("bottom chamber", "1st shield")
    rad3 = RadiationShield("upper chamber", "2nd shield")
    rad4 = RadiationShield("bottom chamber", "2nd shield")

    plotter(rad1, rad2, rad3, rad4)
