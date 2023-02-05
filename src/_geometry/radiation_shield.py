import json
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

from json_reader import read_json_file
from rotation_matrix import rotation_matrix


class RadiationShield:
    """Class creating 3D representation of ECRH protective shields against stray radiation."""

    def __init__(self, chamber_position: str, selected_shield: str):
        """Construct all the necessary attributes for the respective protective shield.

        Args:
            chamber_position (str): defines the position of the spectrometers vacuum chamber;
            accepts either "upper chamber" or "bottom chamber"

            shield_nr (str): defines the position of the protection shield for a given vacuum chamber;
            accepts either "1st shield" or "2nd shield"
        """
        self.chamber_position = chamber_position
        self.selected_shield = selected_shield
        self.loaded_file = read_json_file()
        self.shields_coordinates = self.get_coordinates()
        self.radius_central_point = self.shields_coordinates["central point"]
        self.radius = self.shields_coordinates["radius"]
        self.orientation_vector = self.shields_coordinates["orientation vector"]
        self.vertices_coordinates = self.make_shield(
            self.radius, self.radius_central_point, self.orientation_vector
        )
        self.radiation_shield = self.make_radiation_protection_shield()

    def get_coordinates(self) -> np.ndarray:
        """Reads coordinates of all port vertices.

        Returns:
            np.ndarray: n points representing port vertices (rows) and 3 columns (representing x,y,z)
        """
        ecrh_coordinates = self.loaded_file["ECRH shield"][f"{self.chamber_position}"][
            f"{self.selected_shield}"
        ]

        return ecrh_coordinates

    def make_shield(
        self,
        radius,
        radius_central_point,
        orientation,
        length=1,
        nlength=2,
        alpha=360,
        nalpha=360,
    ):
        """_summary_

        Args:
            radius (_type_): _description_
            radius_central_point (_type_): _description_
            orientation (_type_): _description_
            length (int, optional): _description_. Defaults to 1.
            nlength (int, optional): _description_. Defaults to 2.
            alpha (int, optional): _description_. Defaults to 360.
            nalpha (int, optional): _description_. Defaults to 360.

        Returns:
            _type_: _description_
        """
        I = np.linspace(0, length, nlength)
        if int(alpha) == 360:
            A = np.linspace(0, alpha, num=nalpha, endpoint=False) / 180 * np.pi
        else:
            A = np.linspace(0, alpha, num=nalpha) / 180 * np.pi

        X = radius * np.cos(A)
        Y = radius * np.sin(A)

        px = np.repeat(X, nlength)
        py = np.repeat(Y, nlength)
        pz = np.tile(I, nalpha)
        points = np.vstack((pz, px, py)).T

        ovec = orientation / (np.linalg.norm(orientation))
        cylvec = np.array([1, 0, 0])

        if np.allclose(cylvec, ovec):
            return points

        oaxis = np.cross(ovec, cylvec)
        rot = np.arccos(np.dot(ovec, cylvec))

        R = rotation_matrix(rot, oaxis)
        cylinder_points = points.dot(R)

        # shift of created
        shifted_cylinder_points = cylinder_points + radius_central_point

        return shifted_cylinder_points

    def make_radiation_protection_shield(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        hull = ConvexHull(self.vertices_coordinates)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)

        return poly


if __name__ == "__main__":

    def plotter(*args):

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
