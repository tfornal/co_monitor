import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from scipy.spatial import ConvexHull
import json
from pathlib import Path


class RadiationShield:
    """A class to represent ECRH protective shield"""

    def __init__(self, chamber_position, shield_nr):
        """Construct all the necessary attributes for the specific protective shield.

        Args:
            chamber_position (str): defines the position of the spectrometers vacuum chamber;
            accepts either "upper chamber" or "bottom chamber"

            shield_nr (str): defines the position of the protection shield for a given vacuum chamber;
            accepts either "1st shield" or "2nd shield"
        """
        self.chamber_position = chamber_position
        self.shield_nr = shield_nr
        self.sields_coordinates = self.get_coordinates(
            self.chamber_position, self.shield_nr
        )
        self.radius_central_point = self.sields_coordinates["central point"]
        self.radius = self.sields_coordinates["radius"]
        self.orientation_vector = self.sields_coordinates["orientation vector"]
        self.vertices_coordinates = self.calculate_circular_shield(
            self.radius, self.radius_central_point, self.orientation_vector
        )
        self.poly_hull = self.make_radiation_protection_shield()

    def get_coordinates(self, chamber_position, shield_nr):
        """Get the position of spectrometers vacuum chamber position (top or bottom)
        and number of protection shield (1 or 2 int)

        Returns:
            _type_: _description_
        """
        with open(Path(__file__).parent.resolve() / "coordinates.json") as file:
            json_file = json.load(file)
            shield_coordinates = json_file["ECRH shield"][f"{chamber_position}"][
                f"{shield_nr}"
            ]
        return shield_coordinates

    def rotation_matrix_3D(self, theta, axis):
        """_summary_

        Args:
            theta (float): _description_
            axis (ndarray): 1D array containing x,y,z coordinates of axis of rotation

        Returns:
            _type_: _description_
        """
        axis = np.asarray(axis) / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a**2, b**2, c**2, d**2
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

        return np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )

    def calculate_circular_shield(
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

        R = self.rotation_matrix_3D(rot, oaxis)
        cylinder_points = points.dot(R)

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
    rad1 = RadiationShield("upper chamber", "1st shield")
    rad2 = RadiationShield("bottom chamber", "1st shield")
    rad3 = RadiationShield("upper chamber", "2nd shield")
    rad4 = RadiationShield("bottom chamber", "2nd shield")

    fig = pv.Plotter()
    fig.set_background("black")
    for protective_shield in [rad1, rad2, rad3, rad4]:
        fig.add_mesh(protective_shield.vertices_coordinates, color="blue", opacity=0.9)
        fig.add_mesh(protective_shield.poly_hull, color="green", opacity=0.9)

    fig.show()
