import numpy as np
import pyvista as pv

from .json_reader import read_json_file
from .rotation_matrix import rotation_matrix


class DispersiveElement:
    """This class creates the curved surface of the selected dispersive element
    with given coordinates.
    """

    def __init__(self, element, crystal_height_step=10, crystal_length_step=20):
        """
        Parameters
        ----------
        element : str
            Elements symbol like B, C, N or O in order to select their coordinates
            from the "optics_coordinates" file containing all the necessary input.
        crystal_height_step : int, optional
            Accuracy of the crystals length mesh (by default 10 points).
        crystal_length_step : int, optional
            Accuracy of the crystals length mesh (by default 20 points).
        """
        self.element = element
        self.height_step = crystal_height_step
        self.length_step = crystal_length_step
        self.loaded_file = read_json_file()
        self.disp_elem_coord = self.read_coord_from_file()
        self._init_crys_coordinates()
        self.crystal_points = self._make_curved_crystal()

    def _init_crys_coordinates(self):
        """Constructor of other coordinates and parameters od respective dispersice element"""
        self.max_reflectivity = self.disp_elem_coord["max reflectivity"]
        self.crystal_central_point = np.array(
            self.disp_elem_coord["crystal central point"]
        )
        self.radius_central_point = np.array(
            self.disp_elem_coord["radius central point"]
        )
        self.AOI = np.array(self.disp_elem_coord["AOI"])
        self.A = np.array(self.disp_elem_coord["vertex"]["A"])
        self.B = np.array(self.disp_elem_coord["vertex"]["B"])
        self.C = np.array(self.disp_elem_coord["vertex"]["C"])
        self.D = np.array(self.disp_elem_coord["vertex"]["D"])
        self.radius = self.distance_between_points(self.A, self.radius_central_point)
        self.crystal_orientation_vector = self.B - self.C
        self.alpha = self.angle_between_lines(self.radius_central_point, self.A, self.B)

    def read_coord_from_file(self) -> np.ndarray:
        """Read the coordinates of a given dispersive element from a JSON file.

        Returns
        -------
        np.ndarray
            n points representing the vertices (rows) of the dispersive element and 3 columns (representing x,y,z).
        """
        disp_elem_coord = self.loaded_file["dispersive element"]["element"][
            f"{self.element}"
        ]
        return disp_elem_coord

    @staticmethod
    def distance_between_points(p1: np.ndarray, p2: np.ndarray) -> int:
        """Calculate the Euclidean distance between two points in a 3D cartesian coordinate system.

        Args:
        -------
        p1 : np.ndarray
            The first point represented as a 3 element numpy array in millimeters.
        p2 : np.ndarray
            The second point represented as a 3 element numpy array in millimeters.

        Returns:
        -------
        int
            The Euclidean distance between the two points rounded to the nearest integer, in millimeters.
        """
        squared_dist = np.sum((p1 - p2) ** 2, axis=0)
        R = np.sqrt(squared_dist).round(2)

        return int(R)

    def angle_between_lines(self, central_point, p1, p2):
        """
        Returns the angle in degrees between two lines in 3D cartesian coordinate system.

        Parameters
        ----------
        central_point : np.array
            The central point in the form of [x, y, z].
        p1 : np.array
            First point in the form of [x, y, z].
        p2 : np.array
            Second point in the form of [x, y, z].

        Returns
        -------
        angle : float
            The angle in degrees between the two lines.
        """
        central_point, p1, p2 = np.array(central_point), np.array(p1), np.array(p2)
        v1 = p1 - central_point
        v2 = p2 - central_point
        angle = np.degrees(
            np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        )
        return angle

    def _make_curved_crystal(self):
        """
        Creates a curved dispersive element/crystal with the given parameters.

        Returns
        -------
        crystal_points : numpy.ndarray
            3D array of the curved crystal's points in cartesian coordinate system.
        """
        crys_height = np.linspace(0, 20, self.height_step)  # crystal height range
        crys_lenght = (
            np.linspace(
                0,
                self.alpha,
                num=self.length_step,
            )
            / 180
            * np.pi
        )
        X = self.radius * np.cos(crys_lenght)
        Y = self.radius * np.sin(crys_lenght)
        px = np.repeat(X, self.height_step)
        py = np.repeat(Y, self.height_step)
        pz = np.tile(crys_height, self.length_step)
        crystal_points = np.vstack((pz, px, py)).T

        ovec = self.crystal_orientation_vector / (
            np.linalg.norm(self.crystal_orientation_vector)
        )
        cylvec = np.array([1, 0, 0])

        if np.allclose(cylvec, ovec):
            return crystal_points

        oaxis = np.cross(ovec, cylvec)
        rot = np.arccos(np.dot(ovec, cylvec))

        rot_matrix = rotation_matrix(rot, oaxis)
        crystal_points = crystal_points.dot(rot_matrix)

        shift = np.array(
            self.radius_central_point - self.crystal_orientation_vector / 2
        )

        angle = self.angle_between_lines(
            self.radius_central_point, self.C, crystal_points[0]
        )
        crystal_points += shift

        ### angle checkout
        angle = self.angle_between_lines(
            self.radius_central_point, self.C, crystal_points[0]
        )

        ### move to starting position
        crystal_points -= shift

        ### rotate by the calculated angle and shift again to the destination place
        rot_matrix = rotation_matrix(np.deg2rad(angle), self.crystal_orientation_vector)
        crystal_points = crystal_points.dot(rot_matrix)
        crystal_points += shift

        return crystal_points


if __name__ == "__main__":

    def plot_dispersive_elements():
        """Helper function to plot all dispersive elements at once."""
        fig = pv.Plotter()
        fig.set_background("black")
        disp_elem = ["B", "O", "N", "C"]
        for element in disp_elem:
            disp = DispersiveElement(element, 20, 80)
            crys = disp._make_curved_crystal()
            fig.add_mesh(
                np.array([disp.A, disp.B, disp.C, disp.D]),
                color="red",
                render_points_as_spheres=True,
                point_size=10,
            )

            fig.add_mesh(
                disp.C, color="yellow", render_points_as_spheres=True, point_size=10
            )
            fig.add_mesh(
                crys[0],
                color="purple",
                render_points_as_spheres=True,
                point_size=10,
            )
            fig.add_mesh(
                crys,
                color="orange",
                render_points_as_spheres=True,
                point_size=3,
            )
        fig.show()

    plot_dispersive_elements()
