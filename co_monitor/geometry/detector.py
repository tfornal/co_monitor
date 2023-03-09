import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

from co_monitor.geometry.json_reader import read_json_file


class Detector:
    """A class to represent a detector object."""

    def __init__(self, element: str, plot=False):
        """It is used to create numerical representation (hull made out of vertices) of the selected detector for observation of B, C, N and O channels.

        Parameters
        ----------
        element : str
            takes as argument on of the considered elements: B, C, N or O
        plot : bool, optional
            If True -> create 3D visualization, by default False
        """
        self.element = element
        self.loaded_file = read_json_file()
        self.vertices = self.get_vertices(self.element)
        self.orientation_vector = self.get_orientation_vector(self.element)
        self.spatial_det_coordinates = self.create_thick_det(self.vertices)
        self.poly_det = self.make_detectors_surface()

        if plot:
            self.plotter()

    def get_vertices(self, element: str) -> np.ndarray:
        """Reads coordinates of selected detector.

        Parameters
        ----------
        element : str
            Element name for which the respective detector is chosen.

        Returns
        -------
        numpy.ndarray
            Stack of detector's vertices, with shape (4, 3).

        Raises
        ------
        AssertionError
            If the shape of the returned coordinates is not (4, 3).

        """
        det_coordinates = [
            self.loaded_file["detector"]["element"][element]["vertex"][vertex]
            for vertex in self.loaded_file["detector"]["element"][element]["vertex"]
        ]
        det_coordinates = np.vstack(det_coordinates)
        return det_coordinates

    def get_orientation_vector(self, element: str) -> np.ndarray:
        """
        Reads orientation vector of selected detector.

        Parameters
        ----------
        element : str
            Element name for which the orientation vector of respective detector is chosen.

        Returns
        -------
        np.ndarray
            Stack of detector's orientation vector.
        """
        orientation_vector = [
            self.loaded_file["detector"]["element"][element]["orientation vector"]
        ]

        orientation_vector = np.vstack(orientation_vector)
        assert orientation_vector.shape == (
            1,
            3,
        ), "Orientation vector should consist of 1 point."
        return orientation_vector

    def create_thick_det(self, vertices_coordinates: np.ndarray) -> np.ndarray:
        """
        Add thickness to a detector.

        Parameters
        ----------
        vertices_coordinates : np.ndarray
            An array of shape (4, 3) representing the stack of the detector's vertices.

        Returns
        -------
        np.ndarray
            An array of shape (8, 3) representing the stack of the spatial detector's vertices with added thickness.

        Raises
        ------
        AssertionError
            If the resulting stack of spatial detector's vertices does not have shape (8, 3).

        """
        det_vertices_with_depth = np.concatenate(
            (
                vertices_coordinates + self.orientation_vector / 2,
                vertices_coordinates - self.orientation_vector / 2,
            )
        )
        assert det_vertices_with_depth.shape == (8, 3), "Wrong number of vertices!"
        return det_vertices_with_depth

    def make_detectors_surface(self) -> pv.PolyData:
        """
        Create surface geometry representing the detector object.

        Returns
        -------
        pv.PolyData
            The surface geometry of the detector object.
        """
        hull = ConvexHull(self.spatial_det_coordinates)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)
        return poly

    def plotter(self):
        """Plots 3D representaiton of calculated detector hull."""
        detector = self.make_detectors_surface()
        fig = pv.Plotter()
        fig.set_background("black")
        fig.add_mesh(detector, color="yellow", opacity=0.99)
        fig.show()


if __name__ == "__main__":

    def plot_all_detectors():
        """Helper function to plot all detectors at once."""

        fig = pv.Plotter()
        for element in ["B", "C", "N", "O"]:
            fig.set_background("black")
            det = Detector(element, plot=False)
            detector = det.make_detectors_surface()
            fig.add_mesh(detector, color="yellow", opacity=0.99)
        fig.show()

    # plot_all_detectors()
    det = Detector("B", plot=True)
    # x = Detector.read_json_file()
    # print(x)
