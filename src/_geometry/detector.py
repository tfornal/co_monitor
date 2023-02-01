import numpy as np
import pyvista as pv
import json
from pathlib import Path
from scipy.spatial import ConvexHull


class Detector:
    """A class to represent a detector object.
    
    It is used to create numerical representation (hull made out of vertices) of the selected detector for observation of B, C, N and O channels. 
    
    """
    def __init__(self, element, plot = False):
        """
        Args:
            element (str): takes as argument on of the considered elements: B, C, N or O
            plot (bool, optional): If True -> create 3D visualization (default is False).
        """
        self.element = element
        self.coordinates_from_file = self.read_json_file()
        self.vertices = self.get_vertices(self.element)
        self.orientation_vector = self.get_orientation_vector(self.element)
        self.spatial_det_coordinates = self.create_thick_det(self.vertices)
        if plot:
            self.plotter()
            
    @classmethod
    def read_json_file(cls) -> dict:
        """Reads json file with set of all diagnostic coordinates."""
        
        with open(Path(__file__).parent.resolve() / "coordinates.json") as file:
            data = json.load(file)

        return data

    def get_vertices(self, element):
        """Reads coordinates of selected detector.

        Args:
            element (str): Element name for which the respective detector is chosen. 

        Returns:
            np.ndarray : Stack of detector's vertices.
        """
        det_coordinates = [self.coordinates_from_file["detector"]["element"][element]["vertex"][vertex] \
                            for vertex in self.coordinates_from_file["detector"]["element"][element]["vertex"]]
        det_coordinates = np.vstack(det_coordinates)
        assert(det_coordinates.shape == (4,3)), "Wrong number of vertices!"
        
        return det_coordinates
    
    def get_orientation_vector(self, element):
        """Reads orientation vector of selected detector.

        Args:
            element (str): Element name for which the orientation vector of respective detector is chosen. 

        Returns:
            np.ndarray: Stack of detector's vertices.
        """
        orientation_vector = [self.coordinates_from_file["detector"]["element"][element]["orientation vector"]]
        
        orientation_vector = np.vstack(orientation_vector)
        assert(orientation_vector.shape == (1,3)), "Orientation vector should consist of 1 point."
        
        return orientation_vector
        

    def create_thick_det(self, vertices_coordinates):
        """Gives the detector a thickness dimension depending on its orientation vector. 

        Args:
            vertices_coordinates (np.ndarray(4,3)): Stack of detector's vertices.

        Returns:
            _type_: Stack of spatial detector's vertices.
        """
        det_vertices_with_depth = np.concatenate(
            (
                vertices_coordinates + self.orientation_vector/2,
                vertices_coordinates - self.orientation_vector/2,
            )
        )
        assert(det_vertices_with_depth.shape == (8,3)), "Wrong number of vertices!"
        
        return det_vertices_with_depth

    def make_detectors_surface(self):
        """Creates surface geometry (poly data) representing detector object."""
        
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
        fig.add_mesh(detector, color = "yellow", opacity=0.99)
        fig.show()
        
        
if __name__ == "__main__":
    def plot_all_detectors():
        """Helper to plot all detectors at once."""
        
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