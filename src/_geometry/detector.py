import numpy as np
import pyvista as pv
import json
from pathlib import Path
from scipy.spatial import ConvexHull
from sympy import Point3D, Plane


class Detector:
    """A class used to represent a detector object.
    
    It is used to create numerical representation (hull made out of vertices) 
    of the selected detector for observation of B, C, N and O channels. 
    
    """
    def __init__(self, element, plot = False):
        """
        Args:
            element (str): takes as argument on of the considered elements: 
                B, C, N or O
            plot (bool, optional): If True -> create 3D visualization.
                Defaults to False.
        """
        self.coordinates_from_file = self.read_json_file()
        self.det_vertices_coord = self.get_all_coordinates(element)[:4]
        self.orientation_vector = self.get_all_coordinates(element)[-1]
        self.spatial_det_coordinates = self.create_thick_det(self.det_vertices_coord)
        if plot:
            self.plotter()
            
            
            
        
    def read_json_file(self) -> dict:
        """Reads json file with set of all port coordinates."""
        
        file = open(Path(__file__).parent.resolve() / "coordinates.json")
        data = json.load(file)

        return data

    
    def get_all_coordinates(self, element):
        """_summary_

        Args:
            element (_type_): _description_

        Returns:
            _type_: _description_
        """
        det_coordinates = np.zeros([5, 3])
        with open(Path(__file__).parent.resolve() / "coordinates.json") as file:
            json_file = json.load(file)

            for nr, vertex in enumerate(
                json_file["detector"]["element"][element]["vertex"]
            ):
                det_coordinates[nr] = json_file["detector"]["element"][element][
                    "vertex"
                ][vertex]
            det_coordinates[-1] = json_file["detector"]["element"][element][
                "orientation vector"
            ]
            
            det_coordinates2 = [self.coordinates_data["port"]["vertices"][vertex] \
                                for point, vertex in enumerate(self.coordinates_data["port"]["vertices"])]
            
            print(det_coordinates)
        return det_coordinates

    def create_thick_det(self, vertices_coordinates):
        """_summary_

        Args:
            vertices_coordinates (_type_): _description_

        Returns:
            _type_: _description_
        """
        det_vertices_with_depth = np.concatenate(
            (
                vertices_coordinates + self.orientation_vector/2,
                vertices_coordinates - self.orientation_vector/2,
            )
        ).reshape(8, 3)
        return det_vertices_with_depth

    def make_detectors_surface(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        hull = ConvexHull(self.spatial_det_coordinates)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)
        return poly

    def plotter(self):
        """Plots selected detector"""
        detector = self.make_detectors_surface()
        fig = pv.Plotter()
        fig.set_background("black")
        fig.add_mesh(detector, color = "yellow", opacity=0.99)
        fig.show()
        
        
if __name__ == "__main__":
    def plot_all_detectors():
        fig = pv.Plotter()
        for element in ["B", "C", "N", "O"]:
            fig.set_background("black")
            det = Detector(element, plot=False)
            detector = det.make_detectors_surface()
            fig.add_mesh(detector, color="yellow", opacity=0.99)
        fig.show()
    
    # plot_all_detectors()
    det = Detector("B", plot=False)
