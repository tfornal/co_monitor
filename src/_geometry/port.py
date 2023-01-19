import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from scipy.spatial import ConvexHull, Delaunay
import json
from pathlib import Path


class Port:
    """_summary_"""

    def __init__(self):
        """_summary_"""
        self.port_coordinates = self.get_coordinates()
        self.vertices_coordinates = self.port_coordinates[:-1]
        self.orientation_vector = self.port_coordinates[-1]
        self.spatial_port_coord = self.calculate_port_thickness(
            self.vertices_coordinates[:4]
        )
        self.port_hull = self.calculate_port_hull()

    def get_coordinates(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        ## TODO absolute path readout under Linux machine; - trzeba dodac /"src"/"_geometry" zeby poprawnie odczytal;
        ## w windowsie nie jest to potrzebne
        f = open(Path(__file__).parent.resolve() / "coordinates.json")
        data = json.load(f)
        port_coordinates = np.zeros([16, 3])
        for nr, point in enumerate(data["port"]):
            port_coordinates[nr] = data["port"][point]
        return port_coordinates

    def calculate_port_thickness(self, vertices_coordinates):
        """_summary_

        Args:
            vertices_coordinates (_type_): _description_

        Returns:
            _type_: _description_
        """
        det_vertices_with_depth = np.concatenate(
            (
                self.vertices_coordinates[:4] + self.orientation_vector,
                self.vertices_coordinates[:4] - self.orientation_vector,
            )
        ).reshape(8, 3)
        return det_vertices_with_depth

    def calculate_port_hull(self):
        """Creates 3D representation of port as polyhull.

        Returns:
            _type_: _description_
        """
        hull = ConvexHull(self.spatial_port_coord)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)
        return poly


if __name__ == "__main__":
    port = Port()
    fig = pv.Plotter()
    fig.set_background("black")
    fig.add_mesh(port.port_hull, color="yellow", opacity=0.9)
    fig.add_mesh(port.spatial_port_coord, color="r")
    fig.show()
