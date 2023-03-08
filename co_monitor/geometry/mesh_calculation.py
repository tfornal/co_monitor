import numpy as np
from pyvista import PolyData, Plotter
from scipy.spatial import ConvexHull

from .json_reader import read_json_file
from .rotation_matrix import rotation_matrix


class CuboidMesh:
    """Creates mesh of the observed plasma volume. Nominally the interval used for plasma meshing is 10 mm.

    As a first step it creates the representation of a plasma vertices out of the
    coordinates given in the optics_coordinates.py file. Based on those information
    the cube dimensions/coordinates are calculated in that way, that the plasma
    volume is fully placed inside the center of this cube. The interior of the cube
    is meshed and the fraction of the cubes's interior grid representing the plasma volume
    is exposed and returned. Nominal vaalue of cuboid_dimensions is: [1350, 800, 250].

    Plots a visualization of meshed plasma volume.
    Possibly save calculated mesh coordinates in *.txt.
    """

    def __init__(
        self,
        distance_between_points: int = 10,
        cuboid_dimensions: list = [
            1350,
            800,
            250,
        ],  ### this size covers the whole plasma volume observed by the system
        plot: bool = False,
        savetxt: bool = False,
    ):
        """Constructor of Plasma Mesh class.

        Parameters
        ----------
        distance_between_points : int, optional
            Integer representing required distance between plasma points (in milimeters), by default 10
        cuboid_dimensions : list, optional
            Array of 3 integers representing dimension of investigated cuboid (x,y,z); nominal, by default [1350, 800, 250]
        plot : bool, optional
           Plots 3D represenetation of investigated plasma volume, by default False
        savetxt : bool, optional
            saves the calculated plasma coordinates to *.txt file, by default False
        """
        assert 2 * distance_between_points <= min(
            cuboid_dimensions
        ), "Density of points is too low!"
        self.distance_between_points = distance_between_points
        self.x, self.y, self.z = cuboid_dimensions
        self.loaded_file = read_json_file()
        self.central_plasma_coord = self.get_coordinate()
        self.outer_cube_mesh = self.mesh_outer_cube()
        if plot:
            self.visualization()
        if savetxt:
            self.savetxt_cuboid_coordinates()

    def get_coordinate(self) -> np.ndarray:
        """
        Get the coordinates (1 x 3 array) representing the central point of the observed plasma volume.
        """
        port_vertices = np.array(self.loaded_file["plasma"]["central point"])
        return port_vertices

    def calculate_cubes_edge_length(self, coord_range: np.ndarray) -> float:
        """
        Calculate the edge length of a cube based on its coordinate range.

        Parameters
        ----------
        coord_range : numpy.ndarray
            An array of shape (2,) representing the minimum and maximum coordinates of the cube.
        """
        edge_length = abs(coord_range[0] - coord_range[1])

        return edge_length

    def calculate_interval(self, coord_range: np.ndarray) -> int:
        """
        Calculate the number of intervals/slices used for meshing the cube's volume.

        Parameters
        ----------
        coord_range : numpy.ndarray
            An array of shape (2,) representing the minimum and maximum coordinates of the cube.
        """
        edge_length = self.calculate_cubes_edge_length(coord_range)
        interval = int(edge_length // self.distance_between_points)
        return interval

    def set_cube_ranges(self) -> np.ndarray:
        """
        Determine the coordinate ranges of the outer cube along all three axes (x, y, z).
        """

        x_range = np.array(
            [
                self.central_plasma_coord[0] - self.x / 2,
                self.central_plasma_coord[0] + self.x / 2,
            ]
        )

        y_range = np.array(
            [
                self.central_plasma_coord[1] - self.y / 2,
                self.central_plasma_coord[1] + self.y / 2,
            ]
        )

        z_range = np.array(
            [
                self.central_plasma_coord[2] - self.z / 2,
                self.central_plasma_coord[2] + self.z / 2,
            ]
        )

        outer_cubes_coordinates = np.vstack((x_range, y_range, z_range))
        return outer_cubes_coordinates

    def mesh_outer_cube(self) -> np.ndarray:
        """
        Creates matrix of the outer cube mesh with a specified distance between points.
        """
        # Determine the range of the x, y, and z coordinates of the cube
        cuboid_coordinates = self.set_cube_ranges()
        x_range, y_range, z_range = cuboid_coordinates

        # Evenly space out points within each range
        x = np.linspace(
            x_range[0], x_range[1], self.calculate_interval(x_range), endpoint=True
        )
        y = np.linspace(
            y_range[0], y_range[1], self.calculate_interval(y_range), endpoint=True
        )
        z = np.linspace(
            z_range[0], z_range[1], self.calculate_interval(z_range), endpoint=True
        )

        x, y, z = np.meshgrid(x, y, z)
        meshed_cuboid = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

        # Shift the mesh so that it is centered on the central plasma coordinate
        meshed_cuboid = meshed_cuboid - self.central_plasma_coord

        # Determine the orientation vector and rotate the mesh based on this vector
        orientation_vector = np.array([1, 0, 0])
        ovec = orientation_vector / (np.linalg.norm(orientation_vector))
        cylvec = np.array([0, 1, 0])
        oaxis = np.cross(ovec, cylvec)
        R = rotation_matrix(1, oaxis)
        rotated_outer_cube_mesh = meshed_cuboid.dot(R)

        # Shift the rotated mesh to its desired location
        meshed_cuboid = (
            rotated_outer_cube_mesh + self.central_plasma_coord + [50, 25, 0]
        )
        print(
            f"Number of points in the considered volume: {len(meshed_cuboid)} points."
        )
        return meshed_cuboid

    def polyhull_faces(self, calculation_input_points: np.ndarray) -> PolyData:
        """
        Calculates the faces of a polyhull from an array of points in a 3D space.

        Parameters
        ----------
        calculation_input_points : np.ndarray, shape (n_points, 3)
            The array representing the points in 3D space for which the polyhull should be calculated.

        Returns
        -------
        poly : PolyData
            A PolyData object representing the polyhull, with the `points` attribute set to `calculation_input_points`,
            and the `faces` attribute set to the faces of the polyhull.
        """

        hull = ConvexHull(calculation_input_points)
        faces = np.column_stack(
            (np.ones((len(hull.simplices), 1), dtype=int) * 3, hull.simplices)
        ).flatten()
        poly = PolyData(hull.points, faces)
        return poly

    def visualization(self):
        """
        Plots the results of the calculated plasma/cuboid volume.
        """
        fig = Plotter()
        fig.set_background("black")
        fig.add_mesh(
            self.polyhull_faces(self.outer_cube_mesh), color="blue", opacity=0.1
        )
        fig.add_mesh(
            self.outer_cube_mesh,
            color="white",
            opacity=0.04,
            render_points_as_spheres=True,
        )
        fig.add_mesh(
            self.central_plasma_coord,
            color="red",
            point_size=50,
            render_points_as_spheres=True,
        )
        fig.show()

    def savetxt_cuboid_coordinates(self):
        """
        Save plasma coordinates to the txt file.
        """
        np.savetxt(
            "plasma_coordinates.txt",
            self.outer_cube_mesh,
            fmt="%.4e",
            header=f"Coordinates X[m], Y[m], Z[m];\nSingle block size: {self.distance_between_points} mm",
        )
        print("Coordinates successfully saved!")


if __name__ == "__main__":
    distance_between_points = 10
    cuboid_dimensions = [1350, 800, 50]
    CuboidMesh(distance_between_points, cuboid_dimensions, plot=True, savetxt=True)
