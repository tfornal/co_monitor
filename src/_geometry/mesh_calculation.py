import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull, Delaunay
from pyvista import PolyData, Plotter
from pyvistaqt import BackgroundPlotter
from icecream import ic
import json
from pathlib import Path


class PlasmaMesh:
    def __init__(
        self,
        distance_between_points: int = 10,
        cuboid_size: list = [1350, 800, 250],
        plot: bool = False,
        savetxt: bool = False,
    ):
        """
        Creates mesh of the observed plasma volume. Nominally the interval used for
        plasma meshing is 10 mm;

        As a first step it creates the representation of a plasma vertices out of the
        coordinates given in the optics_coordinates.py file. Based on those information
        the cube dimensions/coordinates are calculated in that way, that the plasma
        volume is fully placed inside the center of this cube. The interior of the cube
        is meshed and the fraction of the cubes's interior grid representing the plasma volume
        is exposed and returned. Nominal vaalue of cuboid_size is: [1350, 800, 250]

        Plots a visualization of meshed plasma volume.

        Possibly save calculated mesh coordinates in *.txt.

        Parameters
        ----------
        distance_between_points: INT
            Integer representing required distance between plasma points (in milimeters).
            Nominally set to 10 mm
        cuboid_size: array INT
            Array of 3 integers representing dimension of investigated cuboid (x,y,z); nominal
        nominally [1350, 800, 250]
        plot: BOOL, optional
            Plots 3D represenetation of investigated plasma volume. The default is False.
        savetxt : BOOL, optional
            saves the calculated plasma coordinates to *.txt file. The default is False.
        #"""

        assert 2 * distance_between_points <= min(
            cuboid_size
        ), "Density of points is too low!"

        self.distance_between_points = distance_between_points
        self.x_length = cuboid_size[0]
        self.y_length = cuboid_size[1]
        self.z_length = cuboid_size[2]

        with open(
            Path(__file__).parent.resolve() / "coordinates.json"  ### windows path
        ) as file:
            json_file = json.load(file)
            self.central_point_of_meshed_volume = np.array(
                json_file["plasma"]["central point"]
            )
        # self.central_point_of_meshed_volume = (
        #     Plasma().center_of_observed_plasma_volume()
        # )

        self.outer_cube_mesh = self.mesh_outer_cube()

        self.visualization(plot)
        self.savetxt_cuboid_coordinates(savetxt)

    #### excepty!!!

    def calculate_cubes_edge_length(self, coord_range):
        """ """
        print(coord_range)
        edge_length = abs(coord_range[0] - coord_range[1])
        return edge_length

    def calculate_interval(self, coord_range):
        """
        Calculates the value of interval/slices used for meshing the cube's volume.

        Returns
        -------
        interval : INT
            value representing the number of slices used for the cube's mesh computation.
        """
        edge_length = self.calculate_cubes_edge_length(coord_range)
        interval = int(edge_length // self.distance_between_points)

        return interval

    def set_cube_ranges(self):
        """
        Calculates the coordinates of all axes (x,y,z) of outer cube.

        Returns
        -------
        outer_cubes_coordinates : 2D np.array (N,3)
            Matrix representing ranges of 3D axes (by rows) describing serving
            as an input for the final cube's calculation.

        """

        x_range = np.array(
            [
                self.central_point_of_meshed_volume[0] - self.x_length / 2,
                self.central_point_of_meshed_volume[0] + self.x_length / 2,
            ]
        )

        y_range = np.array(
            [
                self.central_point_of_meshed_volume[1] - self.y_length / 2,
                self.central_point_of_meshed_volume[1] + self.y_length / 2,
            ]
        )

        z_range = np.array(
            [
                self.central_point_of_meshed_volume[2] - self.z_length / 2,
                self.central_point_of_meshed_volume[2] + self.z_length / 2,
            ]
        )

        outer_cubes_coordinates = np.vstack((x_range, y_range, z_range))

        return outer_cubes_coordinates

    ###### TODO poprawic obracanie! moze jakis plik zewnetrzny albo metode wrzucic w jakis plik zawierajacy rozne wielokrotnie uzywane metody; zapytac dominika;

    def mesh_outer_cube(self):
        """
        Mesh an outer cube with the given distance between points.

        Returns
        -------
        outer_cube_mesh : 2D np.array (N,3)
            Matrix representing coordinates of each point in a given cube.

        """
        cuboid_coordinates = self.set_cube_ranges()
        x_range, y_range, z_range = (
            cuboid_coordinates[0],
            cuboid_coordinates[1],
            cuboid_coordinates[2],
        )

        x, y, z = (
            np.linspace(
                x_range[0], x_range[1], self.calculate_interval(x_range), endpoint=True
            ),
            np.linspace(
                y_range[0], y_range[1], self.calculate_interval(y_range), endpoint=True
            ),
            np.linspace(
                z_range[0], z_range[1], self.calculate_interval(z_range), endpoint=True
            ),
        )
        x, y, z = np.meshgrid(x, y, z)
        meshed_cuboid = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

        def rotation_matrix_3D(theta, axis):
            """
            Rotation matrix based on the Euler-Rodrigues formula for matrix conversion of 3d object.

            Input - angle, axis in carthesian coordinates [x,y,z].
            Return - rotated  matrix.
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

        def shift_matrix(object_coordinates, crystal_orientation_vector):
            """
            Adds vector to 2D array (N, 3) of 3D object row by row.

            Input - 2d array of interest, vector to add.
            Return - 2d array with included vector.
            """
            coordinates_plus_vector = np.zeros([len(object_coordinates), 3])
            for row_nr, j in enumerate(object_coordinates):
                coordinates_plus_vector[row_nr, 0] = (
                    j[0] + crystal_orientation_vector[0]
                )
                coordinates_plus_vector[row_nr, 1] = (
                    j[1] + crystal_orientation_vector[1]
                )
                coordinates_plus_vector[row_nr, 2] = (
                    j[2] + crystal_orientation_vector[2]
                )

            return coordinates_plus_vector

        ### shift to center
        meshed_cuboid = shift_matrix(
            meshed_cuboid, -self.central_point_of_meshed_volume
        )
        ### get orthogonal axis and rotation
        orientation_vector = np.array([1, 0, 0])
        ovec = orientation_vector / (np.linalg.norm(orientation_vector))
        cylvec = np.array([0, 1, 0])

        oaxis = np.cross(ovec, cylvec)

        R = rotation_matrix_3D(1, oaxis)
        rotated_outer_cube_mesh = meshed_cuboid.dot(R)
        ### shift from center to the desired location
        meshed_cuboid = shift_matrix(
            rotated_outer_cube_mesh, self.central_point_of_meshed_volume + [50, 25, 0]
        )

        print(f"Number of points in the considered volume: {len(meshed_cuboid)} points.")

        return meshed_cuboid

    def polyhull(self, calculation_input_points):
        """
        Calcualtes faces of polyhull out of an array representing givenn points in a 3D space
        """

        hull = ConvexHull(calculation_input_points)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = PolyData(hull.points, faces)

        return poly

    def visualization(self, plot):

        """
        Plots the results of calculated plasma/cuboid volume.
        """
        if plot:
            fig = Plotter()
            fig.set_background("black")
            fig.add_mesh(self.polyhull(self.outer_cube_mesh), color="blue", opacity=0.1)
            fig.add_mesh(
                self.outer_cube_mesh,
                color="white",
                opacity=0.04,
                render_points_as_spheres=True,
            )
            fig.add_mesh(
                self.central_point_of_meshed_volume,
                color="red",
                point_size=10,
                render_points_as_spheres=True,
            )
            fig.show()
        # else:
        # pass

    def savetxt_cuboid_coordinates(self, savetxt):
        """
        Save plasma coordinates to the txt file.
        """
        if savetxt:
            np.savetxt(
                "plasma_coordinates.txt",
                self.plasma_coordinates,
                fmt="%.4e",
                header=f"Coordinates X[m], Y[m], Z[m];\nSingle block size: {self.distance_between_points} mm",
            )
            print("Coordinates successfully saved!")

def main():

    cuboid_size = [1350, 800, 250]
    PlasmaMesh(10, cuboid_size, plot=True, savetxt=False)


if __name__ == "__main__":
    main()
