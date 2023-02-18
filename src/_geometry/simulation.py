__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

"""
Kod musi pobierac dane z pliku wyjsciowego uzyskanego za pomoca rest api.
Step 1 - wyznaczenie objetosci jaka obserwowalby kazdy kanal.
Step 2 - Nastepnie kazdy punkt sprawdzany bylby modulem ponizej.
TODO - reflectivity readout -> not a gaussian but experimental profile
TODO - zapis do bazy danych (sql???? czy cos innego?) a nie csv!!!!!!
TODO - testy dla slits number closing top i bottom # for slit in slits_number:
TODO - poprawić nazewnictwo zapisywanych plików!
"""

from functools import reduce
from pathlib import Path
import sys

import dask.array as da
import dask.dataframe as dd
import numpy as np
from opt_einsum import contract
import pyvista as pv
from scipy.spatial import Delaunay

from collimator import Collimator
from detector import Detector
from decorators import timer
from dispersive_element import DispersiveElement
from mesh_calculation import CuboidMesh
from port import Port


@timer
class Simulation:
    def __init__(
        self,
        element,
        slits_number=10,
        distance_between_points=10,
        crystal_height_step=20,
        crystal_length_step=80,
        savetxt=False,
        plot=True,
    ):
        """
        The class runs the simulation of a geometric module of a CO Monitor.
        Calculations include ray tracing, plasma point to crystal point distance
        as well as angle of incident calculations. It includes also other obstacles
        like port and protection ECRH shields.
        """
        print(f"\nInitializing element {element}.")

        self.element = element
        self.slits_number = slits_number
        self.distance_between_points = distance_between_points
        self.crystal_height_step = crystal_height_step
        self.crystal_length_step = crystal_length_step
        self.plot = plot

        self._init_cuboid_coordinates()
        self._init_collimator()
        self._init_dispersive_element()
        self._init_port()
        self._init_detector()

        (
            self.reflected_points_location,
            self.crys_plas_data_arr,
        ) = self.calculate_radiation_reflection()
        self.selected_intersections = self.check_ray_transmission()
        self.selected_indices = self.calculate_indices()
        self.plas_points_indices = self.calculate_plasma_points_indices()
        self.distances = self.calculate_distances()
        self.angles_of_incident = self.calculate_angles()
        self.ddf = self.calculate_radiation_fraction()

        if savetxt:
            self.save_to_file()

    def _init_cuboid_coordinates(self):
        """Generate outer cuboid coordinates."""
        self.cuboid_coordinates = self.generate_cuboid_coordinates()

    def _init_collimator(self):
        """Retrieve collimator coordinates."""
        self.collim = Collimator(self.element, "top closing side", self.slits_number)
        self.collim_vector_front_back = self.collim.vector_front_back

        (
            self.collimator_spatial,
            self.slit_coord_crys_side,
            self.slit_coord_plasma_side,
        ) = self.collim.read_colim_coord()

    def _init_dispersive_element(self):
        """Initiate data related to the respective dispersive element."""
        de = DispersiveElement(
            self.element, self.crystal_height_step, self.crystal_length_step
        )
        self.disp_elem_coord = de.make_curved_crystal()
        self.AOI = de.AOI
        self.max_reflectivity = de.max_reflectivity
        self.radius_central_point = de.radius_central_point
        self.B = de.B
        self.C = de.C

        self.crystal_point_area = self.calculate_crystal_point_area()
        self.crystal_coordinates = da.from_array(self.disp_elem_coord, chunks=(2000, 3))

    def _init_port(self):
        """Initiate port coordinates."""
        self.port_vertices_coordinates = Port().vertices_coordinates
        self.port_orientation_vector = Port().orientation_vector

    def _init_detector(self):
        """Initiate detector coordinates."""
        self.detector_vertices_coordinates = Detector(self.element).vertices
        self.detector_orientation_vector = Detector(self.element).orientation_vector

    def generate_cuboid_coordinates(self) -> da.array:
        """Generate the block of points covering the volume of a plasma observed by each spectroscopic channel.

        Returns
        -------
        cuboid_coordinates : da.array, shape=(n_points, 3)
            Array of points covering the volume of a plasma.
        """
        cm = CuboidMesh(self.distance_between_points)
        cuboid_coordinates = cm.outer_cube_mesh
        return da.from_array(cuboid_coordinates, chunks=(2000, 3))

    def calculate_crystal_point_area(self) -> float:
        """
        Returns the value of the area on the crystal which represents the fraction
        of surface on which the radiation from one plasma points is directed.

        Returns
        -------
        crystal_point_area : float
            The value of the area on the crystal.
        """
        crystal_point_area = round(
            20 * 80 / self.crystal_height_step / self.crystal_length_step, 2
        )
        print(f"\nCrystal area per investigated point is: {crystal_point_area} mm^2.")
        return crystal_point_area

    @staticmethod
    def line_plane_intersection(
        plane_normal, plane_point, ray_direction, ray_point, epsilon=1e-6
    ) -> da.array:
        """Calculates cross section points of line and plane.

        Parameters
        ----------
        plane_normal : numpy.ndarray
            Normal vector of the plane.
        plane_point : numpy.ndarray
            A point on the plane.
        ray_direction : da.array
            Direction vector of the line.
        ray_point : da.array
            A point on the line.
        epsilon : float, optional
            Tolerance value, by default 1e-6.

        Returns
        -------
        Psi : da.array, shape=(n_intersections, 3)
            Array of cross section points of the line and the plane.
        """
        ndotu = np.tensordot(plane_normal, ray_direction, axes=(0, 2))
        ndotu[np.abs(ndotu) < epsilon] = np.nan
        w = ray_point - plane_point
        x = np.tensordot(plane_normal, w, axes=(0, 2))
        si = -(x / ndotu)[:, :, np.newaxis]
        Psi = w + si * ray_direction + plane_point
        return Psi

    def find_intersection_points(self, p1, p2, p3, plasma, crystal):
        """
        Calcualte intersection points on a plane given by three points (p1, p2, p3)
        and a line between plasma and crystal points.

        Parameters
        ----------
        p1 : ndarray, shape (N, 3)
            First point on the plane.
        p2 : ndarray, shape (N, 3)
            Second point on the plane.
        p3 : ndarray, shape (N, 3)
            Third point on the plane.
        plasma : ndarray, shape (N, 3)
            Starting points of lines.
        crystal : ndarray, shape (N, 3)
            Ending points of lines.

        Returns
        -------
        intersection_points : ndarray, shape (N, 3)
            Intersection points of plane and lines.
        """
        v12 = da.from_array(p2 - p1)
        v13 = da.from_array(p3 - p2)
        plane_normal = np.cross(v12, v13)
        point_on_plane = p1
        ray_directions = plasma - crystal
        intersection_points = self.line_plane_intersection(
            plane_normal, point_on_plane, ray_directions, crystal
        )

        return intersection_points

    def make_spatial_object(
        self, vertices_coordinates: np.ndarray, orientation_vector: np.ndarray
    ) -> np.ndarray:
        """Construct a 3D spatial object from given vertices coordinates and orientation vector.

        Parameters
        ----------
        vertices_coordinates : np.ndarray
            A 2D array of shape (n_vertices, n_dims) representing the coordinates of the vertices that define the object.
        orientation_vector : np.ndarray
            A 1D array representing the orientation vector of the given object.

        Returns
        -------
        np.ndarray
            A 2D array of shape (2 * n_vertices, n_dims) representing the 3D vertices of the object.
        """
        obj_3D_vertices = np.concatenate(
            (
                vertices_coordinates + orientation_vector,
                vertices_coordinates - orientation_vector,
            )
        ).reshape(-1, 3)
        return obj_3D_vertices

    def check_in_hull(
        self, intersection_points, vertices_coordinates, orientation_vector
    ):
        """Check if the given points are within the space defined by the vertices coordinates (its convex hull).

        Parameters
        ----------
        points : numpy.ndarray
            A 2D array of shape (n_points, n_dims) representing the points to be checked whether in hull.
        vertices_coordinates : numpy.ndarray
            A 2D array of shape (n_vertices, n_dims) representing the coordinates of the vertices that define the space.
        orientation_vector : np.ndarray
            A 1D array representing the orientation vector of the given object.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (n_points,) with the result of the in/out check for each point. The value is True if the point is inside or on the hull, and False if outside.
        """

        def is_point_in_hull(
            points, vertices_coordinates, orientation_vector
        ) -> np.ndarray:
            vertices_with_depth = self.make_spatial_object(
                vertices_coordinates, orientation_vector
            )
            hull = Delaunay(vertices_with_depth)
            return hull.find_simplex(points) >= 0

        tested_in_hull = da.map_blocks(
            is_point_in_hull,
            intersection_points,
            vertices_coordinates,
            orientation_vector,
            dtype=bool,
            drop_axis=2,
        )
        return tested_in_hull

    def calculate_radiation_reflection(self):
        """
        Calculate the reflection of a beam.

        Returns
        -------
        reflected_points_location : da.ndarray
            Array of reflected points locations.
        crys_plas_data_arr : da.ndarray
            Array of all input points: plasma, crystal and reflected.
        """
        print("\n--- Calculating reflected beam ---")

        def closest_point_on_line(a, b, p):
            """Find the closest point on a line."""
            ap = p - a
            ab = b - a
            result = a + da.dot(ap, ab) / da.dot(ab, ab) * ab
            return result

        plasma_coordinates = self.cuboid_coordinates.reshape(-1, 1, 3)
        crystal_coordinates = self.crystal_coordinates.reshape(1, -1, 3)

        # calculates reflection angle
        crystal_vector = self.B - self.C
        crys_axis1 = self.radius_central_point
        crys_axis2 = self.radius_central_point + crystal_vector

        on_axis_crystal_normal_points = np.array(
            [
                closest_point_on_line(crys_axis1, crys_axis2, i)
                for i in crystal_coordinates[0]
            ]
        )
        on_axis_crystal_normal_points = on_axis_crystal_normal_points.reshape(1, -1, 3)

        d = crystal_coordinates - plasma_coordinates
        n = crystal_coordinates - on_axis_crystal_normal_points
        # points on the axis of curvature of the crystal
        n /= da.linalg.norm(n, axis=2, keepdims=True)
        r = d - 2 * (d * n).sum(axis=-1, keepdims=True) * n
        reflected_points_location = crystal_coordinates + r

        crys_plas_data_arr = da.stack(
            (
                crystal_coordinates - d,  # plasma
                plasma_coordinates + d,  # crystal
                reflected_points_location,
            ),
            axis=0,
        ).rechunk("auto")
        print("--- Reflected beam calculated ---")
        return reflected_points_location, crys_plas_data_arr

    def check_ray_transmission(self):
        """Checks the transmission of each plasma-crystal combination through a collimator
        and returns a 2D array with all plasma points. It check the collision between the
        plasma and component (crystal sides -front and back, port, detector) by finding
        the intersection points between the ray originating from plasma point and point on
        the respective dispersive elements surface.

        Returns
        -------
        selected_intersections : da.ndarray
            2D array with all plasma points (rows) and all crystal points (columns).
        """
        print("\n--- Calculating photon transmission ---")

        plasma_side_in_hull = []
        crys_side_in_hull = []
        # check the transmission through collimator
        for slit in range(self.slits_number):
            # check the collision with collimator's crystal side
            p1, p2, p3 = self.slit_coord_crys_side[slit, :3]
            all_intersection_points_crys_side = self.find_intersection_points(
                p1, p2, p3, self.crys_plas_data_arr[0], self.crys_plas_data_arr[1]
            )
            crys_side_in_hull.append(
                self.check_in_hull(
                    all_intersection_points_crys_side,
                    self.slit_coord_crys_side[slit],
                    self.collim_vector_front_back,
                )
            )

            # check the collision with collimator's plasma side
            p4, p5, p6 = self.slit_coord_plasma_side[slit, :3]
            all_intersection_points_plasma_side = self.find_intersection_points(
                p4, p5, p6, self.crys_plas_data_arr[0], self.crys_plas_data_arr[1]
            )
            plasma_side_in_hull.append(
                self.check_in_hull(
                    all_intersection_points_plasma_side,
                    self.slit_coord_plasma_side[slit],
                    self.collim_vector_front_back,
                )
            )

        # check the transmission through port
        p7, p8, p9 = self.port_vertices_coordinates[:3]
        all_intersection_points_port = self.find_intersection_points(
            p7, p8, p9, self.crys_plas_data_arr[0], self.crys_plas_data_arr[1]
        )
        port_pts_in_hull = da.array(
            [
                self.check_in_hull(
                    all_intersection_points_port,
                    self.port_vertices_coordinates,
                    self.port_orientation_vector,
                )
            ]
        )

        # check the transmission through port
        p10, p11, p12 = self.detector_vertices_coordinates[:3]
        all_intersetion_points_detector = self.find_intersection_points(
            p10, p11, p12, self.crys_plas_data_arr[2], self.crys_plas_data_arr[1]
        )

        det_pts_in_hull = da.array(
            [
                self.check_in_hull(
                    all_intersetion_points_detector,
                    self.detector_vertices_coordinates,
                    self.detector_orientation_vector,
                )
            ]
        )

        # checks the transmission of all photons over input/output of the collimators slits (all -> axis = 3);
        # next checks whether there was transmission over any of the investigated slits (any -> axis = 2);
        # axis = 0 --> all plasma points
        # axis = 1 --> all crystal points
        # dask_array (plasma points, crystal points) boolean array presenting
        # transmission of each plasma-crystal combination ()
        is_transmited_through_collim = da.stack(
            (da.array(crys_side_in_hull), da.array(plasma_side_in_hull)), axis=0
        )
        transmited_through_collim = is_transmited_through_collim.all(axis=0).any(axis=0)
        crystal_point = self.crystal_height_step * self.crystal_length_step
        transmited_through_collim = transmited_through_collim.reshape(
            -1, len(self.cuboid_coordinates), crystal_point
        )

        selected_intersections = (
            da.concatenate(
                (
                    transmited_through_collim,
                    det_pts_in_hull,
                    port_pts_in_hull,
                ),
                axis=0,
            )
            .rechunk("auto")
            .all(axis=0)
        )

        print("--- Photon transmission calculation finished ---")

        return selected_intersections

    def calculate_indices(self) -> dd.DataFrame:
        """
        Calculate a list of indices representing the combination of plasma and crystal
        that are not blocked by the collimator.

        Returns
        -------
        dd.DataFrame
            A dataframe containing the selected indices, with the index column reset.

        """
        print("\n--- Calculating indices ---")
        plas_points, crys_points = self.selected_intersections.shape
        all_indices = da.arange(plas_points * crys_points).reshape(
            plas_points, crys_points
        )
        selected_indices = all_indices[self.selected_intersections]
        try:
            selected_indices = dd.from_dask_array(
                selected_indices.compute_chunk_sizes()
            )
        except ValueError:
            sys.exit("Not enough points to perform calculations!")

        selected_indices = selected_indices.to_frame().reset_index()
        print("--- Indices calculated ---")

        return selected_indices

    def calculate_plasma_points_indices(self) -> dd.DataFrame:
        """Calculates the indices of the selected plasma points.

        Returns
        -------
        plas_points_indices : dd.DataFrame
            A dask.DataFrame containing the indices of the selected plasma points.
        """
        print("\n--- Calculating indices of selected plasma points ---")

        plas_points_indices = self.selected_indices
        plas_points_indices.columns = ["index", "idx_sel_plas_points"]
        plas_points_indices["idx_sel_plas_points"] = plas_points_indices[
            "idx_sel_plas_points"
        ] // len(self.crystal_coordinates)

        print("--- Selected plasma points indices calculated ---")

        return plas_points_indices

    def calculate_distances(self) -> dd.Series:
        """
        Calculate distances between plasma and crystal points.

        Returns
        -------
        dd.Series
            Dask series containing the calculated distances.
        """
        print("\n--- Calculating distances ---")

        plasma_coordinates = self.crys_plas_data_arr[0]
        crystal_coordinates = self.crys_plas_data_arr[1]
        distance_vectors = plasma_coordinates - crystal_coordinates
        distances = (da.sqrt(da.sum(distance_vectors**2, axis=-1)))[
            self.selected_intersections
        ]
        distances = dd.from_dask_array(distances.compute_chunk_sizes())
        distances = distances.to_frame().reset_index()
        print("--- Distances calculated ---")

        return distances

    def calculate_angles(self) -> dd.Series:
        """
        Calculate the angle of incidence between the plasma ray and the curved surface of the crystal.

        Returns
        -------
        dd.Series
            Dask series containing the calculated angles.
        """
        print("\n--- Calculating angles ---")
        plasma, crystal, reflected_points = self.crys_plas_data_arr

        vector_plasma_to_crystal = plasma - crystal
        reflected_crystal = reflected_points - crystal
        cosine_angle_matrix = contract(
            "ijk -> ji", vector_plasma_to_crystal * reflected_crystal
        ).T / (
            da.linalg.norm(vector_plasma_to_crystal, axis=2)
            * da.linalg.norm(reflected_crystal, axis=2)
        )
        angle_radians = da.arccos(cosine_angle_matrix)
        # angle between ray and normal to the crystal at the incidence point
        angle_degrees = da.degrees(angle_radians)
        angle_of_incident = ((180 - angle_degrees.round(2)) / 2)[
            self.selected_intersections
        ]
        angle_of_incident = dd.from_dask_array(angle_of_incident.compute_chunk_sizes())
        angle_of_incident = angle_of_incident.to_frame().reset_index()
        print("--- Angles calculated ---")

        return angle_of_incident

    def calculate_radiation_fraction(self) -> dd.DataFrame:
        """Calculates the final dataset with selected plasma coordinates and the total intensities.

        Total intensity fraction takes into account distance of the plasma point from crystal
        and angle of incident (AOI) of the ray and the surface of the crystal.
        Since it is assumed that the radiation is emitted into the full solid angle,
        the fraction of this solid angle to the outer sphere's surface is calculated.
        AOI represents the angle of incidend of each ray to the crystals surface.
        The reflection fraction is then calculated checking the AOI and calculating
        the particular fraction using calculate_reflectivity function.

        Returns
        -------
        da.DataFrme
            Dask DataFrame containing final dataset with columns representing indices
            of selected plasma points, plasma x, y, and z coordinates, and the total intensity fraction.
        """
        data_frames = [
            self.plas_points_indices,
            self.distances,
            self.angles_of_incident,
        ]
        ddf = reduce(lambda left, right: dd.merge(left, right, on="index"), data_frames)
        ddf = ddf.drop(["index"], axis=1)
        ddf.columns = ["idx_sel_plas_points", "distances", "angle"]

        def calculate_sphere_area(radius):
            return 4 * np.pi * radius**2

        def calc_reflect_prof(angle):
            A1, x1, w1 = (self.max_reflectivity, self.AOI, 2.2)
            profile = A1 * np.exp(-((angle - x1) ** 2) / (2 * w1**2))

            return profile

        ddf["fraction"] = self.crystal_point_area / calculate_sphere_area(
            ddf["distances"]
        )
        ddf["calc_reflect"] = calc_reflect_prof(ddf["angle"])
        ddf["total_intensity_fraction"] = ddf["fraction"] * ddf["calc_reflect"]
        ddf = ddf.drop(["distances", "angle", "fraction", "calc_reflect"], axis=1)
        ddf = ddf.groupby("idx_sel_plas_points").sum().reset_index()

        indices = ddf["idx_sel_plas_points"].values
        ddf["plasma_x"] = self.cuboid_coordinates[indices][:, 0].round(1)
        ddf["plasma_y"] = self.cuboid_coordinates[indices][:, 1].round(1)
        ddf["plasma_z"] = self.cuboid_coordinates[indices][:, 2].round(1)
        # rearrange columns in dask dataframe
        ddf = ddf[
            [
                "idx_sel_plas_points",
                "plasma_x",
                "plasma_y",
                "plasma_z",
                "total_intensity_fraction",
            ]
        ]

        def plotter():
            """Plots observed plasma volume including distribution of radiation intensity regions."""
            fig = pv.Plotter()
            fig.set_background("black")

            df = ddf.compute()
            intensity = df["total_intensity_fraction"]
            plasma_coordinates = df.to_numpy()[:, 1:4]
            point_cloud = pv.PolyData(plasma_coordinates)
            point_cloud["Intensity"] = intensity
            fig.add_mesh(
                pv.PolyData(point_cloud), point_size=8, render_points_as_spheres=True
            )
            fig.show()

        if self.plot:
            plotter()

        return ddf

    def save_to_file(self):
        """Save dataframe with plasma coordinates and calculated radiation intensity fractions to csv file."""

        self.ddf.to_csv(
            Path(__file__).parent.parent.resolve()
            / "_Input_files"
            / "Geometry"
            / "Observed_plasma_volume"
            / f"{self.element}"
            / f"{self.element}_plasma_coordinates-{self.distance_between_points}_mm_spacing-height_{self.crystal_height_step}-length_{self.crystal_length_step}-slit_{self.slits_number}*.dat",
            sep=";",
            header=True,
            index=False,
        )
        print("\nFile successfully saved!")


if __name__ == "__main__":
    elements_list = ["C", "B", "N", "O"]
    testing_settings = dict(
        slits_number=10,
        distance_between_points=50,
        crystal_height_step=40,
        crystal_length_step=20,
        savetxt=True,
        plot=False,
    )
    for element in elements_list:
        simul = Simulation(element, **testing_settings)
