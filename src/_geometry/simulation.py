"""
Kod musi pobierac dane z pliku wyjsciowego uzyskanego za pomoca rest api.
Step 1 - wyznaczenie objetosci jaka obserwowalby kazdy kanal.
Step 2 - Nastepnie kazdy punkt sprawdzany bylby modulem ponizej.
TODO - reflectivity readout -> not a gaussian but experimental profile
"""

from functools import reduce
from pathlib import Path
import time

import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
from opt_einsum import contract
import pyvista as pv
from scipy.spatial import ConvexHull, Delaunay

from collimator import Collimator  # funkcja make_hull
from detector import Detector
from dispersive_element import DispersiveElement  # funkcja make_curved_crystal
from mesh_calculation import CuboidMesh
from port import Port


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

        # calculate ray reflection
        (
            self.reflected_points_location,
            self.full_input_array,
        ) = self.calculate_radiation_reflection()
        self.selected_intersections = self.check_ray_transmission()
        self.selected_indices = self.calculate_indices()
        self.plas_points_indices = self.calculate_plasma_points_indices()
        self.distances = self.grab_distances()
        self.angles_of_incident = self.grab_angles()
        self.ddf = self.calculate_radiation_fraction()

        if savetxt:
            self.save_to_file()

    def _init_cuboid_coordinates(self):
        """Generate outer cuboid coordinates."""
        self.cuboid_coordinates = self.generate_cuboid_coordinates()

    def _init_collimator(self):
        """Retrieve collimator coordinates."""
        self.collim = Collimator(self.element, "top closing side", self.slits_number)
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
        self.disp_elem = de.make_curved_crystal()
        self.AOI = de.AOI
        self.max_reflectivity = de.max_reflectivity
        self.radius_central_point = de.radius_central_point
        self.B = de.B
        self.C = de.C

        self.crystal_point_area = self.calculate_crystal_point_area()
        self.crystal_coordinates = da.from_array(self.disp_elem, chunks=(2000, 3))

    def _init_port(self):
        """Initiate port coordinates."""
        self.port_vertices_coordinates = Port().vertices_coordinates
        self.port_orientation_vector = Port().orientation_vector

    def _init_detector(self):
        """Initiate detector coordinates."""
        self.detector_vertices_coordinates = Detector(self.element).vertices
        self.detector_orientation_vector = Detector(self.element).orientation_vector

    def generate_cuboid_coordinates(self):
        """
        Generate the block of points covering the volume of a plasma observed by each spectroscopic channel.
        """
        cm = CuboidMesh(self.distance_between_points)
        cuboid_coordinates = cm.outer_cube_mesh
        return da.from_array(cuboid_coordinates, chunks=(2000, 3))

    def calculate_crystal_point_area(self):
        """
        Returns the value of the area on the crystal which represents the fraction
        of surface on which the radiation from one plasma points is directed.
        """
        crystal_point_area = round(
            20 * 80 / self.crystal_height_step / self.crystal_length_step, 2
        )
        print(f"\nCrystal area per investigated point is: {crystal_point_area} mm^2.")
        return crystal_point_area

    @staticmethod
    def line_plane_intersection(
        plane_normal, plane_point, ray_direction, ray_point, epsilon=1e-6
    ):
        """Calculates cross section points of line and plane.

        Returns:
            _type_: _description_
        """
        ndotu = np.tensordot(plane_normal, ray_direction, axes=(0, 2))
        ndotu[np.abs(ndotu) < epsilon] = np.nan
        w = ray_point - plane_point
        x = np.tensordot(plane_normal, w, axes=(0, 2))
        si = -(x / ndotu)[:, :, np.newaxis]
        Psi = w + si * ray_direction + plane_point
        return Psi

    def find_intersection_points_basic(self, p1, p2, p3, plasma, crystal):
        """
        Calcualte intersection points on a plane given by three points (p1, p2, p3)
        and line between plasma and crystal points.
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

    ############## implementacja ogolnego check in hull
    ############## implementacja ogolnego check in hull

    def check_in_hull_general(
        self, intersection_points, vertices_coordinates, orientation_vector
    ):
        vertices_with_depth = self.make_thick_obj(
            vertices_coordinates, orientation_vector
        )
        hull = Delaunay(vertices_with_depth)

        return hull.find_simplex(intersection_points) >= 0

    #############
    def make_thick_obj(self, vertices_coordinates, orientation_Vector):
        obj_3D_vertices = np.concatenate(
            (
                vertices_coordinates[:4] + orientation_Vector,
                vertices_coordinates[:4] - orientation_Vector,
            )
        ).reshape(8, 3)
        return obj_3D_vertices

    ############## tutaj koniec
    ############## tutaj koniec

    def check_in_hull(
        self, intersection_points: da.array, hull_vertices: np.ndarray
    ) -> da.array:
        """
        Checks whether intersection points are inside the hull defined by its vertices.
        """
        tested_in_hull = da.map_blocks(
            self.collim.check_in_hull,
            intersection_points,
            hull_vertices,
            dtype=bool,
            drop_axis=2,
        )
        return tested_in_hull

    def check_in_hull_port(self, intersection_points, hull_points):
        """
        Checks whether intersection points are inside the considered hull.
        """

        tested_in_hull = da.map_blocks(
            self.check_in_hull_general,
            intersection_points,
            hull_points,
            self.port_orientation_vector,
            dtype=bool,
            drop_axis=2,
        )

        return tested_in_hull

    def calculate_radiation_reflection(self):
        print("\n--- Calculating reflected beam ---")

        def closest_point_on_line(a, b, p):
            ap = p - a
            ab = b - a
            result = a + da.dot(ap, ab) / da.dot(ab, ab) * ab
            return result

        plasma_coordinates = self.cuboid_coordinates.reshape(-1, 1, 3)
        crystal_coordinates = self.crystal_coordinates.reshape(1, -1, 3)

        ### calculate reflection angle
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

        full_input_array = da.stack(
            (
                crystal_coordinates - d,  # plasma
                plasma_coordinates + d,  # crystal
                reflected_points_location,
            ),
            axis=0,
        ).rechunk("auto")
        return reflected_points_location, full_input_array

    def check_ray_transmission(self):
        """Checks the transmission of each plasma-crystal combination through a collimator.

        Returns:
            dask array: selected_intersections
            2D array with all plasma points (rows) and all crystal points (columns).
        """
        tests_crys_side = []
        found_intersection_points_crys_side = []
        tests_plasma_side = []
        found_intersection_points_plasma_side = []
        tests_port = []
        found_intersection_points_port = []
        tests_detector = []
        found_intersection_points_detector = []

        """TODO odseparowac poszczegolne elementy z mozliwoscia ich 'wylaczenia' """

        # check the transmission through collimator
        for i in range(self.slits_number):
            # check the collision with collimator's crystal side
            p1, p2, p3 = self.slit_coord_crys_side[i, :3]
            all_intersection_points_crys_side = self.find_intersection_points_basic(
                p1, p2, p3, self.full_input_array[0], self.full_input_array[1]
            )
            found_intersection_points_crys_side.append(
                all_intersection_points_crys_side
            )
            tests_crys_side.append(
                self.check_in_hull(
                    all_intersection_points_crys_side, self.slit_coord_crys_side[i]
                )
            )

            # check the collision with collimator's plasma side
            p4, p5, p6 = self.slit_coord_plasma_side[i, :3]
            all_intersection_points_plasma_side = self.find_intersection_points_basic(
                p4, p5, p6, self.full_input_array[0], self.full_input_array[1]
            )
            found_intersection_points_plasma_side.append(
                all_intersection_points_plasma_side
            )
            tests_plasma_side.append(
                self.check_in_hull(
                    all_intersection_points_plasma_side, self.slit_coord_plasma_side[i]
                )
            )

        # check the collision with port
        p7, p8, p9 = self.port_vertices_coordinates[:3]
        all_intersection_points_port = self.find_intersection_points_basic(
            p7, p8, p9, self.full_input_array[0], self.full_input_array[1]
        )
        found_intersection_points_port.append(all_intersection_points_port)
        tested_in_hull = self.check_in_hull_port(
            all_intersection_points_port, self.port_vertices_coordinates
        )

        # tested_in_hull = self.check_in_hull(all_intersection_points_port, (self.port_vertices_coordinates[:4]))
        tests_port.append(tested_in_hull)

        # check the collision with detectors surface
        p10, p11, p12 = self.detector_vertices_coordinates[:3]
        all_intersetion_points_detector = self.find_intersection_points_basic(
            p10, p11, p12, self.full_input_array[2], self.full_input_array[1]
        )

        found_intersection_points_detector.append(all_intersetion_points_detector)
        tested_in_hull = self.check_in_hull_general(
            all_intersetion_points_detector,
            self.detector_vertices_coordinates[:4],
            self.detector_orientation_vector,
        )
        tests_detector.append(tested_in_hull)

        selected_intersections = da.stack(
            (da.array(tests_crys_side), da.array(tests_plasma_side)), axis=0
        )
        selected_intersections = selected_intersections.all(axis=0)
        selected_intersections = selected_intersections.any(axis=0)
        krysztal_pkt = self.crystal_height_step * self.crystal_length_step
        selected_intersections = selected_intersections.reshape(
            -1, len(self.cuboid_coordinates), krysztal_pkt
        )

        selected_intersections = da.concatenate(
            (selected_intersections, da.array(tests_detector), da.array(tests_port)),
            axis=0,
        ).rechunk("auto")
        selected_intersections = selected_intersections.all(axis=0)
        ### checks the transmission of each ray over input/output of the collimators slits (all -> axis = 3);
        ### next checks whether there was transmission over any of the investigated slits (any -> axis = 2);
        ### axis = 0 --> all plasma points
        ### axis = 1 --> all crystal points
        ### dask_array (plasma points, crystal points) boolean array presenting
        ### transmission of each plasma-crystal combination ()

        def plotter():

            fig = pv.Plotter()
            fig.set_background("black")
            fig.add_mesh(self.detector_vertices_coordinates[:4], color="red")
            # fig.add_mesh(wszystkie_detector,color = "yellow", opacity = 0.01)
            # fig.add_mesh(wszystkie_detector[przeszly_detector] ,color = "red")
            make_detectors_surface = Detector(self.element).make_detectors_surface()
            fig.add_mesh(make_detectors_surface, color="purple", opacity=0.4)

            fig.add_mesh(
                self.full_input_array.compute()[0].reshape(-1, 3),
                color="red",
                render_points_as_spheres=True,
            )  ### plasma
            fig.add_mesh(
                self.full_input_array.compute()[1].reshape(-1, 3),
                color="blue",
                render_points_as_spheres=True,
            )  ### crystal
            fig.add_mesh(
                self.full_input_array.compute()[2].reshape(-1, 3),
                color="green",
                render_points_as_spheres=True,
            )  ### reflected plasma

            fig.add_mesh(
                self.cuboid_coordinates.compute().flatten().reshape(-1, 3),
                color="yellow",
                opacity=0.3,
                render_points_as_spheres=True,
            )
            fig.add_mesh(
                self.reflected_points_location.compute().flatten().reshape(-1, 3),
                color="orange",
                opacity=0.3,
                render_points_as_spheres=True,
            )
            fig.add_points(
                self.crystal_coordinates.compute(),
                color="red",
                point_size=10,
                label="Crystal coordinates",
                render_points_as_spheres=True,
            )
            # fig.add_points(on_axis_crystal_normal_points[0], color="yellow", point_size = 10, label="Crystal axis")
            fig.show()

        if self.plot:
            plotter()

        print("\n--- Ray transmission calculation finished ---")

        return selected_intersections

    def percentage_transmission(self):
        """
        Returns percentage transmission of amount of observable plasma points in reference to all used for calculation
        """
        all_plas_crys_combinations = len(self.cuboid_coordinates) * len(
            self.crystal_coordinates
        )
        indices = da.arange(all_plas_crys_combinations).reshape(
            len(self.cuboid_coordinates), len(self.crystal_coordinates)
        )
        selected_indices = indices[self.selected_intersections]
        transmission = round(
            len(selected_indices.compute())
            / (len(self.cuboid_coordinates) * len(self.crystal_coordinates))
            * 100,  ### TODO zahardkodowane wartosci;
            2,
        )
        print(f"\nTrnasmission is: {transmission}%\n")

    def calculate_indices(self):
        """
        Returns list of indices representing combination of plasma-crystal
        if not blocked by collimator.
        Returns dask_series. TODO
        """
        plas_points, crys_points = self.selected_intersections.shape
        all_indices = da.arange(plas_points * crys_points).reshape(
            plas_points, crys_points
        )
        selected_indices = all_indices[self.selected_intersections]
        selected_indices = dd.from_dask_array(selected_indices.compute_chunk_sizes())
        selected_indices = selected_indices.to_frame().reset_index()
        print("\n--- Indices calculated ---")

        return selected_indices

    def calculate_plasma_points_indices(self):
        plas_points_indices = self.selected_indices
        plas_points_indices.columns = ["index", "idx_sel_plas_points"]
        plas_points_indices["idx_sel_plas_points"] = plas_points_indices[
            "idx_sel_plas_points"
        ] // len(self.crystal_coordinates)
        print("\n--- Selected plasma points indices calculated ---")

        return plas_points_indices

    def grab_distances(self, save=False):
        """
        Calculate distances between plasma and crystal points.
        Returns dask_series.
        """
        all_data_input = self.full_input_array
        plasma_coordinates = all_data_input[0]
        crystal = all_data_input[1]
        distance_vectors = plasma_coordinates - crystal
        distances = (da.sqrt(da.sum(distance_vectors**2, axis=-1)))[
            self.selected_intersections
        ]
        if save is not False:
            self.selected_indices = (
                self.selected_indices.compute_chunk_sizes().rechunk()
            )
            distances = distances.compute_chunk_sizes().rechunk()

            #### TODO zahardkodowane nazwy do poprawienia
            da.to_zarr(
                self.selected_indices.astype(np.int64),
                "selected_indices.zarr",
                overwrite=True,
            )
            da.to_zarr(distances.astype(np.float32), "distances.zarr", overwrite=True)
        distances = dd.from_dask_array(
            distances.compute_chunk_sizes()
        )  ### WASKIE GARDLO
        distances = distances.to_frame().reset_index()
        print("\n--- Distances calculated ---")

        return distances

    def grab_angles(self, save=False):
        """
        Calculate incident angles between plasma ray and crystals surface.
        Returns dask_series.
        """
        all_data_input = self.full_input_array
        plasma_coordinates = all_data_input[0]
        crystal_coordinates = all_data_input[1]
        reflected_points = all_data_input[2]

        vector_plasma_to_crystal = plasma_coordinates - crystal_coordinates
        reflected_crystal = reflected_points - crystal_coordinates
        cosine_angle_matrix = contract(
            "ijk -> ji", vector_plasma_to_crystal * reflected_crystal
        ).T / (
            da.linalg.norm(vector_plasma_to_crystal, axis=2)
            * da.linalg.norm(reflected_crystal, axis=2)
        )
        angle_radians = da.arccos(cosine_angle_matrix)
        angle_degrees = da.degrees(
            angle_radians
        )  ## angle between ray and normal to the crystal at the incidence point
        angle_of_incident = ((180 - angle_degrees.round(2)) / 2)[
            self.selected_intersections
        ]
        if save is not False:
            self.selected_indices = (
                self.selected_indices.compute_chunk_sizes().rechunk()
            )
            angle_of_incident = angle_of_incident.compute_chunk_sizes().rechunk()

            # TODO zahardkodowane nazwy do poprawienia
            da.to_zarr(
                self.selected_indices.astype(np.int64),
                "selected_indices.zarr",
                overwrite=True,
            )
            da.to_zarr(
                angle_of_incident.astype(np.float32),
                "angle_of_incident.zarr",
                overwrite=True,
            )

        angle_of_incident = dd.from_dask_array(angle_of_incident.compute_chunk_sizes())
        angle_of_incident = angle_of_incident.to_frame().reset_index()
        print("\n--- Angles calculated ---")

        return angle_of_incident

    def calculate_radiation_fraction(self):
        """
        Calculates fina`l dataset with selected plasma coordinates and the total intensities.
        Total intensity fraction takes into account distance of the plasma point from crystal
        and angle of incident (AOI) of the ray and the surface of the crystal.
        Since it is assumed that the radiation is emitted into the full solid angle,
        the fraction of this solid angle to the outer sphere's surface is calculated.
        AOI represents the angle of incidend of each ray to the crystals surface.
        The reflection fraction is then calculated checking the AOI and calculating
        the particular fraction using calculate_reflectivity function.`
        """
        data_frames = [
            self.plas_points_indices,
            self.distances,
            self.angles_of_incident,
        ]
        ddf = reduce(lambda left, right: dd.merge(left, right, on="index"), data_frames)
        ddf = ddf.drop(["index"], axis=1)
        ddf.columns = ["idx_sel_plas_points", "distances", "angle"]

        def calculate_sphere_area(R):
            area_of_a_sphere = 4 * np.pi * R**2
            return area_of_a_sphere

        def calcualte_reflectivity(calculated_angle):
            A1, x1, w1 = (self.max_reflectivity, self.AOI, 2.2)
            profile = A1 * np.exp(-((calculated_angle - x1) ** 2) / (2 * w1**2))

            return profile

        ddf["fraction"] = self.crystal_point_area / calculate_sphere_area(
            ddf["distances"]
        )
        ddf["calc_reflect"] = calcualte_reflectivity(ddf["angle"])
        ddf["total_intensity_fraction"] = ddf["fraction"] * ddf["calc_reflect"]
        ddf = ddf.drop(["distances", "angle", "fraction", "calc_reflect"], axis=1)
        ddf = ddf.groupby("idx_sel_plas_points").sum().reset_index()

        indices = ddf["idx_sel_plas_points"].values
        ddf["plasma_x"] = self.cuboid_coordinates[indices][:, 0]
        ddf["plasma_y"] = self.cuboid_coordinates[indices][:, 1]
        ddf["plasma_z"] = self.cuboid_coordinates[indices][:, 2]
        ddf = ddf[
            [
                "idx_sel_plas_points",
                "plasma_x",
                "plasma_y",
                "plasma_z",
                "total_intensity_fraction",
            ]
        ]
        return ddf

    def save_to_file(self):
        """Save dataframe with plasma coordinates and calculated radiation intensity fractions"""
        """TODO - zapis do bazy danych (sql???? czy cos innego?) a nie csv!!!!!!"""
        self.ddf.to_csv(
            Path(__file__).parent.parent.resolve()
            / "_Input_files"
            / "Geometric_data"
            / f"{self.element}"  ####  +top/bottom closing side -resolve
            / f"{self.element}_plasma_coordinates-{self.distance_between_points}_mm_spacing-height_{self.crystal_height_step}-length_{self.crystal_length_step}-slit_{self.slits_number}*.dat",
            sep=";",
            header=True,
            index=False,
        )
        print("\nFile successfully saved!")


#### TODO zahardkodowane nazwy do poprawienia
### TODO zrobic testy dla slits number closing top i bottom # for slit in slits_number:

# elements_list = ["B", "C", "N", "O"]
elements_list = ["C"]

testing_settings = dict(
    slits_number=10,
    distance_between_points=80,
    crystal_height_step=3,
    crystal_length_step=3,
    savetxt=False,
    plot=True,
)

start = time.time()
if __name__ == "__main__":
    for element in elements_list:
        simul = Simulation(element, **testing_settings)


print(f"\nExecution time is {round((time.time() - start), 2)} s")
