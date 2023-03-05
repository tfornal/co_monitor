__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import ConvexHull

from collimator import Collimator
from detector import Detector
from dispersive_element import DispersiveElement
from mesh_calculation import CuboidMesh
from port import Port
from radiation_shield import RadiationShield


class PlasmaVolume:
    """Class to create a plasma volume from observed plasma coordinates and Reff coordinates."""

    def __init__(self, element: str):
        """
        Parameters
        ----------
        element : str
            Element name used for generating the observed plasma volume. Accepts B, C, N and O elements.
        """
        self.element = element
        self.observed_cuboid_coords = self._get_plasma_coordinates()
        self.Reff_VMEC_calculated = self._get_Reff_coordinates()
        self.observed_plasma_volume = self._assign_indexes()
        self.point_cloud = self._create_point_cloud()

    def _get_Reff_coordinates(self) -> np.ndarray:
        """
        Load Reff coordinates from file.

        Returns
        -------
        Reff_VMEC_calculated : np.ndarray
            Array of Reff coordinates in mm.
        """
        Reff = (
            Path(__file__).parent.parent.parent
            / "input_files"
            / "geometry"
            / "reff"
            / "Reff_coordinates-10_mm.dat"
        )

        Reff_VMEC_calculated = (
            np.loadtxt(Reff, delimiter=";", skiprows=1) * 1000
        )  # convert from [m] to [mm]
        return Reff_VMEC_calculated

    def _get_plasma_coordinates(self) -> np.ndarray:
        """
        Load observed plasma coordinates calculated by simulation module.

        Returns
        -------
        np.ndarray
            Array of observed plasma volume coordinates in mm.
        """
        calculated_plasma_coordinates = (
            Path(__file__).parent.parent.parent.resolve()
            / "input_files"
            / "geometry"
            / "observed_plasma_volume"
            / f"{self.element}"
            / f"{self.element}_plasma_coordinates-10_mm_spacing-height_30-length_20-slit_100.dat"
        )

        observed_cuboid_coords = np.loadtxt(
            calculated_plasma_coordinates, delimiter=";", skiprows=1
        )
        return observed_cuboid_coords

    def _assign_indexes(self) -> np.ndarray:
        """
        Assign indices of Reff coordinates to observed plasma coordinates.

        Returns
        -------
        observed_cuboid_coords : np.ndarray
            Array of plasma volume coordinates with Reff values.
        """
        indexes = self.observed_cuboid_coords[:, 0].astype(int)
        Reff_VMEC_calculated_with_idx = np.zeros((len(self.Reff_VMEC_calculated), 5))
        Reff_VMEC_calculated_with_idx[:, 0] = np.arange(len(self.Reff_VMEC_calculated))
        Reff_VMEC_calculated_with_idx[:, 1:] = self.Reff_VMEC_calculated[:, 1:]
        Reff_VMEC_calculated_with_idx = Reff_VMEC_calculated_with_idx[indexes]
        observed_plasma_volume = Reff_VMEC_calculated_with_idx[
            ~np.isnan(Reff_VMEC_calculated_with_idx).any(axis=1)
        ]
        print(f"Observed plasma volume of {self.element} generated!")

        return observed_plasma_volume

    def _create_point_cloud(self) -> pv.PolyData:
        """
        Creates a point cloud using the plasma coordinates and Reff values from an observed plasma volume.

        Parameters
        ----------
        observed_plasma_volume : np.ndarray
            The observed plasma volume, with plasma coordinates in the first columns and Reff values in the last column.

        Returns
        -------
        point_cloud : pv.PolyData
            A point cloud object with plasma coordinates and Reff values.
        """
        # Extract plasma coordinates and Reff values
        plasma_coordinates = self.observed_plasma_volume[:, 1:-1]
        reff = self.observed_plasma_volume[:, -1]

        # Create PyVista point cloud object and add Reff as point data
        point_cloud = pv.PolyData(plasma_coordinates)
        point_cloud["Reff [m]"] = reff

        return point_cloud


class W7X:
    """
    Initializes reprezentation of W7X plasmas - axis and shape of the last closed flux surface.

    Parameters
    ----------
    phi_range : int, optional
    The number of angles to use for generating plasma surfaces, by default 360.
    """

    def __init__(self, phi_range=360):
        self.phi_range = phi_range
        self.plasma_axis = self._make_axis()
        self.plasma_surfaces = self._get_plasma_surface()

    def _polar2cart(self, r, theta, phi):
        """
        Converts polar to Cartesian coordinates.

        Parameters
        ----------
        r : float
            The radial distance.
        theta : float
            The polar angle.
        phi : float
            The azimuthal angle.

        Returns
        -------
        tuple : x, y, z
            The corresponding Cartesian coordinates.
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def _get_plasma_surface(
        self, layer_of_plasma_surface=-1
    ) -> np.ndarray:  ### -1 = last surface
        """
        Returns the data representing last closed magnetif flux surface of the W7-X plasma for all angles.

        Parameters
        ----------
        layer_of_plasma_surface : int, optional
            The index of the plasma surface to retrieve, by default -1 (i.e., the last surface).

        Returns
        -------
        np.ndarray
            The plasma surface data for all angles.
        """
        all_plasma_surfaces = []
        for phi in range(self.phi_range):
            local_plasma_surface = self._calculate_plasma_surface(
                phi, layer_of_plasma_surface
            )
            all_plasma_surfaces.append(local_plasma_surface)
        all_plasma_surfaces = np.array(all_plasma_surfaces).reshape(-1, 3)
        print("Plasma surface generated!")

        return all_plasma_surfaces

    def _make_axis(self) -> np.ndarray:
        """
        Returns the plasma axis data for all angles.

        Returns
        -------
        np.ndarray
            The plasma axis data for all angles.
        """
        plasma_axis = []
        for phi in range(self.phi_range):
            local_axis_point = self._calculate_plasma_surface(phi, 0)[0]
            plasma_axis.append(local_axis_point)
        plasma_axis = np.array(plasma_axis).reshape(-1, 3)
        print("Axis generated!")

        return plasma_axis

    def _calculate_plasma_surface(self, phi, surface_number) -> np.ndarray:
        """
        Calculates the plasma surface for a given angle and surface number.

        Parameters
        ----------
        phi : int
            The angle for which to calculate the plasma surface.
        surface_number : int
            The index of the plasma surface to retrieve.

        Returns
        -------
        np.ndarray
            The plasma surface data.
        """
        plasma_surface = []
        w7x_surface = (
            Path(__file__).parent.parent.parent.resolve()
            / "input_files"
            / "visualization"
            / "W7-X_plasma_shape"
            / f"angle{phi}.json"
        )
        with open(w7x_surface, "r") as json_file:
            data = json.load(json_file)
            surfaces = data["surfaces"][surface_number]
            for i in range(len(surfaces["x1"])):
                R = surfaces["x1"][i]
                z = surfaces["x3"][i]
                r = np.sqrt(R**2 + z**2)
                theta = np.arccos(z / r)
                coord = self._polar2cart(r, theta, np.radians(phi))
                plasma_surface.append(coord)
        plasma_surface = np.array(plasma_surface).reshape(-1, 3) * 1000

        return plasma_surface


class Visualization:
    def __init__(self, elements_list, polydata=True):
        self.elements_list = elements_list
        self.polydata = polydata

        self._init_W7X()
        self._init_cuboid()
        self._init_plasma_volume()
        self._init_port()
        self._init_radiation_shield()
        self._init_collimator()
        self._init_dispersive_element()
        self._init_detector()

        self.plotter()

    def _init_W7X(self):
        """Constructor of W7-X plasma shape."""
        w7x = W7X()
        self.plasma_axis = w7x.plasma_axis
        self.plasma_surfaces = w7x.plasma_surfaces
        self.phi_range = w7x.plasma_surfaces

    def _init_cuboid(self):
        """Constructor of outer cuboid mesh."""
        cub = CuboidMesh(
            distance_between_points=100, cuboid_dimensions=[1800, 800, 2000]
        )
        self.cuboid_coordinates = cub.outer_cube_mesh
        print("Cuboid generated!")

    def _init_plasma_volume(self):
        """Constructor of observed plasma volume."""
        self.observed_plasmas = []
        self.point_clouds = []

        for element in self.elements_list:
            plas_vol = PlasmaVolume(element)
            observed_plasma_volume = plas_vol.observed_plasma_volume
            self.point_clouds.append(plas_vol.point_cloud)
            self.observed_plasmas.append(observed_plasma_volume)

    def _init_port(self):
        """Constructor of W7-X port."""
        pt = Port()
        self.port_hull = pt.port_hull

    def _init_radiation_shield(self):
        """Constructor of prodective stray radiation shields."""
        self.radiation_shields = []
        protection_shields = {
            "upper chamber": ["1st shield", "2nd shield"],
            "bottom chamber": ["1st shield", "2nd shield"],
        }

        for chamber, value in protection_shields.items():
            for shield_nr in value:
                rs = RadiationShield(chamber, shield_nr)
                self.radiation_shields.append(rs)

    def _init_collimator(
        self, closing_side: str = "top closing side", slits_number: int = 10
    ):
        """Initialize collimator objects for each observation channel.

        Parameters
        ----------
        closing_side : str, optional
            The closing side of the collimator. Defaults to "top closing side".
        slits_number : int, optional
            The number of slits in the collimator. Defaults to 10.
        """
        self.collimators = []
        for element in self.elements_list:
            col = Collimator(element, closing_side, slits_number)
            for slit in range(col.slits_number):
                (
                    colimator_spatial,
                    slit_coord_crys_side,
                    slit_coord_plasma_side,
                ) = col.read_colim_coord()
                self.collimators.append(col.make_collimator(colimator_spatial[slit]))

    def _init_dispersive_element(self, crystal_height=20, crystal_length=80):
        """
        Initializes the dispersive elements for the respectuve observation channel (B, C, N or O).

        Parameters
        ----------
        crystal_height : float, optional
            The height of the crystal in millimeters. Default is 20.
        crystal_length : float, optional
            The length of the crystal in millimeters. Default is 80.
        """
        self.dispersive_elements = []
        for element in self.elements_list:
            de = DispersiveElement(element, crystal_height, crystal_length)
            crystal_surface = de.crystal_points
            self.dispersive_elements.append(crystal_surface)

    def _init_detector(self):
        """Constructor of detectors for the respective observation channel
        (B, C, N or O)."""
        self.detectors = []
        for element in self.elements_list:
            det = Detector(element)
            self.detectors.append(det.poly_det)

    def _make_hull(self, points):
        """
        Method to make the convex hull of a set of input points.

        Parameters
        ----------
        points : ndarray
            The points to compute the convex hull of points.

        Returns
        -------
        poly : pv.PolyData
            Hull as a PolyData object, representing the surface of
            the convex hull.
        """
        hull = ConvexHull(points)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)

        return poly

    def plotter(self):
        """Plots all the initialized objects."""
        fig = pv.Plotter()
        fig.set_background("black")

        # W7-X surface visualization
        x, y, z = (
            self.plasma_surfaces[:, 0],
            self.plasma_surfaces[:, 1],
            self.plasma_surfaces[:, 2],
        )
        fig.add_mesh(
            pv.StructuredGrid(x, y, z),
            line_width=3,
            color="grey",
            opacity=0.5,
        )

        # W7-X axis visualization
        fig.add_mesh(
            pv.Spline(self.plasma_axis, 400),
            line_width=3,
            color="gold",
            opacity=0.7,
        )

        # cuboid visualization
        fig.add_mesh(
            self._make_hull(self.cuboid_coordinates), color="white", opacity=0.1
        )

        # port hull visualization
        fig.add_mesh(self.port_hull, color="purple", opacity=0.9)

        # detector hull visualization
        for detector in self.detectors:
            fig.add_mesh(detector, color="red", opacity=1)

        # radiation shields points and hull visualization
        for shield in self.radiation_shields:
            fig.add_mesh(
                shield.vertices_coordinates,
                color="blue",
                opacity=0.9,
                render_points_as_spheres=True,
            )
            fig.add_mesh(
                shield.radiation_shield,
                color="green",
                opacity=0.9,
            )
        # collimators' hull visualization
        for collimator in self.collimators:
            fig.add_mesh(collimator, color="yellow", opacity=0.9)

        # dispersive elements' hull visualization
        for disp_elem in self.dispersive_elements:
            fig.add_mesh(
                disp_elem, color="blue", point_size=3, render_points_as_spheres=True
            )

        if self.polydata:
            for pc in self.point_clouds:
                fig.add_mesh(
                    pv.PolyData(pc),
                    point_size=8,
                    render_points_as_spheres=True,
                )
        else:
            for observed_plasma_volume in self.observed_plasmas:
                # points = self._make_hull(observed_plasma_volume[:, 1:-1])
                fig.add_mesh(
                    observed_plasma_volume[:, 1:-1],
                    point_size=8,
                    render_points_as_spheres=True,
                    color=np.random.rand(
                        3,
                    ),
                )
                points = self._make_hull(observed_plasma_volume[:, 1:-1])
                fig.add_mesh(
                    points,
                    color=np.random.rand(
                        3,
                    ),
                    opacity=0.1,
                )

        fig.show()


if __name__ == "__main__":
    elements = ["B", "C", "N", "O"]
    vis = Visualization(elements, polydata=False)
