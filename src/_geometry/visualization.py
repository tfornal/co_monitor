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
    def __init__(self, element):
        self.element = element
        self.Reff_VMEC_calculated = self.get_Reff_coordinates()
        self.plasma_coordinates, self.pc = self.make_observed_plasma_volume()

    def get_Reff_coordinates(self):
        Reff = (
            Path(__file__).parent.parent
            / "_Input_files"
            / "Geometry"
            / "Reff"
            / "Reff_coordinates-10_mm.dat"
        )

        Reff_VMEC_calculated = (
            np.loadtxt(Reff, delimiter=";", skiprows=1) * 1000
        )  # convert from [m] to [mm]
        return Reff_VMEC_calculated

    def make_observed_plasma_volume(self):
        calculated_plasma_coordinates = (
            Path(__file__).parent.parent.resolve()
            / "_Input_files"
            / "Geometry"
            / "Observed_plasma_volume"
            / f"{self.element}"
            / f"{self.element}_plasma_coordinates-10_mm_spacing-height_30-length_20-slit_100.dat"
        )

        observed_cuboid_coords = np.loadtxt(
            calculated_plasma_coordinates, delimiter=";", skiprows=1
        )
        idx_observed_cuboid_coords = observed_cuboid_coords[:, 0].astype(int)
        Reff_VMEC_calculated_with_idx = np.zeros((len(self.Reff_VMEC_calculated), 5))
        Reff_VMEC_calculated_with_idx[:, 0] = np.arange(len(self.Reff_VMEC_calculated))
        Reff_VMEC_calculated_with_idx[:, 1:] = self.Reff_VMEC_calculated[:, 1:]
        Reff_VMEC_calculated_with_idx = Reff_VMEC_calculated_with_idx[
            idx_observed_cuboid_coords
        ]
        observed_plasma_volume = Reff_VMEC_calculated_with_idx[
            ~np.isnan(Reff_VMEC_calculated_with_idx).any(axis=1)
        ]
        reff = observed_plasma_volume[:, -1]
        reff = reff
        plasma_coordinates = observed_plasma_volume[:, 1:-1]

        def create_point_cloud(coordinates, reff):
            point_cloud = pv.PolyData(coordinates)
            point_cloud["Reff [m]"] = reff

            return point_cloud

        pc = create_point_cloud(plasma_coordinates, reff)
        print(f"Observed plasma volume of {self.element} generated!")

        return plasma_coordinates, pc


class W7X:
    def __init__(self, phi_range=360):
        self.phi_range = phi_range
        self.plasma_axis = self.make_axis()
        self.plasma_surfaces = self.get_plasma_surface()

    def _polar2cart(self, r, theta, phi):
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        return x, y, z

    def calculate_plasma_surfaces(self, phi, surface_number):
        plasma_surfaces = []
        W7X_surface = (
            Path(__file__).parent.parent.resolve()
            / "_Input_files"
            / "__Visualization"
            / "W7-X_plasma_shape"
            / f"angle{phi}.json"
        )
        with open(W7X_surface, "r") as json_file:
            data = json.load(json_file)
            surfaces = data["surfaces"][surface_number]
            for i in range(len(surfaces["x1"])):
                R = surfaces["x1"][i]
                z = surfaces["x3"][i]
                r = np.sqrt(R**2 + z**2)
                theta = np.arccos(z / r)
                coord = self._polar2cart(r, theta, math.radians(phi))
                plasma_surfaces.append(coord)
        plasma_surfaces = np.array(plasma_surfaces).reshape(-1, 3) * 1000

        return plasma_surfaces

    def make_axis(self):
        plasma_axis = []
        for phi in range(self.phi_range):
            local_axis_point = self.calculate_plasma_surfaces(phi, 0)[0]
            plasma_axis.append(local_axis_point)
        plasma_axis = np.array(plasma_axis).reshape(-1, 3)
        print("Axis generated!")

        return plasma_axis

    def get_plasma_surface(self, layer_of_plasma_surface=-1):  ### -1 = last surface
        all_plasma_surfaces = []
        for phi in range(self.phi_range):
            local_plasma_surface = self.calculate_plasma_surfaces(
                phi, layer_of_plasma_surface
            )
            all_plasma_surfaces.append(local_plasma_surface)
        all_plasma_surfaces = np.array(all_plasma_surfaces).reshape(-1, 3)
        print("Plasma surface generated!")

        return all_plasma_surfaces


class Visualization:
    def __init__(self, elements_list):
        self.elements_list = elements_list
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
        w7x = W7X()
        self.plasma_axis = w7x.plasma_axis
        self.plasma_surfaces = w7x.plasma_surfaces
        self.phi_range = w7x.plasma_surfaces

    def _init_cuboid(self):
        cub = CuboidMesh(
            distance_between_points=100, cuboid_dimensions=[1800, 800, 2000]
        )
        self.cuboid_coordinates = cub.outer_cube_mesh
        print("Cuboid generated!")

    def _init_plasma_volume(self):
        for element in self.elements_list:
            plas_vol = PlasmaVolume(element)
            self.Reff_VMEC_calculated = plas_vol.Reff_VMEC_calculated
            self.plasma_coordinates = plas_vol.plasma_coordinates
            self.pc = plas_vol.pc

    def _init_port(self):
        pt = Port()
        self.port_hull = pt.port_hull

    def _init_radiation_shield(self):
        self.radiation_shields = []
        protection_shields = {
            "upper chamber": ["1st shield", "2nd shield"],
            "bottom chamber": ["1st shield", "2nd shield"],
        }

        for chamber, value in protection_shields.items():
            for shield_nr in value:
                rs = RadiationShield(chamber, shield_nr)
                self.radiation_shields.append(rs)

    def _init_collimator(self, closing_side="top closing side", slits_number=10):
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
        self.dispersive_elements = []
        for element in self.elements_list:
            de = DispersiveElement(element, crystal_height, crystal_length)
            crystal_surface = de.crystal_points
            self.dispersive_elements.append(crystal_surface)

    def _init_detector(self):
        self.detectors = []
        for element in self.elements_list:
            det = Detector(element)
            self.detectors.append(det.poly_det)

    def _make_hull(self, points):
        hull = ConvexHull(points)
        faces = np.column_stack(
            (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
        ).flatten()
        poly = pv.PolyData(hull.points, faces)

        return poly

    def plotter(self):
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

        def plot_observed_plasma(Reff_VMEC_calculated, element, color, polydata=False):
            plasma_coordinates, pc = make_observed_plasma_volume(
                Reff_VMEC_calculated, element
            )
            if polydata:
                fig.add_mesh(
                    pv.PolyData(pc), point_size=8, render_points_as_spheres=True
                )  # , opacity = 0.8)
            else:
                points = self._make_hull(plasma_coordinates)
                fig.add_mesh(points, color=color, opacity=0.89)
                fig.add_mesh(
                    plasma_coordinates,
                    point_size=8,
                    render_points_as_spheres=True,
                    color=color,
                )

        fig.show()


if __name__ == "__main__":
    elements = ["B", "C", "N", "O"]
    vis = Visualization(elements)
