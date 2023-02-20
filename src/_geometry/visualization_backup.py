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


def make_hull(points):
    hull = ConvexHull(points)
    faces = np.column_stack(
        (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
    ).flatten()
    poly = pv.PolyData(hull.points, faces)

    return poly


def make_cuboid():
    pm = CuboidMesh(distance_between_points=100, cuboid_dimensions=[1800, 800, 2000])
    reduced_cuboid_coordinates = pm.outer_cube_mesh
    print("Cuboid generated!")

    return reduced_cuboid_coordinates


def read_Reff_coordinates():
    Reff = (
        Path(__file__).parent.parent
        / "_Input_files"
        / "Reff"
        / "Reff_coordinates-10_mm.dat"
    )

    Reff_VMEC_calculated = (
        np.loadtxt(Reff, delimiter=";", skiprows=1) * 1000
    )  # convert from [m] to [mm]
    return Reff_VMEC_calculated


def make_mapped_plasma_volume(Reff_VMEC_calculated):
    mapped_plasma_volume = Reff_VMEC_calculated[
        ~np.isnan(Reff_VMEC_calculated).any(axis=1)
    ]
    mapped_plasma_volume = mapped_plasma_volume[:, 1:4]
    print("Mapped plasma volume generated!")

    return mapped_plasma_volume


def make_observed_plasma_volume(Reff_VMEC_calculated, element):
    calculated_plasma_coordinates = (
        Path(__file__).parent.parent.resolve()
        / "_Input_files"
        / "Geometric_data"
        / f"{element}"
        / f"{element}_plasma_coordinates-10_mm_spacing-height_30-length_20-slit_100.dat"
    )

    df = pd.read_csv(calculated_plasma_coordinates, sep=";")
    indexes = df.iloc[:, 0].values
    Reff_VMEC_calculated_with_idx = np.zeros((len(Reff_VMEC_calculated), 5))
    Reff_VMEC_calculated_with_idx[:, 0] = np.arange(len(Reff_VMEC_calculated))
    Reff_VMEC_calculated_with_idx[:, 1:] = Reff_VMEC_calculated[:, 1:]

    Reff_VMEC_calculated_with_idx = Reff_VMEC_calculated_with_idx[indexes]
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
    print(f"Observed plasma volume of {element} generated!")

    return plasma_coordinates, pc


def polar2cart(r, theta, phi):
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)

    return x, y, z


def calculate_plasma_surfaces(phi, surface_number):
    plasma_surfaces = []
    W7X_surface = (
        Path(__file__).parent.parent
        # / "src"
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
            coord = polar2cart(r, theta, math.radians(phi))
            plasma_surfaces.append(coord)
    plasma_surfaces = np.array(plasma_surfaces).reshape(-1, 3) * 1000

    return plasma_surfaces


def make_plasma_surface(phi_range, layer_of_plasma_surface):
    all_plasma_surfaces = []
    for phi in phi_range:
        local_plasma_surface = calculate_plasma_surfaces(phi, layer_of_plasma_surface)
        all_plasma_surfaces.append(local_plasma_surface)
    all_plasma_surfaces = np.array(all_plasma_surfaces).reshape(-1, 3)
    print("Plasma surface generated!")

    return all_plasma_surfaces


def make_axis(phi_range):
    plasma_axis = []
    for phi in phi_range:
        local_axis_point = calculate_plasma_surfaces(phi, 0)[0]
        plasma_axis.append(local_axis_point)
    plasma_axis = np.array(plasma_axis).reshape(-1, 3)
    print("Axis generated!")

    return plasma_axis


def plotter():
    fig = pv.Plotter()
    fig.set_background("black")

    def plot_W7X():
        phi_range = np.arange(0, 360, 1)
        axis = make_axis(phi_range)
        plasma_surfaces = make_plasma_surface(phi_range, -1)

        ### W7-X
        fig.add_mesh(pv.Spline(axis, 400), line_width=3, color="gold", opacity=0.7)
        fig.add_mesh(
            pv.StructuredGrid(
                plasma_surfaces[:, 0], plasma_surfaces[:, 1], plasma_surfaces[:, 2]
            ),
            line_width=3,
            color="grey",
            opacity=0.5,
        )

    def plot_cuboid():
        reduced_cuboid_coordinates = make_cuboid()
        fig.add_mesh(
            reduced_cuboid_coordinates,
            point_size=8,
            color="white",
            render_points_as_spheres=True,
            opacity=0.005,
        )
        points = make_hull(reduced_cuboid_coordinates)
        fig.add_mesh(points, color="white", opacity=0.1)

    def plot_mapped_plasma(Reff_VMEC_calculated):
        mapped_plasma_volume = make_mapped_plasma_volume(Reff_VMEC_calculated)
        points = make_hull(mapped_plasma_volume)
        # fig.add_mesh(points, color='yellow', opacity = 0.3)
        # fig.add_mesh(mapped_plasma_volume, point_size = 8, render_points_as_spheres = True, color = "yellow", opacity = 0.02)

    def plot_observed_plasma(Reff_VMEC_calculated, element, color, polydata=False):
        plasma_coordinates, pc = make_observed_plasma_volume(
            Reff_VMEC_calculated, element
        )
        if polydata:
            fig.add_mesh(
                pv.PolyData(pc), point_size=8, render_points_as_spheres=True
            )  # , opacity = 0.8)
        else:
            points = make_hull(plasma_coordinates)
            fig.add_mesh(points, color=color, opacity=0.89)
            fig.add_mesh(
                plasma_coordinates,
                point_size=8,
                render_points_as_spheres=True,
                color=color,
            )

    def port():
        pt = Port()
        port = pt.make_port_hull()
        fig.add_mesh(port, color="purple", opacity=0.9)

    def radiation_shields():
        protection_shields = {
            "upper chamber": ["1st shield", "2nd shield"],
            "bottom chamber": ["1st shield", "2nd shield"],
        }

        for key, value in protection_shields.items():
            for shield_nr in value:
                protective_shield = RadiationShield(key, shield_nr)
                fig.add_mesh(
                    protective_shield.vertices_coordinates, color="blue", opacity=0.9
                )
                fig.add_mesh(
                    protective_shield.radiation_shield,
                    color="green",
                    opacity=0.9,
                )

    def collimators(element, closing_side):
        # col = Collimator(element, 10, plot = False)
        col = Collimator(element, closing_side, slits_number=10, plot=False)
        for slit in range(col.slits_number):
            (
                colimator_spatial,
                slit_coord_crys_side,
                slit_coord_plasma_side,
            ) = col.read_colim_coord()
            collimator = col.make_collimator(colimator_spatial[slit])
            fig.add_mesh(collimator, color="yellow", opacity=0.9)
            fig.add_mesh(col.A1, color="red", point_size=10)
            fig.add_mesh(col.A2, color="blue", point_size=10)

    def dispersive_elements(element, crystal_height=20, crystal_length=80):
        disp_elem = DispersiveElement(element, crystal_height, crystal_length)
        crys = disp_elem.make_curved_crystal()
        fig.add_mesh(disp_elem.crystal_central_point, color="yellow", point_size=10)
        vertices = np.concatenate(
            (disp_elem.A, disp_elem.B, disp_elem.C, disp_elem.D), axis=0
        ).reshape(4, 3)
        fig.add_mesh(vertices, color="red", point_size=10)
        fig.add_mesh(crys, color="blue")

    def detectors(element: str):
        det = Detector(element, plot=False)
        detector = det.make_detectors_surface()
        fig.add_mesh(detector, color="red", opacity=1)

    ###########################################################################

    list_of_elements = {"B": "red", "C": "blue", "N": "green", "O": "orange"}
    Reff_VMEC_calculated = read_Reff_coordinates()
    plot_W7X()
    plot_cuboid()
    plot_mapped_plasma(Reff_VMEC_calculated)
    port()
    radiation_shields()

    for element in list_of_elements:
        plot_observed_plasma(
            Reff_VMEC_calculated,
            element,
            color=list_of_elements[element],
            polydata=True,
        )
        dispersive_elements(element, 10, 40)
        collimators(element, closing_side="top closing side")
        detectors(element)

    ###########################################################################
    fig.show()


if __name__ == "__main__":
    plotter()