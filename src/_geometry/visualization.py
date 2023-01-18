import pyvista as pv
from dispersive_element_OOP import DispersiveElement
from collimator_OOP import Collimator
from detector_OOP import Detector
from port import Port
import numpy as np
from mesh_calculation_OOP import PlasmaMesh
from radiation_shield_OOP import RadiationShield
import pandas as pd
import pathlib
import math
import json
from scipy.spatial import ConvexHull

# TODO poprawienie dokumentacji!


def make_hull(points):
    """_summary_

    Args:
        points (_type_): _description_

    Returns:
        _type_: _description_
    """
    hull = ConvexHull(points)
    faces = np.column_stack(
        (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
    ).flatten()
    poly = pv.PolyData(hull.points, faces)

    return poly


def make_cuboid():
    """_summary_

    Returns:
        _type_: _description_
    """
    pm = PlasmaMesh(distance_between_points=100, cuboid_size=[1800, 800, 2000])
    reduced_cuboid_coordinates = pm.outer_cube_mesh
    print("Cuboid generated!")
    return reduced_cuboid_coordinates


def read_Reff_coordinates():
    """_summary_

    Returns:
        _type_: _description_
    """
    Reff = (
        pathlib.Path.cwd().parents[0]
        / "_Input_files"
        / "Reff"
        / "Reff_coordinates-10_mm.txt"
    )
    print(Reff)
    Reff_VMEC_calculated = np.loadtxt(Reff)  ### [mm]

    return Reff_VMEC_calculated


def make_mapped_plasma_volume(Reff_VMEC_calculated):
    """_summary_

    Args:
        Reff_VMEC_calculated (_type_): _description_

    Returns:
        _type_: _description_
    """
    mapped_plasma_volume = Reff_VMEC_calculated[
        ~np.isnan(Reff_VMEC_calculated).any(axis=1)
    ]
    mapped_plasma_volume = mapped_plasma_volume[:, 1:4]
    print("Mapped plasma volume generated!")

    return mapped_plasma_volume


def make_observed_plasma_volume(Reff_VMEC_calculated, element):
    """_summary_

    Args:
        Reff_VMEC_calculated (_type_): _description_
        element (_type_): _description_

    Returns:
        _type_: _description_
    """
    calculated_plasma_coordinates = (
        pathlib.Path.cwd().parents[0]
        # / "src"
        / "_Input_files"
        / "__Visualization"
        / f"{element}_plasma_coordinates.csv"
    )  # / "Reff" / f"{reff_file_name}.txt"

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
        """Creates the point cloud of the

        Args:
            coordinates (_type_): _description_
            reff (_type_): _description_

        Returns:
            _type_: _description_
        """
        point_cloud = pv.PolyData(coordinates)
        point_cloud["Reff [m]"] = reff

        return point_cloud

    pc = create_point_cloud(plasma_coordinates, reff)
    print(f"Observed plasma volume of {element} generated!")

    return plasma_coordinates, pc


def polar2cart(r, theta, phi):
    """_summary_

    Args:
        r (_type_): _description_
        theta (_type_): _description_
        phi (_type_): _description_

    Returns:
        _type_: _description_
    """
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)

    return x, y, z


def calculate_plasma_surfaces(phi, surface_number):
    """_summary_

    Args:
        phi (_type_): _description_
        surface_number (_type_): _description_

    Returns:
        _type_: _description_
    """
    plasma_surfaces = []
    W7X_surface = (
        pathlib.Path.cwd().parents[0]
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
    """_summary_

    Args:
        phi_range (_type_): _description_
        layer_of_plasma_surface (_type_): _description_

    Returns:
        _type_: _description_
    """
    all_plasma_surfaces = []
    for phi in phi_range:
        local_plasma_surface = calculate_plasma_surfaces(phi, layer_of_plasma_surface)
        all_plasma_surfaces.append(local_plasma_surface)
    all_plasma_surfaces = np.array(all_plasma_surfaces).reshape(-1, 3)
    print("Plasma surface generated!")

    return all_plasma_surfaces


def make_axis(phi_range):
    """_summary_

    Args:
        phi_range (_type_): _description_

    Returns:
        _type_: _description_
    """
    plasma_axis = []
    for phi in phi_range:
        local_axis_point = calculate_plasma_surfaces(phi, 0)[0]
        plasma_axis.append(local_axis_point)
    plasma_axis = np.array(plasma_axis).reshape(-1, 3)
    print("Axis generated!")

    return plasma_axis


if __name__ == "__main__":

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
            color="red",
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
        if polydata == True:
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
        port = pt.calculate_port_hull()
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
                fig.add_mesh(protective_shield.poly_hull, color="green", opacity=0.9)

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

    elements_list = {"B": "red", "C": "blue", "N": "green", "O": "orange"}
    Reff_VMEC_calculated = read_Reff_coordinates()
    plot_W7X()
    plot_cuboid()
    plot_mapped_plasma(Reff_VMEC_calculated)
    port()
    radiation_shields()

    for element in elements_list:
        plot_observed_plasma(
            Reff_VMEC_calculated, element, color=elements_list[element], polydata=False
        )
        dispersive_elements(element, 10, 40)
        collimators(element, closing_side="top")
        detectors(element)

    ###########################################################################
    fig.show()
