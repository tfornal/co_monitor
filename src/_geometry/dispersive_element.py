import json
from math import degrees
from pathlib import Path

import numpy as np
import pyvista as pv
from sympy import Point3D, Line3D

from rotation_matrix import rotation_matrix


class DispersiveElement:
    def __init__(self, element, crystal_height_step = 10, crystal_length_step = 20):
        """
        This class creates the curved surface of the selected dispersive element
        with given coordinates.
        Parameters
        ----------
            Elements symbol like B, C, N or O in order to select proper coordinates
            from the "optics_coordinates" file containing all the necessary input; .
        crystal_height_step : int, optional
            Accuracy of the crystals length mesh (by default 10 points).
        crystal_length_step : int, optional
            Accuracy of the crystals length mesh (by default 20 points).
        """
        self.height_step = crystal_height_step
        self.length_step = crystal_length_step
        self.disp_elem_coord = self.get_coordinates(element)
        self.AOI = self.disp_elem_coord["AOI"]
        self.max_reflectivity = self.disp_elem_coord["max reflectivity"]
        self.crystal_central_point = np.array(
            self.disp_elem_coord["crystal central point"]
        )
        self.radius_central_point = np.array(
            self.disp_elem_coord["radius central point"]
        )
        self.AOI = np.array(self.disp_elem_coord["AOI"])
        self.A = np.array(self.disp_elem_coord["vertex"]["A"])
        self.B = np.array(self.disp_elem_coord["vertex"]["B"])
        self.C = np.array(self.disp_elem_coord["vertex"]["C"])
        self.D = np.array(self.disp_elem_coord["vertex"]["D"])
        self.R = self.distance_between_poinst(self.A, self.radius_central_point)
        self.crystal_orientation_vector = self.B - self.C
        self.alpha = self.angle_between_lines(self.radius_central_point, self.A, self.B)

    @staticmethod
    def get_coordinates(element):
        """_summary_
        Args:
            element (_type_): _description_
        Returns:
            _type_: _description_
        """

        with open(
            Path(__file__).parent.resolve() / "coordinates.json") as file:
            json_file = json.load(file)
            disp_elem_coord = json_file["dispersive element"]["element"][f"{element}"]
        return disp_elem_coord

    @staticmethod
    def distance_between_poinst(p1, p2):
        """Accepts only np.array containing 3 int/float
        Args:
            p1 (np.array): first point in carthesian coordinate system (in mm)
            p2 (np.array): second point in carthesian coordinate system (in mm)
        Returns:
            int: distance (in mm) between the points
        """
        squared_dist = np.sum((p1 - p2) ** 2, axis=0)
        R = np.sqrt(squared_dist).round(2)
        return int(R)


    def shift_cylinder(self, object_coordinates):
        """
        Adds vector to 2D array (N, 3) of 3D object row by row.
        Input - 2d array of interest, vector to add.
        Return - 2d array with included vector.
        """
        coordinates_plus_vector = object_coordinates + self.crystal_orientation_vector

        return coordinates_plus_vector

    def angle_between_lines(self, central_point, p1, p2):
        """
        Returns angle between two lines in 3D carthesian coordinate system.
        Returns
        -------
        angle : float
            angle in degrees.
        """
        central_point, p1, p2 = Point3D(central_point), Point3D(p1), Point3D(p2)
        l1, l2 = Line3D(central_point, p1), Line3D(central_point, p2)
        angle = degrees(l1.angle_between(l2))

        return angle
    

    def make_curved_crystal(self):
        """_summary_
        Returns:
            _type_: _description_
        """

        crys_height = np.linspace(0, 20, self.height_step)  # crystal height range

        crys_lenght = (
            np.linspace(
                0,
                self.alpha,
                num=self.length_step,
            )
            / 180
            * np.pi
        )
        X = self.R * np.cos(crys_lenght)
        Y = self.R * np.sin(crys_lenght)

        px = np.repeat(X, self.height_step)
        py = np.repeat(Y, self.height_step)
        pz = np.tile(crys_height, self.length_step)
        crstal_points = np.vstack((pz, px, py)).T

        ovec = self.crystal_orientation_vector / (
            np.linalg.norm(self.crystal_orientation_vector)
        )
        cylvec = np.array([1, 0, 0])

        if np.allclose(cylvec, ovec):
            return crstal_points

        oaxis = np.cross(ovec, cylvec)
        rot = np.arccos(np.dot(ovec, cylvec))

        rot_matrix = rotation_matrix(rot, oaxis)
        crstal_points = crstal_points.dot(rot_matrix)

        shift = np.array(
            self.radius_central_point - self.crystal_orientation_vector / 2
        )

        angle = self.angle_between_lines(
            self.radius_central_point, self.C, crstal_points[0]
        )
        crstal_points += shift

        ### angle checkout
        angle = self.angle_between_lines(
            self.radius_central_point, self.C, crstal_points[0]
        )

        ### move to starting position
        crstal_points -= shift

        ### rotate by the calculated angle and shift again to the destination place
        rot_matrix = rotation_matrix(
            np.deg2rad(angle), self.crystal_orientation_vector
        )
        crstal_points = crstal_points.dot(rot_matrix)
        crstal_points += shift

        return crstal_points


if __name__ == "__main__":
    
    def plot_dispersive_elements():
        fig = pv.Plotter()
        fig.set_background("black")
        disp_elem = ["B", "O", "N", "C"]
        for element in disp_elem:
            disp = DispersiveElement(element, 20, 80)
            crys = disp.make_curved_crystal()
            fig.add_mesh(
                np.array([disp.A, disp.B, disp.C, disp.D]),
                color="red",
                render_points_as_spheres=True,
                point_size=10,
            )
    
            fig.add_mesh(
                disp.C, color="yellow", render_points_as_spheres=True, point_size=10
            )
            fig.add_mesh(
                crys[0],
                color="purple",
                render_points_as_spheres=True,
                point_size=10,
            )
            fig.add_mesh(
                crys,
                color="orange",
                render_points_as_spheres=True,
                point_size=3,
            )
        fig.show()
        
    plot_dispersive_elements()
