import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from sympy import Point3D, Line3D
from math import degrees
from optics_coordinates_OOP import DispersiveElementsCoordinates
import pathlib
import json


class DispersiveElement:
    def __init__(self, element, crystal_height_step=10, crystal_length_step=20):
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
        TODO test!!!!
        ALE CHUJNIA! POPRAWIC TODO
        """

        self.disp_elem_coord = self.get_coordinates(element)

        self.AOI = self.disp_elem_coord["AOI"]
        print(self.AOI)
        self.max_reflectivity = self.disp_elem_coord["max reflectivity"]
        self.crystal_central_point = self.disp_elem_coord["crystal central point"]
        self.radius_central_point = self.disp_elem_coord["radius central point"]
        self.A = self.disp_elem_coord["vertex"]["A"]
        self.B = self.disp_elem_coord["vertex"]["B"]
        self.C = self.disp_elem_coord["vertex"]["C"]
        self.D = self.disp_elem_coord["vertex"]["D"]
        print(self.crystal_central_point)
        self.R = self.distance_between_poinst(
            np.array(self.crystal_central_point), np.array(self.radius_central_point)
        )
        print(self.R)
        # self.dec = DispersiveElementsCoordinates()
        # self.element = element
        # self.crystals_coordinates = self.choose_element()
        # (
        #     self.vertices_coordinates,
        #     self.A,
        #     self.B,
        #     self.C,
        #     self.D,
        #     self.crystal_central_point,
        #     self.radius_central_point,
        #     self.R,
        #     self.AOI,
        #     self.max_reflectivity,
        # ) = self.crystals_coordinates
        # self.crystal_orientation_vector = self.check_orientation()
        # self.shifted_radius_central_point = self.define_radius_central_point()
        # self.crystal_height_step = crystal_height_step
        # self.alpha = self.angle_between_lines(self.radius_central_point, self.A, self.B)
        # self.crystal_length_step = crystal_length_step
        # self.shift_angle = self.calculate_shift_angle()

    def get_coordinates(self, element):
        """_summary_

        Args:
            element (_type_): _description_

        Returns:
            _type_: _description_
        """
        with open(
            pathlib.Path.cwd() / "src" / "_geometry" / "coordinates.json"
        ) as file:
            json_file = json.load(file)
            disp_elem_coord = json_file["dispersive element"]["element"][f"{element}"]
        return disp_elem_coord

    def distance_between_poinst(self, p1, p2):
        """Accepts only np.array containing 3 int/float

        Args:
            p1 (_type_): _description_
            p2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        squared_dist = np.sum((p1 - p2) ** 2, axis=0)
        R = np.sqrt(squared_dist).round(2)

        return R

    def choose_element(self):
        """
        Checks wich of the element should be investigated.

        Return - array of dispersive elements coordinates and orientation
                of lable each crystal (from its top to bottom) readed from the database.
        """
        # namedtuple?
        if str(self.element) == "B":
            crystal_coordinates = self.dec.vertices_crys_B()
        elif str(self.element) == "C":
            crystal_coordinates = self.dec.vertices_crys_C()
        elif str(self.element) == "N":
            crystal_coordinates = self.dec.vertices_crys_N()
        elif str(self.element) == "O":
            crystal_coordinates = self.dec.vertices_crys_O()

        return crystal_coordinates

    def check_orientation(self):
        """
        Checks the orientation of a crystal from its top to bottom - explain later

        Input - element symbol.
        Return - array of orientation of each crystal (from its top to bottom) readed from the database.
        """
        orientation = self.B - self.C

        return orientation

    def define_radius_central_point(self):
        if str(self.element) in ["N", "B"]:
            shifted_radius_central_point = self.radius_central_point - (
                self.crystal_orientation_vector / 2
            )
        elif str(self.element) in ["O", "C"]:
            shifted_radius_central_point = self.radius_central_point - (
                self.crystal_orientation_vector / 2
            )

        return shifted_radius_central_point

    def shift_cylinder(self, object_coordinates, crystal_orientation_vector):
        """
        Adds vector to 2D array (N, 3) of 3D object row by row.

        Input - 2d array of interest, vector to add.
        Return - 2d array with included vector.
        """
        coordinates_plus_vector = np.zeros([len(object_coordinates), 3])
        for row_nr, j in enumerate(object_coordinates):
            coordinates_plus_vector[row_nr, 0] = j[0] + crystal_orientation_vector[0]
            coordinates_plus_vector[row_nr, 1] = j[1] + crystal_orientation_vector[1]
            coordinates_plus_vector[row_nr, 2] = j[2] + crystal_orientation_vector[2]

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

    def rotation_matrix_3D(self, theta, axis):
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

    def calculate_shift_angle(self):
        """TODO!!!!!!!!!!!!!!!!"""
        I = np.linspace(0, 20, self.crystal_height_step)  # crystal height range

        if int(self.alpha) == 360:
            A = (
                np.linspace(0, self.alpha, num=self.crystal_length_step, endpoint=False)
                / 180
                * np.pi
            )
        else:
            A = np.linspace(0, self.alpha, num=self.crystal_length_step) / 180 * np.pi

        X = self.R * np.cos(A)
        Y = self.R * np.sin(A)

        px = np.repeat(X, self.crystal_height_step)
        py = np.repeat(Y, self.crystal_height_step)
        pz = np.tile(I, self.crystal_length_step)
        points = np.vstack((pz, px, py)).T

        ovec = self.crystal_orientation_vector / (
            np.linalg.norm(self.crystal_orientation_vector)
        )
        cylvec = np.array([1, 0, 0])

        if np.allclose(cylvec, ovec):
            return points

        oaxis = np.cross(ovec, cylvec)
        rot = np.arccos(np.dot(ovec, cylvec))

        R = self.rotation_matrix_3D(rot, oaxis)
        cylinder_points = points.dot(R)

        shifted_cylinder_points = self.shift_cylinder(
            cylinder_points, self.shifted_radius_central_point
        )
        cylinders_first_coordinate = shifted_cylinder_points[0]
        shift_angle = self.angle_between_lines(
            self.shifted_radius_central_point, self.A, cylinders_first_coordinate
        )

        return shift_angle

    def make_curved_crystal(self):
        """TODO!!!!!!!!!!!!!!!!"""

        """I tutaj zaczyna sie powtorzenie - trzeba to inaczej napisac!!!!"""
        I = np.linspace(0, 20, self.crystal_height_step)  # crystal height range

        """  POPRAWKA DLA KRYSZATLU BV - sprawdzic keidy sin(a) <0 a kiedy >0 - 
        na tej podstawie okreslic czy shift angle ma wartosc + czy -"""
        """  POPRAWKA DLA KRYSZATLU OVIII "-0.3" STOPNIA ABY CYLINDER SIE DODATKOWO OBROCIL
        KONIECZNOSC WPROWADZENIA ZMIAN WE WSPOLRZEDNYCH PRAWDOPDOOBNIE PUNKTU OSI PROMIENIA KRYSZATLU"""

        if self.R == 1419.8:
            self.shift_angle = -self.shift_angle

            if int(self.alpha) == 360:
                A = (
                    np.linspace(
                        0, self.alpha, num=self.crystal_length_step, endpoint=False
                    )
                    / 180
                    * np.pi
                )
            else:
                A = (
                    np.linspace(
                        self.shift_angle,
                        self.shift_angle - self.alpha,
                        num=self.crystal_length_step,
                    )
                    / 180
                    * np.pi
                )

        if self.R == 680:
            self.shift_angle = self.shift_angle - 0.3

            if int(self.alpha) == 360:
                A = (
                    np.linspace(
                        0, self.alpha, num=self.crystal_length_step, endpoint=False
                    )
                    / 180
                    * np.pi
                )
            else:
                A = (
                    np.linspace(
                        self.shift_angle,
                        self.shift_angle - self.alpha,
                        num=self.crystal_length_step,
                    )
                    / 180
                    * np.pi
                )
        else:
            if int(self.alpha) == 360:
                A = (
                    np.linspace(
                        0, self.alpha, num=self.crystal_length_step, endpoint=False
                    )
                    / 180
                    * np.pi
                )
            else:
                A = (
                    np.linspace(
                        self.shift_angle,
                        self.shift_angle - self.alpha,
                        num=self.crystal_length_step,
                    )
                    / 180
                    * np.pi
                )

        X = self.R * np.cos(A)
        Y = self.R * np.sin(A)

        px = np.repeat(X, self.crystal_height_step)
        py = np.repeat(Y, self.crystal_height_step)
        pz = np.tile(I, self.crystal_length_step)
        points = np.vstack((pz, px, py)).T

        ovec = self.crystal_orientation_vector / (
            np.linalg.norm(self.crystal_orientation_vector)
        )
        cylvec = np.array([1, 0, 0])

        if np.allclose(cylvec, ovec):
            return points

        oaxis = np.cross(ovec, cylvec)
        rot = np.arccos(np.dot(ovec, cylvec))

        R = self.rotation_matrix_3D(rot, oaxis)
        cylinder_points = points.dot(R)

        positioned_crystal_surface = self.shift_cylinder(
            cylinder_points, self.shifted_radius_central_point
        )

        return positioned_crystal_surface


if __name__ == "__main__":

    disp = DispersiveElement("C", 20, 80)

    # disp_elem = ["B", "C", "N", "O"]
    # fig = pv.Plotter()
    # fig.set_background("black")
    # for element in disp_elem:
    #     disp = DispersiveElement(element, 20, 80)
    #     crys = disp.make_curved_crystal()

    #     fig.add_mesh(disp.crystal_central_point, color="blue", point_size=10)
    #     fig.add_mesh(disp.A, color="red", point_size=10)
    #     fig.add_mesh(disp.B, color="red", point_size=10)
    #     fig.add_mesh(disp.C, color="red", point_size=10)
    #     fig.add_mesh(disp.D, color="red", point_size=10)
    #     fig.add_mesh(crys, color="orange", render_points_as_spheres=True)

    # fig.show()
