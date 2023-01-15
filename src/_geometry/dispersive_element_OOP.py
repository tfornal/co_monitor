import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from sympy import Point3D, Line3D
from math import degrees
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
        TODO - zmiana promienia krzywizny krysztalu na dowolna wartoć - zahardkodowanie pozycji krysztalu w konkretnej pozycji
        """

        self.height_step = crystal_height_step
        self.length_step = crystal_length_step
        self.disp_elem_coord = self.get_coordinates(element)
        self.AOI = self.disp_elem_coord["AOI"]
        self.max_reflectivity = self.disp_elem_coord["max reflectivity"]
        self.crystal_central_point = np.array(
            self.disp_elem_coord["crystal central point"]
        )
        self.radius_central_point = self.disp_elem_coord["radius central point"]
        self.A = np.array(self.disp_elem_coord["vertex"]["A"])
        self.B = np.array(self.disp_elem_coord["vertex"]["B"])
        self.C = np.array(self.disp_elem_coord["vertex"]["C"])
        self.D = np.array(self.disp_elem_coord["vertex"]["D"])
        self.R = self.distance_between_poinst(
            self.crystal_central_point, self.radius_central_point
        )
        self.crystal_orientation_vector = self.B - self.C

        self.crys_ax = (
            np.array(self.radius_central_point) + self.crystal_orientation_vector
        )
        #############
        self.srodek = (self.A + self.B) / 2  # - (self.C + self.D) / 2
        ##########

        self.shifted_radius_central_point = self.define_radius_central_point()

        self.alpha = self.angle_between_lines(self.radius_central_point, self.A, self.B)
        # self.alpha = 350
        # self.shift_angle = self.calculate_shift_angle()
        print(element, self.R)

    @staticmethod
    def get_coordinates(element):
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

    @staticmethod
    def distance_between_poinst(p1, p2):
        """Accepts only np.array containing 3 int/float
        Args:
            p1 (_type_): _description_
            p2 (_type_): _description_
        Returns:
            _type_: _description_
        """
        squared_dist = np.sum((p1 - p2) ** 2, axis=0)
        R = np.sqrt(squared_dist).round(2)

        return int(R)

    def define_radius_central_point(self):
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
        coordinates_plus_vector = object_coordinates + crystal_orientation_vector

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
        Input - angle (in radians), axis in carthesian coordinates [x,y,z].
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

    def make_curved_crystal(self):
        """_summary_
        Returns:
            _type_: _description_
        """
        crys_height = np.linspace(0, 20, self.height_step)  # crystal height range

        crys_lenght = (
            np.linspace(
                0,
                # self.alpha,
                270,
                num=self.length_step,
            )
            / 180
            * np.pi
        )
        ### definiowanie macierzy punktow reprezentujacych powierzchnie krysztalu
        X = self.R * np.cos(crys_lenght)
        Y = self.R * np.sin(crys_lenght)

        px = np.repeat(X, self.height_step)
        py = np.repeat(Y, self.height_step)
        pz = np.tile(crys_height, self.length_step)
        points = np.vstack((pz, px, py)).T

        ## ustawianie macierzy powierzchni zgodnie z orientacja osi krysztalu

        ovec = self.crystal_orientation_vector / (
            np.linalg.norm(self.crystal_orientation_vector)
        )
        cylvec = np.array([1, 0, 0])

        if np.allclose(cylvec, ovec):
            return points

        oaxis = np.cross(ovec, cylvec)
        rot = np.arccos(np.dot(ovec, cylvec))

        # rot = 15
        # oaxis = self.crystal_orientation_vector

        rot_matrix = self.rotation_matrix_3D(rot, oaxis)
        points = points.dot(rot_matrix)
        ### pozostalo zdefiniowac kąt obrotu/?
        ### ustawic odpowiedni kąt...
        #### przesuniecie
        x = self.rotation_matrix_3D(np.deg2rad(27.5), self.crystal_orientation_vector)
        points = points.dot(x)

        ## pnowny obrot

        shift = np.array(
            self.radius_central_point - self.crystal_orientation_vector / 2
        )

        angle = self.angle_between_lines(self.crys_ax, self.C, points[-1])
        print(angle)  #### nieprawidlowo liczy kąt!!!!!!!!!

        ### kolejne przesuniecie
        # angle = self.angle_between_lines(self.crys_ax, self.C, points[0])
        # x = self.rotation_matrix_3D(np.deg2rad(angle), self.crystal_orientation_vector)
        # print(angle)
        # points = points.dot(x)

        # points += shift

        fig = pv.Plotter()
        fig.set_background("black")
        fig.add_mesh(
            self.crys_ax, color="red", render_points_as_spheres=True, point_size=10
        )
        # fig.add_mesh(
        #     self.A, color="yellow", render_points_as_spheres=True, point_size=10
        # )
        # fig.add_mesh(
        #     self.B, color="yellow", render_points_as_spheres=True, point_size=10
        # # )
        fig.add_mesh(
            self.C, color="yellow", render_points_as_spheres=True, point_size=10
        )
        # fig.add_mesh(
        #     self.D, color="yellow", render_points_as_spheres=True, point_size=10
        # )
        # fig.add_mesh(points, color="green", render_points_as_spheres=True)
        fig.add_mesh(
            points[0], color="purple", render_points_as_spheres=True, point_size=10
        )
        points += shift
        # fig.add_mesh(np.array(disp.crystal_central_point), color="blue", point_size=10)
        # fig.add_mesh(disp.srodek, color="green", point_size=20)
        # # fig.add_mesh(np.array([0, 0, 0]), color="green", render_points_as_spheres=True)
        # fig.add_mesh(self.A, color="red", point_size=20)
        # fig.add_mesh(self.B, color="red", point_size=10)
        # fig.add_mesh(self.C, color="red", point_size=10)
        # fig.add_mesh(self.D, color="red", point_size=10)
        # fig.add_mesh(self.crys_ax, color="purple", point_size=15)
        # fig.add_mesh(points, color="orange", render_points_as_spheres=True)

        fig.show()

        return points


if __name__ == "__main__":
    disp = DispersiveElement("B", 20, 8000)
    disp.make_curved_crystal()

    # fig = pv.Plotter()
    # fig.set_background("black")
    # disp_elem = ["B"]
    # for element in disp_elem:
    #     disp = DispersiveElement(element, 20, 8000)
    #     crys = disp.make_curved_crystal()

    #     # fig.add_mesh(np.array(disp.crystal_central_point), color="blue", point_size=10)
    #     fig.add_mesh(disp.srodek, color="green", point_size=20)
    #     # fig.add_mesh(np.array([0, 0, 0]), color="green", render_points_as_spheres=True)
    #     fig.add_mesh(disp.A, color="red", point_size=20)
    #     fig.add_mesh(disp.B, color="red", point_size=10)
    #     fig.add_mesh(disp.C, color="red", point_size=10)
    #     fig.add_mesh(disp.D, color="red", point_size=10)
    #     fig.add_mesh(disp.crys_ax, color="purple", point_size=15)
    #     fig.add_mesh(crys, color="orange", render_points_as_spheres=True)
    #     # fig.add_mesh(
    #     #     crys[-1], color="red", render_points_as_spheres=True, point_size=20
    #     # )

    # fig.show()


###########BACKUP


# import numpy as np

# import pyvista as pv
# from pyvistaqt import BackgroundPlotter
# from sympy import Point3D, Line3D
# from math import degrees
# import pathlib
# import json


# class DispersiveElement:

#     def __init__(self, element, crystal_height_step=10, crystal_length_step=20):
#         """
#         This class creates the curved surface of the selected dispersive element
#         with given coordinates.

#         Parameters
#         ----------
#             Elements symbol like B, C, N or O in order to select proper coordinates
#             from the "optics_coordinates" file containing all the necessary input; .
#         crystal_height_step : int, optional
#             Accuracy of the crystals length mesh (by default 10 points).
#         crystal_length_step : int, optional
#             Accuracy of the crystals length mesh (by default 20 points).
#         TODO test!!!!
#         ALE CHUJNIA! POPRAWIC TODO
#         TODO - zmiana promienia krzywizny krysztalu na dowolna wartoć - zahardkodowanie pozycji krysztalu w konkretnej pozycji
#         """

#         self.height_step = crystal_height_step
#         self.length_step = crystal_length_step
#         self.disp_elem_coord = self.get_coordinates(element)
#         self.AOI = self.disp_elem_coord["AOI"]
#         self.max_reflectivity = self.disp_elem_coord["max reflectivity"]
#         self.crystal_central_point = np.array(
#             self.disp_elem_coord["crystal central point"]
#         )
#         self.radius_central_point = self.disp_elem_coord["radius central point"]
#         self.A = np.array(self.disp_elem_coord["vertex"]["A"])
#         self.B = np.array(self.disp_elem_coord["vertex"]["B"])
#         self.C = np.array(self.disp_elem_coord["vertex"]["C"])
#         self.D = np.array(self.disp_elem_coord["vertex"]["D"])
#         self.R = self.distance_between_poinst(
#             self.crystal_central_point, self.radius_central_point
#         )
#         self.crystal_orientation_vector = self.B - self.C
#         self.shifted_radius_central_point = self.define_radius_central_point()

#         self.alpha = self.angle_between_lines(self.radius_central_point, self.A, self.B)

#         self.shift_angle = self.calculate_shift_angle()

#     @staticmethod
#     def get_coordinates(element):
#         """_summary_

#         Args:
#             element (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         with open(
#             pathlib.Path.cwd() / "coordinates.json"
#         ) as file:
#             json_file = json.load(file)
#             disp_elem_coord = json_file["dispersive element"]["element"][f"{element}"]
#         return disp_elem_coord

#     @staticmethod
#     def distance_between_poinst(p1, p2):
#         """Accepts only np.array containing 3 int/float

#         Args:
#             p1 (_type_): _description_
#             p2 (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         squared_dist = np.sum((p1 - p2) ** 2, axis=0)
#         R = np.sqrt(squared_dist).round(2)

#         return R

#     def define_radius_central_point(self):
#         shifted_radius_central_point = self.radius_central_point - (
#             self.crystal_orientation_vector / 2
#         )

#         return shifted_radius_central_point

#     def shift_cylinder(self, object_coordinates, crystal_orientation_vector):
#         """
#         Adds vector to 2D array (N, 3) of 3D object row by row.

#         Input - 2d array of interest, vector to add.
#         Return - 2d array with included vector.
#         """
#         coordinates_plus_vector = object_coordinates + crystal_orientation_vector

#         return coordinates_plus_vector

#     def angle_between_lines(self, central_point, p1, p2):
#         """
#         Returns angle between two lines in 3D carthesian coordinate system.

#         Returns
#         -------
#         angle : float
#             angle in degrees.

#         """
#         central_point, p1, p2 = Point3D(central_point), Point3D(p1), Point3D(p2)
#         l1, l2 = Line3D(central_point, p1), Line3D(central_point, p2)
#         angle = degrees(l1.angle_between(l2))

#         return angle

#     def rotation_matrix_3D(self, theta, axis):
#         """
#         Rotation matrix based on the Euler-Rodrigues formula for matrix conversion of 3d object.

#         Input - angle, axis in carthesian coordinates [x,y,z].
#         Return - rotated  matrix.
#         """
#         axis = np.asarray(axis) / np.sqrt(np.dot(axis, axis))
#         a = np.cos(theta / 2.0)
#         b, c, d = -axis * np.sin(theta / 2.0)
#         aa, bb, cc, dd = a**2, b**2, c**2, d**2
#         bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

#         return np.array(
#             [
#                 [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
#                 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
#                 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
#             ]
#         )

#     def helper():
#         pass

#     def calculate_shift_angle(self):
#         """TODO!!!!!!!!!!!!!!!!"""
#         crys_height = np.linspace(0, 20, self.height_step)
#         crys_lenght = np.linspace(0, self.alpha, num=self.length_step) / 180 * np.pi

#         X = self.R * np.cos(crys_lenght)
#         Y = self.R * np.sin(crys_lenght)

#         px = np.repeat(X, self.height_step)
#         py = np.repeat(Y, self.height_step)
#         pz = np.tile(crys_height, self.length_step)
#         points = np.vstack((pz, px, py)).T
#         breakpoint()

#         ovec = self.crystal_orientation_vector / (
#             np.linalg.norm(self.crystal_orientation_vector)
#         )
#         cylvec = np.array([1, 0, 0])

#         if np.allclose(cylvec, ovec):
#             return points

#         oaxis = np.cross(ovec, cylvec)
#         rot = np.arccos(np.dot(ovec, cylvec))

#         R = self.rotation_matrix_3D(rot, oaxis)
#         cylinder_points = points.dot(R)

#         shifted_cylinder_points = self.shift_cylinder(
#             cylinder_points, self.shifted_radius_central_point
#         )
#         cylinders_first_coordinate = shifted_cylinder_points[0]
#         shift_angle = self.angle_between_lines(
#             self.shifted_radius_central_point, self.A, cylinders_first_coordinate
#         )
#         print(float("{:.2f}".format(shift_angle)))
#         return float("{:.2f}".format(shift_angle))

#     def make_curved_crystal(self):
#         """TODO!!!!!!!!!!!!!!!!"""

#         """I tutaj zaczyna sie powtorzenie - trzeba to inaczej napisac!!!!"""
#         I = np.linspace(0, 20, self.height_step)  # crystal height range

#         """  POPRAWKA DLA KRYSZATLU BV - sprawdzic keidy sin(a) <0 a kiedy >0 -
#         na tej podstawie okreslic czy shift angle ma wartosc + czy -"""
#         """  POPRAWKA DLA KRYSZATLU OVIII "-0.3" STOPNIA ABY CYLINDER SIE DODATKOWO OBROCIL
#         KONIECZNOSC WPROWADZENIA ZMIAN WE WSPOLRZEDNYCH PRAWDOPDOOBNIE PUNKTU OSI PROMIENIA KRYSZATLU"""
#         if self.R == 1419.8:
#             self.shift_angle = -self.shift_angle

#             if int(self.alpha) == 360:
#                 A = (
#                     np.linspace(
#                         0, self.alpha, num=self.length_step, endpoint=False
#                     )
#                     / 180
#                     * np.pi
#                 )
#             else:
#                 A = (
#                     np.linspace(
#                         self.shift_angle,
#                         self.shift_angle - self.alpha,
#                         num=self.length_step,
#                     )
#                     / 180
#                     * np.pi
#                 )

#         if self.R == 680:
#             self.shift_angle = self.shift_angle - 0.3

#             if int(self.alpha) == 360:
#                 A = (
#                     np.linspace(
#                         0, self.alpha, num=self.length_step, endpoint=False
#                     )
#                     / 180
#                     * np.pi
#                 )
#             else:
#                 A = (
#                     np.linspace(
#                         self.shift_angle,
#                         self.shift_angle - self.alpha,
#                         num=self.length_step,
#                     )
#                     / 180
#                     * np.pi
#                 )
#         else:
#             if int(self.alpha) == 360:
#                 A = (
#                     np.linspace(
#                         0, self.alpha, num=self.length_step, endpoint=False
#                     )
#                     / 180
#                     * np.pi
#                 )
#             else:
#                 A = (
#                     np.linspace(
#                         self.shift_angle,
#                         self.shift_angle - self.alpha,
#                         num=self.length_step,
#                     )
#                     / 180
#                     * np.pi
#                 )

#         X = self.R * np.cos(A)
#         Y = self.R * np.sin(A)

#         px = np.repeat(X, self.height_step )
#         py = np.repeat(Y, self.height_step )
#         pz = np.tile(I, self.length_step)
#         points = np.vstack((pz, px, py)).T

#         ovec = self.crystal_orientation_vector / (
#             np.linalg.norm(self.crystal_orientation_vector)
#         )
#         cylvec = np.array([1, 0, 0])

#         if np.allclose(cylvec, ovec):
#             return points

#         oaxis = np.cross(ovec, cylvec)
#         rot = np.arccos(np.dot(ovec, cylvec))

#         R = self.rotation_matrix_3D(rot, oaxis)
#         cylinder_points = points.dot(R)

#         positioned_crystal_surface = self.shift_cylinder(
#             cylinder_points, self.shifted_radius_central_point
#         )

#         return positioned_crystal_surface


# if __name__ == "__main__":
#     disp_elem = ["B", "C", "N", "O"]
#     fig = pv.Plotter()
#     fig.set_background("black")
#     for element in disp_elem:
#         disp = DispersiveElement(element, 20, 80)
#         crys = disp.make_curved_crystal()

#         fig.add_mesh(np.array(disp.crystal_central_point), color="blue", point_size=10)
#         fig.add_mesh(disp.A, color="red", point_size=10)
#         fig.add_mesh(disp.B, color="red", point_size=10)
#         fig.add_mesh(disp.C, color="red", point_size=10)
#         fig.add_mesh(disp.D, color="red", point_size=10)
#         fig.add_mesh(crys, color="orange", render_points_as_spheres=True)

#     fig.show()
