import numpy as np
from sympy import Point3D, Plane
import math
import json


class Methods:
    ### wrzucic w inne miejsce
    def define_plane_equation(self, p1, p2, p3):
        plane = Plane(Point3D(p1), Point3D(p2), Point3D(p3))
        plane_equation = plane.equation()

        return plane, plane_equation

    def find_coplanar_vertex(self, A, B, C, D):
        plane = self.define_plane_equation(A, B, C)[0]
        line_perpendicular_to_plane = plane.perpendicular_line(Point3D(D))
        point_E = list(*plane.intersection(line_perpendicular_to_plane))
        point_E = np.array(point_E)
        """  sdasda """
        return point_E

    def distance_between_poinst(self, p1, p2):
        squared_dist = np.sum((p1 - p2) ** 2, axis=0)
        R = np.sqrt(squared_dist).round(2)

        return R


class Plasma:
    def center_of_observed_plasma_volume(self):
        plasma_central_point = np.array([-3430.25, 4305.75, -320.75])
        return plasma_central_point


# class DetectorCoordinates(Methods):
#     def crys_det_orientation_vectors_B(self):
#         orientation_vector = np.array([192.48, 19.95, 8.545])

#         return orientation_vector

#     def crys_det_orientation_vectors_C(self):
#         orientation_vector = np.array([34.568, 190.338, 9.817])

#         return orientation_vector

#     def crys_det_orientation_vectors_N(self):
#         orientation_vector = np.array([59.377, 184.189, -8.255])

#         return orientation_vector

#     def crys_det_orientation_vectors_O(self):
#         orientation_vector = np.array([134.775, 139.114, 1.66])

#         return orientation_vector

#     def vertices_detector_B(self):
#         A = np.array([-6259.727, 6802.972, -589.214])
#         B = np.array([-6257.067, 6776.564, -587.463])
#         C = np.array([-6257.406, 6776.972, -580.784])
#         D = np.array([-6260.065, 6803.381, -582.53])
#         E = self.find_coplanar_vertex(A, B, C, D)
#         vertices_coordinates = np.vstack((A, B, C, D, E)).astype("float64")

#         return vertices_coordinates, A, B, C, D

#     def vertices_detector_C(self):
#         A = np.array([-5438.446, 7589.626, 2.706])
#         B = np.array([-5412.307, 7584.961, 1.106])
#         C = np.array([-5412.643, 7585.367, -5.573])
#         D = np.array([-5438.782, 7590.032, -3.98])
#         E = self.find_coplanar_vertex(A, B, C, D)
#         vertices_coordinates = np.vstack((A, B, C, D, E)).astype("float64")

#         return vertices_coordinates, A, B, C, D

#     def vertices_detector_N(self):
#         A = np.array([-5408.529, 7278.827, -602.019])
#         B = np.array([-5433.813, 7286.898, -603.795])
#         C = np.array([-5434.151, 7287.307, -597.116])
#         D = np.array([-5408.868, 7279.236, -595.3])
#         E = self.find_coplanar_vertex(A, B, C, D)
#         vertices_coordinates = np.vstack((A, B, C, D, E)).astype("float64")
#         return vertices_coordinates, A, B, C, D

#     def vertices_detector_O(self):
#         A = np.array([-5901.357, 6489.582, -67.661])
#         B = np.array([-5920.415, 6508.022, -65.581])
#         C = np.array([-5920.752, 6508.427, -72.261])
#         D = np.array([-5901.693, 6489.988, -74])
#         E = self.find_coplanar_vertex(A, B, C, D)
#         vertices_coordinates = np.vstack((A, B, C, D, E)).astype("float64")

#         return vertices_coordinates, A, B, C, D

#     def define_plane_equation(self, p1, p2, p3):
#         return super().define_plane_equation(p1, p2, p3)

#     def find_coplanar_vertex(self, A, B, C, D):
#         return super().find_coplanar_vertex(A, B, C, D)


class CollimatorsCoordinates(Methods):
    """
    Collimators collimators coordinates (C and O).
    Calculates plane of front and back side of the collimator.
    Finds coplanar vertex from a given list of three vertices (A,B,C).
    Units: [mm]
    TODO: colimatory B i N! (na razie brak wspolrzednych)
    """

    def plas_colim_orientation_vectors(self):
        upper_chamber_vector_direction = np.array([-58.553, 70.592, 7.236])
        bottom_chamber_vector_direction = np.array([-7.637, 9.207, -0.95])

        return upper_chamber_vector_direction, bottom_chamber_vector_direction

    def vertices_collim_B(self, closing_side):
        vector_front_back = np.array([-50.837, 61.29, -6.328])

        if closing_side == "top":
            A1 = np.array([-5495.041, 6707.251, -530.817])
            B1 = np.array([-5494.991, 6707.190, -531.813])
            A2 = np.array([-5418.072, 6771.093, -530.817])
            B2 = np.array([-5418.022, 6771.032, -531.813])
            G = self.find_coplanar_vertex(A1, B1, A2, B2)
            vertices_coordinates = np.vstack((A1, B1, A2, B2, G))
            vector_top_bottom = np.array([0.05, -0.061, -0.996])

        elif closing_side == "bottom":

            A1 = np.array([-5417.111, 6769.934, -549.757])
            B1 = np.array([-5417.162, 6769.995, -548.760])
            A2 = np.array([-5494.080, 6706.093, -549.757])
            B2 = np.array([-5494.131, 6706.154, -548.760])
            G = self.find_coplanar_vertex(A1, B1, A2, B2)
            vertices_coordinates = np.vstack((A1, B1, A2, B2, G))
            vector_top_bottom = np.array([-0.05, 0.061, 0.996])
        return (
            vertices_coordinates,
            A1,
            B1,
            A2,
            B2,
            G,
            vector_top_bottom,
            vector_front_back,
        )

    def vertices_collim_C(self, closing_side):
        vector_front_back = np.array([-50.839, 61.293, 6.282])

        if closing_side == "top":
            A1 = np.array([-5416.597, 6769.376, -59.739])
            B1 = np.array([-5416.547, 6769.316, -58.743])
            A2 = np.array([-5493.566, 6705.535, -59.739])
            B2 = np.array([-5493.516, 6705.474, -58.743])
            G = self.find_coplanar_vertex(A1, B1, A2, B2)
            vertices_coordinates = np.vstack((A1, B1, A2, B2, G))
            vector_top_bottom = np.array([0.05, -0.06, 0.996])

        elif closing_side == "bottom":
            A1 = np.array([-5492.612, 6704.384, -40.798])
            B1 = np.array([-5492.662, 6704.445, -41.795])
            A2 = np.array([-5415.643, 6768.226, -40.798])
            B2 = np.array([-5415.693, 6768.587, -41.795])
            G = self.find_coplanar_vertex(A1, B1, A2, B2)
            vertices_coordinates = np.vstack((A1, B1, A2, B2, G))
            vector_top_bottom = np.array([-0.05, 0.061, -0.997])

        return (
            vertices_coordinates,
            A1,
            B1,
            A2,
            B2,
            G,
            vector_top_bottom,
            vector_front_back,
        )

    def vertices_collim_O(self, closing_side):
        vector_front_back = np.array([-50.839, 61.292, 6.283])

        if closing_side == "top":
            A1 = np.array([-5418.053, 6771.132, -88.65])
            B1 = np.array([-5418.003, 6771.071, -87.653])
            A2 = np.array([-5495.022, 6707.290, -88.65])
            B2 = np.array([-5494.972, 6707.230, -87.653])
            G = self.find_coplanar_vertex(A1, B1, A2, B2)
            vertices_coordinates = np.vstack((A1, B1, A2, B2, G))
            vector_top_bottom = np.array([0.05, -0.061, 0.997])

        elif closing_side == "bottom":
            A1 = np.array([-5494.068, 6706.140, -69.709])
            B1 = np.array([-5494.118, 6706.200, -70.705])
            A2 = np.array([-5417.099, 6769.982, -69.709])
            B2 = np.array([-5417.150, 6770.042, -70.705])
            G = self.find_coplanar_vertex(A1, B1, A2, B2)
            vertices_coordinates = np.vstack((A1, B1, A2, B2, G))
            vector_top_bottom = np.array([-0.05, 0.06, -0.996])

        return (
            vertices_coordinates,
            A1,
            B1,
            A2,
            B2,
            G,
            vector_top_bottom,
            vector_front_back,
        )

    def vertices_collim_N(self, closing_side):
        vector_front_back = np.array([-50.836, 61.29, -6.328])

        if closing_side == "top":
            A1 = np.array([-5493.574, 6705.483, -559.725])
            B1 = np.array([-5493.524, 6705.422, -560.722])
            A2 = np.array([-5416.606, 6769.325, -559.725])
            B2 = np.array([-5416.555, 6769.264, -560.722])
            G = self.find_coplanar_vertex(A1, B1, A2, B2)
            vertices_coordinates = np.vstack((A1, B1, A2, B2, G))
            vector_top_bottom = np.array([0.05, -0.061, -0.997])

        elif closing_side == "bottom":
            A1 = np.array([-5415.645, 6768.166, -578.666])
            B1 = np.array([-5415.695, 6768.227, -577.669])
            A2 = np.array([-5492.613, 6704.324, -578.666])
            B2 = np.array([-5492.664, 6704.385, -577.669])
            G = self.find_coplanar_vertex(A1, B1, A2, B2)
            vertices_coordinates = np.vstack((A1, B1, A2, B2, G))
            vector_top_bottom = np.array([-0.05, 0.061, 0.997])

        return (
            vertices_coordinates,
            A1,
            B1,
            A2,
            B2,
            G,
            vector_top_bottom,
            vector_front_back,
        )

    def define_plane_equation(self, p1, p2, p3):
        return super().define_plane_equation(p1, p2, p3)

    def find_coplanar_vertex(self, A, B, C, D):
        return super().find_coplanar_vertex(A, B, C, D)


class DispersiveElementsCoordinates(Methods):
    """
    Contains information about vertices of each cyllindrically curved crystal,
    crystal central point as well as Radius (R) and central point of its axis.
    Units: [mm]
    Dispersive elements coordinates
    """

    def vertices_crys_B(self):
        AOI = 29.07  ### Angle of Incident [degrees]
        max_reflectivity = 0.255  ### 25.5%
        A = np.array([-5596.52, 6877.643, -545.887])
        B = np.array([-5522.616, 6847.528, -540.295])
        C = np.array([-5521.605, 6846.308, -560.232])
        D = np.array([-5595.509, 6876.423, -565.824])

        E = self.find_coplanar_vertex(A, B, C, D)
        vertices_coordinates = np.vstack((A, B, C, D, E))
        crystal_central_point = np.array([-5558.849, 6862.497, -553.08])
        radius_central_point = np.array([-6097.687, 5549.987, -500.133])
        R = self.distance_between_poinst(crystal_central_point, radius_central_point)

        return (
            vertices_coordinates,
            A,
            B,
            C,
            D,
            crystal_central_point,
            radius_central_point,
            R,
            AOI,
            max_reflectivity,
        )

    def vertices_crys_C(self):
        AOI = 24.94  ### Angle of Incident [degrees]
        max_reflectivity = 0.257  ### 25.7%

        A = np.array([-5546.404, 6821.796, -30.959])
        B = np.array([-5566.607, 6898.993, -25.254])
        C = np.array([-5567.611, 6900.203, -45.192])
        D = np.array([-5547.408, 6823.007, -50.90])
        E = self.find_coplanar_vertex(A, B, C, D)
        vertices_coordinates = np.vstack((A, B, C, D, E))
        crystal_central_point = np.array([-5557.455, 6860.881, -38.06])
        radius_central_point = np.array([-3888.684, 7301.849, -95.335])
        R = self.distance_between_poinst(crystal_central_point, radius_central_point)

        return (
            vertices_coordinates,
            A,
            B,
            C,
            D,
            crystal_central_point,
            radius_central_point,
            R,
            AOI,
            max_reflectivity,
        )

    def vertices_crys_N(self):
        AOI = 29.71  ### Angle of Incident [degrees]
        max_reflectivity = 0.063  ### 6.3%
        A = np.array([-5549.23, 6822.393, -568.999])
        B = np.array([-5564.785, 6900.667, -574.57])
        C = np.array([-5563.773, 6899.448, -594.513])
        D = np.array([-5548.218, 6821.173, -588.936])

        E = self.find_coplanar_vertex(A, B, C, D)
        vertices_coordinates = np.vstack((A, B, C, D, E))
        crystal_central_point = np.array([-5557.392, 6860.741, -581.79])
        radius_central_point = np.array([-4695.14, 7034.455, -548.667])
        R = self.distance_between_poinst(crystal_central_point, radius_central_point)

        return (
            vertices_coordinates,
            A,
            B,
            C,
            D,
            crystal_central_point,
            radius_central_point,
            R,
            AOI,
            max_reflectivity,
        )

    def vertices_crys_O(self):
        AOI = 46.86  ### Angle of Incident [degrees]
        max_reflectivity = 0.05  ### 5% zalozone; TODO

        A = np.array([-5598.349, 6862.422, -54.464])
        B = np.array([-5518.524, 6859.231, -58.679])
        C = np.array([-5519.528, 6860.442, -78.617])
        D = np.array([-5599.353, 6863.353, -74.4])
        E = self.find_coplanar_vertex(A, B, C, D)
        vertices_coordinates = np.vstack((A, B, C, D, E))
        crystal_central_point = np.array([-5558.888, 6862.606, -66.472])
        radius_central_point = np.array([-5588.096, 6184.396, -106.185])
        R = self.distance_between_poinst(crystal_central_point, radius_central_point)

        return (
            vertices_coordinates,
            A,
            B,
            C,
            D,
            crystal_central_point,
            radius_central_point,
            R,
            AOI,
            max_reflectivity,
        )

    def define_plane_equation(self, p1, p2, p3):
        return super().define_plane_equation(p1, p2, p3)

    def find_coplanar_vertex(self, A, B, C, D):
        return super().find_coplanar_vertex(A, B, C, D)


# class ECRHSieldCoordinates(Methods):
#     """
#     Contains coordinates of ECRH Shields mounted inside the bellows connecting
#     CO Monitor to the W7-X.
#     Units: [mm]
#     """

#     def distance(self, x1, y1, z1, x2, y2, z2):
#         ###Alternatywa
#         # squared_dist = np.sum((central_point-edge1)**2, axis=0)
#         # dist = np.sqrt(squared_dist)
#         d = math.sqrt(
#             math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) + math.pow(z2 - z1, 2) * 1.0
#         )
#         return d / 2

#     def calculate(self, edge1, central_point, central_point2):
#         """
#         Calculates radius of the shield and its orientation vector.
#         """
#         vector1 = central_point - edge1
#         edge2 = central_point + vector1
#         R = self.distance(edge1[0], edge1[1], edge1[2], edge2[0], edge2[1], edge2[2])
#         orientation_vector = central_point2 - central_point
#         return R, orientation_vector, edge2

#     def protection_shield_coordinates_channel_CO_1(self):
#         central_point = np.array([-5366.156, 6630.245, -79.757])
#         edge1 = np.array([-5395.061, 6612.241, -138.019])
#         central_point2 = np.array([-5368.065, 6632.546, -79.521])

#         R, orientation_vector, edge2 = self.calculate(
#             edge1, central_point, central_point2
#         )
#         return central_point, R, orientation_vector, edge1, edge2

#     def protection_shield_coordinates_channel_CO_2(self):
#         central_point = np.array([-5223.943, 6458.79, -97.331])
#         edge1 = np.array([-5220.554, 6454.704, -30.04])
#         central_point2 = np.array([-5225.216, 6460.325, -97.174])

#         R, orientation_vector, edge2 = self.calculate(
#             edge1, central_point, central_point2
#         )
#         return central_point, R, orientation_vector, edge1, edge2

#     def protection_shield_coordinates_channel_BN_1(self):
#         central_point = np.array([-5201.634, 6431.894, -525.812])
#         edge1 = np.array([-5198.605, 6427.459, -593.098])
#         central_point2 = np.array([-5203.543, 6434.196, -526.05])

#         R, orientation_vector, edge2 = self.calculate(
#             edge1, central_point, central_point2
#         )
#         return central_point, R, orientation_vector, edge1, edge2

#     def protection_shield_coordinates_channel_BN_2(self):
#         central_point = np.array([-5324.748, 6580.323, -541.138])
#         edge1 = np.array([-5321.341, 6576.216, -608.282])
#         central_point2 = np.array([-5326.021, 6581.857, -541.297])

#         R, orientation_vector, edge2 = self.calculate(
#             edge1, central_point, central_point2
#         )
#         return central_point, R, orientation_vector, edge1, edge2


class PortCoordinates(Methods):
    """
    TODO - more concise approach needed!!!!
    Contains coordinates of the inner wall edge of AEK30 port.
    """

    def vertices_port(self):
        A = np.array([-4072.79, 4787.42, -219.45])
        B = np.array([-3764.65, 4982.96, -219.45])
        C = np.array([-4006.47, 4707.47, -518.92])
        D = np.array([-3681.30, 4882.47, -518.92])
        D1 = np.array([-3949.091, 4938, -38.43])
        D2 = np.array([-3783, 4738, -699])
        D3 = np.array([-4055, 4839, -99])
        D4 = np.array([-4016, 4881, -55])
        D5 = np.array([-3844, 4988, -86])
        D6 = np.array([-3811, 4995, -123])
        D7 = np.array([-3975, 4691, -588])
        D8 = np.array([-3882, 4697, -676])
        D9 = np.array([-3697, 4802, -658])
        D10 = np.array([-3674.5, 4842, -603])
        E = self.find_coplanar_vertex(A, B, C, D)
        vertices_coordinates = np.vstack(
            (A, B, C, D, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, E)
        ).astype("float64")
        # vertices_coordinates = vertices_coordinates.astype('float64')
        return vertices_coordinates

    def port_orientation_vector(self):
        orientation_vector = np.array([87.144, -105.063, 0.0]) / 3
        return orientation_vector

    def find_coplanar_vertex(self, A, B, C, D):
        return super().find_coplanar_vertex(A, B, C, D)


if __name__ == "__main__":
    pc = PortCoordinates()
    # vertices = pc.vertices_port()
    # pc.port_orientation_vector()
    # v = DetectorCoordinates().vertices_detector_B()
    # # print(v[0].shape)
    # de = DispersiveElementsCoordinates()
    # de.vertices_crys_B()
    # ecrh = ECRHSieldCoordinates()
    # ecrh.protection_shield_coordinates_channel_CO_1()
    # print(ecrh.protection_shield_coordinates_channel_CO_1())
    # print(ecrh.protection_shield_coordinates_channel_CO_2())
    # print(ecrh.protection_shield_coordinates_channel_BN_1())
    # print(ecrh.protection_shield_coordinates_channel_BN_2())
    # ecrh.protection_shield_coordinates_channel_CO_2()
    # ecrh.protection_shield_coordinates_channel_BN_1()
    # ecrh.protection_shield_coordinates_channel_BN_2()
