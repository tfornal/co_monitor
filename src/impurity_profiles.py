__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"


"""WORK IN PROGRESS
TODO - implementation of radial impurity profile """
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy import interpolate


# class ImpurityProfile:
#     def __init__(
#         self,
#         file_name,
#         range_and_nr_of_reff_points,
#         kinetic_profiles,
#         element,
#         plaski_rozklad=False,
#         plot=True,
#     ):
#         pass


# def get_impurity_profile(
#     file_name,
#     range_and_nr_of_reff_points,
#     kinetic_profiles,
#     element,
#     plaski_rozklad=False,
#     plot=True,
# ):
#     """
#     file_name = IMPURITY from strahl
#     range_and_nr.... - do wyznaczenia min i max reff
#     kinetic_profiles: df z Reff [m], T_e [eV], n_e [1/cm3]
#     dodac lepszy opis

#     """

#     def interp_profile(x, y, x_point_range):
#         tck = interpolate.splrep(x.ravel(), y.ravel())

#         return interpolate.splev(x_point_range, tck)

#     def integral_over_cylindrical_structure(point_range, curve_to_integrate):
#         x_2d = []
#         for number, reff in enumerate(point_range):
#             if number == 0:
#                 x_2d.append(reff**2)
#             else:
#                 x_2d.append(point_range[number] ** 2 - point_range[number - 1] ** 2)
#         wagi = x_2d
#         integral_over_cylindrical_volume = round(
#             sum(wagi * curve_to_integrate) / sum(wagi), 4
#         )

#         return integral_over_cylindrical_volume

#     def calc_imp_fraction(interpolated_concentration, point_range):
#         df = pd.DataFrame()

#         df["Reff"] = point_range
#         df["interpolated_concentration"] = interpolated_concentration

#         curve = df["interpolated_concentration"]
#         imp_fraction = integral_over_cylindrical_structure(point_range, curve)
#         print(f"AVG IMPURITY CONCENTRATION: {imp_fraction * 100}%")

#         return imp_fraction

#     def read_file():

#         min_reff = range_and_nr_of_reff_points.min()
#         max_reff = range_and_nr_of_reff_points.max()

#         strahl_path_file = file_name
#         df = pd.read_csv(strahl_path_file, usecols=[f"{element}_total", "r/a"], sep=";")
#         df = df[(df["r/a"] <= 1)]
#         df["Reff"] = df["r/a"] * max_reff
#         df = df[~(df["Reff"] < min_reff)]
#         df = df.drop(["r/a"], axis=1)
#         # print(df)
#         return df

#     def calculate_impurity_profile():

#         df = read_file()

#         if plaski_rozklad:
#             avg_density = integral_over_cylindrical_structure(
#                 df["Reff"].to_numpy(), df[f"{element}_total"].to_numpy()
#             )
#             df[f"{element}_total_flat"] = avg_density

#         Reff = df["Reff"].to_numpy()
#         imp_density = df[f"{element}_total"].to_numpy()

#         interp_imp_density = interp_profile(
#             Reff, imp_density, range_and_nr_of_reff_points
#         )
#         interp_imp_density[interp_imp_density < 0] = 0

#         interp_impurity_profiles = pd.DataFrame()
#         interp_impurity_profiles["interp_reff"] = range_and_nr_of_reff_points
#         interp_impurity_profiles["interp_imp_density"] = interp_imp_density

#         if plaski_rozklad:
#             interp_impurity_profiles[
#                 "interp_imp_density"
#             ] = integral_over_cylindrical_structure(
#                 interp_impurity_profiles["interp_reff"], interp_imp_density
#             )
#         return interp_impurity_profiles

#     def calculate_concentration(kinetic_profiles):

#         interp_impurity_profiles = calculate_impurity_profile()

#         interp_ne = interp_profile(
#             kinetic_profiles["Reff"].to_numpy(),
#             kinetic_profiles["n_e"].to_numpy(),
#             range_and_nr_of_reff_points,
#         )

#         interp_ne[interp_ne < 0] = 0

#         df = pd.DataFrame()

#         df["reff"] = range_and_nr_of_reff_points
#         df["impurity_density"] = interp_impurity_profiles["interp_imp_density"]
#         df["impurity_concentration"] = (
#             interp_impurity_profiles["interp_imp_density"] / interp_ne
#         )

#         avg_density = (
#             integral_over_cylindrical_structure(df["reff"], df["impurity_density"])
#             * 1e6
#         )
#         print(
#             f"AVG IMPURITY DENSITY: {avg_density} [1/m3]",
#         )
#         return df, avg_density

#     def plot_results(imp_fraction):

#         plt.plot(range_and_nr_of_reff_points, interpolated_ne["impurity_concentration"])
#         plt.plot(
#             range_and_nr_of_reff_points,
#             np.ones(len(range_and_nr_of_reff_points)) * imp_fraction,
#         )
#         plt.xlim(0, max(range_and_nr_of_reff_points))
#         plt.ylabel("Impurity concentration")
#         plt.xlabel("Reff [m]")
#         plt.show()

#         plt.close("All")
#         plt.plot(range_and_nr_of_reff_points, interpolated_ne["impurity_density"])
#         plt.xlim(0, max(range_and_nr_of_reff_points))
#         plt.ylabel("Impurity density")
#         plt.xlabel("Reff [m]")
#         plt.show()

#     interpolated_ne, avg_density = calculate_concentration(kinetic_profiles)
#     imp_fraction = calc_imp_fraction(
#         interpolated_ne["impurity_concentration"].to_numpy(), interpolated_ne["reff"]
#     )
#     # imp_density = integral_over_cylindrical_structure(df["reff"],df["impurity_density"])*1E6
#     if plot:
#         plot_results(imp_fraction)

#     return interpolated_ne, imp_fraction, avg_density


# if __name__ == "__main__":
#     ### Input files------------------
#     impurity_file_name = "20181011_012@5_5000_conv--100_diff-2000.0.csv"
#     range_and_nr_of_reff_points = np.linspace(
#         0.009, 0.518358, 100
#     )  ### ten range jest kluczowy do okreslenia max reff;
#     from kinetic_profiles import ExperimentalProfile

#     ep = ExperimentalProfile(
#         "report_20180906_038@2_5000_v_",
#     )
#     kinetic_profile = ep.profile_df
#     element = "C"
#     interp_conc, imp_fraction, avg_density = get_impurity_profile(
#         impurity_file_name,
#         range_and_nr_of_reff_points,
#         kinetic_profile,
#         element,
#         plaski_rozklad=False,
#         plot=True,
#     )
