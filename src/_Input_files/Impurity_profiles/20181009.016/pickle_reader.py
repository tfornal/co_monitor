# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:50:23 2021

@author: Tomek
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import repeat
import pathlib


# def first_file():
#     CXRS_file = pathlib.Path.cwd() / "20180920.042.pkl"
#     blips_names = []
#     with open(CXRS_file, 'rb') as file:
#         data = pickle.load(file)
#         blips_number = 0
#         number_of_rows = 0
#         for i in data:
#             if 'blip' in i:
#                 print(i)
#                 blips_names.append(i)
#                 blips_number +=1
#                 Reff = data[i]['Carbon']['a']
#                 number_of_rows=len(Reff)
#                 concentration = data[i]['Carbon']['concentration']
#                 densities = data[i]['Carbon']['densities']

#         new_array = np.zeros((number_of_rows, blips_number*2+1))
#         counter = 1
#         for i in data:
#             if 'blip' in i:

#                 Reff = data[i]['Carbon']['a']
#                 number_of_rows=len(Reff)
#                 concentration = data[i]['Carbon']['concentration']
#                 densities = data[i]['Carbon']['densities']
#                 def plotter(i):

#                     x = np.array(Reff)
#                     # y = np.array(concentration)
#                     y = np.array(densities)
#                     plt.plot(x,y)
#                     plt.xlim([0, 0.50])
#                     plt.title("20180920.042_CVI")
#                     plt.legend(i)
#                 plotter(i)
#                 new_array[:, 0] = Reff
#                 new_array[:, (counter)*2-1] = densities
#                 new_array[:, (counter)*2] = concentration
#                 counter +=1
#         plt.show()
#     df = pd.DataFrame(new_array)

#     nowa_lista = [x for item in blips_names for x in repeat(item, 2)]
#     xx = []
#     for i in range(len(nowa_lista)):
#         if i%2==0:
#             xx.append("Density for "+ nowa_lista[i])
#         else:
#             xx.append("Concentration for " + nowa_lista[i])

#     nowa_lista2 = ["Reff"]
#     final_list=  nowa_lista2 + xx
#     df.columns = [final_list]
#     # df.to_csv("20180920.042_CVI.csv",sep = ";")
#     # print(final_list)


#     from scipy import interpolate
#     print(sum(df["Concentration for blip at 6.5"].values.ravel()))
#     def interpolate_cxrs(xxx):
#         x = df["Reff"].values.ravel()
#         y = df["Concentration for blip at 6.5"].values.ravel()
#         x = list(x)[::-1]
#         y = list(y)[::-1]
#         tck = interpolate.splrep(x, y)
#         # print(tck)

#         """
#         from scipy.interpolate import UnivariateSpline
#         spl = UnivariateSpline(x, y)
#         xs = np.linspace(0, 0.5, 1000)
#         plt.plot(xs,spl(xs), lw = 3)
#         spl.set_smoothing_factor(1200)
#         plt.plot(xs, spl(xs), 'b', lw=3)
#         """

#         return interpolate.splev(xxx, tck)


#     www = np.linspace(0,0.5, 2000)
#     # print(www)
#     zzz = interpolate_cxrs(www)
#     plt.scatter(df["Reff"].values.ravel(),
#              df["Concentration for blip at 6.5"].values.ravel())
#     plt.plot(www,zzz)
#     plt.xlim(0,0.5)
#     print(sum(df["Concentration for blip at 6.5"].values.ravel()))
#     print(interpolate_cxrs(0.5))
#     return interpolate_cxrs


# def second_file():
#     CXRS_file = pathlib.Path.cwd() / "20180920.049.pkl"
#     with open(CXRS_file, 'rb') as file:
#         data = pickle.load(file)
#         time_stamps_amount = len(data.items())

#         # elements_list = ['Carbon', "Oxygen", "Nitrogen"]#, "Neon"]
#         # print(data.items())
#         element = "Oxygen"
#         kinetics = np.zeros((time_stamps_amount,100,3))
#         profiles = np.zeros((time_stamps_amount,117,2))
#         counter =0
#         for time, value in data.items():
#             s = data[f'{time}']['kinetic_information']['s']
#             n_e = data[f'{time}']['kinetic_information']['n_e']
#             T_e = data[f'{time}']['kinetic_information']['T_e']

#             Reff = data[f'{time}']['impurity_information']['Constant_nz'][f'{element}']['s_locations']
#             concentration = np.sqrt(data[f'{time}']['impurity_information']['Constant_nz'][f'{element}']['n_z_values'])

#             kinetics[counter,:,0] = s
#             kinetics[counter,:,1] = n_e
#             kinetics[counter,:,2] = T_e

#             profiles[counter,:,0] = Reff
#             profiles[counter,:,1] = concentration

#             plt.scatter(Reff, concentration),
#             plt.title(f"densities_20180920.049 - {element} - {time}")
#             plt.show()
#             counter +=1

#         def save_to_csv():

#             for i in range(len(profiles)):
#                 np.savetxt(f"{i}{element}.txt", profiles[i], header = "Reff, density")

#             for i in range(len(kinetics)):
#                 np.savetxt(f"{i}_profiles.txt",kinetics[i], header = "s, n_e, T_e")


#         # save_to_csv()

# # first_file()
# # second_file()


import matplotlib.pyplot as plt


def generate_kinetic_information():
    file_name = "densities_20181009.016.pkl"
    CXRS_file = (
        pathlib.Path.cwd() / "src" / "_Input_files" / "Impurity_profiles" / file_name
    )
    with open(CXRS_file, "rb") as file:
        data = pickle.load(file)
        counter = 0
        selected_time_frames = [
            1.0998,
            2.0998,
            3.0998,
            3.3998,
            3.6998,
            3.9998,
            4.2998,
            4.5998,
            4.8998,
        ]

        for time in selected_time_frames:
            dane = data[f"t = {time}"]["kinetic_information"]
            reff = np.sqrt(np.array(dane["s"]))  ################ tu!!!!
            # print(s )
            ne = np.array(dane["n_e"])
            Te = np.array(dane["T_e"])
            Ti = np.array(dane["T_i"])
            # print(reff)
            df_kinetics = pd.DataFrame(columns=["reff", "ne", "Te", "Ti"])
            # print(df_kinetics)
            df_kinetics["reff"] = reff
            df_kinetics["ne"] = ne / 1e14
            df_kinetics["Te"] = Te
            df_kinetics["Ti"] = Ti

            df_kinetics = df_kinetics.drop(df_kinetics[df_kinetics["reff"] > 1].index)
            df_kinetics["reff"] = df_kinetics["reff"] * 0.53
            # print(df_kinetics)
            # df_kinetics.to_csv(f"{file_name[10:22]}_{time}ms_kinetic_profiles.csv", sep = ";", index=False)
            # np.savetxt(
            #     f"{file_name[10:22]}_{time}ms_kinetic_profiles.dat",
            #     df_kinetics.to_numpy(),
            #     header = "!      r     n_20m3_e      T_keV_e      T_keV_H",
            #     comments = ""
            #     )
            # print(df_kinetics.to_numpy())
            # print(len(imp_concentrations))
            ### zrobic interpolacje
            plt.scatter(df_kinetics["reff"], df_kinetics["Te"])
            plt.legend()

        for key in data.keys():
            print(data["t = 4.1998"])


# generate_kinetic_information()
#


### chosen time frames [1.0998, 2.0998, 3.0998, 3.3998, 3.6998, 3.9998, 4.2998, 4.5998, 4.8998]
def generate_impurity_information():
    file_name = "densities_20181009.016.pkl"
    CXRS_file = (
        pathlib.Path.cwd() / "src" / "_Input_files" / "Impurity_profiles" / file_name
    )
    # print(CXRS_file)

    with open(CXRS_file, "rb") as file:
        data = pickle.load(file)
        # for time in data:
        #     print(data[f"{time}"]["impurity_information"]["Fit_profile"]["Carbon"]['rho_locations'])
        #     print(len(data[f"{time}"]["impurity_information"]["Fit_profile"]["Carbon"]['rho_n_z_values']))
        # df_impurities = pd.DataFrame(columns = [""])
        counter = 0
        time_array = []
        imp_concentrations = np.zeros((52, 41))
        # print(imp_concentrations)
        counter = 1
        selected_time_frames = [
            1.0998,
            2.0998,
            3.0998,
            3.3998,
            3.6998,
            3.9998,
            4.2998,
            4.5998,
            4.8998,
        ]

        for time in selected_time_frames:
            time_array.append(time)

            dane = data[f"t = {time}"]["impurity_information"]["Fit_profile"]["Carbon"]
            # location = np.array(dane["rho_locations"])
            # location = np.sqrt(np.array(dane["s_locations"]))
            # print(data[f"{time}"]["impurity_information"]["Fit_profile"]["Carbon"]['rho_locations'])
            # print(dane["rho_n_z_values"])

            # plt.plot(dane["s_locations"], dane["rho_n_z_values"])
            # imp_concentrations = np.append(imp_concentrations, dane["rho_n_z_values"], axis = 1)
            # for key,values in dane.items():
            # print(key)

            # imp_concentrations[:, counter] = dane["rho_n_z_values"]
            imp_concentrations[:, counter] = dane["rho_n_z_values"].T
            imp_concentrations[:, 0] = dane["s_locations"]
            counter += 1

            s_locations = dane["rho_locations"]

            df_impurities = pd.DataFrame(columns=["reff", "imp_density"])
            df_impurities["reff"] = s_locations
            df_impurities["imp_density"] = dane["rho_n_z_values"]

            # print(file_name)
            # df_impurities.to_csv(f"{file_name[10:22]}_{time}ms_imp_profiles.csv", sep = ";", index=False)
            # np.savetxt(
            #     f"{file_name[10:22]}_{time}ms_kinetic_profiles.dat",
            #     df_impurities.to_numpy(),
            #     header = "!      r     n_20m3_e      T_keV_e      T_keV_H",
            #     comments = ""
            #     )

    for j, i in enumerate(imp_concentrations.T):
        # print(len(imp_concentrations))
        plt.plot(s_locations, i)
        plt.legend()

    # # plt.plot(dane["s_locations"], imp_concentrations.T)

    # # print(len(time_array))
    # time_array.insert(0, 's_location')
    # # print(time_array)
    # df = pd.DataFrame(imp_concentrations, columns=time_array)

    # # print(df)
    # # df.to_csv("20181009.016 - CVI_Imp_fitted_profiles_.csv",sep = ";")
    # # df.columns(time_array)
    # # df = pd.DataFrame()
    # # df = imp_concentrations.to_csv()
    # # print(len(imp_concentrations))


generate_kinetic_information()
# generate_impurity_information()
