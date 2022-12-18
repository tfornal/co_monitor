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



def generate_kinetic_information():
    file_name = "densities_20181009.016.pkl"
    CXRS_file = pathlib.Path.cwd() / file_name
    with open(CXRS_file, 'rb') as file:
        data = pickle.load(file)
        counter = 0
        for time in data:
            dane = data[time]["kinetic_information"]
            s = np.array(dane["s"])
            # print(s )
            ne = np.array(dane["n_e"])
            Te = np.array(dane["T_e"])
            Ti = np.array(dane["T_i"])
            np.savetxt(f"{file_name[10:22]}_{time[4:]}ms_kinetic_profiles.dat", 
                        np.c_[s*0.53, ne/1E14, Te, Te], 
                        header = "!      r     n_20m3_e      T_keV_e      T_keV_H",
                        comments = "")
            print(np.c_[np.sqrt(s), ne, Te, Te])
# 
generate_kinetic_information()
# 

def generate_impurity_information():
    file_name = "densities_20181009.016.pkl"
    CXRS_file = pathlib.Path.cwd() / file_name
    print(CXRS_file)
    
    with open(CXRS_file, 'rb') as file:
        data = pickle.load(file)
        counter = 0
        time_array = []
        imp_concentrations = np.zeros((52, 41))
        # print(imp_concentrations)
        counter = 1
        for time in data:
            # print(time)
            time_array.append(time)
            
            # print(data[time]["impurity_information"])
            # for i in data[time]["impurity_information"]["Fit_profile"]["Carbon"]["rho_locations"]:
                # print(i)
            # import time
            # time.slep(3)
            dane = data[time]["impurity_information"]["Fit_profile"]["Carbon"]
            location = np.array(dane["s_locations"])
            # print(dane["rho_n_z_values"])
            # print(dane["s_locations"])
            import matplotlib.pyplot as plt
            # plt.plot(dane["s_locations"], dane["rho_n_z_values"])
            # imp_concentrations = np.append(imp_concentrations, dane["rho_n_z_values"], axis = 1)
            # for key,values in dane.items():
                # print(key)
            
            # imp_concentrations[:, counter] = dane["rho_n_z_values"]
            imp_concentrations[:, counter] = dane["rho_n_z_values"].T
            imp_concentrations[:, 0] = dane["s_locations"]
            counter += 1
            lista  = dane["rho_n_z_values"][0]
            
            # dane["rho_n_z_values"] = [x for item in [dane["rho_n_z_values"][0]] for x in repeat(item, 52)]
            # *len(dane["rho_n_z_values"])
            # print(dane["rho_n_z_values"])
            # print(lista)
            # imp_concentrations[:, counter] = dane["rho_n_z_values"][0] 
            # np.savetxt(f"{file_name[10:22]}_{time[4:]}ms_imp_profiles.txt", 
            #             np.c_[dane["s_locations"], dane["rho_n_z_values"]], 
            #             header = "Reff, imp_conc",
            #             comments = "")
    # print(len(time_array))
    # print(len(location))
    # print(imp_concentrations)
    # print(imp_concentrations.T)
    for i in imp_concentrations.T:
        # print(len(imp_concentrations))
        plt.plot(dane["s_locations"], i)
        plt.legend()
    
    # plt.plot(dane["s_locations"], imp_concentrations.T)
    
    # print(len(time_array))
    time_array.insert(0, 's_location')
    # print(time_array)
    df = pd.DataFrame(imp_concentrations, columns=time_array)

    # print(df)
    # df.to_csv("20181009.016 - CVI_Imp_fitted_profiles_.csv",sep = ";")
    # df.columns(time_array)
    # df = pd.DataFrame()
    # df = imp_concentrations.to_csv()
    # print(len(imp_concentrations))

generate_impurity_information()