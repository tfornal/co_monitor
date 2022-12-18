# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 08:52:58 2021

@author: t_fornal
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import natsort


def interpolate_profiles(interpolation_range, reff, kinetic_profile):
    x = reff.ravel()
    y = kinetic_profile.ravel()
    tck = interpolate.splrep(x, y)

    return interpolate.splev(interpolation_range, tck)


def FA(df_kinetics, impurity_file_name, element, ion_state, step=1):
    ### read file with impurity profile
    # impurity_file_name = "C_corona_equilibrium.csv"############## PODMIENIONE
    number_of_columns = {
        "B": 6,
        "C": 7,
        "N": 8,
        "O": 9,
    }

    col_nums = number_of_columns[f"{element}"]

    def read_impurity_profile():

        df = pd.read_csv(impurity_file_name, sep=";")
        df = df.drop(["Unnamed: 0"], axis=1)

        indexNames = df["r/a"] <= 1
        df = df[indexNames]
        return df

    df = read_impurity_profile()

    col_names = [x for x in df.columns]

    ####3 interpolacja
    point_range = np.linspace(0, 1, 1000)
    df_interpolated = pd.DataFrame()

    for column in col_names:
        interp = interpolate_profiles(
            point_range, df["r/a"].to_numpy(), df[column].to_numpy()
        )
        # print(interp)
        df_interpolated[column] = abs(interp)

    max_reff = 1

    df = df_interpolated

    ### read profile

    indexNames = df_kinetics["Reff"] <= max_reff
    df_kinetics = df_kinetics[indexNames]

    interp_Te = interpolate_profiles(
        point_range, df_kinetics["Reff"].to_numpy(), df_kinetics["T_e"].to_numpy()
    )

    Reff = df_kinetics["Reff"].to_numpy()
    point_range = df["r/a"]

    T_e = df_kinetics["T_e"].to_numpy()
    point_range = point_range * Reff.max()
    interp_Te = interpolate_profiles(point_range, Reff, T_e)

    df["T_e"] = interp_Te

    basic_array = np.zeros((len(df[f"{element}_total"]), col_nums))

    for i in range(basic_array.shape[-1]):
        basic_array[:, i] = df[f"{element}{i}+"] / df[f"{element}_total"]

    ionization_states = [f"Z{i}" for i in range(basic_array.shape[-1])]

    fractional_abundances = dict(zip(ionization_states, basic_array.T))

    reff = df["r/a"]
    x_axis = reff
    # sns.set_theme()
    fig, ax = plt.subplots()
    ax.stackplot(
        x_axis,
        fractional_abundances.values(),
        labels=fractional_abundances.keys(),
        # colors = ["red", "green", "blue", "orange", "purple", "yellow", "c", "yellow"]
    )

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.legend(loc="upper left")
    # ax.set_title(f'Fractional abundance of carbon (STRAHL), \nD = {file_name[:-4]}')
    ax.set_xlabel(f"{x_axis.name}")  #### zrobic tak, aby bylo powyzej 1!!!!
    ax.set_ylabel("Fractional abundance")

    plt.show()

    for key, value in fractional_abundances.items():
        plt.plot(x_axis, value)

    fa = pd.DataFrame(fractional_abundances)
    fa["T"] = df["T_e"]
    cols = fa.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    fa = fa[cols]

    return fa


if __name__ == "__main__":

    discharge_id = "20181011_012@5_5000"
    V = [0]
    element = "C"
    ne = [
        # "19600000000000.0",
        "70300000000000.0"
    ]
    profile_type = [
        "peaked",
        # "flat"
    ]
    print(profile_type[0])

    impurity_path = rf"D:\1. PRACA\_Programy\CO_Monitor\COMonitor_simulation_anal_3\_Input_files\Impurity_profiles\calculated_by_STRAHL\raw\test\{element}\{profile_type[0]}\nie_zmieniajac_source"
    all_files = glob.glob(impurity_path + "/*.csv")
    all_files = natsort.natsorted(all_files, reverse=False)

    print(all_files)
    df_kinetics = pd.read_csv(
        f"{discharge_id}_kinetic_profiles.dat", sep="  ", comment="!", engine="python"
    )
    df_kinetics.columns = ["Reff", "n_e", "T_e", "T_i"]
    print(df_kinetics)

    for imp_profile in all_files[:]:
        FA(df_kinetics.copy(), imp_profile, element, 2)
