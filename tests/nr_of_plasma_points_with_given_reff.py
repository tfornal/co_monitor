"""
Created on Wed Jul 28 13:40:42 2021

@author: t_fornal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_Reff(Reff_path):  ### TODO np > pandas - zmiana formatu liczb
    """
    Load file with plasma coordinates and calculated Reff value.
    """
    Reff_path = open(
        Path().absolute() / "src" / "_Input_files" / "Reff" / f"{Reff_path}.txt"
    )

    Reff = np.loadtxt(Reff_path)
    reff_coordinates = pd.DataFrame(Reff, columns=["idx_plasma", "x", "y", "z", "Reff"])

    return reff_coordinates

    Reff_path = open(
        Path().absolute()
        / "src"
        / "_Input_files"
        / "Geometric_data"
        / "C"
        / "top"
        / "C_plasma_coordinates-10_mm_spacing-height_40-length_30-slit_100.csv"
    )
    reff_coordinates = pd.read_csv(Reff_path, sep=";")

    return reff_coordinates


def oobserved_plasma(reff):
    observed_plasma = load_calculated_observed_plasma()
    indexy = observed_plasma["idx_sel_plas_points"].to_numpy()

    reff_wybrane = reff.iloc[indexy]
    reff_wybrane = reff_wybrane.dropna(axis=0)
    posortowane_wybrane = reff_wybrane["Reff"].sort_values()
    policzone_wybrane_reff = posortowane_wybrane.value_counts()
    policzone_wybrane_reff = policzone_wybrane_reff.reset_index().set_index(
        "Reff", drop=False
    )
    policzone_wybrane_reff = policzone_wybrane_reff.to_numpy()

    plt.scatter(policzone_wybrane_reff[:, 0], policzone_wybrane_reff[:, 1])
    plt.show()


# %%
def total_plasma(reff):
    reff_wybrane = reff.dropna(axis=0)
    posortowane_wybrane = reff_wybrane["Reff"].sort_values()
    policzone_wybrane_reff = posortowane_wybrane.value_counts()
    policzone_wybrane_reff = policzone_wybrane_reff.reset_index().set_index(
        "Reff", drop=False
    )
    policzone_wybrane_reff = policzone_wybrane_reff.to_numpy()
    plt.scatter(policzone_wybrane_reff[:, 0], policzone_wybrane_reff[:, 1])
    plt.show()


if __name__ == "__main__":
    reff_file_name = "Reff_coordinates-10_mm"
    reff = load_Reff(reff_file_name)
    oobserved_plasma(reff)
    total_plasma(reff)
