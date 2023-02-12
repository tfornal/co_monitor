__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import interpolate
from pathlib import Path

# TODO dolaczyc fractional abundance ze strahla


class FractionalAbundance:
    """
    The script reads the files representing fractional abundances (FA) of a given
    ion and interpolates the data.
    It is also possible to plot fractional abundances of selected ions.

    Parameters:
        element (str): select atoms symbol called in parentheses e.g. "C"
        ion_state (str): atomic number of selected ion called in parentheses e.g. "Z5"
        step (int): step of the interpolation - nominally 1 [eV] (optional)
    """

    def __init__(self, element, ion_state, step=1, plot=False):
        self.element = element
        self.ion_state = ion_state
        self.step = step
        self.plot = plot
        self.loaded_file_df = self.read_file()
        self.df_interpolated_frac_ab = self.interpolated_fractional_abundance()
        if plot:
            self.plotter()

    def read_file(self):
        """
        Reads the file representing fractional abundance of the given element.

        Returns:
            temperature (float list): temperature values [eV]
            fractional_abundance (float list): fractional values of selected ion

        """
        fractional_abundance = (
            Path(__file__).parent.resolve()
            / "_Input_files"
            / "Fractional_abundance"
            / f"fractional_abundance_{self.element}.dat"
        )
        Path(__file__).parent.resolve()

        loaded_file_df = pd.read_csv(fractional_abundance, delimiter=" ")

        return loaded_file_df

    def interpolated_fractional_abundance(self):
        """
        Interpolate the temperature and fractional abundance with given step.

        Returns:
            fractional_abundance_array wiextensions.ignoreRecommendations 2 columns: 1st - electron temeprature [eV],
            2nd - fractional abundance [%]
        """

        loaded_file_df = self.loaded_file_df
        temp_col = loaded_file_df["T"]
        Te = np.arange(temp_col.iloc[0], temp_col.iloc[-1], self.step)
        list_of_interp_ionisation_states = []

        # interpolate over all ionisation states and saves in a list
        for ionisation_state in loaded_file_df.iloc[:, 1:]:
            func = interpolate.interp1d(
                temp_col.iloc[:], loaded_file_df[ionisation_state]
            )
            fractional_abundance = func(Te) / 100
            list_of_interp_ionisation_states.append(list(fractional_abundance))

        # put list into dataframe
        column_names = loaded_file_df.columns
        df_interpolated_frac_ab = np.array(list_of_interp_ionisation_states).T

        df_interpolated_frac_ab = pd.DataFrame(df_interpolated_frac_ab)
        df_interpolated_frac_ab.insert(0, "Te", Te)
        df_interpolated_frac_ab.columns = column_names

        return df_interpolated_frac_ab

    def save_to_file(self, element, ion_state):
        """
        Saves inteprolated temperautre/fractional_abundance to the file
        """
        df_interpolated_frac_ab = self.interpolated_fractional_abundance()
        np.savetxt(f"{element} {ion_state}.txt", df_interpolated_frac_ab)
        print(f"Fractional abundance of {element}-{ion_state} ion successfuly saved!")

    def plotter(self):
        """
        Plots interpolated T_e [eV] and all fractional abundance [%].
        """
        column_list = self.df_interpolated_frac_ab.columns[1:]
        Te = self.df_interpolated_frac_ab["T"]

        for column in column_list:
            Fa = self.df_interpolated_frac_ab[column]
            plt.plot(Te, Fa)

        plt.xlim(0, 220)
        plt.xlabel("Te [eV]")
        plt.ylabel("Fractional Abundance [%]")
        plt.title(f"Fractional abundance of {self.element} ({self.ion_state})")
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    fa = FractionalAbundance("O", "Z7", plot=False)
