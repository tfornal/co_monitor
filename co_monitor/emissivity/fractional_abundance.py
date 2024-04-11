__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


class FractionalAbundance:
    """The script reads the files representing fractional abundances (FA) of a given
    ion and interpolates the data.
    """

    def __init__(self, element, ionization_state, interpolation_step=1, plot=False):
        """
        Parameters
        ----------
        element : str
            Select atoms symbol called in parentheses e.g. "C"
        ionization_state : str
            Atomic number of selected ion called in parentheses e.g. "Z5"
        interpolation_step : int, optional
            Step of the interpolation - nominally 1 [eV] (optional), by default 1
        plot : bool, optional
            Plots retrieved Fractional Abundance, by default False
        """
        self.element = element
        self.ionization_state = ionization_state
        self.interpolation_step = interpolation_step
        self.plot = plot
        self.loaded_file_df = self._read_file()
        self.df_interpolated_frac_ab = self._interpolate()
        if plot:
            self._plotter()

    def _read_file(self) -> pd.DataFrame:
        """
        Reads the file representing fractional abundance of the given element.

        Returns
        -------
        loaded_file_df : pandas.DataFrame
            The loaded file data as a pandas dataframe.
        """
        fractional_abundance = (
            Path.cwd()
            / "input_files"
            / "fractional_abundance"
            / f"fractional_abundance_{self.element}.txt"
        )
        Path(__file__).parent.resolve()
        loaded_file_df = pd.read_csv(fractional_abundance, delimiter=" ")

        return loaded_file_df

    def _interpolate(self) -> np.ndarray:
        """
        Interpolates the temperature and fractional abundance with the given step.

        Returns
        -------
        df_interpolated_frac_ab : np.ndarray
            The fractional abundance dataframe with two columns:
            1st - electron temperature [eV], 2nd - fractional abundance [%].
        """

        loaded_file_df = self.loaded_file_df
        temp_col = loaded_file_df["T"]
        Te = np.arange(temp_col.iloc[0], temp_col.iloc[-1], self.interpolation_step)
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
        df_interpolated_frac_ab.insert(0, "T_e [eV]", Te)
        df_interpolated_frac_ab.columns = column_names

        return df_interpolated_frac_ab

    def _save_to_file(self, element, ion_state):
        """Saves inteprolated temperautre/fractional_abundance to the file.

        Parameters
        ----------
        element : str
            Symbol for the selected atom - "B", "C", "N" and "O".
        ion_state : str
            Atomic number of selected hydrogen like ion, for example "Z5" when considering "C" line.
        """
        df_interpolated_frac_ab = self.interpolated_fractional_abundance()
        np.savetxt(f"{element} {ion_state}.txt", df_interpolated_frac_ab)
        print(f"Fractional abundance of {element}-{ion_state} ion successfuly saved!")

    def _plotter(self):
        """
        Plots interpolated T_e [eV] and all fractional abundance [%].
        """
        column_list = self.df_interpolated_frac_ab.columns[1:]
        Te = self.df_interpolated_frac_ab["T"]

        for column in column_list:
            Fa = self.df_interpolated_frac_ab[column]
            plt.plot(Te, Fa)

        plt.xlim(5, 50)
        plt.xlabel("Te [eV]")
        plt.ylabel("Fractional Abundance [%]")
        plt.title(f"Fractional abundance of {self.element} ({self.ionization_state})")
        plt.tight_layout()

        plt.show()
        # plt.savefig("frac_ab_lukasza.png", dpi=300)


if __name__ == "__main__":
    fa = FractionalAbundance("O", "Z7", plot=False)
