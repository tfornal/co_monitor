__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


class Profile:
    def plot(self, profile_df):
        """
        Plot the electron temperature and density profiles.
            Parameters
        ----------
        profile_df : pd.DataFrame
            DataFrame containing Reff, n_e, and T_e columns.
        """
        Reff = profile_df["Reff"]
        n_e = profile_df["n_e"]
        T_e = profile_df["T_e"]

        fig, ax1 = plt.subplots()
        plt.title("Electron temperature and density profiles")
        color = "red"
        ax1.set_xlabel("Reff [m]")
        ax1.set_ylabel("Te [eV]", color=color)
        ax1.plot(Reff, T_e, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()

        color = "blue"
        ax2.set_xlabel("Reff [m]")
        ax2.set_ylabel("ne [1/cm-3]", color=color)
        ax2.plot(Reff, n_e, color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        fig.tight_layout()

        plt.show()

    def save_to_txt(self, profile_df):
        """
        Save the generated profile to a text file.

        Parameters
        ----------
        profile_df : pd.DataFrame
            DataFrame containing the profile information to be saved.
        """
        directory = (
            Path(__file__).parent.resolve() / "src" / "results" / "plasma_profiles"
        )
        if not Path.is_dir(directory):
            Path(directory).mkdir(parents=True, exist_ok=True)
        np.savetxt(PurePath(directory, "plasma_profile.txt"), profile_df, fmt="%.3e")

        print("Profile saved to file!")


class TwoGaussSumProfile(Profile):
    """
    A class that calculates a two-Gaussian sum profile of electron temperature (te) and density (ne).

    Parameters
    ----------
    ne_equation_coefficients : list of float
        A list of six coefficients used to define the two-Gaussian sum equation for the electron density profile.
    Te_equation_coefficients : list of float
        A list of six coefficients used to define the two-Gaussian sum equation for the electron temperature profile.
    max_Reff : float, optional
        Maximum value of radial position Reff [m], by default 0.539
    """

    def __init__(self, ne_coeff, te_coeff, max_Reff=0.539, plot=False):
        self.max_Reff = max_Reff  # [m]
        self.Reff = np.arange(0, self.max_Reff, 0.001)
        self.ne_coeff = ne_coeff
        self.te_coeff = te_coeff
        self.profile_df = self.create_df()
        if plot:
            self.plot()

    def profile_equation(self, two_gauss_coefficients) -> np.ndarray:
        """
        The function calculates a two-Gaussian sum equation.

        Parameters
        ----------
        two_gauss_coefficients : list of float
            A list of six coefficients used to define the two-Gaussian sum equation.

        Returns
        -------
        np.ndarray
            Profile calculated from the two-Gaussian sum equation.
        """
        A1, x1, w1, A2, x2, w2 = two_gauss_coefficients
        profile = A1 * np.exp(-((self.Reff - x1) ** 2) / (2 * w1**2)) + A2 * np.exp(
            -((self.Reff - x2) ** 2) / (2 * w2**2)
        )

        return profile

    def create_df(self):
        """
        Creates dataframe with Reff, Te and ne datasets.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the radial position Reff [m], electron temperature T_e [eV], and electron density n_e [1/cm-3].
        """
        ne_profile = self.profile_equation(self.ne_coeff)
        Te_profile = self.profile_equation(self.te_coeff)

        profile_df = pd.DataFrame(data=[self.Reff, Te_profile, ne_profile]).T
        profile_df = profile_df.rename(columns={0: "Reff", 1: "T_e", 2: "n_e"})

        return profile_df

    def plot(self):
        return super().plot(self.profile_df)

    def save_txt(self):
        return super().save_txt(self.profile_df)


class ExperimentalProfile(Profile):
    """The class creates an object repesenting
    experimental kinetic profiles registerd during experimental campaign.

    Parameters
    ----------
    Profile : _type_
        _description_

    """

    """
    

    Parameters:
        file_name: file name with fitted experimental data;
        interval: (optional) precision of interpolation along Reff (e.g. step = 100
        separates given range of Reff into 100 pieces)
    """

    def __init__(self, file_name, max_Reff=1, interval=10000, plot=False):
        super().__init__()
        self.file_name = file_name
        self.file_path = self._get_file_path()
        self.te_idx_range, self.ne_idx_range = self._get_index_ranges()
        self.raw_profile_df = self.create_raw_profile_df()
        self.interpolation_interval = interval
        self.max_Reff = max_Reff  # [m]
        self.profile_df = self.interpolate_raw_profile()
        if plot:
            self.plot()

    def _get_file_path(self):
        """
        Readout of electron temperature (T_e) and electron density (n_e)
        profiles with experimental data stored in "..\_Input_files\Kinetic_profiles\Experimental" directory;

        Returns:
            file_path: path of a file in the
            ".../_Input_files/Profiles/Experimental/file_name.txt" directory;
        """
        file_path = (
            Path(__file__).parent.resolve()
            / "_Input_files"
            / "Kinetic_profiles"
            / "Experimental"
            / f"{self.file_name}.txt"
        )

        return file_path

    def _get_index_ranges(self):
        """Selects indexes of the fitted T_e and n_e profiles in the read file.

        Returns
        -------
        te_idx_range : list of int
            The indexes of the beginning and end of the fitted T_e section.
        ne_idx_range : list of int
        The indexes of the beginning and end of the fitted n_e section.
        """
        
        te_index = -1
        ne_index = -1
        with open(self.file_path, "r") as file:
            # finds indexes of the fitted T_e and n_e profiles
            for idx, line in enumerate(file):
                if "T_e (keV)" in line:
                    te_index = idx - 1
                if "n_e (10^{19}m^{-3})" in line:
                    ne_index = idx - 1

        if te_index == -1:
            print("No electron temperature (t_e) profile available!")
            return [], []
        if ne_index == -1:
            print("No electron density (n_e) profile available!")
            return [], []
        
        number_of_lines = abs(ne_index - te_index)
        # creates the range of investigated file indexes of fitted T_e and n_e
        te_idx_range = [te_index, ne_index]
        ne_idx_range = [ne_index, ne_index + number_of_lines]
        
        self.te_idx_range = te_idx_range
        self.ne_idx_range = ne_idx_range
        print(te_idx_range, ne_idx_range)
        return te_idx_range, ne_idx_range

    def read_file_sections(self):
        print("tak")
        """
        Reads the specified sections from the file containing fitted T_e and n_e profiles.

        Returns:
        -------
        te_section : list of strings
            Separated T_e section of the read file with given T_e index range.
        ne_section : list of strings
            Separated n_e section of the read file with given n_e index range.
        """
        te_section = []
        ne_section = []
        with open(self.file_path, "r") as file:
            for idx, line in enumerate(file):
                if idx >= self.te_idx_range[0]:
                    te_section.append(line)
                if idx == self.te_idx_range[1]:
                    break
            file.seek(0)
            for idx, line in enumerate(file):
                if idx >= self.ne_idx_range[0]:
                    ne_section.append(line)
                if idx == self.ne_idx_range[1]:
                    break
        # print(te_section)
        return te_section, ne_section

    def _remove_comments(self, array):
        """
        Removes comments lines from the input array. Only raw values are left.
        """
        for line in array[:]:
            if line.startswith("#"):
                array.remove(line)

        return array

    def replace_delimiter(self, array):
        """
        Replaces all delimiters to spacebar.
        """
        array = np.asarray(array)
        for i in range(len(array)):
            array[i] = array[i].replace(" \n", "")
            array[i] = array[i].replace("   ", " ")
            array[i] = array[i].replace("  ", " ")

        return array

    def split_to_columns(self, array):
        """
        Splits readed array of Reff and profile of interest into separated columns.
        This is both [Reff, n_e] and [Reff T_e].
        """
        df = pd.Series(array)
        df = df.str.split(" ", expand=True)
        df.columns = ["Reff", "-", "-", "profile", "-", "-", "-", "-", "-", "-"]
        df = df.astype({"Reff": float, "profile": float})
        df = df.loc[:, df.columns.intersection(["Reff", "profile"])]

        return df

    def concat_profile_df(self, file_name, arr1, arr2):
        """
        Concatenates two data frames [Reff, n_e] and [Reff, T_e] into one [Reff, n_e, T_e].
        """
        df = pd.concat([arr1, arr2], axis=1)
        df.columns = ["Reff", "n_e", "Reff", "T_e"]
        df = df.loc[:, ~df.columns.duplicated()]
        df["Reff"] = df["Reff"]  # [m]
        df["n_e"] = 1e19 * df["n_e"]  # [1/m-3]
        df["T_e"] = 1e3 * df["T_e"]  # [eV]

        return df

    def create_raw_profile_df(self):
        """
        Designation of the final data frame containing Reff, n_e and T_e profiles.
        This module calls subsequent functions - each next function takes as an argument
        returned values by funcions called previously.

        Returns:
            profile_df: dataframe with [Reff, n_e, T_e]
        """
        # te_idx_range, self.ne_idx_range = self._get_index_ranges()
        te_section, ne_section = self.read_file_sections()

        te, ne = self._remove_comments(te_section), self._remove_comments(ne_section)
        te, ne = self.replace_delimiter(te), self.replace_delimiter(ne)
        ne = self.split_to_columns(ne)
        te = self.split_to_columns(te)
        profile_df = self.concat_profile_df(self.file_name, ne, te)

        return profile_df

    def interpolate_raw_profile(self):
        """
        Interpolation of selected ne, te profiles.

        Returns:
            interpolated_profiles: array of interpolated values of Reff, ne, Te with given precision
        """

        Reff = self.raw_profile_df["Reff"]
        n_e = self.raw_profile_df["n_e"]
        T_e = self.raw_profile_df["T_e"]
        f1_te_interp = interpolate.interp1d(Reff, n_e)
        f2_te_interp = interpolate.interp1d(Reff, T_e)

        def find_index():
            idx = 0
            for i, j in enumerate(Reff):
                if j > self.max_Reff:
                    idx = i
                    break
                else:
                    idx = len(Reff) - 1  ########## TODO łopatoligicznie, poprawic!!!!!!
                    break
            return idx

        max_reff_idx = find_index()
        Reff = np.linspace(
            0, Reff[max_reff_idx], self.interpolation_interval, endpoint=True
        )

        n_e = f1_te_interp(Reff) / 1e6
        T_e = f2_te_interp(Reff)

        profile_df = pd.DataFrame(data=[Reff, T_e, n_e]).T
        profile_df = profile_df.rename(columns={0: "Reff", 1: "T_e", 2: "n_e"})
        # print(profile_df)
        return profile_df

    def plot(self):
        return super().plot(self.profile_df)

    def save_txt(self):
        return super().save_txt(self.profile_df)


if __name__ == "__main__":
    ne = [7e13, 0, 0.37, 9.8e12, 0.5, 0.11]
    Te = [1870, 0, 0.155, 210, 0.38, 0.07]
    tgsp = TwoGaussSumProfile(ne, Te, plot=True)

    ep = ExperimentalProfile("report_20181011_012@5_5000_v_1", plot=True)
