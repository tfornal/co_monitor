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
        Reff = profile_df["Reff [m]"]
        n_e = profile_df["n_e [m-3]"]
        T_e = profile_df["T_e [eV]"]

        fig, ax1 = plt.subplots()
        plt.title("Electron temperature and density profiles")
        color = "red"
        ax1.set_xlabel("Reff [m]")
        ax1.set_ylabel("T_e [eV]", color=color)
        ax1.plot(Reff, T_e, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()

        color = "blue"
        ax2.set_xlabel("Reff [m]")
        ax2.set_ylabel("n_e [m-3]", color=color)
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
        self.profiles_df = self.create_df()
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

        profiles_df = pd.DataFrame(data=[self.Reff, Te_profile, ne_profile]).T
        profiles_df = profiles_df.rename(
            columns={0: "Reff [m]", 1: "T_e [eV]", 2: "n_e [m-3]"}
        )

        return profiles_df

    def plot(self):
        return super().plot(self.profiles_df)

    def save_txt(self):
        return super().save_txt(self.profiles_df)


class TestExperimentalProfile(Profile):
    """The class creates an object repesenting
    experimental kinetic profiles registerd during experimental campaign.

    Parameters
    ----------
    Profile : _type_
        _description_

    Parameters:
        file_name: file name with fitted experimental data;
        interval: (optional) precision of interpolation along Reff (e.g. step = 100
        separates given range of Reff into 100 pieces)
    """

    def __init__(self, file_name, time, max_Reff=0.539, interp_step=10000, plot=False):
        """
        Parameters
        ----------
        file_name : str
            Name of the loaded file contaitning kinetic data.
        max_Reff : int, optional
            _description_, by default 1
        interp_step : int, optional
            Defines the precision of kinetic profiles interpolation, by default 10000
        plot : bool, optional
            _description_, by default False
        """
        super().__init__()
        self.file_name = file_name
        self.time = time
        self.max_Reff = max_Reff  # [m]
        self.interp_step = interp_step

        self.file_path = self._get_file_path()
        self.profiles_df = self._get_data_from_file()
        self.profiles_df = self._interpolate()
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
        file_path_ne = (
            Path.cwd()
            / "input_files"
            / "kinetic_profiles"
            / "experimental_new"
            / f"{self.file_name}-n_e.csv"
        )
        file_path_Te = (
            Path.cwd()
            / "input_files"
            / "kinetic_profiles"
            / "experimental_new"
            / f"{self.file_name}-T_e.csv"
        )
        return file_path_ne, file_path_Te

    def _get_data_from_file(self):
        """
        Reads the specified sections from the file containing fitted T_e and n_e profiles.

        Returns:
        -------
        te_section : list of strings
            Separated T_e section of the read file with given T_e index range.
        ne_section : list of strings
            Separated n_e section of the read file with given n_e index range.
        """
        reff = []
        # time = "8.0105"

        # def get_all_time_frames():
        #     for file in self.file_path:
        #         df = pd.read_csv(file, sep="\t")
        #         col_names = [col for col in df.columns if "rho" not in col]
        #         return col_names

        # get_all_time_frames()
        # breakpoint()
        for file in self.file_path:
            df = pd.read_csv(file, sep="\t")
            # col_names = [col for col in df.columns if "rho" not in col]
            rho = df["rho"]
            reff = rho * 0.539
            if "n_e" in str(file):
                ne_section = df[self.time]
            elif "T_e" in str(file):
                te_section = df[self.time]
        data = (rho, ne_section)
        df2 = pd.DataFrame()
        df2["Reff [m]"] = reff
        df2["n_e [m-3]"] = ne_section
        df2["T_e [eV]"] = te_section
        df2 = df2.loc[df2["Reff [m]"] < 0.539]
        return df2

    def _interpolate(self) -> pd.DataFrame:
        """Interpolate selected ne and Te profiles.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the following columns:
            - "Reff [m]": Effective radius values in meters.
            - "T_e [eV]": Interpolated electron temperature values in electron volts (eV).
            - "n_e [m-3]": Interpolated electron density values in meters^-3.

        """
        # Merge the ne and Te profiles into a single DataFrame.
        df = self._get_data_from_file()

        # Interpolate the ne and Te profiles.
        Reff = df["Reff [m]"]
        n_e = df["n_e [m-3]"]
        T_e = df["T_e [eV]"]
        f1_te_interp = interpolate.interp1d(Reff, n_e)
        f2_te_interp = interpolate.interp1d(Reff, T_e)

        # Create a new DataFrame with the interpolated profiles.
        shortened_prof_df = df.drop(df[df["Reff [m]"] > self.max_Reff].index)
        Reff_interp = np.linspace(
            0, shortened_prof_df["Reff [m]"].max(), self.interp_step, endpoint=True
        )
        T_e_interp = f2_te_interp(Reff_interp)
        n_e_interp = f1_te_interp(Reff_interp) / 1e6
        profiles_df = (
            pd.DataFrame(data=[Reff_interp, T_e_interp, n_e_interp]).round(3).T
        )
        profiles_df.columns = ["Reff [m]", "T_e [eV]", "n_e [m-3]"]
        return profiles_df

    def plot(self):
        return super().plot(self.profiles_df)

    def save_txt(self):
        return super().save_txt(self.profiles_df)


class ExperimentalProfile(Profile):
    """The class creates an object repesenting
    experimental kinetic profiles registerd during experimental campaign.

    Parameters
    ----------
    Profile : _type_
        _description_

    Parameters:
        file_name: file name with fitted experimental data;
        interval: (optional) precision of interpolation along Reff (e.g. step = 100
        separates given range of Reff into 100 pieces)
    """

    def __init__(self, file_name, max_Reff=0.539, interp_step=10000, plot=True):
        """
        Parameters
        ----------
        file_name : str
            Name of the loaded file contaitning kinetic data.
        max_Reff : int, optional
            _description_, by default 1
        interp_step : int, optional
            Defines the precision of kinetic profiles interpolation, by default 10000
        plot : bool, optional
            _description_, by default False
        """
        super().__init__()
        self.file_name = file_name
        self.max_Reff = max_Reff  # [m]
        self.interp_step = interp_step
        self.file_path = self._get_file_path()
        self.te_idx_range, self.ne_idx_range = self._get_index_ranges()
        self.te_section, self.ne_section = self._get_data_from_file()
        self.profiles_df = self._interpolate()
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
            Path.cwd()
            / "input_files"
            / "kinetic_profiles"
            / "experimental"
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

        return te_idx_range, ne_idx_range

    def _get_data_from_file(self):
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
                if self.te_idx_range[0] <= idx <= self.te_idx_range[1]:
                    if line.startswith("#"):
                        pass
                    else:
                        te_section.append(line.split())
                if self.ne_idx_range[0] <= idx <= self.ne_idx_range[1]:
                    if line.startswith("#"):
                        pass
                    else:
                        ne_section.append(line.split())

        return te_section, ne_section

    def _merge_to_df(self) -> pd.DataFrame:
        """Designate the final data frame containing Reff, n_e, and T_e profiles.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the following columns:
            - "Reff [m]": Effective radius values in meters.
            - "T_e [eV]": Electron temperature values in electron volts (eV).
            - "n_e [m-3]": Electron density values in meters^-3.

        """
        te_arr = np.array(self.te_section)[:, [0, 3]]
        ne_arr = np.array(self.ne_section)[:, 3].reshape(-1, 1)

        # Combine the columns into a single 2D array and convert to a DataFrame.
        all_data = np.concatenate((te_arr, ne_arr), axis=1).astype(float)
        df = pd.DataFrame(all_data, columns=["Reff [m]", "T_e [eV]", "n_e [m-3]"])

        # Convert temperature and density units and return the DataFrame.
        df["T_e [eV]"] = df["T_e [eV]"] * 1e3
        df["n_e [m-3]"] = df["n_e [m-3]"] * 1e19
        return df

    def _interpolate(self) -> pd.DataFrame:
        """Interpolate selected ne and Te profiles.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the following columns:
            - "Reff [m]": Effective radius values in meters.
            - "T_e [eV]": Interpolated electron temperature values in electron volts (eV).
            - "n_e [m-3]": Interpolated electron density values in meters^-3.

        """
        # Merge the ne and Te profiles into a single DataFrame.
        df = self._merge_to_df()

        # Interpolate the ne and Te profiles.
        Reff = df["Reff [m]"]
        n_e = df["n_e [m-3]"]
        T_e = df["T_e [eV]"]
        f1_te_interp = interpolate.interp1d(Reff, n_e)
        f2_te_interp = interpolate.interp1d(Reff, T_e)

        # Create a new DataFrame with the interpolated profiles.
        shortened_prof_df = df.drop(df[df["Reff [m]"] > self.max_Reff].index)
        Reff_interp = np.linspace(
            0, shortened_prof_df["Reff [m]"].max(), self.interp_step, endpoint=True
        )
        T_e_interp = f2_te_interp(Reff_interp)
        n_e_interp = f1_te_interp(Reff_interp) / 1e6
        profiles_df = (
            pd.DataFrame(data=[Reff_interp, T_e_interp, n_e_interp]).round(3).T
        )
        profiles_df.columns = ["Reff [m]", "T_e [eV]", "n_e [m-3]"]
        return profiles_df

    def plot(self):
        return super().plot(self.profiles_df)

    def save_txt(self):
        return super().save_txt(self.profiles_df)


if __name__ == "__main__":
    ne = [7e13, 0, 0.37, 9.8e12, 0.5, 0.11]
    Te = [1870, 0, 0.155, 210, 0.38, 0.07]
    # tgsp = TwoGaussSumProfile(ne, Te, plot=True)

    ep = ExperimentalProfile("report_20181011_012@5_5000_v_1", plot=True)
    # TestExperimentalProfile("20230215.058").profiles_df
