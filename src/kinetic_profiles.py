import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from scipy import interpolate


class Profile:
    def plot(self, profile_df):
        """
        Plots interpolated Reff, n_e and T_e.
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
        # ax1.set_ylim(0, 25000)

        ax2 = ax1.twinx()

        color = "blue"
        ax2.set_xlabel("Reff [m]")
        ax2.set_ylabel("ne [1/cm-3]", color=color)
        ax2.plot(Reff, n_e, color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        # ax2.set_ylim(0, 1.2E14)
        fig.tight_layout()

        plt.show()

    def save_txt(self, profile_df):
        """
        Saves generated profile to *.txt file.
        """
        directory = Path.cwd() / "src" / "results" / "plasma_profiles"
        if not pathlib.Path.is_dir(directory):
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        np.savetxt(
            pathlib.PurePath(directory, "plasma_profile.txt"), profile_df, fmt="%.3e"
        )

        print("Profile saved to file!")


class PredefinedProfile(Profile):
    """
    The class is responsible for appropriate readout and interpolation of electron density and
    temperature profiles predefined by Y. Turkin.

    Parameters:
        file_nr (int): file number representing particular file from a given directory
        interval: (optional) precision of interpolation along Reff (e.g. step = 100
        separates given range of Reff into 100 pieces)
    """

    def __init__(self, file_nr, interval=1000):
        self.file_nr = file_nr
        self.interval = interval
        self.profile_df = self.make_interpolation()

    def load_predefined_profiles(self):
        """
        Readout of electron temperature (Te) and electron density (ne)
        from predefined file.

        Returns:
            R_raw: array of Reff values
            ne_raw: array of Ne values for given Reff values [1/cm-3]
            Te_raw: array of Te values for given Reff values [eV]
        """

        heating_scenario = os.path.join(
            Path.cwd()
            / "src"
            / "_Input_files"
            / "Kinetic_profiles"
            / "Theoretical"
            / (f"8MWECRH_{self.file_nr}.txt")
        )
        read_file = np.loadtxt(heating_scenario, delimiter=" ")

        Reff_raw, T_e_raw, n_e_raw = (
            read_file[:, 0] / 1e2,  # cm -> m
            read_file[:, 1],
            read_file[:, 2],
        )
        return Reff_raw, T_e_raw, n_e_raw

    def make_interpolation(self):
        """
        Interpolation of selected ne, te profiles.

        Returns:
            profile_array: array of interpolated values of Reff, ne [1/cm-3], Te [eV] with given precision
        """
        Reff_raw, T_e_raw, n_e_raw = self.load_predefined_profiles()
        f1 = interpolate.interp1d(Reff_raw, T_e_raw)
        f2 = interpolate.interp1d(Reff_raw, n_e_raw)

        Reff = np.linspace(0, Reff_raw[-1], self.interval, endpoint=True)
        T_e = f1(Reff)
        n_e = f2(Reff)

        profile_df = pd.DataFrame({"Reff": Reff.round(3), "T_e": T_e, "n_e": n_e})

        return profile_df

    def plot(self):
        return super().plot(self.profile_df)

    def save_txt(self):
        return super().save_txt(self.profile_df)


class TwoGaussSumProfile(Profile):

    """
    The class is responsible for calculation of the electron temperature (Te)
    and density (ne) using sum of two Gauss profiles.

    Parameters:
        ne_equation_coefficients (tuple): takes tuple of 6 values (A1, A2, x1, x2, w1, w2) in
            as an input for the calculations  - ne [cm3]
            e.g. parameters from discharge 201011.012 from W7-X)
            #n_e - A1 (max_density1) = 7E13, x1 = 0, w1 = 0.37, A2 (max_density2) = 9.8E12, x2 = 0.5, w2 = 0.11
        Te_equation_coefficients (tuple): takes tuple of 6 values (A1, A2, x1, x2, w1, w2)
            as an input for the calculations - Te [eV]
            e.g. parameters from discharge 201011.012 from W7-X)
            #T_e - A1 (max_temp1) = 1870, x1 = 0, w1 = 0.155, A2 (max_temp2) = 210, x2 = 0.38, w2 = 0.07

        max_Reff (float): takes float number representing the maximum value of Reff
        for which the calculations are going to be performed
        ne (bool): True if ne profile should be calculated, False if not; nominally "True"
        Te (bool): True if Te profile should be calculated, False if not; nominally "True"
    """

    def __init__(
        self, ne_equation_coefficients, Te_equation_coefficients, max_Reff=0.539
    ):
        self.max_Reff = max_Reff  # [m]
        self.Reff = np.arange(0, self.max_Reff, 0.001)
        self.ne_equation_coefficients = ne_equation_coefficients
        self.Te_equation_coefficients = Te_equation_coefficients
        self.profile_df = self.calculate_profiles()

    def profile_equation(self, two_gauss_coefficients):
        """
        This function is a sum of two gaussian distributions. It calculates the
        value of electron density (n_e) for given Reff. This approach allows for
        accurate simulation of electron density profile in W7-X plasmas.

        Parameters:
            gauss_coefficients (tuple): takes tuple of 6 values (A1, A2, x1, x2, w1, w2)
            as an input for the calculations

        Returns:
            profile (np.array): returns array of investigated Te and ne profiles
        """
        A1, x1, w1, A2, x2, w2 = two_gauss_coefficients
        profile = A1 * np.exp(-((self.Reff - x1) ** 2) / (2 * w1**2)) + A2 * np.exp(
            -((self.Reff - x2) ** 2) / (2 * w2**2)
        )

        return profile

    def calculate_profiles(self):
        """
        Creates dataframe with Reff, Te and ne arrays.

        Returns:
            profile_array: array of interpolated values of Reff, Te [eV], ne [1/cm-3], with given precision
        """
        ne_profile = self.profile_equation(self.ne_equation_coefficients)
        Te_profile = self.profile_equation(self.Te_equation_coefficients)

        profile_df = pd.DataFrame(data=[self.Reff, Te_profile, ne_profile]).T
        profile_df = profile_df.rename(columns={0: "Reff", 1: "T_e", 2: "n_e"})

        return profile_df

    def plot(self):
        return super().plot(self.profile_df)

    def save_txt(self):
        return super().save_txt(self.profile_df)


class ExperimentalProfile(Profile):
    """
    The class creates an object repesenting
    temperature profiles from predefined theoretical profiles given by Y. Turkin.

    Parameters:
        file_name: file name with fitted experimental data;
        interval: (optional) precision of interpolation along Reff (e.g. step = 100
        separates given range of Reff into 100 pieces)
    """

    def __init__(self, file_name, max_Reff=1, interval=10000):
        self.file_name = file_name
        self.file_path = self.read_file_path()
        self.te_idx_range, self.ne_idx_range = self.read_index_ranges()
        self.raw_profile_df = self.create_raw_profile_df()
        self.interpolation_interval = interval
        self.max_Reff = max_Reff  # [m]
        self.profile_df = self.interpolate_raw_profile()

        def gen_for_STRAHL():
            columns_titles = ["Reff", "n_e", "T_e"]
            self.profile_df = self.profile_df.reindex(columns=columns_titles)
            # print(self.profile_df)
            self.profile_df["n_e"] = self.profile_df["n_e"] / 1e14
            self.profile_df["T_e"] = self.profile_df["T_e"] / 1e3
            self.profile_df["T_i (fake)"] = 1
            # print(self.profile_df)
            # np.savetxt(f"{file_name}_kinetic_profiles.dat",
            #            self.profile_df.to_numpy(),
            #            header = "!      r     n_20m3_e      T_keV_e      T_keV_H", fmt='%.6e', delimiter = "  ")

        # gen_for_STRAHL()

    def read_file_path(self):
        """
        Readout of electron temperature (T_e) and electron density (n_e)
        profiles with experimental data stored in "..\_Input_files\Kinetic_profiles\Experimental" directory;

        Returns:
            file_path: path of a file in the
            ".../_Input_files/Profiles/Experimental/file_name.txt" directory;
        """
        file_path = (
            Path.cwd()
            / "src"
            / "_Input_files"
            / "Kinetic_profiles"
            / "Experimental"
            / f"{self.file_name}.txt"
        )

        return file_path

    def read_index_ranges(self):
        """
        Selects indexes of all T_e and n_e profiles in the readed file. Since, the
        fitted profiles are stored in the files on the last position, only the indexes
        corresponding to those profiles are stored.

        Returns:
            te_idx_range: indexes of the beginning and end of the fitted T_e section
            ne_idx_range: indexes of the beginning and end of the fitted n_e section
        """
        te_all_indexes = []
        ne_all_indexes = []

        with open(self.file_path, "r") as file:
            # finds indexes of the all T_e and n_e profiles
            for idx, line in enumerate(file):
                if "T_e (keV)" in line:
                    te_all_indexes.append(idx - 1)
                if "n_e (10^{19}m^{-3})" in line:
                    ne_all_indexes.append(idx - 1)

            # select indexes of the last fitted profiles of T_e and n_e
            try:
                te_last_idx = te_all_indexes[-1]

            except:
                print("No T_e profile available!")

            try:
                ne_last_idx = ne_all_indexes[-1]
            except:
                print("No n_e profile available!")
            number_of_lines = ne_last_idx - te_last_idx

            # creates the range of investigated file indexes of fitted T_e and n_e
            te_idx_range = [te_last_idx, ne_last_idx]
            ne_idx_range = [ne_last_idx, ne_last_idx + number_of_lines]

        return te_idx_range, ne_idx_range

    def read_file_sections(self):
        """
        Opens the file with fitted profiles and reads the separated section with
        specified index ranges both for T_e and n_e.

        Returns:
            te_section: separated T_e section of the readed file with given T_e index range
            ne_section: separated n_e section of the readed file with given n_e index range
        """
        te_section = []
        ne_section = []
        file_path = self.read_file_path()
        with open(file_path, "r") as file:
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
        return te_section, ne_section

    def remove_comments(self, array):
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
        te_idx_range, ne_idx_range = self.read_index_ranges()
        te_section, ne_section = self.read_file_sections()

        te, ne = self.remove_comments(te_section), self.remove_comments(ne_section)
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

        Reff = (self.create_raw_profile_df().Reff).to_list()
        n_e = (self.create_raw_profile_df().n_e).to_list()
        T_e = (self.create_raw_profile_df().T_e).to_list()
        f1 = interpolate.interp1d(Reff, n_e)
        f2 = interpolate.interp1d(Reff, T_e)

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

        n_e = f1(Reff) / 1e6
        T_e = f2(Reff)

        profile_df = pd.DataFrame(data=[Reff, T_e, n_e]).T
        profile_df = profile_df.rename(columns={0: "Reff", 1: "T_e", 2: "n_e"})
        # print(profile_df)
        return profile_df

    def plot(self):
        return super().plot(self.profile_df)

    def save_txt(self):
        return super().save_txt(self.profile_df)


def main():
    pp = PredefinedProfile(5)
    pp.plot()

    # def profile_maker():
    #     ne1 = [7e13, 0, 0.37, 9.8e12, 0.5, 0.11]
    #     ne2 = [3.5e13, 0, 0.4, 2.5e12, 0.38, 0.15]

    #     Te1 = [1870, 0, 0.155, 210, 0.38, 0.07]
    #     Te2 = [2900, 0, 0.195, 80, 0.38, 0.07]

    #     # ne1 = [1,2,3]
    #     # ne2 = [10,20,30]
    #     ne = np.linspace(ne1, ne2, 2)
    #     Te = np.linspace(Te1, Te2, 2)
    #     return ne, Te

    # ne, Te = profile_maker()
    # for i, j in enumerate(ne):
    #     tgsp = TwoGaussSumProfile(ne[i], Te[i])
    #     tgsp.plot()

    # ne = [7e13, 0, 0.37, 9.8e12, 0.5, 0.11]
    # Te = [1870, 0, 0.155, 210, 0.38, 0.07]
    # tgsp = TwoGaussSumProfile(ne, Te)
    # tgsp.plot()

    # 20180816_022@3_9500_v_; 20181011_012@5_500
    # ep = ExperimentalProfile("report_20181011_012@5_5000_v_1")
    # print(ep.profile_df)
    # ep.plot()


if __name__ == "__main__":
    main()
