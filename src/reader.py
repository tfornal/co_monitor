import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pyvista as pv

from pathlib import Path, PurePath
from pyvistaqt import BackgroundPlotter
from collections import Counter, OrderedDict, namedtuple
from pec import PEC
from fractional_abundance import FractionalAbundance
from kinetic_profiles import (
    PredefinedProfile,
    TwoGaussSumProfile,
    ExperimentalProfile,
)
from impurity_profiles import get_impurity_profile


class Emissivity:
    def __init__(
        self,
        # observed_plasma_volume,
        reff_file_name,
        plasma_profiles,
        impurity_file_name,
        element,
        ion_state,
        wavelength,
        impurity_fraction,
        transitions,
        plot=False,
    ):
        """
        Calculates radial emission and total emissivity of investigated ion.
        Creates an object representing dataframe with all plasma parameters, fractional
        abundances and photon emissivity coefficients assigned to each Reff.
        Plots obtained results.

        Parameters
        ----------
        reff_file_name : STR
            Name of file containing Reff values.
        plasma_profiles : 2D array (N, 3)
            2D array containing Reff [m], ne [cm3], Te[eV].
        element : STR
            Investigated element.
        ion_state : STR
            Ion_charge including atomic number symbol (e.g. "Z7").
        wavelength : FLOAT
            Represents the wavelength in Angstroms. Needs to strictly represent
            the value from the PEC file.
        impurity_fraction : FLOAT
            Represents the impurity fraction in plasma.
        transitions : LIST OF STR
            Represents the transition types. Possibly "EXCIT", "RECOM" and "CHEXC".

        Returns
        -------
        None.

        """

        self.plasma_profiles = plasma_profiles
        self.imp_file_name = impurity_file_name
        self.reff_coordinates = self.load_Reff(reff_file_name)
        self.observed_plasma_volume = self.load_observed_plasma(element)
        self.reff_coordinates_with_radiation_fractions = (
            self.read_plasma_coordinates_with_radiation_fractions()
        )
        max_reff_from_geometr_file = self.reff_coordinates_with_radiation_fractions[
            "Reff"
        ].max()
        self.element = element
        self.impurity_concentration = impurity_fraction  # [%]
        self.ion_state = ion_state
        self.wavelength = wavelength
        self.transitions = transitions
        self.ne_te_profiles, self.reff_boundary = self.read_ne_te()
        self.frac_ab = self.read_fractional_abundance()
        self.df_prof_frac_ab_pec = self.assign_temp_accodring_to_indexes()

        self.interpolated_pec_df, self.transitions_list = PEC(
            element, wavelength, transitions, 20, 20
        ).analyse_pec()
        self.df_prof_frac_ab_pec = self.read_pec()
        self.df_prof_frac_ab_pec_emissivity = self.calculate_intensity(
            self.impurity_concentration
        )
        self.total_emissivity = self.calculate_total_emissivity()

    def load_Reff(self, reff_file_name):
        """
        Load file with plasma coordinates and their calculated Reff value (if exists).
        """
        Reff_path = (
            Path(__file__).parent.resolve() / "_Input_files" / "Reff" / f"{reff_file_name}.txt"
        )

        reff_coordinates = pd.read_csv(Reff_path, sep=" ")
        reff_coordinates.columns = ["idx_plasma", "x", "y", "z", "Reff"]
        reff_coordinates = reff_coordinates.astype(
            {"idx_plasma": int, "x": float, "y": float, "z": float, "Reff": float}
        )
        return reff_coordinates

    def load_observed_plasma(self, element):
        """
        Load file with observed plasma volume by each spectroscopic channel.
        """
        observed_plasma = (
            Path(__file__).parent.resolve()
            / "_Input_files"
            / "Geometric_data"
            / f"{element}"
            / "top"
            / f"{element}_plasma_coordinates-10_mm_spacing-height_40-length_30-slit_100.csv"
        )
        observed_plasma_volume = pd.read_csv(observed_plasma, sep=";")
        return observed_plasma_volume

    def read_plasma_coordinates_with_radiation_fractions(self):
        """
        Load file with plasma coordinates and calculated Reff value.
        """
        df = self.observed_plasma_volume

        lista_indexow = df["idx_sel_plas_points"].tolist()
        plasma_points_po_indeksowaniu = self.reff_coordinates.iloc[lista_indexow]
        Reff = plasma_points_po_indeksowaniu["Reff"]
        Reff = plasma_points_po_indeksowaniu["Reff"].tolist()

        df["Reff"] = Reff
        df = df.dropna()

        def plotter():
            to_numpy = df.to_numpy()

            def create_point_cloud(coordinates, reff):
                point_cloud = pv.PolyData(coordinates)
                point_cloud["Reff [mm]"] = reff

                return point_cloud

            reff = to_numpy[:, -1]
            intensity = to_numpy[:, -2]
            plasma_coordinates = to_numpy[:, 1:4]
            pc = create_point_cloud(plasma_coordinates, intensity)

            fig = pv.Plotter()
            fig.set_background("black")

            fig.add_mesh(pv.PolyData(pc), point_size=8, render_points_as_spheres=True)
            fig.show()

        plotter()

        return df

    def read_ne_te(self):  ### basic version!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        Assigns electron temperature and density to each of Reff values based
        on a given plasma profiles.
        """
        plasma_params = pd.DataFrame(
            self.plasma_profiles, columns=["Reff", "n_e", "T_e"]
        )

        indexes = []
        rounded_Reff = plasma_params["Reff"].round(3)

        def cutoff_Reff_above_max_boundary():
            max_mapped_reff = self.reff_coordinates_with_radiation_fractions[
                "Reff"
            ].max()
            max_profiles_reff = plasma_params["Reff"].max()
            reff_boundary = min(max_mapped_reff, max_profiles_reff)
            print(f"\nMax reff from mapped vmec: {max_mapped_reff}")
            print(f"\nMax reff from profiles: {reff_boundary}")
            return reff_boundary

        reff_boundary = cutoff_Reff_above_max_boundary()

        for Reff in self.reff_coordinates_with_radiation_fractions["Reff"]:
            idx = (rounded_Reff - Reff).abs().idxmin()
            indexes.append(idx)

        selected_plasma_parameters = plasma_params.iloc[indexes]
        selected_plasma_parameters = selected_plasma_parameters[["n_e", "T_e"]]

        selected_plasma_parameters.reset_index(drop=True, inplace=True)
        self.reff_coordinates_with_radiation_fractions.reset_index(
            drop=True, inplace=True
        )
        plasma_with_parameters = pd.concat(
            [
                self.reff_coordinates_with_radiation_fractions,
                selected_plasma_parameters,
            ],
            axis=1,
        )

        plasma_with_parameters = plasma_with_parameters[
            ~(plasma_with_parameters["Reff"] >= reff_boundary)
        ].reset_index(drop=True)

        return plasma_with_parameters, reff_boundary

    def read_fractional_abundance(self):
        """
        Assigns fractional abundance to each Reff value.
        """
        fa = FractionalAbundance(self.element, self.ion_state)
        frac_ab = fa.df_interpolated_frac_ab
        emission_types = list(self.transitions)
        fa_column_names = ["Te"]

        selected_fractional_abundances = []
        if len(emission_types) >= 1 and "RECOM" not in emission_types:
            df = frac_ab[self.ion_state]
            selected_fractional_abundances.append(df)
            fa_column_names.append(self.ion_state)

        elif len(emission_types) >= 1 and "RECOM" in emission_types:
            df = frac_ab[self.ion_state]
            selected_fractional_abundances.append(df)
            fa_column_names.append(self.ion_state)

            df = frac_ab.iloc[:, -1]
            selected_fractional_abundances.append(df)
            fa_column_names.append("Fully stripped")

        selected_fractional_abundances = np.array(selected_fractional_abundances).T
        Te = frac_ab["T"]
        frac_ab = pd.DataFrame(selected_fractional_abundances)
        frac_ab.insert(0, "Te", Te)
        frac_ab.columns = fa_column_names

        return frac_ab

    def find_index_of_closest_temperature(self):
        """
        Iterates over the values out of two Te lists and returns a list of indexes
        representing the closest Te values.
        """
        all_min_indexes = []
        for i, plasma_point_temp in self.ne_te_profiles.iterrows():
            index = (plasma_point_temp["T_e"] - self.frac_ab["Te"]).abs().idxmin()
            all_min_indexes.append(index)

        return all_min_indexes

    def assign_temp_accodring_to_indexes(self):
        """
        Returns dataframe represented by assigned indexes.
        """
        all_min_indexes = self.find_index_of_closest_temperature()
        frac_ab = self.frac_ab.iloc[all_min_indexes]
        frac_ab = frac_ab.drop(columns=["Te"])
        frac_ab.reset_index(drop=True, inplace=True)  # reset indexes
        df_prof_frac_ab = pd.concat(
            [self.ne_te_profiles, frac_ab], axis=1
        )  # concatenate two dataframes

        return df_prof_frac_ab

    def read_pec(self):
        """
        Runs routines to read the PEC files

        Returns
        -------
        df_prof_frac_ab_pec : DATAFRAME
            Dataframe with all infomation required to calculate the radiance intensity.

        """
        df_prof_frac_ab_pec = self.assign_temp_accodring_to_indexes()
        for idx, value in enumerate(self.transitions_list.values()):
            pec = []
            for i, row in df_prof_frac_ab_pec.iterrows():
                ne_idx = (
                    np.abs(row["n_e"] - self.interpolated_pec_df[idx, :, 0, 0])
                ).argmin()
                te_idx = (
                    np.abs(row["T_e"] - self.interpolated_pec_df[idx, ne_idx, :, 1])
                ).argmin()
                pec.append(self.interpolated_pec_df[idx, ne_idx, te_idx, 2])
            df_prof_frac_ab_pec[f"pec_{value}"] = pec

        return df_prof_frac_ab_pec

    def calculate_intensity(self, impurity_concentration):
        """
        Runs routines to read the PEC files

        Returns
        -------
        df_prof_frac_ab_pec : DATAFRAME
            Dataframe with containing calculated intensity and sum of intensities
            named TOTAL_Intensity if more transition types were choosen.

        """

        pec_cols = [col for col in self.df_prof_frac_ab_pec.columns if "pec" in col]
        ne = self.df_prof_frac_ab_pec["n_e"]

        ### TODO !!!!!!!!!!!!!!
        ### wyprowadzic to na gore funkcji!!!!!

        if type(impurity_concentration) in [int, float]:
            impurity_concentration = self.impurity_concentration / 100
        elif impurity_concentration in ["cxrs-peaked", "cxrs-flat", "cxrs-linear"]:
            impurity_concentration = self.df_prof_frac_ab_pec["impurity_concentration"]

        for pec in pec_cols:
            if pec == "pec_RECOM":
                self.df_prof_frac_ab_pec[f"Emissivity_{pec[4:]}"] = (
                    self.df_prof_frac_ab_pec["Fully stripped"]
                    * self.df_prof_frac_ab_pec[f"{pec}"]
                    * ne**2
                    * impurity_concentration
                )
            else:
                self.df_prof_frac_ab_pec[f"Emissivity_{pec[4:]}"] = (
                    self.df_prof_frac_ab_pec[self.ion_state]
                    * self.df_prof_frac_ab_pec[f"{pec}"]
                    * ne**2
                    * impurity_concentration
                )
        intensity_cols = [
            col for col in self.df_prof_frac_ab_pec.columns if "Emissivity" in col
        ]
        self.df_prof_frac_ab_pec["Emissivity_TOTAL"] = self.df_prof_frac_ab_pec[
            intensity_cols
        ].sum(axis=1)
        self.df_prof_frac_ab_pec.sort_values("Reff", inplace=True)
        df_prof_frac_ab_pec_emissivity = self.df_prof_frac_ab_pec

        return df_prof_frac_ab_pec_emissivity

    def calculate_total_emissivity(self):
        """
        Saves an output dataframe containing all the calculated information in
        the given directory. Creates 'results' / 'numerical_results' path if not exists.
        """
        emissivity_excit = (
            self.df_prof_frac_ab_pec["Emissivity_EXCIT"]
            * self.df_prof_frac_ab_pec["total_intensity_fraction"]
        ).sum()

        emissivity_recom = (
            self.df_prof_frac_ab_pec["Emissivity_RECOM"]
            * self.df_prof_frac_ab_pec["total_intensity_fraction"]
        ).sum()

        print("\nEXCIT is:", emissivity_excit)
        print("RECOM is:", emissivity_recom)

        total_emissivity = (
            self.df_prof_frac_ab_pec["Emissivity_TOTAL"]
            * self.df_prof_frac_ab_pec["total_intensity_fraction"]
        ).sum()
        total_emissivity = "{:.2e}".format(total_emissivity)
        print(f"TOTAL is: {total_emissivity}")

        return total_emissivity

    def savefile(self):
        """
        Saves an output dataframe containing all the calculated information in
        the given directory. Creates 'results' / 'numerical_results' path if not exists.
        """
        directory = Path(__file__).parent.resolve() / "_Results" / "numerical_results"
        if not Path.is_dir(directory):
            Path(directory).mkdir(parents=True, exist_ok=True)
        np.savetxt(
            PurePath(
                directory,
                f"Emissivity ({self.element} - {self.ion_state} - file).dat",
            ),
            self.df_prof_frac_ab_pec,
            header=f"{self.df_prof_frac_ab_pec.columns} \n Total emissivity: {self.total_emissivity}",
        )

    def plot(self, savefig=False):
        """
        Plots the radial  distribution of selected line emissions.
        Saves an output chart in the given directory. Creates 'results' / 'figures'
        path if not exists.

        Parameters
        ----------
        savefig : bool, optional
            True if the chart is to be saved. The default is False.

        Returns
        -------
        None.
        """
        intensity_colname_list = [
            col for col in self.df_prof_frac_ab_pec.columns if "Emissivity" in col
        ]
        Reff = self.df_prof_frac_ab_pec["Reff"]
        plt.figure(figsize=(8, 5), dpi=100)
        plt.title(f"Emissivity radial distribution ({self.element} - {self.ion_state})")
        for col_name in intensity_colname_list:
            plt.plot(
                Reff, self.df_prof_frac_ab_pec[f"{col_name}"], label=f"{col_name[-5:]}"
            )

        # plt.yscale('log')
        plt.legend()
        plt.xlabel("Reff [m]")
        plt.ylabel("Emissivity [ph/cm3/s]")
        plt.axvline(self.reff_boundary, ls="--", color="black")

        def save():
            if savefig == True:
                directory = Path.cwd() / "_Results" / "figures"
                if not Path.is_dir(directory):
                    Path(directory).mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    PurePath(
                        directory, f"Emissivity ({self.element} - {self.ion_state})"
                    )
                )
            else:
                pass

        plt.show()


def main():
    reff_file_name = "Reff_coordinates-10_mm"

    def profile_maker():
        ne1 = [7.03e13, 0.01, 0.36, 1.00e13, 0.47, 0.08]
        ne2 = [1.96e13, 0, 0.37, 2.80e12, 0.45, 0.15]

        Te1 = [1870, 0, 0.155, 210, 0.38, 0.07]
        Te2 = [8029, 0, 0.155, 210, 0.38, 0.07]
        ne = np.linspace(ne1, ne2, 1, endpoint=True)
        Te = np.linspace(Te1, Te2, 1, endpoint=True)

        return ne, Te

    ne, Te = profile_maker()
    imp_conc_profiles = [
        # "cxrs-peaked",
        "cxrs-flat"
    ]
    impurity_file_name = "20181011_012@5_5000_conv--100_diff-2000.0.csv"
    for number, profile in enumerate(imp_conc_profiles):
        for idx_ne, n in enumerate(ne):
            for idx_te, t in enumerate(Te):
                plasma_profile = TwoGaussSumProfile(ne[idx_ne], Te[idx_te])
                # plasma_profile.plot()
                plasma_profile = plasma_profile.profile_df

                lyman_alpha_lines = ["O"]

                Element = namedtuple(
                    "Element", "ion_state wavelength impurity_fraction"
                )
                transitions = ["EXCIT", "RECOM"]  # , "CHEXC"]
                ### wpisac albo "cxrs" albofloat zawartosci
                lyman_alpha_line = {
                    "B": Element("Z4", 48.6, 2),
                    "C": Element("Z5", 33.7, 1),
                    "N": Element("Z6", 24.8, 2),
                    "O": Element("Z7", 19.0, 1),
                }

                for element in lyman_alpha_lines:
                    line = lyman_alpha_line[element]
                    observed_plasma = (
                        Path.cwd()
                        / "src"
                        / "_Input_files"
                        / "Geometric_data"
                        / f"{element}"
                        / "top"
                        / f"{element}_plasma_coordinates-10_mm_spacing-height_40-length_30-slit_100.csv"
                    )
                    em = Emissivity(
                        reff_file_name,
                        plasma_profile,
                        impurity_file_name,
                        element,
                        line.ion_state,
                        line.wavelength,
                        line.impurity_fraction,
                        transitions,
                    )


start = time.time()

if __name__ == "__main__":
    main()

print(
    f"\nFinished within {round(time.time() - start, 2)}s",
)
