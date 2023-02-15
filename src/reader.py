__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

from collections import namedtuple
from pathlib import Path, PurePath
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

from fractional_abundance import FractionalAbundance
from kinetic_profiles import (
    TwoGaussSumProfile,
    ExperimentalProfile,
)
from pec import PEC


class Emissivity:
    def __init__(
        self,
        reff_magnetic_config,
        plasma_profiles,
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
        """

        self.plasma_profiles = plasma_profiles#.rename(index={0: "Reff", 1: "T_e", 2: "n_e"})
        self.plasma_profiles.columns = ["Reff [m]","T_e [eV]", "n_e [m-3]"]
        # df.columns = ["Reff [m]", "T_e [eV]", "n_e [m-3]"]
        print(self.plasma_profiles)
        # breakpoint()
        
        self.reff_magnetic_config = reff_magnetic_config
        self.reff_coordinates = self.load_Reff()
        self.observed_plasma_volume = self.load_observed_plasma(element)
        self.reff_coordinates_with_radiation_fractions = (
            self.read_plasma_coordinates_with_radiation_fractions()
        )
        self.transitions = transitions
        self.pec_data = self._get_pec_data()
        self.element = element
        self.impurity_concentration = impurity_fraction  # [%]
        self.ion_state = ion_state
        self.wavelength = wavelength
        self.transitions = transitions
        self.ne_te_profiles, self.reff_boundary = self.read_ne_te()
        self.frac_ab = self.read_fractional_abundance()
        self.df_prof_frac_ab_pec = self.assign_temp_accodring_to_indexes()

        self.df_prof_frac_ab_pec = self.read_pec()
        self.df_prof_frac_ab_pec_emissivity = self.calculate_intensity(
            self.impurity_concentration
        )
        self.total_emissivity = self.calculate_total_emissivity()
        
    def _get_pec_data(self):
        lista = []
        for transition in self.transitions:
            self.interpolated_pec = PEC(
                "C", 33.7, transition, interp_step=2000, plot=False
            ).interpolated_pec
            lista.append(self.interpolated_pec)

        pec_interp = np.array(lista)
        return pec_interp

    def load_Reff(self) -> pd.DataFrame:
        """
        Load a file containing plasma coordinates and their calculated Reff value (if available).

        Parameters
        ----------
        reff_file_name : str
            The name of the file containing the Reff values.

        Returns
        -------
        reff_coordinates : pd.DataFrame
            A DataFrame with columns ['idx_plasma', 'x', 'y', 'z', 'Reff'], containing the plasma coordinates
            and their corresponding Reff values. The data types of the columns are:
            int for 'idx_plasma', float for 'x', 'y', 'z', and 'Reff'.
        """

        Reff_path = (
            Path(__file__).parent.resolve()
            / "_Input_files"
            / "Reff"
            / f"{self.reff_magnetic_config}.txt"
        )

        reff_coordinates = pd.read_csv(Reff_path, sep=" ")
        reff_coordinates.columns = ["idx_plasma", "x", "y", "z", "Reff [m]"]
        reff_coordinates = reff_coordinates.astype(
            {"idx_plasma": int, "x": float, "y": float, "z": float, "Reff [m]": float}
        )
        return reff_coordinates

    def load_observed_plasma(self, element: str) -> pd.DataFrame:
        """
        Load a file containing observed plasma volume for each spectroscopic channel.

        Parameters
        ----------
        element : str
            The element of interest (B, C, N or O).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the observed plasma volume for each spectroscopic channel.
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

        idx_list = df["idx_sel_plas_points"].tolist()
        plasma_points_po_indeksowaniu = self.reff_coordinates.iloc[idx_list]
        Reff = plasma_points_po_indeksowaniu["Reff [m]"]
        Reff = plasma_points_po_indeksowaniu["Reff [m]"].tolist()
        df["Reff [m]"] = Reff
        df = df.dropna()

        def plotter():
            reff = df["Reff [m]"]
            intensity = df["total_intensity_fraction"]
            plasma_coordinates = df.to_numpy()[:, 1:4]

            def create_point_cloud(coordinates, reff):
                """
                Create point cloud from plasma coordinates and their Reff value.
                """
                point_cloud = pv.PolyData(coordinates)
                point_cloud["Reff [mm]"] = reff

                return point_cloud

            pc = create_point_cloud(plasma_coordinates, intensity)

            fig = pv.Plotter()
            fig.set_background("black")
            fig.add_mesh(pv.PolyData(pc), point_size=8, render_points_as_spheres=True)
            fig.show()

        plotter()

        return df

    def read_ne_te(self):
        """
        Assigns electron temperature and density to each Reff value based
        on a given plasma profiles.

        Returns
        -------
        plasma_with_parameters : pd.DataFrame
            A DataFrame containing the observed plasma volume for each spectroscopic channel.
        """
        plasma_params = pd.DataFrame(
            self.plasma_profiles, columns=["Reff [m]", "n_e [m-3]", "T_e [eV]"]
        )

        indexes = []
        rounded_Reff = plasma_params["Reff [m]"].round(3)

        def cutoff_Reff_above_max_boundary():
            max_mapped_reff = self.reff_coordinates_with_radiation_fractions[
                "Reff [m]"
            ].max()
            max_profiles_reff = plasma_params["Reff [m]"].max()
            reff_boundary = min(max_mapped_reff, max_profiles_reff)
            print(f"\nMax reff from mapped vmec: {max_mapped_reff}")
            print(f"\nMax reff from profiles: {reff_boundary}")
            return reff_boundary

        reff_boundary = cutoff_Reff_above_max_boundary()

        for Reff in self.reff_coordinates_with_radiation_fractions["Reff [m]"]:
            idx = (rounded_Reff - Reff).abs().idxmin()
            indexes.append(idx)

        selected_plasma_parameters = plasma_params.iloc[indexes]
        selected_plasma_parameters = selected_plasma_parameters[["n_e [m-3]", "T_e [eV]"]]

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
            ~(plasma_with_parameters["Reff [m]"] >= reff_boundary)
        ].reset_index(drop=True)

        return plasma_with_parameters, reff_boundary

    def read_fractional_abundance(self):
        """
        Assigns fractional abundance to each Reff value.
        """
        fa = FractionalAbundance(self.element, self.ion_state)
        frac_ab = fa.df_interpolated_frac_ab
        emission_types = list(self.transitions)
        fa_column_names = ["T_e [eV]"]

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
        frac_ab.insert(0, "T_e [eV]", Te)
        frac_ab.columns = fa_column_names

        return frac_ab

    def find_index_of_closest_temperature(self):
        """
        Iterates over the values out of two Te lists and returns a list of indexes
        representing the closest Te values.
        """
        all_min_indexes = []
        for i, plasma_point_temp in self.ne_te_profiles.iterrows():
            index = (plasma_point_temp["T_e [eV]"] - self.frac_ab["T_e [eV]"]).abs().idxmin()
            all_min_indexes.append(index)

        return all_min_indexes

    def assign_temp_accodring_to_indexes(self):
        """
        Returns dataframe represented by assigned indexes.
        """
        all_min_indexes = self.find_index_of_closest_temperature()
        frac_ab = self.frac_ab.iloc[all_min_indexes]
        frac_ab = frac_ab.drop(columns=["T_e [eV]"])
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
        # TODO przepisac do xarray, poprawic
        df_prof_frac_ab_pec = self.assign_temp_accodring_to_indexes()
        for idx, trans in enumerate(self.transitions):
            pec = []
            for i, row in df_prof_frac_ab_pec.iterrows():
                ne_idx = (np.abs(row["n_e [m-3]"] - self.pec_data[idx, :, 0, 0])).argmin()
                te_idx = (
                    np.abs(row["T_e [eV]"] - self.pec_data[idx, ne_idx, :, 1])
                ).argmin()
                pec.append(self.pec_data[idx, ne_idx, te_idx, 2])
            df_prof_frac_ab_pec[f"pec_{trans}"] = pec

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
        ne = self.df_prof_frac_ab_pec["n_e [m-3]"]

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
        self.df_prof_frac_ab_pec.sort_values("Reff [m]", inplace=True)
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

        """

        intensity_colname_list = [
            col for col in self.df_prof_frac_ab_pec.columns if "Emissivity" in col
        ]
        Reff = self.df_prof_frac_ab_pec["Reff [m]"]
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

        def _save_fig():
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


if __name__ == "__main__":

    lyman_alpha_lines = ["O"]  # , "B", "C", "N"]
    Element = namedtuple("Element", "ion_state wavelength impurity_fraction")

    lyman_alpha_line = {
        "B": Element("Z4", 48.6, 0.02),
        "C": Element("Z5", 33.7, 0.02),
        "N": Element("Z6", 24.8, 0.02),
        "O": Element("Z7", 19.0, 0.02),
    }
    transitions = ["EXCIT", "RECOM"]
    n_e = [7e13, 0, 0.37, 9.8e12, 0.5, 0.11]
    T_e = [1870, 0, 0.155, 210, 0.38, 0.07]

    # Select kinetic profiles
    # kinetic_profiles = ExperimentalProfile("report_20181011_012@5_5000_v_1").profiles_df
    kinetic_profiles = TwoGaussSumProfile(n_e, T_e).profiles_df
    
    reff_magnetic_config = "Reff_coordinates-10_mm"
    for element in lyman_alpha_lines:
        line = lyman_alpha_line[element]
        ce = Emissivity(
            reff_magnetic_config,
            kinetic_profiles,
            element,
            line.ion_state,
            line.wavelength,
            line.impurity_fraction,
            transitions,
        )
        # ce.savefile()
        ce.plot(savefig=False)
