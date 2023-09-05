__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

from collections import namedtuple
from pathlib import Path, PurePath
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

from co_monitor.emissivity.fractional_abundance import FractionalAbundance
from co_monitor.emissivity.kinetic_profiles import (
    TwoGaussSumProfile,
    ExperimentalProfile,
)
from co_monitor.emissivity.pec import PEC


class Emissivity:
    def __init__(
        self,
        element,
        ion_state,
        wavelength,
        impurity_concentration,
        transitions,
        reff_magnetic_config,
        kinetic_profiles,
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
        kinetic_profiles : 2D array (N, 3)
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
        self.element = element
        self.ionization_state = ion_state
        self.wavelength = wavelength
        self.transitions = transitions
        self.impurity_concentration = self._convert_imp_frac(impurity_concentration)
        self.reff_magnetic_config = reff_magnetic_config
        self.kinetic_profiles = kinetic_profiles

        self.reff_coordinates = self._get_Reff()
        self.observed_plasma_volume = self.load_observed_plasma()
        self.plas_coords_with_rad_frac = self.get_plas_coords_with_rad_frac(plot=False)

        self.pec_data = self._get_pec_data()
        self.ne_te_profiles, self.reff_boundary = self.read_ne_te()
        self.df_sel_frac_ab = self.get_frac_ab()
        self.df_prof_frac_ab_pec = self.assign_temp_accodring_to_indexes()
        self.df_prof_frac_ab_pec = self.read_pec()
        self.df_prof_frac_ab_pec_emissivity = self.calculate_intensity()
        self.total_emissivity = self.calculate_total_emissivity()
        self.savefile()
        if plot:
            self.plot()

    def _get_pec_data(self):
        lista = []
        interp_step = 2000
        for transition in self.transitions:
            self.interpolated_pec = PEC(
                self.element,
                self.wavelength,
                self.ionization_state[-1],
                transition,
                interp_step,
            ).interpolated_pec
            lista.append(self.interpolated_pec)

        pec_interp = np.array(lista)
        return pec_interp

    def _get_Reff(self) -> pd.DataFrame:
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
            Path(__file__).parent.parent.parent.resolve()
            / "input_files"
            / "geometry"
            / "reff"
            / f"{self.reff_magnetic_config}.dat"
        )

        reff_coordinates = pd.read_csv(Reff_path, sep=";")
        reff_coordinates.columns = ["idx_plasma", "x", "y", "z", "Reff [m]"]
        reff_coordinates = reff_coordinates.astype(
            {"idx_plasma": int, "x": float, "y": float, "z": float, "Reff [m]": float}
        )

        return reff_coordinates

    def _convert_imp_frac(self, impurity_fraction):
        """Converts percentage fraction of impurity to float value."""
        return impurity_fraction / 100

    def load_observed_plasma(self) -> pd.DataFrame:
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
            Path(__file__).parent.parent.parent.resolve()
            / "input_files"
            / "geometry"
            / "observed_plasma_volume"
            / f"{self.element}"
            / f"{self.element}_plasma_coordinates-10_mm_spacing-height_30-length_20-slit_100.dat"
        )
        observed_plasma_volume = pd.read_csv(observed_plasma, sep=";")
        return observed_plasma_volume

    def get_plas_coords_with_rad_frac(self, plot=False):
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

        if plot:
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

        def get_cutoff_reff():
            max_mapped_reff = self.plas_coords_with_rad_frac["Reff [m]"].max()
            max_profiles_reff = self.kinetic_profiles["Reff [m]"].max()
            reff_boundary = min(max_mapped_reff, max_profiles_reff)
            print(f"\nMax reff from mapped vmec: {max_mapped_reff}")
            print(f"\nMax reff from profiles: {reff_boundary}")
            return reff_boundary

        reff_boundary = get_cutoff_reff()

        reff_arr = np.asarray(self.plas_coords_with_rad_frac["Reff [m]"])
        kin_prof_arr = np.asarray(self.kinetic_profiles["Reff [m]"])

        indexes = list(map(lambda x: self._find_nearest(x, kin_prof_arr), reff_arr))

        selected_plasma_parameters = self.kinetic_profiles.iloc[indexes]
        selected_plasma_parameters = selected_plasma_parameters[
            ["n_e [m-3]", "T_e [eV]"]
        ]

        selected_plasma_parameters.reset_index(drop=True, inplace=True)
        self.plas_coords_with_rad_frac.reset_index(drop=True, inplace=True)
        plasma_with_parameters = pd.concat(
            [
                self.plas_coords_with_rad_frac,
                selected_plasma_parameters,
            ],
            axis=1,
        )

        plasma_with_parameters = plasma_with_parameters[
            ~(plasma_with_parameters["Reff [m]"] >= reff_boundary)
        ].reset_index(drop=True)
        return plasma_with_parameters, reff_boundary

    def get_frac_ab(self):
        """
        Assigns fractional abundance to each Reff value.
        """
        fa = FractionalAbundance(self.element, self.ionization_state)
        frac_ab = fa.df_interpolated_frac_ab

        sel_frac_ab = []
        sel_frac_ab.append(frac_ab.iloc[:, 0])
        sel_frac_ab.append(frac_ab[self.ionization_state])
        sel_frac_ab.append(frac_ab.iloc[:, -1])
        col_names = ["T_e [eV]", f"{self.ionization_state}", "Fully stripped"]

        sel_frac_ab = np.asarray(sel_frac_ab).T
        df_sel_frac_ab = pd.DataFrame(sel_frac_ab)
        df_sel_frac_ab.columns = col_names
        return df_sel_frac_ab

    @staticmethod
    def _find_nearest(array, value) -> int:
        """Finds the index of the closest value in a given array to a given value.

        Parameters
        ----------
        array : list, np.array
            The array to search.
        value : float
            The value to search for.

        Returns
        ----------
        idx : int
            The index of the closest value in the array to the given value.
        """
        idx = (np.abs(np.array(array) - value)).argmin()
        return idx

    def find_closest_temp_idx(self) -> List[int]:
        """
        Finds the index of the closest temperature value in a given array for each value in another array.

        Returns
        ----------
        all_min_indexes : list
            A list containing the index of the closest temperature value in the `frac_ab` array
            for each value in the `ne_te_profiles` array.
        """
        # convert the "T_e [eV]" column of the "frac_ab" dataframe to a numpy array
        array = np.asarray(self.df_sel_frac_ab["T_e [eV]"])
        # convert the "T_e [eV]" column of the "ne_te_profiles" dataframe to a numpy array
        value = np.asarray(self.ne_te_profiles["T_e [eV]"])
        # find the index of the closest value in the "array" array for each value in the "value" array,
        # and return a list of all the index values
        all_min_indexes = list(map(lambda x: self._find_nearest(x, array), value))
        return all_min_indexes

    def assign_temp_accodring_to_indexes(self):
        """
        Returns dataframe represented by assigned indexes.
        """

        all_min_indexes = self.find_closest_temp_idx()
        frac_ab = self.df_sel_frac_ab.iloc[all_min_indexes]
        frac_ab = frac_ab.drop(columns=["T_e [eV]"])
        frac_ab.reset_index(drop=True, inplace=True)  # reset indexes
        df_prof_frac_ab = pd.concat(
            [self.ne_te_profiles, frac_ab], axis=1
        )  # concatenate two dataframes

        return df_prof_frac_ab

    def read_pec(self):
        """Reads the PEC data for each transition in the object's "transitions" attribute,
        and updates the object's "df_prof_frac_ab_pec" dataframe with the PEC data for each transition.

        Returns
        -------
        df_prof_frac_ab_pec : pd.DataFrame
            The updated "df_prof_frac_ab_pec" dataframe.
        """
        for idx, trans in enumerate(self.transitions):
            # find the indices of the nearest values in the "array" arrays for each value in the "value" arrays,
            # and store the indices in "ne_idx" and "te_idx" lists
            array = np.asarray(self.pec_data[idx, :, 0, 0])
            value = np.asarray(self.df_prof_frac_ab_pec["n_e [m-3]"])
            ne_idx = list(map(lambda x: self._find_nearest(x, array), value))

            array = np.asarray(self.pec_data[idx, 0, :, 1])
            value = np.asarray(self.df_prof_frac_ab_pec["T_e [eV]"])
            te_idx = list(map(lambda x: self._find_nearest(x, array), value))

            # update the "df_prof_frac_ab_pec" dataframe with the PEC data for the current transition
            self.df_prof_frac_ab_pec[f"pec_{trans}"] = self.pec_data[
                idx, ne_idx, te_idx, 2
            ]

        return self.df_prof_frac_ab_pec

    def calculate_intensity(self):
        """
        Returns
        -------
        df_prof_frac_ab_pec : pd.DataFrame
            Dataframe with containing calculated intensity and sum of intensities
            named TOTAL_Intensity if more transition types were choosen.
        """
        pec_cols = [col for col in self.df_prof_frac_ab_pec.columns if "pec" in col]
        ne = self.df_prof_frac_ab_pec["n_e [m-3]"]

        for pec in pec_cols:
            if "RECOM" in pec:
                self.df_prof_frac_ab_pec[f"Emissivity_RECOM"] = (
                    self.df_prof_frac_ab_pec["Fully stripped"]
                    * self.df_prof_frac_ab_pec[f"{pec}"]
                    * ne**2
                    * self.impurity_concentration
                )
            else:
                self.df_prof_frac_ab_pec[f"Emissivity_{pec[4:]}"] = (
                    self.df_prof_frac_ab_pec[self.ionization_state]
                    * self.df_prof_frac_ab_pec[f"{pec}"]
                    * ne**2
                    * self.impurity_concentration
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
        emissivities = {
            trans: (
                self.df_prof_frac_ab_pec[f"Emissivity_{trans}"]
                * self.df_prof_frac_ab_pec["total_intensity_fraction"]
            ).sum()
            for trans in self.transitions
        }
        total_emissivity = (
            self.df_prof_frac_ab_pec["Emissivity_TOTAL"]
            * self.df_prof_frac_ab_pec["total_intensity_fraction"]
        ).sum()
        total_emissivity = f"{total_emissivity:.2e}"
        for key, value in emissivities.items():
            print(f"{key} is: {value:.2e}")
        print(f"TOTAL is: {total_emissivity}")

        return total_emissivity

    def savefile(self):
        """
        Saves an output dataframe containing all the calculated information in
        the given directory. Creates 'results' / 'numerical_results' path if not exists.
        """
        directory = (
            Path(__file__).parent.parent.parent.resolve()
            / "results"
            / "numerical_results"
        )
        if not Path.is_dir(directory):
            Path(directory).mkdir(parents=True, exist_ok=True)
        np.savetxt(
            PurePath(
                directory,
                f"Emissivity ({self.element} - {self.ionization_state} - file).dat",
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
        plt.title(
            f"Emissivity radial distribution ({self.element} - {self.ionization_state})"
        )
        for col_name in intensity_colname_list:
            plt.plot(
                Reff, self.df_prof_frac_ab_pec[f"{col_name}"], label=f"{col_name[-5:]}"
            )
        plt.legend()
        plt.xlabel("Reff [m]")
        plt.ylabel("Emissivity [ph/cm3/s]")
        plt.axvline(self.reff_boundary, ls="--", color="black")

        if savefig:
            directory = Path.cwd() / "results" / "figures"
            if not Path.is_dir(directory):
                Path(directory).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                PurePath(
                    directory,
                    f"Emissivity ({self.element} - {self.ionization_state})",
                )
            )
            print("Figure saved!")

        plt.show()


def main():
    lyman_alpha_lines = ["O"]  # , "C", "N", "O"]
    Element = namedtuple("Element", "ion_state wavelength impurity_concentration")

    lyman_alpha_line = {
        "B": Element("Z4", 48.6, 2),
        "C": Element("Z5", 33.7, 2),
        "N": Element("Z6", 24.8, 2),
        "O": Element("Z7", 19.0, 2),
        # "O": Element("Z6", 18.6, 2),
    }
    transitions = ["EXCIT", "RECOM"]
    reff_magnetic_config = "Reff_coordinates-10_mm"
    n_e = [4e13, 0, 0.37, 9.8e12, 0.5, 0.11]
    T_e = [570, 0, 0.155, 210, 0.38, 0.07]

    # Select kinetic profiles
    # kinetic_profiles = ExperimentalProfile("report_20181011_012@5_5000_v_1").profiles_df
    kinetic_profiles = TwoGaussSumProfile(n_e, T_e, plot=True).profiles_df

    for element in lyman_alpha_lines:
        line = lyman_alpha_line[element]
        ce = Emissivity(
            element,
            line.ion_state,
            line.wavelength,
            line.impurity_concentration,
            transitions,
            reff_magnetic_config,
            kinetic_profiles,
        )
        ce.plot(savefig=True)


if __name__ == "__main__":
    main()
