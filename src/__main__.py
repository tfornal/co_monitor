from collections import namedtuple
from reader import Emissivity
from kinetic_profiles import (
    PredefinedProfile,
    TwoGaussSumProfile,
    ExperimentalProfile,
)
import pathlib
import numpy as np

lyman_alpha_lines = ["B", "C", "N", "O"]
Element = namedtuple("Element", "ion_state wavelength impurity_fraction")

lyman_alpha_line = {
    "B": Element("Z4", 48.6, 0.02),
    "C": Element("Z5", 33.7, 0.02),
    "N": Element("Z6", 24.8, 0.02),
    "O": Element("Z7", 19.0, 0.02),
}
transitions = ["EXCIT", "RECOM"]



def main():
    """
    Script runs the code for the plasma emissivity calculation. Requires to
    choose the profile type (calculated, experimental or theoretical), select
    lines of interest and emission type (EXCIT, RECOM, CHEXC).
    All input files are stored in the "_Input_files" directory.
    """

    ne = [7e13, 0, 0.37, 9.8e12, 0.5, 0.11]
    Te = [1870, 0, 0.155, 210, 0.38, 0.07]

    # profile = two_gauss_prof(ne, Te)
    # profile = experimental_prof()
    profile = predefined_profile(4)

    reff_file_name = "Reff_coordinates-1cm"

    for element in lyman_alpha_lines:
        line = lyman_alpha_line[element]
        ce = Emissivity(
            reff_file_name,
            profile,
            element,
            line.ion_state,
            line.wavelength,
            line.impurity_fraction,
            transitions,
        )
        # ce.savefile()
        ce.plot(savefig=False)


def calculate_coeficients(a, b, num):
    """
    Generates interpolated list of two lists (a, b) with given steps number.
    Useful for making a sequence of coefficients used to calculate the plasma profiles.
    profile_maker(a, b, step).

    Returns
    -------
    list_of_array : 2D array
        2D array of interpolated lists.

    """
    list_of_array = np.array([np.linspace(i, j, num) for i, j in zip(a, b)]).T

    return list_of_array


def predefined_profile(profile_file_number, plot=False):
    """Runs routine to calculate plasma profiles from a given."""
    profile = PredefinedProfile(profile_file_number).profile_df

    return profile


def two_gauss_prof(ne, Te):
    """Runs routine to calculate plasma profiles out of sum of two  Gaussian distributions."""
    profile = TwoGaussSumProfile(ne, Te).profile_df

    return profile


def experimental_prof():
    def experimental_profile_files():
        """Reads the input file list from a directory"""
        file_path = (
            pathlib.Path.cwd() / "_Input_files" / "Kinetic_profiles" / "Experimental"
        )
        experimental_profiles = []
        for filename in pathlib.Path.iterdir(file_path):
            if filename.endswith(".txt"):
                experimental_profiles.append(filename.split(".")[0])

        return experimental_profiles

    # #### sequence of files
    # file_list = experimental_profile_files()
    # for file in file_list:
    #     experimental_profile = pt.experimental(file)

    #### one selected file
    experimental_profile_file = "report_20181016_037@3_3000_v_"
    profile = ExperimentalProfile(experimental_profile_file).profile_df

    return profile


if __name__ == "__main__":
    main()
