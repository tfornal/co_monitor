import numpy as np
from collections import namedtuple
from reader import Emissivity
from kinetic_profiles import (
    PredefinedProfile,
    TwoGaussSumProfile,
    ExperimentalProfile,
)


lyman_alpha_lines = ["C", "B", "O", "N"]  #
Element = namedtuple("Element", "ion_state wavelength impurity_fraction")

lyman_alpha_line = {
    "B": Element("Z4", 48.6, 0.02),
    "C": Element("Z5", 33.7, 0.02),
    "N": Element("Z6", 24.8, 0.02),
    "O": Element("Z7", 19.0, 0.02),
}
transitions = ["EXCIT", "RECOM"]


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
    #### one selected file
    experimental_profile_file = "report_20181016_037@3_3000_v_"
    profile = ExperimentalProfile(experimental_profile_file).profile_df

    return profile


if __name__ == "__main__":
    """
    Script runs  the plasma emissivity calculation. It requires to
    choose the profile type (calculated, experimental or theoretical), select
    lines of interest and emission type (EXCIT, RECOM, CHEXC).
    All input files are stored in the "_Input_files" directory.
    """

    n_e = [7e13, 0, 0.37, 9.8e12, 0.5, 0.11]
    T_e = [1870, 0, 0.155, 210, 0.38, 0.07]

    # Select kinetic profiles
    # kinetic_profiles = experimental_prof()
    # kinetic_profiles = predefined_profile(1)
    kinetic_profiles = two_gauss_prof(n_e, T_e)

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
