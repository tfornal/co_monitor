__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

"""
Script runs  the plasma emissivity calculation. It requires to
choose the profile type (calculated or experimental), select
lines of interest and emission type (EXCIT, RECOM, CHEXC).
All input files are stored in the "_Input_files" directory.
"""

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

from emissivity.reader import Emissivity
from emissivity.kinetic_profiles import TwoGaussSumProfile, ExperimentalProfile


def main():
    elements = ["C", "B", "O", "N"]
    Element = namedtuple("Element", "ion_state wavelength impurity_fraction")
    lyman_alpha_line = {
        "B": Element("Z4", 48.6, 0.02),
        "C": Element("Z5", 33.7, 0.02),
        "N": Element("Z6", 24.8, 0.02),
        "O": Element("Z7", 19.0, 0.02),
    }
    transitions = ["EXCIT", "RECOM"]
    reff_magnetic_config = "Reff_coordinates-10_mm"

    # Select kinetic profiles
    n_e = [7e13, 0, 0.37, 9.8e12, 0.5, 0.11]
    T_e = [1870, 0, 0.155, 210, 0.38, 0.07]
    kinetic_profiles = TwoGaussSumProfile(n_e, T_e).profiles_df
    # kinetic_profiles = ExperimentalProfile("report_20181011_012@5_5000_v_1").profile_df

    for element in elements:
        line = lyman_alpha_line[element]
        ce = Emissivity(
            element,
            line.ion_state,
            line.wavelength,
            line.impurity_fraction,
            transitions,
            reff_magnetic_config,
            kinetic_profiles,
        )


if __name__ == "__main__":
    main()


# def geometry():
#     """Executes the simulation code in order to calculate
#     plasma volume observed by each spectroscopic channel.
#     Requires to input list of elements (B, C, N or O)"""

#     from simulation import Simulation

#     elements_list = ["C"]  # , "B" , "N", "O"]
#     settings = dict(
#         slits_number=10,
#         distance_between_points=20,  # the lower value, the higher mesh precision - the longer computation
#         crystal_height_step=15,
#         crystal_length_step=10,
#         savetxt=False,
#         plot=True,
#     )


# if __name__ == "__main__":
#     geometry()
#     for element in elements_list:
#         geom = geometry()
#         simul = Simulation(element, **settings)
